
import datetime
import pickle
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from SR3D.scripts.data import get_data, get_norm_dataset, get_dataloader
from SR3D.scripts.unet_modules import Upsample
from SR3D.scripts.diffusion import Diffusion
from SR3D.scripts.scheduler import linear_beta_schedule, TimestepSampler
from SR3D.scripts.unet import Unet

class Unet_pl(pl.LightningModule):
    def __init__(self, 
                channels = 1,  
                dim = 64,
                init_dim = None,
                out_dim = None,
                batch_size = 64,
                learning_rate = 1e-5,
                learning_rate_decay = 0.99,
                num_epochs = 300,
                timesteps = 1000,
                beta_start = 0.0001,
                beta_end = 0.02,
                loss_type="huber", 
                sampler=None, 
                conditional=False):
        super().__init__()
        self.model = Unet(dim=dim, channels=channels, init_dim=init_dim, out_dim=out_dim, self_condition=conditional)
        self.batch_size = batch_size
        self.lr_init = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.num_epochs = num_epochs

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.diffusion = Diffusion(betas)

        self.init_condition = nn.Upsample(scale_factor = 2, mode='trilinear')

        self.loss_type = loss_type
        self.sampler = sampler
        self.conditional = conditional
        self.loss_spike_flg = 0

    def training_step(self, batch, batch_idx):
        if self.conditional:
            hr, lr = batch 
            lr = self.init_condition(lr)
            x = (hr -lr)
            labels = lr
        else:
            x = batch

        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, loss_type=self.loss_type, labels=labels if self.conditional else None)
        self.log('train_loss', loss)

        if self.sampler.type == 'loss_aware':
            loss_timewise = self.diffusion.timewise_loss(self.model, x, t, loss_type=self.loss_type, labels=labels if self.conditional else None)
            self.sampler.update_history(t, loss_timewise)

        if loss.item() > 0.1 and self.current_epoch > 300 and (self.loss_spike_flg < 2):
            badbdict = {'batch': batch.detach().cpu().numpy(), 'itn': self.current_epoch, 't': t.detach().cpu().numpy(), 'loss': loss.item()}
            pickle.dump(badbdict, open(f'largeloss_{self.current_epoch}.pkl', 'wb'))
            self.loss_spike_flg += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.conditional:
            hr, lr = batch 
            lr = self.init_condition(lr)
            x = hr - lr
            labels = lr
        else:
            x = batch

        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, loss_type=self.loss_type, labels=labels if self.conditional else None)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.learning_rate_decay)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 

### Functions for training
def setup_trainer(num_epochs, logger=None, patience=10):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=0,
        mode="min"
    )

    dt = datetime.datetime.now()
    name = dt.strftime('Run_%m-%d_%H-%M')

    checkpoint_callback = ModelCheckpoint(
        filename= name + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        save_last=True,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', 
        devices=1,
        logger=logger
    )
    return trainer


def main():
    print("Hello World!")
    pl.seed_everything(1234)

    ### training params
    num_epochs = 300
    batch_size =4
    learning_rate = 1e-5
    learning_rate_decay = 0.99

    ### diffusion params
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02

    ### data params
    LR_dir = "/gpfs02/work/tanimura/ana/UNet/data/dens_magneticum_snap25_Box128_grid32_CIC_noRSD/"
    HR_dir = "/gpfs02/work/tanimura/ana/UNet/data/dens_magneticum_snap25_Box128_grid128_CIC_noRSD/"
    mid_dir = "/gpfs02/work/tanimura/ana/UNet/data/dens_magneticum_snap25_Box128_grid64_CIC_noRSD/"

    n_maps=100
    rate_train =0.8

    ### model parameters
    save_dir = '/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/ckpt_logs'
    logger = TensorBoardLogger(save_dir=save_dir, name='SR3D_diffusion')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ### data
    data_hr = get_data(mid_dir, n_maps=n_maps)
    data_lr = get_data(LR_dir, n_maps=n_maps)
    combined_dataset, transforms, inverse_transforms, RANGE_MIN, RANGE_MAX = get_norm_dataset(data_hr, data_lr)
    train_loader, val_loader = get_dataloader(combined_dataset, rate_train, batch_size)

    sampler = TimestepSampler(sampler_type='uniform', timesteps=timesteps, device=device)
    model = Unet_pl(
        channels = 1,
        dim = 64,
        init_dim = 64,
        out_dim = 1,
        batch_size = batch_size,
        learning_rate = learning_rate,
        learning_rate_decay = learning_rate_decay,
        num_epochs = num_epochs,
        timesteps = timesteps,
        beta_start = beta_start,
        beta_end = beta_end,
        loss_type="huber", 
        sampler=sampler, 
        conditional=True).to(device)

    trainer = setup_trainer(num_epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()