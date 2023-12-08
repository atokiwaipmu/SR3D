

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import random
from scripts.diffusion.diffusion import Diffusion
from scripts.diffusion.scheduler import linear_beta_schedule, cosine_beta_schedule

from scripts.utils.diffusion_utils import extract_masked_region, get_masked_grid
from scripts.utils.utils import extract

'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
'''

class DDPM(pl.LightningModule):
    def __init__(self, model, params, sampler=None):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = model(params)
        self.batch_size = params["train"]['batch_size']
        self.learning_rate = params["train"]['learning_rate']
        self.gamma = params["train"]['gamma']
        print("We are using Adam with lr = {}, gamma = {}".format(self.learning_rate, self.gamma))
        self.ifmask = params["architecture"]["mask"]

        timesteps = int(params['diffusion']['timesteps'])
        if params['diffusion']['schedule'] == "cosine":
            betas = cosine_beta_schedule(timesteps=timesteps, s=params['diffusion']['cosine_beta_s'])
            print("The schedule is cosine with s = {}".format(params['diffusion']['cosine_beta_s']))
        elif params['diffusion']['schedule'] == "linear":
            betas = linear_beta_schedule(timesteps=timesteps, beta_start=params['diffusion']['linear_beta_start'], beta_end=params['diffusion']['linear_beta_end'])
            print("The schedule is linear with beta_start = {}, beta_end = {}".format(params['diffusion']['linear_beta_start'], params['diffusion']['linear_beta_end']))
        self.diffusion = Diffusion(betas)
        self.loss_type = params['diffusion']['loss_type']

        self.sampler = sampler
        self.masks = [[1,1,1,1,1,1,1,1], #Full mask
                    [1,1,1,1,0,0,0,0], [1,1,0,0,1,1,0,0], [1,0,1,0,1,0,1,0], # half mask
                    [1,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0], [1,0,0,0,1,0,0,0], # quarter mask
                    [1,0,0,0,0,0,0,0]] # 1/8 mask

    def training_step(self, batch, batch_idx):
        x, cond = batch
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        if self.ifmask:
            mask = [random.choice(self.masks) for _ in range(x.shape[0])]
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            x_masked = torch.stack([get_masked_grid(x[i], mask=mask[i], x_t=x_t[i]) for i in range(x.shape[0])], dim=0)
            predicted_noise = self.model(x_masked, t, condition=cond)

            corr_noise = torch.cat([extract_masked_region(noise[i], mask[i]) for i in range(x.shape[0])], dim=0)
            corr_pred = torch.cat([extract_masked_region(predicted_noise[i], mask[i]) for i in range(x.shape[0])], dim=0)
            loss = F.smooth_l1_loss(corr_noise, corr_pred)
        else:
            loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, cond = batch
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        if self.ifmask:
            mask = [random.choice(self.masks) for _ in range(x.shape[0])]
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            x_masked = torch.stack([get_masked_grid(x[i], mask=mask[i], x_t=x_t[i]) for i in range(x.shape[0])], dim=0)
            predicted_noise = self.model(x_masked, t, condition=cond)

            corr_noise = torch.cat([extract_masked_region(noise[i], mask[i]) for i in range(x.shape[0])], dim=0)
            corr_pred = torch.cat([extract_masked_region(predicted_noise[i], mask[i]) for i in range(x.shape[0])], dim=0)
            loss = F.smooth_l1_loss(corr_noise, corr_pred)
        else:
            loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 