
import pytorch_lightning as pl

from scripts.dataloader.data import get_data_from_params, get_normalized_from_params, get_loaders, Transforms
from scripts.models.Unet_base import Unet
from scripts.diffusion.ddpm import DDPM
from scripts.diffusion.scheduler import TimestepSampler
from scripts.utils.run_utils import setup_trainer, get_parser
from scripts.params import set_params

def main():
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    ### get training data
    lr, hr = get_data_from_params(params)
    lr, hr = lr.unsqueeze(1), hr.unsqueeze(1)
    log2linear_transform = Transforms("log2linear", None, None)
    lr, hr = log2linear_transform.inverse_transform(lr), log2linear_transform.inverse_transform(hr)
    data_input, data_cond, transforms_input, transforms_cond = get_normalized_from_params(lr, hr, params)
    train_loader, val_loader = get_loaders(data_input, data_cond, params["train"]['train_rate'], params["train"]['batch_size'])
    print("train:validation = {}:{}, batch_size: {}".format(len(train_loader), len(val_loader), params["train"]['batch_size']))

    #get sampler type
    sampler = TimestepSampler(
        timesteps=int(params['diffusion']['timesteps']), 
        sampler_type=params['diffusion']['sampler_type'])
    print("sampler type: {}, timesteps: {}".format(params['diffusion']['sampler_type'], params['diffusion']['timesteps']))

    #get model
    model = DDPM(Unet, params, sampler = sampler)

    trainer = setup_trainer(params)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()