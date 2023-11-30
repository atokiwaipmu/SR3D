
import os
import numpy as np
from glob import glob

def set_data_params(params=None, 
                    n_maps=None, 
                    transform_type="sigmoid"):
    if params is None:
        params = {}
    if "data" not in params.keys():
        params["data"] = {}
    params["data"]["HR_dir"]: str = "/gpfs02/work/tanimura/ana/UNet/data/dens_magneticum_snap25_Box128_grid64_CIC_noRSD/"
    params["data"]["LR_dir"]: str = "/gpfs02/work/tanimura/ana/UNet/data/dens_magneticum_snap25_Box128_grid32_CIC_noRSD/"
    params["data"]["n_maps"]: int = n_maps if n_maps is not None else len(glob(params["data"]["LR_dir"] + "*.npy"))
    params["data"]["transform_type"]: str = transform_type
    params["data"]["upsample_scale"]: float = 2.0
    return params

def set_diffusion_params(params=None, scheduler="linear"):
    if params is None:
        params = {}
    if "diffusion" not in params.keys():
        params["diffusion"] = {}
    params['diffusion']['timesteps']: int = 2000
    params['diffusion']['loss_type']: str = "huber"
    params['diffusion']['schedule']: str = scheduler
    if params['diffusion']['schedule'] == "linear":
        params['diffusion']['linear_beta_start']: float = 10**(-6)
        params['diffusion']['linear_beta_end']: float = 10**(-2)
    elif params['diffusion']['schedule'] == "cosine":
        params['diffusion']['cosine_beta_s']: float = 0.015
    params['diffusion']['sampler_type']: str = "uniform"
    return params

def set_architecture_params(params=None, 
                            model="diffusion", 
                            conditioning="concat",
                            norm_type="batch",
                            act_type="mish",
                            block="biggan",
                            mask=False,
                            use_attn=False,
                            flash_attn=False):
    if params is None:
        params = {}
    if "architecture" not in params.keys():
        params["architecture"] = {}
    params["architecture"]["model"]: str = model
    params["architecture"]["mults"] = [1, 2, 4, 4]
    params["architecture"]["block"]: str = block
    params["architecture"]["skip_factor"]: float = 1/np.sqrt(2)
    params["architecture"]["conditional"]: bool = True
    params["architecture"]["conditioning"]: str = conditioning 
    params["architecture"]["mask"]: bool = mask
    params["architecture"]["dim_in"]: int = 1 if params["architecture"]["conditioning"] != "concat" else 2
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    params["architecture"]["norm_type"]: str = norm_type
    params["architecture"]["act_type"]: str = act_type
    params["architecture"]["use_attn"]: bool = use_attn
    params["architecture"]["flash_attn"]: bool = flash_attn
    return params

def set_train_params(params=None, base_dir=None, target="HR", batch_size=4):
    if params is None:
        params = {}
    if "train" not in params.keys():
        params["train"] = {}
    params["train"]['target']: str = target
    params["train"]['train_rate']: float = 0.8
    params["train"]['batch_size']: int = batch_size
    params["train"]['learning_rate'] = 10**-4
    params["train"]['n_epochs']: int = 1000
    params["train"]['gamma']: float = 0.9999
    params["train"]['save_dir']: str = f"{base_dir}/ckpt_logs/{params['architecture']['model']}/"
    os.makedirs(params["train"]['save_dir'], exist_ok=True)
    if "diffusion" in params.keys():
        params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_{params['diffusion']['schedule']}_{params['architecture']['conditioning']}_b{params['train']['batch_size']}"
    else:
        params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_{params['architecture']['conditioning']}_b{params['train']['batch_size']}"
    params["train"]['patience']: int = 30
    params["train"]['save_top_k']: int = 3
    params["train"]['early_stop']: bool = True
    return params

def set_params(
        base_dir="/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D",
        n_maps=None,
        transform_type="sigmoid",
        model="diffusion",
        conditioning="concat",
        norm_type="group",
        act_type="silu",
        block="biggan",
        mask=False,
        use_attn=False,
        flash_attn=False,
        scheduler="linear",
        target="HR", 
        batch_size=4
        ):
    params = {}
    params = set_data_params(params, n_maps=n_maps, transform_type=transform_type)
    params = set_architecture_params(params, model=model, conditioning=conditioning, 
                                    norm_type=norm_type, act_type=act_type, block=block, mask=mask, use_attn=use_attn, flash_attn=flash_attn)
    if model == "diffusion":
        params = set_diffusion_params(params, scheduler=scheduler)
    params = set_train_params(params, base_dir=base_dir, target=target, batch_size=batch_size)
    return params