# Description: Implementation of the diffusion model from the paper
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scripts.utils.utils import extract

class Diffusion():
    # implement arxiv:2104.07636
    def __init__(self, alphas):
        self.alphas = alphas
        self.one_minus_alphas = 1. - alphas
        self.gammas = torch.cumprod(self.alphas, axis=0)
        self.sqrt_gammas = torch.sqrt(self.gammas)
        self.sqrt_one_minus_gammas = torch.sqrt(1. - self.gammas)
        self.gammas_prev = F.pad(self.gammas[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.one_minus_alphas * (1. - self.gammas_prev) / (1. - self.gammas)

class Diffusion():
    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # alpha_bar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # y_t = sqrt_alphas_cumprod* x_0 + sqrt_one_minus_alphas_cumprod * eps_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        #self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas
        # y_t-1 = sqrt_recip_alphas * (y_t - betas_t * MODEL(x_t, t) / sqrt_one_minus_alphas_cumprod_t)
        #         + sqrt_one_minus_alphas_cumprod_t * eps_t
        self.timesteps = len(self.betas)

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", condition=None):
        # L_CE <= L_VLB ~ Sum[eps_t - MODEL(x_t(x_0, eps_t), t) ]
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, condition=condition)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def timewise_loss(self, denoise_model, x_start, t, noise=None, loss_type="l1", condition=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, condition=condition)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise, reduction='none')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise, reduction='none')
        else:
            raise NotImplementedError()
        loss = torch.mean(loss, dim=[-3, -2, -1]) #mean over all spatial dims
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, condition=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_output = model(x, t) if condition is None else model(x, t, condition=condition)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + posterior_variance_t * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition=None):
        device = next(model.parameters()).device
        print('sample device', device)
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        if condition is not None:
            assert condition.shape[0] == shape[0]

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, condition=condition)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1, condition=None):
        return self.p_sample_loop(model, shape=(batch_size, image_size, channels), condition=condition)