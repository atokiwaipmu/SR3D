
import torch
import numpy as np

def linear_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def cosine_beta_schedule(timesteps, s=0.015):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999) 

class TimestepSampler():
    def __init__(self, sampler_type='uniform', history=None, nstart=None, timesteps=None, uniweight=None, device='cuda'):
        self.type= sampler_type
        print('Sampler type', self.type)
        if self.type not in ['uniform', 'loss_aware']:
            raise NotImplementedError()
        if self.type=='loss_aware':
            self.sqloss_history = torch.ones((history, timesteps), device=device)*np.nan #L^2[b, t]
            self.nstart = nstart
            self.uniweight = 1/timesteps if uniweight is None else uniweight
            print('Nstart', nstart, type(nstart))
            print('History', history, type(history))
            print('Uniweight', self.uniweight, type(self.uniweight))

        self.timesteps = timesteps
        self.uniform = 1/timesteps
        self.device=device
        self.history_per_term = torch.zeros(timesteps, device=device, dtype=int)
        self.not_enough_history = True

    def get_weights(self, batch_size, iteration):
        if (iteration<self.nstart) or self.not_enough_history:
            return np.ones(self.timesteps)*self.uniform
        else:
            laweights = torch.sqrt(torch.mean(self.sqloss_history**2, dim=0))
            laweights /= laweights.sum()
            laweights *= (1-self.uniweight)
            laweights += self.uniweight/self.timesteps
            return laweights
        #fast way to evaluate / store the loss for different timesteps??
        #do you need a different sampler for storing history?

    def update_history(self, tl, loss_timewise):
        if self.not_enough_history:  # not-full loss history array
            for (t, tloss) in zip(tl, loss_timewise):
                if self.history_per_term[t] == self.sqloss_history.shape[0]:  # enough history
                    self.sqloss_history[:-1, t] = self.sqloss_history[1:, t]
                    self.sqloss_history[-1, t] = tloss
                else:
                    self.sqloss_history[self.history_per_term[t], t] = tloss
                    self.history_per_term[t] += 1
                    if self.history_per_term.min()==self.sqloss_history.shape[0]:
                        self.not_enough_history = False
                        print('Enough history for all')
        else:#enough history for all terms
            #test if this works fine
            self.sqloss_history[:-1, tl] = self.sqloss_history[1:, tl]
            self.sqloss_history[-1, tl] = loss_timewise
        return

    def get_timesteps(self, batch_size, iteration):
        if self.type=='uniform':
            return torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        elif self.type=='loss_aware':
            weights = self.get_weights(batch_size, iteration)
            return torch.tensor(list(torch.utils.data.WeightedRandomSampler(weights, batch_size, replacement=True)), device=self.device).long()
        else:
            raise NotImplementedError()