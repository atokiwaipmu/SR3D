
import numpy as np
import torch
from torch import nn

from einops import rearrange, repeat
from SR3D.scripts.utils import default, exists

class SinusoidalPositionEmbeddings(nn.Module):
    #embeds time in the phase
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :] #t1: [40, 1], t2: [1, 32]. Works on cpu, not on mps
        #^ is matmul: torch.allclose(res, torch.matmul(t1.float(), t2)): True when cpu
        #NM: added float for mps
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings #Bx64

class Upsample(nn.Module):
    def __init__(self, dim, dim_out = None, kernel_size=3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2, mode='trilinear')
        self.conv = nn.Conv3d(dim, default(dim_out, dim), kernel_size, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim, dim_out = None, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv3d(dim*8, default(dim_out, dim), kernel_size)

    def forward(self, x):
        x = rearrange(x, 'b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1 = 2, p2 = 2, p3 = 2)
        x = self.conv(x)
        return x
    
class Block(nn.Module):
    """
    Basic building block for the Unet architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.LeakyReLU(0.1) if out_channels > 1 else nn.Identity()

    def forward(self, x, scale_shift = None):
        x = self.conv(x)
        x = self.norm(x) 

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ResnetBlock(nn.Module):
    """
    Residual block composed of two basic blocks. https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, padding=1, groups = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(in_channels, out_channels, kernel_size, padding, groups)
        self.block2 = Block(out_channels, out_channels, kernel_size, padding, groups) 
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()


    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)