from torch import nn
from einops import rearrange
from scripts.layers.normalization import Norms
from scripts.layers.activation import Acts
from scripts.layers.timeembedding import TimeEmbed
from scripts.utils.utils import exists, default

'''
The code defines various neural network blocks, specifically for ResNet architecture.
These blocks include standard convolutional layers, normalization, and activation functions.
Additionally, it includes a specialized block for BigGAN's upsampling/downsampling operations.
The implementation is designed to be modular, allowing easy integration into larger neural network models.
'''

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
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm_type="group", act_type="silu"):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = Norms(in_channels, norm_type)
        self.act = Acts(act_type) if out_channels > 1 else nn.Identity()

    def forward(self, x):
        x = self.norm(x) 
        x = self.act(x)
        x = self.conv(x)
        return x
        
class ResnetBlock_BG(nn.Module):
    """
    Up/Downsampling Residual block of BigGAN. https://arxiv.org/abs/1809.11096
    """
    def __init__(self, in_channels, out_channels, updown_sample="Identity", time_emb_dim=None, norm_type="group", act_type="silu"):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp1 = TimeEmbed(time_emb_dim, in_channels * 2)
            self.mlp2 = TimeEmbed(time_emb_dim, out_channels * 2)

        self.norm1 = Norms(in_channels, norm_type)
        self.norm2 = Norms(out_channels, norm_type)

        self.act1 = Acts(act_type) 
        self.act2 = Acts(act_type)

        if updown_sample == "Upsample":
            self.updown_sample = Upsample(in_channels, in_channels)
        elif updown_sample == "Downsample":
            self.updown_sample = Downsample(in_channels, in_channels)
        elif updown_sample == "Identity":
            self.updown_sample = nn.Identity()

        self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time_emb=None):
        h = x
        res = self.conv_res(self.updown_sample(x))

        if exists(time_emb):
            t_e = self.mlp1(time_emb)
            scale, shift = t_e.chunk(2, dim = 1)
            h = h * (scale + 1) + shift

        h = self.norm1(h)
        h = self.act1(h)
        h = self.updown_sample(h)
        h = self.conv1(h)

        if exists(time_emb):
            t_e = self.mlp2(time_emb)
            scale, shift = t_e.chunk(2, dim = 1)
            h = h * (scale + 1) + shift

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + res
    
class ResnetBlock(nn.Module):
    """
    Up/Downsampling Residual block implemented from https://arxiv.org/abs/2311.05217.
    Originally https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, in_channels, out_channels, updown_sample="Identity", time_emb_dim=None, norm_type="group", act_type="silu"):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp = TimeEmbed(time_emb_dim, out_channels * 2)

        if updown_sample == "Upsample":
            self.updown_sample = Upsample(in_channels, out_channels)
        elif updown_sample == "Downsample":
            self.updown_sample = Downsample(in_channels, out_channels)
        elif updown_sample == "Identity":
            self.updown_sample = nn.Identity()

        self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()
        self.block1 = Block(in_channels, out_channels, norm_type=norm_type, act_type=act_type)
        self.block2 = Block(out_channels, out_channels, norm_type=norm_type, act_type=act_type)

    def forward(self, x, time_emb=None):
        h = x
        res = self.conv_res(self.updown_sample(x))

        h = self.block1(h)
        h = self.updown_sample(h)

        if exists(time_emb):
            t_e = self.mlp(time_emb)
            scale, shift = t_e.chunk(2, dim = 1)
            h = h * (scale + 1) + shift

        h = self.block2(h)
        return h + res
