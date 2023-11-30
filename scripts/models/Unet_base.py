
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial

from scripts.layers.timeembedding import SinusoidalPositionEmbeddings
from scripts.layers.attention import Attention, LinearAttention

from scripts.blocks.Resnetblock import Block, ResnetBlock, ResnetBlock_BG
from scripts.utils.utils import exists
    
class Unet(pl.LightningModule):
    """
    Full Unet architecture composed of an encoder (downsampler), a bottleneck, and a decoder (upsampler).
    The architecture is inspired by the one used in the (https://arxiv.org/abs/2311.05217).
    """
    def __init__(self, params):
        super().__init__()

        self.dim_in = params["architecture"]["dim_in"]    
        self.dim_out = params["architecture"]["dim_out"]
        self.dim = params["architecture"]["inner_dim"]
        self.dim_mults = [self.dim * factor for factor in params["architecture"]["mults"]]
        self.skip_factor = params["architecture"]["skip_factor"]
        self.conditioning = params["architecture"]["conditioning"]
        self.use_attn = params["architecture"]["use_attn"]
        self.flash_attn = params["architecture"]["flash_attn"]

        self.depth = len(self.dim_mults)

        # time embeddings
        self.time_dim = self.dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.cond_upsample = nn.Upsample(scale_factor = params["data"]["upsample_scale"], mode='trilinear')

        self.init_conv = nn.Conv3d(self.dim_in, self.dim, kernel_size = 7, padding = 3)
        if self.conditioning == "addconv":
            self.init_conv_cond = nn.Conv3d(self.dim_in, self.dim, kernel_size = 7, padding = 3)

        self.block_type = params["architecture"]["block"]
        if self.block_type == "resnet":
            block_ud = partial(ResnetBlock, time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])
        elif self.block_type == "biggan":
            block_ud = partial(ResnetBlock_BG, time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])
        block_id = partial(ResnetBlock, time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out in zip(self.dim_mults[:-1], self.dim_mults[1:]):
            self.down_blocks.append(
                nn.ModuleList([
                    block_id(dim_in, dim_in),
                    block_ud(dim_in, dim_out, updown_sample = "Downsample")
                    ])
                )
        
        self.mid_block1 = block_id(self.dim_mults[-1], self.dim_mults[-1])
        self.mid_block2 = block_id(self.dim_mults[-1], self.dim_mults[-1])
        if self.use_attn:
            self.attn = Attention(self.dim_mults[-1], heads=4, dim_head=32, flash=self.flash_attn, norm_type=params["architecture"]["norm_type"])

        self.mid_block3 = block_id(self.dim_mults[-1], self.dim_mults[-1])
        self.mid_block4 = block_id(2 * self.dim_mults[-1], self.dim_mults[-1])
        
        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1])):
            self.up_blocks.append(
                nn.ModuleList([
                    block_id(2*dim_in, dim_in),
                    block_ud(dim_in, dim_out, updown_sample = "Upsample"),
                    block_id(2*dim_out, dim_out)
                    ])
                )

        self.final_conv = nn.Sequential(
            block_id(2 * self.dim_mults[0], self.dim), 
            nn.Conv3d(self.dim, self.dim_out, kernel_size = 1, padding = 0))

    def forward(self, x, time, condition=None):
        skip_connections = []
        if condition is not None:
            if self.conditioning == "concat":
                x = self.init_conv(torch.cat([x, self.cond_upsample(condition)], dim=1))
            elif self.conditioning == "addconv":
                x = self.init_conv(x) + self.init_conv_cond(self.cond_upsample(condition))
        else:
            x = self.init_conv(x)
        skip_connections.append(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # downsample
        for block1, block2 in self.down_blocks:
            x = block1(x, t)
            skip_connections.append(x)
            x = block2(x, t)
            skip_connections.append(x)

        # bottleneck
        x = self.mid_block1(x, t)
        skip_connections.append(x)
        x = self.mid_block2(x, t)
        if self.use_attn:
            x = self.attn(x)
        x = self.mid_block3(x, t)
        tmp_connection = skip_connections.pop() * self.skip_factor
        x = torch.cat([x, tmp_connection], dim=1)
        x = self.mid_block4(x, t)

        # upsample
        for block1, block2, block3  in self.up_blocks:
            tmp_connection = skip_connections.pop() * self.skip_factor
            x = torch.cat([x, tmp_connection], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            tmp_connection = skip_connections.pop() * self.skip_factor
            x = torch.cat([x, tmp_connection], dim=1)
            x = block3(x, t)
        
        tmp_connection = skip_connections.pop() * self.skip_factor
        x = torch.cat([x, tmp_connection], dim=1)
        return self.final_conv(x)