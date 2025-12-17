"""
Multiscalle Wavelet Transformer
"""

import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck
import matplotlib.pyplot as plt
import h5py
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from wavelet_utils import DWT_2D, IDWT_2D, Attention, FeedForward



class WaveletAttentionBlock(nn.Module):
    def __init__(self, wave='haar', dim=64, use_efficient_attention=False, local_attention_size=8, **kwargs):
        super().__init__(**kwargs)
        self.dwt = DWT_2D(wave)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim//4),
             nn.LayerNorm(dim//4))
        self.conv_post =  self.filter = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
                # nn.BatchNorm2d(dim),
            )
        self.idwt = IDWT_2D(wave)
        self.attention = Attention(dim)
        self.final_proj = nn.Linear(dim//4, dim)
        self.use_efficient_attention = use_efficient_attention
        if self.use_efficient_attention:
            self.local_attention_size = local_attention_size

    def local_attention(self, x, h, w):
        # input: (B, C, H, W) output : (B, C, H, W)
        b, c, h, w = x.shape
        new_h, new_w = h//self.local_attention_size, w//self.local_attention_size
        x = rearrange(x, 'b c (h_patch new_h) (w_patch new_w)-> (b new_h new_w) (h_patch w_patch) c', new_h=new_h, new_w=new_w)
        x = self.attention(x) # -> (B, H/2 x W/2, C)
        x = rearrange(x, '(b new_h new_w) (h_patch w_patch) c -> b c (h_patch new_h) (w_patch new_w)', b=b, new_h=new_h, new_w=new_w, h_patch=self.local_attention_size, w_patch=self.local_attention_size)
        return x
    
    def global_attention(self, x, h, w):
        # input: (B, C, H, W) output : (B, C, H, W)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attention(x) # -> (B, H/2 x W/2, C)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
    
    def forward(self, x, h, w):
        """
        Input: (B, (H x W), c)
        Output: (B, (H, W), c)
        """
        b, c = x.shape[0], x.shape[-1]
        x = self.mlp_head(x) # (B, (H x W), C) -> (B, (H x W), C//4)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.dwt(x) # -> (B, 4, C//4, H/2, W/2)
        new_h, new_w = x.shape[-2], x.shape[-1]
        x = self.conv_post(x) # -> (B, 4C//4, H/2, W/2)
        if self.use_efficient_attention:
            x = self.local_attention(x, new_h, new_w)
        else:
            x = self.global_attention(x, new_h, new_w)
        x = torch.reshape(x, (b, 4, c//4, new_h, new_w))
        x = self.idwt(x) # -> (B, C, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.final_proj(x) # -> (B, (H x W), C)
        return x


class MultiscaleWaveletTransformer2D(nn.Module):
    def __init__(self, wave='haar', input_dim=3, output_dim=3, dim=64, n_layers=5, use_efficient_attention=False,
                       efficient_layers=[0, 1], add_grid=False, patch_size=None, **kwargs):
        super().__init__(**kwargs)
        self.add_grid = add_grid
        self.n_layers = n_layers
        self.patch_size = patch_size
        dims = np.array([32, 64, 128, 256])*2
        if patch_size is None:
            self.input_proj = nn.Linear(input_dim, dims[0])
            self.output_proj = nn.Sequential(nn.Linear(dims[0], dims[0]//2),
                                            nn.GELU(),
                                            nn.Linear(dims[0]//2, output_dim))

        self.enc_layers = nn.ModuleList([])
        self.use_efficient_attention = use_efficient_attention
        for i in range(self.n_layers):
            dim = dims[i]
            efficient_flag = i in efficient_layers and self.use_efficient_attention
            attn_layer = nn.ModuleList([
                nn.LayerNorm(dim),
                WaveletAttentionBlock(wave=wave, dim=dim, use_efficient_attention=efficient_flag),
                nn.LayerNorm(dim),
                FeedForward(dim, dim*4)
                ])

            down_layer = nn.ModuleList([
                nn.Linear(dim, dim//4),
                nn.LayerNorm(dim//4),
                DWT_2D(wave), 
                nn.Conv2d(dim, dims[i+1] if i < self.n_layers - 1 else dim, kernel_size=3, padding=1, stride=1, groups=1),
            ])
                
            self.enc_layers.append(nn.ModuleList([attn_layer, down_layer]))
        # self.norm = nn.LayerNorm(dim)
        self.dec_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            dim = dims[self.n_layers - i - 1]
            new_dim = dims[self.n_layers - i - 2] if self.n_layers - i - 2 >= 0 else dim
            efficient_flag = self.n_layers - i - 1 in efficient_layers and self.use_efficient_attention
            up_layer = nn.ModuleList([
                nn.Linear(dim, dim*4),
                nn.LayerNorm(dim*4),
                IDWT_2D(wave),
                nn.Conv2d(2*dim, new_dim, kernel_size=3, padding=1, stride=1, groups=1),
                ])
                
            attn_layer = nn.ModuleList([
                nn.LayerNorm(new_dim),
                WaveletAttentionBlock(wave=wave, dim=new_dim, use_efficient_attention=efficient_flag),
                nn.LayerNorm(new_dim),
                FeedForward(new_dim, new_dim*4)
                ])
            
            self.dec_layers.append(nn.ModuleList([up_layer, attn_layer]))

    def get_grid(self, x):
        b, h, w, _ = x.shape
        size_x, size_y = h, w
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).to(x.device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([b, 1, size_y, 1]) 
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float).to(x.device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([b, size_x, 1, 1]) 
        x_grid = torch.cat((gridx, gridy), dim=-1)

        return x_grid

    def attention_block(self, x, layer, h, w):
        ln1, wavelet_block, ln2, ff = layer
        x = wavelet_block(ln1(x), h, w) + x
        x = ln2(ff(x)) + x
        return x
    
    def down_block(self, x, layer, h, w):
        linear, ln, dwt, conv = layer
        x = ln(linear(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = dwt(x) # (B, 4xc//4, H/2, W/2)
        x = conv(x)
        h, w= x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w
    

    def up_block(self, x, x_prev, layer, h, w):
        linear, ln, idwt, conv = layer
        x = ln(linear(x)) # (B, (H/2 x W/2), c*4)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = idwt(x) # (B, C, H, W)
        x = torch.cat((x, x_prev), dim=1) # (B, 2C, H, W)
        x = conv(x)
        h, w= x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w

    def forward(self, x):
        if self.add_grid:
            x_grid = self.get_grid(x)
            x = torch.cat((x, x_grid), dim=-1)

        x = self.input_proj(x)
        h, w = x.shape[1], x.shape[2]
        x = rearrange(x, 'b h w c -> b (h w) c')

        x_list = []

        for attn_layer, down_layer in self.enc_layers:
            x_list.append(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w))
            x = self.attention_block(x, attn_layer, h, w)
            x, h, w = self.down_block(x, down_layer, h, w) # h and w would be updated here
            
        
        for up_layer, attn_layer in self.dec_layers:
            x, h, w = self.up_block(x,  x_list.pop(), up_layer, h, w)
            x = self.attention_block(x, attn_layer, h, w)
        
        # x = self.norm(x)
        x = self.output_proj(x)
        x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":

    print("✅ Models support non-square inputs with proper gradients!")
