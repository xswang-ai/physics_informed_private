import math
from re import escape
from matplotlib.legend import Patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from wavelet_transform import DWT_2D, IDWT_2D,  FeedForward,  Attention, WaveletAttentionBlock
from einops import einsum


class SingleScaleWaveletTransformer2D(nn.Module):
    def __init__(self, wave='haar', input_dim=3, output_dim=3, dim=64, n_layers=5, add_grid=False, patch_size=None, **kwargs):
        super().__init__(**kwargs)
        self.add_grid = add_grid
        self.n_layers = n_layers
        self.patch_size = patch_size
        if patch_size is None:
            self.input_proj = nn.Linear(input_dim, dim)
            self.output_proj = nn.Sequential(nn.Linear(dim, dim//2),
                                            nn.GELU(),
                                            nn.Linear(dim//2, output_dim))
        else:
            self.input_proj = nn.Sequential(nn.Conv2d(input_dim, dim, kernel_size=patch_size, stride=patch_size),
                                            nn.BatchNorm2d(dim),
                                            nn.GELU())
            self.output_proj = nn.Sequential(nn.ConvTranspose2d(dim, dim//2, kernel_size=patch_size, stride=patch_size),
                                            nn.GELU(),
                                            nn.Conv2d(dim//2, output_dim, kernel_size=1, stride=1))
        self.layers = nn.ModuleList([])
        for i in range(self.n_layers):
            layer = nn.ModuleList([
                nn.LayerNorm(dim),
                WaveletAttentionBlock(wave=wave, dim=dim),
                nn.LayerNorm(dim),
                FeedForward(dim, dim*4)
                ])
            self.layers.append(layer)
        # self.norm = nn.LayerNorm(dim)

    def get_grid(self, x):
        b, h, w, _ = x.shape
        size_x, size_y = h, w
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).to(x.device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([b, 1, size_y, 1]) 
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float).to(x.device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([b, size_x, 1, 1]) 
        x_grid = torch.cat((gridx, gridy), dim=-1)
        return x_grid
    

    def get_input_proj(self, x):
        if self.patch_size is not None:
            x = rearrange(x, 'b h w c -> b c h w')
            x = self.input_proj(x)
            h, w = x.shape[-2], x.shape[-1]
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            x = self.input_proj(x)
            h, w = x.shape[1], x.shape[2]
            x = rearrange(x, 'b h w c -> b (h w) c')
        return x, h, w

    def transformer_blocks(self, x, h, w):
        for ln1, wavelet_block, ln2, ff in self.layers:
            x = wavelet_block(ln1(x), h, w) + x
            x = ln2(ff(x)) + x
        return x
    
    def get_output_proj(self, x, h, w):
        if self.patch_size is None:
            x = self.output_proj(x)
            x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.output_proj(x)
        return x

    def forward(self, x):
        if self.add_grid:
            x_grid = self.get_grid(x)
            x = torch.cat((x, x_grid), dim=-1)
        
        x, h, w = self.get_input_proj(x)
        x = self.transformer_blocks(x, h, w)
        x = self.get_output_proj(x, h, w)
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RescalingLayer(nn.Module):
    def __init__(self, dim, trainable: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim, dim), requires_grad=trainable)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=trainable)

    def set_trainable(self, trainable: bool):
        # Toggle grads without recreating the parameters so weights are reused.
        self.scale.requires_grad = trainable
        self.bias.requires_grad = trainable

    def forward(self, x):
        x = torch.einsum("b c h w, c d -> b d h w", x, self.scale)
        x = x + self.bias.reshape(1, -1, 1, 1)
        return x

class WaveletAttentionBlockResidual(nn.Module):
    def __init__(self, wave='haar', dim=64, use_efficient_attention=False, local_attention_size=8, rescaling=False, **kwargs):
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
        self.rescaling = rescaling
        if rescaling:
            self.rescaling_layer = RescalingLayer(dim)
        else:
            self.rescaling_layer = None
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
        x = self.dwt(x) # -> (B, 4*C//4, H/2, W/2)
        new_h, new_w = x.shape[-2], x.shape[-1]
        if self.rescaling:
            x = self.rescaling_layer(x)
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


class MSWTResidual2D(nn.Module):
    def __init__(self, wave='haar', input_dim=3, output_dim=3, dim=64, n_layers=5, use_efficient_attention=False,
                       efficient_layers=[0, 1], add_grid=False, patch_size=None, rescaling=False, **kwargs):
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
                WaveletAttentionBlock(wave=wave, dim=dim, use_efficient_attention=efficient_flag, rescaling=rescaling),
                nn.LayerNorm(dim),
                FeedForward(dim, dim*4)
                ])

            down_layer = nn.ModuleList([
                nn.Linear(dim, dim//4),
                nn.LayerNorm(dim//4),
                DWT_2D(wave), 
                RescalingLayer(dim),
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
                RescalingLayer(dim*4),
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

        # Cache all rescaling layers so we can toggle their trainability later without
        # needing two model instances.
        self._rescaling_layers = [m for m in self.modules() if isinstance(m, RescalingLayer)]

    def _set_trainable(self, *, rescaling_trainable: bool, others_trainable: bool):
        # First toggle everything, then override the rescaling blocks.
        for param in self.parameters():
            param.requires_grad = others_trainable
        for layer in self._rescaling_layers:
            layer.set_trainable(rescaling_trainable)

    def enable_mean_training(self):
        """Train everything except the rescaling layers (mean model)."""
        self._set_trainable(rescaling_trainable=False, others_trainable=True)

    def enable_residual_training(self):
        """Freeze everything except the rescaling layers (residual model)."""
        self._set_trainable(rescaling_trainable=True, others_trainable=False)

    def trainable_param_groups(self):
        """Return a single param group of currently trainable parameters."""
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params}]

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
        linear, ln, dwt, rw, conv = layer
        x = ln(linear(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = dwt(x) # (B, 4xc//4, H/2, W/2)
        x = rw(x)
        x = conv(x)
        h, w= x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, h, w
    

    def up_block(self, x, x_prev, layer, h, w):
        linear, ln, rw,idwt, conv = layer
        x = ln(linear(x)) # (B, (H/2 x W/2), c*4)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = rw(x)
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

class MSWT2DStable(nn.Module):
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

    def inverse_spectral_mapping(self, phi_spatial, x0):
        # compute fft of x_0
        # x0: (B,H,W,C) real
        B, H, W, C = x0.shape
        if phi_spatial.shape[-1] != C:
            x0 = x0[..., :phi_spatial.shape[-1]] # crop the grid part
        X0 = torch.fft.rfft2(x0, dim=(1, 2))  # (B,H,W//2+1,C) complex

        # Easiest drop-in: take rfft2 of the model output and use its angle.
        P = torch.fft.rfft2(phi_spatial, dim=(1, 2))               # complex
        phi = torch.atan2(P.imag, P.real)                          # real phase, same shape as X0
        phase = torch.exp(1j * phi)                                 # unit magnitude
        X = X0 * phase

        x = torch.fft.irfft2(X, s=(H, W), dim=(1, 2))               # (B,H,W,C) real
        return x
        

    def forward(self, x):
        x0 = x.clone()
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


        #stable spectral to guarantee stability
        x = self.inverse_spectral_mapping(x, x0.detach())
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class MSWT2DStableSoftControl(nn.Module):
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
                                            nn.Linear(dims[0]//2, output_dim*2))
            self.initialize_output_proj() # initial phase ≈ 1 and amp ≈ 1

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
   
    def initialize_output_proj(self):
        last = self.output_proj[-1]               
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


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

    def inverse_spectral_mapping(self, out_spatial, x0, amp_max=0.1, eps=1e-6):
        # compute fft of x_0
        # x0: (B,H,W,C) real
        B, H, W, C = x0.shape
        
        if out_spatial.shape[-1] != C:
            x0 = x0[..., :out_spatial.shape[-1]] # crop the grid part in the input
        
        X0 = torch.fft.rfft2(x0, dim=(1, 2))  # (B,H,W//2+1,C) base spectrum
        
        # Easiest drop-in: take rfft2 of the model output and use its angle.
        phase_sp, loggain_sp = out_spatial.split(2, dim=-1)
        print("phase_sp shape:", phase_sp.shape, "loggain_sp shape:", loggain_sp.shape)
       
        # model the phase
        P = torch.fft.rfft2(phase_sp, dim=(1, 2))               # complex
        phase = P / (P.abs() + eps)
    
        # model the amplitude
        A = torch.fft.rfft2(loggain_sp, dim=(1, 2))
        log_gain = A.real
        
        # bound the gain to keep dynamics stable
        log_gain = amp_max * torch.tanh(log_gain/amp_max)
        log_gain = log_gain - log_gain.mean(dim=(1,2,3), keepdim=True)
        amp = torch.exp(log_gain)
        

         # ---- enforce rfft boundary constraints (recommended) ----
        # DC bin must be real -> phase = 1 there
        phase = phase.clone()
        phase[:, 0, 0, :] = 1.0 + 0.0j

        # Nyquist column (W even) should be real
        if W % 2 == 0:
            phase[:, :, -1, :] = phase[:, :, -1, :].real + 0.0j

        # Nyquist row (H even) at k_y=H/2 and k_x=0 should be real
        if H % 2 == 0:
            phase[:, H // 2, 0, :] = phase[:, H // 2, 0, :].real + 0.0j
                
        
        X = X0 * amp * phase
        x = torch.fft.irfft2(X, s=(H, W), dim=(1, 2))               # (B,H,W,C) real

        x_reg = (log_gain ** 2).mean()
        return x, x_reg
        

    def forward(self, x):
        x0 = x.clone()
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


        #stable spectral to guarantee stability
        x, x_reg = self.inverse_spectral_mapping(x, x0.detach())
        if self.training:
            return x, x_reg
        else:
            return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def verify_energy_stability(model, x):
    import matplotlib.pyplot as plt
    def compute_energy(x):
        x_freq = torch.abs(torch.fft.rfft2(x, dim=[1, 2]))**2
        x_energy = torch.sum(x_freq[0, :, :, 0], dim=0) # pick the first channel, and sum over the collumns
        return x_energy.detach().cpu().numpy()    
    
    x = x[0].unsqueeze(0) # just one sample
    x0_energy = compute_energy(x)
    
    x_pred_energy_list = []
    for t in range(10):
        x = model(x)
        x = x[0] if isinstance(x, tuple) else x
        x_pred_energy_list.append(compute_energy(x))
    
    fig, ax = plt.subplots(1, 1)
    for i in range(len(x_pred_energy_list)):
        ax.loglog(x_pred_energy_list[i], label=f't={i}')
    ax.loglog(x0_energy, label='x0')
    ax.legend()
    plt.show()
    return



if __name__ == "__main__":

    x = torch.randn(2, 64, 64, 3)
    # model = InnerWaveletTransformer2D(input_dim=3, output_dim=3, dim=256, n_layers=4, patch_size=4)
    model = MSWT2DStableSoftControl(input_dim=3, output_dim=1,  n_layers=4, use_efficient_attention=True)
    print("number of parameters:", model.count_parameters())
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        output = output[0] if isinstance(output, tuple) else output
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")

    # verify the energy stability
    verify_energy_stability(model, x)