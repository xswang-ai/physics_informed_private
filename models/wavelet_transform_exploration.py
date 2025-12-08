import math
from matplotlib.legend import Patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from wavelet_transform import DWT_2D, IDWT_2D
from wavelet_transform import RelativePositionBias, Transformer, FeedForward, Attention


class WaveletTransformer2D(nn.Module):
    """
    Variant with overlapping patch embeddings and a light deblocking head to
    mitigate 4x4 blocking artifacts without changing the wavelet/transformer core.
    """
    def __init__(self,  wave='haar', in_chans=3, out_chans=3, in_timesteps = 1,  
    dim=64, depth=5, patch_size=(4, 4), normalize=False, meanstd=False, patch_stride=2, use_deblock=False, 
    learnable_scaling_factor=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = dim
        patch_h, patch_w = self.patch_size
        padding = (patch_h // 2, patch_w // 2)

        self.in_chans = in_chans
        self.out_chans = out_chans

        # Overlapping convs replace the stride=patch embedding
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim,
                      kernel_size=self.patch_size, stride=self.patch_stride,
                      padding=0),
            nn.BatchNorm2d(self.embed_dim),
            nn.ELU(inplace=True),
        )
        self.output_proj = nn.Sequential(nn.ConvTranspose2d(self.embed_dim, self.embed_dim//4,
                                              kernel_size=self.patch_size, stride=self.patch_stride,
                                              padding=0),
                                          nn.ELU(inplace=True),
                                          nn.Conv2d(self.embed_dim//4, out_chans,
                                              kernel_size=1, stride=1,
                                              padding=0),
                                          )
        self.dwt = DWT_2D(wave, format='stack')
        self.idwt = IDWT_2D(wave)
        self.rel_pos = RelativePositionBias(dim=dim)
        self.subband_embed = nn.Parameter(torch.randn(4, dim))
        # transformer for cross scale attention
        self.transformer =Transformer(dim=dim, depth=depth, heads=8, dim_head=64, mlp_dim=dim*4)
        self.token_norm = nn.LayerNorm(dim)
        self.learnable_scaling_factor = learnable_scaling_factor
        if learnable_scaling_factor:
            self.scaling_factor = nn.Parameter(torch.ones(4)*0.25) # scalling factor for (LL, LH, HL, HH)
        else:
            self.scaling_factor = 0.25
    

    def get_scaling_factor(self):
        return F.softmax(self.scaling_factor, dim=0)

    def get_pos_embed(self, h, w, s, device, dtype):
        # spatial pos: (h*w, dim) -> (h, w, 1, dim)
        pos_spatial = self.rel_pos(h, w, device=device).to(dtype)
        pos_spatial = pos_spatial.view(h, w, 1, -1)

        subband = self.subband_embed
        if subband.shape[0] < s:
            subband = F.pad(subband, (0, 0, 0, s - subband.shape[0]))
        elif subband.shape[0] > s:
            subband = subband[:s]
        subband = subband.to(device=device, dtype=dtype).view(1, 1, s, -1)

        pos = pos_spatial + subband  # (h, w, s, dim)
        pos = rearrange(pos, 'h w s d -> (h w s) d')
        return pos

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.input_proj(x)
        x = self.dwt(x) # (b, 4, D, H/2, W/2) (ll, lh, hl, hh)
        if self.learnable_scaling_factor:
            scaling_factor = self.get_scaling_factor()
            x = x * scaling_factor.view(1, 4, 1, 1, 1).to(x.device)
        # reshape to a long token (b, d, H//2 * H//2 * 4)
        h, w = x.shape[-2], x.shape[-1]
        s = x.shape[1]
        x = rearrange(x, 'b s d h w -> b (h w s) d')
        pos_embed = self.get_pos_embed(h, w, s, x.device, x.dtype)
        x = x + pos_embed.unsqueeze(0)
        x = self.token_norm(x)

        x = self.transformer(x)
        x = rearrange(x, 'b (h w s) d -> b s d h w', h=h, w=w)
        x = self.idwt(x)
        x = self.output_proj(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WaveletTransformer3D(WaveletTransformer2D):
    def __init__(self, temporal_depth=1, in_timesteps=1, **kwargs):
        super().__init__(in_timesteps=in_timesteps, **kwargs)
        self.in_timesteps = in_timesteps
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.in_timesteps, self.embed_dim))
        self.temporal_transformer = Transformer(
            dim=self.embed_dim,
            depth=temporal_depth,
            heads=4,
            dim_head=64,
            mlp_dim=self.embed_dim * 2,
        )

    def forward(self, x):
        # x: (B, H, W, T, C)
        b, _, _, t, _ = x.shape
        x0 = rearrange(x[..., 0, :], 'b h w c -> b c h w')

        x0 = self.input_proj(x0)
        x0 = self.dwt(x0)
        if self.learnable_scaling_factor:
            scaling_factor = self.get_scaling_factor()
            x0 = x0 * scaling_factor.view(1, 4, 1, 1, 1).to(x0.device)

        h, w = x0.shape[-2], x0.shape[-1]
        s = x0.shape[1]
        tokens = h * w * s

        x_tokens = rearrange(x0, 'b s d h w -> b (h w s) d')

        pos_embed = self.get_pos_embed(h, w, s, x_tokens.device, x_tokens.dtype)
        x_tokens = x_tokens + pos_embed.unsqueeze(0)

        x_tokens = self.token_norm(x_tokens)
        x_tokens = self.transformer(x_tokens)

        temporal_tokens = x_tokens.new_zeros(b * tokens, t, self.embed_dim)
        temporal_tokens[:, 0, :] = x_tokens.reshape(-1, self.embed_dim)

        temporal_pos = self.temporal_pos_embed
        if temporal_pos.shape[1] < t:
            temporal_pos = F.pad(temporal_pos, (0, 0, 0, t - temporal_pos.shape[1]))
        else:
            temporal_pos = temporal_pos[:, :t, :]
        temporal_tokens = temporal_tokens + temporal_pos

        temporal_tokens = self.temporal_transformer(temporal_tokens)
        temporal_tokens = rearrange(temporal_tokens, '(b n) t d -> b n t d', b=b)
        temporal_tokens = rearrange(temporal_tokens, 'b (h w s) t d -> b t s d h w', h=h, w=w, s=s)

        temporal_tokens = rearrange(temporal_tokens, 'b t s d h w -> (b t) s d h w')
        out = self.idwt(temporal_tokens)
        out = self.output_proj(out)
        out = rearrange(out, '(b t) c h w -> b h w t c', b=b, t=t)
        return out


class WaveletBlock(nn.Module):
    def __init__(self, wave='haar', dim=64, **kwargs):
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
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attention(x) # -> (B, H/2 x W/2, C)
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_h, w=new_w)
        x = torch.reshape(x, (b, 4, c//4, new_h, new_w))
        x = self.idwt(x) # -> (B, C, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.final_proj(x) # -> (B, (H x W), C)
        return x

class InnerWaveletTransformer2D(nn.Module):
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
                WaveletBlock(wave=wave, dim=dim),
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

        # gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        # gridx = gridx.reshape(1, self.ref, 1, 1).repeat([b, 1, self.ref, 1])
        # gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        # gridy = gridy.reshape(1, 1, self.ref, 1).repeat([b, self.ref, 1, 1])
        # grid_ref = torch.cat((gridx, gridy), dim=-1).to(x.device)  

        # x_grid = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
        #     reshape(b, size_x, size_y, self.ref * self.ref).contiguous()  
        return x_grid

    def forward(self, x):
        if self.add_grid:
            x_grid = self.get_grid(x)
            x = torch.cat((x, x_grid), dim=-1)
        if self.patch_size is not None:
            x = rearrange(x, 'b h w c -> b c h w')
            x = self.input_proj(x)
            h, w = x.shape[-2], x.shape[-1]
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            x = self.input_proj(x)
            h, w = x.shape[1], x.shape[2]
            x = rearrange(x, 'b h w c -> b (h w) c')
        for ln1, wavelet_block, ln2, ff in self.layers:
            x = wavelet_block(ln1(x), h, w) + x
            x = ln2(ff(x)) + x
        # x = self.norm(x)
        if self.patch_size is None:
            x = self.output_proj(x)
            x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.output_proj(x)
            x = rearrange(x, 'b c h w -> b h w c')
        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiscaleWaveletTransformer2D(nn.Module):
    def __init__(self, wave='haar', input_dim=3, output_dim=3, dim=64, n_layers=5, add_grid=False, patch_size=None, **kwargs):
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
        
        for i in range(self.n_layers):
            dim = dims[i]
            attn_layer = nn.ModuleList([
                nn.LayerNorm(dim),
                WaveletBlock(wave=wave, dim=dim),
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
            up_layer = nn.ModuleList([
                nn.Linear(dim, dim*4),
                nn.LayerNorm(dim*4),
                IDWT_2D(wave),
                nn.Conv2d(2*dim, new_dim, kernel_size=3, padding=1, stride=1, groups=1),
                ])
                
            attn_layer = nn.ModuleList([
                nn.LayerNorm(new_dim),
                WaveletBlock(wave=wave, dim=new_dim),
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

        # gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        # gridx = gridx.reshape(1, self.ref, 1, 1).repeat([b, 1, self.ref, 1])
        # gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        # gridy = gridy.reshape(1, 1, self.ref, 1).repeat([b, self.ref, 1, 1])
        # grid_ref = torch.cat((gridx, gridy), dim=-1).to(x.device)  

        # x_grid = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
        #     reshape(b, size_x, size_y, self.ref * self.ref).contiguous()  
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
    # x = torch.randn(2, 64, 64, 3)
    # # x = torch.randn(2, 96, 192, 7, 3)
    # print("x shape:", x.shape)
    # model = WaveletTransformer2D(in_chans=x.shape[-1],out_chans=x.shape[-1], patch_size=(4, 4), patch_stride=4, dim=512, depth=5)
    # # model = WaveletTransformerHFSKip(in_chans=x.shape[-1],out_chans=x.shape[-1], in_timesteps=x.shape[-2], dim=96, depth=4, num_levels=4)
    # print("number of parameters:", model.count_parameters())
    # with torch.autograd.set_detect_anomaly(True):
    #     output = model(x)
    #     print("output shape:", output.shape)
    #     loss = output.mean()
    #     loss.backward()
    #     print("backward done")
    
    # x = torch.randn(2, 64, 64, 70, 4)
    # model = WaveletTransformer3D(in_chans=x.shape[-1],out_chans=1, patch_size=(4, 4), patch_stride=4, dim=512, depth=5, temporal_depth=2,
    #                              learnable_scaling_factor=False)
    # print("number of parameters:", model.count_parameters())
    # with torch.autograd.set_detect_anomaly(True):
    #     output = model(x)
    #     print("output shape:", output.shape)
    #     loss = output.mean()
    #     loss.backward()
    #     print("backward done")

    x = torch.randn(2, 64, 64, 3)
    # model = InnerWaveletTransformer2D(input_dim=3, output_dim=3, dim=256, n_layers=4, patch_size=4)
    model = MultiscaleWaveletTransformer2D(input_dim=3, output_dim=3,  n_layers=4)
    print("number of parameters:", model.count_parameters())
    with torch.autograd.set_detect_anomaly(True):
        output = model(x)
        print("output shape:", output.shape)
        loss = output.mean()
        loss.backward()
        print("backward done")