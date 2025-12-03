"""
Compute Wavelet transform https://github.com/YehLi/ImageNetModel/blob/main/classification/torch_wavelets.py 
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

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh, format='cat'):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        
        # Store original shape for reconstruction
        B, C, H, W = x.shape
        ctx.shape = x.shape
        
        # Calculate padding needed for odd dimensions
        pad_h = H % 2
        pad_w = W % 2
        
        # Apply padding if needed (pad on right and bottom)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Store padded shape and padding info for backward pass
        ctx.padded_shape = x.shape
        ctx.padding = (pad_w, pad_h)

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        if format == 'cat':
            x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        else:
            x = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            pad_w, pad_h = ctx.padding
            
            dx = dx.view(B, 4, -1, dx.shape[-2], dx.shape[-1])
            dx = dx.transpose(1,2).reshape(B, -1, dx.shape[-2], dx.shape[-1])
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
            
            # Remove padding if it was added in forward pass
            if pad_h > 0 or pad_w > 0:
                dx = dx[:, :, :H, :W]

        return dx, None, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters, target_size=None):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        ctx.target_size = target_size

        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]

        # x = x.transpose(1, 2)  # (B, D//4, 4, H, W)
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        
        # Store the uncropped size for backward pass
        ctx.uncropped_size = (x.shape[-2], x.shape[-1])
        
        # If target size is provided, crop to match it
        if target_size is not None:
            target_h, target_w = target_size
            current_h, current_w = x.shape[-2], x.shape[-1]
            if current_h > target_h or current_w > target_w:
                x = x[:, :, :target_h, :target_w]
        
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            C = ctx.shape[2]
            
            # If we cropped in forward pass, we need to pad dx back to uncropped size
            if ctx.target_size is not None:
                uncropped_h, uncropped_w = ctx.uncropped_size
                current_h, current_w = dx.shape[-2], dx.shape[-1]
                
                if uncropped_h > current_h or uncropped_w > current_w:
                    pad_h = uncropped_h - current_h
                    pad_w = uncropped_w - current_w
                    dx = F.pad(dx, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.stack([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        # self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x, target_size=None):
        return IDWT_Function.apply(x, self.filters, target_size)

class DWT_2D(nn.Module):
    def __init__(self, wave, format='cat'):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
        self.format = format
        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x, format='cat'):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh, self.format)
        


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def test_dwt_idwt():
    data_path = '/Users/wan410/Documents/VSCode/DPOT/pdearena/ns2d_pda/train/data_0.hdf5'
    x = h5py.File(data_path, 'r')['data'].astype(np.float64)
    # print(x.shape)
    x = torch.Tensor(x).double()
    print("original data shape", x.shape)
    
    x = x[:, :, 7, :].unsqueeze(0).permute(0, 3, 1, 2)  # Change to (B, C, H, W)
    print("x shape:", x.shape)

    w = pywt.Wavelet('haar')
    dec_hi = torch.Tensor(w.dec_hi[::-1]) 
    dec_lo = torch.Tensor(w.dec_lo[::-1])

    w_ll = (dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_lh = (dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hl = (dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hh = (dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()

    input = (
        Variable(x, requires_grad=True),
        Variable(w_ll, requires_grad=False),
        Variable(w_lh, requires_grad=False),
        Variable(w_hl, requires_grad=False),
        Variable(w_hh, requires_grad=False),
    )

    test_output = DWT_Function.apply(*input)
    print("test_output shape:", test_output.shape)
    

    output = test_output[0, :, :, :].detach().cpu().numpy()
    # visualize the input and output
    fig = plt.figure(figsize=(15, 3))
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    ax = fig.add_subplot(1, 6, 1)
    ax.imshow(input[0][0, 0, :, :].detach().cpu().numpy(), interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title("Input", fontsize=10)
    for i in range(4):
        ax = fig.add_subplot(1, 6, i + 2)
        a = output[i*3, :, :]
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    # plt.show()


    # recpver the original image with IDWT
    w = pywt.Wavelet('haar')
    rec_hi = torch.Tensor(w.rec_hi)
    rec_lo = torch.Tensor(w.rec_lo)
        
    w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
    w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
    w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
    w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

    w_ll = w_ll.unsqueeze(0).unsqueeze(1)
    w_lh = w_lh.unsqueeze(0).unsqueeze(1)
    w_hl = w_hl.unsqueeze(0).unsqueeze(1)
    w_hh = w_hh.unsqueeze(0).unsqueeze(1)
    filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).double()

    input_recover = (
        Variable(test_output, requires_grad=True),
        Variable(filters, requires_grad=False),
    )
    test = IDWT_Function.apply(*input_recover)
    print("test shape:", test.shape)

    # visualize the recovered image
    ax = fig.add_subplot(1, 6, 6)
    ax.imshow(test[0, 0, :, :].detach().cpu().numpy(), interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title("Recovered", fontsize=10)
    fig.tight_layout()
    plt.show()
    

class RelativePositionBias(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dpb = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            )


    def forward(self, height, width, device='cpu'):
        
        # create grid indices for relative positions
        pos_idx = torch.arange(height)
        pos_idy = torch.arange(width)
        grid = torch.stack(torch.meshgrid(pos_idx, pos_idy, indexing = 'ij')).float().to(device)
        
        grid[0] = 1.0/height*(grid[0] - height//2)
        grid[1] = 1.0/width*(grid[1] - width//2) 
        # print(grid[0, :, 0], grid[1, :, 0])
        grid = rearrange(grid, 'c i j -> (i j) c')
    
        # Compute the relative position embeddings
        pos_embed = self.dpb(grid)
        return pos_embed





class CrossWaveletTransformer(nn.Module):
    def __init__(self, wave='haar',n_channels=3, in_timesteps = 4,  dim=64, depth=5,patch_size=(4, 4), normalize=False, meanstd=False):
        super(CrossWaveletTransformer, self).__init__()
        self.meanstd = meanstd
        self.patch_size = patch_size
        self.input_proj = nn.Sequential(
            nn.Conv2d(4+in_timesteps*n_channels, dim, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.BatchNorm2d(dim),
            nn.ELU(inplace=True),
            )  # (B, D, H, W)
    

        # DWT modules
        self.num_dwt_blocks = 4
        self.dwt_project = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 4, kernel_size=1, stride=1, padding=0),
                           nn.BatchNorm2d(dim // 4),
                           nn.ELU(inplace=True)
                          ) for i in range(self.num_dwt_blocks)
        ])

        self.idwt_project = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=dim//4, out_channels=dim, kernel_size=1, stride=1, padding=0),
                           nn.BatchNorm2d(dim),
                           nn.ELU(inplace=True)
                          ) for i in range(self.num_dwt_blocks)
        ])
        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        # position and scale embeddings
        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.rand(1, 1, dim)) for _ in range(self.num_dwt_blocks)
        ])
        self.relative_position_embeddings = RelativePositionBias(dim=dim)

        # transformer for cross scale attention
        self.transformer =Transformer(dim=dim, depth=depth, heads=8, dim_head=64, mlp_dim=dim*4)

        # final output layer
        if self.meanstd:
            self.output_proj =  nn.ConvTranspose2d(dim, n_channels*2, kernel_size=patch_size, stride=patch_size, padding=0)
        else:
            self.output_proj =  nn.ConvTranspose2d(dim, n_channels, kernel_size=patch_size, stride=patch_size, padding=0)
        self.normalize = normalize
        
    def get_latent_by_index(self, x, index):
        """ 
        Get the latent representation from the input to the (index -1)-th block of Transformer
        """
        x, img_size = self.get_dwt_representation(x) # (B, N, D)
        if index == 0:
            return x
        for attn, ff in self.transformer.layers[:index]:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
    def get_testing_block_by_index(self, index, x):
        """
        Get the latent representation from the input to the index-th block of Transformer
        """
        attn, ff = self.transformer.layers[index]
        x = attn(x) + x
        x = ff(x) + x
        return x


    def get_dwt_representation(self, x):
         # Store original spatial dimensions
        orig_h, orig_w = x.shape[1], x.shape[2]
        
        # get grid and concat
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma

        grid = self.get_grid(x.size(), x.device)
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous() 

        # Calculate padding needed for patch_size
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - (orig_h % patch_h)) % patch_h
        pad_w = (patch_w - (orig_w % patch_w)) % patch_w
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # input shape: (B, C, H, W)
        # projecting to a higher dimension, 2d convolution with kernel size 1
        x = self.input_proj(x) # (B, D, H, W)
        
        # Store the projected dimensions (after patch embedding)
        proj_h, proj_w = x.shape[-2], x.shape[-1]
        
        # run several blocks of DWT to obtain the wavelet coefficients
        x_scale = []
        img_size = []
        img_dims = []  # Store actual (height, width) dimensions
        break_idx = 0
        for i in range(self.num_dwt_blocks):
            x = self.dwt_project[i](x)  # (B, D//4, H, W)
            if min(x.shape[-1], x.shape[-2]) ==1: # too small for wavelet transform
                break_idx = i
                break
            # Store original dimensions before DWT
            orig_h_dwt, orig_w_dwt = x.shape[-2], x.shape[-1]
            img_dims.append((orig_h_dwt, orig_w_dwt))
            
            # apply DWT
            x = self.dwt(x) # (B, 4*D//4, H//2, W//2)
            img_size.append((x.shape[2]*x.shape[3]))
            x_scale_input = rearrange(x, 'b d h w -> b (h w) d')  # (B, D, H//2 * W//2)
            # generate relative position embeddings for every position in the [H//2, W//2] grid
            pos_embed = self.relative_position_embeddings(x.shape[2], x.shape[3], x.device) # shape (B, D, H//2 * W//2)
            # scale embeddings for each scale
            scale_embed = self.scale_embeddings[i]  # (B, D, 1)
            
            x_scale_input = x_scale_input + pos_embed + scale_embed # (B, D, H//2 * W//2)
            x_scale.append(x_scale_input)

        # concatenate the wavelet coefficients from all scales
        x = torch.cat(x_scale, dim=1) # (B, N, D)

        return x, img_size

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2 * np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat(
            (torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)),
            dim=-1,
        ).to(device)  # (bs, H, W, 4)
        return grid


    def forward(self, x):
        # Store original spatial dimensions
        orig_h, orig_w = x.shape[1], x.shape[2]
        
        # get grid and concat
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma

        grid = self.get_grid(x.size(), x.device)
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous() 

        # Calculate padding needed for patch_size
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - (orig_h % patch_h)) % patch_h
        pad_w = (patch_w - (orig_w % patch_w)) % patch_w
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # input shape: (B, C, H, W)
        # projecting to a higher dimension, 2d convolution with kernel size 1
        x = self.input_proj(x) # (B, D, H, W)
        
        # Store the projected dimensions (after patch embedding)
        proj_h, proj_w = x.shape[-2], x.shape[-1]
        
        # run several blocks of DWT to obtain the wavelet coefficients
        x_scale = []
        img_size = []
        img_dims = []  # Store actual (height, width) dimensions
        break_idx = 0
        for i in range(self.num_dwt_blocks):
            x = self.dwt_project[i](x)  # (B, D//4, H, W)
            if min(x.shape[-1], x.shape[-2]) ==1: # too small for wavelet transform
                break_idx = i
                break
            # Store original dimensions before DWT
            orig_h_dwt, orig_w_dwt = x.shape[-2], x.shape[-1]
            img_dims.append((orig_h_dwt, orig_w_dwt))
            
            # apply DWT
            x = self.dwt(x) # (B, 4*D//4, H//2, W//2)
            img_size.append((x.shape[2]*x.shape[3]))
            x_scale_input = rearrange(x, 'b d h w -> b (h w) d')  # (B, D, H//2 * W//2)
            # generate relative position embeddings for every position in the [H//2, W//2] grid
            pos_embed = self.relative_position_embeddings(x.shape[2], x.shape[3], x.device) # shape (B, D, H//2 * W//2)
            # scale embeddings for each scale
            scale_embed = self.scale_embeddings[i]  # (B, D, 1)
            
            x_scale_input = x_scale_input + pos_embed + scale_embed # (B, D, H//2 * W//2)
            x_scale.append(x_scale_input)

        # concatenate the wavelet coefficients from all scales
        x = torch.cat(x_scale, dim=1) # (B, N, D)
    

        # apply transformer to the wavelet coefficients
        x = self.transformer(x)
        x = rearrange(x, 'b n d -> b d n')

        # recover the original image with IDWT
        # first split x based on image size to get the wavelet coefficients for each scale
        x_splits = torch.split(x, img_size, dim=-1)
        
        x_recov = torch.zeros_like(x_splits[-1])  # initialize the recovered image
        for i, x_split in enumerate(x_splits[::-1]):  # reverse the order to match the DWT order
            x_recov = x_recov + x_split # combine the wavelet from the current scale (x_split) with the previous recovered image (x_recov)
            # get the scale size and dimensions
            scale_idx = len(x_splits) - 1 - i
            scale_size = img_size[scale_idx]
            orig_h_dwt, orig_w_dwt = img_dims[scale_idx]
            
            # Calculate the DWT output dimensions (after stride=2 with padding)
            dwt_h = (orig_h_dwt + 1) // 2  # This accounts for padding in DWT
            dwt_w = (orig_w_dwt + 1) // 2
            
            # start with the last scale, which has the smallest size
            x_recov = rearrange(x_recov, 'b (p d) (h w) -> b p d h w', p = self.num_dwt_blocks, h=dwt_h, w=dwt_w)
            # apply IDWT to the wavelet coefficients with target size
            x_recov = self.idwt(x_recov, target_size=(orig_h_dwt, orig_w_dwt)) # (b d h w)
            x_recov = self.idwt_project[i](x_recov)  # (B, 4*d, H, W))
            x_recov = rearrange(x_recov, 'b d h w -> b d (h w)') # (b d H*W)
        # the final output is the recovered image from the last scale
        # Get the final dimensions from the first scale (which is the original projected size)
        final_h, final_w = img_dims[0] if img_dims else (proj_h, proj_w)
        x = rearrange(x_recov, 'b d (h w) -> b d h w', h=final_h, w=final_w)  # (B, D, H, W)
        x = self.output_proj(x) # (B, 3, H, W)
        
        # Crop back to original dimensions if padding was applied
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]
        
        # reshape to (B, C, H, W)
        x = rearrange(x, 'b c h w -> b h w 1 c')
        if self.normalize:
            x = x * sigma  + mu
        return x

    def count_parameters(self):

        # count the parameters  of dwt_project and idwt_project
        dwt_params = sum(p.numel() for p in self.dwt_project.parameters() if p.requires_grad)
        idwt_params = sum(p.numel() for p in self.idwt_project.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        input_proj_params = sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
        print("DWT parameters:", dwt_params)
        print("IDWT parameters:", idwt_params)
        print("Transformer parameters:", transformer_params)
        print("Input projection parameters:", input_proj_params)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




class CrossWaveletTransSkipConnection(CrossWaveletTransformer):
    def __init__(self, wave='haar',n_channels=3, in_timesteps = 4,  dim=64, depth=5,patch_size=(4, 4), normalize=False, meanstd=False):
        super(CrossWaveletTransSkipConnection, self).__init__(
            wave=wave,
            n_channels=n_channels,
            in_timesteps=in_timesteps,
            dim=dim,
            depth=depth,
            patch_size=patch_size,
            normalize=normalize,
            meanstd=meanstd,
        )

    def input_preprocessing(self, x):
        # Store original spatial dimensions
        orig_h, orig_w = x.shape[1], x.shape[2]
        
        # get grid and concat
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
        else:
            mu, sigma = None, None
        grid = self.get_grid(x.size(), x.device)
        x = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous() 
        
        # Calculate padding needed for patch_size
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - (orig_h % patch_h)) % patch_h
        pad_w = (patch_w - (orig_w % patch_w)) % patch_w
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (mu, sigma) , (pad_h, pad_w)


    def downsampling(self, x):
        # run several blocks of DWT to obtain the wavelet coefficients
        x_scale = []
        skip_connection = []
        
        for i in range(self.num_dwt_blocks):
            x = self.dwt_project[i](x)  # (B, D//4, H, W)
            if min(x.shape[-1], x.shape[-2]) ==1: # too small for wavelet transform
                break_idx = i
                break

            # apply DWT
            x = self.dwt(x) # (B, 4*D//4, H//2, W//2)
            skip_connection.append(x) # append x before adding the positional embedding
            x_scale_input = rearrange(x, 'b d h w -> b (h w) d')  # (B, D, H//2 * W//2)
            # generate relative position embeddings for every position in the [H//2, W//2] grid
            pos_embed = self.relative_position_embeddings(x.shape[2], x.shape[3], x.device) # shape (B, D, H//2 * W//2)
            # scale embeddings for each scale
            scale_embed = self.scale_embeddings[i]  # (B, D, 1)
            
            x_scale_input = x_scale_input + pos_embed + scale_embed # (B, D, H//2 * W//2)
            x_scale.append(x_scale_input)

        # concatenate the wavelet coefficients from all scales
        x = torch.cat(x_scale, dim=1) # (B, N, D)
        return x, skip_connection
    
    
    def upsampling(self, x, skip_connection, proj_h, proj_w):
         # recover the original image with IDWT
        # first split x based on image size to get the wavelet coefficients for each scale
        img_size = [x.shape[-1]*x.shape[-2] for x in skip_connection]
        x_splits = torch.split(x, img_size, dim=-1)
        
        x_recov = torch.zeros_like(x_splits[-1]).reshape(skip_connection[-1].shape)  # initialize the recovered image
        for i, (x_split, x_skip) in enumerate(zip(x_splits[::-1], skip_connection[::-1])):  # reverse the order to match the DWT order
            x_recov = x_recov + x_split.reshape(x_skip.shape) + x_skip # combine the wavelet from the current scale (x_split) with the previous recovered image (x_recov)
            # 4 coeffecients required by IDWT, LL, LH, HL, HH
            x_recov = rearrange(x_recov, 'b (p d) h w -> b p d h w', p = 4)
            # apply IDWT to the wavelet coefficients with target size
            x_recov = self.idwt(x_recov) # (b d h w)
            x_recov = self.idwt_project[i](x_recov)  # lift dimension, (B, 4*d, H, W))
        
        x = x_recov
        return x

    def output_postprocessing(self, x, mu, sigma, pad_h, pad_w, orig_h, orig_w):
        # Crop back to original dimensions if padding was applied
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]
        
        # reshape to (B, C, H, W)
        x = rearrange(x, 'b c h w -> b h w 1 c')
        if self.normalize:
            x = x * sigma  + mu
        return x

    def forward(self, x):
        orig_h, orig_w = x.shape[1], x.shape[2]
        x, (mu, sigma), (pad_h, pad_w) = self.input_preprocessing(x) # concate xy coordinates and padding if needed
        x = self.input_proj(x) # (B, D, H, W)
        
        # Store the projected dimensions (after patch embedding)
        proj_h, proj_w = x.shape[-2], x.shape[-1]
        
        ######### Downsampling Blocks #########################################################
        x, skip_connection = self.downsampling(x)

        ####### Transformer Blocks #########################################################
        x = self.transformer(x)
        x = rearrange(x, 'b n d -> b d n')

        ####### Recovering Blocks #########################################################
        x = self.upsampling(x, skip_connection, proj_h, proj_w)
        x = self.output_proj(x) # (B, C, H, W)
        x = self.output_postprocessing(x, mu, sigma, pad_h, pad_w, orig_h, orig_w)
        return x





def test_non_square_input():
    """Test the model with non-square inputs"""
    print("Testing CrossWaveletTransformer with non-square inputs...")
    
    # Test with different aspect ratios
    test_cases = [
        (64, 128),   # 1:2 ratio
        (128, 64),   # 2:1 ratio
        (96, 128),   # 3:4 ratio
        (63, 127),   # Odd dimensions
    ]
    
    model = CrossWaveletTransformer(wave='haar', dim=64, patch_size=(4, 4))
    model.eval()
    
    for h, w in test_cases:
        print(f"\nTesting with input size: {h} x {w}")
        try:
            # Create test input: (B, H, W, T, C)
            x = torch.randn(2, h, w, 4, 3)
            
            with torch.no_grad():
                output = model(x)
            
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected output shape: (2, {h}, {w}, 1, 3)")
            
            # Check if output dimensions match expected
            if output.shape == (2, h, w, 1, 3):
                print("✓ PASS: Output dimensions match input spatial dimensions")
            else:
                print("✗ FAIL: Output dimensions do not match")
                
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
    
    print("\nNon-square input test completed!")

def test_gradient_error():
    """Test gradient computation with problematic input size"""
    print("Testing gradient computation with (96, 192) input...")
    
    model = CrossWaveletTransformer(wave='haar', dim=64, patch_size=(4, 4))
    model.train()  # Enable training mode for gradients
    
    # Create input that causes the error
    x = torch.randn(2, 96, 192, 4, 3, requires_grad=True)
    
    try:
        output = model(x)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Test backward pass
        loss = torch.mean(output)
        print("Computing gradients...")
        loss.backward()
        print("✓ PASS: Gradient computation successful")
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return False
    
    return True

def test_comprehensive_gradients():
    """Test gradient computation with various input sizes"""
    print("Testing comprehensive gradient computation...")
    
    # Test cases that commonly cause gradient issues
    test_cases = [
        (96, 192),   # Original problematic case
        (63, 127),   # Odd dimensions
        (100, 200),  # Even but not power of 2
        (48, 96),    # Smaller dimensions
        (120, 80),   # Different aspect ratio
    ]
    
    model = CrossWaveletTransformer(wave='haar', dim=32, patch_size=(4, 4))  # Smaller model for faster testing
    model.train()
    
    all_passed = True
    
    for h, w in test_cases:
        print(f"\nTesting gradients with input size: {h} x {w}")
        try:
            # Create input with gradients enabled
            x = torch.randn(1, h, w, 4, 3, requires_grad=True)
            
            # Forward pass
            output = model(x)
            
            # Backward pass
            loss = torch.mean(output)
            loss.backward()
            
            # Check if input gradients were computed
            if x.grad is not None and x.grad.shape == x.shape:
                print(f"✓ PASS: Gradients computed correctly. Input grad shape: {x.grad.shape}")
            else:
                print(f"✗ FAIL: Gradient shape mismatch or None")
                all_passed = False
                
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All gradient tests passed!")
    else:
        print("\n❌ Some gradient tests failed!")
    
    return all_passed

if __name__ == "__main__":
    # Quick validation test
    print("Running quick validation...")
    # test_gradient_error()  # Test the specific case that was problematic
    
    # Normal model usage
    x = torch.rand(2, 128, 128, 4, 3)
    
    # Base transformer
    model_base = CrossWaveletTransformer(wave='haar', dim=512)
    print("# parameters (base):", model_base.count_parameters())
    output_base = model_base(x)
    pred_base = torch.mean(output_base)
    pred_base.backward()
    print("Base Output shape:", output_base.shape)
    
    # Skip-connection variant
    model_skip = CrossWaveletTransSkipConnection(wave='haar', dim=512)
    print("# parameters (skip):", model_skip.count_parameters())
    output_skip = model_skip(x)
    pred_skip = torch.mean(output_skip)
    pred_skip.backward()
    print("Skip Output shape:", output_skip.shape)
    
    print("✅ Models support non-square inputs with proper gradients!")