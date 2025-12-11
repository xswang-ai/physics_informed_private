"""
Codes for backpropagation of wavelet transforms are referenced from https://github.com/YehLi/ImageNetModel/blob/main/classification/torch_wavelets.py 
"""

import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange

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
        ctx.is_cat_format = (x.dim() == 4)

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
            if ctx.is_cat_format:
                # Input was (B, 4*C, H, W) so recover channel count from original shape
                C = ctx.shape[1] // 4
            else:
                # Input was stacked (B, 4, C, H, W)
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
            # Restore original layout to keep autograd shapes consistent with the forward input
            if ctx.is_cat_format:
                dx = dx.reshape(ctx.shape[0], 4 * C, dx.shape[-2], dx.shape[-1])
            else:
                dx = dx.reshape(ctx.shape[0], 4, C, dx.shape[-2], dx.shape[-1])
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

