import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable, gradcheck



ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class DWT_Function(Function): 
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh): 
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh) 
        ctx.shape = x.shape 

        dim = x.shape[1] 
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim) 
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1) 
        return x

    @staticmethod
    def backward(ctx, dx): 
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape 
            dx = dx.view(B, 4, -1, H//2, W//2)  

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2) 
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1) 
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0) 
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None



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
        self.filters = self.filters  

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
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
        
        self.w_ll = self.w_ll 
        self.w_lh = self.w_lh 
        self.w_hl = self.w_hl 
        self.w_hh = self.w_hh 

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)

elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)


class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()



class BaseMask(object):
    @property
    def bool_matrix(self):
        """Return a bool (uint8) matrix with 1s to all places that should be
        kept."""
        raise NotImplementedError()

    @property
    def float_matrix(self):
        """Return the bool matrix as a float to be used as a multiplicative
        mask for non softmax attentions."""
        if not hasattr(self, "_float_matrix"):
            with torch.no_grad():
                self._float_matrix = self.bool_matrix.float()
        return self._float_matrix

    @property
    def lengths(self):
        """If the matrix is of the following form

            1 1 1 0 0 0 0
            1 0 0 0 0 0 0
            1 1 0 0 0 0 0

        then return it as a vector of integers

            3 1 2.
        """
        if not hasattr(self, "_lengths"):
            with torch.no_grad():
                lengths = self.bool_matrix.long().sum(dim=-1)
                # make sure that the mask starts with 1s and continues with 0s
                # this should be changed to something more efficient, however,
                # I chose simplicity over efficiency since the LengthMask class
                # will be used anyway (and the result is cached)
                m = self.bool_matrix.view(-1, self.shape[-1])
                for i, l in enumerate(lengths.view(-1)):
                    if not torch.all(m[i, :l]):
                        raise ValueError("The mask is not a length mask")
                self._lengths = lengths
        return self._lengths

    @property
    def shape(self):
        """Return the shape of the boolean mask."""
        return self.bool_matrix.shape

    @property
    def additive_matrix(self):
        """Return a float matrix to be added to an attention matrix before
        softmax."""
        if not hasattr(self, "_additive_matrix"):
            with torch.no_grad():
                self._additive_matrix = torch.log(self.bool_matrix.float())
        return self._additive_matrix

    @property
    def additive_matrix_finite(self):
        """Same as additive_matrix but with -1e24 instead of infinity."""
        if not hasattr(self, "_additive_matrix_finite"):
            with torch.no_grad():
                self._additive_matrix_finite = (
                        (~self.bool_matrix).float() * (-1e24)
                )
        return self._additive_matrix_finite

    @property
    def all_ones(self):
        """Return true if the mask is all ones."""
        if not hasattr(self, "_all_ones"):
            with torch.no_grad():
                self._all_ones = torch.all(self.bool_matrix)
        return self._all_ones

    @property
    def lower_triangular(self):
        """Return true if the attention is a triangular causal mask."""
        if not hasattr(self, "_lower_triangular"):
            self._lower_triangular = False
            with torch.no_grad():
                try:
                    lengths = self.lengths
                    if len(lengths.shape) == 1:
                        target = torch.arange(
                            1,
                            len(lengths)+1,
                            device=lengths.device
                        )
                        self._lower_triangular = torch.all(lengths == target)
                except ValueError:
                    pass
        return self._lower_triangular


class FullMask(BaseMask):
    """Thin wrapper over a pytorch tensor that provides the BaseMask
    interface.

    The arguments can be given both by keyword arguments and positional
    arguments. To imitate function overloading, the constructor checks the type
    of the first argument and if it is a tensor it treats it as the mask.
    otherwise it assumes that it was the N argument.

    Arguments
    ---------
        mask: The mask as a PyTorch tensor.
        N: The rows of the all True mask to be created if the mask argument is
           not provided.
        M: The columns of the all True mask to be created if the mask argument
           is not provided. If N is given M defaults to N.
        device: The device to create the mask in (defaults to cpu)
    """
    def __init__(self, mask=None, N=None, M=None, device="cpu"):
        # mask is a tensor so we ignore N and M
        if mask is not None and isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                raise ValueError("FullMask expects the mask to be bool")
            with torch.no_grad():
                self._mask = mask.clone()
            return

        # mask is an integer, N is an integer and M is None so assume they were
        # passed as N, M
        if mask is not None and M is None and isinstance(mask, int):
            M = N
            N = mask

        if N is not None:
            M = M or N
            with torch.no_grad():
                self._mask = torch.ones(N, M, dtype=torch.bool, device=device)
            self._all_ones = True
            return

        raise ValueError("Either mask or N should be provided")

    @property
    def bool_matrix(self):
        return self._mask


class LengthMask(BaseMask):
    """Provide a BaseMask interface for lengths. Mostly to be used with
    sequences of different lengths.

    Arguments
    ---------
        lengths: The lengths as a PyTorch long tensor
        max_len: The maximum length for the mask (defaults to lengths.max())
        device: The device to be used for creating the masks (defaults to
                lengths.device)
    """
    def __init__(self, lengths, max_len=None, device=None):
        self._device = device or lengths.device
        with torch.no_grad():
            self._lengths = lengths.clone().to(self._device)
        self._max_len = max_len or self._lengths.max()

        self._bool_matrix = None
        self._all_ones = torch.all(self._lengths == self._max_len).item()

    @property
    def bool_matrix(self):
        if self._bool_matrix is None:
            with torch.no_grad():
                indices = torch.arange(self._max_len, device=self._device)
                self._bool_matrix = (
                        indices.view(1, -1) < self._lengths.view(-1, 1)
                )
        return self._bool_matrix


class TriangularCausalMask(LengthMask):
    """A square matrix with everything masked out above the diagonal.

    Arguments
    ---------
        N: The size of the matrix
        device: The device to create the mask in (defaults to cpu)
    """
    def __init__(self, N, device="cpu"):
        lengths = torch.arange(1, N+1, device=device)
        super(TriangularCausalMask, self).__init__(lengths, N, device)
        self._lower_triangular = True


class Spectral_Attention_Structured_Mesh_2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, sr_ratio=1, is_filter=True):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor

        self.w1 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size))

        dim = self.hidden_size
        self.num_heads = num_blocks
        head_dim = dim // num_blocks 
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.is_filter = is_filter

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
        )
        if self.is_filter:
            self.filter = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
                nn.BatchNorm2d(dim),
            )
        
        self.qkv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3)
        )
        self.proj = nn.Linear(dim+dim//4, dim)
        
        self.merge_linear = nn.Linear(dim+dim, dim)
        
        self.apply(self._init_weights)
        self.inner_attention = LinearAttention(dim)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x_ori = x
        B, N, C = x.shape 
        
        ####################### Fourier Attention ##############################
        x = x.reshape(B, H, W, C) 
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = W // 2 + 1  
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1) 
        x = torch.view_as_complex(x).reshape(B, x.shape[1], x.shape[2], C) 
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x_fft = x + x_ori
        
        ####################### Wavelet Attention ##############################
        x = x_ori.view(B, H, W, C).permute(0, 3, 1, 2) 
        x_dwt = self.dwt(self.reduce(x)) 
        Bd, Cd, Hd, Wd = x_dwt.shape 
        if self.is_filter:
            x_dwt = self.filter(x_dwt) 
        
        attn_mask = FullMask(Hd*Wd, device=x.device)
        length_mask = LengthMask(x_dwt.new_full((B,), Hd*Wd, dtype=torch.int64))
        kv = x_dwt.reshape(B, C, -1).permute(0, 2, 1) 
        kv = self.qkv(kv).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = kv[0], kv[1], kv[2]
        x_dwt_attn = self.inner_attention(
            q, 
            k,
            v,
            attn_mask,
            length_mask,
            length_mask
        ).view(B, N, -1).reshape(B, Hd, Wd, C).permute(0, 3, 1, 2) 
        
        x_idwt = self.idwt(x_dwt_attn)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)  
        x_wave = self.proj(torch.cat([x_ori, x_idwt], dim=-1)) 
        
#########################  Gated Fusion #####################################
        x_merged = torch.cat([x_fft, x_wave], dim=-1)
        weight_map = F.sigmoid(self.merge_linear(x_merged))
        x_final_output = weight_map*x_fft + (1-weight_map)*x_wave
        
        return x_final_output



class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class SAOT_Transformer_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85,
            is_filter= True
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Spectral_Attention_Structured_Mesh_2D(hidden_dim, num_heads, sr_ratio=1, is_filter = is_filter)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        
        self.H = H
        self.W = W
    
    def forward(self, fx):  
        fx = self.Attn(self.ln_1(fx), self.H, self.W) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class SAOTModel(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 H=85,
                 W=85,
                 is_filter= True
                 ):
        super(SAOTModel, self).__init__()
        self.__name__ = 'SAOT_Transformer_block'
        self.H = H
        self.W = W
        self.pad_h = 0
        self.pad_w = 0
        if self.H%2 ==1:
            self.H += 1
            self.pad_h = 1
        if self.W%2 ==1:
            self.W += 1
            self.pad_w = 1
        
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:  
            self.pos = self.get_grid()  
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([SAOT_Transformer_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=self.H,
                                                      W=self.W,
                                                      last_layer=(_ == n_layers - 1),
                                                      is_filter = is_filter)
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float)) 

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1]) 
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1]) 
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()  
        
        return pos 

    def forward(self, x, T=None):  #x: B, H, W, 2+input_dim;    T: B,1; 
       # print('x, fx: ', x.shape, fx.shape) #x, fx:  torch.Size([4, 85, 85, 2]) torch.Size([4, 85, 85, 1])
        x_ori = x
        x = F.pad(x, [0,0,0,self.pad_w, 0,self.pad_h])
        
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        
        if self.unified_pos:     
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        

        fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1) 
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
        
        for block in self.blocks: 
            fx = block(fx)  
        
        fx = torch.reshape(fx, (fx.shape[0], self.H, self.W, fx.shape[-1]))

        if self.pad_h !=0:
            fx = fx[:, :-self.pad_h, :, :]  
        if self.pad_w !=0:
            fx = fx[:, :, :-self.pad_w, :] 
            
        return fx




if __name__ == '__main__':
    model = SAOTModel(space_dim=2,
                    n_layers=3,
                    n_hidden=64,
                    dropout=0.0,
                    n_head=4,
                    Time_Input=False,
                    mlp_ratio=1,
                    fun_dim=1,
                    out_dim=1,
                    slice_num=32,
                    ref=8,
                    unified_pos=0,
                    H=64, W=64,
                    is_filter=True)


    x = torch.randn(1, 64, 64, 3)
    out = model(x)
    print(out.shape)