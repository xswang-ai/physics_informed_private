# ---------------------------------------------------------------------------------------------
# Author: Xuesong
# Date: 08/28/2025
# This code is developed with reference to the following GitHub repo:
#  https://github.com/SiaK4/HFS_ResUNet/blob/main/Models/ResUnet.py
# ---------------------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveSwish(nn.Module):
    def __init__(self,beta_init=1.0):
        super(AdaptiveSwish,self).__init__()

        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)

class AdaptiveTanh(nn.Module):
    def __init__(self,alpha_init=1.0):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.parameter(torch.tensor(alpha_init))

    def forward(self,x):
        return torch.tanh(self.alpha*x)

class Rowdy(nn.Module):
    def __init__(self, beta_init=1.0, cos_terms=2):
        super(Rowdy, self).__init__()
        self.amplitudes = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(cos_terms)])
        self.frequencies = nn.ParameterList([nn.Parameter(0.1*torch.ones(1)) for _ in range(cos_terms)])

        self.base_frequencies = torch.arange(10, 10*(cos_terms+1), 10, dtype=torch.float32)

        self.beta = nn.Parameter(torch.tensor(beta_init))

def get_activation(activation_name):
    if activation_name =='adaptive_swish':
        return AdaptiveSwish()
    if activation_name =='adaptive_tanh':
        return AdaptiveTanh()
    if activation_name =='Rowdy':
        return Rowdy()
    if activation_name =='GELU':
        return nn.GELU(approximate='tanh')

class featscale(nn.Module):
    def __init__(self, patch_size,channels):
        super(featscale, self).__init__()
        self.patch_size = patch_size
        self.lambda1 = nn.Parameter(1.0*torch.ones(1,channels,1,1,1))
        self.lambda2 = nn.Parameter(1.0*torch.ones(1,channels,1,1,1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reshape into patches of shape (batch_size, channels, num_patches, patch_size, patch_size)
        X_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        num_patches = (height//self.patch_size)* (width//self.patch_size)
        X_patches = X_patches.reshape(batch_size, channels, num_patches, self.patch_size, self.patch_size)
        X_mean_patch = X_patches.mean(dim=2)
        X_mean_expanded = X_mean_patch.unsqueeze(2).expand(-1, -1, num_patches, -1, -1)
        
        #Generate X_d and X_h
        X_d = X_mean_expanded
        X_h = X_patches - X_d

        # #Combine X_d and X_h
        X = X_patches + self.lambda1*X_d + self.lambda2*X_h
        X = X.reshape(batch_size, channels, height//self.patch_size, width//self.patch_size, self.patch_size, self.patch_size)
        X = X.permute(0,1,2,4,3,5).reshape(batch_size, channels, height, width)
        return X

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh')
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
            #nn.GELU(approximate='tanh')

        )
    def forward(self, x):
        shortcut = self.skip(x)
        out = self.residual(x)
        return out+shortcut

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock2, self).__init__()
        self.in_channels = in_channels

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            # nn.GELU(approximate='tanh'),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            # nn.GELU(approximate='tanh')
            activation
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(1,out_channels)
            #nn.GELU(approximate='tanh')

        )
    def forward(self, x):
        shortcut = self.skip(x)
        out = self.residual(x)
        return out+shortcut

def get_model_config(target_params='medium', custom_width_multiplier=None):
    """
    Get model configuration based on target parameter count.
    
    Args:
        target_params: Target parameter count. Options:
            - 'small': ~16M parameters (width_multiplier ~0.5)
            - 'medium': ~32-40M parameters (power-of-two preset)
            - 'large': ~64M parameters (width_multiplier ~1.0, default)
            - int: Custom target parameter count in millions
        custom_width_multiplier: Optional custom width multiplier (overrides target_params if provided)
    
    Returns:
        Dictionary with 'features', 'bottleneck_feature', 'width_multiplier'
    """
    # Base configuration (roughly 64M parameters at multiplier 1.0)
    base_features = [64, 128, 256, 512, 512]
    base_bottleneck = 1024
    
    preset_features = None
    preset_bottleneck = None

    if custom_width_multiplier is not None:
        width_multiplier = custom_width_multiplier
    elif isinstance(target_params, str):
        target = target_params.lower()
        if target == 'small':
            width_multiplier = 0.5  # ~16M params
        elif target == 'medium':
            # Explicit power-of-two preset to land in ~32-40M range
            preset_features = [32, 64, 128, 256, 512]
            preset_bottleneck = 1024
            width_multiplier = None
        elif target in ['large', 'huge']:
            width_multiplier = 1.0  # ~64M params (original)
        else:
            raise ValueError(f"Unknown target_params string: {target_params}. Use 'small', 'medium', 'large', or a number.")
    elif isinstance(target_params, (int, float)):
        # Interpolate width multiplier based on target parameter count
        if target_params <= 20:
            width_multiplier = 0.5  # ~16M
        elif target_params <= 40:
            width_multiplier = 0.8  # ~32M
        else:
            width_multiplier = 1.0  # ~64M
    else:
        width_multiplier = 1.0  # Default to large
    
    if preset_features is not None:
        features = preset_features
        bottleneck_feature = preset_bottleneck
    else:
        # Scale features and bottleneck
        features = [int(f * width_multiplier) for f in base_features]
        bottleneck_feature = int(base_bottleneck * width_multiplier)
    
    # Ensure minimum channel sizes
    features = [max(f, 16) for f in features]  # At least 16 channels
    bottleneck_feature = max(bottleneck_feature, 64)  # At least 64 bottleneck channels
    
    # Round to nearest 8 for better GPU utilization
    features = [((f + 4) // 8) * 8 for f in features]
    bottleneck_feature = ((bottleneck_feature + 4) // 8) * 8
    
    return {
        'features': features,
        'bottleneck_feature': bottleneck_feature,
        'width_multiplier': width_multiplier
    }


class ResUNet(nn.Module):
    def __init__(self, in_c, out_c, features=None, bottleneck_feature=None, 
                 patch_size_enc=[16,8,4,2,1], patch_size_dec=[16,8,4,2,1],
                 activation_name='GELU', device=torch.device('cpu'),
                 target_params='large', width_multiplier=None):
        """
        ResUNet with configurable model size.
        
        Args:
            in_c: Input channels
            out_c: Output channels
            features: List of feature dimensions for encoder/decoder. If None, determined by target_params or width_multiplier
            bottleneck_feature: Bottleneck feature dimension. If None, determined by target_params or width_multiplier
            patch_size_enc: Patch sizes for encoder featscale layers
            patch_size_dec: Patch sizes for decoder featscale layers
            activation_name: Activation function name
            device: Device to run on
            target_params: Target parameter count ('small'~16M, 'medium'~32M, 'large'~64M, or number)
            width_multiplier: Custom width multiplier (overrides target_params if provided)
        """
        super(ResUNet, self).__init__()
        
        # Get model configuration
        if features is None or bottleneck_feature is None:
            config = get_model_config(target_params, width_multiplier)
            if features is None:
                features = config['features']
            if bottleneck_feature is None:
                bottleneck_feature = config['bottleneck_feature']
            self.width_multiplier = config.get('width_multiplier', None)
        else:
            # If both features and bottleneck are explicitly provided, width_multiplier is None
            self.width_multiplier = None
        
        self.in_c = in_c
        self.out_c = out_c
        self.features = features  # Store features for reference
        self.bottleneck_feature = bottleneck_feature  # Store bottleneck for reference
        self.lamb1_history = []
        self.lamb2_history = []
        self.activation = get_activation(activation_name)
        self.device = device
        #Encoder and featscale
        self.encoder = nn.ModuleList()
        self.featscale = nn.ModuleList()
        self.featscale2 = nn.ModuleList()
        self.featscale3 = nn.ModuleList()

        num_layers = len(features)  # Assuming featscale lists match encoder layers
        self.w1 = nn.Parameter(0.1*torch.ones(num_layers))
        self.w2 = nn.Parameter(0.1*torch.ones(num_layers))
        self.w3 = nn.Parameter(0.1*torch.ones(num_layers))

        for i,feature in enumerate(features):
            self.encoder.append(ResidualBlock2(in_c,feature,self.activation))
            self.featscale.append(featscale(patch_size_enc[i],feature))

            self.featscale2.append(featscale(max(patch_size_enc[i]//2,1),feature))
            self.featscale3.append(featscale(max(patch_size_enc[i]//4,1),feature))

            in_c = feature

        # Bottleneck layer
        self.bottleneck = ResidualBlock2(features[-1], bottleneck_feature,self.activation)
        self.fs_bottleneck = featscale(patch_size=1, channels=bottleneck_feature)

        #Upsample and Decoder and featscale
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.featscale_up = nn.ModuleList()
        self.featscale_up2 = nn.ModuleList()
        self.featscale_up3 = nn.ModuleList()

        for i, feature in enumerate(reversed(features)):
            self.upsample.append(
                nn.ConvTranspose2d(bottleneck_feature, bottleneck_feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock(bottleneck_feature+feature, feature,self.activation))
            bottleneck_feature = feature

            self.featscale_up.append(featscale(patch_size_dec[-i-1],feature))
            self.featscale_up2.append(featscale(max(patch_size_dec[-i-1]//2,1),feature))
            self.featscale_up3.append(featscale(max(patch_size_dec[-i-1]//4,1),feature))
        
        self.final_conv = nn.Conv2d(features[0],self.out_c,kernel_size=1)

    def save_lambdas(self):
        self.lamb1_history.append(self.lamb1.item())
        self.lamb2_history.append(self.lamb2.item())


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid
    
    def forward(self, x):
        # Support inputs shaped (B, H, W, C) or (B, H, W, T, C)
        if x.dim() == 4:
            B, H, W, C = x.shape
            T = 1
        elif x.dim() == 5:
            B, H, W, T, C = x.shape
        else:
            raise ValueError(f'Unsupported input shape {x.shape}')

        x = x.view(B, H, W, -1)           # B, H, W, T*C
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T*C, H, W)
        
        #Downsampling path
        skip_connections = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            x1 = self.featscale[i](x)
            x2 = self.featscale2[i](x)
            x3 = self.featscale3[i](x)
            x = self.w1[i]*x1 + self.w2[i]*x2 + self.w3[i]*x3
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        x = self.bottleneck(x)
        x = self.fs_bottleneck(x)

        #Upsampling path
        skip_connections = skip_connections[::-1]
        for up in range(len(self.decoder)):
            x = self.upsample[up](x)
            x = torch.cat((x, skip_connections[up]),dim=1)
            x = self.decoder[up](x)
        out = self.final_conv(x)

        # reshape back to (B, C_out, H, W) -> (B, H, W, T, C)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, H, W, -1, self.out_c)
        # Collapse time/channel if single-step, single-channel
        if out.shape[-1] == 1 and out.shape[-2] == 1:
            return out[..., 0, 0]
        if out.shape[-2] == 1:
            return out.squeeze(-2)
        return out
       
    def set_input(self,input_data):
        x = input_data[:,0:20,:,:]
        y = input_data[:,20:25,:,:]
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y


    def get_latent_by_index(self, x, index):
        """ 
        Get the latent representation from the input to the (index -1)-th block of UNet
        """
        if index < 5: # convolution blocks
             # absort the time dimension into the channel dimensionx = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
            B, H, W, T, C = x.shape
            x = x.view(*x.shape[:-2], -1)           #### B, H, W, T*C
            grid = self.get_grid(x)
            x = torch.cat((x, grid), dim=-1)        #### B, H, W, T*C +2
            x = x.permute(0, 3, 1, 2).contiguous() # (B, T*C+2, H, W)
            if index == 0:
                return x.permute(0, 2, 3, 1).unsqueeze(-2)
            #Downsampling path
            skip_connections = []
            for i, down in enumerate(self.encoder[:index]): # :index because the index is the (index -1)
                x = down(x)
                x1 = self.featscale[i](x)
                x2 = self.featscale2[i](x)
                x3 = self.featscale3[i](x)
                x = self.w1[i]*x1 + self.w2[i]*x2 + self.w3[i]*x3
                skip_connections.append(x)
                x = F.max_pool2d(x, kernel_size=2)
            return x.permute(0, 2, 3, 1).unsqueeze(-2)
        else:
            pass

    def get_testing_block_by_index(self, index, x):
        i = index # (index -1) because the 
        down = self.encoder[i]
        # preprocess in the input
        x = x.squeeze(-2) # (B, H, W, T, C) -> (B, H, W, C)
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        if x.shape[1] > down.in_channels: # very naive way to handle different channels
            x = x[:, :down.in_channels, :, :]
        x = down(x)
        featscale = self.featscale[i]
        featscale2 = self.featscale2[i]
        featscale3 = self.featscale3[i]
        x = self.w1[i]*featscale(x) + self.w2[i]*featscale2(x) + self.w3[i]*featscale3(x)
        x = x.permute(0, 2, 3, 1).unsqueeze(-2) # (B, C, H, W) -> (B, H, W, 1, C) 
        return x



class featscale2(nn.Module):
    def __init__(self, patch_size,channels):
        super(featscale2, self).__init__()
        self.patch_size = patch_size
        self.lambda1 = nn.Parameter(torch.ones(1,channels,1,1,1))
        self.lambda2 = nn.Parameter(torch.ones(1,channels,1,1,1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reshape into patches of shape (batch_size, channels, num_patches, patch_size, patch_size)
        X_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        num_patches = (height//self.patch_size)* (width//self.patch_size)
        X_patches = X_patches.reshape(batch_size, channels, num_patches, self.patch_size, self.patch_size)
        X_mean_patch = X_patches.mean(dim=2)
        X_mean_expanded = X_mean_patch.unsqueeze(2).expand(-1, -1, num_patches, -1, -1)
        
        #Generate X_d and X_h
        X_d = X_mean_expanded
        X_h = X_patches - X_d

        # #Combine X_d and X_h
        X = X_patches + self.lambda1*X_d + self.lambda2*X_h
        X = X.reshape(batch_size, channels, height//self.patch_size, width//self.patch_size, self.patch_size, self.patch_size)
        X = X.permute(0,1,2,4,3,5).reshape(batch_size, channels, height, width)
        return X


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = 3
    T_in = 1
    T_ar = 1

    # Example 1: Use predefined size ('small', 'medium', 'large')
    print("=" * 60)
    print("Example 1: Small model (~16M parameters)")
    print("=" * 60)
    model_small = ResUNet(in_c=n_channels, out_c=n_channels, 
                         target_params='small',
                         device=device).to(device)
    n_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Small model parameters: {n_params_small:,} ({n_params_small/1e6:.2f}M)")
    print(f"Features: {model_small.features}")
    print(f"Bottleneck: {model_small.bottleneck_feature}")
    print(f"Width multiplier: {model_small.width_multiplier}")
    
    print("\n" + "=" * 60)
    print("Example 2: Medium model (~32-40M parameters, power-of-two preset)")
    print("=" * 60)
    model_medium = ResUNet(in_c=n_channels, out_c=n_channels, 
                          target_params='medium',
                          device=device).to(device)
    n_params_medium = sum(p.numel() for p in model_medium.parameters())
    print(f"Medium model parameters: {n_params_medium:,} ({n_params_medium/1e6:.2f}M)")
    print(f"Features: {model_medium.features}")
    print(f"Bottleneck: {model_medium.bottleneck_feature}")
    print(f"Width multiplier: {model_medium.width_multiplier}")
    
    print("\n" + "=" * 60)
    print("Example 3: Large model (~64M parameters, default)")
    print("=" * 60)
    model_large = ResUNet(in_c=n_channels, out_c=n_channels, 
                         target_params='large',
                         device=device).to(device)
    n_params_large = sum(p.numel() for p in model_large.parameters())
    print(f"Large model parameters: {n_params_large:,} ({n_params_large/1e6:.2f}M)")
    print(f"Features: {model_large.features}")
    print(f"Bottleneck: {model_large.bottleneck_feature}")
    print(f"Width multiplier: {model_large.width_multiplier}")
    
    print("\n" + "=" * 60)
    print("Example 4: Custom width multiplier")
    print("=" * 60)
    model_custom = ResUNet(in_c=n_channels, out_c=n_channels, 
                          width_multiplier=0.6,  # Custom size
                          device=device).to(device)
    n_params_custom = sum(p.numel() for p in model_custom.parameters())
    print(f"Custom model parameters: {n_params_custom:,} ({n_params_custom/1e6:.2f}M)")
    print(f"Features: {model_custom.features}")
    print(f"Bottleneck: {model_custom.bottleneck_feature}")
    print(f"Width multiplier: {model_custom.width_multiplier}")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    x = torch.rand(2, 256, 256, T_in, n_channels)
    y = model_small(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Parameter Count Summary")
    print("=" * 60)
    print(f"Small  (target_params='small'):     {n_params_small/1e6:6.2f}M")
    print(f"Medium (target_params='medium'):    {n_params_medium/1e6:6.2f}M")
    print(f"Large  (target_params='large'):     {n_params_large/1e6:6.2f}M")
    print(f"Custom (width_multiplier=0.6):      {n_params_custom/1e6:6.2f}M")
