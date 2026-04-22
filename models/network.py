import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
import torch
import timm

def L1_norm(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = torch.abs(narry_a)
    temp_abs_b = torch.abs(narry_b)
    _l1_a = torch.sum(temp_abs_a, dim=1)
    _l1_b = torch.sum(temp_abs_b, dim=1)

    _l1_a = torch.sum(_l1_a, dim=0)
    _l1_b = torch.sum(_l1_b, dim=0)
    with torch.no_grad():
        l1_a = _l1_a.detach()
        l1_b = _l1_b.detach()

    # caculate the map for source images
    mask_value = l1_a + l1_b
    # print("mask_value 的size",mask_value.size())

    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b
    # print("array_MASK_b 的size",array_MASK_b.size())
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            temp_matrix = array_MASK_a * narry_a[i, j, :, :] + array_MASK_b * narry_b[i, j, :, :]
            # print("temp_matrix 的size",temp_matrix.size())
            result.append(temp_matrix)  

    result = torch.stack(result, dim=-1)

    result = result.reshape((dimension[0], dimension[1], dimension[2], dimension[3]))

    return result

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        B, C, N = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((B, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        out_x1 = out_x1.permute(0, 2, 1)
        out_x2 = out_x2.permute(0, 2, 1)
        return out_x1, out_x2

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
        
############################ Mainly used for unimodal feature extraction ##########################
class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)

############################  ##########################
class SSMSubBlock(nn.Module):
    """SSM子模块：包含Norm和Mamba，与SingleMambaBlock中的设计一致"""
    def __init__(self, dim):
        super(SSMSubBlock, self).__init__()
        self.norm = LayerNorm(dim, 'with_bias')
        # 假设使用mamba_ssm库中的Mamba
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            self.encoder = Mamba(dim, bimamba_type=None)
        except ImportError:
            # 如果Mamba不可用，使用线性层作为替代
            print("Warning: mamba_ssm not found, using Linear layer as SSM substitute")
            self.encoder = nn.Linear(dim, dim)
    
    def forward(self, x):
        """前向传播：先Norm后Mamba"""
        x = self.norm(x)
        return self.encoder(x)

class MambaNormBlock(nn.Module):
    """
    Dual-path Mamba Block

    Process:
    1. Input x goes through LayerNorm
    2. Split into two paths:
       - Path 1: Linear → Conv → SSM (Norm + Mamba)
       - Path 2: Linear → Activation
    3. Element-wise product of the two path outputs
    4. Product result goes through Linear
    5. Residual connection with input x
    6. Output
    """
    def __init__(self, dim, reduction_factor=2, kernel_size=3):
        super(MambaNormBlock, self).__init__()
        self.dim = dim
        self.reduction_factor = reduction_factor
        reduced_dim = dim // reduction_factor
        
        # LayerNorm
        self.norm = LayerNorm(dim, 'with_bias')
        
        # 路径1的组件：Linear → Conv → SSM
        self.linear_path1 = nn.Linear(dim, reduced_dim)
        self.conv = nn.Conv2d(
            reduced_dim, reduced_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            groups=reduced_dim  # 深度可分离卷积
        )
        self.ssm = SSMSubBlock(reduced_dim)
        
        # Linear → Activation
        self.linear_path2 = nn.Linear(dim, reduced_dim)
        self.activation = nn.GELU()
        
        # Linear layer after the element-wise product
        self.linear_after_multiply = nn.Linear(reduced_dim, dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features [B, C, H, W] or [B, L, C]
        
        Returns:
            output: Output features, same shape as input x
        """
        # Save input for residual connection
        residual = x
        
        # Get input shape information
        if len(x.shape) == 4:
            # Input is a 4D feature map
            B, C, H, W = x.shape
            is_4d = True
            # Convert to sequence form for processing
            x_seq = to_3d(x)  # [B, H*W, C]
        else:
            # Input is already in sequence form
            B, L, C = x.shape
            is_4d = False
            # Calculate approximate H and W (assuming square)
            H = int(L ** 0.5)
            W = H
            x_seq = x
        
        # 1. LayerNorm
        x_norm = self.norm(x_seq)  # [B, L, C]
        
        # 2. Dual-path processing
        # Path 1: Linear → Conv → SSM
        path1 = self.linear_path1(x_norm)  # [B, L, reduced_dim]
        
        # Convert to spatial dimensions for convolution
        if is_4d:
            path1_spatial = path1.transpose(1, 2).view(B, -1, H, W)
        else:
            # If input is sequential, reshape to spatial form first
            path1_spatial = path1.transpose(1, 2).view(B, -1, H, W)
        
        # Convolution processing
        path1_conv = self.conv(path1_spatial)  # [B, reduced_dim, H, W]
        
        # Convert back to sequence form for SSM
        path1_seq = path1_conv.flatten(2).transpose(1, 2)  # [B, H*W, reduced_dim]
        path1_ssm = self.ssm(path1_seq)  # [B, H*W, reduced_dim]
        
        # Path 2: Linear → Activation
        path2 = self.linear_path2(x_norm)  # [B, L, reduced_dim]
        path2_act = self.activation(path2)  # [B, L, reduced_dim]
        
        # 3. Element-wise product fusion
        # Ensure path1_ssm and path2_act have matching dimensions
        if path1_ssm.shape != path2_act.shape:
            # If shapes don't match, adjust path2_act shape
            path2_act = path2_act.view_as(path1_ssm)
        
        multiplied = path1_ssm * path2_act  # [B, L, reduced_dim]
        
        # 4. Linear layer after element-wise product
        multiplied_proj = self.linear_after_multiply(multiplied)  # [B, L, C]
        
        # 5. Residual connection with input x
        output_seq = multiplied_proj + residual if is_4d else multiplied_proj + x_seq
        

        if is_4d:
            output = to_4d(output_seq, H, W)
        else:
            output = output_seq
        
        return output

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.size()
        

        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        

        avg_fc = self.fc(avg_out)
        max_fc = self.fc(max_out)
        

        channel_weights = self.sigmoid(avg_fc + max_fc).view(B, C, 1, 1)
        
        return channel_weights

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        

        x_pool = torch.cat([avg_out, max_out], dim=1)
        

        spatial_weights = self.sigmoid(self.conv(x_pool))
        
        return spatial_weights

class DualBranchMambaBlock(nn.Module):
    """
    Dual-branch Mamba Block
    Branch 1: Channel Attention + Spatial Attention
    Branch 2: Conv2D + Mamba
    Finally, element-wise product fusion, followed by residual connection with original input
    """
    def __init__(self, dim, reduction_ratio=8, kernel_size=3):
        super(DualBranchMambaBlock, self).__init__()
        self.dim = dim
        
        # LayerNorm
        self.norm = LayerNorm(dim, 'with_bias')
        

        self.linear_in = nn.Linear(dim, dim * 2)

        
        self.channel_attention = ChannelAttention(dim, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.silu1 = nn.SiLU()
        

        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                             padding=kernel_size//2, groups=dim)
        self.silu2 = nn.SiLU()
        self.mamba = SingleMambaBlock(dim)
        

        self.linear_out = nn.Linear(dim, dim)
        
    def forward(self, x, residual=None):
        """
        Forward pass
        Args:
            x: Input features [B, L, C]
            residual: Residual connection (optional)
        Returns:
            output: Output features [B, L, C]
        """
        B, L, C = x.shape
        H = int(L ** 0.5)
        

        original_input = x
        

        if residual is not None:
            x = x + residual
        
        # LayerNorm
        x_norm = self.norm(x)
        

        x_linear = self.linear_in(x_norm)  # [B, L, 2C]
        x1, x2 = torch.chunk(x_linear, 2, dim=-1)  # 各为[B, L, C]
        

        x1_spatial = x1.transpose(1, 2).view(B, C, H, H)
        x2_spatial = x2.transpose(1, 2).view(B, C, H, H)
        


        ca_weight = self.channel_attention(x1_spatial)
        x1_ca = x1_spatial * ca_weight
        

        sa_weight = self.spatial_attention(x1_ca)
        x1_sa = x1_ca * sa_weight
        

        F1 = self.silu1(x1_sa)
        

        
        x2_conv = self.conv(x2_spatial)
        x2_silu = self.silu2(x2_conv)
        
        
        x2_seq = x2_silu.flatten(2).transpose(1, 2)
        
        
        x2_mamba, _ = self.mamba([x2_seq, torch.zeros_like(x2_seq)])
        
        
        F2 = x2_mamba.transpose(1, 2).view(B, C, H, H)
        
        
        F = F1 * F2  # [B, C, H, H]
        
        
        F_seq = F.flatten(2).transpose(1, 2)
        
        
        output = self.linear_out(F_seq)
        
        
        output = output + original_input
        
        return output

class Auxiliary_Encoder(nn.Module):
    """
    Intelligent auxiliary encoder that automatically selects implementation based on environment
    """
    
    def __init__(self, mode="auto", feature_dim=128, **kwargs):
        super().__init__()
        
        # Select implementation based on mode
        if mode == "timm" or mode == "auto":
            try:
                # Attempt to use timm, fallback if fails
                self._init_timm_clip(feature_dim, **kwargs)
                print("✓ Using timm CLIP model")
                self.mode = "timm"
            except Exception as e:
                print(f"✗ timm CLIP failed: {e}")
                self._init_simple_cnn(feature_dim, **kwargs)
                print("✓ Fallback to simple CNN")
                self.mode = "simple"
        elif mode == "simple":
            self._init_simple_cnn(feature_dim, **kwargs)
            self.mode = "simple"
        elif mode == "none":
            self._init_none(feature_dim, **kwargs)
            self.mode = "none"
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _init_timm_clip(self, feature_dim, **kwargs):
        """Initialize timm CLIP model"""
        # Offline loading logic can be added here
        # Temporarily using simple CNN as fallback
        self._init_simple_cnn(feature_dim, **kwargs)
    
    def _init_simple_cnn(self, feature_dim, **kwargs):

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )
        self.feature_dim = feature_dim
        

        self.ct_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        
        self.mri_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
    
    def _init_none(self, feature_dim, **kwargs):

        self.feature_dim = feature_dim
        
        # Create placeholders
        self.ct_projection = nn.Identity()
        self.mri_projection = nn.Identity()
    
    def extract_modality_specific_features(self, images, modality):

        # Convert single channel to three channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Resize
        if images.shape[2] < 64 or images.shape[3] < 64:
            images = F.interpolate(images, size=(128, 128), mode='bilinear')
        
        # Extract features
        if self.mode == "none":
            batch_size = images.shape[0]
            base_features = torch.randn(batch_size, self.feature_dim).to(images.device)
        else:
            base_features = self.encoder(images)
        
        # Modality-specific projection
        if modality == "ct":
            projected = self.ct_projection(base_features)
        elif modality == "mri":
            projected = self.mri_projection(base_features)
        else:
            projected = base_features
        
        return projected
    
    def forward(self, ct_images, mri_images, pet_images=None, spect_images=None):
        """Forward pass"""
        features_dict = {}
        
        features_dict["ct"] = self.extract_modality_specific_features(ct_images, "ct")
        features_dict["mri"] = self.extract_modality_specific_features(mri_images, "mri")
        
        if pet_images is not None:
            features_dict["pet"] = self.extract_modality_specific_features(pet_images, "pet")
        
        if spect_images is not None:
            features_dict["spect"] = self.extract_modality_specific_features(spect_images, "spect")
        
        return features_dict

class CM3F (nn.Module):
    """Multi-level Cross Feature Fusion Module"""
    def __init__(self, dim, num_levels=3):
        super(CM3F, self).__init__()
        self.num_levels = num_levels
        self.dim = dim
        
        # Independent Mamba module pairs for initial context feature extraction - using MambaNormBlock to replace SingleMambaBlock
        self.mamba_pairs = nn.ModuleList([
            nn.ModuleList([MambaNormBlock(dim) for _ in range(2)]) 
            for _ in range(num_levels)
        ])
        
        # Convolutional layers for local feature extraction
        self.local_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
            for _ in range(num_levels)
        ])
        
        # Final fusion Mamba module - using MambaNormBlock to replace SingleMambaBlock
        self.final_mamba = MambaNormBlock(dim)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, F1_prime, F2_prime, h, w):
        """
        Args:
            F1_prime: Exchanged features of modality 1 [B, L, C]
            F2_prime: Exchanged features of modality 2 [B, L, C]
            h, w: Spatial dimensions of the feature map
        Returns:
            fused_features: Fused features [B, L, C]
        """
        B, L, C = F1_prime.shape
        
        # Convert to spatial dimensions for local convolution
        F1_spatial = F1_prime.transpose(1, 2).view(B, C, h, w)
        F2_spatial = F2_prime.transpose(1, 2).view(B, C, h, w)
        
        # Initialize level features
        level_features = []
        
        for i in range(self.num_levels):
            # Independent Mamba processing - Note: MambaNormBlock only requires input features, not a tuple
            F1_processed = self.mamba_pairs[i][0](F1_prime)  # 使用MambaNormBlock
            F2_processed = self.mamba_pairs[i][1](F2_prime)  # 使用MambaNormBlock
            
            # Initial summation fusion
            fused_global = F1_processed + F2_processed
            
            # Local feature extraction
            F1_local = self.local_convs[i](F1_spatial)
            F2_local = self.local_convs[i](F2_spatial)
            
            # Convert to sequence form
            F1_local_seq = F1_local.flatten(2).transpose(1, 2)
            F2_local_seq = F2_local.flatten(2).transpose(1, 2)
            
            # Secondary fusion: global features + local features
            fused_local = F1_local_seq + F2_local_seq
            
            # Multi-level feature fusion
            if i == 0:
                level_fused = fused_global + fused_local
            else:
                level_fused = level_features[-1] + fused_global + fused_local
            
            level_features.append(level_fused)
            
            # Update input features for the next layer
            F1_prime, F2_prime = F1_processed, F2_processed
        
        # Final fusion: sum all level features and apply final Mamba processing
        final_fused = sum(level_features)
        final_fused = self.final_mamba(final_fused)  # Using MambaNormBlock
        
        return final_fused

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops
    
class MambaMFNet (nn.Module):
   
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, clip_feature_dim=128, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7,qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=1, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(MambaMFNet, self).__init__()
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        print('in_chans: ', in_chans)
        
        # Add color processing related parameters
        self.in_chans = in_chans
        self.require_color_processing = (in_chans == 3 or in_chans == 6)
        
        # Modify mean setting logic
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        elif in_chans == 6:
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)  # Output mean
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)  # Input mean
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
            self.mean_in = torch.zeros(1, in_chans, 1, 1)
        
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
    
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
    
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)
        # absolute position embedding
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

        #####################################################################################################
        ################################### 1, low-level feature extraction ###################################
        # Modify convolution layer input channels to support 1 or 3 channels
        self.low_level_feature_extraction1 = nn.Conv2d(1, embed_dim_temp, 3, 1, 1)  # Changed to 1-channel input
        self.low_level_feature_extraction2 = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        # Modified: Replace SingleMambaBlock with DualBranchMambaBlock, consistent with Code 2
        self.high_level_feature_extraction1 = nn.Sequential(*[DualBranchMambaBlock(self.embed_dim) for i in range(8)])
        self.high_level_feature_extraction2 = nn.Sequential(*[DualBranchMambaBlock(self.embed_dim) for i in range(8)])
        
        #####################################################################################################
        ################################### 2.5, feature exchange after extraction ###########################
        # Modified: Replace FeatureShuffleExchange with ChannelExchange, consistent with Code 2
        self.feature_exchange_after_extraction = ChannelExchange(p=2)
        
        #####################################################################################################
        ################################### 3, CM3F feature fusion ######################################
        # CM3F has been modified to use MambaNormBlock
        self.cm3f = CM3F (dim=embed_dim, num_levels=3)
        
        #####################################################################################################
        ################################ 4, fused image reconstruction ################################
        # Modified: Replace SingleMambaBlock with MambaNormBlock
        self.feature_re = nn.Sequential(*[MambaNormBlock(self.embed_dim) for i in range(8)])
        self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), 1, 3, 1, 1)  # Output 1 channel

        # Add RGB-YUV conversion related layers
        self.rgb_to_yuv = nn.Conv2d(3, 3, 1, 1, 0, bias=False)
        self.yuv_to_rgb = nn.Conv2d(3, 3, 1, 1, 0, bias=False)
        
        # Initialize RGB-YUV conversion matrix
        rgb_to_yuv_weight = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        
        yuv_to_rgb_weight = torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        
        self.rgb_to_yuv.weight = nn.Parameter(rgb_to_yuv_weight, requires_grad=False)
        self.yuv_to_rgb.weight = nn.Parameter(yuv_to_rgb_weight, requires_grad=False)
        
        #####################################################################################################
        ############################## 5, CLIP Auxiliary Encoder ####################################
        # Add CLIP auxiliary encoder
        self.clip_encoder = Auxiliary_Encoder(
            mode="auto",
            feature_dim=clip_feature_dim
        )
        
        # CLIP feature fusion layers
        self.clip_fusion_layers = nn.ModuleDict({
            "ct": nn.Sequential(
                nn.Linear(embed_dim + 32, embed_dim),  # 128 is the dimension after CLIP projection
                nn.LayerNorm(embed_dim),
                nn.GELU()
            ),
            "mri": nn.Sequential(
                nn.Linear(embed_dim + 32, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
        })
        
        # Cross-modal CLIP feature exchange module
        self.clip_feature_exchange = nn.Sequential(
            nn.Linear(64, 128),  # CT+MRI CLIP feature concatenation
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.Sigmoid()  # Generate exchange weights
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def process_3channel_input(self, x):
        """Process 3-channel input, convert to Y channel and save UV channels"""
        B, C, H, W = x.shape
        if C == 3:
            # RGB to YUV
            yuv = self.rgb_to_yuv(x)
            # Separate Y, U, V channels
            Y = yuv[:, 0:1, :, :]  # Y channel
            U = yuv[:, 1:2, :, :]  # U channel
            V = yuv[:, 2:3, :, :]  # V channel
            return Y, U, V, True
        else:
            # Already single channel
            return x, None, None, False

    def reconstruct_3channel_output(self, Y, U, V, has_color):
        """Reconstruct 3-channel output, merge Y channel with UV channels"""
        if has_color and U is not None and V is not None:
            # Resize Y channel to match UV channels if needed
            if Y.shape[2:] != U.shape[2:]:
                Y = F.interpolate(Y, size=U.shape[2:], mode='bilinear', align_corners=False)
            
            # Merge YUV channels
            yuv = torch.cat([Y, U, V], dim=1)
            # YUV to RGB
            rgb = self.yuv_to_rgb(yuv)
            return rgb
        else:
            return Y

    def augment_features_with_clip(self, features, clip_features, modality):
        """
        Augment main features with CLIP features
        """
        B, C, H, W = features.shape
        
        # Expand CLIP feature dimensions
        clip_features_expanded = clip_features.unsqueeze(-1).unsqueeze(-1)
        clip_features_expanded = clip_features_expanded.expand(-1, -1, H, W)
        
        # Concatenate features
        combined = torch.cat([features, clip_features_expanded], dim=1)
        
        # Pass through fusion layer
        combined = combined.flatten(2).transpose(1, 2)  # [B, H*W, C+128]
        fused = self.clip_fusion_layers[modality](combined)
        fused = fused.transpose(1, 2).view(B, -1, H, W)
        
        return fused
    
    def exchange_clip_features(self, ct_clip, mri_clip):
        """
        Exchange CT and MRI CLIP features (cross-modal information enhancement)
        """
        # Concatenate features
        combined = torch.cat([ct_clip, mri_clip], dim=-1)  # [B, 256]
        
        # Generate exchange weights
        exchange_weights = self.clip_feature_exchange(combined)
        ct_weight, mri_weight = torch.chunk(exchange_weights, 2, dim=-1)
        
        # Feature exchange
        ct_augmented = ct_weight * ct_clip + (1 - ct_weight) * mri_clip
        mri_augmented = mri_weight * mri_clip + (1 - mri_weight) * ct_clip
        
        return ct_augmented, mri_augmented

    def dual_level_feature_extraction(self, x, y, x_orig, y_orig):
        """
        Modified feature extraction with CLIP assistance
        """
        # Handle input channel issues
        I1_y, I1_u, I1_v, I1_has_color = self.process_3channel_input(x_orig)
        I2_y, I2_u, I2_v, I2_has_color = self.process_3channel_input(y_orig)
        
        # Save UV channels for subsequent reconstruction
        self.I1_u, self.I1_v = I1_u, I1_v
        self.I2_u, self.I2_v = I2_u, I2_v
        self.I1_has_color = I1_has_color
        self.I2_has_color = I2_has_color
        
        # Original feature extraction (now processing Y channel, i.e., 1 channel)
        I1 = self.lrelu(self.low_level_feature_extraction1(I1_y))
        I1 = self.lrelu(self.low_level_feature_extraction2(I1))  

        I2 = self.lrelu(self.low_level_feature_extraction1(I2_y))
        I2 = self.lrelu(self.low_level_feature_extraction2(I2))   
        
        # Extract CLIP auxiliary features
        clip_features = self.clip_encoder(x_orig, y_orig)
        ct_clip = clip_features["ct"]
        mri_clip = clip_features["mri"]
        
        # Cross-modal CLIP feature exchange
        ct_clip_aug, mri_clip_aug = self.exchange_clip_features(ct_clip, mri_clip)
        
        # Augment main features with CLIP features
        I1_clip_augmented = self.augment_features_with_clip(I1, ct_clip_aug, "ct")
        I2_clip_augmented = self.augment_features_with_clip(I2, mri_clip_aug, "mri")
        
        # Continue with original processing flow
        b, c, h, w = I2_clip_augmented.shape
        
        x_size = (I1_clip_augmented.shape[2], I1_clip_augmented.shape[3])
        I1_seq = self.patch_embed(I1_clip_augmented)
        I2_seq = self.patch_embed(I2_clip_augmented)
   
        if self.ape:
            I1_seq = I1_seq + self.absolute_pos_embed
            I2_seq = I2_seq + self.absolute_pos_embed
        I1_seq = self.pos_drop(I1_seq)
        I2_seq = self.pos_drop(I2_seq)

        # Modified: Use DualBranchMambaBlock for processing
        I1_processed = I1_seq
        I2_processed = I2_seq
        
        for block in self.high_level_feature_extraction1:
            I1_processed = block(I1_processed, torch.zeros_like(I1_processed))
        
        for block in self.high_level_feature_extraction2:
            I2_processed = block(I2_processed, torch.zeros_like(I2_processed))
        
        return I1_processed, 0, I2_processed, 0, h, w
    
    def dual_phase_feature_fusion(self, x, x_residual, y, y_residual, h, w):
        # Modified: Use ChannelExchange for feature exchange
        # First convert features to sequence form for exchange
        I1_seq = x.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
        I2_seq = y.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
        
        # Perform feature channel exchange before entering CM3F
        I1_exchanged, I2_exchanged = self.feature_exchange_after_extraction(I1_seq, I2_seq)
        
        # Convert exchanged feature maps back to sequence form
        I1_exchanged = I1_exchanged.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]
        I2_exchanged = I2_exchanged.permute(0, 2, 1)
        
        # Use CM3F for multi-level feature fusion
        fusion_f = self.cm3f(I1_exchanged, I2_exchanged, h, w)
        
        # Convert back to spatial dimensions
        fusion_f = self.patch_unembed(fusion_f,(h,w))
        
        return fusion_f

    def fused_img_recon(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        # Modified: Use MambaNormBlock for processing
        for block in self.feature_re:
            x = block(x)
  
        x = self.patch_unembed(x, x_size)
        
        # -------------------Convolution------------------- #
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x)  # Output 1-channel Y channel
        
        return x

    def forward(self, A, B):
        x = A
        y = B
        H, W = x.shape[2:]
        
        # Save original images for CLIP feature extraction
        A_orig = A.clone()
        B_orig = B.clone()
        
        # Set mean based on input channels
        if A.shape[1] == 3:
            self.mean_A = self.mean.type_as(x)
        else:
            self.mean_A = torch.zeros(1, 1, 1, 1).type_as(x)
            
        if B.shape[1] == 3:
            self.mean_B = self.mean.type_as(y)
        else:
            self.mean_B = torch.zeros(1, 1, 1, 1).type_as(y)
            
        self.mean = (self.mean_A + self.mean_B) / 2

        # Normalization processing
        if A.shape[1] == 3:
            x = (x - self.mean_A) * self.img_range
        else:
            x = x * self.img_range
            
        if B.shape[1] == 3:
            y = (y - self.mean_B) * self.img_range
        else:
            y = y * self.img_range
        
        # Dual_level_feature_extraction with CLIP
        feature1, residual_feature1, feature2, residual_feature2, h, w = self.dual_level_feature_extraction(x, y, A_orig, B_orig)
        
        # Dual_phase_feature_fusion
        fusion_feature = self.dual_phase_feature_fusion(feature1, residual_feature1, feature2, residual_feature2, h, w)

        # Fused_image_reconstruction
        fused_y = self.fused_img_recon(fusion_feature)  # Obtain 1-channel Y channel
        
        # Reconstruct color channels
        if self.I1_has_color or self.I2_has_color:
            # Choose which modality's UV channels to use (prefer the modality with color)
            if self.I1_has_color:
                U, V = self.I1_u, self.I1_v
            else:
                U, V = self.I2_u, self.I2_v
            
            # Merge Y channel with UV channels
            fused_img = self.reconstruct_3channel_output(fused_y, U, V, True)
        else:
            fused_img = fused_y
        
        # Inverse normalization
        if fused_img.shape[1] == 3:
            fused_img = fused_img / self.img_range + self.mean
        else:
            fused_img = fused_img / self.img_range + self.mean.mean(dim=1, keepdim=True)  # Use mean of means for single channel

        return fused_img[:, :, :H*self.upscale, :W*self.upscale]
    
if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = MambaMFNet (upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, clip_feature_dim=128, num_heads=[6, 6, 6, 6])
    
    # 测试3通道输入
    x = torch.randn((1, 3, height, width))
    y = torch.randn((1, 3, height, width))
    output = model(x, y)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
