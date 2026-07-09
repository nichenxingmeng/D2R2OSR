import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp, DCNv2Pack

from d2r2osr.archs.hat_arch import RHAG, RHAG_Prompt
from d2r2osr.utils.e2p import e2p_patch
from d2r2osr.archs.transformer_arch import TransformerBlock
from d2r2osr.archs.swinir_arch import BasicLayer, BasicLayer_Prompt_2D, BasicLayer_Prompt_2D_SM, BasicLayer_Prompt_Attn


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
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
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """ Image to Patch Unembedding
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
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
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
    def __init__(self, scale, num_feat, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
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
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class ChannelAttentionFusion(nn.Module):
    def __init__(self, embed_dim, embed_dim_ppr):
        super().__init__()
        self.Dual_Attention_q = nn.Conv2d(embed_dim_ppr, embed_dim, 1, 1, 0)
        self.Dual_Attention_k = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        self.Dual_Attention_v = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        self.Dual_Attention_w = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        batch_size, channels, height, width = x.size()
        q = self.Dual_Attention_q(y).view(batch_size, -1, height*width)
        k = self.Dual_Attention_k(x).view(batch_size, -1, height*width)
        v = self.Dual_Attention_v(x).view(batch_size, -1, height*width)

        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v).view(batch_size, channels, height, width)
        out = self.Dual_Attention_w(out) + x

        return out


class ChannelAttention3DFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Dual_Attention_q = nn.Conv3d(embed_dim, embed_dim, 1)
        self.Dual_Attention_v = nn.Conv3d(embed_dim, embed_dim, 1)
        self.Dual_Attention_w = nn.Conv3d(embed_dim, embed_dim, 1)
        self.Dual_Attention_o = nn.Conv3d(embed_dim, 1, 1)

    def forward(self, x):
        x = x.contiguous()
        B, C, D, H, W = x.size()

        # 生成 q, v
        q = self.Dual_Attention_q(x)  # [B,C,D,H,W]
        v = self.Dual_Attention_v(x)

        # 🔥 改为 depth attention
        # 先把 depth 放到第二维
        q = q.permute(0, 2, 1, 3, 4).contiguous()  # [B,D,C,H,W]
        v = v.permute(0, 2, 1, 3, 4).contiguous()

        # flatten spatial+channel
        q = q.view(B, D, -1)   # [B,D,C*H*W]
        v = v.view(B, D, -1)

        # depth x depth attention
        attn = torch.bmm(q, q.transpose(1, 2))  # [B,D,D]
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v)  # [B,D,C*H*W]

        # reshape 回原形状
        out = out.view(B, D, C, H, W)
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,D,H,W]

        # residual
        out = self.Dual_Attention_w(out) + x

        # 输出 depth 权重图
        out = self.Dual_Attention_o(out).view(B, D, H, W)

        return out
    

class SimpleFusion(nn.Module):
    def __init__(self, dim_erp, dim_pers):
        super().__init__()
        mid_dim = dim_erp // 2
        self.proj = nn.Sequential(
            nn.Conv2d(dim_pers, mid_dim, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_dim, dim_erp, 1, 1, 0),
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # residual scaling

    def forward(self, x_erp, x_pers):
        x_pers = self.proj(x_pers)
        return x_erp + self.alpha * x_pers

class AdaptiveFusion(nn.Module):
    def __init__(self, dim_erp, dim_pers):
        super().__init__()

        self.proj = nn.Conv2d(dim_pers, dim_erp, 1, 1, 0)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim_erp * 2, dim_erp, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_erp, 1, 1, 1, 0),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_erp, dim_erp // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_erp // 4, dim_erp, 1),
            nn.Sigmoid()
        )

    def forward(self, x_erp, x_pers):
        x_pers = self.proj(x_pers)

        cat = torch.cat([x_erp, x_pers], dim=1)

        spatial_weight = self.spatial_gate(cat)
        channel_weight = self.channel_gate(x_erp)
        out = x_erp + spatial_weight * channel_weight * x_pers

        return out

class SpatialCrossAttentionFusion(nn.Module):
    def __init__(self, dim_erp, dim_ppr, num_heads=4):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim_erp
        self.head_dim = dim_erp // num_heads
        self.scale = self.head_dim ** -0.5

        self.ppr_proj = nn.Conv2d(dim_ppr, dim_erp, 1)

        # Q K V
        self.q = nn.Conv2d(dim_erp, dim_erp, 1)
        self.k = nn.Conv2d(dim_erp, dim_erp, 1)
        self.v = nn.Conv2d(dim_erp, dim_erp, 1)

        self.proj = nn.Conv2d(dim_erp, dim_erp, 1)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_erp, x_ppr):
        """
        x_erp: (B, C, H, W)
        x_ppr: (B, C2, H, W)
        """

        B, C, H, W = x_erp.shape

        x_ppr = self.ppr_proj(x_ppr)

        # Q from ERP
        q = self.q(x_erp)
        # K V from PPR
        k = self.k(x_ppr)
        v = self.v(x_ppr)

        # reshape for multi-head
        q = q.reshape(B, self.num_heads, self.head_dim, H*W)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W)

        q = q.permute(0,1,3,2)   # (B, heads, HW, head_dim)
        k = k.permute(0,1,2,3)   # (B, heads, head_dim, HW)
        v = v.permute(0,1,3,2)   # (B, heads, HW, head_dim)

        # Spatial attention
        attn = (q @ k) * self.scale     # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)

        out = attn @ v                  # (B, heads, HW, head_dim)
        out = out.permute(0,1,3,2).reshape(B, C, H, W)
        out = self.proj(out)

        # residual injection
        return x_erp + self.alpha * out

class FIFB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_equi = nn.Sequential(
            nn.Conv2d(2*dim, dim,1,1,0),
            nn.GELU(),
        )
        self.conv_cube = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1,1,0),
            nn.GELU(),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1,1,0),
            nn.GELU()
        )

    def forward(self, f_erp, f_cmp):
        f_cat = torch.concat([f_erp, f_cmp], dim=1)
        f_cat = self.conv_cat(f_cat)
        f_cmp = torch.concat([f_cat, f_cmp], dim=1)
        f_cmp = self.conv_cat(f_cmp)
        f_erp = torch.concat([f_cat, f_erp], dim=1)
        f_erp = self.conv_cat(f_erp)
        return f_erp,f_cmp,f_cat

@ARCH_REGISTRY.register()
class D2R2OSR(nn.Module):
    """
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        condition_dim (int): The dimension of input conditions
        c_dim (int): The dimension of condition-related hidden layers
        vit_condition (list): whether to apply DAAB. Default: None
        dcn_condition (list): whether to apply DACB. Default: None
    """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 condition_dim=1,
                 vit_condition=None,
                 vit_condition_type='1conv',
                 dcn_condition=None,
                 dcn_condition_type='1conv',
                 window_condition=False,
                 window_condition_only=False,
                 c_dim=60,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mode='train',
                 **kwargs):
        super(D2R2OSR, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        # self.embed_dim_2D = embed_dim
        self.embed_dim_2D = 60
        self.mode = mode

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_first_perspective = nn.Conv2d(num_in_ch, self.embed_dim_2D, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        if dcn_condition is None:
            dcn_condition = [0 for _ in range(self.num_layers + 1)]
        if vit_condition is None:
            vit_condition = [0 for _ in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_embed_2D = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.embed_dim_2D,
            embed_dim=self.embed_dim_2D,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_2D = self.patch_embed_2D.num_patches
        patches_resolution_2D = self.patch_embed_2D.patches_resolution
        self.patches_resolution_2D = patches_resolution_2D

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_unembed_2D = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.embed_dim_2D,
            embed_dim=self.embed_dim_2D,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            self.absolute_pos_embed_2D = nn.Parameter(torch.zeros(1, num_patches_2D, self.embed_dim_2D))
            trunc_normal_(self.absolute_pos_embed_2D, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_2D = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build RHAG
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
            )
            self.layers.append(layer)

        self.layers_2D = nn.ModuleList()
        for i_layer_2D in range(self.num_layers):
            layer_2D = BasicLayer_Prompt_Attn(
                dim=self.embed_dim_2D,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer_2D],
                num_heads=num_heads[i_layer_2D],
                window_size=(window_size, window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer_2D]):sum(depths[:i_layer_2D + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,

                prompt_dim=18,
                prompt_len=5,
                )
            self.layers_2D.append(layer_2D)
        self.norm = norm_layer(self.num_features)
        # self.norm_2D = norm_layer(self.embed_dim_2D)
        # self.norm_3D = norm_layer(self.num_features + self.embed_dim_2D)

        # bi
        self.layers_fusion = nn.ModuleList()
        for i_layer_fusion in range(self.num_layers):
            layer_fusion = AdaptiveFusion(
                dim_erp=embed_dim,
                dim_pers=self.embed_dim_2D
            )
            self.layers_fusion.append(layer_fusion)
        
        '''
        self.layers_fusion_2D = nn.ModuleList()
        for i_layer_fusion_2D in range(self.num_layers):
            layer_fusion_2D = Dual_Attention(embed_dim)
            self.layers_fusion_2D.append(layer_fusion_2D)
        '''
        self.layer_fusion_3D = ChannelAttention3DFusion(self.num_layers)
        '''
        self.layers_fusion_2D_Conv = nn.ModuleList()
        for i_layer_fusion_2D_Conv in range(self.num_layers-1):
            layer_fusion_2D_Conv = nn.Sequential(nn.Conv2d(self.embed_dim_2D, embed_dim, 3, 1, 1),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.layers_fusion_2D_Conv.append(layer_fusion_2D_Conv)
        '''
        # self.fusion_align = nn.Sequential(nn.Conv2d(self.embed_dim_2D + embed_dim, embed_dim, 3, 1, 1),
        #                                         nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # build the last conv layer in deep feature extraction
        # self.conv_after_body_2D = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
        #                                         nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.fusion_temp = nn.Sequential(nn.Conv2d(self.embed_dim_2D, embed_dim, 3, 1, 1),
        #                                         nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.transformer_2D = TransformerBlock(dim=embed_dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
        # self.conv_after_body_pers = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1), nn.LeakyReLU(inplace=True))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat, input_resolution=(patches_resolution[0], patches_resolution[1]))
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward_features_3d_fusion(self, x, x_ppr, condition):
        x_size = (x.shape[2], x.shape[3])
        x_ppr_size = (x_ppr.shape[2], x_ppr.shape[3])

        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': self.relative_position_index_SA,
            'rpi_oca': self.relative_position_index_OCA
        }

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_ppr = self.patch_embed_2D(x_ppr)
        if self.ape:
            x_ppr = x_ppr + self.absolute_pos_embed_2D
        x_ppr = self.pos_drop_2D(x_ppr)

        x_fusion_list = []

        for i in range(self.num_layers):
            # x = self.layers[i](x, x_size, params, condition)
            # x_ppr = self.layers_2D[i](x_ppr, x_ppr_size, condition)
            x = self.layers[i](x, x_size, params)
            x_ppr = self.layers_2D[i](x_ppr, x_ppr_size)

            # layer_ppr_Conv = self.layers_fusion_2D_Conv[i]
            # x_ppr_align = layer_ppr_Conv(x_ppr)

            x_map = self.patch_unembed(x, x_size)
            x_ppr_map = self.patch_unembed_2D(x_ppr, x_ppr_size)
            
            x_fusion_map = self.layers_fusion[i](x_map, x_ppr_map)
            # x_fusion_map = torch.cat([x_map, x_ppr_map], dim=1)
            # print(x_fusion_map.shape)
            x_fusion_map = x_fusion_map.permute(0, 2, 3, 1)  # B,H,W,C
            x_fusion_map = self.norm(x_fusion_map)
            x_fusion_map = x_fusion_map.permute(0, 3, 1, 2)  # B,C,H,W

            x_fusion_list.append(x_fusion_map)

        return x_fusion_list
    
    def perspective_patch_test(self, x, perspective):
        """
        Patch-wise testing with provided perspective tiles.
        x: tensor, (B, C, H, W) equirectangular image
        perspective: list of [y_start, x_start, tile_h, tile_w] for each batch
        """
        B, C, H, W = x.shape
        device = x.device

        x_p_full = torch.zeros((B, C, H, W), device=device)
        count_map = torch.zeros((B, 1, H, W), device=device)

        y_start, x_start, tile_h, tile_w = perspective
        y_end = y_start + tile_h
        x_end = x_start + tile_w
        y_end = min(y_end, H)
        x_end = min(x_end, W)
        actual_patch_h = y_end - y_start
        actual_patch_w = x_end - x_start
        with torch.no_grad():

            temp = torch.zeros((1, C, 256, 512), device=device)
            temp[:, :, y_start:y_end, x_start:x_end] = x[:, :, y_start:y_end, x_start:x_end]

            w_p = 180 * ((x_start + actual_patch_w / 2) / 256 - 1)
            h_p = 90 * (1 - (y_start + actual_patch_h / 2) / 128)
            p_p = max(180 * actual_patch_h / 256, 360 * actual_patch_w / 512)

            x_temp = e2p_patch(temp, p_p, w_p, h_p, (actual_patch_h, actual_patch_w))

            x_p_full[:, :, y_start:y_end, x_start:x_end] += x_temp
            count_map[:, :, y_start:y_end, x_start:x_end] += 1

            del temp, x_temp
            torch.cuda.empty_cache()

        x_p_full /= count_map
        return x_p_full


    def perspective_patch_train(self, x, perspective):
        patch_h = 64
        patch_w = 64
        b, c, h_, w_ = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x_p = []
        for i in range(b):
            h, w = perspective[i][0].item() // 4, perspective[i][1].item() // 4
            temp = torch.zeros((1, c, 256, 512)).to(x.device)
            # temp[:, :, h:h + patch_h, w:w + patch_w] = x[i, :, :, :].clone().detach().unsqueeze(0)
            temp[:, :, h:h + patch_h, w:w + patch_w] = x[i, :, :, :].unsqueeze(0)
            w_p = 180 * ((w + patch_w // 2) / 256 - 1)
            h_p = 90 * (1 - (h + patch_h // 2) / 128)
            p_p = max(180 * patch_h / 256, 360 * patch_w / 512)
            x_temp = e2p_patch(temp, p_p, w_p, h_p, (patch_h, patch_w))
            x_p.append(x_temp.squeeze(0))
        x_p = torch.stack(x_p, dim=0)
        
        return x_p

    def forward(self, x, condition, perspective):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            if self.mode == 'train':
                x_ppr = self.perspective_patch_train(x, perspective)
            else:
                x_ppr = self.perspective_patch_test(x, perspective)

            x_ppr = self.conv_first_perspective(x_ppr)
            x = self.conv_first(x)
            
            # x = self.forward_features_inter_fusion(x, x_ppr, condition) + x
            # x = self.forward_features_simple_fusion(x, x_ppr, condition) + x
            x_fusion_list = self.forward_features_3d_fusion(x, x_ppr, condition)
            x_temp = self.layer_fusion_3D(torch.stack(x_fusion_list, dim=1))
            # x = self.fusion_align(x_temp) + x
            x = x_temp + x

            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops