import torch
import numpy as np
import torch.nn as nn
from mmdet.models.necks.fpn import FPN
from transformers import AutoImageProcessor, AutoModel,pipeline
from torchvision import transforms
from .grid_mask import GridMask, PatchGridMask
import timm
import torch.nn.functional as F
from timm.data import resolve_model_data_config
import io, zstandard as zstd


class ImgEncoder(nn.Module):
    def __init__(self, config,num_feature_levels=2):
        super().__init__()
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        num_cams = 4

        self.num_cams = num_cams
        self.use_cams_embeds = True
        _num_levels_ = 1
        _dim_ = self.embed_dims

        self.use_lidar=False

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        #self.patch_grid_mask = PatchGridMask(use_h=True, use_w=True, d_min=4, d_max=16, ratio=0.5, rotate=1.0, offset=False, rescale=True, prob=0.7, same_on_batch=False)
        self.use_grid_mask = True

        #self.img_backbone = timm.create_model( "resnet34", pretrained=True, features_only=True )
        # self.img_backbone = timm.create_model("timm/vit_base_patch14_reg4_dinov2.lvd142m",
        #                                pretrained=True,
        #                                img_size=(448,768),
        #                                num_classes=0,
        #                                )
        #self.img_backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.img_backbone = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
        self.transform = make_transform(512)
                                       
        # self.bridge = nn.Sequential(
        #     nn.Conv2d(384, self.embed_dims, kernel_size=1, bias=False),
        #     nn.GroupNorm(32, self.embed_dims),
        #     nn.GELU(),
        #     # 轻量下采样 ×2，把 32×54 降到 16×27（更接近原 14×24）
        #     nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.GELU(),
        # )                                
                # 注册 mean/std，前向时做 (x-mean)/std

        original_mean = torch.tensor([[123.675, 116.28, 103.53]]).view(1,3,1,1)
        original_std = torch.tensor([[58.395, 57.12, 57.375]]).view(1,3,1,1)
        self.register_buffer("original_img_mean", original_mean, persistent=False)
        self.register_buffer("original_img_std", original_std, persistent=False)

        self.with_img_neck=True

        self.num_outs=1

        self.img_neck=FPN(
            in_channels=[64,128,256,1280][-self.num_outs:],#[64,128,256,512]
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=self.num_outs,
            relu_before_extra_convs=True
        )
        self.level_embeds = nn.Parameter(torch.randn( self.num_feature_levels, self.embed_dims))#,dtype=torch.float16
        self.cams_embeds = nn.Parameter(
            torch.randn([self.num_cams, self.embed_dims]))

    # def _tokens_to_map(self, x, B, N, H,W,patch_size=16):
    #     # 使用模型内部的网格大小，避免硬编码 32×54
    #     gh, gw = H//patch_size, W//patch_size
    #     # 判断前缀 token 数（cls+register，共5）
    #     extra = x.shape[1] - gh * gw
    #     x = x[:, extra:]                     # 去掉前缀 token
    #     x = x.transpose(1, 2).reshape(B*N, -1, gh, gw)  # B*N, C=384, H=gh, W=gw
    #     return x

    
    def _tokens_to_map(self, x, B, N, H, W,
                    patch_size=16,      # DINOv2 常见14
                    keep=0.5,           # 保留概率
                    training=None,      # 传 self.training
                    fill='mean',        # 'mean' 或 'zero'
                    apply_prob=0.7,     # 触发dropout的概率
                    eps=1e-6,
                    generator=None):
        """
        x: [B*N, T, C] (含前缀token)
        return: [B*N, C, gh, gw]
        """
        # if training is None:
        #     training = getattr(self, "training", True)

        gh, gw = H // patch_size, W // patch_size
        extra = x.shape[1] - gh * gw
        # 仅对 patch tokens 做 dropout（不动前缀）
        patch = x[:, extra:]                           # [B*N, gh*gw, C]

        # # 训练且需要dropout且可能触发
        # if training and (keep < 1.0) and (apply_prob > 0.0):
        #     BN, T, C = patch.shape

        #     # 每个样本以 apply_prob 的概率启用 dropout（不触发则直接保留原patch）
        #     apply_gate = (torch.rand(BN, 1, 1, device=patch.device, generator=generator) < apply_prob)
        #     apply_gate_full = apply_gate.expand(BN, T, C)

        #     # 掩码：1=keep, 0=drop（按 token 维度）
        #     m_keep = (torch.rand(BN, T, 1, device=patch.device, generator=generator) < keep).to(patch.dtype)

        #     if fill == 'mean':
        #         # 被drop的token用同一序列的token均值来填（不做缩放，期望会略变平滑）
        #         fill_val = patch.mean(dim=1, keepdim=True)                # [BN,1,C]
        #         dropped = patch * m_keep + fill_val * (1.0 - m_keep)      # [BN,T,C]
        #     else:
        #         # 'zero'：用0填+inverted dropout保持期望不变
        #         scale = 1.0 / max(float(keep), eps)
        #         dropped = patch * (m_keep * scale)

        #     # 仅对启用了dropout的样本替换
        #     patch = torch.where(apply_gate_full, dropped, patch)

        # 转成 [B*N, C, gh, gw]
        patch = patch.transpose(1, 2).reshape(B * N, -1, gh, gw)
        return patch

    def forward(self,img,len_queue=None,**kwargs):
        #img = F.interpolate(img.flatten(0,1), size=(518, 518), mode='bilinear', align_corners=False).view(img.shape[0], img.shape[1], img.shape[2], 518, 518)
        B = img.size(0)
        if img is not None:
            # if img.dim() == 5 and img.size(0) == 1:
            #     img.squeeze_()
            # elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

            img = img*self.original_img_std + self.original_img_mean
            rgb_seq = [2,1,0]
            img = img[:,rgb_seq]/255.0

            img = self.transform(img)
            if self.training and self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(pixel_values=img)['last_hidden_state']
            img_feats = self._tokens_to_map(img_feats,B,N,img.shape[2],img.shape[3])
            # if self.training and self.use_grid_mask:
            #     img_feats = self.patch_grid_mask(img_feats)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck([img_feats])

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats_reshaped):
            bs, num_cam, c, h, w = feat.shape#1,6,256,12,20
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)#6,1,240,256
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)


            #feat = feat +lidar2img_embed[:,:,None]

            spatial_shape = torch.as_tensor(
                [spatial_shape], dtype=torch.long, device=feat.device)
            level_start_index = torch.cat((spatial_shape.new_zeros(
                (1,)), spatial_shape.prod(1).cumsum(0)[:-1]))

            feat = feat.permute(  0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)#6,1,240,256

        return feat_flatten[-1],spatial_shapes[-1],level_start_index,kwargs

def make_transform(resize_size: int = 224):
    #resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([normalize])

def letterbox_pad_bchw(x, fill=0.0):
    """
    将张量 x[B,C,H,W] 做 letterbox 补边到方形 B,C,T,T（不缩放），T=max(H,W)。
    fill: 常数填充值（float）或每通道填充值序列（len==C，如 ImageNet 均值）。
    """
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(x) or x.ndim != 4:
        raise TypeError("Expected a 4D tensor [B,C,H,W].")
    B, C, H, W = x.shape
    T = max(H, W)
    if H == T and W == T:
        return x  # 已是方形

    pad_h, pad_w = T - H, T - W
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    # 标量填充值直接用 F.pad，更高效
    if isinstance(fill, (int, float)):
        return F.pad(x, (left, right, top, bottom), mode="constant", value=float(fill))

    # 每通道不同填充值：手动构造画布
    fill = torch.as_tensor(fill, dtype=x.dtype, device=x.device)
    if fill.numel() != C:
        raise ValueError(f"len(fill) must equal C ({C}), got {fill.numel()}.")
    canvas = fill.view(1, C, 1, 1).expand(B, C, T, T).clone()
    canvas[:, :, top:T-bottom if bottom else T, left:T-right if right else T] = x
    return canvas