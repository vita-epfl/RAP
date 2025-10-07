import torch
import numpy as np
import torch.nn as nn
from mmdet.models.necks.fpn import FPN
from transformers import AutoImageProcessor, AutoModel,pipeline
from torchvision import transforms
from .grid_mask import GridMask, PatchGridMask
import timm
import torch.nn.functional as F



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
        self.use_grid_mask = True

        self.img_backbone = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
       # self.transform = make_transform(512)
                                   
        # original_mean = torch.tensor([[123.675, 116.28, 103.53]]).view(1,3,1,1)
        # original_std = torch.tensor([[58.395, 57.12, 57.375]]).view(1,3,1,1)
        # self.register_buffer("original_img_mean", original_mean, persistent=False)
        # self.register_buffer("original_img_std", original_std, persistent=False)

        self.with_img_neck=True

        self.num_outs=1

        self.img_neck=FPN(
            in_channels=[64,128,256,1280][-self.num_outs:],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=self.num_outs,
            relu_before_extra_convs=True
        )
        self.level_embeds = nn.Parameter(torch.randn( self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.randn([self.num_cams, self.embed_dims]))

    
    def _tokens_to_map(self, x, B, N, H, W,
                    patch_size=16,      
                    keep=0.5,           
                    training=None,      
                    fill='mean',        
                    apply_prob=0.7,     
                    eps=1e-6,
                    generator=None):
        """
        x: [B*N, T, C] 
        return: [B*N, C, gh, gw]
        """

        gh, gw = H // patch_size, W // patch_size
        extra = x.shape[1] - gh * gw
        patch = x[:, extra:]                           # [B*N, gh*gw, C]


        patch = patch.transpose(1, 2).reshape(B * N, -1, gh, gw)
        return patch

    def forward(self,img,len_queue=None,**kwargs):
        B = img.size(0)
        if img is not None:

            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            # img = img*self.original_img_std + self.original_img_mean
            # rgb_seq = [2,1,0]
            # img = img[:,rgb_seq]/255.0
            #img = self.transform(img)
            if self.training and self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(pixel_values=img)['last_hidden_state']
            img_feats = self._tokens_to_map(img_feats,B,N,img.shape[2],img.shape[3])

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
