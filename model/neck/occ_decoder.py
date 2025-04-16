# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch.utils.checkpoint import checkpoint as cp
from mmengine.model import BaseModule

@MODELS.register_module()
class Occ_Decoder(BaseModule):
    def __init__(
        self,
        in_channels,
        num_level=1,
        soft_weights=False,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        with_cp=False,
        use_deblock=True,
    ):
        super(Occ_Decoder, self).__init__()

        self.fp16_enabled=False
      
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.with_cp = with_cp
        self.use_deblock = use_deblock
        self.in_channels = in_channels
        self.num_level = num_level

        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        if self.use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=self.in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))

    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}

        if self.use_deblock:
            if self.with_cp and voxel_feats[0].requires_grad:
                x0 = cp(self.deblock, voxel_feats[0])
            else:
                x0 = self.deblock(voxel_feats[0])
            output_occs.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp  and feats.requires_grad:
                x = cp(occ_conv, feats)
            else:
                x = occ_conv(feats)
            output_occs.append(x)

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W, D= output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        output['representation'] = out_voxel_feats

        return output
     
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, representation, **kwargs):
        if isinstance(representation, torch.Tensor):
            voxel_feats = [representation]
        else:
            voxel_feats = representation
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        output = self.forward_coarse_voxel(voxel_feats)

        return output
