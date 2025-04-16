import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F


@OPENOCC_LOSS.register_module()
class FlowRegLoss2d(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'semantic_mask': 'semantic_mask',
                'ms_rays': 'ms_rays',
                'scene_flows': 'scene_flows',
                }
        else:
            self.input_dict = input_dict
        self.img_size = kwargs.get('img_size', [768, 1600])
        
        self.loss_func = self.flow_reg_loss_2d
    
    def flow_reg_loss_2d(
            self,  
            ms_rays,
            scene_flows, 
            semantic_mask=None,
            sampled_flows=None,
            weights = None,
            dynamic_mask=None):
        # curr_imgs: B, N, C, H, W
        # depth: B, N, R
        # rays: R, 2
        bs, num_cams = scene_flows.shape[:2]
        num_rays = ms_rays.shape[0]
        assert bs == 1
        assert ((semantic_mask is not None) and (dynamic_mask is None)) or ((semantic_mask is None) and (dynamic_mask is not None))

        def sample_pixel(pixel, imgs):
            # imgs: B, N, 3, H, W
            # pixel: B, N, 1, R, 2
            pixel = pixel.clone()
            pixel[..., 0] /= self.img_size[1]
            pixel[..., 1] /= self.img_size[0]
            pixel = 2 * pixel - 1
            pixel_rgb = F.grid_sample(
                imgs.flatten(0, 1), 
                pixel.flatten(0, 1), 
                mode='nearest',
                padding_mode='border',
                align_corners=True) # BN, 3, 1, R
            pixel_rgb = pixel_rgb.reshape(bs, num_cams, pixel_rgb.shape[-1])
            return pixel_rgb
        if semantic_mask is not None:
            semantic_mask = semantic_mask.unsqueeze(2)
            pix_curr = ms_rays.reshape(1, 1, 1, num_rays, 2).repeat(1,num_cams,1,1,1) # bs, N, 1, r, 2
            sample_mask = sample_pixel(pix_curr, semantic_mask)
            sample_mask = sample_mask == 0
        else:
            sample_mask = ~dynamic_mask
        if sample_mask.sum()>0:
            flow_loss = torch.abs(scene_flows[sample_mask]).mean()
        else:
            flow_loss = torch.tensor(0).cuda().float()
        if weights is not None:
            reg_loss = []
            for cam in range(num_cams):
                weight_mask = weights[cam]<1e-3
                reg_loss.append(torch.abs(sampled_flows[cam].reshape(-1,2)[weight_mask]).sum(1))
            flow_loss += 0.2 * torch.cat(reg_loss).mean()
        return flow_loss