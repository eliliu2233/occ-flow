import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F


@OPENOCC_LOSS.register_module()
class OccLoss3d(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'uniform_occ_3d': 'uniform_occ_3d',
                'occ_mask': 'occ_mask'
                }
        else:
            self.input_dict = input_dict
        
        self.loss_func = self.occ_loss_3d
    
    def occ_loss_3d(
            self, 
            uniform_occ_3d,
            occ_mask
        ):
        # flow_3d_gt: B, W, H, D, 2
        # uniform_flow_3d: B, W, H, D, 2
        device = occ_mask.device
        # occ_prob = occ_mask.sum() / occ_mask.numel()
        # class_weights = torch.tensor([1.0/(1-occ_prob), 1.0/occ_prob]).float().to(device)
        # occ_loss = nn.BCELoss(weight=class_weights, reduction="mean")(uniform_occ_3d, occ_mask.float())
        occ_loss = nn.BCELoss(reduction="mean")(uniform_occ_3d, occ_mask.float())

        return occ_loss