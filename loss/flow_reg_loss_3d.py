import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F


@OPENOCC_LOSS.register_module()
class FlowRegLoss3d(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'uniform_flow_3d': 'uniform_flow_3d',
                'occ_mask': 'occ_mask'
                }
        else:
            self.input_dict = input_dict
        
        self.loss_func = self.flow_reg_loss_3d
    
    def flow_reg_loss_3d(
            self, 
            uniform_flow_3d,
            occ_mask=None,
        ):
        # uniform_flow_3d: B, W, H, D, 2
        bs, w, h, z, _ = uniform_flow_3d.size()
        assert bs == 1
        if occ_mask is not None:
            uniform_flow_3d = uniform_flow_3d[occ_mask]
        flow_loss = torch.abs(uniform_flow_3d).mean()

        return flow_loss