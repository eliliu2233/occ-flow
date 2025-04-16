import os, numpy as np

import torch.nn.functional as F
import torch
from .unimatch.unimatch import UniMatch
from .unimatch.geometry import forward_backward_consistency_check

def build_unimatch_model(unimatch_ckpt):
    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     num_head=1,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task='flow').eval().cuda()
    checkpoint = torch.load(unimatch_ckpt, map_location="cuda:{}".format(torch.cuda.current_device()))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)

    return model

def forward_unimatch_model(model, curr_imgs, prev_imgs, next_imgs):
    
    with torch.no_grad():
        results = {}
        if prev_imgs is not None:
            flow_pr_prev = model(curr_imgs.flip(dims=[1]), prev_imgs.flip(dims=[1]),
                                attn_type='swin',
                                attn_splits_list=[2, 8],
                                corr_radius_list=[-1, 4],
                                prop_radius_list=[-1, 1],
                                num_reg_refine=6,
                                task='flow',
                                pred_bidir_flow=True,)['flow_preds'][0]
            fwd_occ_prev, _ = forward_backward_consistency_check(flow_pr_prev[:curr_imgs.size(0)], flow_pr_prev[curr_imgs.size(0):])
            flow_curr_prev = flow_pr_prev[:curr_imgs.size(0)]
            # flow_curr_prev = flow_curr_prev*(1-fwd_occ_prev.unsqueeze(1)) + 1e7*fwd_occ_prev.unsqueeze(1)
            results.update(flow_curr_prev=flow_curr_prev.unsqueeze(0))
            # torch.cuda.empty_cache()
        if next_imgs is not None:
            flow_pr_next = model(curr_imgs.flip(dims=[1]), next_imgs.flip(dims=[1]),
                                attn_type='swin',
                                attn_splits_list=[2, 8],
                                corr_radius_list=[-1, 4],
                                prop_radius_list=[-1, 1],
                                num_reg_refine=6,
                                task='flow',
                                pred_bidir_flow=True,)['flow_preds'][0]
            fwd_occ_next, _ = forward_backward_consistency_check(flow_pr_next[:curr_imgs.size(0)], flow_pr_next[curr_imgs.size(0):])
            flow_curr_next = flow_pr_next[:curr_imgs.size(0)].permute(0,2,3,1)
            flow_curr_next[fwd_occ_next.bool()] = 1e7
            results.update(flow_curr_next=flow_curr_next.permute(0,3,1,2).unsqueeze(0), fwd_occ=fwd_occ_next.unsqueeze(0))
            # torch.cuda.empty_cache()
    return results