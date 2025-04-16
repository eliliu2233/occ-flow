import torch, torch.nn as nn
import numpy as np
from mmseg.models import SEGMENTORS, builder
from mmseg.models import build_backbone
from mmengine.registry import MODELS
import model
import torch.nn.functional as F
from .base_segmentor import CustomBaseSegmentor
import torch.distributed as dist
from mmengine.logging import MMLogger
logger = MMLogger.get_instance('occflow')
from utils.tb_wrapper import WrappedTBWriter
if 'occflow' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('occflow')
else:
    writer = None

@SEGMENTORS.register_module()
class TPVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        queue_length=1,
        is_corner=True,
        fuse_with_bev=False,
        fuse_only_bev=False,
        volume_size=[257,257,33],
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.use_post_fusion = use_post_fusion
        self.queue_length = queue_length
        self.volume_size = volume_size
        self.is_corner = is_corner
        self.fuse_with_bev = fuse_with_bev
        if fuse_only_bev:
            _dim_ = self.encoder.embed_dims
            self.combine_coeff = nn.Conv3d(_dim_, 1, kernel_size=1, bias=True)
            self.process_net = builder.build_backbone(dict(type='CustomResNet3D', numC_input=_dim_,
                                                           num_layer=[1,], num_channels=[_dim_,], stride=[1,], backbone_output_ids=[0,]))
        self.fuse_only_bev = fuse_only_bev
        if fuse_with_bev:
            self.combine_coeff = nn.Conv3d(self.encoder.embed_dims, 1, kernel_size=1, bias=True)
        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def extract_img_feat(self, imgs, metas, len_queue=None, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if metas[0]['flip']:
                # img_feat = torch.fliplr(img_feat)
                img_feat = torch.flip(img_feat, [-1])
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            if metas[0]['flip']:
                img_feat_backbone = torch.flip(img_feat_backbone, [-1])
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return {
            'ms_img_feats_backbone': img_feats_backbone_reshaped,
            'ms_img_feats': img_feats_reshaped}
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped
    
    def transform_bev(self, tmp_bev, prev_to_curr_rt):
        # This converts BEV indices to meters
        # Generate grid
        n, mc, h, w, z = tmp_bev.shape
        if self.is_corner:
            xs = torch.linspace(0, w - 1, w, dtype=tmp_bev.dtype, device=tmp_bev.device).view(1, w, 1).expand(h, w, z)
            ys = torch.linspace(0, h - 1, h, dtype=tmp_bev.dtype, device=tmp_bev.device).view(h, 1, 1).expand(h, w, z)
            zs = torch.linspace(0, z - 1, z, dtype=tmp_bev.dtype, device=tmp_bev.device).view(1, 1, z).expand(h, w, z)
        else:
            xs = torch.linspace(0.5, w - 0.5, w, dtype=tmp_bev.dtype, device=tmp_bev.device).view(1, w, 1).expand(h, w, z)
            ys = torch.linspace(0.5, h - 0.5, h, dtype=tmp_bev.dtype, device=tmp_bev.device).view(h, 1, 1).expand(h, w, z)
            zs = torch.linspace(0.5, z - 0.5, z, dtype=tmp_bev.dtype, device=tmp_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((ys, xs, zs), -1).view(1, h, w, z, 3).expand(n, h, w, z, 3)
        xyzs = self.encoder.mapping.grid2meter(grid)
        xyzs = torch.cat((xyzs, torch.zeros_like(xyzs[...,0:1])), dim=-1)
        xyzs = prev_to_curr_rt.inverse().view(n,1,1,1,4,4) @ xyzs.unsqueeze(-1)
        sample_grid = self.encoder.mapping.meter2grid(xyzs.squeeze(-1)[...,0:3], normalize=True)
        sample_grid = 2.0 * sample_grid - 1.0
        if not self.is_corner:
            volume_size = torch.FloatTensor([h, w, z]).to(sample_grid.device)
            sample_grid = sample_grid * volume_size / (volume_size-1)

        sampled_history_bev = torch.nn.functional.grid_sample(tmp_bev, sample_grid[...,[2,1,0]], align_corners=True, mode='bilinear', padding_mode='border')
            
        return sampled_history_bev

    def tpv_to_volume(self, representation):
        tpv_hw, tpv_zh, tpv_wz = representation
        B = tpv_hw.size(0)
        tpv_hw = tpv_hw.reshape(B, self.volume_size[0], self.volume_size[1], 1, -1)
        tpv_hw = tpv_hw.expand(-1, -1, -1, self.volume_size[2], -1)

        tpv_zh = tpv_zh.reshape(B, self.volume_size[2], self.volume_size[0], 1, -1).permute(0, 2, 3, 1, 4)
        tpv_zh = tpv_zh.expand(-1, -1, self.volume_size[1], -1, -1)

        tpv_wz = tpv_wz.reshape(B, self.volume_size[1], self.volume_size[2], 1, -1).permute(0, 3, 1, 2, 4)
        tpv_wz = tpv_wz.expand(-1, self.volume_size[0], -1, -1, -1)

        tpv = tpv_hw + tpv_zh + tpv_wz
        return tpv
    
    def obtain_history_bev(self, prev_imgs, metas, **kwargs):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        is_training = self.training
        self.eval()
        prev_bev_lst = []
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = prev_imgs.shape
            prev_imgs = prev_imgs.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(prev_imgs, metas, len_queue=len_queue)['ms_img_feats']
            for i in range(len_queue):
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                results = {'ms_img_feats':img_feats, 'metas':metas}
                outs = self.lifter(**results)
                results.update(outs)
                outs = self.encoder(**results)
                results.update(outs)
                prev_to_curr_rt = torch.from_numpy(np.stack([metas[j]['prev_to_curr_rt'][i] for j in range(len(metas))])).float().to(prev_imgs.device)
                if not self.fuse_only_bev:
                    prev_bev = self.tpv_to_volume(results['representation']).permute(0,4,1,2,3)
                else:
                    prev_bev = results['representation'][0].reshape(bs, self.volume_size[0], self.volume_size[1], -1).permute(0,3,1,2)
                if getattr(self, 'pre_process', None) is not None:
                    prev_bev = self.pre_process(prev_bev)[0]
                
                prev_bev = self.transform_bev(prev_bev, prev_to_curr_rt)
                prev_bev_lst.append(prev_bev)
        if is_training:
            self.train()
        # (bs, num_queue, embed_dims, H, W)
        return torch.cat(prev_bev_lst, dim=1)

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                img_feat_only=False,
                extra_backbone=False,
                occ_only=False,
                prepare=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        B,N,C,H,W = imgs.size()
        imgs = imgs.reshape(B,self.queue_length,-1,C,H,W)
        results = {
            'imgs': imgs[:,0,...],
            'metas': metas,
            'points': points,
            'prev_imgs':imgs[:,1:,...] if self.queue_length>1 else None
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        # outs['ms_img_feats'] = [feat.float() for feat in outs['ms_img_feats']]
        results.update(outs)
        if img_feat_only:
            return results['ms_img_feats_backbone']
        # with torch.cuda.amp.autocast(enabled=False):
        outs = self.lifter(**results)
        results.update(outs)
        outs = self.encoder(**results)
        bev_embed = []
        if results['prev_imgs'] is not None:
            prev_bev = self.obtain_history_bev(**results)
            if not self.fuse_only_bev:
                curr_bev = self.tpv_to_volume(outs['representation']).permute(0,4,1,2,3)
            else:
                curr_bev = outs['representation'][0].reshape(B, self.volume_size[0], self.volume_size[1], -1).permute(0,3,1,2)
            if getattr(self, 'pre_process', None) is not None:
                curr_bev = self.pre_process(curr_bev)[0]
            bev = torch.cat([curr_bev, prev_bev], dim=1)
            if getattr(self, 'temporal_fusion', None):
                B, _, volume_h, volume_w, volume_z = bev.size()
                bev = bev.reshape(B, self.queue_length, -1, volume_h, volume_w, volume_z)
                C = bev.size(2)
                if self.temporal_fusion.bev_pooling:
                    bev = bev.permute(0,1,3,4,5,2).mean(-2)
                    bev = self.temporal_fusion(bev.flatten(2,3)).reshape(B, volume_h, volume_w, -1).permute(0,3,1,2)
                    bev = curr_bev + bev.unsqueeze(-1)
                else:
                    # (B,Q,C,H,W,Z) --> (B*Z,Q,H,W,C)
                    x = bev.permute(0,5,1,3,4,2).flatten(0,1)
                    if self.fuse_with_bev:
                        # (B,Q,C,H,W,Z) --> (B,Q,H,W,C)
                        x_bev = bev.mean(-1).permute(0,1,3,4,2)
                        x = torch.cat((x_bev, x), dim=0) # (B*(Z+1),Q,H,W,C)
                        bev = self.temporal_fusion(x.flatten(2,3)).reshape(-1, volume_h, volume_w, C).permute(0,3,1,2)
                        x_bev, x = bev[:B], bev[B:]
                        x = x.reshape(B, volume_z, -1, volume_h, volume_w).permute(0,2,3,4,1)
                        coeff = self.combine_coeff(x).sigmoid()
                        bev = x + coeff * x_bev.unsqueeze(-1)
                    else:
                        x = self.temporal_fusion(x.flatten(2,3))
                        bev = x.reshape(B, volume_z, volume_h, volume_w, -1).permute(0,4,2,3,1)
            else:
                if self.fuse_with_bev:
                    bev = torch.cat((bev.mean(-1).unsqueeze(-1), bev), dim=-1)
                
        else:
            bev = self.tpv_to_volume(outs['representation']).permute(0,4,1,2,3)
        if getattr(self, 'temporal_encoder_backbone', None) is not None:
            bev = self.temporal_encoder_backbone(bev)
        if getattr(self, 'temporal_encoder_neck', None) is not None:
            bev = self.temporal_encoder_neck(bev)
        if self.fuse_only_bev:
            results.update(bev_embed=bev)
        else:
            results.update(representation=bev, bev_embed=bev_embed)
            
        if getattr(self, 'decoder', None) is not None:
            outs = self.decoder(**results)
            results.update(outs)
        
        if self.fuse_only_bev:
            curr_volume = self.tpv_to_volume(results['representation']).permute(0,4,1,2,3)
            results['representation'] = self.process_net(curr_volume + self.combine_coeff(curr_volume).sigmoid()*results['bev_embed'].unsqueeze(-1))[0].permute(0,2,3,4,1)
        else:
            results['representation'] = (results['representation']).permute(0,2,3,4,1) #+curr_bev
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        elif prepare and hasattr(self.head, "prepare"):
            if not results.get('prepare', False):
                outs = self.head.prepare(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results