import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import numpy as np
import torch.nn.functional as F

def paint(img, uv, color):
    if uv.shape[0]>0:
        img[uv[:,1], uv[:,0]] = color
        img[uv[:,1]+1, uv[:,0]] = color
        img[uv[:,1], uv[:,0]+1] = color
        img[uv[:,1]+1, uv[:,0]+1] = color
    return img
    
@OPENOCC_LOSS.register_module()
class FlowLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_keys = {
                'flow_curr_prev': 'flow_curr_prev',
                'flow_curr_next': 'flow_curr_next',
                'ray_indices': 'ray_indices',
                'weights': 'weights',
                'ts': 'ts',
                'metas': 'metas',
                'ms_rays': 'ms_rays',
                'prev_warps': 'prev_warps',
                'next_warps': 'next_warps'
                }
        else:
            self.input_dict = input_dict
        self.img_size = kwargs.get('img_size', [768, 1600])
        self.dims = kwargs.get('dims', 2)
        self.epe_thre = kwargs.get('epe_thre', 8.0)
        self.debug = kwargs.get('debug', False)
        self.loss_func = self.reproj_loss
        self.dyn_iters = kwargs.get('dyn_iters', 1e8)
        self.with_static = kwargs.get('with_static', True)
        # self.iter_counter = 0
    
    def reproj_loss(
            self, 
            ray_indices,
            weights,
            ts,
            metas, 
            ms_rays, 
            flow_curr_next, flow_curr_prev=None,
            points_warps=None,
            sampled_flows=None,
            static_flows=None,
            semantic_mask=None,
            curr_imgs=None,
            deltas=None):
        # curr_imgs: B, N, C, H, W
        # depth: B, N, R
        # rays: R, 2
        bs, num_cams = flow_curr_next.shape[:2]
        device = flow_curr_next.device
        num_rays = ms_rays.shape[0]
        assert bs == 1

        # prepare transformation matrices
        lidar2warpImgs = []
        img2prevImg, img2nextImg = [], []
        extra_output = {}
        for meta in metas:
            lidar2warpImgs.append(meta['lidar2nextImg'])
            img2prevImg.append(meta['img2prevImg'])
            img2nextImg.append(meta['img2nextImg'])
        
        def list2tensor(trans):
            if isinstance(trans[0], (np.ndarray, list)):
                trans = np.asarray(trans)
                trans = weights[0].new_tensor(trans) # B, 36(6tem * 6cur), 4, 4
            else:
                trans = torch.stack(trans, dim=0)
            trans = trans.reshape(bs, num_cams, 1, 4, 4)
            return trans

        lidar2warpImgs = list2tensor(lidar2warpImgs)
        img2prevImg = list2tensor(img2prevImg)
        img2nextImg = list2tensor(img2nextImg)
        
        with torch.no_grad():
            pixels = ms_rays.reshape(1, 1, 1, num_rays, 2).repeat(bs, num_cams, 1, 1, 1)
            pixels[...,0] /= self.img_size[1]
            pixels[..., 1] /= self.img_size[0]
            pixels = 2 * pixels - 1
            sample_flow = F.grid_sample(flow_curr_next.flatten(0, 1), pixels.flatten(0, 1)).permute(0,2,3,1).reshape(bs,num_cams,num_rays,2)
            dynamic_mask = torch.zeros([bs, num_cams, num_rays], dtype=bool, device=ms_rays.device)
            dynamic_dis = False
            if static_flows is not None:
                dynamic_dis = True
                epe = torch.sqrt(((sample_flow-static_flows)**2).sum(axis=-1))
                epe_mask = (epe > self.epe_thre) & ((abs(sample_flow[...,0]) < 2e3) & (abs(sample_flow[...,1]) < 2e3))
                dynamic_mask = epe_mask
                if semantic_mask is not None:
                    semantic_mask = semantic_mask.unsqueeze(2)
                    sample_mask = F.grid_sample(semantic_mask.flatten(0, 1), pixels.flatten(0, 1), 
                                            mode='nearest',padding_mode='border',align_corners=True).permute(0,2,3,1).reshape(bs,num_cams,num_rays)
                    sample_mask = sample_mask != 0
                    dynamic_mask = dynamic_mask & sample_mask
                extra_output.update(dynamic_mask=dynamic_mask)
                if self.debug:
                    import cv2
                    img = curr_imgs[0,0].permute(1,2,0).cpu().numpy()*255.0
                    uv = ms_rays.reshape(num_rays,2).cpu().numpy().astype(int)
                    m_epe = epe_mask[0,0].cpu().numpy()
                    img_epe = paint(img, uv[m_epe], [0,0,255])
                    img_epe = paint(img_epe, uv[~m_epe], [255,0,0])
                    cv2.imwrite('./epe_img.png', img_epe)
                    m_sem = sample_mask[0,0].cpu().numpy()
                    img_sem = paint(img, uv[m_sem], [0,0,255])
                    img_sem = paint(img_sem, uv[~m_sem], [255,0,0])
                    cv2.imwrite('./sem_img.png', img_sem)
                    m_dyn = dynamic_mask[0,0].cpu().numpy()
                    img_dyn = paint(img, uv[m_dyn], [0,0,255])
                    img_dyn = paint(img_dyn, uv[~m_dyn], [255,0,0])
                    cv2.imwrite('./dyn_img.png', img_dyn)
                             
        
        tot_loss, tot_dynamic = [], []
        tot_count = torch.tensor(0., device=device)
        for cam, (ray_idx, weight, t) in enumerate(zip(ray_indices, weights, ts)):

            rays = ms_rays[ray_idx]
            if deltas is not None:
                delta = deltas[cam].detach()
                eps = torch.finfo(delta.dtype).eps
                weight = weight.clone()
                weight[delta < eps] = 0.
                weight = weight / delta.clamp_min(eps)
            
            pixel_coords = torch.ones((bs, 1, len(rays), 4), device=device) # B, N, R, 4
            pixel_coords[..., :2] = rays.reshape(1, 1, -1, 2)
            pixel_coords[..., :3] *= t.reshape(1, 1, -1, 1)
            pixel_coords = pixel_coords.unsqueeze(-1)
            weight_mask = weight>2e-3
            
            @torch.cuda.amp.autocast(enabled=False)
            def cal_pixel(trans, coords):
                trans = trans.float()
                coords = coords.float()
                eps = 1e-5
                pixel = torch.matmul(trans, coords).squeeze(-1) # bs, N, R, 4
                mask = pixel[..., 2] > 0.0
                pixel = pixel[..., :2] / torch.maximum(torch.ones_like(pixel[..., :1]) * eps, pixel[..., 2:3])
                mask = mask & (pixel[..., 0] > -100) & (pixel[..., 0] < self.img_size[1]+100) & \
                              (pixel[..., 1] > -50) & (pixel[..., 1] < self.img_size[0]+50)
                return pixel, mask
            # prev_warp = prev_warp.reshape(1,1,len(rays),3)
            # prev_warp = torch.cat((prev_warp, torch.ones_like(prev_warp[..., :1])), -1).unsqueeze(-1)
            # pixel_prev, prev_mask = cal_pixel(lidar2prevImg[:, cam:(cam+1), ...], prev_warp) # bs, N, 1, R, 2
            # next_warp = next_warp.reshape(1,1,len(rays),3)
            # next_warp = torch.cat((next_warp, torch.ones_like(next_warp[..., :1])), -1).unsqueeze(-1)
            # pixel_next, next_mask = cal_pixel(lidar2nextImg[:, cam:(cam+1), ...], next_warp)
            
            def sample_pixel(pixel, imgs):
                # imgs: B, N, 3, H, W
                # pixel: B, N, 1, R, 2
                pixel = pixel.clone()
                pixel[..., 0] /= self.img_size[1]
                pixel[..., 1] /= self.img_size[0]
                pixel = 2 * pixel - 1
                pixel_rgb = F.grid_sample(
                    imgs.flatten(0, 1), 
                    pixel.flatten(0, 1),)
                    # mode='bilinear',
                    # padding_mode='border',
                    # align_corners=True) # BN, 3, 1, R
                tmp_dim = pixel_rgb.shape[1]
                pixel_rgb = pixel_rgb.reshape(bs, 1, tmp_dim, pixel_rgb.shape[-1])
                pixel_rgb = pixel_rgb.permute(0, 1, 3, 2) # B, N, R, 3
                return pixel_rgb
            
            def get_diff(x1, x2):
                return torch.mean(torch.abs(x1 - x2), dim=-1)
            
            pix_curr = ms_rays.reshape(1, 1, 1, num_rays, 2) # bs, N, 1, r, 2
            num_rays = pix_curr.shape[-2]
            pix_curr_ = torch.gather(pix_curr.squeeze(2), dim=-2, index=ray_idx.reshape(1, 1, -1, 1).repeat(1, 1, 1, 2)) # B, N, R, 3

            def compute_l1_loss(points_warps, sample_flow_gt, lidar2warpImg, dyn_mask):
                if points_warps is not None:
                    dyn_mask_ = torch.gather(dyn_mask, -1, ray_idx).reshape(1,1,-1)
                    points_warp = points_warps[cam].reshape(1,1,-1,3)
                    if sampled_flows is not None:
                        sampled_flow = sampled_flows[cam].reshape(1,1,-1,2).clone()
                        if dynamic_dis:
                            sampled_flow[~dyn_mask_] = 0.0
                        points_warp = points_warp + torch.cat([sampled_flow, torch.zeros_like(sampled_flow[...,0:1])],dim=-1)
                    points_warp = torch.cat((points_warp, torch.ones_like(points_warp[..., :1])), -1).unsqueeze(-1)
                    pixel, mask = cal_pixel(lidar2warpImg, points_warp) # bs, N, 1, R, 2
                else:
                    pixel, mask = cal_pixel(img2nextImg[:, cam:(cam+1), ...], pixel_coords) # bs, N, 1, R, 2
                sample_flow_ = torch.gather(sample_flow_gt, dim=-2, index=ray_idx.reshape(1, 1, -1, 1).repeat(1, 1, 1, self.dims)) # B, N, R, 3
                diff = get_diff(pixel-pix_curr_, sample_flow_) # B, N, R
                mask = mask & weight_mask
                diff[~mask] = 0.
                
                weight_ = weight.clone()
                # stop the gradient of sdf field for dynamic objects...
                if self.iter_counter < self.dyn_iters and dyn_mask.sum()>0:
                    dyn_mask_ = torch.gather(dyn_mask, -1, ray_idx)
                    weight_[dyn_mask_] = weight[dyn_mask_].detach()
                weight_[~mask.flatten()] = 0.
                weight_sum = torch.zeros(num_rays, dtype=weight.dtype, device=weight.device)
                weight_sum.index_add_(-1, ray_idx, weight_)
                weight_sum = weight_sum.clamp_min(torch.finfo(weight.dtype).eps) #.detach() # r
                weight_sum = torch.gather(weight_sum, -1, ray_idx) # R
                weight_ = weight_ / weight_sum # R
                
                l1_loss = torch.zeros(num_rays, dtype=diff.dtype, device=weight.device)
                l1_loss.index_add_(-1, ray_idx, weight_ * diff.flatten()) # r
                invalid_mask = (abs(sample_flow_gt[...,0]) > 2e3) & (abs(sample_flow_gt[...,1]) > 2e3)
                l1_loss[invalid_mask.flatten()] = 0.
                return l1_loss
            
            dyn_mask = dynamic_mask[:, cam:(cam+1), ...].flatten()
            flow_loss = compute_l1_loss(points_warps, sample_flow[:, cam:(cam+1), ...], lidar2warpImgs[:, cam:(cam+1), ...], dyn_mask)
            if (not self.with_static) and dynamic_dis:
                flow_loss[~dyn_mask] = 0.0
            valid_next_mask = flow_loss>0
            cnt = valid_next_mask.to(torch.float)
            if flow_curr_prev is not None:
                pixel_prev, prev_mask = cal_pixel(img2prevImg[:, cam:(cam+1), ...], pixel_coords) # bs, N, 1, R, 2
                flow_prev = sample_pixel(pix_curr, flow_curr_prev[:, cam:(cam+1), ...])
                flow_prev_ = torch.gather(flow_prev, dim=-2, index=ray_idx.reshape(1, 1, -1, 1).repeat(1, 1, 1, self.dims)) # B, N, R, 3
                diff_prev = get_diff(pixel_prev-pix_curr_, flow_prev_) # B, N, R
                prev_mask = prev_mask & weight_mask
                diff_prev[~prev_mask] = 0.
                weight_ = weight.clone()
                weight_[~prev_mask.flatten()] = 0.
                weight_sum = torch.zeros(num_rays, dtype=weight.dtype, device=weight.device)
                weight_sum.index_add_(-1, ray_idx, weight_)
                weight_sum = weight_sum.clamp_min(torch.finfo(weight.dtype).eps) # r
                weight_sum = torch.gather(weight_sum, -1, ray_idx) # R
                weight_ = weight_ / weight_sum # R
                prev_loss = torch.zeros(num_rays, dtype=diff_prev.dtype, device=weight.device)
                prev_loss.index_add_(-1, ray_idx, weight_ * diff_prev.flatten()) # r
                invalid_mask = (abs(flow_prev[...,0]) > 2e3) & (abs(flow_prev[...,1]) > 2e3)
                prev_loss[invalid_mask.flatten()] = 0.
                # remove dynamic prev flow loss 
                prev_loss[dyn_mask] = 0.0
                valid_prev_mask = (prev_loss>0)
                cnt += valid_prev_mask.to(torch.float)
                flow_loss += prev_loss
                # flow_loss[static_mask] = (flow_loss[static_mask] + prev_loss[static_mask])/2.0
            general_mask = cnt > 0
            cnt = torch.clamp(cnt, 1.0)
            flow_loss = flow_loss / cnt
            if general_mask.sum()>0:
                valid_loss = flow_loss[general_mask].clone()
                dyn_mask = dyn_mask[general_mask]
                sort_values, sort_inds = torch.sort(valid_loss.data, dim=0)
                median = sort_values[valid_loss.numel()//2]
                loss_data = valid_loss.data
                valid_mask = (loss_data<median*100.0) & (loss_data<1e3)
                # valid_mask = torch.ones_like(loss_data)
                if (valid_loss[valid_mask]>1e3).sum()>0:
                    debug=True
                tot_loss.append(valid_loss[valid_mask])
                tot_dynamic.append(dyn_mask[valid_mask])
        if len(tot_loss)>0:
            tot_loss = torch.cat(tot_loss, dim=0)
            tot_dynamic = torch.cat(tot_dynamic, dim=0)
            dyn_num = tot_dynamic.sum().item()
            if dyn_num >0:
                tot_loss[tot_dynamic] *= min(tot_dynamic.numel() / dyn_num, 20.0)
            tot_loss = tot_loss.mean()
        # elif len(tot_loss)>0 and not self.with_static:
        #     tot_loss = torch.cat(tot_loss, dim=0)
        #     tot_dynamic = torch.cat(tot_dynamic, dim=0)
        #     tot_loss = tot_loss[tot_dynamic].mean() if tot_dynamic.sum()>0 else torch.tensor(0., device=device).float()
        else:
            tot_loss = torch.tensor(0., device=device).float()
        # self.iter_counter += 1

        return tot_loss, extra_output