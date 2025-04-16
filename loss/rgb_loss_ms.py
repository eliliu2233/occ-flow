import torch.nn as nn, torch
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import numpy as np
from utils.vis_utils import save_surroundimg, semantic_colors

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@OPENOCC_LOSS.register_module()
class RGBLossMS(BaseLoss):

    def __init__(
            self, 
            weight=1.0, 
            img_size=None, 
            no_ssim=True, 
            ray_resize=None, 
            input_dict=None, 
            **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ms_colors': 'ms_colors',
                'ms_rays': 'ms_rays',
                'gt_imgs': 'gt_imgs'}
        else:
            self.input_dict = input_dict
        assert img_size is not None
        self.img_size = img_size
        self.loss_func = self.rgb_loss
        self.no_ssim = no_ssim or ray_resize is None
        self.ray_resize = ray_resize
        if not self.no_ssim:
            self.ssim = SSIM()        
    
    def rgb_loss(self, ms_colors, ms_rays, gt_imgs):
        # rgb: B, N, R, 3
        # rays: R, 2
        # curr_imgs: B, N, 3, H, W
        bs, num_cams = gt_imgs.shape[:2]

        if not isinstance(ms_rays, list):
            rays = ms_rays
            pixels = rays.reshape(1, 1, -1, 2).repeat(bs*num_cams, 1, 1, 1) # B*N, 1, R, 2
            pixels[..., 0] /= self.img_size[1]
            pixels[..., 1] /= self.img_size[0]
            gt = F.grid_sample(
                gt_imgs.flatten(0, 1),
                pixels * 2 - 1,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True) # B*N, 3, 1, R
            gt = gt.reshape(bs, num_cams, 3, -1).transpose(-1, -2) # B, N, R, 3
        else:
            raise NotImplementedError
        
        tot_loss = 0.
        for color in ms_colors:
            loss = torch.abs(color - gt).mean()
            if not self.no_ssim:
                color_reshaped = color.reshape(bs*num_cams, *self.ray_resize, 3).permute(0, 3, 1, 2)
                gt_reshaped = gt.reshape(bs*num_cams, *self.ray_resize, 3).permute(0, 3, 1, 2)
                ssim_loss = self.ssim(color_reshaped, gt_reshaped).mean()
                loss = 0.15 * loss + 0.85 * ssim_loss
            tot_loss += loss

        return tot_loss / len(ms_colors)


@OPENOCC_LOSS.register_module()
class SemLossMS(BaseLoss):

    def __init__(
            self, 
            weight=1.0, 
            img_size=None, 
            ray_resize=None, 
            input_dict=None, 
            **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'sem': 'sem',
                'metas': 'metas',
                'ms_rays': 'ms_rays'}
        else:
            self.input_dict = input_dict
        assert img_size is not None
        self.img_size = img_size
        self.loss_func = self.sem_loss
        self.ray_resize = ray_resize
    
    def sem_loss(self, sem, metas, ms_rays):
        # sem: B, N, R, c
        # rays: R, 2
        # curr_imgs: B, N, 3, H, W
        gt_imgs = []
        for meta in metas:
            gt_imgs.append(meta['sem'])
        if isinstance(gt_imgs[0], np.ndarray):
            gt_imgs = np.asarray(gt_imgs)
            gt_imgs = sem[0].new_tensor(gt_imgs, dtype=torch.long) # B, N, H, W
        elif isinstance(gt_imgs[0], torch.Tensor):
            gt_imgs = torch.stack(gt_imgs).to(sem[0].device)
        else:
            raise NotImplementedError
        num_cls = sem[0].shape[-1]

        if not isinstance(ms_rays, list):
            rays = ms_rays.to(torch.long)
            gt = gt_imgs[:, :, rays[:, 1], rays[:, 0]] # B, N, R
            gt = F.one_hot(gt, num_classes=num_cls).to(torch.float) # B, N, R, C
        else:
            raise NotImplementedError
        
        tot_loss = 0.
        for s in sem:
            s = torch.clamp(s, 0, 1)
            loss = F.binary_cross_entropy(s, gt)
            tot_loss += loss

        return tot_loss / len(sem)



@OPENOCC_LOSS.register_module()
class SemCELossMS(BaseLoss):

    def __init__(
            self, 
            weight=1.0, 
            img_size=None, 
            ray_resize=None, 
            input_dict=None, 
            **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'sem': 'sem',
                'metas': 'metas',
                'ms_rays': 'ms_rays'}
        else:
            self.input_dict = input_dict
        assert img_size is not None
        self.img_size = img_size
        self.loss_func = self.sem_loss
        self.ray_resize = ray_resize
    
    def sem_loss(self, sem, metas, ms_rays):
        # sem: B, N, R, c
        # rays: R, 2
        # curr_imgs: B, N, 3, H, W
        gt_imgs = []
        for meta in metas:
            gt_imgs.append(meta['sem'])
        if isinstance(gt_imgs[0], np.ndarray):
            gt_imgs = np.asarray(gt_imgs)
            gt_imgs = sem[0].new_tensor(gt_imgs, dtype=torch.long) # B, N, H, W
        elif isinstance(gt_imgs[0], torch.Tensor):
            gt_imgs = torch.stack(gt_imgs).to(sem[0].device)
        else:
            raise NotImplementedError
        num_cls = sem[0].shape[-1]

        if not isinstance(ms_rays, list):
            rays = ms_rays.to(torch.long)
            gt = gt_imgs[:, :, rays[:, 1], rays[:, 0]] # B, N, R
            gt = F.one_hot(gt, num_classes=num_cls).to(torch.float) # B, N, R, C
        else:
            raise NotImplementedError
        
        tot_loss = 0.
        for s in sem:
            s = torch.clamp(s, 1e-6, 1)
            loss = torch.mean(torch.sum((-1) * torch.log(s) * gt, dim=-1))
            # loss = F.binary_cross_entropy(s, gt)
            tot_loss += loss

        return tot_loss / len(sem)
    
@OPENOCC_LOSS.register_module()
class SemLoss(BaseLoss):

    def __init__(
            self, 
            weight=1.0, 
            img_size=None, 
            ray_resize=None, 
            input_dict=None, 
            **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'sem': 'sem',
                'metas': 'metas',
                'ms_rays': 'ms_rays'}
        else:
            self.input_dict = input_dict
        assert img_size is not None
        self.img_size = img_size
        self.with_sem_free = kwargs.get('with_sem_free', False)
        self.class_frequencies = [0, 18164955800, 734218842, 2336187448, 10996756106, 395414611,
                                           638889260, 651023279, 117046208, 985341947, 7303776233, 43131984997,
                                           0, 9342674867, 9525824742, 51204832885, 22848065525]
        if self.with_sem_free:
            self.class_frequencies.append(31841090018)
        self.class_weights = 1.0 / np.sqrt(np.array(self.class_frequencies, dtype=np.float32))
        self.class_weights = np.nan_to_num(self.class_weights, posinf=0)
        self.class_weights = self.class_weights / np.mean(self.class_weights)
        self.sem_criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.class_weights).cuda(),
            ignore_index=-1, reduction="mean")
        self.debug = kwargs.get('debug', False)
        self.loss_func = self.sem_loss
        self.ray_resize = ray_resize
    
    def sem_loss(self, sem, semantic_mask, metas, ms_rays=None, semantic_gt=None, curr_imgs=None):
        # sem: B, N, R, c
        # rays: R, 2
        # curr_imgs: B, N, 3, H, W
        num_cls = sem[0].shape[-1]
        bs, num_cams = sem[0].shape[:2]
        if semantic_gt is None:
            gt_imgs = semantic_mask
            if ms_rays is not None:
                rays = ms_rays.to(torch.long)
                gt = gt_imgs[:, :, rays[:, 1], rays[:, 0]] # B, N, R
                # gt = F.one_hot(gt, num_classes=num_cls).to(torch.float) # B, N, R, C
            else:
                gt = F.interpolate(semantic_mask.flatten(0, 1).unsqueeze(1).float(), size=self.ray_resize, mode='nearest').squeeze(1)
        else:
            gt = semantic_gt
        if self.debug:
            import cv2
            sem_imgs = []
            for i in range(num_cams):
                img = cv2.resize(curr_imgs[0,i].permute(1,2,0).cpu().numpy()*255.0, (curr_imgs.size(-1)//2, curr_imgs.size(-2)//2))
                uv = (metas[0]['lidarseg'][i][:,:2]/2).astype(int)
                img[uv[:,1], uv[:,0]] = semantic_colors[metas[0]['lidarseg'][i][:,2].astype(int)][:,-2::-1]
                sem_imgs.append(img.copy())
            save_surroundimg(sem_imgs, './lidarseg.png', metas[0]['cam_types'])
        if not self.with_sem_free:
            sky_mask = gt != 17
            semantic_loss = self.sem_criterion(sem[0][sky_mask].view(-1,num_cls).float(), gt[sky_mask].view(-1).long())
        else:
            semantic_loss = self.sem_criterion(sem[0].view(-1,num_cls).float(), gt.view(-1).long())
        return semantic_loss
