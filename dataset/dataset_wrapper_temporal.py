
import numpy as np, torch
from torch.utils import data
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage, \
    RandomFlip, ImageAug3D
import torch.nn.functional as F
import cv2
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('occflow')
from . import OPENOCC_DATAWRAPPER

# img_norm_cfg = dict(
    # mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_temporal(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train',
            mode=1, 
            data_config=dict(image_size=[900,1600], input_size=[896,1600], render_size=[900,1600]),
            resize_config=dict(input_scale=1.0, render_scale=1.0),
            scale_rate=1,
            photometric_aug=dict(
                use_swap_channel=False,
            ),
            use_temporal_aug=False,
            temporal_aug_list=[],
            img_norm_cfg=img_norm_cfg,
            supervision_img_size=None,
            supervision_scale_rate=None,
            use_flip=False,
            ref_focal_len=None,
            pad_img_size=None,
            random_scale=None,
            pad_scale_rate=None,
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.scale_rate = scale_rate
        self.mode = mode
        self.use_temporal_aug = use_temporal_aug
        if use_temporal_aug:
            assert len(temporal_aug_list) > 0
        self.temporal_aug_list = temporal_aug_list

        photometric = PhotoMetricDistortionMultiViewImage(**photometric_aug)
        logger.info('using photometric augmentation: '+ str(photometric_aug))

        train_transforms = [
            photometric,
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)
        ]
        val_transforms = [
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)
        ]
        if scale_rate != 1 or ref_focal_len is not None or random_scale is not None or pad_scale_rate is not None:
            if mode != 1:
                train_transforms.insert(2,ImageAug3D(dict(image_size=data_config['image_size'], final_dim=data_config['input_size']), [resize_config['input_scale'],resize_config['input_scale']], [0.0,0.0], False, False))
                val_transforms.insert(1, ImageAug3D(dict(image_size=data_config['image_size'], final_dim=data_config['input_size']), [resize_config['input_scale'],resize_config['input_scale']], [0.0,0.0], False, False))
                self.temporal_transform = [ImageAug3D(dict(image_size=data_config['image_size'], final_dim=data_config['render_size']), [resize_config['render_scale'],resize_config['render_scale']], [0.0,0.0], False, False)]
            else:
                train_transforms.insert(2, RandomScaleImageMultiViewImage([scale_rate], ref_focal_len, random_scale, pad_scale_rate))
                val_transforms.insert(1, RandomScaleImageMultiViewImage([scale_rate], ref_focal_len, pad_scale_rate=pad_scale_rate))
        if use_flip:
            train_transforms.append(RandomFlip(0.5))
        
        if phase == 'train':
            self.transforms = train_transforms
        else:
            self.transforms = val_transforms
        if use_temporal_aug:
            self.temporal_transforms = [
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=4)]
            if supervision_scale_rate != 1:
                self.temporal_transforms.insert(1, RandomScaleImageMultiViewImage([supervision_scale_rate]))
        self.supervision_img_size = supervision_img_size

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            return self.deal_with_length2_dataset(data)
        elif len(data) == 3 and len(data[0])==2 and len(data[1])==3:
            return self.deal_with_selectframes_dataset(data)
        elif len(data) == 3:
            return self.deal_with_length3_dataset(data)
    
    def deal_with_length3_dataset(self, data):
        input_imgs, anchor_imgs, img_metas = data
        img_metas['img_shape'] = input_imgs[0].shape[:2]

        # deal with img augmentations
        input_imgs, imgs_dict = forward_aug(input_imgs, img_metas, self.transforms)
        if self.mode != 1:
            if 'img_aug_matrix' in imgs_dict:
                # img_metas['cam2lidar'] = img_metas['img2lidar'] @ img_metas['intrinsic']
                img_metas['img_aug_matrix'] = np.asarray(imgs_dict['img_aug_matrix'][0])
                img_metas['img2lidar'] = img_metas['img2lidar'] @ np.linalg.inv(img_metas['img_aug_matrix'])
                img_metas['lidar2img'] = img_metas['img_aug_matrix'] @ img_metas['lidar2img']
            if 'img_shape' in imgs_dict:
                img_metas['img_shape'] = imgs_dict['img_shape']
        input_imgs = self.to_tensor(input_imgs)
        if self.mode != 1:
            for i in range(len(anchor_imgs)):
                anchor_imgs[i], imgs_dict = forward_aug(anchor_imgs[i], img_metas, self.temporal_transform)
                if 'img_aug_matrix' in imgs_dict:
                    img_metas['render_img_aug_matrix'] = np.asarray(imgs_dict['img_aug_matrix'])
                    img_metas['temImg2lidars'][i] = img_metas['temImg2lidars'][i] @ np.linalg.inv(imgs_dict['img_aug_matrix'])
        #     img_metas['temImg2lidar'] = img_metas['cam2lidar'] @ np.linalg.inv(img_metas['intrinsic']) @ np.linalg.inv(img_metas['render_img_aug_matrix'])
        # else:
        #     img_metas['temImg2lidar'] = img_metas['cam2lidar'] @ np.linalg.inv(img_metas['intrinsic'])
        # img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['scale_rate'] = self.scale_rate
        # if 'focal_ratios' in imgs_dict:
        #     img_metas['focal_ratios'] = imgs_dict['focal_ratios']
        if 'focal_ratios_x' in imgs_dict:
            img_metas['focal_ratios_x'] = imgs_dict['focal_ratios_x']
        if 'focal_ratios_y' in imgs_dict:
            img_metas['focal_ratios_y'] = imgs_dict['focal_ratios_y']
        img_metas['flip'] = imgs_dict.get('flip', False)

        anchor_imgs = np.asarray(anchor_imgs).astype(np.float32)
        anchor_imgs = torch.from_numpy(anchor_imgs)
        anchor_imgs = anchor_imgs.permute(0, 1, 4, 2, 3)

        data_tuple = (input_imgs, 
                      anchor_imgs / 256.,
                      img_metas)
        return data_tuple
    
    def deal_with_length2_dataset(self, data):
        imgs, img_metas = data
        if len(imgs) == 4:
            input_imgs, curr_imgs, prev_imgs, next_imgs = imgs
            color_imgs = deepcopy(curr_imgs)
            lidar_rays = None
        elif len(imgs) == 3:
            curr_imgs, prev_imgs, next_imgs = imgs
            input_imgs = deepcopy(curr_imgs)
            color_imgs = deepcopy(curr_imgs)
            lidar_rays = None
        elif len(imgs) == 5:
            input_imgs, curr_imgs, prev_imgs, next_imgs, lidar_rays = imgs
            color_imgs = deepcopy(curr_imgs)
            if isinstance(lidar_rays, np.ndarray):
                lidar_rays = torch.from_numpy(lidar_rays)
        img_metas['img_shape'] = input_imgs[0].shape[:2]

        # deal with img augmentations
        input_imgs, imgs_dict = forward_aug(input_imgs, img_metas, self.transforms)
        img_metas['cam2lidar'] = img_metas['img2lidar'] @ img_metas['intrinsic']
        img_metas['render_img_aug_matrix'] = np.eye(4)
        if self.mode != 1:
            if 'img_aug_matrix' in imgs_dict:
                img_metas['img_aug_matrix'] = np.asarray(imgs_dict['img_aug_matrix'][0])
                img_metas['img2lidar'] = img_metas['img2lidar'] @ np.linalg.inv(img_metas['img_aug_matrix'])
                img_metas['lidar2img'] = img_metas['img_aug_matrix'] @ img_metas['lidar2img']
            if 'img_shape' in imgs_dict:
                img_metas['img_shape'] = imgs_dict['img_shape']
        input_imgs = self.to_tensor(input_imgs)
        if self.mode != 1:
            curr_imgs, imgs_dict = forward_aug(curr_imgs, img_metas, self.temporal_transform)
            for l in ['sem', 'ins']:
                if l in img_metas.keys():
                    for i in range(len(img_metas[l])):
                        sem = cv2.resize(img_metas[l][i], imgs_dict['resize_dims'], interpolation=cv2.INTER_NEAREST)
                        crop = imgs_dict['crop']
                        sem = sem[crop[1]:crop[3], crop[0]:crop[2]]
                        img_metas[l][i] = sem
            prev_imgs, _ = forward_aug(prev_imgs, img_metas, self.temporal_transform)
            next_imgs, _ = forward_aug(next_imgs, img_metas, self.temporal_transform)
            if 'img_aug_matrix' in imgs_dict:
                img_metas['render_img_aug_matrix'] = np.asarray(imgs_dict['img_aug_matrix'])
                img_metas['render_intrinsic'] = imgs_dict['img_aug_matrix'] @ img_metas['intrinsic']
                img_metas['render_c2w'] = img_metas['temImg2lidar'] @ img_metas['intrinsic']
                img_metas['temImg2lidar'] = img_metas['temImg2lidar'] @ np.linalg.inv(imgs_dict['img_aug_matrix'])
                img_metas['img2prevImg'] = imgs_dict['img_aug_matrix'] @ img_metas['img2prevImg'] @ np.linalg.inv(imgs_dict['img_aug_matrix'])
                img_metas['img2nextImg'] = imgs_dict['img_aug_matrix'] @ img_metas['img2nextImg'] @ np.linalg.inv(imgs_dict['img_aug_matrix'])
                img_metas['lidar2prevImg'] = imgs_dict['img_aug_matrix'] @ img_metas['lidar2prevImg']
                img_metas['lidar2nextImg'] = imgs_dict['img_aug_matrix'] @ img_metas['lidar2nextImg']
                if 'lidar2flowImgs' in img_metas.keys():
                    img_metas['lidar2flowImgs'] = imgs_dict['img_aug_matrix'][0] @ img_metas['lidar2flowImgs']
        curr_aug = prev_aug = next_aug = None
        if 'curr_imgs' in self.temporal_aug_list:
            curr_aug, _ = forward_aug(curr_imgs, {}, self.temporal_transforms)
            curr_aug = self.to_tensor(curr_aug)
        if 'prev_imgs' in self.temporal_aug_list:
            prev_aug, _ = forward_aug(prev_imgs, {}, self.temporal_transforms)
            prev_aug = self.to_tensor(prev_aug)
        if 'next_imgs' in self.temporal_aug_list:
            next_aug, _ = forward_aug(next_imgs, {}, self.temporal_transforms)
            next_aug = self.to_tensor(next_aug)
            
        curr_imgs = self.to_tensor(curr_imgs)
        prev_imgs = self.to_tensor(prev_imgs)
        next_imgs = self.to_tensor(next_imgs)
        color_imgs = self.to_tensor(color_imgs)
        if self.supervision_img_size is not None:
            curr_imgs = F.interpolate(curr_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)
            prev_imgs = F.interpolate(prev_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)
            next_imgs = F.interpolate(next_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)

        # img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['scale_rate'] = self.scale_rate
        # if 'focal_ratios' in imgs_dict:
        #     img_metas['focal_ratios'] = imgs_dict['focal_ratios']
        if 'focal_ratios_x' in imgs_dict:
            img_metas['focal_ratios_x'] = imgs_dict['focal_ratios_x']
        if 'focal_ratios_y' in imgs_dict:
            img_metas['focal_ratios_y'] = imgs_dict['focal_ratios_y']
        img_metas['flip'] = imgs_dict.get('flip', False)

        data_tuple = (input_imgs, 
                      curr_imgs / 256., 
                      prev_imgs / 256., 
                      next_imgs / 256., 
                      color_imgs / 256.,
                      img_metas,
                      curr_aug,
                      prev_aug,
                      next_aug,
                      lidar_rays)
        return data_tuple

    def deal_with_selectframes_dataset(self, data):
        imgs, depth_data, img_metas = data
        curr_imgs, next_imgs = imgs
        depth_loc, depth_gt, depth_mask = depth_data

        if self.mode != 1:
            curr_imgs, imgs_dict = forward_aug(curr_imgs, img_metas, self.temporal_transform)
            for l in ['sem', 'ins', 'next_ins']:
                if l in img_metas.keys():
                    for i in range(len(img_metas[l])):
                        sem = cv2.resize(img_metas[l][i], imgs_dict['resize_dims'], interpolation=cv2.INTER_NEAREST)
                        crop = imgs_dict['crop']
                        sem = sem[crop[1]:crop[3], crop[0]:crop[2]]
                        img_metas[l][i] = sem
            next_imgs, _ = forward_aug(next_imgs, img_metas, self.temporal_transform)
            if 'img_aug_matrix' in imgs_dict:
                img_metas['render_img_aug_matrix'] = np.asarray(imgs_dict['img_aug_matrix'])
                img_metas['temImg2lidar'] = img_metas['temImg2lidar'] @ np.linalg.inv(imgs_dict['img_aug_matrix'])
                img_metas['img2nextImg'] = imgs_dict['img_aug_matrix'][0] @ img_metas['img2nextImg'] @ np.linalg.inv(imgs_dict['img_aug_matrix'][0])
        
        curr_imgs = self.to_tensor(curr_imgs)
        next_imgs = self.to_tensor(next_imgs)
        if isinstance(depth_gt, np.ndarray):
                depth_loc = torch.from_numpy(depth_loc)
                depth_gt = torch.from_numpy(depth_gt)
                depth_mask = torch.from_numpy(depth_mask)
        if self.supervision_img_size is not None:
            curr_imgs = F.interpolate(curr_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)
            next_imgs = F.interpolate(next_imgs, size=self.supervision_img_size, mode='bilinear', align_corners=True)

        data_tuple = ( 
                      curr_imgs, 
                      next_imgs, 
                      depth_loc,
                      depth_gt,
                      depth_mask,
                      img_metas,
                      )
        return data_tuple

def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple

def forward_aug(imgs, metas, transforms):
    imgs_dict = {
        'img': imgs,
        'metas': metas,
    }
    for t in transforms:
        imgs_dict = t(imgs_dict)
    aug_imgs = imgs_dict['img']
    return aug_imgs, imgs_dict
