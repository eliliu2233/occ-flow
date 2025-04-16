import os, numpy as np, random, mmengine, math
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import mmcv
# mmcv.use_backend('pillow')
from mmcv.image.io import imread
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET
if 'HFAI' in os.environ:
    hfai = True
    from dataset.loading import LoadMultiViewImageFromFilesHF, \
        LoadPtsFromFilesHF
else:
    hfai = False
from nuscenes import NuScenes
import time 
from tqdm import tqdm

def get_img2global(img_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(img_dict['camera_intrinsics']) if 'camera_intrinsics' in img_dict.keys() else np.asarray(img_dict['cam_intrinsic'])
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(img_dict['sensor2ego_rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(img_dict['sensor2ego_translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(img_dict['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(img_dict['ego2global_translation']).T

    img2global = ego2global @ cam2ego @ img2cam
    return img2global

def get_lidar2global(lidar_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(lidar_dict['sensor2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(lidar_dict['sensor2ego_translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(lidar_dict['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(lidar_dict['ego2global_translation']).T

    lidar2global = ego2global @ lidar2ego
    return lidar2global


@OPENOCC_DATASET.register_module()
class nuScenes_Tracking_Dataset:
    def __init__(
            self, 
            data_path, 
            queue_length=1,
            imageset='train', 
            crop_size=[768, 1600],
            input_img_crop_size=None,
            aux_sync=False,
            resize_scale=1.0,
            input_resize_scale=1.0,
            min_dist=0.4,
            max_dist=10.0,
            sensor_type=['CAM_FRONT '],
            strict=True,
            return_depth=False,
            return_depth_imgs=False,
            eval_depth=80,
            cur_prob=1.0,
            prev_prob=0.5,
            choose_nearest=False,
            ref_sensor='CAM_FRONT',
            composite_prev_next=False,
            sensor_mus=[3.0, 0.5],
            sensor_sigma=0.5,
            ego_centric=False,
            with_nearest_next=False,
            with_lidar_next=False,
            return_sem=False,
            debug=False,
            select_frames=None,
            label_path=None,
            select_scene=None,
            step=1,
            ignore_exist_flows=False,
            **kwargs):
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.step = step
        self.keyframes = data['keyframes']
        self.queue_length = queue_length
        self.with_nearest_next = with_nearest_next
        self.with_lidar_next = with_lidar_next
        self.data_path = data_path
        self.debug = debug
        self.origin_size = [900, 1600]
        self.crop_size = crop_size
        self.aug_rot_render = np.eye(2)* resize_scale
        self.aug_trans_render = -np.array([self.origin_size[1]*resize_scale-self.crop_size[1], self.origin_size[0]*resize_scale-self.crop_size[0]])
        self.input_img_crop_size = crop_size if input_img_crop_size is None else input_img_crop_size
        self.aug_rot = np.eye(2)* input_resize_scale
        self.aug_trans = -np.array([self.origin_size[1]*input_resize_scale-self.input_img_crop_size[1], self.origin_size[0]*input_resize_scale-self.input_img_crop_size[0]])
        self.strict = strict
        self.aux_sync = aux_sync
        self.return_depth = return_depth
        self.return_depth_imgs = return_depth_imgs
        self.eval_depth = eval_depth
        self.cur_prob = cur_prob
        self.prev_prob = prev_prob
        self.choose_nearest = choose_nearest
        self.composite_prev_next = composite_prev_next
        self.sensor_mus = sensor_mus[0]
        self.sensor_sigma = sensor_sigma
        self.sensor_types = sensor_type
        self.ego_centric = ego_centric
        self.return_sem = return_sem
        self.label_path = label_path
        self.semantic_map = np.array([
            0,   # ignore
            4,   # sedan      -> car
            11,  # highway    -> driveable_surface
            3,   # bus        -> bus
            10,  # truck      -> truck
            14,  # terrain    -> terrain
            16,  # tree       -> vegetation
            13,  # sidewalk   -> sidewalk
            2,   # bicycle    -> bycycle
            1,   # barrier    -> barrier
            7,   # person     -> pedestrian
            15,  # building   -> manmade
            6,   # motorcycle -> motorcycle
            5,   # crane      -> construction_vehicle
            9,   # trailer    -> trailer
            8,   # cone       -> traffic_cone
            17   # sky        -> ignore
        ], dtype=np.int8)


        if hfai:
            self.img_loader = LoadMultiViewImageFromFilesHF(
                to_float32=True,
                file_client_args=dict(
                    backend='ffrecord',
                    fname=data_path+'CAM',
                    filename2idx='img_fname2idx.pkl'))
            self.pts_loader = LoadPtsFromFilesHF(
                to_float32=True,
                file_client_args=dict(
                    backend='ffrecord',
                    fname=data_path+'LIDAR',
                    filename2idx='pts_fname2idx.pkl'))
        else:
            self.img_loader = self.pts_loader = None
        
        self.lidar2cam_rect = np.eye(4)
        if select_frames is not None and self.debug:
            key_frames = []
            for i in range(len(select_frames)):
                scene_token, index = select_frames[i].split('_')
                key_frames.append((scene_token, int(index)))
            self.keyframes_compute = key_frames
        if select_scene is not None:
            select_scene_tokens = [scene_token for scene_token in self.scene_infos.keys() if self.scene_infos[scene_token][0]['LIDAR_TOP']['scene_name'] in select_scene]
            self.keyframes_compute = [key_frame for key_frame in self.keyframes if key_frame[0] in select_scene_tokens][::step] # 
            print('finish select scene:', select_scene, len(self.keyframes_compute))
        elif ignore_exist_flows:
            flag = 0
            self.keyframes_compute = []
            for (scene_token, index) in tqdm(self.keyframes): #:self.keyframes
                info = self.scene_infos[scene_token][index]
                # for i in range(6):
                filename = os.path.join('/Data/lyl/nuscenes_flow', info[self.sensor_types[-1]]['filename'][:-4]+'_flow.npz')
                if not os.path.exists(filename) and index<len(self.scene_infos[scene_token])-1:
                    self.keyframes_compute.append((scene_token, index))
        else:
            self.keyframes_compute = deepcopy(self.keyframes)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.keyframes_compute)
    
    def get_depth_from_lidar(self, lidar_path, lidar2img, image_size, is_input=False):
        # lidar2img: N, 4, 4
        scan = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0
        # points_hcoords = scan[scan[:, 0] > 0, :]
        points_hcoords = np.expand_dims(scan.T, 0) # 1, 4, n
        img_points = np.transpose(lidar2img @ points_hcoords, (0, 2, 1)) # N, n, 4

        depth = img_points[..., 2] # N, n
        img_points = img_points[..., :2] # N, n, 2
        N, n, _ = img_points.shape
        # mask = (depth < self.eval_depth) & (depth > 1e-3)  # get points with depth < max_sample_depth
        mask = (depth < self.eval_depth) & (depth > 1.0)  # get points with depth < max_sample_depth

        img_points = img_points / np.expand_dims(depth, axis=2)  # scale 2D points
        if not is_input:
            img_points = (img_points.reshape(-1,2) @ self.aug_rot_render.T + self.aug_trans_render[None,:]).reshape(N,n,2)
            img_points[..., 0] = img_points[..., 0] / self.crop_size[1]
            img_points[..., 1] = img_points[..., 1] / self.crop_size[0]
        else:
            img_points = (img_points.reshape(-1,2) @ self.aug_rot.T + self.aug_trans[None,:]).reshape(N,n,2)
            img_points[..., 0] = img_points[..., 0] / self.input_img_crop_size[1]
            img_points[..., 1] = img_points[..., 1] / self.input_img_crop_size[0]
        # img_points = np.round(img_points).astype(int)
        mask = mask & (img_points[..., 0] >0) & \
                    (img_points[..., 1] > 0) & \
                    (img_points[..., 0] < 1) & \
                    (img_points[..., 1] < 1)

        return img_points, depth, mask
    
    
    def composite_next_dict(self, anchor_info, key_index):
        datas = []
        try:
            key_scene_token, curr_index = self.keyframes_compute[key_index]
            all_index = self.keyframes.index((key_scene_token, curr_index))
            next_scene_token, next_index = self.keyframes[all_index+1]
        except:
            return datas, False
        if next_scene_token != key_scene_token:
            return datas, False
        else:
            while True:
                data = dict()
                curr_index += 1
                if curr_index > next_index+2:
                    is_final = False
                    break
                try:
                    sensor_info = self.scene_infos[key_scene_token][curr_index]
                    for sensor_type in self.sensor_types:
                        data.update({sensor_type: sensor_info[sensor_type]})
                    datas.append(data)
                except:
                    is_final = True
                    break  
        return datas, is_final

    def __getitem__(self, index):
        #### 1. get color, temporal_depth choice if necessary
        cv2.setNumThreads(0)

        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        while True:
            key_index = index
            scene_token, index = self.keyframes_compute[index]
            info = deepcopy(self.scene_infos[scene_token][index])

            anchor_info = deepcopy(info)

            anchor_next, scene_final = self.composite_next_dict(anchor_info, key_index)
            if len(anchor_next) == 0:
                continue
            break

        #### 3. prepare img_metas
        imgs_info = self.get_data_info(info)
            
        anchor_dict = self.get_data_info_anchor(info, anchor_info)
        next_dict = self.get_data_info_temporal(anchor_info, anchor_next)

        img_metas = {
            'curr_imgs_path': anchor_dict['image_paths'],
            'next_imgs_path': next_dict['image_paths'],
            'lidar2img': imgs_info['lidar2img'],
            'img2lidar': imgs_info['img2lidar'],
            'intrinsic': imgs_info['cam_intrinsic'],
            'cam2ego': imgs_info['cam2ego'],
            'temImg2lidar': anchor_dict['temImg2lidar'],
            'ego2lidar': imgs_info['ego2lidar'],
            'token': info['LIDAR_TOP']['token'],
            # 'timestamp': info['timestamp'],
            'img2nextImg': next_dict['img2temImg'],
            'lidar2nextImg': next_dict['lidar2temImg'],
            'scene_token':scene_token, 'index':index,
            'scene_final':scene_final,
            'keyframe_index':key_index, 'scene_name':info['LIDAR_TOP']['scene_name'], 'cam_types':self.sensor_types}
            
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                info['LIDAR_TOP']['filename'], img_metas['lidar2img'], self.crop_size)
            img_metas.update({
                'depth_loc': depth_loc,
                'depth_gt': depth_gt,
                'depth_mask': depth_mask})
            
        if self.ego_centric:
            ego2lidar = img_metas['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            ego2img = img_metas['lidar2img'] @ ego2lidar[None, ...]
            img2ego = lidar2ego[None, ...] @ img_metas['img2lidar']
            temImg2ego = lidar2ego[None, ...] @ img_metas['temImg2lidar']
            ego2nextImg = img_metas['lidar2nextImg'] @ ego2lidar[None, ...]
            img_metas.update({
                'lidar2img': ego2img,
                'img2lidar': img2ego,
                'temImg2lidar': temImg2ego,
                'lidar2nextImg': ego2nextImg,
                'ego2lidar': np.eye(4)})
        
        #### 4. read imgs
        curr_img_paths = [os.path.join(self.data_path, path) for path in img_metas['curr_imgs_path']]
        curr_imgs = self.read_surround_imgs(curr_img_paths, self.crop_size)
        next_imgs = self.read_surround_imgs(img_metas['next_imgs_path'], self.crop_size)
        if self.return_sem:
            sem = []
            for sensor_type in self.sensor_types:
                cam_filename = anchor_info[sensor_type]['filename']
                semantic = np.fromfile(os.path.join(self.label_path['semantic'], cam_filename[:-4] + '_mask.bin'), dtype=np.int8).reshape(900, 1600)
                semantic = self.semantic_map[semantic]
                # Convert thing classes to 1 and background classes to 0
                semantic = np.isin(semantic, self.thing_class).astype(np.float32)
                sem.append(semantic)
            img_metas.update({'sem': sem})
        data_tuple = (
            [curr_imgs, next_imgs], [None]*3, img_metas)
        return data_tuple   
    
    def read_surround_imgs(self, img_paths, crop_size):
        cv2.setNumThreads(0)
        if hfai:
            imgs = self.img_loader.load(img_paths)
        else:
            imgs = []
            for filename in img_paths:
                # imgs.append(imread(filename, 'unchanged', 'bgr').astype(np.float32))
                imgs.append(cv2.imread(filename, -1).astype(np.float32))
        # imgs = [img[:crop_size[0], :crop_size[1], :] for img in imgs]
        return imgs
            
            
        
    def get_data_info_temporal(self, info, info_tems):
        image_paths = []
        img2temImgs = []
        lidar2temImgs = []
        lidar2global = get_lidar2global(info['LIDAR_TOP'])
        for info_tem in info_tems:
            for cam_type in self.sensor_types:

                cam_info_tem = info_tem[cam_type]
                cam_info = info[cam_type]
                image_paths.append(os.path.join(self.data_path, cam_info_tem['filename']))

                temImg2global = get_img2global(cam_info_tem)
                img2global = get_img2global(cam_info)

                img2temImg = np.linalg.inv(temImg2global) @ img2global            
                img2temImgs.append(img2temImg)
                lidar2temImg = np.linalg.inv(temImg2global) @ lidar2global
                lidar2temImgs.append(lidar2temImg)

        out_dict = dict(
            image_paths=image_paths,
            img2temImg=np.asarray(img2temImgs),
            lidar2temImg=np.asarray(lidar2temImgs))
        return out_dict
    
    def get_data_info_anchor(self, info, info_tem):
        image_paths = []
        temImg2lidars = []

        lidar2global = get_lidar2global(info['LIDAR_TOP'])

        for cam_type in self.sensor_types:

            cam_info_tem = info_tem[cam_type]
            image_paths.append(cam_info_tem['filename'])

            temImg2global = get_img2global(cam_info_tem)

            temImg2lidar = np.linalg.inv(lidar2global) @ temImg2global
            temImg2lidars.append(temImg2lidar)

        out_dict = dict(
            image_paths=image_paths,
            temImg2lidar=np.asarray(temImg2lidars))
        return out_dict

    def get_data_info(self, info):
        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []

        lidar2ego_r = Quaternion(info['LIDAR_TOP']['sensor2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['LIDAR_TOP']['sensor2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['LIDAR_TOP'])

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info[cam_type]['filename']))

            img2global = get_img2global(info[cam_type])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            img2lidar = np.linalg.inv(lidar2global) @ img2global

            cam2ego_r = Quaternion(info[cam_type]['sensor2ego_rotation']).rotation_matrix
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = cam2ego_r
            cam2ego[:3, 3] = np.array(info[cam_type]['sensor2ego_translation']).T

            intrinsic = info[cam_type]['camera_intrinsics'] if 'camera_intrinsics' in info[cam_type].keys() else info[cam_type]['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic

            lidar2img_rts.append(lidar2img)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            
        input_dict =dict(
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ego2lidar=ego2lidar,
            cam2ego=np.asarray(cam2ego_rts))
        return input_dict
