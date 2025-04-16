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
class nuScenes_Sweeps_Dist_Lidar:
    def __init__(
            self, 
            data_path, 
            split='train',
            queue_length=1,
            step=1,
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
            return_lidar_rays=False,
            sample_lidar_rays=4096,
            with_nearest_next=False,
            with_lidar_next=False,
            return_flow_3d=False,
            return_sem=False,
            label_path=None,
            return_tracking_flow=False,
            select_scene=None,
            flow_step=1,
            debug=False,
            **kwargs):
        self.split = split
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes_all = data['keyframes']
        self.queue_length = queue_length
        self.step = step
        self.keyframes = self.keyframes_all[::step]
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
        self.return_lidar_rays = return_lidar_rays
        self.sample_lidar_rays = sample_lidar_rays
        self.return_flow_3d = return_flow_3d
        self.return_tracking_flow = return_tracking_flow
        self.flow_step = flow_step if split == 'train' else 1
        self.return_sem = return_sem
        self.label_path = label_path
        self.thing_class = [2,3,4,5,6,7,9,10]
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
        if select_scene is not None:
            select_scene_tokens = [scene_token for scene_token in self.scene_infos.keys() if self.scene_infos[scene_token][0]['LIDAR_TOP']['scene_name'] in select_scene]
            self.keyframes = [keyframe for keyframe in self.keyframes if keyframe[0] in select_scene_tokens] #and keyframe[1]>100
            print('finsish select scene:', select_scene, len(self.keyframes))
        #### collect temporal information
        for scene_idx, (scene_token, scene_samples) in enumerate(self.scene_infos.items()):
            if (scene_idx + 1) % 50 == 0:
                print(f'One_Frame_Sweeps: processed {scene_idx + 1} scenes.')
            if select_scene is not None and scene_token not in select_scene_tokens:
                continue
            length = len(scene_samples)
            for i, sample in enumerate(scene_samples):
                curr_xyz = np.array(sample[ref_sensor]['ego2global_translation'])
                prev_samples, next_samples = [], []
                prev_dists, next_dists = [], []
                prev_flags, next_flags = [], []
                for j in range(i - 1, -1, -1):
                    temp_xyz = np.array(scene_samples[j][ref_sensor]['ego2global_translation'])
                    temp_dist = np.linalg.norm(curr_xyz - temp_xyz)
                    if temp_dist > max_dist:
                        break
                    if temp_dist > min_dist:
                        prev_samples.append((scene_token, j))
                        prev_dists.append(temp_dist)
                        prev_flags.append(scene_samples[j]['LIDAR_TOP']['is_key_frame'])
                
                for j in range(i + 1, length, 1):
                    temp_xyz = np.array(scene_samples[j][ref_sensor]['ego2global_translation'])
                    temp_dist = np.linalg.norm(curr_xyz - temp_xyz)
                    if temp_dist > max_dist:
                        break
                    if temp_dist > min_dist:
                        next_samples.append((scene_token, j))
                        next_dists.append(temp_dist)
                        next_flags.append(scene_samples[j]['LIDAR_TOP']['is_key_frame'])
                
                if not strict:
                    prev_samples.append((scene_token, i))
                    prev_dists.append(0.)
                    next_samples.append((scene_token, i))
                    next_dists.append(0.)
                
                sample.update({
                    'prev_samples': prev_samples,
                    'prev_dists': prev_dists,
                    'prev_flags': prev_flags,
                    'next_samples': next_samples,
                    'next_dists': next_dists,
                    'next_flags': next_flags,})

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug and self.split=='train':
            return 2000
        return len(self.keyframes)
    
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
    
    
    def composite_dict(self, anchor_info, key_index):
        datas = []
        for prefix in ['prev_', 'next_']:
            data = dict()
            dists = np.asarray(anchor_info[prefix + 'dists'])
            mu = self.sensor_mus
            sigma = self.sensor_sigma
            probs = 1 / math.sqrt(2 * math.pi) / sigma * np.exp(-1 / (2 * sigma * sigma) * ((dists - mu) ** 2))
            probs = probs / np.sum(probs)
            if prefix == 'next_' and self.with_nearest_next:
                try:
                    key_scene_token, _ = self.keyframes_all[key_index]
                    next_scene_token, next_index = self.keyframes_all[key_index+1]
                except:
                    datas.append(None)
                    continue
                if next_scene_token != key_scene_token:
                    datas.append(None)
                    continue
                else:
                    if self.return_lidar_rays: #and self.with_aux_lidar_rays
                        data.update(LIDAR_TOP=self.scene_infos[key_scene_token][next_index]['LIDAR_TOP'])
                    for sensor_type in self.sensor_types:
                        data.update({sensor_type: self.scene_infos[key_scene_token][next_index][sensor_type]})
                    datas.append(data)
                    continue
            if self.aux_sync:
                idx = np.random.choice(len(dists), p=probs)
                scene_token, sample_idx = anchor_info[prefix + 'samples'][idx]
                if self.return_lidar_rays:
                    data.update(LIDAR_TOP=self.scene_infos[scene_token][sample_idx]['LIDAR_TOP'])
                for sensor_type in self.sensor_types:
                    data.update({sensor_type: self.scene_infos[scene_token][sample_idx][sensor_type]})
            else:
                if self.return_lidar_rays:
                    # probs_keyframe = deepcopy(probs)
                    # flags = np.asarray(anchor_info[prefix + 'flags'])
                    # probs_keyframe = flags * probs_keyframe
                    # probs_keyframe = probs_keyframe / np.sum(probs_keyframe)
                    idx = np.random.choice(len(dists), p=probs)
                    scene_token, sample_idx = anchor_info[prefix + 'samples'][idx]
                    data.update(LIDAR_TOP=self.scene_infos[scene_token][sample_idx]['LIDAR_TOP'])
                for sensor_type in self.sensor_types:
                    idx = np.random.choice(len(dists), p=probs)
                    scene_token, sample_idx = anchor_info[prefix + 'samples'][idx]
                    data.update({sensor_type: self.scene_infos[scene_token][sample_idx][sensor_type]})
            datas.append(data)
        return datas[0], datas[1]

    def __getitem__(self, index):
        #### 1. get color, temporal_depth choice if necessary
        cv2.setNumThreads(0)
        if random.random() < self.cur_prob:
            temporal_supervision = 'curr'
        elif random.random() < self.prev_prob:
            temporal_supervision = 'prev'
        else:
            temporal_supervision = 'next'

        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        while True:
            if self.debug and self.split == 'train':
                index = np.random.randint(len(self.keyframes))
            scene_token, index = self.keyframes[index]
            key_index = self.keyframes_all.index((scene_token, index))
            info = deepcopy(self.scene_infos[scene_token][index])

            if temporal_supervision == 'curr':
                anchor_info = deepcopy(info)
            elif temporal_supervision == 'prev':
                if len(info['prev_samples']) == 0:
                    index = np.random.randint(len(self.keyframes))
                    continue
                anchor_scene_token, anchor_info_id = info['prev_samples'][np.random.randint(len(info['prev_samples']))]
                # anchor_scene_token, anchor_info_id = np.random.choice(info['prev_samples'])
                assert anchor_scene_token == scene_token and anchor_info_id <= index
                anchor_info = deepcopy(self.scene_infos[scene_token][anchor_info_id])
            else:
                if len(info['next_samples']) == 0:
                    index = np.random.randint(len(self.keyframes))
                    continue
                anchor_scene_token, anchor_info_id = info['next_samples'][np.random.randint(len(info['next_samples']))]
                # anchor_scene_token, anchor_info_id = np.random.choice(info['next_samples'])
                assert anchor_scene_token == scene_token and anchor_info_id >= index
                anchor_info = deepcopy(self.scene_infos[scene_token][anchor_info_id])

            if len(anchor_info['prev_samples']) == 0 or \
                len(anchor_info['next_samples']) == 0:
                index = np.random.randint(len(self.keyframes))
                continue
            
            if self.return_tracking_flow:
                flow_infos = []
                try:
                    final_scene_token, final_index = self.keyframes_all[key_index+self.flow_step]
                    if final_scene_token != scene_token:
                        index = np.random.randint(len(self.keyframes))
                        continue
                except:
                    index = np.random.randint(len(self.keyframes))
                    continue
                for i in range(self.flow_step):
                    curr_scene_token, curr_index = self.keyframes_all[key_index+i]
                    flow_infos.append(self.keyframes_all[key_index+i])
                flow_infos.append((final_scene_token, final_index))

            if self.composite_prev_next:
                anchor_prev, anchor_next = self.composite_dict(anchor_info, key_index)
            else:
                if self.choose_nearest:
                    anchor_prev_scene_token, anchor_prev_idx = anchor_info['prev_samples'][0]
                    anchor_next_scene_token, anchor_next_idx = anchor_info['next_samples'][0]
                else:
                    anchor_prev_scene_token, anchor_prev_idx = anchor_info['prev_samples'][np.random.randint(len(anchor_info['prev_samples']))]
                    anchor_next_scene_token, anchor_next_idx = anchor_info['next_samples'][np.random.randint(len(anchor_info['next_samples']))]
                assert anchor_prev_scene_token == scene_token and anchor_next_scene_token == scene_token
                anchor_prev = deepcopy(self.scene_infos[scene_token][anchor_prev_idx])
                anchor_next = deepcopy(self.scene_infos[scene_token][anchor_next_idx])
            if anchor_next == None:
                index = np.random.randint(len(self.keyframes))
                continue
            break

        #### 3. prepare img_metas
        imgs_info = self.get_data_info(info)
        temporal_info = deepcopy(info)
        prev_to_curr_rt = []
        for i in range(1, self.queue_length):
            temporal_scene_token, temporal_index = self.keyframes_all[max(0,key_index-i)]
            if temporal_scene_token == scene_token:
                temporal_info = self.scene_infos[temporal_scene_token][temporal_index]
            for cam_type in self.sensor_types:
                imgs_info['img_filename'].append(os.path.join(self.data_path, temporal_info[cam_type]['filename']))
            prev_to_curr_rt.append(np.linalg.inv(get_lidar2global(info['LIDAR_TOP'])) @ get_lidar2global(temporal_info['LIDAR_TOP']))
            
        anchor_dict = self.get_data_info_anchor(info, anchor_info)
        prev_dict = self.get_data_info_temporal(anchor_info, anchor_prev)
        next_dict = self.get_data_info_temporal(anchor_info, anchor_next)

        img_metas = {
            'input_imgs_path': imgs_info['img_filename'],
            'curr_imgs_path': anchor_dict['image_paths'],
            'prev_imgs_path': prev_dict['image_paths'],
            'next_imgs_path': next_dict['image_paths'],
            'lidar2img': imgs_info['lidar2img'],
            'img2lidar': imgs_info['img2lidar'],
            'intrinsic': imgs_info['cam_intrinsic'],
            'cam2ego': imgs_info['cam2ego'],
            'temImg2lidar': anchor_dict['temImg2lidar'],
            'ego2lidar': imgs_info['ego2lidar'],
            'prev_to_curr_rt': prev_to_curr_rt,
            'token': info['LIDAR_TOP']['token'],
            # 'timestamp': info['timestamp'],
            'img2prevImg': prev_dict['img2temImg'],
            'img2nextImg': next_dict['img2temImg'],
            'lidar2prevImg': prev_dict['lidar2temImg'],
            'lidar2nextImg': next_dict['lidar2temImg'],
            'scene_token':scene_token, 'index':index,
            'keyframe_index': key_index, 'scene_name': info['LIDAR_TOP']['scene_name'], 'cam_types':self.sensor_types}
        
        if self.return_flow_3d:
            token = info['LIDAR_TOP']['token']
            # scene_name = self.nusc.get('scene', scene_token)['name']
            scene_name = info['LIDAR_TOP']['scene_name']
            label_file = os.path.join(self.data_path, f'openocc_v2/{scene_name}/{token}/labels.npz')
            label_data = np.load(label_file)
            flow_3d = label_data['flow'].astype(np.float32)
            semantic_3d = label_data['semantics'].astype(np.float32)
            occ_mask = semantic_3d != 16
            img_metas.update(flow_3d=flow_3d, occ_mask=occ_mask)
        
        if self.return_tracking_flow:
            flows = []
            lidar2global = get_lidar2global(anchor_info['LIDAR_TOP'])
            lidar2temImgs, next_times = [], []
            for step in range(self.flow_step):
                scene_token, curr_index = flow_infos[step]
                curr_info = self.scene_infos[scene_token][curr_index]
                scene_token, next_index = flow_infos[step+1]
                for sensor_type in self.sensor_types:
                    flow_file = os.path.join(self.label_path['flow'], curr_info[sensor_type]['filename'][:-4]+'_flow.npz')
                    flow = np.load(flow_file)['flow'][-1:].transpose(1,2,0,3)
                    flow = flow.reshape(flow.shape[0], flow.shape[1], flow.shape[2]*flow.shape[3])
                    flows.append(flow)
                start_index = next_index # curr_index+1
                for temp_index in range(start_index, next_index+1):
                    info_tem = self.scene_infos[scene_token][temp_index]
                    for sensor_type in self.sensor_types:
                        cam_info_tem = info_tem[sensor_type]
                        temImg2global = get_img2global(cam_info_tem)
                        lidar2temImg = np.linalg.inv(temImg2global) @ lidar2global
                        lidar2temImgs.append(lidar2temImg)
                        next_times.append(round(cam_info_tem['timestamp']*1e-6-curr_info[sensor_type]['timestamp']*1e-6, 2))
            img_metas.update({'tracking_flow': np.asarray(flows), 
                              'lidar2flowImgs': np.asarray(lidar2temImgs).reshape(self.flow_step, -1, len(self.sensor_types), 4, 4), 
                              'next_times': np.asarray(next_times).reshape(self.flow_step, -1, len(self.sensor_types))})
            
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
            
            
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                info['LIDAR_TOP']['filename'], img_metas['lidar2img'], self.crop_size)
            img_metas.update({
                'depth_loc': depth_loc,
                'depth_gt': depth_gt,
                'depth_mask': depth_mask})
        if self.return_depth_imgs:
            img_points, depth, mask = self.get_depth_from_lidar(info['LIDAR_TOP']['filename'], img_metas['lidar2img'], self.input_img_crop_size)
            u = (img_points[...,0] * self.input_img_crop_size[1]).astype(int)
            v = (img_points[...,1] * self.input_img_crop_size[0]).astype(int)
            depth_imgs = np.zeros([u.shape[0], *self.input_img_crop_size])
            for i in range(u.shape[0]):
                depth_imgs[i, v[i][mask[i]],u[i][mask[i]]] = depth[i][mask[i]]
            img_metas.update(input_depth_imgs=depth_imgs)
            
        if self.ego_centric:
            ego2lidar = img_metas['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            ego2img = img_metas['lidar2img'] @ ego2lidar[None, ...]
            img2ego = lidar2ego[None, ...] @ img_metas['img2lidar']
            temImg2ego = lidar2ego[None, ...] @ img_metas['temImg2lidar']
            ego2prevImg = img_metas['lidar2prevImg'] @ ego2lidar[None, ...]
            ego2nextImg = img_metas['lidar2nextImg'] @ ego2lidar[None, ...]
            prev_to_curr_rt = [lidar2ego @ img_metas['prev_to_curr_rt'][i] @ ego2lidar for i in range(len(img_metas['prev_to_curr_rt']))]
            if 'lidar2flowImgs' in img_metas.keys():
                img_metas['lidar2flowImgs'] = img_metas['lidar2flowImgs'] @ ego2lidar[None, ...]
            img_metas.update({
                'lidar2img': ego2img,
                'img2lidar': img2ego,
                'temImg2lidar': temImg2ego,
                'lidar2prevImg': ego2prevImg,
                'lidar2nextImg': ego2nextImg,
                'prev_to_curr_rt': prev_to_curr_rt,
                'ego2lidar': np.eye(4)})
        
        #### 4. read imgs
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'], self.input_img_crop_size)
        curr_imgs = self.read_surround_imgs(img_metas['curr_imgs_path'], self.crop_size)
        prev_imgs = self.read_surround_imgs(img_metas['prev_imgs_path'], self.crop_size)
        next_imgs = self.read_surround_imgs(img_metas['next_imgs_path'], self.crop_size)
        lidar_rays = None
        if self.return_lidar_rays:
            lidar2global = get_lidar2global(info['LIDAR_TOP'])
            if self.ego_centric:
                lidar2global = lidar2global @ imgs_info['ego2lidar']
            lidar_infos = [anchor_info]
            lidar_result = self.read_lidar_rays(lidar_infos, lidar2global, self.crop_size)
            lidar_rays = lidar_result['lidar_rays']
            lidar_points = lidar_result['lidar_points']
        # t3 = time.time()
        # print('loading images time:', t2-t1, 'loading lidar rays time:', t3-t2)
        data_tuple = (
            [input_imgs, curr_imgs, prev_imgs, next_imgs, lidar_rays], img_metas)
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

    def read_lidar_rays(self, infos, lidar2global, image_size):
        lidar_rays = []
        lidar_points = []
        result_dict = {}
        anchor_info = infos[0]
        for idx, info in enumerate(infos):
            temLidar2global = get_lidar2global(info['LIDAR_TOP'])
            sensor2lidar = np.linalg.inv(lidar2global) @ temLidar2global
            scan = np.fromfile(os.path.join(self.data_path, info['LIDAR_TOP']['filename']), dtype=np.float32).reshape((-1, 5))[:, :3]
            scan_mask = np.zeros(scan.shape[0]).astype(bool)
            scan_ins = np.zeros_like(scan[:,0], dtype=int)
            for cam_id, cam_type in enumerate(self.sensor_types):
                cam_filename = anchor_info[cam_type]['filename']
                temImg2global = get_img2global(anchor_info[cam_type])
                temLidar2temImg = np.linalg.inv(temImg2global) @ temLidar2global
                pixel = scan @ temLidar2temImg[:3,:3].T + temLidar2temImg[:3,3][None,:]
                mask =  np.logical_and(pixel[:,2]>1.0, pixel[:,2]<self.eval_depth)
                pixel = pixel[:,:2] / pixel[:,2:3]
                pixel = (pixel @ self.aug_rot_render.T + self.aug_trans_render[None,:])
                mask = mask & (pixel[:, 0] > 0) & (pixel[:, 0] < self.crop_size[1]) & (pixel[:, 1] > 0) & (pixel[:, 1] < self.crop_size[0])
                scan_mask = scan_mask | mask
            # scan = scan[scan_mask]
            depth_range = np.linalg.norm(scan, axis=1)
            rays_d = (scan / depth_range[:,None]) @ sensor2lidar[:3,:3].T
            rays_o = sensor2lidar[:3,3][None,:].repeat(rays_d.shape[0],axis=0)
            frame_id = idx *np.ones_like(depth_range)
            if idx == 0:
                lidar_rays.append(np.concatenate((rays_o[scan_mask], rays_d[scan_mask], frame_id[scan_mask][:,None], depth_range[scan_mask][:,None]), axis=1))
                lidar_points.append(np.concatenate((scan[scan_mask]@sensor2lidar[:3,:3].T + sensor2lidar[:3,3][None,:], 
                                                  np.zeros_like(scan[scan_mask][:,0:1])), axis=1))
        lidar_rays = np.vstack(lidar_rays)
        inds = np.random.permutation(lidar_rays.shape[0])[:self.sample_lidar_rays]
        result_dict.update(lidar_rays=lidar_rays[inds], lidar_points=np.vstack(lidar_points))
        return result_dict
            
            
        
    def get_data_info_temporal(self, info, info_tem):
        image_paths = []
        img2temImgs = []
        lidar2temImgs = []
        lidar2global = get_lidar2global(info['LIDAR_TOP'])
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
            image_paths.append(os.path.join(self.data_path, cam_info_tem['filename']))

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