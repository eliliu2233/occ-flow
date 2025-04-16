import glob, os, time, random

import numpy as np
import torch
# from torchvision import transforms
from dataset.kitti.helpers import dump_xyz, read_calib, read_poses, get_data_path
from dataset.kitti.helpers import get_poses_kittimot, get_kittimot_calib, tracking_calib_from_txt
from dataset.kitti.params import val_error_frames
# from . import io_data as SemanticKittiIO
from copy import deepcopy
from mmcv.image.io import imread
from .. import OPENOCC_DATASET
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm
if 'HFAI' in os.environ:
    hfai = True
    from dataset.loading import LoadMultiViewImageFromFilesHF, LoadPtsFromFilesHF
else:
    hfai = False


@OPENOCC_DATASET.register_module()
class Kitti_Select_Dataset:
    def __init__(
            self,
            split,
            root, preprocess_root,
            input_img_crop_size=None,
            resize_scale=1.0,
            input_resize_scale=1.0,
            frames_interval=0.4,
            sequence_distance=10,
            n_sources=1,
            eval_depth=80,
            sequences=None,
            selected_frames=None, 
            n_rays=1200,
            cur_prob=1.0,
            crop_size=[370, 1220],
            strict=True,
            return_depth=False,
            prev_prob=0.5,
            choose_nearest=False,
            return_sem=False,
            sem_path=None,
            splits=None,
            queue_length=1,
            temporal_step=1,
            interval=1,
            return_lidar_rays=False,
            sample_lidar_rays=2048,
            dataset_type='odometry',
            **kwargs,
    ):
        self.root = root
        self.dataset_type = dataset_type
        self.temporal_step = temporal_step
        self.interval = interval
        self.preprocess_root = preprocess_root
        self.depth_preprocess_root = os.path.join(preprocess_root, "depth")
        self.transform_preprocess_root = os.path.join(preprocess_root, "transform")
        if isinstance(selected_frames, str):
            with open(selected_frames, 'r') as f:
                content = f.read()
                selected_frames = content.split('\n')[:-1]
        self.input_resize_scale = input_resize_scale
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.img_H, self.img_W = crop_size[0], crop_size[1]
        self.input_img_crop_size = crop_size if input_img_crop_size is None else input_img_crop_size
        
        self.n_classes = 20
        self.n_sources = n_sources
        self.eval_depth = eval_depth
        self.n_rays = n_rays
        self.cur_prob = cur_prob
        self.return_depth = return_depth
        self.prev_prob = prev_prob
        self.choose_nearest = choose_nearest
        self.return_sem = return_sem
        # assert (not return_sem) or os.path.exists(sem_path)
        # self.sem_path = sem_path
        
        self.transxy = [
            [0, -1., 0, 0],
            [1., 0, 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]]
        self.transxy = np.array(self.transxy)

        if splits == None:
            splits = {
                "train": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "10"],#            
                "val": ["09"],
                "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            }
        
        self.split = split
        self.queue_length = queue_length
        self.return_lidar_rays = return_lidar_rays
        self.sample_lidar_rays = sample_lidar_rays
        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = splits[split]
        self.output_scale = 1
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.frames_interval = frames_interval
        if not isinstance(sequence_distance, list):
            sequence_distance = [sequence_distance] * 2
        self.sequence_distance = sequence_distance
        self.strict = strict

        self.voxel_size = 0.2  # 0.2m
        max_frames_interval = 10
      
        start_time = time.time()
        self.scans = []
        self.frame2scan = {}
        self.all_gt_global_poses = {}

        for sequence in self.sequences:
            t1 = time.time()
            if self.dataset_type == 'mot':
                calibration_path = os.path.join(self.root, "calib", sequence + ".txt")
                tracking_calibration = tracking_calib_from_txt(calibration_path)
                
                oxt_path = os.path.join(self.root, "oxts", sequence + ".txt")
                poses_imu_w_tracking, _, _ = get_poses_kittimot(self.root, oxt_path) # (n_frames, 4, 4) imu pose
                
                tr_imu2velo = tracking_calibration["Tr_imu2velo"]
                poses_velo_w_tracking = np.matmul(poses_imu_w_tracking, np.linalg.inv(tr_imu2velo))  # (n_frames, 4, 4) velodyne pose
                gt_global_poses = np.matmul(poses_velo_w_tracking, np.linalg.inv(tracking_calibration['Tr_velo2cam']))
                self.all_gt_global_poses[sequence] = gt_global_poses

                calib = get_kittimot_calib(gt_global_poses, tracking_calibration, scene_no=int(sequence))
            elif self.dataset_type == 'odometry':
                pose_path = os.path.join(self.root, "dataset", "poses", sequence + ".txt")
                gt_global_poses = read_poses(pose_path)
                self.all_gt_global_poses[sequence] = gt_global_poses

                calib = read_calib(
                    os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
                )
            elif self.dataset_type == 'kitti360':
                os.environ['KITTI360_DATASET'] = self.root
                from dataset.kitti.kitti360Viewer3DRaw import Kitti360Viewer3DRaw
                self.velo = Kitti360Viewer3DRaw(mode='velodyne', seq=int(sequence))
                from dataset.kitti.helpers import get_kitti360_calib, get_poses_kitti360
                calib_path = os.path.join(self.root, 'calibration')
                sequence_folder = f'2013_05_28_drive_{sequence}_sync'
                calib = get_kitti360_calib(calib_path)
                gt_global_poses = get_poses_kitti360(calib, os.path.join(self.root, 'data_poses', sequence_folder, 'poses.txt'))
                self.all_gt_global_poses[sequence] = gt_global_poses
                
            P = calib["P2"]

            T_cam0_2_cam2 = calib['T_cam0_2_cam2']
            T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
            T_velo_2_cam = T_cam0_2_cam2 @ calib["Tr"]
            
            glob_path = os.path.join(
                self.root, get_data_path(self.dataset_type, sequence, 'image_2'), "*.png"
            )

            
            seq_img_paths = glob.glob(glob_path)
            t2 = time.time()
            max_length = 0
            min_length = 50
            cnt = 0
            for seq_img_path in tqdm(seq_img_paths, desc=f'loading sequence {sequence}'):
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]
                is_included = True if selected_frames is None else sequence+'_'+frame_id in selected_frames
                if not is_included:
                    continue
                
                current_img_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'image_2'), frame_id + ".png")
                current_sem_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'semantic'), frame_id + ".png")
                current_lid_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'velodyne'), frame_id + ".bin")
                if not os.path.exists(current_lid_path) or (not os.path.exists(current_sem_path) and self.return_sem):
                    print('wrong input:', current_img_path)
                    continue
    
                frame_list = list(gt_global_poses.keys()) if dataset_type == 'kitti360' else list(range(len(gt_global_poses)))
                # temporal_frames
                flag = True
                for i in range(queue_length):
                    temporal_frame_id = int(frame_id) - self.temporal_step * i
                    flag = flag & (temporal_frame_id in frame_list)
                # next_frame
                flag =  flag & (int(frame_id)+1 in frame_list)
                if not flag:
                    continue
                
                next_frame_id = str(int(frame_id)+1).zfill(len(frame_id))
                next_pose = gt_global_poses[int(next_frame_id)]
                next_img_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'image_2'), next_frame_id + ".png")
                next_lid_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'velodyne'), next_frame_id + ".bin")
                cnt += 1
                if cnt % self.interval!=0:
                    continue
                
                self.frame2scan.update({str(sequence) + "_" + frame_id: len(self.scans)})
                self.scans.append(
                    {
                        "frame_id": frame_id,
                        "sequence": sequence,
                        "img_path": current_img_path,
                        "lid_path": current_lid_path,
                        'sem_path': current_sem_path,
                        "pose": gt_global_poses[int(frame_id)],

                        "next_img_path": next_img_path,
                        "next_lid_path": next_lid_path,
                        'next_pose': next_pose,

                        "T_velo_2_cam": T_velo_2_cam,
                        "P": P,
                        "T_cam0_2_cam2": T_cam0_2_cam2,
                        "T_cam2_2_cam0": T_cam2_2_cam0,
                        "next_frame_id": next_frame_id
                    }
                    )
            print(sequence, min_length, max_length)
        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))

    def get_depth_from_lidar(self, lidar_path, lidar2img, image_size):
        # lidar2img: N, 4, 4
        frame = int(os.path.splitext(os.path.basename(lidar_path))[0])
        scan = np.fromfile(lidar_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        if self.dataset_type == 'kitti360':    
            scan = self.velo.curlVelodyneData(frame, scan)
        scan[:, 3] = 1.0
        # points_hcoords = scan[scan[:, 0] > 0, :]
        points_hcoords = np.expand_dims(self.transxy @ scan.T, 0) # 1, 4, n
        img_points = np.transpose(lidar2img @ points_hcoords, (0, 2, 1)) # N, n, 4

        depth = img_points[..., 2] # N, n
        img_points = img_points[..., :2] # N, n, 2
        N, n, _ = img_points.shape
        mask = (depth < self.eval_depth) & (depth > 1.0)  # get points with depth < max_sample_depth

        img_points = img_points / np.expand_dims(depth, axis=2)  # scale 2D points
        img_points = (img_points.reshape(-1,2) @ self.aug_rot_render.T + self.aug_trans_render[None,:]).reshape(N,n,2)
        img_points[..., 0] = img_points[..., 0] / self.crop_size[1]
        img_points[..., 1] = img_points[..., 1] / self.crop_size[0]
        # img_points = np.round(img_points).astype(int)
        mask = mask & (img_points[..., 0] > 0) & \
                    (img_points[..., 1] > 0) & \
                    (img_points[..., 0] < 1) & \
                    (img_points[..., 1] < 1)

        return img_points, depth, mask
    
    def load_2d_sem_label(self, scan):
        sequence = scan['sequence']
        filename = os.path.basename(scan['img_path'])
        sem_path = os.path.join(self.sem_path, sequence, 'image_02', filename + '.npy')
        sem = np.load(sem_path)[None, ...]
        return sem
    
    def __len__(self):
        return len(self.scans)
    
    def prepare_img_metas(self, scan, anchor_scan):
        img_metas = {}
        img_metas.update({
            'curr_imgs_path': [anchor_scan['img_path']],
            'next_imgs_path': [anchor_scan['next_img_path']]
        })

        intrinsic = np.eye(4)
        intrinsic[:3, :3] = scan['P'][:3, :3]
        lidar2img = intrinsic @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
        img2lidar = np.linalg.inv(lidar2img)

        temImg2lidar = self.transxy @ np.linalg.inv(scan['T_velo_2_cam']) @ \
            scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(scan['pose']) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        img2nextImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['next_pose']) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
                        
        img_metas.update({
            'lidar2img': np.expand_dims(lidar2img, axis=0),
            'img2lidar': np.expand_dims(img2lidar,axis=0),
            'img2nextImg': np.expand_dims(img2nextImg,axis=0),
            'token': scan['frame_id'],
            'sequence': scan['sequence'],
            'temImg2lidar': np.expand_dims(temImg2lidar,axis=0),
            'intrinsic': np.expand_dims(intrinsic,axis=0),
        })

        return img_metas

    def read_surround_imgs(self, img_paths):
        cv2.setNumThreads(0)
        if hfai:
            imgs = self.img_loader.load(img_paths)
        else:
            imgs = []
            for filename in img_paths:
                imgs.append(cv2.imread(filename, -1).astype(np.float32))
        return imgs
    
    def __getitem__(self, index):
        #### 1. get color, temporal_depth choice if necessary
        cv2.setNumThreads(0)

        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        scan = deepcopy(self.scans[index])
        sequence = scan['sequence']

        anchor_scan = deepcopy(scan)
            
        img_metas = self.prepare_img_metas(scan, anchor_scan)
        # read 6 cams
        curr_imgs = self.read_surround_imgs(img_metas['curr_imgs_path'])
        next_imgs = self.read_surround_imgs(img_metas['next_imgs_path'])
        origin_size = [curr_imgs[0].shape[0], curr_imgs[0].shape[1]]
        self.aug_rot_render = np.eye(2)* self.resize_scale
        self.aug_trans_render = -np.array([int(max(0, origin_size[1]*self.resize_scale-self.crop_size[1]) / 2), int(origin_size[0]*self.resize_scale)-self.crop_size[0]])
        self.aug_rot = np.eye(2)* self.input_resize_scale
        self.aug_trans = -np.array([int(max(0, origin_size[1]*self.input_resize_scale-self.input_img_crop_size[1])/2), int(origin_size[0]*self.input_resize_scale)-self.input_img_crop_size[0]])
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                scan['lid_path'], img_metas['lidar2img'], self.crop_size)
        if self.return_sem:
            # sem = self.load_2d_sem_label(anchor_scan)
            sem = self.read_surround_imgs([anchor_scan['sem_path']])
            img_metas.update({'sem': sem})

        data_tuple = ([curr_imgs, next_imgs], [depth_loc, depth_gt, depth_mask], img_metas)
        return data_tuple   


if __name__ == "__main__":
    import os, sys
    print(sys.path)

    dataset = Kitti_Smart(
        'train',
        root='data/kitti',
        preprocess_root='data/kitti/preprocess',
        cur_prob=1.0,
        crop_size=[352, 1216])
    
    batch = dataset[0]
    
    import pdb; pdb.set_trace()