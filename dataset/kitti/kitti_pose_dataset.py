import glob, os, time, random
import math
import numpy as np
import torch
# from torchvision import transforms
from dataset.kitti.helpers import dump_xyz, get_poses_kittimot, get_kittimot_calib, tracking_calib_from_txt, get_data_path
from dataset.kitti.helpers import read_calib, read_poses
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
class Kitti_EGO_Dataset:
    def __init__(
            self,
            split,
            root, 
            frames_interval=0,
            sequence_distance=10,
            n_sources=1,
            eval_depth=80,
            sequences=None,
            selected_frames=None, 
            exclude_selected=False,
            step=1,
            n_rays=1200,
            cur_prob=1.0,
            crop_size=[370, 1220],
            prev_prob=0.5,
            splits=None,
            dataset_type='mot',
            **kwargs,
    ):
        self.root = root
        self.dataset_type = dataset_type
        if isinstance(selected_frames, str):
            with open(selected_frames, 'r') as f:
                content = f.read()
                selected_frames = content.split('\n')[:-1]
        self.crop_size = crop_size
        self.img_H, self.img_W = crop_size[0], crop_size[1]
        
        self.n_classes = 20
        self.n_sources = n_sources
        self.eval_depth = eval_depth
        self.n_rays = n_rays
        self.cur_prob = cur_prob
        self.prev_prob = prev_prob
        
        self.transxy = [
            [0, -1., 0, 0],
            [1., 0, 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]]
        self.transxy = np.array(self.transxy)

        if splits == None:
            splits = {
                "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],#            
                "val": ["08"],
                "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            }
        self.split = split

        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = splits[split]
        self.output_scale = 1
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        if not isinstance(sequence_distance, list):
            frames_interval = [frames_interval] * 2
        self.frames_interval = frames_interval
        if not isinstance(sequence_distance, list):
            sequence_distance = [sequence_distance] * 2
        self.sequence_distance = sequence_distance

        self.voxel_size = 0.2  # 0.2m
      
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
            elif self.dataset_type == 'odometry':
                pose_path = os.path.join(self.root, "dataset", "poses", sequence + ".txt")
                gt_global_poses = read_poses(pose_path)
                self.all_gt_global_poses[sequence] = gt_global_poses

                calib = read_calib(os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt"))
            P = calib["P2"]

            T_cam0_2_cam2 = calib['T_cam0_2_cam2']
            T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
            T_velo_2_cam = T_cam0_2_cam2 @ calib["Tr"]

            if split == "val" and self.dataset_type=='odometry':
                glob_path = os.path.join(self.root, "dataset", "sequences", sequence, "voxels", "*.bin")
            else:
                glob_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'image_2'), "*.png")

            
            seq_img_paths = sorted(glob.glob(glob_path))[::step]
            t2 = time.time()
            max_length = 0
            min_length = 50

            for seq_img_path in tqdm(seq_img_paths, desc=f'loading sequence {sequence}'):
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]
                # ignore error frame
                if split == "val" and self.dataset_type == 'odometry' and frame_id in val_error_frames:
                    continue
                
                is_included = True if selected_frames is None else frame_id in selected_frames #sequence+'_'+
                if exclude_selected:
                    is_included = not is_included
                if not is_included:
                    continue
                
                current_img_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'image_2'), frame_id + ".png")
                current_lid_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'velodyne'), frame_id + ".bin")
                if not os.path.exists(current_lid_path):
                    continue

                prev_frame_ids, next_frame_ids = [], []
                prev_img_paths, next_img_paths = [], []
                prev_lid_paths, next_lid_paths = [], []
                prev_poses, next_poses = [], []
                prev_dists, next_dists = [], []
        
                frame_list = list(gt_global_poses.keys()) if dataset_type == 'kitti360' else list(range(len(gt_global_poses)))
                # deal with next
                cnt = 0
                dist = 0.
                pos_step = 1
                neg_step = -1
                curr_xyz = dump_xyz(gt_global_poses[int(frame_id)])
                next_poses.append(gt_global_poses[int(frame_id)])
                next_frame_ids.append(frame_id)
                next_dists.append(0.0)
                while True:
                    # cnt += step
                    cnt += pos_step
                    rel_frame_id = str(int(frame_id) + cnt).zfill(len(frame_id))
                    if abs(cnt) > 40:
                        break
                    img_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'image_2'), rel_frame_id + ".png")

                    if not os.path.exists(img_path) or not int(rel_frame_id) in frame_list:
                        continue

                    next_xyz = dump_xyz(gt_global_poses[int(rel_frame_id)])
                    dist = np.sqrt((next_xyz[0] - curr_xyz[0]) ** 2 + (next_xyz[1] - curr_xyz[1]) ** 2)
                    if dist < frames_interval[1]:
                        continue
                    if dist > sequence_distance[1]:
                        break

                    next_frame_ids.append(rel_frame_id)
                    next_img_paths.append(img_path)
                    lidar_path = os.path.join(self.root, get_data_path(self.dataset_type, sequence, 'velodyne'), rel_frame_id + ".bin")
                    next_lid_paths.append(lidar_path)
                    next_poses.append(gt_global_poses[int(rel_frame_id)])
                    next_dists.append(dist)

                prev_next_len = len(prev_poses) + len(next_poses)
                if prev_next_len > max_length:
                    max_length = prev_next_len
                if prev_next_len < min_length:
                    min_length = prev_next_len

                self.frame2scan.update({str(sequence) + "_" + frame_id: len(self.scans)})

                scan = {
                        "frame_id": frame_id,
                        "sequence": sequence,
                        "img_path": current_img_path,
                        "lid_path": current_lid_path,
                        "pose": gt_global_poses[int(frame_id)],

                        "prev_img_paths": prev_img_paths,
                        "prev_lid_paths": prev_lid_paths,
                        "next_img_paths": next_img_paths,
                        "next_lid_paths": next_lid_paths,

                        "T_velo_2_cam": T_velo_2_cam,
                        "P": P,
                        "T_cam0_2_cam2": T_cam0_2_cam2,
                        "T_cam2_2_cam0": T_cam2_2_cam0,

                        "next_poses": next_poses,
                        "next_dists": next_dists,
                        "next_frame_ids": next_frame_ids
                    }
                self.scans.append(scan)
            print(sequence, min_length, max_length)
        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))
    
    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, index):
        #### 1. get color, temporal_depth choice if necessary
        cv2.setNumThreads(0)
        scan = deepcopy(self.scans[index])
        ref_sample_token = scan['sequence'] + '_' + scan['frame_id']
        output_origin_list = []
        curr_ego2global = scan['pose'] @ scan['T_cam2_2_cam0'] @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
        for i in range(len(scan['next_poses'])):
            temp_lidar2global = scan['next_poses'][i] @ scan['T_cam2_2_cam0'] @ scan['T_velo_2_cam']
            temp2curr = np.linalg.inv(curr_ego2global) @ temp_lidar2global
            origin_tf = np.array(temp2curr[:3, 3], dtype=np.float32)
            output_origin_list.append(origin_tf)
            
        select_num = 8
        if len(output_origin_list) > select_num:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, select_num)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]
        
        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]
        return (ref_sample_token, output_origin_tensor) 
