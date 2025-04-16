import glob, os, time, random
import math
import numpy as np
import torch
# from torchvision import transforms
from dataset.kitti.helpers import dump_xyz, read_calib, read_poses
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
class Kitti_One_Frame:
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
            step=1,
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
            return_lidar_rays=False,
            sample_lidar_rays=2048,
            debug=False,
            **kwargs,
    ):
        self.root = root
        # self.dataset_type = dataset_type
        self.preprocess_root = preprocess_root
        self.depth_preprocess_root = os.path.join(preprocess_root, "depth")
        self.transform_preprocess_root = os.path.join(preprocess_root, "transform")
        self.debug = debug
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

        # splits = {
        #     "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],                   
        #     "val": ["08"],
        #     "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        # }
        if splits == None:
            splits = {
                "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],#            
                "val": ["08"],
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
        max_frames_interval = 1
      
        start_time = time.time()
        self.scans = []
        self.frame2scan = {}
        self.all_gt_global_poses = {}

        for sequence in self.sequences:
            t1 = time.time()
            pose_path = os.path.join(self.root, "dataset", "poses", sequence + ".txt")
            gt_global_poses = read_poses(pose_path)
            self.all_gt_global_poses[sequence] = gt_global_poses

            calib = read_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]

            T_cam0_2_cam2 = calib['T_cam0_2_cam2']
            T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
            T_velo_2_cam = T_cam0_2_cam2 @ calib["Tr"]

            if split == "val":
                glob_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
                )

            else:
                glob_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "image_2", "*.png"
                )

            
            seq_img_paths = glob.glob(glob_path)[::step]
            t2 = time.time()
            max_length = 0
            min_length = 50
            # paired_dists = {}
            # dist_step = 1 if split == 'train' else 5
            # for seq_img_path in seq_img_paths:
            #     filename = os.path.basename(seq_img_path)
            #     frame_id = os.path.splitext(filename)[0]
            #     prev_frame_id = "{:06d}".format(int(frame_id) - dist_step)
            #     img_path = os.path.join(
            #         self.root, "dataset", "sequences", sequence, "image_2", prev_frame_id + ".png")
            #     if not os.path.exists(img_path):
            #         paired_dists[frame_id] = 0.
            #     else:
            #         curr_pose = gt_global_poses[int(frame_id)]
            #         prev_pose = gt_global_poses[int(prev_frame_id)]
            #         prev_xyz = dump_xyz(prev_pose)
            #         curr_xyz = dump_xyz(curr_pose)
            #         rel_distance = np.sqrt(
            #             (prev_xyz[0] - curr_xyz[0]) ** 2 + (prev_xyz[2] - curr_xyz[2]) ** 2)
            #         paired_dists[frame_id] = rel_distance
            # t3 = time.time()
            # print('loading pose time:', t2-t1, 'computing dist time:', t3-t2)
            for seq_img_path in tqdm(seq_img_paths, desc=f'loading sequence {sequence}'):
                filename = os.path.basename(seq_img_path)
                frame_id = os.path.splitext(filename)[0]
                
                # ignore error frame
                # if split == "val" and frame_id in val_error_frames:
                #     continue

                current_img_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "image_2", frame_id + ".png")
                current_sem_path = os.path.join(self.root, "dataset", "sequences", sequence, "semantic", frame_id + ".png")
                current_lid_path = os.path.join(
                    self.root, "dataset", "sequences", sequence, "velodyne", frame_id + ".bin")

                prev_frame_ids, next_frame_ids = [], []
                prev_img_paths, next_img_paths = [], []
                prev_lid_paths, next_lid_paths = [], []
                prev_poses, next_poses = [], []
                prev_dists, next_dists = [], []
                
                is_included = True if selected_frames is None else sequence+'_'+frame_id in selected_frames
                if not is_included:
                    continue
                
                if self.split == 'val':
                    prev_frame_id = "{:06d}".format(max(0, int(frame_id)-1))
                    prev_frame_ids.append(prev_frame_id)
                    next_frame_id = "{:06d}".format(min(len(gt_global_poses)-1, int(frame_id)+1))
                    next_frame_ids.append(next_frame_id)
                    prev_poses.append(gt_global_poses[int(prev_frame_id)])
                    next_poses.append(gt_global_poses[int(next_frame_id)])
                    prev_img_paths.append(os.path.join(self.root, "dataset", "sequences", sequence, "image_2", prev_frame_id + ".png"))
                    next_img_paths.append(os.path.join(self.root, "dataset", "sequences", sequence, "image_2", next_frame_id + ".png"))
                    prev_xyz = dump_xyz(prev_poses[0])
                    next_xyz = dump_xyz(next_poses[0])
                    curr_xyz = dump_xyz(gt_global_poses[int(frame_id)])
                    prev_distance = np.sqrt((prev_xyz[0] - curr_xyz[0]) ** 2 + (prev_xyz[2] - curr_xyz[2]) ** 2)
                    next_distance = np.sqrt((next_xyz[0] - curr_xyz[0]) ** 2 + (next_xyz[2] - curr_xyz[2]) ** 2)
                    prev_dists.append(prev_distance)
                    next_dists.append(next_distance)
                elif self.split == 'train':
                    pos_step = 1
                    neg_step = -1
                    curr_xyz = dump_xyz(gt_global_poses[int(frame_id)])
                    # deal with prev
                    cnt = 0
                    dist = 0.
                    while True:
                        # cnt += step
                        cnt += neg_step
                        rel_frame_id = "{:06d}".format(int(frame_id) + cnt)

                        img_path = os.path.join(
                            self.root, "dataset", "sequences", sequence, "image_2", rel_frame_id + ".png")

                        if not os.path.exists(img_path):
                            break

                        # dist += paired_dists["{:06d}".format(int(rel_frame_id) + pos_step)]
                        prev_xyz = dump_xyz(gt_global_poses[int(rel_frame_id)])
                        dist = np.sqrt((prev_xyz[0] - curr_xyz[0]) ** 2 + (prev_xyz[1] - curr_xyz[1]) ** 2)
                        if dist < frames_interval:
                            continue
                        if dist > sequence_distance[0]:
                            break
                        if abs(cnt) > max_frames_interval:
                            break
                        if split == "val" and rel_frame_id in val_error_frames:
                            continue

                        prev_frame_ids.append(rel_frame_id)
                        prev_img_paths.append(img_path)
                        lidar_path = os.path.join(
                            self.root, "dataset", "sequences", sequence, "velodyne", rel_frame_id + ".bin")
                        prev_lid_paths.append(lidar_path)
                        prev_poses.append(gt_global_poses[int(rel_frame_id)])
                        prev_dists.append(dist)

                    # deal with next
                    cnt = 0
                    dist = 0.
                    while True:
                        # cnt += step
                        cnt += pos_step
                        rel_frame_id = "{:06d}".format(int(frame_id) + cnt)

                        img_path = os.path.join(
                            self.root, "dataset", "sequences", sequence, "image_2", rel_frame_id + ".png")

                        if not os.path.exists(img_path):
                            break

                        # dist += paired_dists[rel_frame_id]
                        next_xyz = dump_xyz(gt_global_poses[int(rel_frame_id)])
                        dist = np.sqrt((next_xyz[0] - curr_xyz[0]) ** 2 + (next_xyz[2] - curr_xyz[2]) ** 2)
                        if dist < frames_interval:
                            continue
                        if dist > sequence_distance[1]:
                            break
                        if abs(cnt) > max_frames_interval:
                            break
                        if split == "val" and rel_frame_id in val_error_frames:
                            continue

                        next_frame_ids.append(rel_frame_id)
                        next_img_paths.append(img_path)
                        lidar_path = os.path.join(
                            self.root, "dataset", "sequences", sequence, "velodyne", rel_frame_id + ".bin")
                        next_lid_paths.append(lidar_path)
                        next_poses.append(gt_global_poses[int(rel_frame_id)])
                        next_dists.append(dist)

                prev_next_len = len(prev_poses) + len(next_poses)
                
                if prev_next_len > max_length:
                    max_length = prev_next_len
                if prev_next_len < min_length:
                    min_length = prev_next_len

                self.frame2scan.update({str(sequence) + "_" + frame_id: len(self.scans)})

                if not self.strict:
                    prev_img_paths.append(current_img_path)
                    prev_lid_paths.append(current_lid_path)
                    next_img_paths.append(current_img_path)
                    next_lid_paths.append(current_lid_path)
                    prev_poses.append(curr_pose)
                    next_poses.append(curr_pose)
                    prev_dists.append(0.)
                    next_dists.append(0.)
                    prev_frame_ids.append(frame_id)
                    next_frame_ids.append(frame_id)

                self.scans.append(
                    {
                        "frame_id": frame_id,
                        "sequence": sequence,
                        "img_path": current_img_path,
                        "lid_path": current_lid_path,
                        'sem_path': current_sem_path,
                        "pose": gt_global_poses[int(frame_id)],

                        "prev_img_paths": prev_img_paths,
                        "prev_lid_paths": prev_lid_paths,
                        "next_img_paths": next_img_paths,
                        "next_lid_paths": next_lid_paths,

                        "T_velo_2_cam": T_velo_2_cam,
                        "P": P,
                        "T_cam0_2_cam2": T_cam0_2_cam2,
                        "T_cam2_2_cam0": T_cam2_2_cam0,

                        "prev_poses": prev_poses,
                        "next_poses": next_poses,
                        "prev_dists": prev_dists,
                        "next_dists": next_dists,
                        "prev_frame_ids": prev_frame_ids,
                        "next_frame_ids": next_frame_ids
                    }
                )
            print(sequence, min_length, max_length)

        # self.to_tensor_normalized = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        # self.to_tensor = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))

    def get_depth_from_lidar(self, lidar_path, lidar2img, image_size):
        # lidar2img: N, 4, 4
        scan = np.fromfile(lidar_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        scan[:, 3] = 1.0
        # points_hcoords = scan[scan[:, 0] > 0, :]
        points_hcoords = np.expand_dims(self.transxy @ scan.T, 0) # 1, 4, n
        img_points = np.transpose(lidar2img @ points_hcoords, (0, 2, 1)) # N, n, 4

        depth = img_points[..., 2] # N, n
        img_points = img_points[..., :2] # N, n, 2
        N, n, _ = img_points.shape
        mask = (depth < self.eval_depth) & (depth > 1e-3)  # get points with depth < max_sample_depth

        img_points = img_points / np.expand_dims(depth, axis=2)  # scale 2D points
        # img_points = (img_points.reshape(-1,2) @ self.aug_rot_render.T + self.aug_trans_render[None,:]).reshape(N,n,2)
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
    
    def prepare_img_metas(self, temporal_scans, anchor_scan, anchor_prev, anchor_next):
        img_metas = {}
        scan = temporal_scans[0]
        img_metas.update({
            'input_imgs_path': [temporal_scans[i]['img_path'] for i in range(len(temporal_scans))],
            'curr_imgs_path': [anchor_scan['img_path']],
            'prev_imgs_path': [anchor_scan['prev_img_paths'][anchor_prev]],
            'next_imgs_path': [anchor_scan['next_img_paths'][anchor_next]]
        })
        prev_to_curr_rt = []
        for i in range(1, len(temporal_scans)):
            temporal_scan = temporal_scans[i]
            transform = self.transxy @ np.linalg.inv(scan['T_velo_2_cam']) @ scan['T_cam0_2_cam2'] @ np.linalg.inv(scan['pose']) @ \
                                temporal_scan['pose'] @ temporal_scan['T_cam2_2_cam0'] @ temporal_scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
            prev_to_curr_rt.append(transform)

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
        
        img2prevImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['prev_poses'][anchor_prev]) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
            
        lidar2prevImg = intrinsic @ anchor_scan['T_cam0_2_cam2'] @ np.linalg.inv(anchor_scan['prev_poses'][anchor_prev]) @ \
                        anchor_scan['pose'] @ anchor_scan['T_cam2_2_cam0'] @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
        
        img2nextImg = intrinsic @ \
            anchor_scan['T_cam0_2_cam2'] @ \
            np.linalg.inv(anchor_scan['next_poses'][anchor_next]) @ \
            anchor_scan['pose'] @ \
            anchor_scan['T_cam2_2_cam0'] @ \
            np.linalg.inv(intrinsic)
        
        lidar2nextImg = intrinsic @ anchor_scan['T_cam0_2_cam2'] @ np.linalg.inv(anchor_scan['next_poses'][anchor_next]) @ \
                        anchor_scan['pose'] @ anchor_scan['T_cam2_2_cam0'] @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
                        
        img_metas.update({
            'lidar2img': np.expand_dims(lidar2img, axis=0),
            'img2lidar': np.expand_dims(img2lidar,axis=0),
            'img2prevImg': np.expand_dims(img2prevImg,axis=0),
            'img2nextImg': np.expand_dims(img2nextImg,axis=0),
            'token': scan['frame_id'],
            'sequence': scan['sequence'],
            'temImg2lidar': np.expand_dims(temImg2lidar,axis=0),
            'intrinsic': np.expand_dims(intrinsic,axis=0),
            'prev_to_curr_rt': prev_to_curr_rt,
            'lidar2prevImg':lidar2prevImg,
            'lidar2nextImg':lidar2nextImg,
            'prev_time': (int(anchor_scan['prev_frame_ids'][anchor_prev])-int(scan['frame_id']))*0.1,
            'next_time': (int(anchor_scan['next_frame_ids'][anchor_next])-int(scan['frame_id']))*0.1,
        })

        return img_metas

    def read_surround_imgs(self, img_paths):
        cv2.setNumThreads(0)
        if hfai:
            imgs = self.img_loader.load(img_paths)
        else:
            imgs = []
            for filename in img_paths:
                # imgs.append(imread(filename, 'unchanged').astype(np.float32))
                imgs.append(cv2.imread(filename, -1).astype(np.float32))
        imgs = [img[:self.img_H, :self.img_W, :] for img in imgs]
        return imgs
    
    def read_lidar_rays(self, scan, anchor_scan, anchor_prev, anchor_next):
        lidar_rays = []
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = scan['P'][:3, :3]
        lidar2global = scan['pose'] @ scan['T_cam2_2_cam0'] @ scan['T_velo_2_cam'] @ np.linalg.inv(self.transxy)
        for idx, temporal_scan in enumerate([anchor_scan]):
            temLidar2global = temporal_scan['pose'] @ temporal_scan['T_cam2_2_cam0'] @ temporal_scan['T_velo_2_cam']
            sensor2lidar = np.linalg.inv(lidar2global) @ temLidar2global
            lidar_scan = np.fromfile(temporal_scan['lid_path'], dtype=np.float32)
            lidar_scan = lidar_scan.reshape((-1, 4))[:, :3]

            temImg2global = anchor_scan['pose'] @ anchor_scan['T_cam2_2_cam0'] @ np.linalg.inv(intrinsic)
            temLidar2temImg = np.linalg.inv(temImg2global) @ temLidar2global
            pixel = lidar_scan @ temLidar2temImg[:3,:3].T + temLidar2temImg[:3,3][None,:]
            mask =  np.logical_and(pixel[:,2]>1.0, pixel[:,2]<self.eval_depth)
            pixel = pixel[:,:2] / pixel[:,2:3]
            pixel = (pixel @ self.aug_rot_render.T + self.aug_trans_render[None,:])
            pixel[:,0] /= self.img_W
            pixel[:,1] /= self.img_H
            scan_mask = mask & (pixel[:, 0] > 0) & (pixel[:, 0] < 1) & (pixel[:, 1] > 0) & (pixel[:, 1] < 1)

            lidar_scan = lidar_scan[scan_mask]
            depth_range = np.linalg.norm(lidar_scan, axis=1)
            rays_d = (lidar_scan / depth_range[:,None]) @ sensor2lidar[:3,:3].T
            rays_o = sensor2lidar[:3,3][None,:].repeat(rays_d.shape[0],axis=0)
            frame_id = idx *np.ones_like(depth_range)
            lidar_rays.append(np.concatenate((rays_o, rays_d, frame_id[:,None], depth_range[:,None]), axis=1))
        lidar_rays = np.vstack(lidar_rays)
        inds = np.random.permutation(lidar_rays.shape[0])[:self.sample_lidar_rays]
        return lidar_rays[inds]
    
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
            if self.split == 'train' and self.debug:
                index = 902
            scan = deepcopy(self.scans[index])
            sequence = scan['sequence']

            if temporal_supervision == 'curr':
                anchor_scan = deepcopy(scan)
            elif temporal_supervision == 'prev':
                if len(scan['prev_frame_ids']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scan_id = np.random.choice(scan['prev_frame_ids'])
                anchor_scan = deepcopy(self.scans[self.frame2scan[str(sequence) + '_' + anchor_scan_id]])
            else:
                if len(scan['next_frame_ids']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scan_id = np.random.choice(scan['next_frame_ids'])
                anchor_scan = deepcopy(self.scans[self.frame2scan[str(sequence) + '_' + anchor_scan_id]])
                            
            if len(anchor_scan['prev_frame_ids']) == 0 or len(anchor_scan['next_frame_ids']) == 0:
                index = np.random.randint(len(self))
                continue
            anchor_prev = 0 if self.choose_nearest else np.random.randint(len(anchor_scan['prev_frame_ids']))
            anchor_next = 0 if self.choose_nearest else np.random.randint(len(anchor_scan['next_frame_ids']))
            break
        
        temporal_step = 2
        temporal_scans = [scan]
        # temporal_scan = deepcopy(scan)
        # for i in range(self.queue_length):
        #     scan_temp = self.scans[max(0,index-temporal_step*i)]
        #     if scan_temp['sequence'] == scan['sequence']:
        #         temporal_scans.append(scan_temp)
        #         temporal_scan = deepcopy(scan_temp)
        #     else:
        #         temporal_scans.append(temporal_scan)
        for i in range(1, self.queue_length):
            temporal_frame_id = max(0,int(self.scans[index]['frame_id'])-temporal_step*i)
            temporal_img_path = os.path.join(os.path.dirname(scan['img_path']), f'{temporal_frame_id:06d}.png')
            temporal_pose = self.all_gt_global_poses[scan['sequence']][temporal_frame_id]
            temporal_scan = dict(frame_id=f'{temporal_frame_id:06d}', sequence=scan['sequence'], img_path=temporal_img_path, pose=temporal_pose,
                                 T_velo_2_cam=scan['T_velo_2_cam'], T_cam2_2_cam0=scan['T_cam2_2_cam0'])
            temporal_scans.append(temporal_scan)
            
        img_metas = self.prepare_img_metas(
            temporal_scans, anchor_scan, anchor_prev, anchor_next)
        # read 6 cams
        input_imgs = self.read_surround_imgs(img_metas['input_imgs_path'])
        curr_imgs = self.read_surround_imgs(img_metas['curr_imgs_path'])
        prev_imgs = self.read_surround_imgs(img_metas['prev_imgs_path'])
        next_imgs = self.read_surround_imgs(img_metas['next_imgs_path'])
        origin_size = [curr_imgs[0].shape[0], curr_imgs[0].shape[1]]
        self.aug_rot_render = np.eye(2)* self.resize_scale
        self.aug_trans_render = -np.array([int(max(0, origin_size[1]*self.resize_scale-self.crop_size[1]) / 2), int(origin_size[0]*self.resize_scale)-self.crop_size[0]])
        self.aug_rot = np.eye(2)* self.input_resize_scale
        self.aug_trans = -np.array([int(max(0, origin_size[1]*self.input_resize_scale-self.input_img_crop_size[1])/2), int(origin_size[0]*self.input_resize_scale)-self.input_img_crop_size[0]])
        
        if self.return_depth:
            depth_loc, depth_gt, depth_mask = self.get_depth_from_lidar(
                scan['lid_path'], img_metas['lidar2img'], [self.img_H, self.img_W])
            img_metas.update({
                'depth_loc': depth_loc,
                'depth_gt': depth_gt,
                'depth_mask': depth_mask})
        if self.return_sem:
            # sem = self.load_2d_sem_label(anchor_scan)
            sem = self.read_surround_imgs([anchor_scan['sem_path']])
            img_metas.update({'sem': sem})

        lidar_rays = None
        if self.return_lidar_rays:
            lidar_rays = self.read_lidar_rays(scan, anchor_scan, anchor_prev, anchor_next)

        data_tuple = ([input_imgs, curr_imgs, prev_imgs, next_imgs, lidar_rays], img_metas)
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