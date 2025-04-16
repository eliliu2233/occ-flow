import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("./")
# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time
import copy
import math
import gzip
import pickle
import argparse
import shutil
import numpy as np
import cv2
import torch
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset.kitti.io_data as SemanticKittiIO
KITTI_ROOT = 'data/kitti'

VIZ = True
dvr = load("dvr", sources=["utils/ray_iou_geo/lib/dvr/dvr.cpp", "utils/ray_iou_geo/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
_pc_range = [-25.6, 0.0, -2.0, 25.6, 51.2, 4.4]
_voxel_size = 0.2

def read_semantic_kitti(sequence, frame_id):
    label_path = os.path.join(
        KITTI_ROOT, "dataset/sequences", sequence, "voxels", "{}.label".format(frame_id))
    invalid_path = os.path.join(
        KITTI_ROOT, "dataset/sequences", sequence, "voxels", "{}.invalid".format(frame_id))

    remap_lut = SemanticKittiIO.get_remap_lut("dataset/kitti/semantic-kitti.yaml")
    LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
    INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
    LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
        np.float32
    )  # Remap 20 classes semanticKITTI SSC
    # import pdb; pdb.set_trace()
    LABEL[
        np.isclose(INVALID, 1)
    ] = 255  # Setting to unknown all voxels marked on invalid mask...
    
    LABEL = LABEL.reshape(256, 256, 32)
    return LABEL

# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []
    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size

    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def process_one_sample(sem_pred, lidar_rays, output_origin, return_xyz=False, sem_type='occ', sem_gt=None):
    T = output_origin.shape[1]
    pred_pcds_t = []
    if sem_type == 'sem':
        free_id = 0
        occ_pred = copy.deepcopy(sem_pred)
        occ_pred[sem_pred != free_id] = 1
        occ_pred[sem_pred == free_id] = 0
        # sem_pred = copy.deepcopy(occ_pred)
    elif sem_type == 'occ':
        occ_pred = copy.deepcopy(sem_pred)
        sem_pred = copy.deepcopy(sem_gt)
    occ_pred = occ_pred.permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
    
    for t in range(T): 
        lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, _, coord_index =  dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, 32, 256, 256],
                "test"
            )
            pred_dist *= _voxel_size

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist[0].cpu().numpy()
        )
        coord_index = coord_index[0, :, :].long().cpu()  # [N, 3]

        pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
        pred_dist = pred_dist[0, :, None].cpu()

        if return_xyz:
            pred_pcds = torch.cat([pred_label, pred_dist, pred_pcds[0]], dim=-1)  # [N, 5]  5: [label, dist, x, y, z]
        else:
            pred_pcds = torch.cat([pred_label, pred_dist], dim=-1)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)

    return pred_pcds_t.numpy()


def main(args):
    if os.path.exists(os.path.join(args.output_dir, 'code')):
        shutil.rmtree(os.path.join(args.output_dir, 'code'))
    os.makedirs(os.path.join(args.output_dir, 'code'), exist_ok=True)
    shutil.copytree('utils/ray_iou_geo', os.path.join(args.output_dir, 'code'+'/ray_iou_geo'))
    occ_save_path = args.pred_occ_path
    # pred_paths = sorted(glob.glob(f'{occ_save_path}/*.npz'))
    # eval_frames = [pred_path.split('/')[-1][:-4] for pred_path in pred_paths]

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)
    from dataset.kitti.kitti_pose_dataset import Kitti_EGO_Dataset
    ego_pose_dataset = Kitti_EGO_Dataset(root='data/kitti', split='val', splits={'val':['08']}, dataset_type='odometry') #, selected_frames=eval_frames

    data_pkl_gt = {}
    data_pkl_pred = {}

    for sample_token, output_origin in tqdm(ego_pose_dataset):
        scene_name = sample_token.split('_')[0]
        frame_id = sample_token.split('_')[1]
        pred_path = os.path.join(occ_save_path, scene_name, frame_id+'.npz')
        if not os.path.exists(pred_path):
            continue
        
        output_origin = output_origin.to(torch.float32).unsqueeze(0)

        pred_data = np.load(pred_path)

        sem_pred = pred_data['occ']
        # sem_pred = pred_data
        # sem_pred = np.flip(pred_data, axis=1).copy().astype(np.float32)
        sem_pred = np.reshape(sem_pred, [256, 256, 32])
        sem_pred = torch.from_numpy(sem_pred).permute(1,0,2) # HWZ-->WHZ

        gt_data = read_semantic_kitti(scene_name, frame_id) 
        sem_gt = torch.flip(torch.from_numpy(gt_data), [1]).permute(1,0,2) # HWZ-->WHZ
        sem_gt[..., 28:] = 0
        sem_gt[:,-6:, ...] = 0
        sem_gt[:6, :, :] = 0
        sem_gt[-6:, :, :] = 0
        pcd_gt = process_one_sample(sem_gt, lidar_rays, output_origin, return_xyz=VIZ, sem_type='sem')
        pcd_pred = process_one_sample(sem_pred, lidar_rays, output_origin, return_xyz=VIZ, sem_gt=sem_gt, sem_type='occ')


        data_pkl_gt[sample_token] = {
            'pcd_cls': pcd_gt[:, 0].astype(np.uint8),
            'pcd_dist': pcd_gt[:, 1].astype(np.float16),
            # 'pcd_points': pcd_gt[:, 2:].astype(np.float16)
        }

        data_pkl_pred[sample_token] = {
            'pcd_cls': pcd_pred[:, 0].astype(np.uint8),
            'pcd_dist': pcd_pred[:, 1].astype(np.float16),
            # 'pcd_points': pcd_pred[:, 2:].astype(np.float16)
        }

    submission_pkl_gt = {
        'method': 'GT',
        'results': data_pkl_gt
    }

    submission_pkl_pred = {
        'method': 'My prediction',
        'results': data_pkl_pred
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path_gt = os.path.join(args.output_dir, os.path.basename(args.data_info).split('.')[0] + '_pcd.gz')
    output_path_pred = os.path.join(args.output_dir, 'my_pred_pcd.gz')
    print("gzip and dumping the data...")

    start = time.time()
    with gzip.GzipFile(output_path_gt, 'wb', compresslevel=9) as f:
        pickle.dump(submission_pkl_gt, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.GzipFile(output_path_pred, 'wb', compresslevel=9) as f:
        pickle.dump(submission_pkl_pred, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-occ-path", default='out/visualization/kitti/occ_odom_static/occupancy')
    parser.add_argument("--data-info", default='data/nuscenes_infos_val_occ.pkl')
    parser.add_argument("--output-dir", default='ray_iou_output/kitti_odom')
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
