import os
# os.environ["DISPLAY"] = "10.166.204.232:5.0"
os.environ["DISPLAY"] = "localhost:12.0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ['QT_QPA_PLATFORM']
# ="offscreen"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from multiprocessing import Pool
from tqdm import tqdm
# import mmcv
import torch
import pickle
import mayavi.mlab as mlab
import numpy as np
from dataset.kitti.helpers import read_calib
mlab.options.offscreen = True

voxel_size = 0.2 # 0.3125
pc_range = [-25.6, 0.0, -2.0, 25.6, 51.2, 4.4]
sequence = '08'
dataset_type = 'odometry'
if dataset_type == 'kitti360':
    data_root = 'data/KITTI_360/'
    from dataset.kitti.helpers import get_kitti360_calib
    calib_path = os.path.join(data_root, 'calibration')
    sequence_folder = f'2013_05_28_drive_{sequence}_sync'
    calib = get_kitti360_calib(calib_path)
elif dataset_type == 'odometry':
    data_root = 'data/kitti/'
    calib = read_calib(os.path.join(data_root, "dataset", "sequences", sequence, "calib.txt"))

P = calib["P2"]
trans_xy = np.array([[0, -1., 0, 0], [1., 0, 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]])
T_cam0_2_cam2 = calib['T_cam0_2_cam2']
T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)
T_velo_2_cam = T_cam0_2_cam2 @ calib["Tr"]
lidar2ego_r = np.eye(3)
lidar2ego_t = np.zeros(3)
cam2lidar = trans_xy @ np.linalg.inv(T_velo_2_cam)
cam_types = ['CAM_FRONT']
color_bins = 25
delta_c = abs(color_bins-1) / (255 * 2)
colors = np.zeros([color_bins, 4],dtype=np.uint8)
for i in range(color_bins):
    color_n = int(i / delta_c)
    if color_n <= 255:
        colors[i, :] = [0, 255 - color_n, 255, 255]
    else:
        colors[i, :] = [color_n - 255, 0, 255, 255]
novel = 0
visual_folder = f'out/visualization/kitti/occ_odom_static/occupancy/{sequence}'

def worker(i):

    visual_path = os.path.join(visual_folder, f'{i:06d}.npz')
    if not os.path.exists(visual_path):
        return
    folder = visual_path.split('.')[0]
    os.makedirs(folder, exist_ok=True)

    data = np.load(visual_path, allow_pickle=True)
    mask = data['occ'] == 1
    occ_size = [256, 256, 32]
    x = torch.linspace(0, occ_size[0]-1, occ_size[0])
    y = torch.linspace(0, occ_size[1]-1, occ_size[1])
    z = torch.linspace(0, occ_size[2]-1, occ_size[2])
    Y, X, Z = torch.meshgrid(y, x, z)
    vv = torch.stack([X, Y, Z], dim=-1)
    vertices = vv.clone()
    vertices[..., 0] = (vertices[..., 0] + 0.5) * voxel_size + pc_range[0]
    vertices[..., 1] = (vertices[..., 1] + 0.5) * voxel_size + pc_range[1]
    vertices[..., 2] = (vertices[..., 2] + 0.5) * voxel_size + pc_range[2]
    vertices = vertices.cpu().numpy()
    fov_voxels = vertices[mask]
    
    metric_z = fov_voxels[:,2]
    z_min, z_max = metric_z.min(), min(metric_z.max(), 2.0)
    metric_z = np.clip((metric_z-z_min)/(z_max-z_min), 0.0, 1.0)*(color_bins-1)
    if fov_voxels.shape[-1]==3:
        fov_voxels = np.concatenate([fov_voxels, np.ones_like(fov_voxels[:,0:1])], axis=1)
    
    for cam in cam_types:
        cam2lidar_r = cam2lidar[:3,:3]
        # cam2lidar_t = cam2lidar[:3,3] + np.array([0.0, 1.5, 5.5]) if novel else cam2lidar[:3,3]
        cam2lidar_t = cam2lidar[:3,3] + np.array([2.0, 0.8, 1.5]) if novel else cam2lidar[:3,3]
        cam_position = (lidar2ego_r @ cam2lidar_t.reshape(-1,1)).reshape(-1) + lidar2ego_t
        f = 0.0055  
        focal_position = (lidar2ego_r @ ((cam2lidar_r @ np.array([0.,0.,f]).reshape(-1,1)).reshape(-1) + cam2lidar_t).reshape(-1,1)).reshape(-1) + lidar2ego_t
        
        render_w = 406
        figure = mlab.figure(size=(render_w, render_w/1408*512), bgcolor=(1, 1, 1))
        # pdb.set_trace()
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            metric_z,
            colormap="viridis",
            scale_factor=voxel_size - 0.05*voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=color_bins-1,
        )

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

        scene = figure.scene
        scene.camera.position = cam_position
        scene.camera.focal_point = focal_position
        scene.camera.view_angle = 35 if cam != 'CAM_BACK' else 55
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.]
        filename = os.path.join(folder, f'./OCC-{cam}.jpg')
        if novel:
            scene.camera.pitch(-18)
            scene.camera.yaw(9)
            # scene.camera.pitch(-18)
            filename = os.path.join(folder, f'./OCC.jpg')
        scene.camera.compute_view_plane_normal()
        scene.render()
        # print(f'saving occupancy image {cam}.....')
        mlab.savefig(filename)
        mlab.close()
        
if __name__ == '__main__':
    po = Pool(10)
    infos = list(range(65,126,10)) #[85]
    pbar = tqdm(total=len(infos))
    pbar.set_description('export occupancy flow')
    update = lambda *args: pbar.update()
    for info in infos:
        po.apply_async(func=worker, args=(info, ), callback=update)
    po.close()
    po.join()