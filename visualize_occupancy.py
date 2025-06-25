import os
os.environ["DISPLAY"] = "localhost:12.0"
# os.environ['QT_QPA_PLATFORM'] ="offscreen"
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# import mmcv
import pickle
import mayavi.mlab as mlab
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pyquaternion import Quaternion
mlab.options.offscreen = True

voxel_size = 0.4 # 0.3125
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

with open('data/nuscenes_infos_val_occ.pkl', 'rb') as f:
    data_info = pickle.load(f)['infos'][0]
# transform from lidar to ego coordinate
lidar2ego_r = Quaternion(data_info['lidar2ego_rotation']).rotation_matrix
lidar2ego_t = np.array(data_info['lidar2ego_translation'])
cam_infos = data_info['cams']
color_bins = 25
delta_c = abs(color_bins-1) / (255 * 2)
colors = np.zeros([color_bins, 4],dtype=np.uint8)
for i in range(color_bins):
    color_n = int(i / delta_c)
    if color_n <= 255:
        colors[i, :] = [0, 255 - color_n, 255, 255]
    else:
        colors[i, :] = [color_n - 255, 0, 255, 255]
vis_flow = 0
cam_types = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
visual_folder = 'out/visualization/nuscenes/occ_flow/occupancy/scene-0797/' # scene-0558

def worker(filename):
    # visual_path = os.path.join(visual_folder, filename)
    visual_path = filename
    if not os.path.exists(visual_path):
        return
    folder = visual_path.split('.')[0]
    os.makedirs(folder, exist_ok=True)
    data = np.load(visual_path, allow_pickle=True)
    mask = data['occ'] == 1
    occ_size = [200, 200, 16]
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
    z_min, z_max = metric_z.min(), min(metric_z.max(), 3.0)
    metric = np.clip((metric_z-z_min)/(z_max-z_min), 0.0, 1.0)*(color_bins-1)
    if vis_flow:
        voxel_flow = data['flow'][mask]
        max_move = np.argmax(np.abs(voxel_flow), axis=1)
        voxel_flow = voxel_flow[np.arange(fov_voxels.shape[0]), max_move]
        metric_f = voxel_flow*2.0
        f_min, f_max = -1.2, 1.2
        metric = np.clip((metric_f-f_min)/(f_max-f_min), 0.0, 1.0)*(color_bins-1)
    if fov_voxels.shape[-1]==3:
        fov_voxels = np.concatenate([fov_voxels, np.ones_like(fov_voxels[:,0:1])], axis=1)
        
    for cam in cam_types:
        cam2lidar_r = cam_infos[cam]['sensor2lidar_rotation']
        cam2lidar_t = cam_infos[cam]['sensor2lidar_translation']
        cam_position = (lidar2ego_r @ cam2lidar_t.reshape(-1,1)).reshape(-1) + lidar2ego_t
        f = 0.0055  
        focal_position = (lidar2ego_r @ ((cam2lidar_r @ np.array([0.,0.,f]).reshape(-1,1)).reshape(-1) + cam2lidar_t).reshape(-1,1)).reshape(-1) + lidar2ego_t
        
        render_w = 352 #300
        figure = mlab.figure(size=(render_w, render_w/1408*512), bgcolor=(1, 1, 1))
        # pdb.set_trace()
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            metric,
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
        scene.camera.view_angle = 35 if cam != 'CAM_BACK' else 60
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.]
        save_file = os.path.join(folder, f'./OCC-{cam}.jpg') if not vis_flow else os.path.join(folder, f'./FLOW-{cam}.jpg')
        scene.camera.compute_view_plane_normal()
        scene.render()
        mlab.savefig(save_file)
        mlab.close()
        
if __name__ == '__main__':
    import glob
    po = Pool(10)
    infos = glob.glob(visual_folder+'*.npz')
    pbar = tqdm(total=len(infos))
    pbar.set_description('export occupancy flow')
    update = lambda *args: pbar.update()
    for info in infos:
        po.apply_async(func=worker, args=(info, ), callback=update)
    po.close()
    po.join()