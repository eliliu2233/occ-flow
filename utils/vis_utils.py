import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image

semantic_colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

def get_cmap_colors(bins):
    if bins < 25:
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap('tab20')(np.linspace(0, 1, bins)))
    else:
        cmap = matplotlib.colors.ListedColormap(np.vstack((
            plt.get_cmap('tab20')(np.linspace(0, 1, int(bins*0.6))),
            plt.get_cmap('Set3')(np.linspace(0, 1, bins-int(bins*0.6)))
        )))
    return cmap

def save_surroundimg(imgs, save_path, cam_types, rgb=False):
    pad_len = int(imgs[0].shape[1] * 0.02)
    channels = imgs[0].shape[-1]
    
    if len(imgs) == 6:
        v_padding = 255.0 * np.ones([imgs[0].shape[0], pad_len, 3]) if len(imgs[0].shape) == 3 else 255.0 * np.ones([imgs[0].shape[0], pad_len])
        
        image_left_front_right = np.concatenate(
            (imgs[cam_types.index('CAM_FRONT_LEFT')], v_padding, imgs[cam_types.index('CAM_FRONT')], v_padding, imgs[cam_types.index('CAM_FRONT_RIGHT')]), axis=1)
            
        image_left_rear_right = np.concatenate(
            (imgs[cam_types.index('CAM_BACK_LEFT')], v_padding, imgs[cam_types.index('CAM_BACK')], v_padding, imgs[cam_types.index('CAM_BACK_RIGHT')]), axis=1)
        h_padding = 255.0 * np.ones([pad_len, image_left_front_right.shape[1], 3]) if len(imgs[0].shape) == 3 else 255.0 * np.ones([pad_len, image_left_front_right.shape[1]])
        surround_view = np.concatenate((image_left_front_right, h_padding, image_left_rear_right), axis=0)
    elif len(imgs) == 1:
        surround_view = imgs[0]
    else:
        raise ValueError(f"Invalid number of images: {len(imgs)}")
    
    if not rgb:
        cv2.imwrite(save_path, surround_view)
    else:
        Image.fromarray(surround_view.astype(np.uint8)).save(save_path)
