# Data Preparation

All paths are relative to the repository root. Datasets go in `data/`.

- [nuScenes](#nuscenes)
- [SemanticKITTI](#semantickitti)

## nuScenes

| Data | Source | Location |
| :-- | :-- | :-- |
| nuScenes v1.0 (full) | [nuScenes](https://www.nuscenes.org/download) | `data/nuscenes/` |
| Occupancy and flow annotations (`openocc_v2.1.zip`) | [OccNet](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq) | `data/nuscenes/openocc_v2/` |
| Validation info file (`nuscenes_infos_val_occ.pkl`, from `infos.zip`) | [OccNet](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq) | `data/` |
| Sweeps info files | [Google Drive](https://drive.google.com/drive/folders/1wcNTaaOgw5TzlfeaQcVJ276joe1_D_SJ?usp=sharing) | `data/` |
| 2D semantic labels | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/564c9ac19b5f4b54a774/?p=%2Fsemantic_labels&mode=list) | `data/nuscenes/nuscenes_semantic/` |
| 2D flow labels | [Google Drive](https://drive.google.com/file/d/1eEof41urMiRb73qvZ5Io-mQMZtBfCL3G/view?usp=sharing) | `data/nuscenes/nuscenes_flow/` |

The 2D flow labels were generated with CoTracker v2. To regenerate them, run:

```shell
python precompute_tracking.py --py-config config/nuscenes/nuscenes_tracking.py
```

Expected structure:

```
data/
├── nuscenes_sweeps_infos_train_anns.pkl
├── nuscenes_sweeps_infos_val_anns.pkl
├── nuscenes_infos_val_occ.pkl
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-test/
    ├── v1.0-trainval/
    ├── openocc_v2/
    ├── nuscenes_semantic/
    └── nuscenes_flow/
```

## SemanticKITTI

The setup follows [SceneRF](https://github.com/astra-vision/SceneRF).

| Data | Source |
| :-- | :-- |
| Calibration files (1 MB) | [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) |
| Color images (65 GB) | [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) |
| Ground-truth poses (4 MB) | [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) |
| Velodyne point clouds (80 GB) | [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) |
| Voxel annotations (700 MB) | [SemanticKITTI](http://www.semantic-kitti.org/dataset.html) |

Expected structure. The dataset config points to `preprocess/`, so create it as an empty folder. Evaluation uses sequence `08`.

```
data/
└── kitti/
    ├── preprocess/
    └── dataset/
        ├── poses/
        │   ├── 00.txt
        │   └── ...
        └── sequences/
            ├── 00/
            │   ├── image_2/
            │   ├── image_3/
            │   ├── labels/
            │   ├── velodyne/
            │   ├── voxels/
            │   ├── calib.txt
            │   ├── poses.txt
            │   └── times.txt
            └── ...
```
