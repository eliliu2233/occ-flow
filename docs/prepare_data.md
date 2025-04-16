<!-- template from bevformer -->

## NuScenes

**a. Download nuScenes data.**

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

**b. Download occupancy and occupancy flow annotations.**

Download the openocc_v2.1.zip and infos.zip from OccNet [HERE](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq) and unzip it.

**c. Download 2D semantic labels and 2D flow labels.**

Download the generated 2D semantic labels from [semantic_labels](https://cloud.tsinghua.edu.cn/d/564c9ac19b5f4b54a774/?p=%2Fsemantic_labels&mode=list) and extract the data to ./data/nuscenes/.

Download the generated 2D flow labels from [flow_labels](https://drive.google.com/file/d/1eEof41urMiRb73qvZ5Io-mQMZtBfCL3G/view?usp=sharing) and extract the data to ./data/nuscenes.

(Optional) If you want to extract 2d flow labels with CotrackerV2 by yourself, you can run the following python file:

```shell
python precompute_tracking.py
```

**d. Download nuScenes pkl files.**

Download the nuscenes sweeps pkl files. [HERE](https://drive.google.com/drive/folders/1wcNTaaOgw5TzlfeaQcVJ276joe1_D_SJ?usp=sharing)

**Folder structure**
```
occ-flow
├── ...
├── data/
│   ├── nuscenes_sweeps_infos_train_anns.pkl
│   ├── nuscenes_sweeps_infos_train_anns.pkl
│   ├── nuscenes_infos_val_occ.pkl
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── openocc_v2/
|   |   ├── nuscenes_semantic/
|   |   ├── nuscenes_flow/
```

## SemanticKITTI

We follow similar instructions as [SceneRF](https://github.com/astra-vision/SceneRF) to prepare SemanticKITTI.

**a. Download calib, rgb, pose and lidar files.**

To train and evaluate novel depths synthesis, please download on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) the following data:

    - Odometry data set (calibration files, 1 MB)
    - Odometry data set (color, 65 GB)
    - Odometry ground truth poses (4 MB)
    - Velodyne laser data, 80 GB

**b. Download occupancy annotations.**

To evaluate 3D occupancy prediction, please download **the SemanticKITTI voxel data (700 MB)** on [Semantic KITTI download website](http://www.semantic-kitti.org/dataset.html).

**c. Create preprocess folder.**

Create an empty folder to store preprocess data at `data/kitti/preprocess`.

**Folder structure**
```
occ-flow
├── ...
├── data/
│   ├── kitti/
│   │   ├── dataset/
|   |   |   ├── poses/
|   |   |   |   ├── 00.txt
|   |   |   |   ├── ...
|   |   |   ├── sequences/
|   |   |   |   ├── 00/
|   |   |   |   |   ├── image_2/
|   |   |   |   |   ├── image_3/
|   |   |   |   |   ├── labels/
|   |   |   |   |   ├── velodyne/
|   |   |   |   |   ├── voxels/
|   |   |   |   |   ├── calib.txt
|   |   |   |   |   ├── poses.txt
|   |   |   |   |   ├── times.txt
|   |   |   |   ├── ...
```