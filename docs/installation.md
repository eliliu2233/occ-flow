<!-- template from bevformer -->
# Step-by-step installation instructions

The following configuration of conda environment is tested on Nvidia A100 and Nvidia A6000.


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n occflow python=3.8.16 -y
conda activate occflow
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Recommended torch>=1.12.1
```

**c. Install mmcv, mmdet, mmseg, mmdet3d.**
```shell
pip install -U openmim  # Following https://mmcv.readthedocs.io/en/latest/get_started/installation.html
mim install mmcv==2.0.1
mim install mmdet==3.0.0 # Following https://mmdetection.readthedocs.io/en/latest/get_started.html
pip install "mmsegmentation==1.0.0" # Following https://mmsegmentation.readthedocs.io/en/latest/get_started.html
mim install "mmdet3d==1.1.1" # Following https://mmdetection3d.readthedocs.io/en/latest/get_started.html
```

**d. Install sdfstudio.**

We develop the SDF rendering part of our code based on [sdfstudio](https://github.com/autonomousvision/sdfstudio).
```shell
git clone https://github.com/eliliu2233/sdfstudio_occ.git
cd sdfstudio_occ
pip install --upgrade pip setuptools
pip install -e .
```

**e. Install other packages and deal with package versions.**
```shell
pip install pillow==8.4.0 typing_extensions==4.8.0 torchmetrics==0.9.3 timm==0.9.2
```

**f. Clone Let Occ Flow.**
```shell
git clone https://github.com/eliliu2233/occ-flow.git
```

**f. Prepare pretrained models.**
```shell
cd occ-flow
mkdir ckpts
```
+ Download convnext-base pretrained backbone. [HERE](https://drive.google.com/file/d/1ZjWbqI1qoBcqeQijI5xX9E-YNkxpJcYV/view)
+ (Optional) Download unimatch pretrained model. [HERE](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth)
+ Download pretrained model weights. [HERE](https://drive.google.com/drive/folders/1a9g0tpu57MR7WggvVwq6AI8zfXXfc6ln?usp=sharing)
