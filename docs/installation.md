# Installation

The code has been tested with Python 3.8, PyTorch 1.12.1 and CUDA 11.3 on NVIDIA A100 and A6000 GPUs. You also need a CUDA toolkit that matches your PyTorch build: the RayIoU evaluation compiles a custom CUDA extension.

## 1. Environment

```shell
conda create -n occflow python=3.8.16 -y
conda activate occflow
```

## 2. PyTorch

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Other builds of `torch>=1.12.1` should also work. See the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

## 3. OpenMMLab

```shell
pip install -U openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
pip install mmsegmentation==1.0.0
mim install mmdet3d==1.1.1
```

## 4. sdfstudio

The SDF rendering module is built on [sdfstudio](https://github.com/autonomousvision/sdfstudio). Install our modified version:

```shell
git clone https://github.com/eliliu2233/sdfstudio_occ.git
pip install --upgrade pip setuptools
pip install -e sdfstudio_occ
```

## 5. Additional dependencies

```shell
pip install pillow==8.4.0 typing_extensions==4.8.0 torchmetrics==0.9.3 timm==0.9.2
```

## 6. Let Occ Flow

```shell
git clone https://github.com/eliliu2233/occ-flow.git
```

Next, [prepare the datasets](prepare_data.md) and download the checkpoints listed in the [Model Zoo](../README.md#model-zoo).
