#!/bin/bash

# Assume using the Docker image: nvcr.io/nvidia/pytorch:23.10-py3

pip install -r requirements.txt

# Pytorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"

# PointNet++ (setup.py is changed according to: https://github.com/erikwijmans/Pointnet2_PyTorch/pull/177 )
pip install pointnet2_ops_lib/

# For Point-MAE
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install extensions/chamfer_dist/
