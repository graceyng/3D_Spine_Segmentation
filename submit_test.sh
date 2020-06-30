#! /bin/bash

CUDA_VISIBLE_DEVICES='get_CUDA_VISIBLE_DEVICES' || exit
export CUDA_VISIBLE_DEVICES
cd /cbica/projects/deepspine/code/3D_Spine_Segmentation/

nvidia-smi

module load python/anaconda/3.5.6+tfgpu+pydicom+opencv

python script_submit_test.py

