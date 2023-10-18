#!/bin/bash
#SBATCH --job-name=3d_diffusion
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/%j.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/%j.err  
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2 # where X is the GPU id of an available GPU

# activate python environment
source ~/.bashrc
conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github
python -m SR3D.scripts.main