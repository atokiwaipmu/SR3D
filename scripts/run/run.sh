#!/bin/bash
#SBATCH --job-name=3d_diffusion
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/SR3D.out
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/SR3D.err
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5 # where X is the GPU id of an available GPU

# activate python environment
source ~/.bashrc
conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D
python -m scripts.run.main --use_attn True --batch_size 4