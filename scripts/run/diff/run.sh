#!/bin/bash
#SBATCH --job-name=3d_diffusion
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/SR3D_%j.out
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/SR3D_%j.err
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-gpu=6

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1 # where X is the GPU id of an available GPU

# activate python environment
source ~/.bashrc
conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D
python -m scripts.run.main --n_maps 2000 --mask True --use_attn True --batch_size 2