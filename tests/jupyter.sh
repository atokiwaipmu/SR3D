#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/jupyter.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D/log/juputer.err  
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=4

source /home/akira.tokiwa/.bashrc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5 # where X is the GPU id of an available GPU

# activate python environment
conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SR3D
jupyter notebook --no-browser --port=8883
