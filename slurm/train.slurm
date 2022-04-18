#!/bin/bash
#SBATCH --job-name=syac-train
#SBATCH --account=ec30
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=30G
#SBATCH --partition=accel
#SBATCH --gpus=1

module purge

nvidia-smi

module load CUDA/11.3.1
module load Anaconda3/5.3.0
source activate /fp/homes01/u01/ec-markuhei/.conda/envs/SYAC-ML
~/.conda/envs/SYAC-ML/bin/python --version
~/.conda/envs/SYAC-ML/bin/python train.py
