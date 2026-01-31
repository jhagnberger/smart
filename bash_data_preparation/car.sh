#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --job-name=prepare-shapenetcar

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate smart

# Change to smart directory
cd ../smart

python3 prepare.py --config-name=car