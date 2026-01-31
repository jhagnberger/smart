#!/bin/bash
#SBATCH --partition=dgx,gpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --job-name=smart-shift-wing

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate smart

# Change to smart directory
cd ../smart

# Read random seed from command line argument or use default
if [ -z "$1" ]; then
    SEED=42
else
    SEED=$1
fi

python3 train.py --config-name=wing ++experiment.random_seed=$SEED
python3 inference.py --config-name=wing ++experiment.random_seed=$SEED