#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --job-name=download-shapenetcar

# Get local dir from command line argument or use default
if [ -z "$1" ]; then
    LOCAL_DIR="./shapenetcar"
else
    LOCAL_DIR=$1
fi

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download the mlcfd_data.zip file
wget -nv "http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip" -O "$LOCAL_DIR/mlcfd_data.zip"

# Unzip the downloaded file
unzip -q "$LOCAL_DIR/mlcfd_data.zip" -d "$LOCAL_DIR"
