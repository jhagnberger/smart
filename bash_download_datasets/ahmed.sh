#!/bin/bash
# Code based on the script provided by N. Ashton at https://huggingface.co/datasets/neashton/ahmedml
# Licensed under the Creative Commons Attribution-ShareAlike 4.0 International License
# For more details, see https://creativecommons.org/licenses/by-sa/4.0/
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --job-name=download-ahmedml

# Set the paths
HF_OWNER="neashton"
HF_PREFIX="ahmedml"

# Get local dir from command line argument or use default
if [ -z "$1" ]; then
    LOCAL_DIR="./ahmedml"
else
    LOCAL_DIR=$1
fi

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Loop through the run folders from 1 to 500
for i in $(seq 1 500); do
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"

    # Create the run directory if it doesn't exist
    mkdir -p "$RUN_LOCAL_DIR"

    # Print the current run directory being processed
    echo "Downloading files for $RUN_DIR..."

    # Download the ahmed_i.stl file
    wget -nv "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/ahmed_$i.stl" -O "$RUN_LOCAL_DIR/ahmed_$i.stl"

    # Download the boundary_i.vtp file
    wget -nv "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/boundary_$i.vtp" -O "$RUN_LOCAL_DIR/boundary_$i.vtp"

    # Download the volume_i.vtu file
    wget -nv "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/volume_$i.vtu" -O "$RUN_LOCAL_DIR/volume_$i.vtu"
done