#!/bin/bash
# Code based on the script provided by N. Ashton at https://huggingface.co/datasets/neashton/ahmedml
# Licensed under the Creative Commons Attribution-ShareAlike 4.0 International License
# For more details, see https://creativecommons.org/licenses/by-sa/4.0/
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --job-name=download-shift-suv

# Set the paths
HF_OWNER="luminary-shift"
HF_PREFIX="SUV"
MODEL_SCALE="full" # options: full, qtr
REAR_GEOMETRY_TYPE="estate" # options: estate, fastback

# Your HF access token
HF_ACCESS_TOKEN="hf_{YOUR_ACCESS_TOKEN}"

# Get local dir from command line argument or use default
if [ -z "$1" ]; then
    LOCAL_DIR="./shift-suv"
else
    LOCAL_DIR=$1
fi

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Loop through the run folders from 1 to 999
for i in $(seq 1 999); do
    RUN_DIR=$(printf "run_%05d" "$i")
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"

    if wget -nv --spider --quiet --header="Authorization: Bearer ${HF_ACCESS_TOKEN}" "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/AeroSUV_${MODEL_SCALE}_scale_${REAR_GEOMETRY_TYPE}_transient/$RUN_DIR/merged_surfaces.stl"; then
        # Create the run directory if it doesn't exist
        mkdir -p "$RUN_LOCAL_DIR"

        # Print the current run directory being processed
        echo "Downloading files for $RUN_DIR..."

        # Download the merged_surfaces.stl file
        wget -nv --header="Authorization: Bearer ${HF_ACCESS_TOKEN}" "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/AeroSUV_${MODEL_SCALE}_scale_${REAR_GEOMETRY_TYPE}_transient/$RUN_DIR/merged_surfaces.stl" -O "$RUN_LOCAL_DIR/merged_surfaces.stl"

        # Download the merged_surfaces.vtp file
        wget -nv --header="Authorization: Bearer ${HF_ACCESS_TOKEN}" "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/AeroSUV_${MODEL_SCALE}_scale_${REAR_GEOMETRY_TYPE}_transient/$RUN_DIR/merged_surfaces.vtp" -O "$RUN_LOCAL_DIR/merged_surfaces.vtp"

        # Download the merged_volumes.vtu file
        wget -nv --header="Authorization: Bearer ${HF_ACCESS_TOKEN}" "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/AeroSUV_${MODEL_SCALE}_scale_${REAR_GEOMETRY_TYPE}_transient/$RUN_DIR/merged_volumes.vtu" -O "$RUN_LOCAL_DIR/merged_volumes.vtu"
    else
        echo "Files for $RUN_DIR do not exist."
    fi
done