#!/bin/bash

#SBATCH --job-name=upload_sim_tar
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate drivoR

export HOME="/fs/nexus-projects/sim2real/aliu"
export HF_HOME="/nfshomes/aliu1237/.cache/huggingface"

DATA_ROOT="$HOME/carla_garage_data_navsim_converted"
TAR_DIR="$HOME/sim_data_tars"
REPO="AL314/m2r_sim_data"

# mkdir -p "$TAR_DIR"

# echo "[$(date)] Tarring openscene_meta_datas (1.2GB)..."
# tar -cf "$TAR_DIR/openscene_meta_datas.tar" -C "$DATA_ROOT" openscene_meta_datas/
# echo "[$(date)] Done."

# echo "[$(date)] Tarring synthetic_scene_pickles (tiny)..."
# tar -cf "$TAR_DIR/synthetic_scene_pickles.tar" -C "$DATA_ROOT" synthetic_scene_pickles/
# echo "[$(date)] Done."

# echo "[$(date)] Tarring sensor_blobs (191GB, no compression — will take ~30-60min)..."
# tar -cf "$TAR_DIR/sensor_blobs.tar" -C "$DATA_ROOT" sensor_blobs/
# echo "[$(date)] Done tarring."

echo "[$(date)] Uploading tarballs to HuggingFace..."
# hf upload "$REPO" "$TAR_DIR/openscene_meta_datas.tar" openscene_meta_datas.tar --repo-type dataset
# hf upload "$REPO" "$TAR_DIR/synthetic_scene_pickles.tar" synthetic_scene_pickles.tar --repo-type dataset
hf upload "$REPO" "$TAR_DIR/sensor_blobs.tar" sensor_blobs.tar --repo-type dataset

echo "[$(date)] All uploads complete."
