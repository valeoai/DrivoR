#!/bin/bash

#SBATCH --job-name=hf_upload_sim_data
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

hf upload-large-folder AL314/m2r_sim_data \
    /fs/nexus-projects/sim2real/aliu/carla_garage_data_navsim_converted/ \
    --repo-type dataset \
    --num-workers 4