#!/bin/bash

#SBATCH --job-name=download_mini
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma

wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz
tar -xzf openscene_metadata_mini.tgz

for split in {0..31}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_${split}.tgz
    echo "Extracting file openscene_sensor_mini_camera_${split}.tgz"
    tar -xzf openscene_sensor_mini_camera_${split}.tgz
    rm openscene_sensor_mini_camera_${split}.tgz
done

# Skipping lidar download: DrivoR uses lidar_pc: [] and does not need lidar sensor data.
# for split in {0..31}; do
#     wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_lidar/openscene_sensor_mini_lidar_${split}.tgz
#     echo "Extracting file openscene_sensor_mini_lidar_${split}.tgz"
#     tar -xzf openscene_sensor_mini_lidar_${split}.tgz
#     rm openscene_sensor_mini_lidar_${split}.tgz
# done

mv openscene-v1.1/meta_datas /fs/nexus-projects/sim2real/aliu/navsim/dataset/navsim_logs/mini
# rm -r openscene_v1.1

mv openscene-v1.1/sensor_blobs /fs/nexus-projects/sim2real/aliu/navsim/dataset/mini_sensor_blobs
# rm -r openscene-v1.1