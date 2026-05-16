#!/bin/bash

#SBATCH --job-name=navhard_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

## GAMMA training config
#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate drivoR

export NAVSIM_DEVKIT_ROOT="/fs/nexus-projects/sim2real/aliu/DrivoR"
export NAVSIM_EXP_ROOT="/fs/nexus-projects/sim2real/aliu/DrivoR/exp"
export OPENSCENE_DATA_ROOT="/fs/nexus-projects/sim2real/aliu/navsim/dataset"

export NUPLAN_MAPS_ROOT="/fs/nexus-projects/sim2real/aliu/navsim/dataset/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
TRAIN_TEST_SPLIT=navtrain
# Resume from partial cache; script skips scenarios where metric_cache.pkl already exists
CACHE_PATH=$NAVSIM_EXP_ROOT/train_metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH \
worker.threads_per_node=32
