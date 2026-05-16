#!/bin/bash

#SBATCH --job-name=drivoR_navhard_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks-per-node=1

## GAMMA training config
#SBATCH --time=10:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate navsim

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/DrivoR"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT

# Change to DrivoR root so relative backbone weight paths in drivoR.yaml resolve correctly
cd $NAVSIM_DEVKIT_ROOT

TRAIN_TEST_SPLIT=navhard_two_stage
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/training_drivoR_Nav1_traj_long_25epochs/05.09_20.03/lightning_logs/version_0/checkpoints/epoch-epoch31-step-step57964.ckpt
CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/training_drivoR_Nav1_traj_long_25epochs/nav2_adapt_dinov2fe/lightning_logs/version_0/checkpoints/best-epoch30-step57965.ckpt
METRIC_CACHE_PATH=/fs/nexus-projects/sim2real/aliu/navsim/metric_cache_navhard
NAVHARD_DATA_ROOT=$OPENSCENE_DATA_ROOT/navhard_two_stage
EXPERIMENT=drivoR_navhard_eval

python /fs/nexus-projects/sim2real/aliu/DrivoR/navsim/planning/script/run_pdm_score_gpu_v2.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXPERIMENT \
    metric_cache_path=$METRIC_CACHE_PATH \
    synthetic_sensor_path=$NAVHARD_DATA_ROOT/sensor_blobs \
    synthetic_scenes_path=$NAVHARD_DATA_ROOT/synthetic_scene_pickles \
    agent=drivoR \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc=10 \
    agent.config.dac=13 \
    agent.config.ddc=6 \
    agent.config.ttc=14 \
    agent.config.ep=15 \
    agent.config.comfort=2 \
    agent.config.use_adapter=true \
    agent.config.image_backbone.use_hf_dinov2=true \
    agent.config.image_backbone.hf_model_name=facebook/dinov2-small
