#!/bin/bash

#SBATCH --job-name=drivoR_warmup_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=32

## GAMMA training config
#SBATCH --time=5:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate drivoR

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/DrivoR"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"

# Change to DrivoR root so relative backbone weight paths in drivoR.yaml resolve correctly
cd $NAVSIM_DEVKIT_ROOT

TRAIN_TEST_SPLIT=warmup_test_e2e
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/training_drivoR_Nav1_traj_long_25epochs/05.03_22.26/lightning_logs/version_0/checkpoints/best-epoch32-step60363.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth
CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/training_drivoR_Nav1_traj_long_25epochs/nav2_adapt_dinov2fe/lightning_logs/version_0/checkpoints/best-epoch30-step57965.ckpt
METRIC_CACHE_PATH=/fs/nexus-projects/sim2real/aliu/DrivoR/metric_cache_warmup

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=drivoR \
    worker=single_machine_thread_pool \
    agent.checkpoint_path=$CHECKPOINT \
    agent.scheduler_args.num_epochs=25 \
    agent.batch_size=1 \
    experiment_name=drivoR_warmup_eval \
    metric_cache_path=$METRIC_CACHE_PATH \
    navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/mini \
    sensor_blobs_path=$OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini \
    agent.config.use_adapter=true \
    agent.config.image_backbone.use_hf_dinov2=true \
    agent.config.image_backbone.hf_model_name=facebook/dinov2-small \
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
    

# Pretrained model: /fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/eval_warmup.out.6739914