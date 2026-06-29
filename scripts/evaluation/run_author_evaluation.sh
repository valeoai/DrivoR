#!/bin/bash

#SBATCH --job-name=drivoR_warmup_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=32

## GAMMA training config
#SBATCH --time=1:00:00
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

cd $NAVSIM_DEVKIT_ROOT

TRAIN_TEST_SPLIT=warmup_test_e2e
METRIC_CACHE_PATH=/fs/nexus-projects/sim2real/aliu/DrivoR/metric_cache_warmup
EXPERIMENT=drivoR_nav2
AGENT=drivoR

# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_realonly_training/lightning_logs/version_7021972/checkpoints/epoch39-step62140.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_realonly_training/lightning_logs/version_7021972/checkpoints/best-epoch35-step57888.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_training/lightning_logs/version_7022063/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_training/lightning_logs/version_7022063/checkpoints/best-epoch34-step57965.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_sim/test_rerun_author_sim_0.4238_6_23_26/lightning_logs/version_7029641/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_sim/sanity_author/lightning_logs/version_7035005/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_tests/author_sim_0.5/lightning_logs/version_7041255/checkpoints/epoch34-step57965.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_tests/author_realonly_sanity/lightning_logs/version_7041248/checkpoints/epoch39-step64420.ckpt
CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/2026-06-28/author_sim_0.1/lightning_logs/version_7042874/checkpoints/epoch39-step54090.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py  \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=$AGENT \
    worker=single_machine_thread_pool \
    agent.checkpoint_path=$CHECKPOINT \
    agent.scheduler_args.num_epochs=25 \
    agent.batch_size=1 \
    experiment_name=$EXPERIMENT \
    metric_cache_path=$METRIC_CACHE_PATH \
    navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/mini \
    sensor_blobs_path=$OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini \
    agent.config.use_adapter=true \
    agent.config.use_matrix_adapter=false \
    agent.config.image_backbone.use_hf_dinov2=false \
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
    agent.config.comfort=2