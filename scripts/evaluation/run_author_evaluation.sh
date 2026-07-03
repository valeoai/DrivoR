#!/bin/bash

#SBATCH --job-name=drivoR_warmup_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=32

## GAMMA training config
#SBATCH --time=10:00:00
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
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT

SCRIPT_PATH=$(readlink -f "$0")

cd $NAVSIM_DEVKIT_ROOT

EXPERIMENT=drivoR_nav2
AGENT=drivoR

CHECKPOINT_BASE=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke
# CHECKPOINT=$CHECKPOINT_BASE/test_rerun_author_sim_0.4238_6_23_26/lightning_logs/version_7029641/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=$CHECKPOINT_BASE/author_sim_0.4238_6_23_26_lr0.00005/lightning_logs/version_7029713/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=$CHECKPOINT_BASE/author_sim_0.4238_6_23_26_val/lightning_logs/version_7029714/checkpoints/best-epoch35-step59256.ckpt
# CHECKPOINT=$CHECKPOINT_BASE/author_sim/sanity_author/lightning_logs/version_7035005/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=$CHECKPOINT_BASE/2026-06-29/test_data_size_realonly_12910/lightning_logs/version_7045334/checkpoints/epoch39-step53120.ckpt
CHECKPOINT=$CHECKPOINT_BASE/2026-06-30/author_sim_0.1_0.4787_6_29_26_12910/lightning_logs/version_7047943/checkpoints/epoch39-step51830.ckpt


MODEL_NAME=$(basename $(dirname $(dirname $(dirname $(dirname $CHECKPOINT)))))
EPOCH=$(basename $CHECKPOINT .ckpt | cut -d'-' -f1)
JOB_NAME=eval_${MODEL_NAME}_${EPOCH}_wp

# Re-submit via sbatch with output next to the checkpoint; skip when already inside SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    sbatch --job-name=$JOB_NAME --output=$(dirname $CHECKPOINT)/%x.out.%j --error=$(dirname $CHECKPOINT)/%x.out.%j "$SCRIPT_PATH"
    exit 0
fi

CONFIG_NAME=author_sim_eval_warmup
# CONFIG_NAME=author_realonly_eval_warmup

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    --config-path $NAVSIM_DEVKIT_ROOT/my_configs \
    --config-name $CONFIG_NAME \
    'hydra.searchpath=[pkg://navsim.planning.script.config.common,pkg://navsim.planning.script.config.training,pkg://navsim.planning.script.config.pdm_scoring]' \
    'hydra.output_subdir=null' \
    'hydra.run.dir=/tmp' \
    experiment_name=$EXPERIMENT \
    agent.checkpoint_path=$CHECKPOINT

