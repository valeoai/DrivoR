#!/bin/bash

#SBATCH --job-name=drivoR_warmup_eval
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

CHECKPOINT_BASE=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_realonly_training/lightning_logs/version_7021972/checkpoints/epoch39-step62140.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_realonly_training/lightning_logs/version_7021972/checkpoints/best-epoch35-step57888.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_training/lightning_logs/version_7022063/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_training/run_author_training/lightning_logs/version_7022063/checkpoints/best-epoch34-step57965.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_sim/test_rerun_author_sim_0.4238_6_23_26/lightning_logs/version_7029641/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_sim/author_sim_0.4238_6_23_26_lr0.00005/lightning_logs/version_7029713/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_sim/author_sim_0.4238_6_23_26_val/lightning_logs/version_7029714/checkpoints/epoch39-step64420.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/author_sim/author_sim_0.4238_6_23_26_val/lightning_logs/version_7029714/checkpoints/best-epoch35-step59256.ckpt
# CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/2026-06-28/author_sim_0.1/lightning_logs/version_7042874/checkpoints/epoch39-step54090.ckpt
CHECKPOINT=$CHECKPOINT_BASE/2026-06-29/author_sim_0.1/lightning_logs/version_7045829/checkpoints/epoch39-step54090.ckpt

MODEL_NAME=$(basename $(dirname $(dirname $(dirname $(dirname $CHECKPOINT)))))
EPOCH=$(basename $CHECKPOINT .ckpt | cut -d'-' -f1)
JOB_NAME=eval_${MODEL_NAME}_${EPOCH}_navhard

# Re-submit via sbatch with output next to the checkpoint; skip when already inside SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    sbatch --job-name=$JOB_NAME --output=$(dirname $CHECKPOINT)/eval_%x.out.%j --error=$(dirname $CHECKPOINT)/eval_%x.out.%j "$SCRIPT_PATH"
    exit 0
fi

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_v2.py \
    --config-path $NAVSIM_DEVKIT_ROOT/my_configs \
    --config-name author_sim_eval_navhard \
    'hydra.searchpath=[pkg://navsim.planning.script.config.common,pkg://navsim.planning.script.config.training,pkg://navsim.planning.script.config.pdm_scoring]' \
    'hydra.output_subdir=null' \
    'hydra.run.dir=/tmp' \
    experiment_name=$EXPERIMENT \
    agent.checkpoint_path=$CHECKPOINT
