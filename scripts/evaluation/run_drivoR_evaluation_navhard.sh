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

# --------------------------- OPTION: TOGGLE THIS ---------------------------
# Set BATCH_DATE_DIR to a date folder name (e.g. 2026-07-05) to submit a
# separate navhard eval job for every run folder inside it (excluding
# "evaluations"). Leave empty to fall back to the single $CHECKPOINT below.
BATCH_DATE_DIR="2026-07-07"
# --------------------------- OPTION: TOGGLE THIS ---------------------------

CHECKPOINT_BASE=/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke

if [ -z "$SLURM_JOB_ID" ] && [ -n "$BATCH_DATE_DIR" ]; then
    for RUN_DIR in $CHECKPOINT_BASE/$BATCH_DATE_DIR/*/; do
        RUN_NAME=$(basename "$RUN_DIR")
        [ "$RUN_NAME" = "evaluations" ] && continue

        CKPT=$(find "$RUN_DIR" -name "epoch*.ckpt" ! -name "best-*" | sort | tail -1)
        if [ -z "$CKPT" ]; then
            echo "No last-epoch checkpoint found in $RUN_DIR, skipping."
            continue
        fi

        EPOCH=$(basename "$CKPT" .ckpt | cut -d'-' -f1)
        JOB_NAME=eval_${RUN_NAME}_${EPOCH}
        sbatch --job-name=$JOB_NAME --export=ALL,CHECKPOINT=$CKPT \
            --output=$(dirname "$CKPT")/%x.out.%j --error=$(dirname "$CKPT")/%x.out.%j "$SCRIPT_PATH"
    done
    exit 0
fi

: "${CHECKPOINT:=$CHECKPOINT_BASE/2026-07-07/sim_0.1_balanced/lightning_logs/version_7074273/checkpoints/epoch39-step51570.ckpt}"
#/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/2026-07-07/sim_0.1_replace/lightning_logs/version_7074272/checkpoints/epoch39-step51830.ckpt
# /fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/2026-07-07/sim_0.1_balanced/lightning_logs/version_7074273/checkpoints/epoch39-step51570.ckpt

MODEL_NAME=$(basename $(dirname $(dirname $(dirname $(dirname $CHECKPOINT)))))
TRAIN_DATE=$(basename $(dirname $(dirname $(dirname $(dirname $(dirname $CHECKPOINT))))))
EPOCH=$(basename $CHECKPOINT .ckpt | cut -d'-' -f1)
JOB_NAME=eval_${MODEL_NAME}_${EPOCH}

# Re-submit via sbatch with output next to the checkpoint; skip when already inside SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    sbatch --job-name=$JOB_NAME --output=$(dirname $CHECKPOINT)/%x.out.%j --error=$(dirname $CHECKPOINT)/%x.out.%j "$SCRIPT_PATH"
    exit 0
fi

CONFIG_NAME=author_sim_eval_navhard
# CONFIG_NAME=author_realonly_eval_navhard

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_v2.py \
    --config-path $NAVSIM_DEVKIT_ROOT/my_configs \
    --config-name $CONFIG_NAME \
    'hydra.searchpath=[pkg://navsim.planning.script.config.common,pkg://navsim.planning.script.config.training,pkg://navsim.planning.script.config.pdm_scoring]' \
    'hydra.output_subdir=null' \
    'hydra.run.dir=/tmp' \
    experiment_name=$EXPERIMENT \
    output_dir=$CHECKPOINT_BASE/$TRAIN_DATE/evaluations/$JOB_NAME \
    agent.checkpoint_path=$CHECKPOINT
