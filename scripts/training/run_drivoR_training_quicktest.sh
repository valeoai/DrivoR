#!/bin/bash

#SBATCH --job-name=drivoR_nav1_train
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=240gb
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

## GAMMA training config
#SBATCH --time=24:00:00     
#SBATCH --qos=huge-long                                    
#SBATCH --account=gamma
#SBATCH --partition=gamma

# --------------------------- UPDATE THIS ---------------------------
CONFIG_NAME=sim_0.25_add_debug
# --------------------------- UPDATE THIS ---------------------------

SCRIPT_PATH=$(readlink -f "$0")
EXP_ROOT=/fs/nexus-projects/sim2real/aliu/DrivoR/exp

if [ -z "$SLURM_JOB_ID" ]; then
    EXPERIMENT=$(date +%Y-%m-%d)
    LOG_DIR=$EXP_ROOT/ke/$EXPERIMENT/$CONFIG_NAME/lightning_logs
    mkdir -p $LOG_DIR
    sbatch --job-name=$CONFIG_NAME --export=ALL,EXPERIMENT=$EXPERIMENT --output=$LOG_DIR/train_%x.out.%j --error=$LOG_DIR/train_%x.out.%j "$SCRIPT_PATH"
    exit 0
fi

eval "$(conda shell.bash hook)"
conda activate drivoR

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/DrivoR"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"
export NAVSIM_TRAIN_METRIC_CACHE="$HOME/navsim/metric_cache_trainval"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Change to DrivoR root so relative backbone weight paths in drivoR.yaml resolve correctly
cd $NAVSIM_DEVKIT_ROOT

# PL's SLURMEnvironment uses SLURM_NTASKS=1 to set world_size=1, ignoring devices:8.
# Unset it so PL uses its subprocess launcher and respects the devices config.
unset SLURM_NTASKS

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py \
    --config-path $NAVSIM_DEVKIT_ROOT/my_configs \
    --config-name $CONFIG_NAME \
    'hydra.searchpath=[pkg://navsim.planning.script.config.common,pkg://navsim.planning.script.config.training]' \
    experiment_name=$EXPERIMENT

# Kick off warmup eval on the last epoch checkpoint
CKPT_DIR=$EXP_ROOT/ke/$EXPERIMENT/$CONFIG_NAME/lightning_logs/version_$SLURM_JOB_ID/checkpoints
CHECKPOINT=$(find $CKPT_DIR -name "epoch*.ckpt" ! -name "best-*" | sort | tail -1)

if [ -z "$CHECKPOINT" ]; then
    echo "No last-epoch checkpoint found in $CKPT_DIR, skipping eval."
    exit 0
fi

EPOCH=$(basename $CHECKPOINT .ckpt | cut -d'-' -f1)
EVAL_JOB_NAME=eval_${CONFIG_NAME}_${EPOCH}_wp

sbatch --job-name=$EVAL_JOB_NAME \
       --output=$(dirname $CHECKPOINT)/%x.out.%j \
       --error=$(dirname $CHECKPOINT)/%x.out.%j << EOF
#!/bin/bash
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=32
#SBATCH --time=10:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "\$(conda shell.bash hook)"
conda activate drivoR

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="\$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="\$HOME/DrivoR"
export NAVSIM_EXP_ROOT="\$NAVSIM_DEVKIT_ROOT/exp"
export OPENSCENE_DATA_ROOT="\$HOME/navsim/dataset"
export SUBSCORE_PATH=\$NAVSIM_EXP_ROOT

cd \$NAVSIM_DEVKIT_ROOT

python \$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    --config-path \$NAVSIM_DEVKIT_ROOT/my_configs \
    --config-name author_sim_eval_warmup \
    'hydra.searchpath=[pkg://navsim.planning.script.config.common,pkg://navsim.planning.script.config.training,pkg://navsim.planning.script.config.pdm_scoring]' \
    'hydra.output_subdir=null' \
    'hydra.run.dir=/tmp' \
    experiment_name=$EXPERIMENT \
    output_dir=$EXP_ROOT/ke/$EXPERIMENT/evaluations/$EVAL_JOB_NAME \
    agent.checkpoint_path=$CHECKPOINT
EOF
