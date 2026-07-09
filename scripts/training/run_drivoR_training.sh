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
CONFIG_NAME=sim_0.25_add_continue10
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
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Change to DrivoR root so relative backbone weight paths in drivoR.yaml resolve correctly
cd $NAVSIM_DEVKIT_ROOT

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


# TODO
# This is train_sim_0.1_add:                                                                                                   
#   [Epoch 30] Data mix summary (all ranks):                                                                                     
#     real :    12874  (90.9%)                                                                                                   
#     sim  :     1286  (9.1%)                                                                                                    
#     total:    14160                                                                                                            
                                                                                                                               
#   Compared to sim_0.1_12910:[Epoch 39] Data mix summary (all ranks):                                                           
#     real :     1280  (50.0%)                                                                                                   
#     sim  :     1280  (50.0%)                                                                                                   
#     total:     2560 
# ● Found it — the two configs use different sim_data_mode, and that's the whole story:

#   sim_0.1_add explicitly sets sim_data_mode: add (my_configs/sim_0.1_add.yaml:5). In run_training_full.py:137-141, "add" mode
#   keeps train_data completely untouched — it computes n_real but never subsets anything with it (that's the dead # TODO: check 
#   if my code actually limits by n_target at line 140). So the full real dataset survives (bounded only by max_scenes=12910 from
#   the base config), and sim is added on top at sim_data_ratio * len(real). Hence 12874 real (≈ max_scenes) + 1286 sim (≈10%) =
#   14160. This matches its docstring exactly and does respect max_scenes.

#   author_sim_0.1_12910 never sets sim_data_mode at all, so it falls back to the default "balanced" (line 123:
#   cfg.get("sim_data_mode", "balanced")). Balanced mode (lines 142-147) explicitly subsamples the real data down to n_real = 
#   min(n_target, len(train_data)) where n_target = int(sim_data_ratio * len(train_data)) — i.e. it throws away nearly everything
#   and shrinks real to match sim count (≈1280 each way, 2560 total). Bumping max_scenes to 12910 barely matters here: it just
#   enlarges the candidate pool that gets randomly downsampled back to ~10% anyway.

#   So author_sim_0.1_12910 is the "wrong" one relative to what its name implies. The config is named as if it trains on 12910
#   real scenes, but because it silently defaults to balanced mode instead of add, it actually only uses ~1280 real scenes per
#   epoch — the max_scenes: 12910 bump is nearly a no-op for real-data volume. If the intent was "use all 12910 real scenes + 10%
#   sim on top," that config is missing sim_data_mode: add.
