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

EXPERIMENT=5-18_5-24
AGENT=drivoR

# Resume from a specific checkpoint by setting RESUME_CHECKPOINT to its path.
# Leave empty to auto-resume from the latest checkpoint of this experiment (if
# one exists), or start fresh if none is found.
# RESUME_CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth
RESUME_CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth

# ── Simulator data config ────────────────────────────────────────────────────
SIM_LOG_PATH=$HOME/carla_garage_data_navsim_converted/openscene_meta_datas
SIM_SCENES_PATH=$HOME/carla_garage_data_navsim_converted/synthetic_scene_pickles # unused?
SIM_SENSOR_PATH=$HOME/carla_garage_data_navsim_converted/sensor_blobs
SIM_DATA_RATIO=0.5
# ────────────────────────────────────────────────────────────────────────────

# Build optional sim args and enable adapter when sim data is present
SIM_ARGS=""
USE_ADAPTER=false
USE_MATRIX_ADAPTER=false
if [ "$(echo "$SIM_DATA_RATIO > 0" | bc -l)" -eq 1 ] && [ -n "$SIM_LOG_PATH" ]; then
    SIM_ARGS="sim_data_ratio=$SIM_DATA_RATIO sim_log_path=$SIM_LOG_PATH"
    if [ -n "$SIM_SENSOR_PATH" ]; then
        SIM_ARGS="$SIM_ARGS sim_sensor_path=$SIM_SENSOR_PATH"
    fi
    if [ -n "$SIM_CACHE_PATH" ]; then
        SIM_ARGS="$SIM_ARGS sim_cache_path=$SIM_CACHE_PATH"
    fi
    USE_ADAPTER=true
    USE_MATRIX_ADAPTER=true
fi

# PL's SLURMEnvironment uses SLURM_NTASKS=1 to set world_size=1, ignoring devices:8.
# Unset it so PL uses its subprocess launcher and respects the devices config.
unset SLURM_NTASKS

echo "USE_ADAPTER=$USE_ADAPTER"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_ckpt_path=$RESUME_CHECKPOINT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.devices=8 \
    trainer.params.max_epochs=40 \
    trainer.params.check_val_every_n_epoch=100 \
    trainer.params.strategy=ddp \
    dataloader.params.prefetch_factor=2 \
    dataloader.params.batch_size=10 \
    dataloader.params.num_workers=4 \
    dataloader.params.pin_memory=false \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=8 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.image_backbone.use_hf_dinov2=true \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.ray_threads=2 \
    agent.config.use_adapter=$USE_ADAPTER \
    agent.config.use_matrix_adapter=true \
    agent.config.freeze_perception=true \
    agent.loss.prev_weight=0.0 \
    trainer.params.limit_train_batches=500 \
    seed=2 \
    $SIM_ARGS
# $USE_MATRIX_ADAPTER
    # agent.config.image_backbone.hf_model_name=facebook/dinov2-small \