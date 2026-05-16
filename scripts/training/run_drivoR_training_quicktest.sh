#!/bin/bash

#SBATCH --job-name=drivoR_quicktest
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/DrivoR/my_dump/%x.out.%j

#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --time=01:30:00
#SBATCH --qos=high
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

cd $NAVSIM_DEVKIT_ROOT

EXPERIMENT=quicktest
AGENT=drivoR
RESUME_CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth

SIM_LOG_PATH=$HOME/carla_garage_data_navsim_converted/openscene_meta_datas
SIM_SENSOR_PATH=$HOME/carla_garage_data_navsim_converted/sensor_blobs
SIM_DATA_RATIO=1

SIM_ARGS=""
USE_ADAPTER=false
if [ "$(echo "$SIM_DATA_RATIO > 0" | bc -l)" -eq 1 ] && [ -n "$SIM_LOG_PATH" ]; then
    SIM_ARGS="sim_data_ratio=$SIM_DATA_RATIO sim_log_path=$SIM_LOG_PATH"
    if [ -n "$SIM_SENSOR_PATH" ]; then
        SIM_ARGS="$SIM_ARGS sim_sensor_path=$SIM_SENSOR_PATH"
    fi
    USE_ADAPTER=true
fi

unset SLURM_NTASKS

echo "USE_ADAPTER=$USE_ADAPTER"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_ckpt_path=null \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.devices=2 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=5 \
    dataloader.params.num_workers=4 \
    dataloader.params.pin_memory=false \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=2 \
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
    agent.config.use_matrix_adapter=false \
    agent.config.freeze_all_except_adapter=true \
    agent.loss.prev_weight=0.0 \
    trainer.params.limit_train_batches=500 \
    seed=2 \
    $SIM_ARGS
