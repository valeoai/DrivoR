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

EXPERIMENT=training_drivoR_realonly_10epochs
AGENT=drivoR

RESUME_CHECKPOINT=/fs/nexus-projects/sim2real/aliu/DrivoR/weights/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth

unset SLURM_NTASKS

# halved lr, is this the difference maker?
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_ckpt_path=$RESUME_CHECKPOINT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=40 \
    dataloader.params.batch_size=10 \
    dataloader.params.num_workers=4 \
    dataloader.params.prefetch_factor=2 \
    dataloader.params.pin_memory=false \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0001 \
    agent.num_gpus=8 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    seed=2 \
    agent.config.freeze_perception=true \
    agent.config.use_adapter=$USE_ADAPTER \
    $SIM_ARGS

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py  \
#     agent=$AGENT \
#     experiment_name=$EXPERIMENT \
#     train_ckpt_path=$RESUME_CHECKPOINT \
#     train_test_split=navtrain \
#     cache_path=null \
#     use_cache_without_dataset=false \
#     trainer.params.max_epochs=40 \
#     dataloader.params.batch_size=10 \
#     dataloader.params.num_workers=4 \
#     dataloader.params.prefetch_factor=2 \
#     dataloader.params.pin_memory=false \
#     agent.lr_args.name=AdamW \
#     agent.lr_args.base_lr=0.0002 \
#     agent.num_gpus=8 \
#     agent.progress_bar=false \
#     agent.config.refiner_ls_values=0.0 \
#     agent.config.image_backbone.focus_front_cam=false \
#     agent.config.one_token_per_traj=true \
#     agent.config.refiner_num_heads=1 \
#     agent.config.tf_d_model=256 \
#     agent.config.tf_d_ffn=1024 \
#     agent.config.area_pred=false \
#     agent.config.agent_pred=false \
#     agent.config.ref_num=4 \
#     agent.loss.prev_weight=0.0 \
#     seed=2


#   Metric                                Baseline  Finetuned      Delta
#   ego_progress                            0.8954     0.7953    -0.1001  ← dominant
#   drivable_area_compliance                1.0000     0.9663    -0.0337
#   time_to_collision_within_bound          0.9929     0.9645    -0.0284
#   no_at_fault_collisions                  0.9956     0.9689    -0.0266

#   ego_progress is the main culprit — a 10pp drop, which accounts for most of the 9pp score drop. The model is making significantly less forward progress along the route after
#   fine-tuning. The secondary issues (more DAC violations, more collisions, worse TTC) are real but smaller.

#   This pattern — less progress + more DAC/collision violations — is consistent with the model becoming more hesitant or erratic after being exposed to Carla sim data. The sim domain has a different
#   scene distribution and driving style, which pulls the model's proposals toward behaviors that don't generalize well to the real warmup scenes.

#   On val loss: it wouldn't tell you much here. You can already see exactly what broke and by how much from the per-metric CSV. Val loss would only confirm whether the model is overfitting (val loss
#   rising while train loss falls), but the specific degradation pattern — ego_progress dropping while comfort and traffic_light stay perfect — points to the sim data changing the proposal
#   distribution toward slower/more conservative trajectories, not overfitting. Val loss is a single scalar that would obscure this structure.

#   The more useful next diagnostic would be checking whether the ego_progress drop correlates with the sim data ratio, or whether it's a consequence of training on a particular subset of real data
#   that has lower speed profiles.