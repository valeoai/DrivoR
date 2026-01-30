import os
import random
from typing import Tuple
from pathlib import Path
import logging
import pickle, datetime
import copy
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_mix"

def dist_ready():
    return dist.is_available() and dist.is_initialized()

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    print("Sensor blobs path: ", sensor_blobs_path)
    print("Data path: ", data_path)

    # print("cfg.train_test_split_synthetic ", cfg.train_test_split_synthetic.scene_filter)
    # print("cfg.train_test_split_synthetic ", type(cfg.train_test_split_synthetic))
    # print("cfg.train_test_split.scene_filter: ", cfg.train_test_split.scene_filter)


    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    
    train_data = Dataset(
            scene_loader=train_scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )
    
    # add synthetic data
    train_scene_filter_synthetic_0: SceneFilter = instantiate(cfg.train_test_split_synthetic.scene_filter)
    train_scene_filter_synthetic_1: SceneFilter = instantiate(cfg.train_test_split_synthetic.scene_filter)
    train_scene_filter_synthetic_2: SceneFilter = instantiate(cfg.train_test_split_synthetic.scene_filter)
    train_scene_filter_synthetic_3: SceneFilter = instantiate(cfg.train_test_split_synthetic.scene_filter)
    train_scene_filter_synthetic_4: SceneFilter = instantiate(cfg.train_test_split_synthetic.scene_filter)
    all_synthetic_log_names = copy.deepcopy(train_scene_filter_synthetic_0.log_names)
    train_scene_filter_synthetic_0.log_names = []
    train_scene_filter_synthetic_1.log_names = []
    train_scene_filter_synthetic_2.log_names = []
    train_scene_filter_synthetic_3.log_names = []
    train_scene_filter_synthetic_4.log_names = []
    for log_name in all_synthetic_log_names:
        if "-000" in log_name:
            train_scene_filter_synthetic_0.log_names.append(log_name)
        elif "-001" in log_name:
            train_scene_filter_synthetic_1.log_names.append(log_name)
        elif "-002" in log_name:
            train_scene_filter_synthetic_2.log_names.append(log_name)
        elif "-003" in log_name:
            train_scene_filter_synthetic_3.log_names.append(log_name)
        elif "-004" in log_name:
            train_scene_filter_synthetic_4.log_names.append(log_name)

    # add synthetic data 0
    data_path_synthetic_0 = Path(cfg.navsim_log_path_synthetic + '-0')
    sensor_blobs_path_synthetic_0 = Path(cfg.sensor_blobs_path_synthetic + '-0')
    print("Sensor blobs path synthetic_0: ", sensor_blobs_path_synthetic_0)
    print("Data path synthetic_0: ", data_path_synthetic_0)
    train_synthetic_scene_loader_0 = SceneLoader(
        sensor_blobs_path=sensor_blobs_path_synthetic_0,
        data_path=data_path_synthetic_0,
        scene_filter=train_scene_filter_synthetic_0,
        sensor_config=agent.get_sensor_config(),
    )
    
    train_data_synthetic_0 = Dataset(
            scene_loader=train_synthetic_scene_loader_0,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )

    # add synthetic data 1
    data_path_synthetic_1 = Path(cfg.navsim_log_path_synthetic + '-1')
    sensor_blobs_path_synthetic_1 = Path(cfg.sensor_blobs_path_synthetic + '-1')
    print("Sensor blobs path synthetic_1: ", sensor_blobs_path_synthetic_1)
    print("Data path synthetic_1: ", data_path_synthetic_1)
    train_synthetic_scene_loader_1 = SceneLoader(
        sensor_blobs_path=sensor_blobs_path_synthetic_1,
        data_path=data_path_synthetic_1,
        scene_filter=train_scene_filter_synthetic_1,
        sensor_config=agent.get_sensor_config(),
    )
    
    train_data_synthetic_1 = Dataset(
            scene_loader=train_synthetic_scene_loader_1,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )

    # add synthetic data 2
    data_path_synthetic_2 = Path(cfg.navsim_log_path_synthetic + '-2')
    sensor_blobs_path_synthetic_2 = Path(cfg.sensor_blobs_path_synthetic + '-2')
    print("Sensor blobs path synthetic_2: ", sensor_blobs_path_synthetic_2)
    print("Data path synthetic_2: ", data_path_synthetic_2)
    train_synthetic_scene_loader_2 = SceneLoader(
        sensor_blobs_path=sensor_blobs_path_synthetic_2,
        data_path=data_path_synthetic_2,
        scene_filter=train_scene_filter_synthetic_2,
        sensor_config=agent.get_sensor_config(),
    )
    
    train_data_synthetic_2 = Dataset(
            scene_loader=train_synthetic_scene_loader_2,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        )

    # # add synthetic data 3
    # data_path_synthetic_3 = Path(cfg.navsim_log_path_synthetic + '-3')
    # sensor_blobs_path_synthetic_3 = Path(cfg.sensor_blobs_path_synthetic + '-3')
    # print("Sensor blobs path synthetic_3: ", sensor_blobs_path_synthetic_3)
    # print("Data path synthetic_3: ", data_path_synthetic_3)
    # train_synthetic_scene_loader_3 = SceneLoader(
    #     sensor_blobs_path=sensor_blobs_path_synthetic_3,
    #     data_path=data_path_synthetic_3,
    #     scene_filter=train_scene_filter_synthetic_3,
    #     sensor_config=agent.get_sensor_config(),
    # )

    # train_data_synthetic_3 = Dataset(
    #         scene_loader=train_synthetic_scene_loader_3,
    #         feature_builders=agent.get_feature_builders(),
    #         target_builders=agent.get_target_builders(),
    #         cache_path=cfg.cache_path,
    #         force_cache_computation=cfg.force_cache_computation,
    #     )

 
    # # add synthetic data 4
    # data_path_synthetic_4 = Path(cfg.navsim_log_path_synthetic + '-4')
    # sensor_blobs_path_synthetic_4 = Path(cfg.sensor_blobs_path_synthetic + '-4')
    # print("Sensor blobs path synthetic_4: ", sensor_blobs_path_synthetic_4)
    # print("Data path synthetic_4: ", data_path_synthetic_4)
    # train_synthetic_scene_loader_4 = SceneLoader(
    #     sensor_blobs_path=sensor_blobs_path_synthetic_4,
    #     data_path=data_path_synthetic_4,
    #     scene_filter=train_scene_filter_synthetic_4,
    #     sensor_config=agent.get_sensor_config(),
    # )
    
    # train_data_synthetic_4 = Dataset(
    #         scene_loader=train_synthetic_scene_loader_4,
    #         feature_builders=agent.get_feature_builders(),
    #         target_builders=agent.get_target_builders(),
    #         cache_path=cfg.cache_path,
    #         force_cache_computation=cfg.force_cache_computation,
    #     )   




    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    # print("len(train_data) ", len(train_data))
    print("len(train_data_synthetic_0) ", len(train_data_synthetic_0))
    print("len(train_data_synthetic_1) ", len(train_data_synthetic_1))
    print("len(train_data_synthetic_2) ", len(train_data_synthetic_2))
    # print("len(train_data_synthetic_3) ", len(train_data_synthetic_3))
    # print("len(train_data_synthetic_4) ", len(train_data_synthetic_4))
    train_data = ConcatDataset([train_data, train_data_synthetic_0, train_data_synthetic_1, train_data_synthetic_2])
    # train_data = train_data_synthetic_0
    print("after concat len(train_data) ", len(train_data))
    # assert False
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False,drop_last=True)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    if cfg.train_ckpt_path is None:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = "/".join(str(cfg.output_dir).split("/")[:-1]) + "/*/lightning_logs/version_*/checkpoints/" + '*.ckpt'
        print("/".join(str(cfg.output_dir).split("/")[:-1]))
        print("search_pattern ", search_pattern)
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
        print("cfg.train_ckpt_path ", cfg.train_ckpt_path)
    
    strategy = DDPStrategy(timeout=datetime.timedelta(seconds=7200))
    trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)
    trainer_params["strategy"] = strategy
    trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
