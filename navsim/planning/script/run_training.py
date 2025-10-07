from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, WaymoCacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from pytorch_lightning.loggers import WandbLogger

import random
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
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

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

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

        cached_logs = [log_name.name.replace(".pkl", "") for log_name in Path(cfg.cache_path).iterdir()]
        train_logs = [log_name for log_name in cached_logs if log_name not in cfg.val_logs]
        val_logs = [log_name for log_name in cached_logs if log_name in cfg.val_logs]

        if 'waymo' in cfg.dataset['_target_']:
            train_data = WaymoCacheOnlyDataset(
                cache_path=cfg.cache_path,
                split='training'
            )
            val_data = WaymoCacheOnlyDataset(
                cache_path=cfg.cache_path,
                split='val',
            )
            # # split val_data by 80/20
            # import random
            # from torch.utils.data import ConcatDataset, Subset
            # N = len(val_data)
            # indices = random.sample(range(N), int(0.8*N))
            # the_rest = [i for i in range(N) if i not in indices]
            # train_data = Subset(val_data, indices)
            # val_data = Subset(val_data, the_rest)
        else:
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
            log_names=train_logs,
        )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=val_logs,
                split='val'
            )

            train_data_perturbed = CacheOnlyDataset(
                cache_path=cfg.cache_path_perturbed,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders())
            N = len(train_data_perturbed)
            indices = random.sample(range(N), int(0.1*N))
            print(f'len(perturbed): {len(indices)}')
            train_data_perturbed = Subset(train_data_perturbed, indices)

            train_data_others = CacheOnlyDataset(
                cache_path=cfg.cache_path_others,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders())
                
            train_data_others.score_mask=False
            N = len(train_data_others)
            indices = random.sample(range(N), int(0.05*N))
            print(f'len(others): {len(indices)}')
            train_data_others = Subset(train_data_others, indices)

            train_data = ConcatDataset([train_data, train_data_perturbed, train_data_others])

    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks(), logger=WandbLogger(project="rap", name=cfg.experiment_name, id=cfg.experiment_name),
            )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path='last'
    )


if __name__ == "__main__":
    main()
