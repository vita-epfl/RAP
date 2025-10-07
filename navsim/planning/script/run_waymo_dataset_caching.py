import cv2
cv2.setNumThreads(1)
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    file_list = [a["file"] for a in args]

    cfg: DictConfig = args[0]["cfg"]


    agent: AbstractAgent = instantiate(cfg.agent)

    dataset = instantiate(
        cfg.dataset,
        file_list=file_list,
        agent=agent,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)

    waymo_raw_path = Path(cfg.waymo_raw_path)
    data_points = [{"file": f,"cfg": cfg} for f in waymo_raw_path.iterdir() if 'tfrecord' in str(f)]

    print('len of data_points', len(data_points))
    cache_features(data_points)
    #_ = worker_map(worker, cache_features, data_points)
    logger.info(f"Finished caching {len(data_points)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()
