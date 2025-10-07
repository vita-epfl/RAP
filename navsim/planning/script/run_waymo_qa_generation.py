import json
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
from tqdm import tqdm
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

    dataset = instantiate(
        cfg.dataset,
        file_list=file_list,
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
    #cache_features(data_points)
    _ = worker_map(worker, cache_features, data_points)
    training_dir = Path(cfg.cache_path)/"training"
    test_dir = Path(cfg.cache_path)/"test"
    val_dir = Path(cfg.cache_path)/"val"
    training_dir_annotated = Path(cfg.cache_path)/"training_annotation"
    test_dir_annotated = Path(cfg.cache_path)/"test_annotation"
    val_dir_annotated = Path(cfg.cache_path)/"val_annotation"
    


    for split_dir in [training_dir, val_dir, test_dir, training_dir_annotated, test_dir_annotated, val_dir_annotated]:
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} 不存在")
            continue

        merged_qas = []

        # 支持 .json 和 .jsonl；若有子目录，用 rglob
        json_files = sorted(split_dir.rglob("*.json*"))
        for fp in tqdm(json_files, desc=f"Collecting {split_dir.name}"):
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    merged_qas.extend(obj)
                else:
                    merged_qas.append(obj)

        # 写出单文件（数组格式）和流式 jsonl 两份，按需使用
        out_json  = split_dir.parent / f"{split_dir.name}_merged.json"

        with out_json.open("w", encoding="utf-8") as f_json:
            json.dump(merged_qas, f_json, ensure_ascii=False, indent=2)


        print(f"[OK] {split_dir.name}: {len(merged_qas)} samples → "
            f"{out_json.name}")
    

if __name__ == "__main__":
    main()
