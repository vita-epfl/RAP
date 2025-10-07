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
import numpy as np
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import os
import math
import tensorflow as tf
import torch
from typing import List, Dict, Tuple

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
logger = logging.getLogger(__name__)

def nms_trajectories(
    model_outputs: List[Dict[str, torch.Tensor]],
    dist_threshold: float = 2.0,  # m，判定“太近”就抑制
    top_k: int = 1,               # 每个样本最多保留多少条轨迹
    metric: str = "endpoint",     # 'endpoint' 或 'average'
) -> Tuple[torch.Tensor, torch.Tensor]:

    # ① 纵向拼接 → [bs, Σmodes, 10, 3] / [bs, Σmodes]
    trajs = torch.cat([m["trajectory"] for m in model_outputs], dim=1)  # 同步 bs 维
    scores = torch.cat([m["score"]      for m in model_outputs], dim=1)

    best   = scores.argmax(dim=1)                                # [bs]
    best_traj  = trajs[torch.arange(trajs.size(0)), best][...,:2]                   # [bs, 10, 3]
    return best_traj



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def evaluation(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    test_data = WaymoCacheOnlyDataset(
        cache_path=cfg.cache_path,
        split='test',
    )

    logger.info("Building Datasets")
    test_dataloader = DataLoader(test_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num test samples: %d", len(test_data))

    logger.info("Building Trainer")

    ckpt_home = Path(cfg.agent.checkpoint_path).parent
    ckpt_paths = ckpt_home.glob("*.ckpt")

    all_model_preds = []  # list over models
    for idx, ckpt in enumerate(ckpt_paths):
        print(f"[Ensemble] Running inference {ckpt}")
        trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks(), logger=WandbLogger(project="unitraj", name=cfg.experiment_name, id=cfg.experiment_name))
        preds = trainer.predict(model=lightning_module, dataloaders=test_dataloader, ckpt_path=ckpt)
        trajs = torch.cat([m["trajectory"] for m in preds], dim=0)  # 同步 bs 维
        scores = torch.cat([m["score"] for m in preds], dim=0)
        frame_names = sum([m["frame_name"] for m in preds], [])
        all_model_preds.append({'trajectory':trajs,'score':scores})
        torch.cuda.empty_cache()
    

    merged_results = nms_trajectories(all_model_preds)

    predictions = []
    print('eval on ', merged_results.shape[0], 'trajectories')
    for i in range(merged_results.shape[0]):
        x = merged_results[i, :, 0].detach().cpu().numpy()
        y = merged_results[i, :, 1].detach().cpu().numpy()
        # 原始帧对应的时间点（偶数帧）
        even_frames = np.arange(2, 21, 2)  # [2, 4, ..., 20]

        # 添加第0帧为起点 (0, 0)
        full_x = [0.0]
        full_y = [0.0]
        full_frames = [0]

        # 插入偶数帧坐标
        full_x.extend(x.tolist())
        full_y.extend(y.tolist())
        full_frames.extend(even_frames.tolist())

        # 所有帧 0~20
        all_frames = np.arange(21)

        # 使用线性插值补齐所有帧
        interp_x = np.interp(all_frames, full_frames, full_x)[1:]
        interp_y = np.interp(all_frames, full_frames, full_y)[1:]

        predicted_trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(pos_x=interp_x,
                                                                    pos_y=interp_y)
        frame_name = frame_names[i]
        frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(frame_name=frame_name, trajectory=predicted_trajectory)
        predictions.append(frame_trajectory)
            
    num_submission_shards = 1  # Please modify accordingly.
    submission_file_base = './MySubmission'  # Please modify accordingly.
    if not os.path.exists(submission_file_base):
        os.makedirs(submission_file_base)
        
    sub_file_names = [
        os.path.join(submission_file_base, part)
        for part in [f'mysubmission.binproto-00000-of-00001']
    ]
    # As the submission file may be large, we shard them into different chunks.
    submissions = []
    num_predictions_per_shard =  math.ceil(len(predictions) / num_submission_shards)
    for i in range(num_submission_shards):
        start = i * num_predictions_per_shard
        end = (i + 1) * num_predictions_per_shard
        submissions.append(
        wod_e2ed_submission_pb2.E2EDChallengeSubmission(
            predictions=predictions[start:end]))
    for i, shard in enumerate(submissions):
        shard.submission_type  =  wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        shard.authors[:] = ['']  # Please modify accordingly.
        shard.affiliation = ''  # Please modify accordingly.
        shard.account_name = ''  # Please modify accordingly.
        shard.unique_method_name = 'RAP'  # Please modify accordingly.
        shard.method_link = ''  # Please modify accordingly.
        shard.description = '3D Rasterization Augmented End-to-End Planning (RAP)'  # Please modify accordingly.
        shard.uses_public_model_pretraining = True # Please modify accordingly.
        shard.public_model_names.extend(['DINO']) # Please modify accordingly.
        shard.num_model_parameters = "1B" # Please modify accordingly.
        with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
            fp.write(shard.SerializeToString())

    # run tar cvf on the generated file
    os.system('rm -rf MySubmission.tar')
    os.system('rm -rf MySubmission.tar.gz')
    os.system('tar cvf MySubmission.tar MySubmission')
    os.system('gzip -f MySubmission.tar')


if __name__ == '__main__':
    # seed everything
    evaluation()
