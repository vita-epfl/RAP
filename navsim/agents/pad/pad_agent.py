from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from navsim.agents.pad.pad_model import PadModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.pad.pad_features import PadTargetBuilder
from navsim.agents.pad.pad_features import PadFeatureBuilder
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
from .score_module.compute_b2d_score import compute_corners_torch
from navsim.agents.transfuser.transfuser_loss import _agent_loss
from navsim.common.waymo_utils import get_rater_feedback_score, interpolate_trajectory
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR

class PadAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr: float,
            checkpoint_path: str = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data=self._config.cache_data

        if not cache_data:
            self._pad_model = PadModel(config)

        if not cache_data:#only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray and self._config.pdm_scorer:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                if self.b2d:
                    self.worker = RayDistributedNoTorch(threads_per_node=8)
                else:
                    self.worker = RayDistributedNoTorch(threads_per_node=16,use_distributed=True)
                self.worker_map=worker_map

            if config.b2d:
                self.train_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz")
                self.test_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/val_fut_boxes.gz")
                from .score_module.compute_b2d_score import get_scores
                self.get_scores = get_scores

                map_file =os.getenv("NAVSIM_EXP_ROOT") +"/map.pkl"

                with open(map_file, 'rb') as f:
                    self.map_infos = pickle.load(f)
                self.cuda_map=False

            else:
                from .score_module.compute_navsim_score import get_scores
                metric_cache = MetricCacheLoader(Path(cfg.train_metric_cache_path))

                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths

                self.get_scores = get_scores

        self.init_from_pretrained()

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.removeprefix('agent.'): v for k, v in state_dict.items()}            
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                print(f"Loading checkpoint on GPU#{torch.cuda.current_device()}")
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
                self.device = torch.device(f"cuda:{device_id}")
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
                self.device = torch.device("cpu")
            self.load_state_dict({k.replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()})
            self.to(self.device)

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )
    
    def get_target_builders(self) :
        return [PadTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [PadFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor],targets=None,return_score=False) -> Dict[str, torch.Tensor]:
        ego_vel = features['ego_status'][:,-1,3:5]
        if targets:
            targets['initial_speed'] = torch.norm(ego_vel, dim=-1)
        return self._pad_model(features,targets,return_score)

    def compute_score_fde(self, targets, proposals,beta=1):
        target_final = targets["trajectory"][:, -1, :2]      # (B, 2)
        proposal_final = proposals[:, :, -1, :2]             # (B, N, 2)

        # 2) 计算 FDE：L2 距离
        fde = torch.norm(proposal_final - target_final.unsqueeze(1), dim=-1)  # (B, N)

        # 3) 转换为 [0,1] 分数：score = exp(-fde / beta)
        #    距离越小，score 越接近 1；距离越大，score → 0
        scores = torch.exp(-fde / beta)  # (B, N)
        best_scores = scores.amax(dim=-1)
        return scores,best_scores
    
    def compute_score_rfs(self, targets, proposals):
        if targets.get('rfs_trajs') is None:
            rfs_trajs = targets['trajectory'].detach().cpu().numpy()
            initial_speed = targets['initial_speed'].detach().cpu().numpy()
            prediction_trajectories = proposals.detach().cpu().numpy()[...,:2]

            rater_specified_trajectories_list = []
            rater_scores_list = []
            prediction_trajectories_list = []
            prediction_probabilities_list = []
            for i in range(rfs_trajs.shape[0]):
                current_rfs = interpolate_trajectory(rfs_trajs[i])
                current_rfs_scores = [10]
                current_prediction_trajectories = prediction_trajectories[i]
                rfs_trajs_list = [current_rfs]

                prediction_trajectories_ = []
                for i in range(current_prediction_trajectories.shape[0]):
                    interp_prediction_trajectory = interpolate_trajectory(current_prediction_trajectories[i])
                    prediction_trajectories_.append(interp_prediction_trajectory)
                interp_prediction_trajectories = np.stack(prediction_trajectories_)

                rater_specified_trajectories_list.append(rfs_trajs_list)
                rater_scores_list.append(current_rfs_scores)
                prediction_trajectories_list.append(interp_prediction_trajectories)
                prediction_probabilities_list.append(np.ones(interp_prediction_trajectories.shape[0]))

            rater_feedback_metrics = get_rater_feedback_score(
                np.stack(prediction_trajectories_list),
                np.stack(prediction_probabilities_list),
                rater_specified_trajectories_list,
                rater_scores_list,
                initial_speed,
                frequency=4,  # Default is 4.
                length_seconds=5, # Default predict 5 seconds.
                output_trust_region_visualization=False,
                default_num_of_rater_specified_trajectories=1
            )
        else:
            rfs_trajs = targets['rfs_trajs'].detach().cpu().numpy()
            rfs_len = targets['rfs_len'].detach().cpu().numpy()
            rfs_scores = targets['rfs_scores'].detach().cpu().numpy()
            initial_speed = targets['initial_speed'].detach().cpu().numpy()
            prediction_trajectories = proposals.detach().cpu().numpy()[...,:2]

            rater_specified_trajectories_list = []
            rater_scores_list = []
            prediction_trajectories_list = []
            prediction_probabilities_list = []
            for i in range(len(prediction_trajectories)):
                current_rfs = rfs_trajs[i]
                current_rfs_len = rfs_len[i]
                current_rfs_scores = rfs_scores[i]
                current_prediction_trajectories = prediction_trajectories[i]
                rfs_trajs_list = [current_rfs[k][:current_rfs_len[k]] for k in range(len(current_rfs))]
                prediction_trajectories_ = []
                for i in range(current_prediction_trajectories.shape[0]):
                    interp_prediction_trajectory = interpolate_trajectory(current_prediction_trajectories[i])
                    prediction_trajectories_.append(interp_prediction_trajectory)
                interp_prediction_trajectories = np.stack(prediction_trajectories_)
                rater_specified_trajectories_list.append(rfs_trajs_list)
                rater_scores_list.append(current_rfs_scores)
                prediction_trajectories_list.append(interp_prediction_trajectories)
                prediction_probabilities_list.append(np.ones(interp_prediction_trajectories.shape[0]))

            rater_feedback_metrics = get_rater_feedback_score(
                np.stack(prediction_trajectories_list),
                np.stack(prediction_probabilities_list),
                rater_specified_trajectories_list,
                rater_scores_list,
                initial_speed,
                frequency=4,  # Default is 4.
                length_seconds=5, # Default predict 5 seconds.
                output_trust_region_visualization=False,
            )

        rater_feedback_score_per_inference = rater_feedback_metrics['rater_feedback_score_per_inference']
        #normalize to 0,1
        rater_feedback_score_per_inference = (rater_feedback_score_per_inference - 4) / 6
        # minimum is zero, rater_feedback_score_per_inference is array
        rater_feedback_score_per_inference = np.maximum(rater_feedback_score_per_inference, 0)
        rater_feedback_score_per_inference = torch.from_numpy(rater_feedback_score_per_inference).to(proposals.device)

        return rater_feedback_score_per_inference, rater_feedback_score_per_inference.amax(dim=-1)

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        if self.b2d:
            data_points = []

            lidar2worlds=targets["lidar2world"]

            all_proposals = torch.cat([proposals, target_trajectory[:,None]], dim=1)

            all_proposals_xy=all_proposals[:, :,:, :2]
            all_proposals_heading=all_proposals[:, :,:, 2:]

            all_pos = all_proposals_xy.reshape(len(target_trajectory),-1, 2)

            mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2

            dists = torch.linalg.norm(all_pos - mid_points[:,None], dim=-1).amax(1) + 5

            xyz = torch.cat(
                [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])], dim=-1)

            xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]

            vel=torch.cat([all_proposals_xy[:, :,:1], all_proposals_xy[:,:, 1:] - all_proposals_xy[:,:, :-1]],dim=2)/ 0.5

            proposals_05 = torch.cat([all_proposals_xy + vel*0.5, all_proposals_heading], dim=-1)

            proposals_1 = torch.cat([all_proposals_xy + vel*1, all_proposals_heading], dim=-1)

            proposals_ttc = torch.stack([all_proposals, proposals_05,proposals_1], dim=3)

            ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(proposals_ttc.shape[0],proposals_ttc.shape[1], proposals_ttc.shape[2], proposals_ttc.shape[3],  4, 2)

            ego_corners_center = torch.cat([ego_corners_ttc[:,:,:,0], all_proposals_xy[:, :, :, None]], dim=-2)

            ego_corners_center_xyz = torch.cat(
                [ego_corners_center, torch.zeros_like(ego_corners_center[..., :1]), torch.ones_like(ego_corners_center[..., :1])], dim=-1)

            global_ego_corners_centers = torch.einsum("nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz)[..., :2]

            accs = torch.linalg.norm(vel[:,:, 1:] - vel[:,:, :-1], dim=-1) / 0.5

            turning_rate=torch.abs(torch.cat([all_proposals_heading[:, :,:1,0]-np.pi/2, all_proposals_heading[:,:, 1:,0]-all_proposals_heading[:,:, :-1,0]],dim=2)) / 0.5

            comforts = (accs[:,:-1] < accs[:,-1:].max()).all(-1) & (turning_rate[:,:-1] < turning_rate[:,-1:].max()).all(-1)

            if self.cuda_map==False:
                for key, value in self.map_infos.items():
                    self.map_infos[key] = torch.tensor(value).to(target_trajectory.device)
                self.cuda_map=True

            for token, town_name, proposal,target_traj, comfort, dist, xy,global_conners,local_corners in zip(targets["token"], targets["town_name"], proposals.cpu().numpy(),  target_trajectory.cpu().numpy(), comforts.cpu().numpy(), dists.cpu().numpy(), xys, global_ego_corners_centers,ego_corners_ttc.cpu().numpy()):
                all_lane_points = self.map_infos[town_name[:6]]

                dist_to_cur = torch.linalg.norm(all_lane_points[:,:2] - xy, dim=-1)

                nearby_point = all_lane_points[dist_to_cur < dist]

                lane_xy = nearby_point[:, :2]
                lane_width = nearby_point[:, 2]
                lane_id = nearby_point[:, -1]

                dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)

                on_road = dist_to_lane < lane_width[:, None, None, None]

                on_road_all = on_road.any(0).all(-1)

                nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None,None], dim=0)

                nearest_lane_id=lane_id[nearest_lane]

                center_nearest_lane_id=nearest_lane_id[:,:,-1]

                nearest_road_id = torch.round(center_nearest_lane_id)

                target_road_id = torch.unique(nearest_road_id[-1])

                on_route_all = torch.isin(nearest_road_id, target_road_id)
                # in_multiple_lanes: if
                # - more than one drivable polygon contains at least one corner
                # - no polygon contains all corners
                corner_nearest_lane_id=nearest_lane_id[:,:,:-1]

                batch_multiple_lanes_mask = (corner_nearest_lane_id!=corner_nearest_lane_id[:,:,:1]).any(-1)

                on_road_all=on_road_all==on_road_all[-1:]
                # on_road_all = on_road_all | ~on_road_all[-1:]# on road or groundtruth offroad

                ego_areas=torch.stack([batch_multiple_lanes_mask,on_road_all,on_route_all],dim=-1)

                data_dict = {
                    "fut_box_corners": metric_cache_paths[token],
                    "_ego_coords": local_corners,
                    "target_traj": target_traj,
                    "proposal":proposal,
                    "comfort": comfort,
                    "ego_areas": ego_areas.cpu().numpy(),
                }
                data_points.append(data_dict)
        else:
            data_points = [
                {
                    "token": metric_cache_paths[token],
                    "poses": poses,
                    "test": test
                }
                for token, poses in zip(targets["token"], proposals.cpu().numpy())
            ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def score_loss(self, pred_logit, pred_logit2,agents_state, pred_area_logits, target_scores, gt_states, gt_valid,
                   gt_ego_areas):

        b,proposal_num,gt_poses,dim=gt_ego_areas.shape
        num_pose = self._config.trajectory_sampling.num_poses*5
        if agents_state is not None:
            pred_states = agents_state[..., :-1].reshape(gt_states.shape)
            pred_logits = agents_state[..., -1:].reshape(gt_valid.shape)

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(b,proposal_num,num_pose,dim)
            
            pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits[:,:,:gt_poses], gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            pred_area_loss = 0

        sub_score_loss = self.bce_logit_loss(pred_logit, target_scores[..., -pred_logit.shape[-1]:])  # .mean()[..., -6:]

        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            sub_score_loss2 = self.bce_logit_loss(pred_logit2, target_scores)  # .mean()[..., -6:-1][..., -6:-1]

            final_score_loss2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            sub_score_loss=(sub_score_loss+sub_score_loss2)/2

            final_score_loss=(final_score_loss+final_score_loss2)/2

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        #dist[dist==0]=10000

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss

    def pad_loss(self,targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config  ):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]
        score_mask = targets['score_mask']

        if self._config.pdm_scorer:
            if score_mask.sum()>0:
                score_targets = {k: v[score_mask] if isinstance(v, torch.Tensor) else [x for x, m in zip(v, score_mask) if m] for k, v in targets.items()}
                score_proposals = proposals[score_mask]

                final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
                    score_targets, score_proposals, test=False)
                best_score = best_scores.mean()

                pred_logit=pred["pred_logit"][score_mask] if pred["pred_logit"] is not None else None
                pred_logit2=pred["pred_logit2"][score_mask] if pred["pred_logit2"] is not None else None
                pred_agents_states=pred["pred_agents_states"][score_mask] if pred["pred_agents_states"] is not None else None
                pred_area_logit=pred["pred_area_logit"][score_mask] if pred["pred_area_logit"] is not None else None
                
                sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                    pred_logit,pred_logit2,
                    pred_agents_states, pred_area_logit
                    , target_scores, gt_states, gt_valid, gt_ego_areas)
                pdm_score = pred["pdm_score"][score_mask].detach()
                top_proposals = torch.argmax(pdm_score, dim=1)
                score = final_scores[np.arange(len(final_scores)), top_proposals].mean()

            else:
                sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0
                best_score = 0
                score = 0
        else:
            final_scores, best_scores = self.compute_score_rfs(targets, proposals)
            final_score_loss = self.bce_logit_loss(pred["pred_logit"][..., -1], final_scores)
            sub_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0
            best_score = best_scores.mean()


        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []

        for proposals_i in proposal_list:

            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(
                1)
            # weight min_loss by score mask
            weight = torch.ones_like(min_loss, dtype=torch.float32,device=min_loss.device)
            weight[~score_mask] = 0.1
            min_loss = (min_loss * weight).mean()
            
            inter_loss = self.diversity_loss(proposals_i)

            trajectory_loss = config.prev_weight * trajectory_loss  + min_loss+ inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]


        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(targets, pred, config)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
                config.trajectory_weight * trajectory_loss
                + config.sub_score_weight * sub_score_loss
                + config.final_score_weight * final_score_loss
                + config.pred_ce_weight * pred_ce_loss
                + config.pred_l1_weight * pred_l1_loss
                + config.pred_area_weight * pred_area_loss
                + config.agent_class_weight * agent_class_loss
                + config.agent_box_weight * agent_box_loss
                + config.bev_semantic_weight * bev_semantic_loss

        )
            


        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            "inter_loss0": inter_loss0,
            # "inter_loss1": inter_loss1,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            # "min_loss1": min_loss1,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config)

    def get_optimizers(self):
        vit_lr = self._lr * 0.2
        vit_params   = list(self._pad_model._backbone.img_backbone.parameters())
        other_params = [p for n, p in self._pad_model.named_parameters()
                        if p.requires_grad and '_backbone.img_backbone' not in n]
        for p in self._pad_model._backbone.img_backbone.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": self._lr, "weight_decay": 1e-4},
                #{"params": vit_params,   "lr": vit_lr,   "weight_decay": 1e-4},
            ]
        )
        scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-5, epochs=20, warmup_epochs=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-4,lr=self._lr)
        # scheduler = WarmupCosLR(
        #     optimizer=optimizer,
        #     lr=self._lr,
        #     min_lr=5e-6,
        #     epochs=20,
        #     warmup_epochs=1,
        # )
        
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}



    def get_training_callbacks(self):

        checkpoint_cb = ModelCheckpoint(
            save_last=True,
            save_top_k=3,
            monitor='val/score',
            filename='{epoch}-{step}',
            mode="max"
            )

        return [checkpoint_cb]