import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple
import torch
from navsim.agents.abstract_agent import AbstractAgent
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from navsim.common.waymo_utils import get_rater_feedback_score, interpolate_trajectory
import numpy as np
import io, torch, zstandard as zstd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.distill_feature = agent._config.distill_feature
        # self.real_feat = []
        # self.synth_feat = []

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        batch_size = features['camera_valid'].shape[0]
        #features['camera_feature'] = features.pop('rendered_camera_feature')
        prediction = self.agent.forward(features,targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if 'rfs_trajs' in targets:
            rfs_trajs = targets['rfs_trajs'].detach().cpu().numpy()
            rfs_len = targets['rfs_len'].detach().cpu().numpy()
            rfs_scores = targets['rfs_scores'].detach().cpu().numpy()
            initial_speed = targets['initial_speed'].detach().cpu().numpy()
            prediction_trajectories = prediction['trajectory'].detach().cpu().numpy()[...,:2]

            rater_specified_trajectories_list = []
            rater_scores_list = []
            prediction_trajectories_list = []
            prediction_probabilities_list = []
            for i in range(batch_size):
                current_rfs = rfs_trajs[i]
                current_rfs_len = rfs_len[i]
                current_rfs_scores = rfs_scores[i]
                current_prediction_trajectories = prediction_trajectories[i]
                current_prediction_probabilities = np.ones(1)
                rfs_trajs_list = [current_rfs[k][:current_rfs_len[k]] for k in range(len(current_rfs))]
                interp_prediction_trajectories = interpolate_trajectory(current_prediction_trajectories)
                rater_specified_trajectories_list.append(rfs_trajs_list)
                rater_scores_list.append(current_rfs_scores)
                prediction_trajectories_list.append(interp_prediction_trajectories[None])
                prediction_probabilities_list.append(current_prediction_probabilities)

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
            loss_dict['rater_feedback_score'] = torch.tensor(rater_feedback_metrics['rater_feedback_score']).mean().to(self.device)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            visualize_idx = 0

            img_mean = [123.675, 116.28, 103.53]
            img_std  = [58.395, 57.12, 57.375]
            camera = features['camera_feature'][visualize_idx, 1].permute(1, 2, 0).cpu().numpy()
            camera = (camera * img_std + img_mean).astype(np.uint8)
            camera = camera[:, :, ::-1]  # BGR->RGB

            ego_status = features['ego_status'][visualize_idx, -1].cpu().numpy()
            pred_traj  = prediction['trajectory'][visualize_idx].detach().cpu().numpy()[:, :2]
            gt_traj    = targets['trajectory'][visualize_idx].cpu().numpy()[:, :2]

            # === 创建 1x2 子图 ===
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # 子图1: Camera
            axs[0].imshow(camera)
            axs[0].set_title("Camera View")
            axs[0].axis('off')
            status_text = "\n".join([f"{i}: {v:.2f}" for i, v in enumerate(ego_status)])
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            axs[0].text(5, 20, status_text, fontsize=10, va='top', bbox=props)

            # 子图2: Trajectory
            axs[1].plot(pred_traj[:, 0], pred_traj[:, 1], 'ro-', label="Predicted")
            axs[1].plot(gt_traj[:, 0],   gt_traj[:, 1],   'go-', label="Ground Truth")
            for i in range(len(pred_traj)):
                axs[1].annotate(str(i), (pred_traj[i, 0], pred_traj[i, 1]), color='red')
                axs[1].annotate(str(i), (gt_traj[i, 0], gt_traj[i, 1]), color='green')
            axs[1].set_title("Trajectory")
            axs[1].set_xlabel("X"); axs[1].set_ylabel("Y")
            axs[1].legend(); axs[1].grid(True); axs[1].axis('equal')

            plt.tight_layout()

            # 上传 wandb
            wandb.log({f"{logging_prefix}/visualization": [wandb.Image(fig)]})
            plt.close(fig)



        if type(loss_dict) is dict:
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return loss_dict

    def _step_distill(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        self.agent._rap_model.progress = (self.current_epoch+1)/20

        real_only = False
        features, targets = batch
        if features.get('frame_name') is not None:
            features.pop('frame_name')
    
        real_valid_mask = features['camera_valid']
        
        batch_size = real_valid_mask.shape[0]
        self.agent._rap_model.batch_size = batch_size

        real_features = {k: v[real_valid_mask] for k, v in features.items() if k not in ['camera_valid','rendered_camera_feature']}

        real_targets = {k: v[real_valid_mask] if isinstance(v, torch.Tensor) else [x for x, m in zip(v, real_valid_mask) if m] for k, v in targets.items()}

        features.pop('camera_valid')
        features['camera_feature'] = features.pop('rendered_camera_feature')
        rendered_features = features
        rendered_targets = targets        

        if not self.training or real_only:
            if real_valid_mask.any():
                prediction = self.agent.forward(real_features,real_targets)
                loss_dict = self.agent.compute_loss(real_features, real_targets, prediction)
                ade_real = torch.mean(torch.norm(prediction['trajectory'][:,:,:2] - real_targets['trajectory'][:,:,:2], dim=-1))
                loss_dict['ade_real'] = ade_real
                if 'rfs_trajs' in real_targets:
                    rfs_trajs = real_targets['rfs_trajs'].detach().cpu().numpy()
                    rfs_len = real_targets['rfs_len'].detach().cpu().numpy()
                    rfs_scores = real_targets['rfs_scores'].detach().cpu().numpy()
                    initial_speed = real_targets['initial_speed'].detach().cpu().numpy()
                    prediction_trajectories = prediction['trajectory'].detach().cpu().numpy()[...,:2]

                    rater_specified_trajectories_list = []
                    rater_scores_list = []
                    prediction_trajectories_list = []
                    prediction_probabilities_list = []
                    for i in range(batch_size):
                        current_rfs = rfs_trajs[i]
                        current_rfs_len = rfs_len[i]
                        current_rfs_scores = rfs_scores[i]
                        current_prediction_trajectories = prediction_trajectories[i]
                        current_prediction_probabilities = np.ones(1)
                        rfs_trajs_list = [current_rfs[k][:current_rfs_len[k]] for k in range(len(current_rfs))]
                        interp_prediction_trajectories = interpolate_trajectory(current_prediction_trajectories)
                        rater_specified_trajectories_list.append(rfs_trajs_list)
                        rater_scores_list.append(current_rfs_scores)
                        prediction_trajectories_list.append(interp_prediction_trajectories[None])
                        prediction_probabilities_list.append(current_prediction_probabilities)

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
                    loss_dict['rater_feedback_score'] = torch.tensor(rater_feedback_metrics['rater_feedback_score']).mean().to(self.device)
            else:
                # prediction = self.agent.forward(rendered_features,rendered_targets)
                # loss_dict = self.agent.compute_loss(rendered_features, rendered_targets, prediction)
                # ade_real = torch.mean(torch.norm(prediction['trajectory'][:,:,:2] - rendered_targets['trajectory'][:,:,:2], dim=-1))
                # loss_dict['ade_real'] = ade_real
                return 0
        else:
            all_features = {}
            for k, v in rendered_features.items():
                all_features[k] = torch.cat([v, real_features[k]], dim=0)
    
            all_targets = {}
            for k, v in rendered_targets.items():
                all_targets[k] = torch.cat([v, real_targets[k]], dim=0) if isinstance(v, torch.Tensor) else v + real_targets[k] 
 

            prediction = self.agent.forward(all_features,all_targets)

            loss_dict = self.agent.compute_loss(all_features, all_targets, prediction)

            if real_valid_mask.any():
                                                
                domain_logits = prediction['domain_logits']
                N_synth = batch_size       # synthetic 数量
                N_real = domain_logits.shape[0] - N_synth

                if N_synth == 0 or N_real == 0:
                    domain_loss = torch.zeros((), device=device, dtype=torch.float32)
                else:

                    pos_weight = torch.tensor([N_synth / max(1, N_real)], device=domain_logits.device)
                    bce_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                    domain_labels = torch.cat([
                        torch.zeros(N_synth, device=domain_logits.device),  # synthetic=0
                        torch.ones(N_real, device=domain_logits.device)     # real=1
                    ], dim=0)
                    domain_loss = bce_logits(domain_logits, domain_labels.float())

                loss_dict['domain_loss'] = domain_loss
                loss_dict['loss'] += 0.001*domain_loss
                
                bev_feature = prediction['bev_feature']
                render_bev = bev_feature[:batch_size][real_valid_mask].detach()
                real_bev = bev_feature[batch_size:]
                loss_render = F.mse_loss(render_bev, real_bev)
                loss_dict['loss'] += self.agent._config.distill_feature_weight * loss_render

                # self.real_feat.append(real_bev.detach().cpu().numpy()[:,1].mean(axis=-2))
                # self.synth_feat.append(render_bev.detach().cpu().numpy()[:,1].mean(axis=-2))
                ade_real = torch.mean(torch.norm(prediction['trajectory'][batch_size:,:,:2] - all_targets['trajectory'][batch_size:,:,:2], dim=-1))
                loss_dict['ade_real'] = ade_real
                loss_dict['loss_render'] = loss_render
                
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]) if k not in ['ade_real', 'loss_render'] else int(real_valid_mask.sum()))
        
        if self.global_step % 10 == 0 and self.global_rank == 0 and False:
            visualize_idx = 0
            projected_feats = prediction['projected_feats'][visualize_idx,1].detach().cpu().numpy()
            dino_feats = prediction['dino_feats'][visualize_idx,1].detach().cpu().numpy()

            rgb_dino, bg_mask, thr = visualize_dino_pca_sklearn(dino_feats)      # (H,W,3)
            rgb_proj, bg_mask2, thr2 = visualize_dino_pca_sklearn(projected_feats)

            # 反归一化相机图像
            img_mean = [123.675, 116.28, 103.53]
            img_std = [58.395, 57.12, 57.375]
            camera = all_features['camera_feature'][visualize_idx,1].permute(1, 2, 0).cpu().numpy()
            camera = (camera * img_std + img_mean).astype(np.uint8)
            camera = camera[:, :, ::-1]

            ego_status = all_features['ego_status'][visualize_idx,-1].cpu().numpy()
            pred_traj = prediction['trajectory'][visualize_idx].detach().cpu().numpy()[:, :2]
            gt_traj   = all_targets['trajectory'][visualize_idx].cpu().numpy()[:, :2]

            # === 创建 2x2 子图 ===
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))

            # 子图1: Camera
            axs[0,0].imshow(camera)
            axs[0,0].set_title("Camera View")
            axs[0,0].axis('off')
            status_text = "\n".join([f"{i}: {v:.2f}" for i, v in enumerate(ego_status)])
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            axs[0,0].text(5, 20, status_text, fontsize=10, verticalalignment='top', bbox=props)

            # 子图2: Dino features (PCA RGB)
            axs[0,1].imshow(rgb_dino)
            axs[0,1].set_title("DINO Features PCA")
            axs[0,1].axis('off')

            # 子图3: Trajectory
            axs[1,0].plot(pred_traj[:, 0], pred_traj[:, 1], 'ro-', label="Predicted")
            axs[1,0].plot(gt_traj[:, 0], gt_traj[:, 1], 'go-', label="Ground Truth")
            for i in range(len(pred_traj)):
                axs[1,0].annotate(str(i), (pred_traj[i, 0], pred_traj[i, 1]), color='red')
                axs[1,0].annotate(str(i), (gt_traj[i, 0], gt_traj[i, 1]), color='green')
            axs[1,0].set_title("Trajectory")
            axs[1,0].set_xlabel("X"); axs[1,0].set_ylabel("Y")
            axs[1,0].legend(); axs[1,0].grid(True); axs[1,0].axis('equal')

            # 子图4: Projected features (可选)
            axs[1,1].imshow(rgb_proj)
            axs[1,1].set_title("Projected Features PCA")
            axs[1,1].axis('off')

            plt.tight_layout()

            # 上传 wandb
            wandb.log({f"{logging_prefix}/visualization": [wandb.Image(fig)]})
            plt.close(fig)

########################
            visualize_idx = 1
            projected_feats = prediction['projected_feats'][visualize_idx,1].detach().cpu().numpy()
            dino_feats = prediction['dino_feats'][visualize_idx,1].detach().cpu().numpy()

            rgb_dino, bg_mask, thr = visualize_dino_pca_sklearn(dino_feats)      # (H,W,3)
            rgb_proj, bg_mask2, thr2 = visualize_dino_pca_sklearn(projected_feats)

            # 反归一化相机图像
            img_mean = [123.675, 116.28, 103.53]
            img_std = [58.395, 57.12, 57.375]
            camera = all_features['camera_feature'][visualize_idx,1].permute(1, 2, 0).cpu().numpy()
            camera = (camera * img_std + img_mean).astype(np.uint8)
            camera = camera[:, :, ::-1]

            ego_status = all_features['ego_status'][visualize_idx,-1].cpu().numpy()
            pred_traj = prediction['trajectory'][visualize_idx].detach().cpu().numpy()[:, :2]
            gt_traj   = all_targets['trajectory'][visualize_idx].cpu().numpy()[:, :2]

            # === 创建 2x2 子图 ===
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))

            # 子图1: Camera
            axs[0,0].imshow(camera)
            axs[0,0].set_title("Camera View")
            axs[0,0].axis('off')
            status_text = "\n".join([f"{i}: {v:.2f}" for i, v in enumerate(ego_status)])
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            axs[0,0].text(5, 20, status_text, fontsize=10, verticalalignment='top', bbox=props)

            # 子图2: Dino features (PCA RGB)
            axs[0,1].imshow(rgb_dino)
            axs[0,1].set_title("DINO Features PCA")
            axs[0,1].axis('off')

            # 子图3: Trajectory
            axs[1,0].plot(pred_traj[:, 0], pred_traj[:, 1], 'ro-', label="Predicted")
            axs[1,0].plot(gt_traj[:, 0], gt_traj[:, 1], 'go-', label="Ground Truth")
            for i in range(len(pred_traj)):
                axs[1,0].annotate(str(i), (pred_traj[i, 0], pred_traj[i, 1]), color='red')
                axs[1,0].annotate(str(i), (gt_traj[i, 0], gt_traj[i, 1]), color='green')
            axs[1,0].set_title("Trajectory")
            axs[1,0].set_xlabel("X"); axs[1,0].set_ylabel("Y")
            axs[1,0].legend(); axs[1,0].grid(True); axs[1,0].axis('equal')

            # 子图4: Projected features (可选)
            axs[1,1].imshow(rgb_proj)
            axs[1,1].set_title("Projected Features PCA")
            axs[1,1].axis('off')

            plt.tight_layout()

            # 上传 wandb
            wandb.log({f"{logging_prefix}/visualization1": [wandb.Image(fig)]})
            plt.close(fig)           
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if self.distill_feature:
            return self._step_distill(batch, "train")
        else:
            return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step_distill(batch, "val") if self.distill_feature else self._step(batch, "val")

    def test_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        features, targets = batch
        self.agent._rap_model.cache=True
        real_valid_mask = features['camera_valid']
        
        batch_size = real_valid_mask.shape[0]

        real_features = {k: v[real_valid_mask] if isinstance(v, torch.Tensor) else [x for x, m in zip(v, real_valid_mask) if m] for k, v in features.items() if k not in ['camera_valid','rendered_camera_feature']}

        real_targets = {k: v[real_valid_mask] if isinstance(v, torch.Tensor) else [x for x, m in zip(v, real_valid_mask) if m] for k, v in targets.items()}

        features.pop('camera_valid')
        features['camera_feature'] = features.pop('rendered_camera_feature')
        rendered_features = features
        rendered_targets = targets
  
        all_features = {}
        for k, v in rendered_features.items():
            all_features[k] = torch.cat([v, real_features[k]], dim=0) if isinstance(v, torch.Tensor) else v + real_features[k]

        all_targets = {}
        for k, v in rendered_targets.items():
            all_targets[k] = torch.cat([v, real_targets[k]], dim=0) if isinstance(v, torch.Tensor) else v + real_targets[k] 

        prediction = self.agent.forward(all_features,all_targets)
        
        if real_valid_mask.any():
            real_dino_feats = prediction[batch_size:]
            real_token_path = real_features['token_path']
            for i in range(len(real_token_path)):
                real_feature_save_path = real_token_path[i]+'_dino_feat_real.pt'
                #torch.save(real_dino_feats[i], real_feature_save_path)
                save_fp16_zstd(real_dino_feats[i], real_feature_save_path)

        rendered_dino_feats = prediction[:batch_size]
        rendered_token_path = rendered_features['token_path']
        for i in range(len(rendered_token_path)):
            rendered_feature_save_path = rendered_token_path[i]+'_dino_feat_rendered.pt'
            #torch.save(rendered_dino_feats[i], rendered_feature_save_path)
            save_fp16_zstd(rendered_dino_feats[i], rendered_feature_save_path)
        return 0

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        self.eval()
        features, targets = batch
        # real_valid_mask: boolean tensor indicating whether the frame is valid for real images.
        # This is needed since we also use rendered images for training.
        real_valid_mask = features['camera_valid']
        frame_name = features.pop('frame_name')
        real_features = {k: v[real_valid_mask] for k, v in features.items() if k not in ['camera_valid','rendered_camera_feature']}

        features.pop('camera_valid')
        features['camera_feature'] = features.pop('rendered_camera_feature')

        prediction = self.agent.forward(real_features,None,return_score=True)
        prediction['frame_name'] = frame_name
        
        return prediction

    # def on_validation_epoch_end(self):

    #     print('START TSNE')
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from sklearn.manifold import TSNE
    #     import wandb
    #     # 1) 拼接成 (N, d)
    #     real = np.concatenate(self.real_feat, axis=0) if len(self.real_feat) else np.empty((0, 1))
    #     synth = np.concatenate(self.synth_feat, axis=0) if len(self.synth_feat) else np.empty((0, 1))

    #     X = np.concatenate([real, synth], axis=0)
    #     y = np.concatenate([
    #         np.zeros(real.shape[0], dtype=int),
    #         np.ones(synth.shape[0], dtype=int)
    #     ], axis=0)

    #     # 2) t-SNE（cosine 距离更稳）
    #     n = X.shape[0]

    #     perplexity = min(30, max(5, n // 50))
    #     if perplexity >= n:
    #         perplexity = max(5, n // 3)

    #     tsne = TSNE(
    #         n_components=2,
    #         init="pca",
    #         perplexity=perplexity,
    #         learning_rate="auto",
    #         n_iter=500,
    #         metric="cosine",
    #         random_state=42,
    #         verbose=0,
    #     )
    #     X2 = tsne.fit_transform(X)

    #     # 3) 画图
    #     fig = plt.figure(figsize=(6, 6))
    #     ax = plt.gca()
    #     ax.scatter(X2[y == 0, 0], X2[y == 0, 1], s=6, alpha=0.65, label="Real")
    #     ax.scatter(X2[y == 1, 0], X2[y == 1, 1], s=6, alpha=0.65, label="Rasterized")
    #     ax.set_title(f"t-SNE: Real vs Rasterized)")
    #     ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    #     ax.legend(loc="best")
    #     plt.tight_layout()

    #     # 4) 用 WandB 上传
    #     wandb.log({"tsne/real_vs_synth": wandb.Image(fig), "epoch": self.current_epoch})

    #     plt.close(fig)

    #     # 5) 清缓存
    #     self.real_feat.clear()
    #     self.synth_feat.clear()


import numpy as np
from sklearn.decomposition import PCA


def visualize_dino_pca_sklearn(feats, eps=1e-8):
    """
    feats: (C,H,W) 的特征（np.ndarray 或 torch.Tensor），C>=3
    返回:
      rgb_uint8: (H,W,3) np.uint8，可视化结果
      bg_mask  : (H,W) bool，背景为 True
      thr      : float，Otsu 阈值（作用在 PC1 上）
    """
    # to numpy
    try:
        import torch
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().float().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(feats, dtype=np.float32)
    C, H, W = x.shape
    assert C >= 3, f"Need C>=3, got {C}"

    X = x.reshape(C, -1).T  # (N, C)

    pca = PCA(n_components=3)
    P_all = pca.fit_transform(X)              # (N,3)
    pc1 = P_all[:, 0]
    thr = _otsu(pc1)
    bg = pc1 < -10e8
    fg = ~bg

    if fg.sum() >= 3:
        X_fg = X[fg]
        P_fg = pca.fit_transform(X_fg)      # (N_fg,3)
        for i in range(3):
            ch = P_fg[:, i]
            mu, sd = ch.mean(), ch.std() + eps
            P_fg[:, i] = (ch - mu) / (sd**2) + 0.5
        P = np.zeros_like(P_all)
        P[fg] = P_fg
        P[bg] = 0.0
    else:
        P = P_all.copy()
        P[bg] = 0.0
        for i in range(3):
            ch = P[:, i]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + eps)
            P[:, i] = ch

    img = P.reshape(H, W, 3)
    return img, bg.reshape(H, W), float(thr)



