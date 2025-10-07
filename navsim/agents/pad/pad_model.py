from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.image_encoder import ImgEncoder
from .bevformer.transformer_decoder import MLP

class LambdaScheduler:
    def __init__(self, gamma=10.0):
        self.gamma = gamma
    def __call__(self, progress: float) -> float:
        # progress ∈ [0,1]
        return 2.0 / (1.0 + torch.exp(torch.tensor(-self.gamma * progress))) - 1.0

# —— Gradient Reversal —— #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None  # 对特征梯度乘以 -λ

def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)

# —— 聚合 + 判别器 —— #
class DomainClassifier(nn.Module):
    """
    接口：
      forward(feat, lambd, agg='mean') 
        feat: (num_cam, num_patch, B, D) 或 (B, D)
        lambd: GRL 系数（训练时 >0；eval 时可设 0）
        agg: 'mean' | 'max' | 'linproj'  (从多相机/多patch聚合到 per-sample)
    输出：
      logits: (B,)  -> 用 BCEWithLogitsLoss 与 domain labels (B,) 对齐
    """
    def __init__(self, d_in, hidden=512, agg='mean'):
        super().__init__()
        self.agg = agg
        if agg == 'linproj':
            # 可学习的线性聚合：把 cam*patch 维度做线性投影
            self.linproj = nn.Linear(d_in, d_in)
        self.classifier = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # 二分类（source=0, target=1）
        )

    def _aggregate(self, feat):
        # 支持两种输入： (C,P,B,D) 或 (B,D)
        if feat.dim() == 4:
            C, P, B, D = feat.shape
            # 先把 cam/patch 合成 token 维度：(C*P, B, D)
            feat = feat.reshape(C*P, B, D)
            if self.agg == 'mean':
                feat = feat.mean(dim=0)         # (B, D)
            elif self.agg == 'max':
                feat = feat.max(dim=0).values   # (B, D)
            elif self.agg == 'linproj':
                # 先线性变换每个 token，再平均
                feat = self.linproj(feat)        # (C*P, B, D)
                feat = feat.mean(dim=0)         # (B, D)
            else:
                raise ValueError(f'Unknown agg: {self.agg}')
            return feat
        elif feat.dim() == 2:
            return feat  # 已是 (B, D)
        else:
            raise ValueError(f'Expect feat dim 2 or 4, got {feat.shape}')

    def forward(self, feat, lambd: float):
        feat_agg = self._aggregate(feat)        # (B, D)
        feat_rev = grad_reverse(feat_agg, lambd)
        logits = self.classifier(feat_rev).squeeze(-1)  # (B,)
        return logits

class PadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.trajectory_sampling.num_poses
        self.state_size=3

        self._backbone = ImgEncoder(config)

        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(11, config.tf_d_model)

        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)

        ref_num=config.ref_num

        shared_refiner=Traj_refiner(config)

        self._trajectory_head=nn.ModuleList([shared_refiner for _ in range(ref_num) ] )

        self.scorer = Scorer(config)
        self.domain_classifier = DomainClassifier(config.tf_d_model)
        self.lambda_scheduler = LambdaScheduler(gamma=10.0)
        self.b2d=config.b2d

    def forward(self, features: Dict[str, torch.Tensor],targets: Dict[str, torch.Tensor],return_score=False) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"][:,-1]
        camera_feature: torch.Tensor = features["camera_feature"]

        batch_size = ego_status.shape[0]

        if self.b2d:
            ego_status[:,1:3]=0

        image_feature = self._backbone(camera_feature,img_metas=features)  # b,64,64,64

        output={}

        ego_feature=self.hist_encoding(ego_status)[:,None]

        bev_feature =ego_feature+self.init_feature.weight[None]

        proposal_list = []

        for i, refine in enumerate(self._trajectory_head):
            bev_feature, proposal_list = refine(bev_feature, proposal_list,image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, bev_feature)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels
        output["bev_feature"]=image_feature[0].permute(2,0,1,3)

        lambda_ = self.lambda_scheduler(self.progress)
        feat = image_feature[0][[1]]   # 假设 shape = [B, D]
        feat_grad = feat[:,:,:16].detach()          
        feat_no_grad = feat[:,:,16:]   
        mixed_feat = torch.cat([feat_grad, feat_no_grad], dim=2)
        domain_logits = self.domain_classifier(mixed_feat, lambd=lambda_)  # (B,)
        output["domain_logits"] = domain_logits

        if pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        if return_score:
            output["trajectory"] = proposals
            output["score"] = pdm_score
        else:
            token = torch.argmax(pdm_score, dim=1)
            trajectory = proposals[torch.arange(batch_size), token]
            output["trajectory"] = trajectory
            output["pdm_score"] = pdm_score

        return output



