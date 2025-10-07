<div align="center">
<h1 align="center" style="
  font-size: 4rem;
  font-weight: 800;
  background: linear-gradient(90deg, #648FFF 35%, #DC267F 65%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
">
  RAP
</h1>
<h3>3D Rasterization Augmented End-to-End Planning</h3>

[Lan Feng](https://alan-lanfeng.github.io/)<sup>1,†</sup>, 
[Yang Gao](https://people.epfl.ch/yang.gao?lang=en/)<sup>1,†</sup>, 
[Éloi Zablocki](https://eloiz.github.io/)<sup>2,‡</sup>, 
[Quanyi Li](https://quanyili.github.io/), 
[Wuyang Li](https://wymancv.github.io/wuyang.github.io/)<sup>1,†</sup>, 
[Sichao Liu](https://sites.google.com/view/sichao-liu/home)<sup>1,†</sup>, 
[Matthieu Cord](https://cord.isir.upmc.fr/)<sup>2,3,‡</sup>, 
[Alexandre Alahi](https://people.epfl.ch/alexandre.alahi)<sup>1,†</sup>  

<sup>1</sup> EPFL, Switzerland <sup>2</sup> Valeo.ai, France <sup>3</sup> Sorbonne Université, France  

🏆 **1st Place** – [Waymo Open Dataset Vision-based E2E Driving Challenge](https://waymo.com/open/challenges/) (UniPlan entry)  
🏆 **#1 on Leaderboards** – [Waymo Open Dataset Vision-based E2E Driving](https://waymo.com/open/challenges/2025/e2e-driving/) & [NAVSIM v1/v2](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest) (RAP entry)  
🏆 **State-of-the-art** – [Bench2Drive](https://thinklab-sjtu.github.io/Bench2Drive/) benchmark

<a href="https://alan-lanfeng.github.io/RAP" target="_blank">
  <img src="https://img.shields.io/badge/_Project_Page-1d72b8?style=for-the-badge&logo=google-chrome&logoColor=white" height="40">
</a>
<a href="https://arxiv.org/abs/2510.04333" target="_blank">
  <img src="https://img.shields.io/badge/_Arxiv-1d72b8?style=for-the-badge&logo=google-chrome&logoColor=white" height="40">
</a>
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/xxxx.xxxxx) -->
</div>

---

🚗 **RAP (Rasterization Augmented Planning)** is a scalable data augmentation pipeline for end-to-end autonomous driving.  
It leverages lightweight **3D rasterization** to generate counterfactual recovery maneuvers and cross-agent views and **Raster-to-Real feature alignment** to bridge the sim-to-real gap in feature space, achieving **state-of-the-art performance** on multiple benchmarks.


---

## News
* **` Oct. 6th, 2025`:** Code released🔥!

---

# Getting Started

## Environment Setup
```bash
conda create -n rap python=3.9 
conda activate rap
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./nuplan-devkit
pip install -e .
```

## Set environment variable
set the environment variable based on where you place the PAD directory. 
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/rap_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/rap_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/rap_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/rap_workspace/dataset"
export Bench2Drive_ROOT="$HOME/rap_workspace/Bench2Drive"
```
## Data Processing
Note: This step generate data that shares the same format (with additional rasterized camera views) as [Navsim](https://github.com/autonomousvision/navsim).
- [Data Processing](process_data/README.md)

Please organize the generated data in the same way as [HERE](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).

## Navsim training and evaluation
1. cache training data and metric
```bash
# data caching ego vehicle
export OPENSCENE_DATA_ROOT="$HOME/rap_workspace/dataset"
python navsim/planning/script/run_dataset_caching.py \
agent=rap_agent \
dataset=navsim_dataset \
agent.config.trajectory_sampling.time_horizon=5 \
agent.config.cache_data=True \
train_test_split=navtrain \
train_test_split.scene_filter.has_route=false \
experiment_name=trainval_test \
worker.threads_per_node=64 \
cache_path=./cache/rap_ego \

# data caching cross-agent
export OPENSCENE_DATA_ROOT="$HOME/rap_workspace/dataset_aug"
python navsim/planning/script/run_dataset_caching.py \
agent=rap_agent \
dataset=navsim_dataset \
agent.config.trajectory_sampling.time_horizon=5 \
agent.config.cache_data=True \
train_test_split=navtrain \
train_test_split.scene_filter.has_route=false \
experiment_name=trainval_test \
worker.threads_per_node=64 \
cache_path=./cache/rap_aug \


# data caching recovery-oriented perturbation
export OPENSCENE_DATA_ROOT="$HOME/rap_workspace/dataset_perturbed"
python navsim/planning/script/run_dataset_caching.py \
agent=rap_agent \
dataset=navsim_dataset \
agent.config.trajectory_sampling.time_horizon=5 \
agent.config.cache_data=True \
train_test_split=navtrain \
train_test_split.scene_filter.has_route=false \
experiment_name=trainval_test \
worker.threads_per_node=64 \
cache_path=./cache/rap_perturbed \

# train metric caching
python navsim/planning/script/run_training_metric_caching.py \
train_test_split=navtrain \
cache.cache_path=./train_metric_cache \
worker.threads_per_node=32 \
```

3. train navsim model
```bash
python navsim/planing/script/run_training.py \
agent=rap_agent \
agent.config.pdm_scorer=True \
agent.config.distill_feature=True \
experiment_name=test \
train_test_split=navtrain \
split=trainval \
cache_path=./cache/rap_ego \
cache_path_perturbed=./cache/rap_perturbed \
cache_path_others=./cache/rap_aug \
use_cache_without_dataset=True \
force_cache_computation=False \
dataloader.params.batch_size=64 \
dataset=navsim_dataset \
agent.config.trajectory_sampling.time_horizon=5
```
4. test navsim model

Please refer to [Navsim](https://github.com/autonomousvision/navsim) for more details.

## Waymo Finetuning
1. Download Waymo E2E Driving Dataset: https://waymo.com/open/download/
2. Dataset caching:
```bash
python navsim/planning/script/run_waymo_dataset_caching.py \
agent=rap_agent \
dataset=waymo_dataset \
dataset.include_val=False \
experiment_name=trainval_test \
train_test_split=navtrain \
worker.threads_per_node=64 \
cache_path=./cache/rap_waymo \
waymo_raw_path=
```
3. finetune pretrained model on Waymo
```bash
python navsim/planing/script/run_training.py \
agent=rap_agent \
agent.config.pdm_scorer=False \
agent.config.distill_feature=False \
experiment_name=waymo_finetune \
train_test_split=navtrain  \
train_test_split.scene_filter=navtrain \
split=trainval   \
trainer.params.max_epochs=20 \
cache_path=./cache/rap_waymo \
use_cache_without_dataset=True  \
force_cache_computation=False \
dataloader.params.batch_size=16 \
agent.config.trajectory_sampling.time_horizon=5 \
dataset=waymo_dataset \
agent.checkpoint_path=$CHECKPOINT \
agent.lr=1e-5 \
```

4. Leaderboard submission
```bash
mkdir waymo_submission
#Put all the ckpt files in the same directory /waymo_submission so that it can run ensembling

Python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_waymo_submission.py \
        agent=navsim_agent \
        agent.config.pdm_scorer=False \
        agent.config.distill_feature=False \
        experiment_name=ldb \
        train_test_split=navtrain  \
        train_test_split.scene_filter=navtrain \
        split=trainval \
        trainer.params.max_epochs=20 \
        cache_path="./cache/rap_waymo" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        dataloader.params.batch_size=8 \
        agent.config.trajectory_sampling.time_horizon=5 \
        dataset=waymo_dataset \
        agent.checkpoint_path=./waymo_submission/1.ckpt
```
## Checkpoints
> Results on NAVSIM

| Method | Model Size | Backbone | PDMS | Weight Download |
| :---: | :---: | :---: | :---:  | :---: |
| RAP-DINO | 888M | DINOv3-h16+ | 93.8 | [Hugging Face](https://huggingface.co/Lanl11/RAP_ckpts/tree/main) |

> Results on Waymo

| Method | Model Size | Backbone | RFS | Weight Download |
| :---: | :---: | :---: | :---:  | :---: |
| RAP-DINO | 888M | DINOv3-h16+ | 8.04 | [Hugging Face](https://huggingface.co/Lanl11/RAP_ckpts/tree/main) |


## Citation

```bibtex
@misc{feng2025rap3drasterizationaugmented,
      title={RAP: 3D Rasterization Augmented End-to-End Planning}, 
      author={Lan Feng and Yang Gao and Eloi Zablocki and Quanyi Li and Wuyang Li and Sichao Liu and Matthieu Cord and Alexandre Alahi},
      year={2025},
      eprint={2510.04333},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.04333}, 
}
```