import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import re
import numpy as np 
from scipy.interpolate import CubicSpline

from typing import Dict, List, Tuple
import numpy as np


_THRESHOLD_TIME_SECONDS = np.array([3, 5], dtype=np.int64)
_BASE_THRESHOLDS = np.array([1.0, 1.8], dtype=np.float64)
_MINIMUM_SCORE_OUTSIDE_TRUST_REGION = 4.0


def get_lat_lng_thresholds(
    init_speed: np.ndarray,  # [B]
    lat_lng_threshold_multipliers: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
  """Get lateral and longitudinal thresholds."""
  # Set and scale thresholds with the initial velocity
  lat_threshold_multiplier, lng_threshold_multiplier = (
      lat_lng_threshold_multipliers
  )
  lat_thresholds = _BASE_THRESHOLDS * lat_threshold_multiplier  # [2]
  lng_thresholds = _BASE_THRESHOLDS * lng_threshold_multiplier  # [2]
  scale_by_init_speed = np.clip(
      0.5 + 0.5 * (init_speed - 1.4) / (11 - 1.4), 0.5, 1.0
  )  # [B]
  lat_thresholds = scale_by_init_speed[..., None] * lat_thresholds  # [B, 2]
  lng_thresholds = scale_by_init_speed[..., None] * lng_thresholds  # [B, 2]

  return lat_thresholds, lng_thresholds


def process_rater_specified_trajectories(
    trajectory_batches: List[List[np.ndarray]],
    trajectory_labels_batches: List[np.ndarray],
    target_num_waypoints: int,
    target_num_trajectories_per_batch: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """Processes rater-specified trajectories by truncating or padding.

  Args:
    trajectory_batches: A list where each element is a batch of trajectories. A
      trajectory is represented as a NumPy array of waypoints.
    trajectory_labels_batches: A list where each element is a NumPy array of
      labels corresponding to a batch of trajectories.
    target_num_waypoints: The fixed number of waypoints each trajectory should
      have after processing. Trajectories longer than this will be truncated,
      and shorter ones will be padded by duplicating their last waypoint.
    target_num_trajectories_per_batch: The fixed number of trajectories each
      batch should have after processing. Batches with more trajectories will be
      truncated, and those with fewer will be padded by duplicating their last
      trajectory (and its label).

  Returns:
    A tuple containing:
      - processed_trajectory_batches: The processed trajectory batches after
        truncation or padding them at both the trajectory and waypoint levels.
      - processed_labels_batches: The corresponding processed label batches.

  Raises:
    ValueError: If the number of trajectory batches and label batches do not
    match, or if within a batch, the number of trajectories and labels do not
    match.
  """
  if len(trajectory_batches) != len(trajectory_labels_batches):
    raise ValueError(
        'The number of trajectory batches and label batches must be the same.'
    )

  processed_trajectory_batches_list = []
  processed_labels_batches_list = []

  # Iterate over each batch of trajectories and their corresponding labels
  for i in range(len(trajectory_batches)):
    current_trajectory_batch = trajectory_batches[i]
    current_labels_batch = trajectory_labels_batches[i]
    if len(current_trajectory_batch) != len(current_labels_batch):
      raise ValueError(
          'In each batch, the number of trajectories and labels must be the'
          ' same.'
      )

    # --- Step 1: Truncate or pad the number of trajectories in the
    # current batch ---
    num_trajectories_in_batch = len(current_trajectory_batch)

    if num_trajectories_in_batch > target_num_trajectories_per_batch:
      # Truncate trajectories and labels
      processed_batch_trajectories = current_trajectory_batch[
          :target_num_trajectories_per_batch
      ]
      processed_batch_labels = current_labels_batch[
          :target_num_trajectories_per_batch
      ]
    elif num_trajectories_in_batch < target_num_trajectories_per_batch:
      # Pad trajectories and labels by duplicating the last element
      num_to_pad = target_num_trajectories_per_batch - num_trajectories_in_batch
      padding_trajectories = [current_trajectory_batch[-1]] * num_to_pad
      padding_labels = [current_labels_batch[-1]] * num_to_pad

      processed_batch_trajectories = (
          current_trajectory_batch + padding_trajectories
      )
      processed_batch_labels = current_labels_batch.tolist() + padding_labels
    else:
      # Number of trajectories already matches the target
      processed_batch_trajectories = current_trajectory_batch
      processed_batch_labels = current_labels_batch

    # --- Step 2: Truncate or pad waypoints for each trajectory
    # in the processed batch ---
    final_trajectories_for_current_batch = []

    for trajectory in processed_batch_trajectories:
      num_waypoints = len(trajectory)
      if num_waypoints > target_num_waypoints:
        # Truncate waypoints
        processed_trajectory = trajectory[:target_num_waypoints]
      elif num_waypoints < target_num_waypoints:
        # Pad waypoints by duplicating the last waypoint
        last_waypoint = trajectory[-1]
        padding_waypoints = np.array(
            [last_waypoint] * (target_num_waypoints - num_waypoints)
        )
        processed_trajectory = np.concatenate(
            (trajectory, padding_waypoints), axis=0
        )

      else:
        # Number of waypoints already matches the target
        processed_trajectory = trajectory
      final_trajectories_for_current_batch.append(processed_trajectory)

    processed_trajectory_batches_list.append(
        np.array(final_trajectories_for_current_batch)
    )

    # Labels are already processed at the batch level (number of trajectories)
    # No per-label processing is typically needed unless labels have internal
    # structure to pad/truncate
    if isinstance(processed_batch_labels, np.ndarray):
      processed_labels_batches_list.append(processed_batch_labels)
    else:  # if it became a list during padding
      processed_labels_batches_list.append(np.array(processed_batch_labels))

  # Convert lists of batches to NumPy arrays

  final_processed_trajectories = np.array(processed_trajectory_batches_list)
  final_processed_labels = np.array(processed_labels_batches_list)

  return final_processed_trajectories, final_processed_labels


def get_rater_feedback_score(
    inference_trajectories: np.ndarray,  # [B, I, T, 2]
    inference_probs: np.ndarray,  # [B, I]
    rater_specified_trajectories: List[
        List[np.ndarray]
    ],  # [[T1, 2], [T2, 2], ...], ...]
    rater_feedback_labels: List[np.ndarray],  #  [[P1], [P2], ...]
    init_speed: np.ndarray,  # [B]
    lat_lng_threshold_multipliers: Tuple[float, float] = (1.0, 4.0),
    decay_factor: float = 0.1,
    frequency: int = 4,
    length_seconds: int = 5,
    default_num_of_rater_specified_trajectories: int = 3,
    output_trust_region_visualization: bool = False,
    minimum_score_outside_trust_region: float = _MINIMUM_SCORE_OUTSIDE_TRUST_REGION,
) -> Dict[str, np.ndarray]:
  """Get rater feedback score (https://waymo.com/open/challenges/2025/e2e-driving/).

  Notations:
  - B: batch size
  - I: number of inference trajectories
  - P: number of rater-specified trajectories
  - T: number of timesteps

  Args:
    inference_trajectories: An array of inference trajectories with shape [B, I,
      T, 2]
    inference_probs: An array of inference probabilities with shape [B, I]
    rater_specified_trajectories: An array of rater-specified trajectories with
      shape [B, P, T, 2]
    rater_feedback_labels: An array of rater feedback labels (scores between 0
      and 10, both inclusive) with shape [B, P]
    init_speed: A batch of initial velocities with shape [B]
    lat_lng_threshold_multipliers: A tuple of latitude and longitude threshold
      multipliers with shape [2]
    decay_factor: A scalar score decay factor outside the trust region
    frequency: The frequency (Hz) of trajectories to be considered.
    length_seconds: The length (seconds) of trajectories to be considered.
    default_num_of_rater_specified_trajectories: The default number of rater
      specified trajectories to be used.
    output_trust_region_visualization: Whether to output trust region
      visualization.
    minimum_score_outside_trust_region: The minimum score for inference
      trajectories that are not fully within the trust region.

  Returns:
    A dictionary of final rater feedback score and output for visualization.
  """
  # We first process the rater-specified trajectories and labels by
  # truncating or padding them to the same length.
  # After processing, the shape of rater_specified_trajectories is
  # [B, P, T, 2], and the shape of rater_feedback_labels is [B, P].
  rater_specified_trajectories, rater_feedback_labels = (
      process_rater_specified_trajectories(
          rater_specified_trajectories,
          rater_feedback_labels,
          target_num_waypoints=length_seconds * frequency,
          target_num_trajectories_per_batch=default_num_of_rater_specified_trajectories,
      )
  )

  if inference_trajectories.shape[-2] != rater_specified_trajectories.shape[-2]:
    raise ValueError(
        'Inference and rater-specified trajectories must have the same number'
        ' of timesteps.'
    )

  if (
      inference_trajectories.shape[-2]
      < _THRESHOLD_TIME_SECONDS.max() * frequency
  ):
    raise ValueError(
        'Inference trajectories must have at least'
        f' {_THRESHOLD_TIME_SECONDS.max()} timesteps.'
    )

  # Make rater-specified trajectories to include the origin
  padded_rater_specified_trajectories = np.pad(
      rater_specified_trajectories,
      ((0, 0), (0, 0), (1, 0), (0, 0)),
      constant_values=0,
  )  # [B, P, T + 1, 2]

  # Compute displacement vectors
  displacement_vectors = (
      padded_rater_specified_trajectories[..., 1:, :]
      - padded_rater_specified_trajectories[..., :-1, :]
  )  # [B, P, T, 2]

  # Get unnormalized directions
  lng_directions = displacement_vectors  # [B, P, T, 2]

  # When displacement is zero, which means the vehicle did not move, we bring
  # longitudinal directions from the previous timestep.
  lng_magnitudes = np.linalg.norm(lng_directions, axis=-1)  # [B, P, T]

  # At the first timestep, we set the longitudinal directions to be (1, 0).
  # This is because the vehicle coordinate is used.
  lng_directions[..., 0, 0] = np.where(
      lng_magnitudes[..., 0] == 0,
      1,
      lng_directions[..., 0, 0],
  )  # x-axis
  lng_directions[..., 0, 1] = np.where(
      lng_magnitudes[..., 0] == 0,
      0,
      lng_directions[..., 0, 1],
  )  # y-axis

  # For the rest of the timesteps, we bring the longitudinal directions from the
  # previous timestep.
  for t in range(1, lng_directions.shape[2]):
    lng_directions[..., t, 0] = np.where(
        lng_magnitudes[..., t] == 0,
        lng_directions[..., t - 1, 0],
        lng_directions[..., t, 0],
    )  # x-axis
    lng_directions[..., t, 1] = np.where(
        lng_magnitudes[..., t] == 0,
        lng_directions[..., t - 1, 1],
        lng_directions[..., t, 1],
    )  # y-axis

  # Lateral directions are 90-degree counterclockwise rotation of longitudinal
  # directions, i.e., (x_new, y_new) = (-y, x)
  lat_directions = np.stack(
      [lng_directions[..., 1] * -1, lng_directions[..., 0]], axis=-1
  )  # [B, P, T, 2]

  # Normalize directions
  lng_directions = lng_directions / np.linalg.norm(
      lng_directions, axis=-1, keepdims=True
  )  # [B, P, T, 2]
  lat_directions = lat_directions / np.linalg.norm(
      lat_directions, axis=-1, keepdims=True
  )  # [B, P, T, 2]

  # Get longitudinal and lateral distances from rater-specified trajectories
  rater_specified_to_inference_vectors = (
      inference_trajectories[..., None, :, :, :]
      - rater_specified_trajectories[..., None, :, :]
  )  # [B, 1, I, T, 2] - [B, P, 1, T, 2] --> [B, P, I, T, 2]
  lng_projections = np.sum(
      lng_directions[..., None, :, :] * rater_specified_to_inference_vectors,
      axis=-1,
  )  # [B, P, I, T], directions are broadcasted to the inference trajectories
  lat_projections = np.sum(
      lat_directions[..., None, :, :] * rater_specified_to_inference_vectors,
      axis=-1,
  )  # [B, P, I, T], directions are broadcasted to the inference trajectories
  lng_distances = np.abs(lng_projections)  # [B, P, I, T]
  lat_distances = np.abs(lat_projections)  # [B, P, I, T]

  # Filter distances at 3 and 5 seconds
  selected_indices = _THRESHOLD_TIME_SECONDS * frequency - 1
  lng_distances = lng_distances[..., selected_indices]  # [B, P, I, 2]
  lat_distances = lat_distances[..., selected_indices]  # [B, P, I, 2]

  lat_thresholds, lng_thresholds = get_lat_lng_thresholds(
      init_speed, lat_lng_threshold_multipliers
  )

  outputs = {}

  # ---------------------------------------------------------------------------
  # Visualization
  # ---------------------------------------------------------------------------
  if output_trust_region_visualization:
    center_x = rater_specified_trajectories[..., selected_indices, :][
        ..., 0
    ]  # [B, P, T (=2)]
    center_y = rater_specified_trajectories[..., selected_indices, :][
        ..., 1
    ]  # [B, P, T (=2)]
    width = 2 * lng_thresholds  # [B, 2]
    height = 2 * lat_thresholds  # [B, 2]
    angle = np.degrees(
        np.arctan2(
            displacement_vectors[..., selected_indices, :][..., 1],
            displacement_vectors[..., selected_indices, :][..., 0],
        )
    )  # [B, P, T (=2)]

    outputs.update({
        'trust_region_center_x': center_x,  # [B, I, 2]
        'trust_region_center_y': center_y,  # [B, I, 2]
        'trust_region_width': width,  # [B, 2]
        'trust_region_height': height,  # [B, 2]
        'trust_region_angle': angle,  # [B, I, 2]
    })

  # ---------------------------------------------------------------------------
  # Hard matching with decaying
  # ---------------------------------------------------------------------------

  # Normalize distances with thresholds
  normalized_lng_distances = (
      lng_distances / lng_thresholds[..., None, None, :]
  )  # [B, P, I, 2]
  normalized_lat_distances = (
      lat_distances / lat_thresholds[..., None, None, :]
  )  # [B, P, I, 2]

  # Pick the maximum of the two normalized distances
  normalized_distances = np.maximum(
      normalized_lng_distances, normalized_lat_distances
  )  # [B, P, I, 2]

  # Mask to indicate if the inference trajectory is fully within the trust
  # region, i.e., distance from trajectory i is near any rated trajectory p.
  # For inferences not fully within the trust region, scores are clipped to
  # `minimum_score_outside_trust_region` during score computation below.
  # [B, P, I, 2] -> [B, I]
  is_fully_within_trust_region = np.any(
      np.all(normalized_distances <= 1.0, axis=3), axis=1)
  outputs['is_fully_within_trust_region'] = (
      is_fully_within_trust_region  # [B, I]
  )

  # Make scores flat within the trust region.
  exponent = np.maximum(normalized_distances - 1.0, 0.0)
  decay = decay_factor**exponent
  # Scores between every inference i and rated trajectory p along x,y axes.
  rater_feedback_scores_per_axis_pairwise = (
      rater_feedback_labels[..., None, None] * decay
  )  # [B, P, I, 2]

  # Scores for each inference trajectory along x,y axes.
  # Each inference trajectory is assigned a score based on its best match with
  # a rated trajectory.
  rater_feedback_score_per_axis_per_inference = np.amax(
      rater_feedback_scores_per_axis_pairwise, axis=1
  )  # [B, I, 2]

  # Scores for each inference trajectory averaged over x,y axes.
  rater_feedback_score_per_inference = np.mean(
      rater_feedback_score_per_axis_per_inference,
      axis=-1,
  )  # [B, I]

  # Clip scores for inferences not fully within the trust region.
  rater_feedback_score_per_inference[~is_fully_within_trust_region] = (
      np.maximum(
          minimum_score_outside_trust_region,
          rater_feedback_score_per_inference[~is_fully_within_trust_region],
      )
  )  # [B, I]

  # Weighted sum over scores for each inference trajectory.
  rater_feedback_score = np.sum(
      rater_feedback_score_per_inference * inference_probs, axis=-1
  )  # [B]
  outputs['rater_feedback_score_per_inference'] = rater_feedback_score_per_inference
  outputs['rater_feedback_score'] = rater_feedback_score  # [B]
  # Updated the truncated or padded rater-specified trajectories.
  # [B, P, T, 2]
  outputs['rater_specified_trajectories'] = rater_specified_trajectories
  # Updated the truncated or padded rater feedback labels.
  # [B, P]
  outputs['rater_feedback_labels'] = rater_feedback_labels

  return outputs

def interpolate_trajectory(current_prediction_trajectories):
                        
    x = current_prediction_trajectories[:,0]
    y = current_prediction_trajectories[:,1]
    even_frames = np.arange(2, 21, 2)  # [2, 4, ..., 20]
    full_x = [0.0]
    full_y = [0.0]
    full_frames = [0]
    full_x.extend(x.tolist())
    full_y.extend(y.tolist())
    full_frames.extend(even_frames.tolist())
    all_frames = np.arange(21)
    interp_x = np.interp(all_frames, full_frames, full_x)[1:]
    interp_y = np.interp(all_frames, full_frames, full_y)[1:]
    interp_xy = np.stack([interp_x, interp_y], axis=-1)
    return interp_xy


def cubic_resample_traj(traj, fs_in=1.0, fs_out=4.0, bc_type='natural', include_endpoint=True):
    """
    用三次样条将二维轨迹从较低采样率插值到较高采样率。

    Parameters
    ----------
    traj : array-like, shape (N, 2)
        原始轨迹点 (x, y)，假设等间隔采样。
    fs_in : float, default=1.0
        原始采样率 (Hz)；1 Hz 表示相邻点间隔 1 秒。
    fs_out : float, default=4.0
        目标采样率 (Hz)；4 Hz 表示每 0.25 秒一个点。
    bc_type : str or 2-tuple, default='natural'
        CubicSpline 边界条件。同 SciPy `CubicSpline`。
    include_endpoint : bool, default=True
        若 True，输出时间序列包含最后一个原始时间点；否则不含。

    Returns
    -------
    t_out : ndarray, shape (M,)
        新时间轴（秒）。
    traj_out : ndarray, shape (M, 2)
        插值后的轨迹点。

    Notes
    -----
    - 假设原始数据是 **匀速采样** 的；如果不是，请传入你自己的时间轴并改写函数。
    - 如果需要速度、加速度，也可直接调用样条对象的 `.derivative()`。
    """
    traj_1hz = np.asarray(traj, dtype=float)
    if traj_1hz.ndim != 2 or traj_1hz.shape[1] != 2:
        raise ValueError("traj_1hz 必须是形状 (N,2) 的数组。")

    # 构造输入时间轴：隐藏点 + 已知 1Hz 点
    N = traj_1hz.shape[0]
    t_known = 1.0 + np.arange(N) / fs_in
    t_in = np.concatenate(([0.0], t_known))
    traj_in = np.vstack([np.asarray((0,0), dtype=float), traj_1hz])

    # 输出时间轴
    dt_out = 1.0 / fs_out
    t_out = np.arange(0.25, 5.0 + 1e-12, dt_out)

    # 构造样条（默认允许外推；如果不想外推，可传 extrapolate=False）
    cs = CubicSpline(t_in, traj_in, axis=0, bc_type=bc_type, extrapolate=False)

    # 检查哪些输出点在范围内
    in_range = (t_out >= t_in[0]) & (t_out <= t_in[-1])
    traj_out = np.full((t_out.shape[0], 2), np.nan, dtype=float)
    traj_out[in_range] = cs(t_out[in_range])

    # 如果你宁愿强制计算越界点（外推），可以改成：
    # traj_out = cs(t_out)  # 并把上面的 extrapolate=False 改为 True

    return traj_out

# ---------- 1. 通用加载函数 ---------- #
def load_json(path) -> List[Dict[str, Any]]:
    """加载标准 JSON（整体是列表的结构）"""
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    result = []
    for smp in samples:
        meta = smp.get("meta_info", {})
        # 找到 assistant 生成的预测文本（若有）
        pred_txt = next(
            (m["content"] for m in smp.get("messages", [])
             if m.get("role") == "assistant" and "Predicted" in m.get("content", "")),
            None
        )
        result.append(
            {
                "id": smp.get("id"),
                "initial_speed": meta.get("initial_speed"),
                "trajectory": meta.get("trajectory"),      # 轨迹点列表
                "rfs_scores": meta.get("rfs_scores"),
                "rfs_trajs": meta.get("rfs_trajs"),
                "rfs_len": meta.get("rfs_len"),
                "predicted_waypoints": pred_txt,
            }
        )
    return result
    

def load_jsonl(path) -> List[Dict[str, Any]]:
    """加载逐行 JSONL，返回每一行对应的字典"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:                       # 跳过空行
                records.append(json.loads(line))
    result = []
    for rec in records:
        result.append(
            rec.get("predict")
        )
    return result
    

def extract_trajs(pred_text):

    pattern = r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    matches = re.findall(pattern, pred_text)

    if not matches:
        print('warning: no match found')
        return np.zeros((5,2))

    # 转成浮点，然后堆叠为 (N, 2) 形状
    coords = np.array([[float(x), float(y)] for x, y in matches], dtype=float)

    return coords

def calculate_rfs(samples_info, jsonl_info):
    predicted_trajs = [extract_trajs(pred_text) for pred_text in jsonl_info]

    rater_specified_trajectories_list = []
    rater_scores_list = []
    prediction_trajectories_list = []
    prediction_probabilities_list = []
    initial_speed_list = []
    for i in range(len(samples_info)):
        rfs_trajs = samples_info[i]['rfs_trajs']
        rfs_len = samples_info[i]['rfs_len']
        rfs_scores = samples_info[i]['rfs_scores']
        initial_speed = samples_info[i]['initial_speed']
        initial_speed_list.append(initial_speed)
        prediction_trajectories = predicted_trajs[i]
        rfs_trajs_list = [rfs_trajs[k][:rfs_len[k]] for k in range(len(rfs_trajs))]

        interp_prediction_trajectories = cubic_resample_traj(prediction_trajectories) if len(prediction_trajectories)==5 else interpolate_trajectory(prediction_trajectories)
        rater_specified_trajectories_list.append(rfs_trajs_list)
        rater_scores_list.append(rfs_scores)
        prediction_trajectories_list.append(interp_prediction_trajectories[None])
        prediction_probabilities_list.append(np.ones(1))


    rater_feedback_metrics = get_rater_feedback_score(
        np.stack(prediction_trajectories_list),
        np.stack(prediction_probabilities_list),
        rater_specified_trajectories_list,
        rater_scores_list,
        np.stack(initial_speed_list),
        frequency=4,  # Default is 4.
        length_seconds=5, # Default predict 5 seconds.
        output_trust_region_visualization=False,
    )
    score = rater_feedback_metrics['rater_feedback_score']#.mean()
    return score

# ---------- 3. 演示入口 ---------- #
if __name__ == "__main__":
    # 假设文件名：samples.json / records.jsonl   —— 你可自行更改
    samples_path = "/home/fenglan/DiffusionDrive/MySubmission/val_merged.json"
    jsonl_path    = "/home/fenglan/DiffusionDrive/MySubmission/qwen_annotated.jsonl"

    samples_info = load_json(samples_path)
    jsonl_info   = load_jsonl(jsonl_path)

    print(calculate_rfs(samples_info, jsonl_info))
    


