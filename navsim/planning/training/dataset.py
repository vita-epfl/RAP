from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from tqdm import tqdm
import numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
import tensorflow as tf
import shutil
import cv2
from scipy.signal import savgol_filter

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import json
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
from navsim.common.dataclasses import AgentInput, EgoStatus, Cameras, Lidar, Camera
logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        split: str = "train",
        use_cache_list=False
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]
        self.split = split
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self.score_mask = True
        cache_file = self._cache_path / f"valid_cache_{self.split}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        

        if use_cache_list:
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    print(f"[Cache] Loading from {cache_file}")
                    self._valid_cache_paths: Dict[str, Path] = pickle.load(f)
            else:
                print("[Cache] Cache not found, building...")
                self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
                    cache_path=self._cache_path,
                    feature_builders=self._feature_builders,
                    target_builders=self._target_builders,
                    log_names=self.log_names,
                )
                with open(cache_file, "wb") as f:
                    pickle.dump(self._valid_cache_paths, f)
                    print(f"[Cache] Saved to {cache_file}")
        else:
            self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
                cache_path=self._cache_path,
                feature_builders=self._feature_builders,
                target_builders=self._target_builders,
                log_names=self.log_names,
            )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)
        if self.score_mask:
            targets['score_mask']=torch.tensor(True)
        else:
            targets['score_mask']=torch.tensor(False)
        return (features, targets)


class WaymoCacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,split: str
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"

        self._cache_path = Path(cache_path)/split
        self.split = split
        self.tokens = [token for token in self._cache_path.iterdir()]

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    def _load_scene_with_token(self, token_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        features: Dict[str, torch.Tensor] = {}
        data_dict_path = token_path / ("features.gz")
        data_dict = load_feature_target_from_pickle(data_dict_path)
        features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        data_dict_path = token_path / ("targets.gz")
        data_dict = load_feature_target_from_pickle(data_dict_path)
        targets.update(data_dict)

        return (features, targets)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)

class NavsimQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)

        image_save_path = os.path.join(self._cache_path,'trainval', metadata.initial_token)
        os.makedirs(image_save_path, exist_ok=True)
        qa = get_qa_navsim(agent_input,data_dict['trajectory'].numpy(),image_save_path)

        qa_path = os.path.join(image_save_path+".json")
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=2, ensure_ascii=False)

        annotation = get_annotation(agent_input,data_dict['trajectory'].numpy(),image_save_path)
        os.makedirs(self._cache_path/'trainval_annotation', exist_ok=True)
        annotation_path = os.path.join(self._cache_path/'trainval_annotation', metadata.initial_token+'.json')
        with open(annotation_path, "w") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)


    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)


class WaymoDataset(torch.utils.data.Dataset):

    def __init__(self,
        file_list: List[str],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        submission_frames,
        include_val=False,
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        ):
        super().__init__()

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._submission_frames = submission_frames
        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self.raw_file_list = file_list
        self._force_cache_computation = force_cache_computation
        self.include_val = include_val
        if self._cache_path is not None:
            self.cache_dataset()

    def cache_dataset(self):
                
        if os.path.exists(self._cache_path) and self._force_cache_computation is False:
            return
        else:
            for tfrecord_path in tqdm(self.raw_file_list):
                if self.include_val and 'val' in str(tfrecord_path):
                    self.process_one_record(tfrecord_path,'training')

                data_split = tfrecord_path.name.split('_')[0]
                self.process_one_record(tfrecord_path,data_split)

    def process_one_record(self, tfrecord_path,data_split):
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        dataset_iter = dataset.as_numpy_iterator()

        for cnt, bytes_example in enumerate(dataset_iter):
            e2eframe = wod_e2ed_pb2.E2EDFrame()
            e2eframe.ParseFromString(bytes_example)
            dirs = os.path.join(self._cache_path, data_split)
            os.makedirs(dirs, exist_ok=True)
            self.process(dirs, e2eframe)

    def process(self, dirs, e2eframe):
        frame_token,frame_index = e2eframe.frame.context.name.split('-')
        frame_index = int(frame_index)
        split = dirs.split('/')[-1]

        if 'val' in split:
            if not (len(e2eframe.preference_trajectories) > 0 and e2eframe.preference_trajectories[0].preference_score != -1):
                return  
            current_rater_trajs = []
            current_rater_scores = []
            current_rater_len = []
            rater_specified_trajs_and_scores_i = e2eframe.preference_trajectories
            for j in range(len(rater_specified_trajs_and_scores_i)):
                rater_traj =  np.stack(
                        [
                            rater_specified_trajs_and_scores_i[j].pos_x,
                            rater_specified_trajs_and_scores_i[j].pos_y,
                        ],
                        axis=-1,
                    )
                rater_len = rater_traj.shape[0]
                rater_traj = np.pad(rater_traj, ((0, 21 - rater_len), (0, 0)), 'constant', constant_values=0)
                current_rater_trajs.append(rater_traj)
                current_rater_len.append(rater_len)
                current_rater_scores.append(rater_specified_trajs_and_scores_i[j].preference_score)


            current_rater_scores = np.array(current_rater_scores).astype(np.float32)
            current_rater_trajs = np.array(current_rater_trajs).astype(np.float32)
            current_rater_len = np.array(current_rater_len).astype(np.int32)
        if 'test' in split:
            if frame_index != self._submission_frames.get(frame_token, None):
                return
        if 'training' in split:
            if (frame_index % 5) != 0:
                return
        vel_x = e2eframe.past_states.vel_x[-1]
        vel_y = e2eframe.past_states.vel_y[-1]
        initial_speed = np.sqrt(vel_x**2 + vel_y**2)
        camera_image_list, camera_calibration_list = return_cameras(e2eframe)
        
        l0,f0,r0,b0 = camera_image_list[0], camera_image_list[1], camera_image_list[2],camera_image_list[3]
        l0_calibration,f0_calibration,r0_calibration,b0_calibration = camera_calibration_list[0], camera_calibration_list[1], camera_calibration_list[2],camera_calibration_list[3]

        intent = e2eframe.intent
        if intent == 2:
            driving_command = np.array([1,0,0,0])
        elif intent == 3:
            driving_command = np.array([0,0,1,0])
        elif intent == 1:
            driving_command = np.array([0,1,0,0])
        else:
            driving_command = np.array([0,0,0,1])
        
        history_pos_x = e2eframe.past_states.pos_x
        history_pos_y = e2eframe.past_states.pos_y
        history_vx = e2eframe.past_states.vel_x
        history_vy = e2eframe.past_states.vel_y
        history_ax = e2eframe.past_states.accel_x
        history_ay = e2eframe.past_states.accel_y
        history_dynamics = np.stack([history_pos_x, history_pos_y, history_vx, history_vy, history_ax, history_ay], axis=1)
        history_index = np.arange(1, 1 + len(history_dynamics),2)
        history_dynamics = history_dynamics[history_index]
        num_history_frames = len(history_dynamics)
        future_waypoints_matrix = np.stack([e2eframe.future_states.pos_x, e2eframe.future_states.pos_y], axis=1)
        future_index = np.arange(1, 1 + len(future_waypoints_matrix),2)
        future_waypoints_matrix = future_waypoints_matrix[future_index]


        if np.random.random() < 0.9 and 'training' in split:
            ego_velocity_2d = history_dynamics[-1, :2]  # [vx, vy]
            ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5
            num_poses, dt = (
                10,
                0.5,
            )
            cv_poses = np.array(
                [[(time_idx + 1) * dt * ego_speed, 0.0] for time_idx in range(num_poses)],
                dtype=np.float32,
            )   
            true_future_pos = future_waypoints_matrix  # shape: (future_frames, 2)
            errors = np.linalg.norm(cv_poses - true_future_pos, axis=1)
            mean_error = np.mean(errors)
            if mean_error < 0.5:
                return 

        whole_pos = np.concatenate([history_dynamics[:, :2], future_waypoints_matrix], axis=0)
        headings = compute_headings(whole_pos)
        local_ego_poses = np.concatenate([whole_pos, headings[:,np.newaxis]], axis=1)

        ego_statuses = []
        cameras = []

        frame_idx = num_history_frames - 1
        ego_status = EgoStatus(
            ego_pose=local_ego_poses[frame_idx].astype(np.float32),
            ego_velocity=history_dynamics[frame_idx, 2:4].astype(np.float32),
            ego_acceleration=history_dynamics[frame_idx, 4:6].astype(np.float32),
            driving_command=driving_command,
        )
        ego_statuses.append(ego_status)
        none_Camera = Camera(
            image=None,
            rendered_image=None,
            sensor2lidar_rotation=None,
            sensor2lidar_translation=None,
            intrinsics=None,
            distortion=None,
            real_valid=False,
        )
        cameras.append(
            Cameras(
                cam_f0=Camera(
                    image=f0,
                    rendered_image=f0,
                    sensor2lidar_rotation=f0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=f0_calibration['sensor2lidar_translation'],
                    intrinsics=f0_calibration['intrinsic'],
                    distortion=f0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_l0=Camera(
                    image=l0,
                    rendered_image=l0,
                    sensor2lidar_rotation=l0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=l0_calibration['sensor2lidar_translation'],
                    intrinsics=l0_calibration['intrinsic'],
                    distortion=l0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_r0=Camera(
                    image=r0,
                    rendered_image=r0,
                    sensor2lidar_rotation=r0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=r0_calibration['sensor2lidar_translation'],
                    intrinsics=r0_calibration['intrinsic'],
                    distortion=r0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_b0=Camera(
                    image=b0,
                    rendered_image=b0,
                    sensor2lidar_rotation=b0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=b0_calibration['sensor2lidar_translation'],
                    intrinsics=b0_calibration['intrinsic'],
                    distortion=b0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_l1=none_Camera,
                cam_l2=none_Camera,
                cam_r1=none_Camera,
                cam_r2=none_Camera,
            )
        )
        agent_input = AgentInput(ego_statuses, cameras, None)

        for builder in self._feature_builders:
            data_dict_path = Path(os.path.join(dirs, frame_token+str(frame_index), "features.gz"))
            os.makedirs(data_dict_path.parent, exist_ok=True)
            data_dict = builder.compute_features(agent_input)
            data_dict['frame_name'] = e2eframe.frame.context.name
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        future_trajs = local_ego_poses[num_history_frames:]
        trajectory = torch.tensor(future_trajs, dtype=torch.float32)
        if 'val' in split:
            targets = {
            "trajectory": trajectory,
            "token": frame_token,
            "rfs_trajs": current_rater_trajs,
            "rfs_scores": current_rater_scores,
            "rfs_len": current_rater_len,
            "initial_speed": initial_speed,
        }
        else:
            targets = {
            "trajectory": trajectory,
            "token": frame_token,
            "initial_speed": initial_speed,
        }
        data_dict_path = os.path.join(dirs, frame_token+str(frame_index), "targets.gz")
        dump_feature_target_to_pickle(data_dict_path, targets)

        return



    
    def __getitem__(self, index):
        file_key = self.data_loaded[index]
        file_path = os.path.join(self._cache_path, file_key)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    

    def __len__(self):
        return len(self.data_loaded)




class WaymoQADataset(torch.utils.data.Dataset):

    def __init__(self,
        file_list: List[str],
        submission_frames,
        include_val=False,
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        ):
        super().__init__()

        self._submission_frames = submission_frames
        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self.raw_file_list = file_list
        self._force_cache_computation = force_cache_computation
        self.include_val = include_val
        if self._cache_path is not None:
            self.cache_dataset()

    def cache_dataset(self):
                
        if os.path.exists(self._cache_path) and self._force_cache_computation is False:
            return
        else:
            for tfrecord_path in tqdm(self.raw_file_list):
                data_split = tfrecord_path.name.split('_')[0]
                self.process_one_record(tfrecord_path,data_split)

    def process_one_record(self, tfrecord_path,data_split):
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        dataset_iter = dataset.as_numpy_iterator()
        for cnt, bytes_example in enumerate(dataset_iter):
            e2eframe = wod_e2ed_pb2.E2EDFrame()
            e2eframe.ParseFromString(bytes_example)
            dirs = os.path.join(self._cache_path, data_split)
            os.makedirs(dirs, exist_ok=True)
            self.process(dirs, e2eframe)

    def process(self, dirs, e2eframe):
        frame_token,frame_index = e2eframe.frame.context.name.split('-')
        frame_index = int(frame_index)
        split = dirs.split('/')[-1]

        if 'val' in split:
            if not (len(e2eframe.preference_trajectories) > 0 and e2eframe.preference_trajectories[0].preference_score != -1):
                return  
            current_rater_trajs = []
            current_rater_scores = []
            current_rater_len = []
            rater_specified_trajs_and_scores_i = e2eframe.preference_trajectories
            for j in range(len(rater_specified_trajs_and_scores_i)):
                rater_traj =  np.stack(
                        [
                            rater_specified_trajs_and_scores_i[j].pos_x,
                            rater_specified_trajs_and_scores_i[j].pos_y,
                        ],
                        axis=-1,
                    )
                rater_len = rater_traj.shape[0]
                rater_traj = np.pad(rater_traj, ((0, 21 - rater_len), (0, 0)), 'constant', constant_values=0)
                current_rater_trajs.append(rater_traj)
                current_rater_len.append(rater_len)
                current_rater_scores.append(rater_specified_trajs_and_scores_i[j].preference_score)


            current_rater_scores = np.array(current_rater_scores).astype(np.float32)
            current_rater_trajs = np.array(current_rater_trajs).astype(np.float32)
            current_rater_len = np.array(current_rater_len).astype(np.int32)
        if 'test' in split:
            return
            if frame_index != self._submission_frames.get(frame_token, None):
                return
        if 'training' in split:
            if (frame_index % 5) != 0:
                return
        vel_x = e2eframe.past_states.vel_x[-1]
        vel_y = e2eframe.past_states.vel_y[-1]
        initial_speed = np.sqrt(vel_x**2 + vel_y**2)
        camera_image_list, camera_calibration_list = return_cameras(e2eframe)
        
        l0,f0,r0,b0 = camera_image_list[0], camera_image_list[1], camera_image_list[2],camera_image_list[3]
        l0_calibration,f0_calibration,r0_calibration,b0_calibration = camera_calibration_list[0], camera_calibration_list[1], camera_calibration_list[2],camera_calibration_list[3]

        intent = e2eframe.intent
        if intent == 2:
            driving_command = np.array([1,0,0,0])
        elif intent == 3:
            driving_command = np.array([0,0,1,0])
        elif intent == 1:
            driving_command = np.array([0,1,0,0])
        else:
            driving_command = np.array([0,0,0,1])
        
        history_pos_x = e2eframe.past_states.pos_x
        history_pos_y = e2eframe.past_states.pos_y
        history_vx = e2eframe.past_states.vel_x
        history_vy = e2eframe.past_states.vel_y
        history_ax = e2eframe.past_states.accel_x
        history_ay = e2eframe.past_states.accel_y
        history_dynamics = np.stack([history_pos_x, history_pos_y, history_vx, history_vy, history_ax, history_ay], axis=1)
        history_index = np.arange(1, 1 + len(history_dynamics),2)
        history_dynamics = history_dynamics[history_index]
        num_history_frames = len(history_dynamics)
        future_waypoints_matrix = np.stack([e2eframe.future_states.pos_x, e2eframe.future_states.pos_y], axis=1)
        future_index = np.arange(1, 1 + len(future_waypoints_matrix),2)
        future_waypoints_matrix = future_waypoints_matrix[future_index]


        # if np.random.random() < 0.9 and 'training' in split:
        #     ego_velocity_2d = history_dynamics[-1, :2]  # [vx, vy]
        #     ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5
        #     num_poses, dt = (
        #         10,
        #         0.5,
        #     )
        #     cv_poses = np.array(
        #         [[(time_idx + 1) * dt * ego_speed, 0.0] for time_idx in range(num_poses)],
        #         dtype=np.float32,
        #     )   
        #     true_future_pos = future_waypoints_matrix  # shape: (future_frames, 2)
        #     errors = np.linalg.norm(cv_poses - true_future_pos, axis=1)
        #     mean_error = np.mean(errors)
        #     if mean_error < 0.5:
        #         return 

        whole_pos = np.concatenate([history_dynamics[:, :2], future_waypoints_matrix], axis=0)
        headings = compute_headings(whole_pos)
        local_ego_poses = np.concatenate([whole_pos, headings[:,np.newaxis]], axis=1)

        ego_statuses = []
        cameras = []

        for frame_idx in range(num_history_frames):
            ego_status = EgoStatus(
                ego_pose=local_ego_poses[frame_idx].astype(np.float32),
                ego_velocity=history_dynamics[frame_idx, 2:4].astype(np.float32),
                ego_acceleration=history_dynamics[frame_idx, 4:6].astype(np.float32),
                driving_command=driving_command,
            )
            ego_statuses.append(ego_status)

        none_Camera = Camera(
            image=None,
            rendered_image=None,
            sensor2lidar_rotation=None,
            sensor2lidar_translation=None,
            intrinsics=None,
            distortion=None,
            real_valid=False,
        )
        cameras.append(
            Cameras(
                cam_f0=Camera(
                    image=f0,
                    rendered_image=f0,
                    sensor2lidar_rotation=f0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=f0_calibration['sensor2lidar_translation'],
                    intrinsics=f0_calibration['intrinsic'],
                    distortion=f0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_l0=Camera(
                    image=l0,
                    rendered_image=l0,
                    sensor2lidar_rotation=l0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=l0_calibration['sensor2lidar_translation'],
                    intrinsics=l0_calibration['intrinsic'],
                    distortion=l0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_r0=Camera(
                    image=r0,
                    rendered_image=r0,
                    sensor2lidar_rotation=r0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=r0_calibration['sensor2lidar_translation'],
                    intrinsics=r0_calibration['intrinsic'],
                    distortion=r0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_b0=Camera(
                    image=b0,
                    rendered_image=b0,
                    sensor2lidar_rotation=b0_calibration['sensor2lidar_rotation'],
                    sensor2lidar_translation=b0_calibration['sensor2lidar_translation'],
                    intrinsics=b0_calibration['intrinsic'],
                    distortion=b0_calibration['distortion'],
                    real_valid=True,
                ),
                cam_l1=none_Camera,
                cam_l2=none_Camera,
                cam_r1=none_Camera,
                cam_r2=none_Camera,
            )
        )
        agent_input = AgentInput(ego_statuses, cameras, None)

        future_trajs = local_ego_poses[num_history_frames:]


        image_save_path = os.path.join(dirs, frame_token+str(frame_index))
        os.makedirs(image_save_path, exist_ok=True)


        qa = get_qa(agent_input,future_trajs,image_save_path)
        if 'val' in split:
            meta_info = {"trajectory": future_trajs.tolist(),
            "token": frame_token,
            "rfs_trajs": current_rater_trajs.tolist(),
            "rfs_scores": current_rater_scores.tolist(),
            "rfs_len": current_rater_len.tolist(),
            "initial_speed": initial_speed}
            qa['meta_info'] = meta_info

        # save qa
        qa_path = os.path.join(dirs, frame_token+str(frame_index)+".json")
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=2, ensure_ascii=False)

        annotation = get_annotation(agent_input,future_trajs,image_save_path)
        os.makedirs(dirs+'_annotation', exist_ok=True)
        annotation_path = os.path.join(dirs+'_annotation', frame_token+str(frame_index)+".json")
        with open(annotation_path, "w") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

def return_cameras(data):
  """Return the front_left, front, and front_right cameras as a list of images"""
  image_list = []
  calibration_list = []
  # CameraName Enum reference:
  # https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50
  order = [2, 1, 3, 7]
  for camera_name in order:
    for index, image_content in enumerate(data.frame.images):
      if image_content.name == camera_name:
        # Decode the raw image string and convert to numpy type.
        calibration = data.frame.context.camera_calibrations[index]
        image = tf.io.decode_image(image_content.image).numpy()
        image_list.append(image)

        extrinsic = np.reshape(
            np.array(list(calibration.extrinsic.transform), dtype=np.float32),
            [4, 4])

        R = extrinsic[:3, :3]   # 提取旋转矩阵
        t = extrinsic[:3, 3:].reshape(-1)   # 提取平移向量（列向量）

        R_inv = R
        tmp0 = R_inv[:,0].copy()
        tmp1 = R_inv[:,1].copy()
        tmp2 = R_inv[:,2].copy()
        R_inv[:,0] = -tmp1
        R_inv[:,1] = -tmp2
        R_inv[:,2] = tmp0


        intrinsic = np.array(list(calibration.intrinsic), dtype=np.float32)
        K = np.array([
            [intrinsic[0],  0,  intrinsic[2]],
            [0,  intrinsic[1],  intrinsic[3]],
            [0,   0,   1]
        ])
        distortion = intrinsic[4:]

        calibration_list.append({'sensor2lidar_rotation': R_inv,'sensor2lidar_translation': t, 'intrinsic': K, 'distortion': distortion})
        break 
  # pad b0.shape[0] to have the same shape as f0
  f0_shape = image_list[0].shape
  b0 = image_list[-1]
  pad_len = f0_shape[0] - b0.shape[0]
  pad_left = pad_len // 2
  pad_right = pad_len - pad_left
  pad_shape = ((pad_left, pad_right),) + ((0, 0),) * (b0.ndim - 1)
  b0_padded = np.pad(b0, pad_shape, mode='constant', constant_values=0)
  image_list[-1] = b0_padded
  return image_list, calibration_list


def compute_headings(whole_pos, ref_frame=7, 
                     window_length=7, polyorder=2):
    """
    计算每一帧的朝向（heading），并平滑去噪，保证帧 ref_frame 的 heading=0。

    参数
    ----
    whole_pos : (n,2) ndarray
        车辆在每一帧的 (x,y) 位置。
    ref_frame : int
        参考帧索引，使该帧 heading 强制为 0。
    window_length : int, odd
        Savitzky–Golay 窗口长度（必须为奇数，且 >= polyorder+2）。
    polyorder : int
        多项式拟合阶数。

    返回
    ----
    headings : (n,) ndarray
        平滑且连续的朝向角度，单位：rad，范围自动展开，无跳跃。
    """
    # 1) 计算速度向量（中间差分）
    dx = np.gradient(whole_pos[:,0])
    dy = np.gradient(whole_pos[:,1])

    # 2) 对速度在时间轴上滤波，去除高频抖动
    #    window_length 必须为奇数，且 > polyorder
    dx_s = savgol_filter(dx, window_length, polyorder, mode='interp')
    dy_s = savgol_filter(dy, window_length, polyorder, mode='interp')

    # 3) 计算原始朝向
    raw_heading = np.arctan2(dy_s, dx_s)

    # 4) 对角度做 unwrap，消除 π 跳变
    heading = np.unwrap(raw_heading)

    # 5) 以第 ref_frame 帧为零参考
    heading = heading - heading[ref_frame]

    return heading

def get_annotation(agent_input, future_trajs,image_save_path):

    #f0,l0,r0 = agent_input.cameras[0].cam_f0.image, agent_input.cameras[0].cam_l0.image, agent_input.cameras[0].cam_r0.image

    # ------------ 1) save images --------------------------------------------
    #os.makedirs(image_save_path, exist_ok=True)
    f0_path = os.path.join(image_save_path, "front.png")
    r0_path = os.path.join(image_save_path, "fr.png")
    l0_path = os.path.join(image_save_path, "fl.png")
    acc_x, acc_y = agent_input.ego_statuses[-1].ego_acceleration[:2]
    vel_x, vel_y = agent_input.ego_statuses[-1].ego_velocity[:2]

    # # resize image to 532x476
    # f0 = cv2.resize(f0, (476, 532))
    # l0 = cv2.resize(l0, (476, 532))
    # r0 = cv2.resize(r0, (476, 532))
    # cv2.imwrite(f0_path, f0[:, :, ::-1])
    # cv2.imwrite(l0_path, l0[:, :, ::-1])
    # cv2.imwrite(r0_path, r0[:, :, ::-1])

    # ------------ 2) format trajectories ------------------------------------
    past_traj = np.stack([agent_input.ego_statuses[i].ego_pose for i in range(len(agent_input.ego_statuses))], axis=0)
    future_trajs = np.asarray(future_trajs).reshape(10, 3)

    def _coords_to_str(arr):
        """[[x,y,h], …] → '[x1, y1]', '[x2, y2]', …  (heading omitted for brevity)"""
        return ', '.join([f"[{x:.2f}, {y:.2f}]" for x, y, _ in arr])

    past_txt   = _coords_to_str(past_traj)
    future_txt = _coords_to_str(future_trajs)
    cmd_map = ["TURN_LEFT", "GO_STRAIGHT", "TURN_RIGHT", "UNKNOWN"]
    cmd_idx = int(np.argmax(agent_input.ego_statuses[-1].driving_command))
    driving_cmd = cmd_map[cmd_idx] if cmd_idx < len(cmd_map) else "UNKNOWN"
    # ------------ 3) compose prompt -----------------------------------------
    system_prompt = "You are an expert labeller of driving scenarios."

    user_prompt = (
        "INPUT\n"
        "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep:\n"
        " 1) front: <image>\n"
        " 2) front-right: <image>\n"
        " 3) front-left: <image>\n"
        f"- Current high-level intent: {driving_cmd}\n"
        f"- 4-second past trajectory (8 steps at 2 Hz): {past_txt}\n"
        f"- Expert 5-second future trajectory (10 steps at 2 Hz): {future_txt}\n"
        "TASK\n"
        "1. Inspect the input and decide, for each object class below, whether at least one critical instance of that class is present (i.e., it materially affects the ego-vehicle’s future trajectory). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego’s future path.\n"
        "Object classes to audit:\n"
        "- nearby_vehicle\n"
        "- pedestrian\n"
        "- cyclist\n"
        "- construction\n"
        "- traffic_element\n"
        "- weather_condition\n"
        "- road_hazard\n"
        "- emergency_vehicle\n"
        "- animal\n"
        "- special_vehicle\n"
        "- conflicting_vehicle\n"
        "- door_opening_vehicle\n"
        "2. Output \"yes\" or \"no\" for every class (no omissions).\n"
        "3. Compose a concise natural-language description explaining why the expert safe driver plans the given future trajectory.\n"
        "- Mention only the classes you marked \"yes\"\n"
        "- Describe how each of those critical objects or conditions influences the trajectory.\n"
        "- Do not invent objects or conditions not present in the input.\n"
        "4. From the expert’s 5-second future trajectory, assign exactly one category from each list\n"
        "   - speed   ∈ { keep, accelerate, decelerate }\n"
        "   - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }\n"
        "Output format (strict JSON, no extra keys, no commentary):\n"
        "{\n"
        "  \"critical_objects\": {\n"
        "    \"nearby_vehicle\": \"yes|no\",\n"
        "    \"pedestrian\": \"yes|no\",\n"
        "    \"cyclist\": \"yes|no\",\n"
        "    \"construction\": \"yes|no\",\n"
        "    \"traffic_element\": \"yes|no\",\n"
        "    \"weather_condition\": \"yes|no\",\n"
        "    \"road_hazard\": \"yes|no\",\n"
        "    \"emergency_vehicle\": \"yes|no\",\n"
        "    \"animal\": \"yes|no\",\n"
        "    \"special_vehicle\": \"yes|no\",\n"
        "    \"conflicting_vehicle\": \"yes|no\",\n"
        "    \"door_opening_vehicle\": \"yes|no\"\n"
        "  },\n"
        "  \"explanation\": 100-word description that references only the classes marked 'yes',\n"
        "  \"meta_behaviour\": {\n"
        "    \"speed\": \"keep|accelerate|decelerate\",\n"
        "    \"command\": \"straight|yield|left_turn|right_turn|lane_follow|lane_change_left|lane_change_right|reverse\"\n"
        "  }\n"
        "}\n"
    )

    # ---------- 4) package -----------------------------------------------------
    qa_sample = {
        "id": image_save_path.split('/')[-1],
        "images": [f0_path, r0_path, l0_path],
        "system": system_prompt,
        "messages": [
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": ""}
        ]
    }
    return qa_sample

def get_qa(agent_input, future_trajs,image_save_path):
    """
    Assemble one training sample for Qwen2.5-VL E2E planner
    WITHOUT a separate system message.

    Returns
    -------
    sample : dict
        {
            "images":   [front_path, fr_path, fl_path],
            "messages": [
                {"role": "user",      "content": USER_PROMPT},
                {"role": "assistant", "content": ASSISTANT_LABEL}
            ]
        }
    """
    # ------- 1) EGO STATE (latest) -------------------------------------
    ego_state = agent_input.ego_statuses[-1]
    acc_x, acc_y = ego_state.ego_acceleration[:2]
    vel_x, vel_y = ego_state.ego_velocity[:2]
    heading_rad  = ego_state.ego_pose[2]

    cmd_map = ["TURN_LEFT", "GO_STRAIGHT", "TURN_RIGHT", "UNKNOWN"]
    cmd_idx = int(np.argmax(ego_state.driving_command))
    driving_cmd = cmd_map[cmd_idx] if cmd_idx < len(cmd_map) else "UNKNOWN"

    # past 4 s @ 2 Hz → 8 pts
    past_xy  = np.array([s.ego_pose[:2] for s in agent_input.ego_statuses])
    past_line = " ".join([f"[{x:.2f},{y:.2f}]" for x, y in past_xy])

    # ------- 2) FUTURE TRAJ label (1 Hz, t=1…5 s) ----------------------
    idx = np.arange(1, 10, 2)                # pick 1,3,5,7,9
    future_xy  = np.asarray(future_trajs)[idx, :2]
    label_line = ", ".join([f"[{x:.2f},{y:.2f}]" for x, y in future_xy])

    # ------- 3) Save multi-view images ---------------------------------
    cam0 = agent_input.cameras[-1]
    front_img, fr_img, fl_img = cam0.cam_f0.image, cam0.cam_r0.image, cam0.cam_l0.image
    front_img = cv2.resize(front_img, (476, 532))
    fr_img    = cv2.resize(fr_img,    (476, 532))
    fl_img    = cv2.resize(fl_img,    (476, 532))
    os.makedirs(image_save_path, exist_ok=True)
    front_path = os.path.join(image_save_path, "front.png")
    fr_path    = os.path.join(image_save_path, "fr.png")
    fl_path    = os.path.join(image_save_path, "fl.png")
    cv2.imwrite(front_path, front_img[:, :, ::-1])
    cv2.imwrite(fr_path,    fr_img[:,   :, ::-1])
    cv2.imwrite(fl_path,    fl_img[:,   :, ::-1])

    # ------- 4) Compose single USER prompt ----------------------------
    system_prompt = "You are an expert driver."
    USER_PROMPT = (
        "INPUT\n"
        "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep:\n"
        " 1) front: <image>\n"
        " 2) front-right: <image>\n"
        " 3) front-left: <image>\n"
        f"- Current high-level intent: {driving_cmd}\n"
        f"- Current acceleration and velocity: [{acc_x:.2f},{acc_y:.2f}], [{vel_x:.2f},{vel_y:.2f}]\n"
        "TASK\n"
        "predict the optimal 5-second future trajectory (5 steps at 1 Hz) of the ego vehicle.\n"
        "Output format (raw text, not markdown or LaTeX):\n"
        "[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5]"
    )

    ASSISTANT_LABEL = label_line   # “[x,y] …” string

    return {
        "images":   [front_path, fr_path, fl_path],
        "system": system_prompt,
        "messages": [
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": ASSISTANT_LABEL}
        ]
    }



def get_qa_navsim(agent_input, future_trajs,image_save_path):
    """
    Assemble one training sample for Qwen2.5-VL E2E planner
    WITHOUT a separate system message.

    Returns
    -------
    sample : dict
        {
            "images":   [front_path, fr_path, fl_path],
            "messages": [
                {"role": "user",      "content": USER_PROMPT},
                {"role": "assistant", "content": ASSISTANT_LABEL}
            ]
        }
    """
    # ------- 1) EGO STATE (latest) -------------------------------------
    ego_state = agent_input.ego_statuses[-1]
    acc_x, acc_y = ego_state.ego_acceleration[:2]
    vel_x, vel_y = ego_state.ego_velocity[:2]
    heading_rad  = ego_state.ego_pose[2]

    cmd_map = ["TURN_LEFT", "GO_STRAIGHT", "TURN_RIGHT", "UNKNOWN"]
    cmd_idx = int(np.argmax(ego_state.driving_command))
    driving_cmd = cmd_map[cmd_idx] if cmd_idx < len(cmd_map) else "UNKNOWN"

    # past 4 s @ 2 Hz → 8 pts
    past_xy  = np.array([s.ego_pose[:2] for s in agent_input.ego_statuses])
    past_line = " ".join([f"[{x:.2f},{y:.2f}]" for x, y in past_xy])

    # ------- 2) FUTURE TRAJ label (1 Hz, t=1…5 s) ----------------------
    idx = np.arange(1, 10, 2)                # pick 1,3,5,7,9
    future_xy  = np.asarray(future_trajs)[idx, :2]
    label_line = " ".join([f"[{x:.2f},{y:.2f}]" for x, y in future_xy])

    # ------- 3) Save multi-view images ---------------------------------
    cam0 = agent_input.cameras[-1]
    front_img, fr_img, fl_img = cam0.cam_f0.image, cam0.cam_r0.image, cam0.cam_l0.image
    front_img = cv2.resize(front_img, (640, 360))
    fr_img    = cv2.resize(fr_img,    (640, 360))
    fl_img    = cv2.resize(fl_img,    (640, 360))
    os.makedirs(image_save_path, exist_ok=True)
    front_path = os.path.join(image_save_path, "front.png")
    fr_path    = os.path.join(image_save_path, "fr.png")
    fl_path    = os.path.join(image_save_path, "fl.png")
    cv2.imwrite(front_path, front_img[:, :, ::-1])
    cv2.imwrite(fr_path,    fr_img[:,   :, ::-1])
    cv2.imwrite(fl_path,    fl_img[:,   :, ::-1])

    # ------- 4) Compose single USER prompt ----------------------------
    USER_PROMPT = (
        # ← 将原 system 指令并入开头
        "You are an expert autonomous-driving planner. "
        "Given current multi-view images in the sequence of Front, Front-right and Front-left, and the ego vehicle's state, "
        "output a 5-step (1 Hz) future global-XY trajectory (metres).\n"
        "### IMAGES\n"
        "<image> <image> <image>\n"
        "### EGO_STATE\n"
        f"VEL [{vel_x:.2f},{vel_y:.2f}] ACC [{acc_x:.2f},{acc_y:.2f}] "
        "### INTENT\n"
        f"{driving_cmd}\n"
        # "### PAST_TRAJ  # 2 Hz, 4 s\n"
        # f"{past_line}\n"
        "### TASK\n"
        "Predict 5-step future trajectory (1 Hz, 1-5 s) as:\n"
        "[x1,y1] ... [x5,y5]"
    )

    ASSISTANT_LABEL = label_line   # “[x,y] …” string

    return {
        "images":   [front_path, fr_path, fl_path],
        "messages": [
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": ASSISTANT_LABEL}
        ]
    }
