from navsim.planning.training.dataset import Dataset, WaymoDataset, WaymoQADataset, NavsimQADataset

def build_navsim_dataset(scene_loader, agent, cache_path, force_cache_computation):
    return Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cache_path,
        force_cache_computation=force_cache_computation,
    )

def build_navsim_qa_dataset(scene_loader, agent, cache_path, force_cache_computation):
    return NavsimQADataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cache_path,
        force_cache_computation=force_cache_computation,
    )


def build_waymo_dataset(agent, file_list, submission_frames, include_val, cache_path, force_cache_computation):
    return WaymoDataset(
        file_list=file_list,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        submission_frames=submission_frames,
        include_val=include_val,
        cache_path=cache_path,
        force_cache_computation=force_cache_computation,
    )

def build_waymo_qa_dataset(file_list, submission_frames, include_val, cache_path, force_cache_computation):
    return WaymoQADataset(
        file_list=file_list,
        submission_frames=submission_frames,
        include_val=include_val,
        cache_path=cache_path,
        force_cache_computation=force_cache_computation,
    )

