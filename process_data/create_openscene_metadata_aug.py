import argparse
import shutil
from typing import Dict, List

# import mmcv
import numpy as np
from os import listdir
from os.path import isfile, join

from pyquaternion import Quaternion
import time
import cv2

from tqdm import tqdm
from multiprocessing import Pool
import os

import multiprocessing
import pickle
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from scenarionet.converter.nuplan.type import get_traffic_obj_type, NuPlanEgoType, set_light_status
from scenarionet.converter.utils import nuplan_to_metadrive_vector, compute_angular_velocity
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_traffic_light_status_for_lidarpc_token_from_db
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from shapely.geometry import Point as Point2D
from shapely.ops import unary_union

from helpers.multiprocess_helper import get_scenes_per_thread
from helpers.canbus import CanBus
from helpers.driving_command import get_driving_command
from helpers.nuplan_cameras_utils import (
    get_log_cam_info, get_closest_start_idx, get_cam_info_from_lidar_pc
)
from helpers.renderer import ScenarioRenderer,save_as_video
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
import geopandas as gpd

NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
filtered_classes = ["traffic_cone", "barrier", "czone_sign", "generic_object"]

def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array([nuplan_to_metadrive_vector([pose.x, pose.y], nuplan_center) for pose in path])
    return points

def get_points_from_boundary(boundary, center):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
    points = nuplan_to_metadrive_vector(points, center)
    return points

def set_light_position(map_api, lane_id, center, target_position=8):
    lane = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
    assert lane is not None, "Can not find lane: {}".format(lane_id)
    path = lane.baseline_path.discrete_path
    acc_length = 0
    point = [path[0].x, path[0].y]
    for k, point in enumerate(path[1:], start=1):
        previous_p = path[k - 1]
        acc_length += np.linalg.norm([point.x - previous_p.x, point.y - previous_p.y])
        if acc_length > target_position:
            break
    return [point.x - center[0], point.y - center[1]]

def extract_map_features(map_api, center, radius=200):
    ret = {}
    np.seterr(all='ignore')
    layer_names = [
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.LANE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,

    ]
    center_for_query = Point2D(*center)
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)
    # Filter out stop polygons in turn stop
    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]
    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            edges = sorted(block.interior_edges, key=lambda lane: lane.index) \
                if layer == SemanticMapLayer.ROADBLOCK else block.interior_edges
            for index, lane_meta_data in enumerate(edges):
                if not hasattr(lane_meta_data, "baseline_path"):
                    continue
                if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                    boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
                    sizes = []
                    for idx, polygon in enumerate(boundary[0]):
                        sizes.append(len(polygon.xy[1]))
                    points = boundary[0][np.argmax(sizes)].xy
                elif isinstance(lane_meta_data.polygon.boundary, LineString):
                    points = lane_meta_data.polygon.boundary.xy
                polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])

                # According to the map attributes, lanes are numbered left to right with smaller indices being on the
                # left and larger indices being on the right.
                # @ See NuPlanLane.adjacent_edges()
                ret[lane_meta_data.id] = {
                    SD.TYPE: MetaDriveType.LANE_SURFACE_STREET \
                        if layer == SemanticMapLayer.ROADBLOCK else MetaDriveType.LANE_SURFACE_UNSTRUCTURE,
                    SD.POLYLINE: extract_centerline(lane_meta_data, center),
                    SD.ENTRY: [edge.id for edge in lane_meta_data.incoming_edges],
                    SD.EXIT: [edge.id for edge in lane_meta_data.outgoing_edges],
                    SD.LEFT_NEIGHBORS: [edge.id for edge in block.interior_edges[:index]] \
                        if layer == SemanticMapLayer.ROADBLOCK else [],
                    SD.RIGHT_NEIGHBORS: [edge.id for edge in block.interior_edges[index + 1:]] \
                        if layer == SemanticMapLayer.ROADBLOCK else [],
                    SD.POLYGON: polygon
                }
                if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                    continue
                left = lane_meta_data.left_boundary
                if left.id not in ret:
                    # only broken line in nuPlan data
                    line_type = MetaDriveType.LINE_BROKEN_SINGLE_WHITE
                    if line_type != MetaDriveType.LINE_UNKNOWN:
                        ret[left.id] = {SD.TYPE: line_type, SD.POLYLINE: get_points_from_boundary(left, center)}

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    # walkway
    for area in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
        ret[area.id] = {
            SD.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
            SD.POLYGON: polygon,
        }

    # corsswalk
    for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
        ret[area.id] = {
            SD.TYPE: MetaDriveType.CROSSWALK,
            SD.POLYGON: polygon,
        }

    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)

    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        block_points = nuplan_to_metadrive_vector(block_points, center)
        id = "boundary_{}".format(idx)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points}
    np.seterr(all='warn')
    return ret



def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-π, π].

    :param angle: The angle in radians.
    :return: The normalized angle in radians.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def create_nuplan_info(db_name_and_args):

    log_idx, log_db_name, args = db_name_and_args
    render_sensor_path = args.nuplan_sensor_path.replace(
        "sensor_blobs", "rendered_sensor_blobs")
    os.makedirs(render_sensor_path, exist_ok=True)

    render_sensor_path_augmented = args.nuplan_sensor_path.replace(
        "sensor_blobs", "rendered_sensor_blobs_augmented"
    )
    os.makedirs(render_sensor_path_augmented, exist_ok=True)

    ORIGINAL_EGO_DIMS = np.array([4.6, 1.8, 5])

    # Include 4 camera views: front, left, right, and back
    renderer = ScenarioRenderer(camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0'])
    frame_infos = []
    scene_list = []

    log_db = NuPlanDB(args.nuplan_root_path, join(nuplan_db_path, log_db_name + ".db"), None)
    log_name = log_db.log_name
    log_token = log_db.log.token
    map_location = log_db.log.map_version
    vehicle_name = log_db.log.vehicle_name

    map_api = get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_location)  # NOTE: lru cached

    log_file = os.path.join(nuplan_db_path, log_db_name + ".db")


    frame_idx = 0

    # list (sequence) of point clouds (each frame).
    lidar_pc_list = log_db.lidar_pc
    lidar_pcs = lidar_pc_list

    log_cam_infos = get_log_cam_info(log_db.log)
    start_idx = get_closest_start_idx(log_db.log, lidar_pcs)

    # Find key_frames (controled by args.sample_interval)
    lidar_pc_list = lidar_pc_list[start_idx :: args.sample_interval]
    index = -1
    img_list = []
    

    ## limit the number of frames to debug
    # lidar_pc_list = lidar_pc_list[:25]  

    for lidar_pc in tqdm(lidar_pc_list):
        index += 1
        # LiDAR attributes.
        lidar_pc_token = lidar_pc.token
        scene_token = lidar_pc.scene_token
        pc_file_name = lidar_pc.filename
        next_token = lidar_pc.next_token
        prev_token = lidar_pc.prev_token
        lidar_token = lidar_pc.lidar_token
        time_stamp = lidar_pc.timestamp
        scene_name = f"log-{log_idx:04d}-{lidar_pc.scene.name}"
        lidar_boxes = lidar_pc.lidar_boxes
        roadblock_ids = [
            str(roadblock_id)
            for roadblock_id in str(lidar_pc.scene.roadblock_ids).split(" ")
            if len(roadblock_id) > 0
        ]

        if scene_token not in scene_list:
            scene_list.append(scene_token)
            frame_idx = 0

        can_bus = CanBus(lidar_pc).tensor
        lidar = log_db.session.query(Lidar).filter(Lidar.token == lidar_token).all()

        traffic_lights = []
        for traffic_light_status in get_traffic_light_status_for_lidarpc_token_from_db(
            log_file, lidar_pc_token
        ):
            lane_connector_id: int = traffic_light_status.lane_connector_id
            is_red: bool = traffic_light_status.status.value == 2
            traffic_light_position = set_light_position(map_api, lane_connector_id, [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y])
            traffic_lights.append((lane_connector_id, is_red, traffic_light_position))

        ego_pose = StateSE2(
            lidar_pc.ego_pose.x,
            lidar_pc.ego_pose.y,
            lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0],
        )
        driving_command = get_driving_command(ego_pose, map_api, roadblock_ids) # unknown

        scenario = {}
        map_features = extract_map_features(map_api, [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y], radius=200)
        scenario['map_features'] = map_features
        scenario['ego_pos'] = [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y]
        scenario['ego_heading'] = lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0]
        scenario['traffic_lights'] = traffic_lights
        info = {
            "token": lidar_pc_token,
            "frame_idx": frame_idx,
            "timestamp": time_stamp,
            "log_name": log_name,
            "log_token": log_token,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "map_location": map_location,
            "map_features": map_features,  
            "roadblock_ids": roadblock_ids,
            "vehicle_name": vehicle_name,
            "can_bus": can_bus,
            "lidar_path": pc_file_name, 
            "lidar2ego_translation": lidar[0].translation_np,
            "lidar2ego_rotation": [
                lidar[0].rotation.w,
                lidar[0].rotation.x,
                lidar[0].rotation.y,
                lidar[0].rotation.z,
            ],
            "ego2global_translation": can_bus[:3],
            "ego2global_rotation": can_bus[3:7],
            "ego_dynamic_state": [
                lidar_pc.ego_pose.vx,
                lidar_pc.ego_pose.vy,
                lidar_pc.ego_pose.acceleration_x,
                lidar_pc.ego_pose.acceleration_y,
            ],
            "traffic_lights": traffic_lights,
            "driving_command": driving_command, 
            "cams": dict(),
        }
        info["sample_prev"] = None
        info["sample_next"] = None

        if index > 0:  # find prev.
            info["sample_prev"] = lidar_pc_list[index - 1].token
        if index < len(lidar_pc_list) - 1:  # find next.
            next_key_token = lidar_pc_list[index + 1].token
            next_key_scene = lidar_pc_list[index + 1].scene_token
            info["sample_next"] = next_key_token
        else:
            next_key_token, next_key_scene = None, None

        if next_key_token == None or next_key_token == "":
            frame_idx = 0
        else:
            if next_key_scene != scene_token:
                frame_idx = 0
            else:
                frame_idx += 1

        # Parse lidar2ego translation.
        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # add lidar2global: map point coord in lidar to point coord in the global
        l2e = np.eye(4)
        l2e[:3, :3] = l2e_r_mat
        l2e[:3, -1] = l2e_t
        e2g = np.eye(4)
        e2g[:3, :3] = e2g_r_mat
        e2g[:3, -1] = e2g_t
        lidar2global = np.dot(e2g, l2e)
        info["ego2global"] = e2g
        info["lidar2ego"] = l2e
        info["lidar2global"] = lidar2global

        cams, camera_exists = get_cam_info_from_lidar_pc(log_db.log, lidar_pc, log_cam_infos)

        info["camera_exists"] = camera_exists
        info["cams"] = cams

        # Parse 3D object labels.
        if not args.is_test:
            if args.filter_instance:
                fg_lidar_boxes = [
                    box for box in lidar_boxes if box.category.name not in filtered_classes
                ]
            else:
                fg_lidar_boxes = lidar_boxes

            instance_tokens = [item.token for item in fg_lidar_boxes]
            track_tokens = [item.track_token for item in fg_lidar_boxes]

            inv_ego_r = lidar_pc.ego_pose.trans_matrix_inv
            ego_yaw = lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0]

            locs = np.array(
                [
                    np.dot(
                        inv_ego_r[:3, :3],
                        (b.translation_np - lidar_pc.ego_pose.translation_np).T,
                    ).T
                    for b in fg_lidar_boxes
                ]
            ).reshape(-1, 3)
            dims = np.array([[b.length, b.width, b.height] for b in fg_lidar_boxes]).reshape(
                -1, 3
            )
            rots = np.array([b.yaw for b in fg_lidar_boxes]).reshape(-1, 1)
            rots = rots - ego_yaw

            velocity_3d = np.array([[b.vx, b.vy, b.vz] for b in fg_lidar_boxes]).reshape(-1, 3)
            for i in range(len(fg_lidar_boxes)):
                velo = velocity_3d[i]
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity_3d[i] = velo

            names = [box.category.name for box in fg_lidar_boxes]
            names = np.array(names)
            gt_boxes_nuplan = np.concatenate([locs, dims, rots], axis=1)
            locs_world = np.array(
                [
                    b.translation_np - lidar_pc.ego_pose.translation_np
                    for b in fg_lidar_boxes
                ]
            ).reshape(-1, 3)
            gt_boxes_world = np.concatenate([locs_world, dims, rots+ego_yaw], axis=1)
                
            info["anns"] = dict(
                gt_boxes=gt_boxes_nuplan,
                gt_boxes_world=gt_boxes_world,
                gt_names=names,
                gt_velocity_3d=velocity_3d.reshape(-1, 3),
                instance_tokens=instance_tokens,
                track_tokens=track_tokens,
            )

        frame_infos.append(info)

    
    ##### Treat all neighboring vehicles as a new ego-vehicle
    '''
    token: copy
    frame_idx: copy
    timestamp: copy
    log_name: f"{log_name}_{vehicle_id}"
    log_token: copy
    scene_name: f"{scene_name}_{vehicle_id}"
    scene_token: f"{scene_token}_{vehicle_id}"
    map_location: copy
    roadblock_ids: []
    vehicle_name: copy
    can_bus: None
    lidar_path: None
    lidar2ego_translation: None
    lidar2ego_rotation: None
    ego2global_translation: re-calculate based on the new virtual ego
    ego2global_rotation: re-calculate based on the new virtual ego
    ego_dynamic_state: re-calculate based on the new virtual ego's 'gt_boxes_world'
    traffic_lights: copy
    driving_command: [0, 0, 0, 1] # unknown driving
    cams: None
    sample_prev: copy
    sample_next: copy
    ego2global: re-calculate based on the new virtual ego
    lidar2ego: copy
    lidar2global: copy
    camera_exists: False
    ego_exists: True/False
    anns: {
        gt_boxes: remove virtual ego, reframe coordinates based new virtual ego,
        gt_boxes_world: remove virtual ego,
        gt_names: remove virtual ego,
        gt_velocity_3d: remove virtual ego, reframe coordinates based new virtual ego,
        instance_tokens: remove virtual ego,
        track_tokens: remove virtual ego
    }"


    '''
    ## create a list to store all appeared vehicles.
    appeared_vehicles = []
    for info in frame_infos:
        track_tokens = info['anns']['track_tokens']
        gt_names = info['anns']['gt_names']
        n_objects = len(track_tokens)
        for i in range(n_objects):
            if gt_names[i] == 'vehicle' and track_tokens[i] not in appeared_vehicles:
                appeared_vehicles.append(track_tokens[i])

    

    # New Filtering Parameters from your colleague
    WINDOW_SIZE = 14  # 7 seconds at 2Hz
    CURRENT_FRAME_IDX_IN_WINDOW = 3  # The current frame is the 4th (0-indexed)
    FUTURE_HORIZON = 10  # 5 seconds prediction
    CV_ERROR_THRESHOLD = 0.5  # meters
    DT = 0.5  # Time step is 0.5s (2Hz)



    ## iterate all neighboring vehicles to create a new ego-vehicle for each of them
    for vehicle_id in tqdm(appeared_vehicles, desc="Augmenting data for vehicles"):
        new_frame_infos = []
        img_list_aug = [] # for visualization

        ## check if we have enough frames
        if len(frame_infos) < WINDOW_SIZE:
            continue

        ## handle the augmented frames
        for i in range(len(frame_infos)):
            info = frame_infos[i]

            # build a new_info 
            new_info = {
                "token": None,
                "frame_idx": None,
                "timestamp": None,
                "log_name": f"{log_name}_{vehicle_id}",
                "log_token": None,
                "scene_name": f"{scene_name}_{vehicle_id}",
                "scene_token": f"{scene_token}_{vehicle_id}",
                "map_location": None,
                "roadblock_ids": [],
                "vehicle_name": None,
                "can_bus": None,
                "lidar_path": None,
                "lidar2ego_translation": None,
                "lidar2ego_rotation": None,
                "ego2global_translation": None,
                "ego2global_rotation": None,
                "ego_dynamic_state": None,
                "traffic_lights": None,
                "driving_command": [0, 0, 0, 1],
                "cams": None,
                "sample_prev": None,
                "sample_next": None,
                "ego2global": None,
                "lidar2ego": None,
                "lidar2global": None,
                "camera_exists": False,
                "ego_exists": False,
                "is_valid": False,
                "anns": {
                    "gt_boxes": None,
                    "gt_boxes_world": None,
                    "gt_names": None,
                    "gt_velocity_3d": None,
                    "instance_tokens": None,
                    "track_tokens": None
                }
            }


            is_window_valid = True
            window_indices = range(i - CURRENT_FRAME_IDX_IN_WINDOW, i - CURRENT_FRAME_IDX_IN_WINDOW + WINDOW_SIZE)

            ## check if current frame existing
            if vehicle_id not in frame_infos[i]['anns']['track_tokens']:
                new_frame_infos.append(new_info)
                continue
            new_info['ego_exists'] = True

            ## Filter 1: check if there is a 14-frame window always valid
            for j in window_indices:
                # Check if the vehicle exists in all frames of the window
                if j < 0 or j >= len(frame_infos) or vehicle_id not in frame_infos[j]['anns']['track_tokens']:
                    is_window_valid = False
                    # new_info['is_valid'] = False
                    break

            ## Filter 2: check if the motion is too simple (after passing filter 1)
            if is_window_valid:
                # --- CONSTANT VELOCITY (CV) MODEL FILTER ---
                # Get current state of the new pseudo-ego vehicle in ACTUAL WORLD COORDINATES
                current_vehicle_idx = info['anns']['track_tokens'].index(vehicle_id)
                current_pos_world_diff = info['anns']['gt_boxes_world'][current_vehicle_idx, :2]
                
                # Convert world coordinate difference to actual world coordinates
                # gt_boxes_world contains world coordinate differences (b.translation_np - ego.translation_np)
                original_ego_world_pos = info['ego2global_translation'][:2]
                current_pos_world = original_ego_world_pos + current_pos_world_diff

                # Calculate velocity using the next frame in world coordinates
                next_info = frame_infos[i+1]
                next_vehicle_idx = next_info['anns']['track_tokens'].index(vehicle_id)
                next_pos_world_diff = next_info['anns']['gt_boxes_world'][next_vehicle_idx, :2]
                
                # Convert next frame's world coordinate difference to actual world coordinates
                next_ego_world_pos = next_info['ego2global_translation'][:2]
                next_pos_world = next_ego_world_pos + next_pos_world_diff
                
                time_interval = (next_info['timestamp'] - info['timestamp']) / 1e6
                velocity_world = (next_pos_world - current_pos_world) / time_interval

                # Predict future trajectory using CV model in world coordinates
                predicted_future_poses = []
                true_future_poses = []
                
                for k in range(1, FUTURE_HORIZON + 1):
                    # Predicted position using constant velocity
                    predicted_pos = current_pos_world + velocity_world * (k * DT)
                    predicted_future_poses.append(predicted_pos)
                    
                    # True position: convert from world coordinate difference to actual world coordinates
                    future_frame_info = frame_infos[i+k]
                    vehicle_idx_future = future_frame_info['anns']['track_tokens'].index(vehicle_id)
                    true_pos_world_diff = future_frame_info['anns']['gt_boxes_world'][vehicle_idx_future, :2]
                    
                    # Convert to actual world coordinates
                    future_ego_world_pos = future_frame_info['ego2global_translation'][:2]
                    true_pos_world = future_ego_world_pos + true_pos_world_diff
                    true_future_poses.append(true_pos_world)
                
                predicted_future_poses = np.array(predicted_future_poses)
                true_future_poses = np.array(true_future_poses)

                # Compare and filter if motion is too simple
                errors = np.linalg.norm(predicted_future_poses - true_future_poses, axis=1)
                mean_error = np.mean(errors)

                if mean_error > CV_ERROR_THRESHOLD:
                    ## Pass both Filters
                    new_info['is_valid'] = True

            ## Process the information for the current frame, regardless passing filter(s) or not

            vehicle_index = info['anns']['track_tokens'].index(vehicle_id)
            new_ego_pose_relative = info['anns']['gt_boxes_world'][vehicle_index]
            
            # Convert world coordinate difference to actual world coordinates for scenario setup
            abs_new_ego_pos = info['ego2global_translation'] + new_ego_pose_relative[:3]

            new_scenario = {} # for rendering
            new_scenario['map_features'] = extract_map_features(map_api, abs_new_ego_pos[:2], radius=200)

    
            new_scenario['traffic_lights'] = info['traffic_lights']

            new_ego_yaw_world = new_ego_pose_relative[6]  
            
            new_scenario['ego_pos'] = abs_new_ego_pos[:2]
            new_scenario['ego_heading'] = new_ego_yaw_world


            
            new_info['frame_idx'] = info['frame_idx']
            new_info['timestamp'] = info['timestamp']
            new_info['log_name'] = f"{info['log_name']}_{vehicle_id}"
            new_info['token'] = f"{info['token']}_{new_info['log_name']}"
            new_info['log_token'] = info['log_token']

            new_info['scene_name'] = f"{info['scene_name']}_{vehicle_id}"
            new_info['scene_token'] = f"{info['scene_token']}_{vehicle_id}"
            new_info['map_location'] = info['map_location']
            new_info['vehicle_name'] = f"{info['vehicle_name']}_{vehicle_id}"


            new_info['traffic_lights'] = info['traffic_lights']
            new_info['driving_command'] = [0, 0, 0, 1] 
            new_info['sample_prev'] = info['sample_prev']
            new_info['sample_next'] = info['sample_next']


            new_ego_global_pose = info['anns']['gt_boxes_world'][vehicle_index]
            
            # Extract yaw_world early since it's needed in multiple places
            yaw_world = new_ego_global_pose[6]  # Yaw is already world-absolute
            
            vehicle_velocity_in_orig_ego_frame = info['anns']['gt_velocity_3d'][vehicle_index, :3]
            orig_ego_quaternion = Quaternion(*info['ego2global_rotation'])
            orig_ego_yaw_world = orig_ego_quaternion.yaw_pitch_roll[0] 
            new_ego_yaw_world = yaw_world  
            
            relative_yaw = new_ego_yaw_world - orig_ego_yaw_world  
            
            c, s = np.cos(relative_yaw), np.sin(relative_yaw)
            R_orig_ego_to_new_ego = np.array([[c, -s], [s, c]]) 
            
            # Transform velocity directly from original ego frame to new ego frame
            velocity_ego = R_orig_ego_to_new_ego @ vehicle_velocity_in_orig_ego_frame[:2]
            
            # For acceleration, use finite differences if next frame is available, otherwise use zero
            if i < len(frame_infos) - 1 and vehicle_id in frame_infos[i+1]['anns']['track_tokens']:
                next_vehicle_idx = frame_infos[i+1]['anns']['track_tokens'].index(vehicle_id)
                next_velocity_in_orig_ego_frame = frame_infos[i+1]['anns']['gt_velocity_3d'][next_vehicle_idx, :3]
                
                # Calculate next frame's relative yaw
                next_orig_ego_quaternion = Quaternion(*frame_infos[i+1]['ego2global_rotation'])
                next_orig_ego_yaw_world = next_orig_ego_quaternion.yaw_pitch_roll[0]
                next_new_ego_yaw_world = frame_infos[i+1]['anns']['gt_boxes_world'][next_vehicle_idx, 6]
                next_relative_yaw = next_new_ego_yaw_world - next_orig_ego_yaw_world
                
                # Transform next velocity to new ego frame
                c_next, s_next = np.cos(next_relative_yaw), np.sin(next_relative_yaw)
                R_next_orig_to_new = np.array([[c_next, -s_next], [s_next, c_next]])
                next_velocity_ego = R_next_orig_to_new @ next_velocity_in_orig_ego_frame[:2]
                
                # Calculate acceleration in new ego frame
                time_interval = (frame_infos[i+1]['timestamp'] - info['timestamp']) / 1e6  # convert microseconds to seconds
                acceleration_ego = (next_velocity_ego - velocity_ego) / time_interval
            else:
                acceleration_ego = np.array([0.0, 0.0])
            
            new_info['ego_dynamic_state'] = [velocity_ego[0], velocity_ego[1], 
                                            acceleration_ego[0], acceleration_ego[1]]



            # Remove the new virtual ego from the annotations (it shouldn't see itself)
            new_info['anns'] = {k: np.copy(v) for k, v in info['anns'].items()}
            if vehicle_id in new_info['anns']['track_tokens']:
                    idx_to_del = np.where(new_info['anns']['track_tokens'] == vehicle_id)[0][0]
                    for key in new_info['anns']:
                        new_info['anns'][key] = np.delete(new_info['anns'][key], idx_to_del, axis=0)

            new_ego_pose_relative = info['anns']['gt_boxes_world'][vehicle_index]
            pos_rel_orig_ego = new_ego_pose_relative[:3]  
            
            original_ego_global_pose_mat = info['ego2global']
            new_ego_pos_world = pos_rel_orig_ego + original_ego_global_pose_mat[:3, 3]
            
            x_world, y_world, z_world = new_ego_pos_world


            new_ego2global = np.eye(4)
            c, s = np.cos(yaw_world), np.sin(yaw_world)  
            new_ego2global[:3, :3] = np.array([
                        [c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]
                    ])
            new_ego2global[:3, 3] = [x_world, y_world, z_world]
            q = Quaternion(axis=[0, 0, 1], angle=yaw_world)

            new_info['ego2global'] = new_ego2global
            new_info['ego2global_translation'] = np.array([x_world, y_world, z_world])
            new_info['ego2global_rotation'] = np.array([q.w, q.x, q.y, q.z])


            # Step 1: Get transformation matrices
            R_orig_ego_to_world = original_ego_global_pose_mat[:2, :2]  # Original ego → World rotation

            new_ego_yaw = yaw_world  
            c, s = np.cos(-new_ego_yaw), np.sin(-new_ego_yaw)  
            R_world_to_new_ego = np.array([[c, -s], [s, c]])

            # Step 2: Transform each object's velocity and position
            for j in range(len(new_info['anns']['track_tokens'])):


                vel_relative_to_orig_ego = new_info['anns']['gt_velocity_3d'][j, :2]
                vel_in_world = (R_orig_ego_to_world @ vel_relative_to_orig_ego)
                v_obj_in_new_ego_frame = R_world_to_new_ego @ vel_in_world
                        
                new_info['anns']['gt_velocity_3d'][j, :2] = v_obj_in_new_ego_frame

                obj_pos_world_diff = new_info['anns']['gt_boxes_world'][j, :3]
                
                obj_pos_world = original_ego_global_pose_mat[:2, 3] + obj_pos_world_diff[:2] 
                obj_z_world = original_ego_global_pose_mat[2, 3] + obj_pos_world_diff[2]
                
                new_ego_pos_world = new_ego2global[:2, 3]
                new_ego_z_world = new_ego2global[2, 3]
                
                obj_pos_rel_new_ego_world = obj_pos_world - new_ego_pos_world
                obj_z_rel_new_ego = obj_z_world - new_ego_z_world
                
                obj_pos_rel_new_ego = R_world_to_new_ego @ obj_pos_rel_new_ego_world
                
                new_info['anns']['gt_boxes_world'][j, :2] = obj_pos_rel_new_ego_world
                new_info['anns']['gt_boxes_world'][j, 2] = obj_z_rel_new_ego

                new_info['anns']['gt_boxes'][j, 0:2] = obj_pos_rel_new_ego
                new_info['anns']['gt_boxes'][j, 2] = obj_z_rel_new_ego
                
                obj_world_yaw = new_info['anns']['gt_boxes_world'][j, 6]  
                new_info['anns']['gt_boxes'][j, 6] = normalize_angle(obj_world_yaw - new_ego_yaw)


            orig_ego_world_pos = original_ego_global_pose_mat[:2, 3]  
            new_ego_world_pos = new_ego2global[:2, 3] 
            orig_ego_pos_world_rel = orig_ego_world_pos - new_ego_world_pos  
            orig_ego_pos_rel_new = R_world_to_new_ego @ orig_ego_pos_world_rel  
            
            orig_ego_z_rel = original_ego_global_pose_mat[2, 3] - new_ego2global[2, 3] 
            orig_ego_quaternion = Quaternion(*info['ego2global_rotation'])
            orig_ego_yaw_world = orig_ego_quaternion.yaw_pitch_roll[0]  
            
   
            orig_ego_velocity_in_orig_frame = np.array([info['ego_dynamic_state'][0], info['ego_dynamic_state'][1], 0])
            
            # Transform original ego velocity to new ego frame
            relative_yaw_orig_to_new = new_ego_yaw - orig_ego_yaw_world  # Yaw difference
            c_rel, s_rel = np.cos(relative_yaw_orig_to_new), np.sin(relative_yaw_orig_to_new)
            R_orig_to_new_ego = np.array([[c_rel, -s_rel], [s_rel, c_rel]])
            orig_ego_velocity_in_new_frame = R_orig_to_new_ego @ orig_ego_velocity_in_orig_frame[:2]
            orig_ego_velocity_3d = np.array([orig_ego_velocity_in_new_frame[0], orig_ego_velocity_in_new_frame[1], 0])
            
            orig_ego_box = np.concatenate([
                orig_ego_pos_rel_new,  # ego-relative position (x, y)
                [orig_ego_z_rel],      # ego-relative z
                ORIGINAL_EGO_DIMS,     # vehicle dimensions
                [normalize_angle(orig_ego_yaw_world - new_ego_yaw)]  # ego-relative yaw
            ])
            
            orig_ego_box_world = np.concatenate([
                orig_ego_pos_world_rel, # world coordinate difference (x, y)
                [orig_ego_z_rel],       # world coordinate difference z
                ORIGINAL_EGO_DIMS,      # vehicle dimensions
                [orig_ego_yaw_world]    # world-absolute yaw
            ])

            # Add original ego to the actual annotations (new_info['anns'])
            new_info['anns']['gt_boxes'] = np.vstack([new_info['anns']['gt_boxes'], orig_ego_box])
            new_info['anns']['gt_boxes_world'] = np.vstack([new_info['anns']['gt_boxes_world'], orig_ego_box_world])
            new_info['anns']['gt_names'] = np.append(new_info['anns']['gt_names'], 'vehicle')
            new_info['anns']['gt_velocity_3d'] = np.vstack([new_info['anns']['gt_velocity_3d'], orig_ego_velocity_3d])
            new_info['anns']['instance_tokens'] = np.append(new_info['anns']['instance_tokens'], 'original_ego_instance')
            new_info['anns']['track_tokens'] = np.append(new_info['anns']['track_tokens'], 'original_ego_track')

            # RENDERING PREPARATION: Create a copy for rendering (same as actual annotations now)
            anns_for_render = {k: np.copy(v) for k, v in new_info['anns'].items()}
            
            new_scenario['anns'] = anns_for_render

            # SCENE RENDERING: Generate camera views from the new ego's perspective
            rendered_cameras_aug = renderer.observe(new_scenario)


            # # # for vis only:
            # img_list_aug.append(rendered_cameras_aug)
            # out_folder = 'out_video'
            # if len(img_list_aug) == 20: # 200
            #         save_as_video(img_list_aug, f"{out_folder}/test_aug_{vehicle_id}.mp4")
            #         break



            new_info['cams'] = {}
            for cam_id, rendered_image in rendered_cameras_aug.items():
                    # Create a new, unique path for the augmented image
                    pseudo_cam_path = os.path.join(
                        f"{log_name}_{vehicle_id}",
                        cam_id,
                        f"{new_info['token']}.jpg"
                    )
                    full_rendered_path = os.path.join(render_sensor_path_augmented, pseudo_cam_path)
                    os.makedirs(os.path.dirname(full_rendered_path), exist_ok=True)
                    cv2.imwrite(full_rendered_path, rendered_image[:, :, ::-1])
                    
                    # Get the camera's metadata from the render.
                    from helpers.renderer import camera_params
                    new_cam_info = {
                        'data_path': pseudo_cam_path,
                        'sensor2lidar_rotation': camera_params[cam_id]['sensor2lidar_rotation'],
                        'sensor2lidar_translation': camera_params[cam_id]['sensor2lidar_translation'],
                        'cam_intrinsic': camera_params[cam_id]['intrinsics'],
                        'distortion': camera_params[cam_id]['distortion']
                    }


                    # Update the data path to point to the new rendered image
                    new_cam_info['data_path'] = pseudo_cam_path
                    
                    # Assign the complete dictionary to the new info
                    new_info['cams'][cam_id] = new_cam_info

            new_info['camera_exists'] = True if rendered_cameras_aug else False


            
            
            
            
            
            new_frame_infos.append(new_info)


        

        num_valid_frames = sum(info['is_valid'] for info in new_frame_infos)
        if num_valid_frames == 0:
            # print(f"No valid samples found for vehicle {vehicle_id}. Skipping save.")
            continue

        pkl_file_path_vehicle = f"{args.out_dir}/{log_name}_{vehicle_id}.pkl"
        with open(pkl_file_path_vehicle, "wb") as f:
            pickle.dump(new_frame_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
        

    del map_api
    return log_idx

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "--thread-num", type=int, default=16, help="number of threads for multi-processing."
    )

    # directory configurations.
    parser.add_argument("--nuplan-root-path", help="the path to nuplan root path.")
    parser.add_argument("--nuplan-db-path", help="the dir saving nuplan db.")
    parser.add_argument("--nuplan-sensor-path", help="the dir to nuplan sensor data.")
    parser.add_argument("--nuplan-map-version", help="nuplan mapping dataset version.")
    parser.add_argument("--nuplan-map-root", help="path to nuplan map data.")
    parser.add_argument("--out-dir", help="output path.")
    parser.add_argument("--start-index", type=int, default=0, help="start index.")
    parser.add_argument("--end-index", type=int, default=100, help="end index.")

    parser.add_argument(
        "--sample-interval", type=int, default=10, help="interval of key frame samples."
    )

    # split.
    parser.add_argument("--is-test", action="store_true", help="Dealing with Test set data.")
    parser.add_argument(
        "--filter-instance", action="store_true", help="Ignore instances in filtered_classes."
    )
    parser.add_argument("--split", type=str, default="train", help="Train/Val/Test set.")

    args = parser.parse_args()
    return args



def load_done_set(checkpoint_path='checkpoint.txt') -> set[int]:
    """读取已完成 index 集合（若文件不存在，则返回空集合）"""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def append_done(idx: int, checkpoint_path='checkpoint.txt') -> None:
    """将新完成的 index 追加到 checkpoint 文件"""
    # 用 'a' 打开保证追加写，单线程（主进程）执行，不需要锁
    with open(checkpoint_path, "a") as f:
        f.write(f"{idx}\n")

if __name__ == "__main__":
    args = parse_args()

    nuplan_root_path = args.nuplan_root_path
    nuplan_db_path = args.nuplan_db_path
    nuplan_sensor_path = args.nuplan_sensor_path
    nuplan_map_version = args.nuplan_map_version
    nuplan_map_root = args.nuplan_map_root
    out_dir = args.out_dir

    db_names = [
        f[:-3]
        for f in os.listdir(args.nuplan_db_path)
        if os.path.isfile(os.path.join(args.nuplan_db_path, f))
    ]
    db_names.sort()
    total = len(db_names)

    done_index = load_done_set()

    tasks = [
        (idx, db_name, args)
        for idx, db_name in enumerate(db_names)
    ]

    
    tasks = tasks[args.start_index:args.end_index]

    tasks = [task for task in tasks if task[0] not in done_index]

    print(f"processing {args.start_index} to {args.end_index} out of {total}")
    print(f'left {len(tasks)} tasks')


    # # --- For debugging a single file ---
    # if tasks: # Make sure there is at least one task to run
    #     print("--- RUNNING IN SINGLE-PROCESS DEBUG MODE ---")
    #     # Now breakpoint() will work inside the function
    #     create_nuplan_info(tasks[0]) 
    #     print("--- FINISHED PROCESSING ONE SCENE ---")
    # else:
    #     print("No tasks found to process.")

    with Pool(processes=args.thread_num) as pool:
        for result in tqdm(pool.imap_unordered(create_nuplan_info, tasks, chunksize=1),total=len(tasks)):
            append_done(result)


