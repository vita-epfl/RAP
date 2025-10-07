import os
import copy
import shapely
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
import numpy as np
from pyquaternion import Quaternion
import pickle
import cv2
import matplotlib.pyplot as plt
import torch

from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_images_from_lidar_tokens,
    get_sensor_data_from_sensor_data_tokens_from_db,
    get_cameras,
    get_scenarios_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel

NUPLAN_DB_PATH = os.environ["NUPLAN_DB_PATH"]
NUPLAN_SENSOR_PATH = os.environ["NUPLAN_SENSOR_PATH"]
from numpy import array
import uuid

camera_params = {'CAM_F0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.00785972, -0.02271912, 0.99971099],
                                                            [-0.99994262, 0.00745516, -0.00769211],
                                                            [-0.00727825, -0.99971409, -0.02277642]]),
                            'sensor2lidar_translation': array([1.65506747, -0.01168732, 1.49112208])},
                 'CAM_L0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.81776776, -0.0057693, 0.57551942],
                                                            [-0.57553938, -0.01377628, 0.81765802],
                                                            [0.0032112, -0.99988846, -0.01458626]]),
                            'sensor2lidar_translation': array([1.63069485, 0.11956747, 1.48117884])},
                 'CAM_L1': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.93120104, 0.00261563, -0.36449662],
                                                            [0.36447127, -0.02048653, 0.93098926],
                                                            [-0.00503215, -0.99978671, -0.0200304]]),
                            'sensor2lidar_translation': array([1.29939471, 0.63819702, 1.36736822])},
                 'CAM_L2': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.63520782, 0.01497516, -0.77219607],
                                                            [0.77232489, -0.00580669, 0.63520119],
                                                            [0.00502834, -0.99987101, -0.01525415]]),
                            'sensor2lidar_translation': array([-0.49561003, 0.54750373, 1.3472672])},
                 'CAM_R0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.82454901, 0.01165722, 0.56567043],
                                                            [-0.56528395, 0.02532491, -0.82450755],
                                                            [-0.02393702, -0.9996113, -0.01429199]]),
                            'sensor2lidar_translation': array([1.61828343, -0.15532203, 1.49007665])},
                 'CAM_R1': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.92684778, 0.02177016, -0.37480562],
                                                            [0.37497631, 0.00421964, -0.92702479],
                                                            [-0.01859993, -0.9997541, -0.01207426]]),
                            'sensor2lidar_translation': array([1.27299407, -0.60973112, 1.37217911])},
                 'CAM_R2': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.62253245, 0.03706878, -0.78171558],
                                                            [0.78163434, -0.02000083, -0.62341618],
                                                            [-0.03874424, -0.99911254, -0.01652307]]),
                            'sensor2lidar_translation': array([-0.48771615, -0.493167, 1.35027683])},
                 'CAM_B0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.00802542, 0.01047463, -0.99991293],
                                                            [0.99989075, -0.01249671, 0.00789433],
                                                            [-0.01241293, -0.99986705, -0.01057378]]),
                            'sensor2lidar_translation': array([-0.47463312, 0.02368552, 1.4341838])}}

def get_log_cam_info(log):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_PATH, log_name + '.db')

    log_cam_infos = {}
    for cam in get_cameras(log_file, [str(channel.value) for channel in CameraChannel]):
        intrinsics = np.array(pickle.loads(cam.intrinsic))
        translation = np.array(pickle.loads(cam.translation))
        rotation = np.array(pickle.loads(cam.rotation))
        rotation = Quaternion(rotation).rotation_matrix
        distortion = np.array(pickle.loads(cam.distortion))
        c = dict(
            intrinsic=intrinsics,
            distortion=distortion,
            translation=translation,
            rotation=rotation,
        )
        log_cam_infos[cam.token] = c

    return log_cam_infos

def get_closest_start_idx(log, lidar_pcs):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_PATH, log_name + '.db')

    # Find the first valid point clouds, with all 8 cameras available.
    found_start_index = False
    for start_idx in range(0, len(lidar_pcs)):
        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx].token], [str(channel.value) for channel in CameraChannel]
        )
        if len(list(retrieved_images)) == 8:
            found_start_index = True
            break

    if not found_start_index:
       return 0

    # Find the true LiDAR start_idx with the minimum timestamp difference with CAM_F0.
    try:
        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx].token], ['CAM_F0']
        )
        diff_0 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx].timestamp)

        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx + 1].token], ['CAM_F0']
        )
        diff_1 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx + 1].timestamp)

 
        start_idx = start_idx if diff_0 < diff_1 else start_idx + 1
        return start_idx
    except Exception as e:
        print(f'{log_name} {start_idx} {e}')
        return 0

def get_cam_info_from_lidar_pc(log, lidar_pc, log_cam_infos):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_PATH, log_name + '.db')
    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pc.token], [str(channel.value) for channel in CameraChannel]
    )
    cams = {}
    camera_exists = True
    for img in retrieved_images:
        channel = img.channel
        filename = img.filename_jpg
        filepath = os.path.join(NUPLAN_SENSOR_PATH, filename)
        if not os.path.exists(filepath):
            camera_exists = False
        cam_info = log_cam_infos[img.camera_token]
        cams[channel] = dict(
            data_path = filename,
            sensor2lidar_rotation = cam_info['rotation'],
            sensor2lidar_translation = cam_info['translation'],
            cam_intrinsic = cam_info['intrinsic'],
            distortion = cam_info['distortion'],
        )
    if len(cams) != 8:
        # find missing camera
        for channel in camera_params.keys():
            if channel not in cams:
                cams[channel] = dict(
                    data_path = os.path.join('missing_camera', uuid.uuid4().hex + '.png'),
                    sensor2lidar_rotation = camera_params[channel]['sensor2lidar_rotation'],
                    sensor2lidar_translation = camera_params[channel]['sensor2lidar_translation'],
                    cam_intrinsic = camera_params[channel]['intrinsics'],
                    distortion = camera_params[channel]['distortion'],
                )
        camera_exists = False
    return cams, camera_exists
