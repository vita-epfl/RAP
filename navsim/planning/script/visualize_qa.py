import json
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import os
from tqdm import tqdm
import tensorflow as tf
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset import dataset_pb2 as open_dataset
import textwrap
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont

def save_with_caption(
        image,                # H×W×3, RGB numpy array
        caption,              # str，任意长度
        out_path="output.png",
        dpi=100,
        font_size=20,
        line_spacing=1.3      # 行距系数
    ):
    """
    把 image 置于上方，把 caption 写在下方并自动换行后保存。
    """
    # —————————————— 1. 估算一行能放多少字符 ——————————————
    H, W = image.shape[:2]
    # 对 12–16 号字体，平均每个字符 ~0.6*fontsize 像素宽
    chars_per_line = max(8, int(W / ( 1 * font_size)))

    wrapped = textwrap.fill(caption, width=chars_per_line, break_long_words=False)
    n_lines = wrapped.count("\n") + 1
    text_height_px = int(font_size * line_spacing * n_lines)

    # —————————————— 2. 创建画布 ——————————————
    fig_height = (H + text_height_px) / dpi
    fig = plt.figure(figsize=(W / dpi, fig_height), dpi=dpi)

    # 用 GridSpec 把画布分成两行：图片区 + 文字区
    gs = fig.add_gridspec(2, 1, height_ratios=[H, text_height_px])

    # 2.1 图片
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.axis("off")

    # 2.2 文字（坐标系反转 y 方向更直观）
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    ax_txt.text(
        0.01, 1.0,            # 左上角
        wrapped,
        ha="left", va="top",
        wrap=True,
        fontsize=font_size
    )

    # —————————————— 3. 保存 ——————————————
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
# ---------- 你给出的工具函数 ---------- #
def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = vehicle_pose
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.flatten()) + [0.0] * 10

#   print(extrinsic)
#   print(intrinsic)
#   print(metadata)
#   print(camera_image_metadata)
#   print(world_points)
  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

def draw_points_on_image(image, points, size):
  """Draws points on an image.

  Args:
    image: The image to draw on.
    points: A numpy array of shape (N, 2) representing the points to draw.
  """
  for point in points:
    cv2.circle(image, (int(point[0]), int(point[1])), size, (255, 0, 0), -1)
  return image


# ---------- 主函数 ---------- #
def visualize(
    json_path: str,
    calib_pkl: str = "/home/fenglan/DiffusionDrive/navsim/planning/script/calibration.pkl",
    save_path = None,
):
    os.makedirs(save_path, exist_ok=True)
    
    # 3) 加载相机标定
    front3_camera_calibration_list = pickle.load(open(calib_pkl, "rb"))
    # 1) 读取 JSON
    with open(json_path, "r") as f:
        samples = json.load(f)
    for index in tqdm(range(0,len(samples),1000)):
        
        sample = samples[index]           
        image_paths = sample["images"]          # [front, fr, fl]
        messages    = sample["messages"]
        content     = messages[-1]["content"]   # assistant 最后一条回复

        # 2) 加载并按 fl → front → fr 排序
        order = [2, 0, 1]
        imgs = [
            cv2.resize(
                cv2.cvtColor(cv2.imread(image_paths[j]), cv2.COLOR_BGR2RGB),
                (972, 1079)          # (width, height)
            )
            for j in order
        ]

        # 4) **从 content 中提取未来轨迹坐标**
        #    回答格式为 …}\n[x1, y1], [x2, y2], ...
        pairs = re.findall(r'\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\]', content)
        if not pairs:
            raise ValueError("未能在 content 中找到坐标点")
        future_xy = np.array([[float(x), float(y)] for x, y in pairs], dtype=np.float32)
        future_xyz = np.concatenate([future_xy, np.zeros((future_xy.shape[0], 1))], axis=1)

        # 5) 单位车辆姿态
        vehicle_pose = np.eye(4, dtype=np.float32)

        # 6) 投影并绘制

        images_with_drawn_points = []
        for j in range(len(front3_camera_calibration_list)):
            waypoints_camera_space = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[j], future_xyz)
            images_with_drawn_points.append(draw_points_on_image(imgs[j], waypoints_camera_space, size=15))
        concatenated_image = np.concatenate(images_with_drawn_points, axis=1)

        image = concatenated_image
        text = content
        save_with_caption(image, text, out_path=os.path.join(save_path, str(index) + ".png"))

# ---------- CLI ---------- #
if __name__ == "__main__":

    visualize("/mnt/vita/scratch/vita-students/users/lfeng/DiffusionDrive/cache/waymo_qa/waymo_train_annotated.json","/mnt/vita/scratch/vita-students/users/lfeng/DiffusionDrive/navsim/planning/script/calibration.pkl", "./visualization")