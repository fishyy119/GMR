from typing import Tuple

import numpy as np
import pyrender
import trimesh

Point3D = Tuple[float, float, float]


def add_world_axes(scene: pyrender.Scene, axis_length=0.5, axis_thickness=0.02):
    """
    在 pyrender 场景中添加世界坐标系可视化
    X: 红色
    Y: 绿色
    Z: 蓝色
    """
    axes_colors = {
        "x": [1.0, 0.0, 0.0, 1.0],  # 红
        "y": [0.0, 1.0, 0.0, 1.0],  # 绿
        "z": [0.0, 0.0, 1.0, 1.0],  # 蓝
    }

    # 每个轴用一个细长的 Box 表示
    for axis, color in axes_colors.items():
        if axis == "x":
            box = trimesh.primitives.Box(extents=[axis_length, axis_thickness, axis_thickness])
            box.apply_translation([axis_length / 2, 0, 0])
        elif axis == "y":
            box = trimesh.primitives.Box(extents=[axis_thickness, axis_length, axis_thickness])
            box.apply_translation([0, axis_length / 2, 0])
        else:  # z
            box = trimesh.primitives.Box(extents=[axis_thickness, axis_thickness, axis_length])
            box.apply_translation([0, 0, axis_length / 2])

        # 设置颜色
        box.visual.vertex_colors = np.tile(color, (box.vertices.shape[0], 1))
        mesh = pyrender.Mesh.from_trimesh(box, smooth=False)
        scene.add(mesh)


def look_at_matrix(
    target: Point3D,
    distance: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    up: Point3D = (0, 0, 1),
):
    """
    世界系：Z朝上
    相机系：Z朝相机后方

    根据球面角生成相机位姿矩阵
    Args:
        target: np.array([x,y,z]) 目标点
        distance: float 相机距离目标点
        yaw: float 偏航角，单位角度
        pitch: float 俯仰角，单位角度
        up: 相机上方向

    Returns:
        4x4 np.float32 相机位姿矩阵
    """
    # 球面坐标 → 相机位置
    _target = np.array(target, dtype=np.float32)
    x = distance * np.cos(np.radians(pitch)) * np.cos(np.radians(yaw))
    y = distance * np.cos(np.radians(pitch)) * np.sin(np.radians(yaw))
    z = distance * np.sin(np.radians(pitch))
    eye = _target + np.array([x, y, z], dtype=np.float32)

    # forward 指向目标
    f = _target - eye
    f /= np.linalg.norm(f)

    # right
    _up = np.array(up, dtype=np.float32)
    r = np.cross(f, _up)
    r /= np.linalg.norm(r)

    # 正交化 up
    u = np.cross(r, f)

    # 4x4 位姿矩阵
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = r
    mat[:3, 1] = u
    mat[:3, 2] = -f  # pyrender 相机朝向
    mat[:3, 3] = eye

    return mat
