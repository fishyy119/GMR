# pyright: standard
# pyright: reportAttributeAccessIssue=false
import time
from pathlib import Path
from typing import List, Tuple

import imageio
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from rich import print
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.smpl import HumanData

RGBA = Tuple[float, float, float, float]


def draw_point(
    pos: np.ndarray,
    viewer: mjv.Handle,
    size: float = 0.025,
    color: RGBA = (1.0, 0.0, 0.0, 1.0),
    offset=np.array([1, 0, 0]),
):
    scn: mj.MjvScene = viewer.user_scn
    geom = scn.geoms[scn.ngeom]

    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[size, size, size],
        pos=pos + offset,
        mat=np.eye(3).flatten(),
        rgba=color,
    )
    scn.ngeom += 1


def draw_frame(
    pos: np.ndarray,
    mat: np.ndarray,
    viewer: mjv.Handle,
    size: float = 0.1,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    offset=np.array([-1, 0, 0]),
    is_robot_frame=False,
):
    if is_robot_frame:
        rgba_list = [[1, 0.5, 0, 1], [0, 1, 0.7, 1], [0, 0.5, 1, 1]]
    else:
        rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

    scn: mj.MjvScene = viewer.user_scn
    for i in range(3):
        geom = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            scn.geoms[scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + offset,
            to=pos + offset + size * (mat @ fix)[:, i],
        )
        scn.ngeom += 1


class RobotMotionViewer:
    def __init__(
        self,
        robot_type,
        camera_follow=True,
        motion_fps=30,
        transparent_robot=0,
        # video recording
        record_video=False,
        video_path: str | Path | None = None,
        video_width=640,
        video_height=480,
        keyboard_callback=None,
    ):
        self.is_first_step = True
        self.previous_lookat = np.zeros(3)

        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)

        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        self.viewer = mjv.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False, key_callback=keyboard_callback
        )

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = Path(video_path)

            video_dir = video_path.parent
            video_dir.mkdir(parents=True, exist_ok=True)

            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")

            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

    def step(
        self,
        # robot data
        root_pos,
        root_rot,
        dof_pos,
        # visualization options
        rate_limit=True,
        follow_camera=True,
        # some refrence visualization
        show_ref_point=False,
        show_ref_frame=False,
        point_size=0.03,
        frame_size=0.1,
        point_offset=np.array([1.2, 0.0, 0.0]),
        frame_offset=np.array([-1.2, 0.0, 0.0]),
        human_motion_data: HumanData | None = None,
        robot_joints_to_show: List[str] | None = None,
    ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.

        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot  # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos

        mj.mj_forward(self.model, self.data)

        point_offset = point_offset if show_ref_point else np.zeros(3)
        frame_offset = frame_offset if show_ref_frame else np.zeros(3)
        need_look: np.ndarray = self.data.xpos[self.model.body(self.robot_base).id] + (point_offset + frame_offset) / 2

        if self.is_first_step:
            self.viewer.cam.lookat = need_look
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 60  # 正面朝向机器人

            self.is_first_step = False
            self.previous_lookat = need_look
        else:
            if follow_camera:
                # ln 表：
                # 0.9 = -0.1
                # 0.8 = -0.22
                # 0.7 = -0.36
                # 0.6 = -0.51
                # 0.5 = -0.7
                # 0.4 = -0.9
                # 0.3 = -1.2
                # 0.2 = -1.6
                # 0.1 = -2.3
                # 指数系数绝对值越大，平滑越弱
                alpha = np.exp(-2.0 * np.abs(need_look - self.previous_lookat))
                smoothed_lookat = (1 - alpha) * need_look + alpha * self.previous_lookat
                self.viewer.cam.lookat = smoothed_lookat
                self.previous_lookat = smoothed_lookat.copy()

        # Clean custom geometry
        self.viewer.user_scn.ngeom = 0  # type: ignore

        if human_motion_data is not None:
            for human_body_name, (pos, rot) in human_motion_data.items():
                if show_ref_frame:
                    draw_frame(
                        pos,
                        R.from_quat(rot, scalar_first=True).as_matrix(),
                        self.viewer,
                        frame_size,
                        offset=frame_offset,
                        joint_name=None,
                    )
                if show_ref_point:
                    draw_point(pos, self.viewer, size=point_size, offset=point_offset, color=(0.0, 1.0, 0.0, 1.0))

        if robot_joints_to_show is not None:
            for name in robot_joints_to_show:
                body_id = self.model.body(name).id
                body_pos = self.data.xpos[body_id]
                R_body = self.data.xmat[body_id].reshape(3, 3)

                if show_ref_frame:
                    draw_frame(
                        body_pos,
                        R_body,
                        self.viewer,
                        frame_size,
                        offset=frame_offset,
                        joint_name=None,
                        is_robot_frame=True,
                    )
                if show_ref_point:
                    draw_point(body_pos, self.viewer, size=point_size, offset=point_offset, color=(1.0, 0.0, 0.0, 1.0))

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
