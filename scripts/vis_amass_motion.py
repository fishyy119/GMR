# pyright: standard
# pip install pyrender==0.1.45

import argparse
from operator import is_
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pyrender
import smplx
import torch
import trimesh
from rich import print
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
is_grab = False  # GRAB数据集的fps字段名写错了


def load_amass(npz_path: Path, device: str = "cpu"):
    data = np.load(npz_path)

    # numpy → torch
    poses = torch.from_numpy(data["poses"]).float().to(device)  # (T, 72)
    trans = torch.from_numpy(data["trans"]).float().to(device)  # (T, 3)

    fps_key = "mocap_frame_rate" if is_grab else "mocap_framerate"

    try:
        fps = int(data[fps_key])
    except KeyError:
        print(f"[yellow]Warning:[/yellow] '{fps_key}' not found in {npz_path}, defaulting to 30 fps.")
        fps = 30

    # AMASS (SMPL) → SMPL-X body pose (21 joints)
    body_pose_smpl = poses[:, 3:72]  # (T, 69)
    body_pose_smplx = body_pose_smpl[:, :63]  # (T, 63)

    motion = {
        "global_orient": poses[:, :3],  # (T, 3)
        "body_pose": body_pose_smplx,  # (T, 63)
        "transl": trans,  # (T, 3)
        "fps": fps,
    }

    return motion


def render_amass_motion(
    smpl,
    motion_path: Path,
    out_path: Path,
    yaw: float,
    pitch: float,
    camera_dist: float,
    show_axes: bool,
    image_size=(960, 720),
):
    motion = load_amass(motion_path)

    w, h = image_size
    renderer = pyrender.OffscreenRenderer(w, h)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.5)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

    # 后面减去center,所以原点是整个基座运动轨迹的中心
    cam_pose = look_at_matrix((0.0, 0.0, 0.0), distance=camera_dist, yaw=yaw, pitch=pitch)
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)

    if show_axes:
        add_world_axes(scene)

    T = motion["transl"].shape[0]
    center = motion["transl"].mean(axis=0)
    mesh_node = None

    target_fps = 30
    step = int(motion["fps"] / target_fps)
    frames = []

    for i in tqdm(range(0, T, step), desc="Rendering AMASS motion", leave=False):
        with torch.no_grad():
            smpl_out = smpl(
                global_orient=motion["global_orient"][None, i],
                body_pose=motion["body_pose"][None, i],
                transl=motion["transl"][None, i],
                # betas=motion["betas"][None],  # 使用中性身体形状
            )

        vertices = smpl_out.vertices[0].cpu().numpy()
        vertices -= center.cpu().numpy()  # 居中显示

        tri_mesh = trimesh.Trimesh(vertices, smpl.faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(mesh)

        res = renderer.render(scene)
        assert res is not None
        color, _ = res
        frames.append(color)

    renderer.delete()

    with iio.get_writer(
        out_path,
        fps=target_fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:  # frame: H×W×3 uint8 RGB
            writer.append_data(frame)  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AMASS motion using SMPL-X and pyrender")

    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="AMASS npz file or directory containing npz files"
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=Path("outputs"), help="Output directory for rendered videos"
    )
    parser.add_argument(
        "--axes", action=argparse.BooleanOptionalAction, default=False, help="Draw world coordinate axes"
    )
    parser.add_argument("--yaw", type=float, default=-90.0, help="Camera yaw angle in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="Camera pitch angle in degrees")
    parser.add_argument("--distance", type=float, default=2, help="Camera distance from the origin")
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[480, 720, 1080],
        default=720,
        help="Video vertical resolution (480 / 720 / 1080)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input.is_dir():
        amass_files = list(args.input.glob("*.npz"))
    else:
        amass_files = [args.input]

    smpl = smplx.create(
        model_path=str(REPO_ROOT / "assets/body_models"),
        model_type="smplx",
        num_pca_comps=45,
        gender="neutral",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if "GRAB" in str(args.input):
        is_grab = True

    from general_motion_retargeting.pyrender_utils import add_world_axes, look_at_matrix

    for amass_file in tqdm(amass_files, desc="Rendering files", unit="file"):
        try:
            render_amass_motion(
                smpl,
                amass_file,
                yaw=args.yaw,
                pitch=args.pitch,
                camera_dist=args.distance,
                show_axes=args.axes,
                image_size=(args.resolution * 16 // 9, args.resolution),
                out_path=args.out_dir / f"{amass_file.stem}.mp4",
            )
        except Exception as e:
            print(f"Failed to render {amass_file}: {e}")
