# pyright: standard
import argparse
import time
from pathlib import Path

import numpy as np
from rich import print


def path_expand(s) -> Path:
    return Path(s).expanduser().resolve(strict=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--smplx-file", help="SMPLX motion file to load.", type=str, required=True)
    parser.add_argument(
        "-o", "--save-path", type=path_expand, default=None, help="Path to save the robot motion. (DIRECTORY)"
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--video-quality", type=int, default=720, choices=[480, 720, 1080], help="Available video resolutions."
    )

    parser.add_argument("--loop", default=False, action="store_true", help="Loop the motion.")
    parser.add_argument("--record-video", default=False, action="store_true", help="Record the video.")
    parser.add_argument("--point", default=False, action="store_true", help="Show reference points.")
    parser.add_argument("--frame", default=False, action="store_true", help="Show reference frames.")
    parser.add_argument("--follow", default=False, action="store_true", help="Camera follow the robot.")
    parser.add_argument(
        "--rate-limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    HERE = Path(__file__).parent

    save_path: Path | None = args.save_path

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"{args.robot}_{Path(args.smplx_file).stem}.pkl"

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(args.smplx_file, SMPLX_FOLDER)

    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=HERE.parent / f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",
        video_width=args.video_quality * 16 // 9,
        video_height=args.video_quality,
    )

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    qpos_list = []

    # Start the viewer
    i = 0

    try:
        while True:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break

            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # retarget
            qpos = retarget.retarget(smplx_data)

            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                # human_motion_data=smplx_data,
                rate_limit=args.rate_limit,
                follow_camera=args.follow,
                show_ref_point=args.point,
                show_ref_frame=args.frame,
                human_motion_data=retarget.scaled_human_data,
                robot_joints_to_show=list(retarget.ik_match_table.keys()),
            )
            if save_path is not None:
                qpos_list.append(qpos)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        robot_motion_viewer.close()

    if save_path is not None:
        import pickle

        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {save_path}")
