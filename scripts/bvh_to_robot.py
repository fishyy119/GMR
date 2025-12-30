import argparse
import os
import time
from pathlib import Path

import numpy as np
from rich import print


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--bvh_file", help="BVH motion file to load.", required=True, type=Path)
    parser.add_argument("-o", "--save_path", default=None, help="Path to save the robot motion.")

    parser.add_argument("--format", choices=["lafan1", "nokov", "neuron"], default="neuron")
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

    parser.add_argument("--motion_fps", default=30, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    HERE = Path(__file__).parent
    args = parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)

    qpos_list = []

    # Load SMPLX trajectory
    bvh_file: Path = args.bvh_file
    lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file, format=args.format)

    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=HERE.parent / f"videos/{args.robot}_{bvh_file.stem}.mp4",
        video_width=args.video_quality * 16 // 9,
        video_height=args.video_quality,
    )

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    print(f"mocap_frame_rate: {motion_fps}")

    # Start the viewer
    i = 0

    try:
        while True:
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            # Update progress bar
            # pbar.update(1)

            # Update task targets.
            smplx_data = lafan1_data_frames[i]

            # retarget
            qpos = retargeter.retarget(smplx_data)

            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                # follow_camera=False,
                rate_limit=args.rate_limit,
                follow_camera=args.follow,
                show_ref_point=args.point,
                show_ref_frame=args.frame,
                human_motion_data=retargeter.scaled_human_data,
                robot_joints_to_show=list(retargeter.ik_match_table.keys()),
            )

            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
                if i >= len(lafan1_data_frames):
                    break

            if args.save_path is not None:
                qpos_list.append(qpos)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        robot_motion_viewer.close()

    if args.save_path is not None:
        import pickle

        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
