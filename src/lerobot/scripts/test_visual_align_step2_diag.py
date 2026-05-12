"""Step 2 diagnostic: Real camera marker detection + core dump diagnosis.

Usage:
    python -m lerobot.scripts.test_visual_align_step2_diag \
        --robot.type supre_robot_follower \
        --robot.cameras.head_cam.type opencv \
        --robot.cameras.head_cam.index 0 \
        --robot.cameras.head_cam.width 640 \
        --robot.cameras.head_cam.height 480 \
        --robot.cameras.head_cam.fps 30 \
        --marker_id 0 \
        --marker_size 0.10 \
        --skip_disconnect false

After detection, optionally skip disconnect() to determine if the
core dump is caused by robot.disconnect() or by CAN SDK cleanup.
"""

import faulthandler
import sys
import time

faulthandler.enable()  # Print Python traceback on SIGABRT/SIGSEGV

import cv2
import numpy as np
from draccus import CLI

from lerobot.agent.visual_align import detect_marker, _get_detector
from lerobot.tasks.config import VisualAlignConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots import make_robot_from_config


@CLI
class Step2DiagConfig:
    robot: RobotConfig
    marker_id: int = 0
    marker_size: float = 0.10
    marker_family: str = "tag36h11"
    skip_disconnect: bool = False  # Set true to test if core dump is from disconnect()


def main(cfg: Step2DiagConfig):
    visual_config = VisualAlignConfig(
        marker_id=cfg.marker_id,
        marker_size=cfg.marker_size,
        marker_family=cfg.marker_family,
    )

    print(">>> BEFORE robot.connect()")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print(">>> AFTER robot.connect()")

    # Get observation with camera image
    print(">>> BEFORE robot.get_observation()")
    obs = robot.get_observation()
    print(">>> AFTER robot.get_observation()")

    images = obs.get("images", {})
    img = images.get("head_cam")
    if img is None:
        print("ERROR: No head_cam image in observation!")
        print(f"Available keys: {list(obs.keys())}")
        if "images" in obs:
            print(f"Image keys: {list(obs['images'].keys())}")
        if not cfg.skip_disconnect:
            print(">>> BEFORE robot.disconnect()")
            robot.disconnect()
            print(">>> AFTER robot.disconnect()")
        sys.exit(1)

    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # Convert to BGR if needed
    bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)

    # Detect marker
    print(">>> BEFORE detect_marker()")
    detector = _get_detector(visual_config.marker_family)
    marker = detect_marker(bgr, visual_config, detector)
    print(">>> AFTER detect_marker()")

    if marker is None:
        print("No marker detected!")
        # Save image for debugging
        cv2.imwrite("debug_no_marker.png", bgr)
        print("Saved debug image to debug_no_marker.png")
    else:
        print(f"Detected! ID={marker['id']}")
        print(f"  tvec: x={marker['tvec'][0]:.3f}m, y={marker['tvec'][1]:.3f}m, z={marker['tvec'][2]:.3f}m")
        print(f"  rvec: rx={marker['rvec'][0]:.3f}, ry={marker['rvec'][1]:.3f}, rz={marker['rvec'][2]:.3f}")

        # Compute AGV movement for validation
        from lerobot.agent.visual_align import compute_agv_movement
        dtheta_deg, forward_dist = compute_agv_movement(marker["tvec"], marker["rvec"], visual_config)
        print(f"  AGV movement: dtheta={dtheta_deg:.2f}deg, forward={forward_dist:.3f}m")

    # === Disconnect with diagnostics ===
    if cfg.skip_disconnect:
        print(">>> SKIPPING robot.disconnect() (process will exit without cleanup)")
        print(">>> If no core dump occurs, the crash is from disconnect()")
    else:
        print(">>> BEFORE robot.disconnect()")
        try:
            robot.disconnect()
            print(">>> AFTER robot.disconnect() — SUCCESS")
        except Exception as e:
            print(f">>> EXCEPTION during disconnect: {e}")

    print(">>> SCRIPT EXITING NORMALLY")
    time.sleep(0.5)  # Let C++ threads settle before exit


if __name__ == "__main__":
    main()