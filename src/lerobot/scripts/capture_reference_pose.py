#!/usr/bin/env python3
"""Capture a reference camera pose relative to an AprilTag marker.

Takes one photo with the head camera, detects the target AprilTag,
and saves the camera pose (tvec, rvec) as a JSON reference file.

Usage:
    # Interactive: show live camera preview, press S to capture
    python -m lerobot.scripts.capture_reference_pose \\
        --robot.type supre_robot_follower \\
        --robot.cameras.head_cam.type opencv \\
        --robot.cameras.head_cam.index 0 \\
        --robot.cameras.head_cam.width 640 \\
        --robot.cameras.head_cam.height 480 \\
        --robot.cameras.head_cam.fps 30 \\
        --marker_id 0 \\
        --marker_size 0.10 \\
        --output ref_station_C.json

    # Headless / non-interactive: capture immediately
    python -m lerobot.scripts.capture_reference_pose \\
        --robot.type supre_robot_follower \\
        ... \\
        --no-preview
"""

import faulthandler
import sys
import time

faulthandler.enable()

import cv2
import numpy as np
from draccus import parse as draccus_parse

from lerobot.agent.visual_align import (
    detect_marker,
    save_reference_pose,
    _get_detector,
)
from lerobot.tasks.config import VisualAlignConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots import make_robot_from_config


class CaptureReferenceConfig:
    robot: RobotConfig
    marker_id: int = 0
    marker_size: float = 0.10
    marker_family: str = "tag36h11"
    output: str = "ref_pose.json"
    no_preview: bool = False
    camera_offset_pitch: float = 0.0
    camera_offset_yaw: float = 0.0
    camera_offset_x: float = 0.0
    camera_offset_y: float = 0.0


def main():
    cfg = draccus_parse(CaptureReferenceConfig)
    visual_config = VisualAlignConfig(
        marker_id=cfg.marker_id,
        marker_size=cfg.marker_size,
        marker_family=cfg.marker_family,
        camera_offset_pitch=cfg.camera_offset_pitch,
        camera_offset_yaw=cfg.camera_offset_yaw,
        camera_offset_x=cfg.camera_offset_x,
        camera_offset_y=cfg.camera_offset_y,
    )

    print("Connecting to robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print("Robot connected.")

    detector = _get_detector(visual_config.marker_family)

    if cfg.no_preview:
        _capture_once(robot, visual_config, detector, cfg.output)
    else:
        _interactive_capture(robot, visual_config, detector, cfg.output)

    print("Disconnecting...")
    robot.disconnect()
    print("Done.")


def _capture_once(robot, visual_config, detector, output_path):
    """Take one photo and try to capture the reference pose."""
    obs = robot.get_observation()
    img = obs.get("images", {}).get("head_cam")
    if img is None:
        print("ERROR: No head_cam image in observation!")
        sys.exit(1)

    bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
    marker = detect_marker(bgr, visual_config, detector)

    if marker is None:
        print(f"ERROR: Marker ID={visual_config.marker_id} not found!")
        cv2.imwrite("debug_no_marker.png", bgr)
        print("Saved debug image to debug_no_marker.png")
        sys.exit(1)

    _save_and_report(marker, output_path)


def _interactive_capture(robot, visual_config, detector, output_path):
    """Show live camera preview. Press S to capture, Q to quit."""
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  S = capture & save reference pose      │")
    print("  │  Q = quit without saving                │")
    print("  └─────────────────────────────────────────┘\n")

    while True:
        obs = robot.get_observation()
        img = obs.get("images", {}).get("head_cam")
        if img is None:
            print("Waiting for camera...")
            time.sleep(0.5)
            continue

        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
        marker = detect_marker(bgr, visual_config, detector)

        # Draw overlay
        display = bgr.copy()
        if marker is not None:
            cv2.putText(
                display,
                f"ID={marker['id']}  z={marker['tvec'][2]:.2f}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            corners = marker["corners"].astype(int)
            cv2.polylines(display, [corners], True, (0, 255, 0), 2)
        else:
            cv2.putText(
                display,
                "No marker detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Capture Reference Pose (S=save, Q=quit)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == ord('S'):
            if marker is None:
                print("No marker visible — cannot capture.")
            else:
                _save_and_report(marker, output_path)
                break
        elif key == ord('q') or key == ord('Q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


def _save_and_report(marker, output_path):
    tvec = marker["tvec"].copy()
    rvec = marker["rvec"].copy()
    save_reference_pose(output_path, tvec, rvec)
    print(f"\nReference pose saved to: {output_path}")
    print(f"  Marker ID: {marker['id']}")
    print(f"  tvec (camera frame): "
          f"x={tvec[0]:.4f}, y={tvec[1]:.4f}, z={tvec[2]:.4f} m")
    print(f"  rvec (Rodrigues):   "
          f"rx={rvec[0]:.4f}, ry={rvec[1]:.4f}, rz={rvec[2]:.4f}")


if __name__ == "__main__":
    main()