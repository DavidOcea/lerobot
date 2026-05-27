#!/usr/bin/env python3
"""Capture a reference camera pose relative to an AprilTag marker.

Takes photos with the head camera, detects the target AprilTag,
and saves the camera pose (tvec, rvec) as a JSON reference file.

Usage:
    # Interactive: show live camera preview, press S to capture
    python -m lerobot.scripts.capture_reference_pose \\
        --config configs/visual_align_test.yaml \\
        --marker-id 0 \\
        --marker-size 0.10 \\
        --output ref_station_C.json

    # Headless / non-interactive: capture immediately
    python -m lerobot.scripts.capture_reference_pose \\
        --config configs/visual_align_test.yaml \\
        --no-preview \\
        --output ref_station_C.json
"""

import argparse
import faulthandler
import sys
import time

faulthandler.enable()

import cv2
import numpy as np

from lerobot.agent.visual_align import (
    detect_marker,
    save_reference_pose,
    _get_detector,
)
from lerobot.tasks.config import VisualAlignConfig, load_config_from_yaml
from lerobot.robots import make_robot_from_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture AprilTag reference pose with head camera",
    )
    p.add_argument("--config", required=True,
                   help="YAML config file (must contain robot_config)")
    p.add_argument("--marker-id", type=int, default=0,
                   help="AprilTag marker ID (default: 0)")
    p.add_argument("--marker-size", type=float, default=0.10,
                   help="Marker physical side length in meters (default: 0.10)")
    p.add_argument("--marker-family", default="tag36h11",
                   help="AprilTag family (default: tag36h11)")
    p.add_argument("--output", default="ref_pose.json",
                   help="Output JSON path (default: ref_pose.json)")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip live preview, capture immediately")
    p.add_argument("--camera-offset-pitch", type=float, default=0.0,
                   help="Camera pitch offset in degrees")
    p.add_argument("--camera-offset-yaw", type=float, default=0.0,
                   help="Camera yaw offset in degrees")
    p.add_argument("--camera-offset-x", type=float, default=0.0,
                   help="Camera forward offset in meters")
    p.add_argument("--camera-offset-y", type=float, default=0.0,
                   help="Camera lateral offset in meters")
    return p.parse_args()


def main():
    args = parse_args()

    # Load robot config from YAML
    orchestrator_cfg = load_config_from_yaml(args.config)
    robot_cfg = orchestrator_cfg.robot_config

    visual_config = VisualAlignConfig(
        marker_id=args.marker_id,
        marker_size=args.marker_size,
        marker_family=args.marker_family,
        camera_offset_pitch=args.camera_offset_pitch,
        camera_offset_yaw=args.camera_offset_yaw,
        camera_offset_x=args.camera_offset_x,
        camera_offset_y=args.camera_offset_y,
    )

    print("Connecting to robot...")
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("Robot connected.")

    detector = _get_detector(visual_config.marker_family)

    if args.no_preview:
        _capture_once(robot, visual_config, detector, args.output)
    else:
        _interactive_capture(robot, visual_config, detector, args.output)

    print("Disconnecting...")
    robot.disconnect()
    print("Done.")


def _capture_once(robot, visual_config, detector, output_path):
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