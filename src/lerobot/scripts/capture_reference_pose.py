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
    _marker_to_agv_xy,
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

    # Read calibrated camera_matrix / dist_coeffs from the YAML's first
    # visual_align task so the reference pose and alignment loop use the
    # SAME solvePnP coordinate system.  Without this the reference is
    # computed with default (uncalibrated) params and alignment with
    # calibrated params → near-guaranteed divergence.
    camera_matrix = None
    dist_coeffs = None
    # Fallback camera offsets from YAML (CLI args override)
    yaml_pitch = args.camera_offset_pitch
    yaml_yaw = args.camera_offset_yaw
    yaml_x = args.camera_offset_x
    yaml_y = args.camera_offset_y
    tag1_id = None
    yaml_marker_size = args.marker_size
    yaml_tag_1_size = None
    for task in orchestrator_cfg.tasks:
        if task.task_type == "visual_align" and task.visual_align_config is not None:
            va = task.visual_align_config
            if va.camera_matrix is not None and va.dist_coeffs is not None:
                camera_matrix = va.camera_matrix
                dist_coeffs = va.dist_coeffs
                print(f"Using calibrated camera params from YAML task '{task.name}'")
            # Use YAML offset values as defaults (CLI defaults are 0)
            if args.camera_offset_pitch == 0.0 and va.camera_offset_pitch != 0.0:
                yaml_pitch = va.camera_offset_pitch
            if args.camera_offset_yaw == 0.0 and va.camera_offset_yaw != 0.0:
                yaml_yaw = va.camera_offset_yaw
            if args.camera_offset_x == 0.0 and va.camera_offset_x != 0.0:
                yaml_x = va.camera_offset_x
            if args.camera_offset_y == 0.0 and va.camera_offset_y != 0.0:
                yaml_y = va.camera_offset_y
            # Dual-tag: read tag_1_id so the reference photo captures both
            tag1_id = va.tag_1_id
            # Read marker_size from YAML (CLI default is 0.10)
            if args.marker_size == 0.10 and va.marker_size != 0.10:
                yaml_marker_size = va.marker_size
            if va.tag_1_size is not None:
                yaml_tag_1_size = va.tag_1_size
            break

    visual_config = VisualAlignConfig(
        marker_id=args.marker_id,
        marker_size=yaml_marker_size,
        marker_family=args.marker_family,
        camera_offset_pitch=yaml_pitch,
        camera_offset_yaw=yaml_yaw,
        camera_offset_x=yaml_x,
        camera_offset_y=yaml_y,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        tag_1_id=tag1_id,
        tag_1_size=yaml_tag_1_size,
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

    # Dual-tag: also detect tag_1
    tag1_marker = None
    if visual_config.tag_1_id is not None:
        tag1_marker = detect_marker(bgr, visual_config, detector,
                                    target_id=visual_config.tag_1_id)
        if tag1_marker is None:
            print(f"WARNING: tag_1 (ID={visual_config.tag_1_id}) not found — "
                  f"saving single-tag only")

    _save_and_report(marker, output_path, tag1_marker)


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
        tag1_marker = None
        if visual_config.tag_1_id is not None:
            tag1_marker = detect_marker(bgr, visual_config, detector,
                                        target_id=visual_config.tag_1_id)

        display = bgr.copy()
        if marker is not None:
            cur_x, cur_y = _marker_to_agv_xy(marker["tvec"], visual_config)
            z_cam = marker["tvec"][2]
            tag1_info = ""
            if tag1_marker is not None:
                t1x, t1y = _marker_to_agv_xy(tag1_marker["tvec"], visual_config)
                tag1_info = f"  T1:ID={tag1_marker['id']}  lateral={t1y:+.3f}m"
                # Draw tag_1 bounding box (blue to distinguish from tag_0 green)
                t1_corners = tag1_marker["corners"].astype(int)
                cv2.polylines(display, [t1_corners], True, (255, 0, 0), 2)
                cv2.putText(display, f"T1", (t1_corners[0][0][0], t1_corners[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # ── Overlay ──────────────────────────────────
            cv2.putText(
                display,
                f"ID={marker['id']}{tag1_info}  z={z_cam:.2f}m  lateral={cur_y:+.3f}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            # Horizontal bar indicator for cur_y (centering guide)
            h, w = display.shape[:2]
            cx, cy, bar_w = w // 2, h - 50, min(w // 2, 300)
            # Background bar
            cv2.rectangle(display, (cx - bar_w // 2, cy - 8), (cx + bar_w // 2, cy + 8), (50, 50, 50), -1)
            # Center line
            cv2.line(display, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
            # Cursor — clamp ±5cm lateral to bar half-width
            cursor = int(cx + cur_y / 0.05 * bar_w // 2)
            cursor = max(cx - bar_w // 2, min(cx + bar_w // 2, cursor))
            cv2.circle(display, (cursor, cy), 8, (0, 255, 255), -1)
            cv2.putText(display, "L", (cx - bar_w // 2 - 20, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(display, "R", (cx + bar_w // 2 + 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(display, "0", (cx - 10, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
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
                _save_and_report(marker, output_path, tag1_marker)
                break
        elif key == ord('q') or key == ord('Q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


def _save_and_report(marker, output_path, tag1_marker=None):
    tvec = marker["tvec"].copy()
    rvec = marker["rvec"].copy()
    t1_t, t1_r = None, None
    if tag1_marker is not None:
        t1_t = tag1_marker["tvec"].copy()
        t1_r = tag1_marker["rvec"].copy()
    save_reference_pose(output_path, tvec, rvec, tag_1_tvec=t1_t, tag_1_rvec=t1_r)
    print(f"\nReference pose saved to: {output_path}")
    print(f"  Marker ID: {marker['id']}")
    print(f"  tvec (camera frame): "
          f"x={tvec[0]:.4f}, y={tvec[1]:.4f}, z={tvec[2]:.4f} m")
    print(f"  rvec (Rodrigues):   "
          f"rx={rvec[0]:.4f}, ry={rvec[1]:.4f}, rz={rvec[2]:.4f}")
    if tag1_marker is not None:
        print(f"  Tag_1 ID: {tag1_marker['id']}  "
              f"tvec: x={t1_t[0]:.4f}, y={t1_t[1]:.4f}, z={t1_t[2]:.4f}")


if __name__ == "__main__":
    main()