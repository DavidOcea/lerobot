#!/usr/bin/env python3
"""Capture ROI regions for workpiece position classification.

Opens the head camera, displays a live frame, and lets you draw
rectangular ROIs with the mouse.  Press S to save to a JSON file
reusable by the ``yolo_roi`` classifier.

Usage:
    python -m lerobot.scripts.capture_roi_regions \
        --config configs/agv_class_Adc.yaml \
        --output roi_regions_long.json

Mouse controls during preview:
    Left-drag  → draw a new ROI rectangle
    Right-click → remove the last ROI
    S          → save to --output and exit
    Q          → quit without saving
"""

import argparse
import json
import time

import cv2
import numpy as np

from lerobot.tasks.config import load_config_from_yaml
from lerobot.robots import make_robot_from_config


# ── global state (shared between mouse callback and main loop) ──
_rois: list[dict[str, int]] = []
_drawing: bool = False
_start_x: int = 0
_start_y: int = 0
_temp_rect: tuple[int, int, int, int] | None = None  # x, y, w, h


def _mouse_callback(event, x, y, flags, param):
    global _drawing, _start_x, _start_y, _temp_rect, _rois
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _start_x, _start_y = x, y
        _temp_rect = None
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _temp_rect = (
            min(_start_x, x),
            min(_start_y, y),
            abs(x - _start_x),
            abs(y - _start_y),
        )
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        if _temp_rect is not None and _temp_rect[2] > 5 and _temp_rect[3] > 5:
            _rois.append({
                "x": _temp_rect[0],
                "y": _temp_rect[1],
                "w": _temp_rect[2],
                "h": _temp_rect[3],
            })
            print(f"  ROI #{len(_rois)}: x={_temp_rect[0]}, y={_temp_rect[1]}, "
                  f"w={_temp_rect[2]}, h={_temp_rect[3]}")
        _temp_rect = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _rois:
            removed = _rois.pop()
            print(f"  Removed last ROI (was x={removed['x']}, y={removed['y']})")


def parse_args():
    p = argparse.ArgumentParser(description="Capture ROI regions for YOLO-ROI classifier")
    p.add_argument("--config", required=True, help="YAML config with robot_config")
    p.add_argument("--output", default="roi_regions.json", help="Output JSON path")
    return p.parse_args()


def main():
    global _rois
    args = parse_args()

    orchestrator_cfg = load_config_from_yaml(args.config)
    robot_cfg = orchestrator_cfg.robot_config

    print("Connecting to robot...")
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("Robot connected. Waiting for first frame...")

    # Wait for a valid camera frame
    frame = None
    for _ in range(50):
        obs = robot.get_observation()
        img = obs.get("images", {}).get("head_cam")
        if img is not None:
            frame = img if img.dtype == np.uint8 else img.astype(np.uint8)
            break
        time.sleep(0.1)

    if frame is None:
        print("ERROR: No head_cam image after 5 seconds")
        robot.disconnect()
        return

    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")
    print("  LEFT-drag  = draw ROI")
    print("  RIGHT-click = undo last ROI")
    print("  S = save & exit,  Q = quit without saving")

    cv2.namedWindow("Capture ROI Regions")
    cv2.setMouseCallback("Capture ROI Regions", _mouse_callback)

    while True:
        display = frame.copy()

        # Draw confirmed ROIs (green)
        for i, r in enumerate(_rois):
            cv2.rectangle(display, (r["x"], r["y"]),
                          (r["x"] + r["w"], r["y"] + r["h"]),
                          (0, 255, 0), 2)
            cv2.putText(display, f"ROI{i+1}", (r["x"] + 4, r["y"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw in-progress rectangle (blue)
        if _temp_rect is not None:
            cv2.rectangle(display,
                          (_temp_rect[0], _temp_rect[1]),
                          (_temp_rect[0] + _temp_rect[2],
                           _temp_rect[1] + _temp_rect[3]),
                          (255, 0, 0), 1)

        # Help text
        cv2.putText(display, "L-drag=draw  R-click=undo  S=save  Q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Total ROIs: {len(_rois)}", (w - 180, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Capture ROI Regions", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s') or key == ord('S'):
            if not _rois:
                print("No ROIs drawn — nothing to save.")
                continue
            data = {
                "image_width": w,
                "image_height": h,
                "regions": {},
            }
            for i, r in enumerate(_rois):
                label = input(f"  Label for ROI #{i+1}: ").strip()
                if not label:
                    label = f"roi_{i+1}"
                data["regions"][label] = r

            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(_rois)} ROIs to {args.output}")
            break

        elif key == ord('q') or key == ord('Q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
