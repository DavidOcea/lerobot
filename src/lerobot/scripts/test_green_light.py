#!/usr/bin/env python3
"""Live green-light detection test for the color_threshold classifier.

Opens the head camera and shows, in real time, whether the color_threshold
classifier sees the green light on/off inside a fixed pixel ROI. Use this to
visually verify the ROI + hue/saturation/value thresholds before wiring the
gate into the box-pickup task YAML.

Usage:
    python -m lerobot.scripts.test_green_light \
        --config /root/workspace/dc_dir/agv_class_AB1B2_boxcheck_fast.yaml \
        --roi 320 240 80 80

Green box + "GREEN ON" = lit; red box + "GREEN OFF" = off. Press Q to quit.
"""

import argparse

import cv2
import numpy as np

import lerobot.agent  # noqa: F401  break classifier<->orchestrator circular import
from lerobot.tasks.classifier import make_classifier
from lerobot.tasks.config import load_config_from_yaml
from lerobot.robots import make_robot_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Live green-light detection test")
    p.add_argument("--config", required=True, help="YAML config with robot_config")
    p.add_argument("--roi", required=True, nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="Pixel ROI [x y w h] in the camera frame")
    p.add_argument("--camera", default="head_cam", help="Camera key (default: head_cam)")
    p.add_argument("--hue-range", nargs=2, type=int, default=[35, 85], metavar=("LO", "HI"),
                   help="OpenCV H (0-180) green band (default: 35 85)")
    p.add_argument("--sat-min", type=int, default=80, help="Min saturation (default: 80)")
    p.add_argument("--val-min", type=int, default=80, help="Min value/brightness (default: 80)")
    p.add_argument("--fraction-threshold", type=float, default=0.20,
                   help="Min green-pixel fraction to count as lit (default: 0.20)")
    return p.parse_args()


def main():
    args = parse_args()
    roi = args.roi  # [x, y, w, h]

    classifier = make_classifier(
        "color_threshold",
        roi=roi,
        hue_range=args.hue_range,
        saturation_min=args.sat_min,
        value_min=args.val_min,
        color_fraction_threshold=args.fraction_threshold,
        default_label="no_green",
    )

    cfg = load_config_from_yaml(args.config)
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print(f"Robot connected. ROI={roi}  hue={args.hue_range}  "
          f"sat>={args.sat_min}  val>={args.val_min}  frac>{args.fraction_threshold}")
    print("Q = quit")

    cv2.namedWindow("Green Light Test")

    while True:
        obs = robot.get_observation()
        frame = obs.get("images", {}).get(args.camera)
        if frame is None:
            cv2.waitKey(20)
            continue
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        result = classifier.classify(frame)
        lit = result.label == "green"
        frac = result.confidence if lit else 1.0 - result.confidence

        display = frame.copy()
        x, y, w, h = roi
        col = (0, 255, 0) if lit else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), col, 3)

        status = "GREEN ON " if lit else "GREEN OFF"
        cv2.putText(display, f"{status} frac={frac:.2f} (thr={args.fraction_threshold:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)

        # zoomed ROI inset (top-right) so the lamp region is visible up close
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 > x0 and y1 > y0:
            zoom = cv2.resize(frame[y0:y1, x0:x1], None, fx=2.0, fy=2.0,
                              interpolation=cv2.INTER_NEAREST)
            zh, zw = zoom.shape[:2]
            fh, fw = display.shape[:2]
            if zh + 8 < fh and zw + 8 < fw:
                display[8:8 + zh, fw - zw - 8:fw - 8] = zoom
                cv2.rectangle(display, (fw - zw - 8, 8), (fw - 8, 8 + zh), (200, 200, 200), 1)

        cv2.imshow("Green Light Test", display)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
