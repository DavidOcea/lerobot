#!/usr/bin/env python3
"""One-shot RGB/BGR diagnostic for the robot's head camera.

Reads a few frames from ``robot.get_observation()["images"][camera]`` and
determines whether the camera returns RGB or BGR, two independent ways:

1. **Model-as-oracle (decisive)**: the box_stack model expects RGB (RGB input
   ~0.94 confidence vs BGR ~0.35). We run the frame through the model twice —
   as-is (no flip) and channel-flipped — and see which interpretation yields the
   higher confidence.
     - as-is wins   -> camera returns RGB  -> feed model directly (no flip)
     - flipped wins -> camera returns BGR  -> flip BGR->RGB before feeding model

2. **Saved images (human check)**: writes the raw frame and its channel-swapped
   twin as PNG. Whichever looks like the real scene confirms the format.

Prerequisite: a box should actually be in the camera's view so the model has
something to detect — otherwise both confidences are near background noise.

Usage:
    python -m lerobot.scripts.diagnose_camera_format \
        --config lerobot/configs/agv_class_longloushang_yanshi_V0.6.yaml \
        --model /path/on/robot/box_stack.onnx
"""

import argparse
import faulthandler
import time

import cv2
import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.tasks.config import load_config_from_yaml

faulthandler.enable()

INPUT_SIZE = 640


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose robot camera RGB/BGR format")
    p.add_argument("--config", required=True, help="Task YAML config (must contain robot_config)")
    p.add_argument("--model", required=True, help="Path to box_stack ONNX model")
    p.add_argument("--camera", default="head_cam", help="Camera key in observation['images']")
    p.add_argument("--output", default="camera_format_diag", help="Output filename prefix for saved PNGs")
    p.add_argument("--frames", type=int, default=3, help="Frames to average the verdict over")
    return p.parse_args()


def make_blob(frame, flip):
    """frame -> NCHW RGB blob. flip=True treats the frame as BGR and converts to RGB."""
    if flip:
        frame = frame[:, :, ::-1]  # BGR -> RGB
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0)


def main():
    args = parse_args()

    import onnxruntime as ort
    session = ort.InferenceSession(args.model)
    print(f"ONNX providers: {session.get_providers()}")

    print("Connecting to robot...")
    cfg = load_config_from_yaml(args.config)
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print("Robot connected.\n")

    # Warm up one frame so the async read thread is populated.
    robot.get_observation()
    time.sleep(0.3)

    as_is_wins = flip_wins = ties = 0
    as_is_sum = flip_sum = 0.0
    n = 0

    for i in range(args.frames):
        obs = robot.get_observation()
        frame = obs.get("images", {}).get(args.camera)
        if frame is None:
            print(f"[{i}] no frame — waiting")
            time.sleep(0.3)
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        n += 1
        h, w = frame.shape[:2]

        # Channel means (channel order unknown: could be BGR or RGB)
        c0, c1, c2 = frame[..., 0].mean(), frame[..., 1].mean(), frame[..., 2].mean()

        preds_as_is = session.run(None, {"images": make_blob(frame, flip=False)})[0][0]
        preds_flip = session.run(None, {"images": make_blob(frame, flip=True)})[0][0]
        conf_as_is = float(preds_as_is[4:].max())  # max class score over all anchors
        conf_flip = float(preds_flip[4:].max())

        as_is_sum += conf_as_is
        flip_sum += conf_flip
        if conf_as_is > conf_flip:
            as_is_wins += 1
        elif conf_flip > conf_as_is:
            flip_wins += 1
        else:
            ties += 1

        print(f"[{i}] {w}x{h}  ch0={c0:.1f} ch1={c1:.1f} ch2={c2:.1f}  "
              f"| as-is(no flip)={conf_as_is:.3f}  flipped={conf_flip:.3f}")

        if i == 0:
            # Human check: raw frame as-is vs channel-swapped twin.
            cv2.imwrite(f"{args.output}_as_observed.png", frame)
            cv2.imwrite(f"{args.output}_channels_swapped.png", frame[:, :, ::-1])

    robot.disconnect()

    avg_as_is = as_is_sum / max(n, 1)
    avg_flip = flip_sum / max(n, 1)
    print("\n===== VERDICT =====")
    print(f"as-is (no flip) higher conf : {as_is_wins}/{n}  avg={avg_as_is:.3f}")
    print(f"flipped       higher conf   : {flip_wins}/{n}  avg={avg_flip:.3f}")
    if as_is_wins > flip_wins:
        print("=> camera returns RGB  (feed model directly, NO channel flip)")
    elif flip_wins > as_is_wins:
        print("=> camera returns BGR  (flip BGR->RGB before feeding model)")
    else:
        print("=> inconclusive — compare the two saved PNGs by eye")
    print(f"\nSaved: {args.output}_as_observed.png  /  {args.output}_channels_swapped.png")
    print("If '_as_observed' looks wrong but '_channels_swapped' looks right -> camera is RGB.")


if __name__ == "__main__":
    main()
