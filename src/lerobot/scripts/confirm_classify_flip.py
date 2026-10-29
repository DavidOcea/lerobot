#!/usr/bin/env python3
"""End-to-end confirmation that classifier.py's color flip is the bug.

On robot1, reads a few head_cam frames (RGB) and compares:

1. ``classifier.classify(frame)`` — the CURRENT deployment path. It flips the
   frame RGB->BGR before feeding the RGB model, so confidence collapses below
   the 0.15 threshold and it returns ``no_detection`` (conf 0.0).

2. No-flip inference — the SAME model fed the raw RGB frame directly. This is
   what the code SHOULD do. Confidence is ~0.95 and it returns a real label.

A gap of >=5x between no-flip and flip raw confidences confirms the flip is the
root cause, and directly ties the deployment code to the diagnosis.

Usage (on robot1):
    python -m lerobot.scripts.confirm_classify_flip \
        --config lerobot/configs/<robot1 config>.yaml \
        --model /path/on/robot/box_stack.onnx \
        --frames 3
"""

import argparse
import faulthandler
import time

import cv2
import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.tasks.config import load_config_from_yaml

# classifier.py imports ``lerobot.agent.visual_align`` at module top, which drags
# in agent/__init__ → orchestrator → classifier — a circular import. It only
# triggers when classifier is imported FIRST (as this standalone script does).
# The robot's normal entry point imports the agent package first, so we mirror
# that: load `lerobot.agent` before classifier to put it in sys.modules and
# break the cycle. Without this line you get:
#   ImportError: cannot import name 'make_classifier' from partially initialized
#   module 'lerobot.tasks.classifier'
import lerobot.agent  # noqa: F401
from lerobot.tasks.classifier import YOLOClassifier

faulthandler.enable()

INPUT_SIZE = 640
DEFAULT_CLASSES = ["box", "box_stack"]


def parse_args():
    p = argparse.ArgumentParser(description="Confirm classifier.py color-flip bug end-to-end")
    p.add_argument("--config", required=True,
                   help="Task YAML config (must contain robot_config with head_cam)")
    p.add_argument("--model", required=True, help="box_stack.onnx path")
    p.add_argument("--camera", default="head_cam",
                   help="Camera key in observation['images'] (default: head_cam)")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                   help="Comma-separated class names, index = class id")
    p.add_argument("--conf", type=float, default=0.15,
                   help="Confidence threshold (default: 0.15, matches YOLOClassifier)")
    p.add_argument("--frames", type=int, default=3,
                   help="Frames to average the verdict over")
    return p.parse_args()


def make_blob(frame, flip):
    """frame -> NCHW blob. flip=True treats the frame as BGR and converts to RGB
    (this is exactly what classifier.py does, and is wrong when frame is RGB)."""
    if flip:
        frame = frame[:, :, ::-1].copy()
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0)


def raw_max_conf(session, frame, flip):
    """Max class score over all anchors for a given color interpretation."""
    preds = session.run(None, {"images": make_blob(frame, flip)})[0][0]
    return float(preds[4:].max())


def decode_best(session, frame, flip, classes, conf_threshold):
    """Full decode (post-NMS) for a given color interpretation -> (label, conf).

    Mirrors YOLOClassifier.classify()'s postprocessing exactly, except the
    color handling is controlled by `flip` instead of hardcoded.
    """
    preds = session.run(None, {"images": make_blob(frame, flip)})[0][0]
    nc = len(classes)
    boxes, scores, cls_ids = [], [], []
    for i in range(preds.shape[1]):
        cx, cy, bw, bh = preds[0:4, i]
        cls_conf = preds[4:4 + nc, i]
        mc = float(cls_conf.max())
        if mc < conf_threshold:
            continue
        cls_id = int(cls_conf.argmax())
        boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
        scores.append(mc)
        cls_ids.append(cls_id)
    if not boxes:
        return "no_detection", 0.0
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
    if len(idxs) == 0:
        return "no_detection", 0.0
    best = max(idxs.flatten(), key=lambda i: scores[i])
    return classes[cls_ids[best]], scores[best]


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print("Connecting to robot...")
    cfg = load_config_from_yaml(args.config)
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print("Robot connected.\n")

    classifier = YOLOClassifier(model_path=args.model, classes=classes,
                                conf_threshold=args.conf)
    if not classifier._init_session():
        print("FAILED to load ONNX model")
        robot.disconnect()
        return
    session = classifier._session
    print(f"ONNX providers: {session.get_providers()}")
    print(f"classes: {classes}  conf_threshold={args.conf}\n")

    # Warm up one frame so the async read thread is populated.
    robot.get_observation()
    time.sleep(0.3)

    cur_confs, noflip_raws, flip_raws = [], [], []
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

        # 1) Current deployment path — has the hardcoded flip.
        result = classifier.classify(frame)

        # 2) No-flip — what the code should do.
        noflip_label, noflip_conf = decode_best(session, frame, flip=False,
                                                classes=classes,
                                                conf_threshold=args.conf)
        noflip_raw = raw_max_conf(session, frame, flip=False)
        flip_raw = raw_max_conf(session, frame, flip=True)

        cur_confs.append(result.confidence)
        noflip_raws.append(noflip_raw)
        flip_raws.append(flip_raw)

        print(f"[{i}]")
        print(f"    classify() 当前路径(带翻转): label={result.label:<14s} conf={result.confidence:.3f}")
        print(f"    无翻转 正确路径           : label={noflip_label:<14s} conf={noflip_conf:.3f}")
        print(f"    raw max conf:  no-flip={noflip_raw:.3f}   flip={flip_raw:.3f}")

    robot.disconnect()

    avg_cur = sum(cur_confs) / max(n, 1)
    avg_noflip = sum(noflip_raws) / max(n, 1)
    avg_flip = sum(flip_raws) / max(n, 1)
    print("\n===== 结论 =====")
    print(f"当前 classify() 平均 conf = {avg_cur:.3f}  (带翻转 → 多返回 no_detection)")
    print(f"无翻转 raw 平均 conf      = {avg_noflip:.3f}")
    print(f"翻转 raw 平均 conf        = {avg_flip:.3f}")
    if avg_noflip >= 5.0 * max(avg_flip, 1e-6):
        print(f"=> 坐实：无翻转/翻转差距 {avg_noflip / max(avg_flip, 1e-6):.1f}x >= 5x，"
              f"classifier.py 的翻转是 bug；删掉翻转后 classify() 应回到 ~{avg_noflip:.2f}")
    else:
        print("=> 差距不明显，需人工复核（可能场景无目标，或模型对颜色不敏感）")


if __name__ == "__main__":
    main()
