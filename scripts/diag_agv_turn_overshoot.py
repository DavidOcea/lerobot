#!/usr/bin/env python3
"""
AGV turn-execution overshoot diagnostic.

Purpose
-------
The production visual-align loop commands small corrective turns (e.g. 3.66°)
via ``SeerAGVController.turn(..., mode=0)``, but on the real robot the AGV
physically rotates ~4x that (measured: command 3.66° -> ~14.7° actual), which
drives the persistent phase1/phase2 oscillation. This script quantifies that
overshoot as a function of turn speed, so we can pick a ``turn_speed`` value
that keeps execution accurate.

Method
------
For each (speed, command_angle) pair it:
  1. coarsely re-centers the marker, then measures the marker bearing
     ``dtheta_before`` using the SAME camera + AprilTag pipeline as
     production (``detect_marker`` + ``compute_agv_movement``),
  2. commands ``turn(command_angle, vw=speed, mode=0)``,
  3. waits for completion + a settle delay,
  4. re-measures ``dtheta_after``,
  5. records ``actual = dtheta_before - dtheta_after`` and the overshoot
     ratio ``actual / command``.

A ratio near 1.0 means accurate execution; >1.0 means overshoot. The summary
table is grouped by turn speed so you can read off the fastest speed whose
ratio is ~1.0 with small spread.

Note on settle
--------------
The sweep uses a FIXED, generous settle delay (``--settle``, default 0.8s) so
the measured overshoot reflects the turn execution itself, not an under-settle.
Production currently sleeps 0.3s — you can pass ``--settle 0.3`` to reproduce
that if you want to isolate the settle effect separately.

Usage
-----
Run ON the robot (needs the head camera + AGV reachable over TCP)::

    python scripts/diag_agv_turn_overshoot.py --config configs/task_agent_tasks.yaml

Use ``--host`` to override the AGV IP from the YAML, ``--speeds`` / ``--angles``
to change the sweep, ``--reps`` to repeat each (speed, angle) cell.

WARNING: this MOVES the AGV (rotation only, small angles). Run it with the
robot in a clear area and the emergency stop reachable.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

import numpy as np

from lerobot.tasks.config import VisualAlignConfig, load_config_from_yaml
from lerobot.robots import make_robot_from_config
from lerobot.robots.agv.seer_agv_controller import SeerAGVController
from lerobot.agent.visual_align import (
    _get_detector,
    compute_agv_movement,
    detect_marker,
)

DEG_TO_RAD = math.pi / 180.0
HEAD_CAM_KEY = "head_cam"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("diag_turn")


# ── camera / bearing helpers ─────────────────────────────────────────────


def _camera_key(robot) -> str:
    cams = getattr(robot, "cameras", {}) or {}
    if HEAD_CAM_KEY in cams:
        return HEAD_CAM_KEY
    keys = list(cams.keys())
    if keys:
        logger.warning("'%s' not in cameras, using first key '%s'", HEAD_CAM_KEY, keys[0])
        return keys[0]
    return HEAD_CAM_KEY


def _grab_bgr(robot):
    obs = robot.get_observation()
    img = obs.get("images", {}).get(_camera_key(robot))
    if img is None:
        return None
    return img if img.dtype == np.uint8 else img.astype(np.uint8)


def measure_bearing(robot, va: VisualAlignConfig, detector) -> float | None:
    """Bearing to marker in degrees (+ = left/CCW). None if marker not found."""
    bgr = _grab_bgr(robot)
    if bgr is None:
        return None
    marker = detect_marker(bgr, va, detector)
    if marker is None:
        return None
    dtheta_deg, _ = compute_agv_movement(marker["tvec"], marker["rvec"], va)
    return dtheta_deg


def turn_and_wait(agv, angle_deg: float, speed_deg_s: float, settle: float) -> None:
    """Command a signed turn and block until it finishes + settle."""
    vw = speed_deg_s * DEG_TO_RAD
    if angle_deg < 0:
        vw = -vw
    agv.turn(angle=abs(angle_deg) * DEG_TO_RAD, vw=vw, mode=0)
    agv.wait_for_turn_complete(timeout=10.0)
    time.sleep(settle)


def find_marker(robot, agv, va, detector, search_speed: float) -> bool:
    """Locate the marker: try in place, then sweep left. True if found."""
    for _ in range(3):
        if measure_bearing(robot, va, detector) is not None:
            return True
        time.sleep(0.3)
    step = va.search_turn_step or 10.0
    max_turn = va.search_max_turn or 90.0
    turned = 0.0
    while turned < max_turn:
        turn_and_wait(agv, step, search_speed, 0.5)
        turned += step
        if measure_bearing(robot, va, detector) is not None:
            logger.warning("Marker found after %.0f° search", turned)
            return True
    return False


def center_marker(
    robot, agv, va, detector, center_speed: float,
    tol: float = 2.0, max_iters: int = 4,
) -> bool:
    """Coarse center the marker so test turns start near 0° bearing."""
    for _ in range(max_iters):
        d = measure_bearing(robot, va, detector)
        if d is None:
            return False
        if abs(d) <= tol:
            return True
        turn_and_wait(agv, d, center_speed, 0.6)
    return True


# ── main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AGV turn overshoot diagnostic")
    p.add_argument(
        "--config", default="configs/visual_align_test.yaml",
        help="orchestrator YAML with a visual_align task "
             "(source of robot + AGV + marker params)",
    )
    p.add_argument("--host", default=None, help="override AGV IP from YAML")
    p.add_argument(
        "--speeds", default="15,10,7,5,3",
        help="comma-separated turn speeds in deg/s to sweep (default %(default)s)",
    )
    p.add_argument(
        "--angles", default="5,-5,10,-10",
        help="comma-separated commanded turn angles in deg, signed (default %(default)s)",
    )
    p.add_argument("--reps", type=int, default=2, help="repeats per (speed, angle) cell")
    p.add_argument("--settle", type=float, default=0.8, help="settle seconds after each turn")
    p.add_argument("--center-speed", type=float, default=3.0, help="deg/s used while re-centering")
    p.add_argument("--yes", action="store_true", help="skip the start confirmation prompt")
    return p.parse_args()


def _parse_csv(s: str, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def main() -> int:
    args = parse_args()

    speeds = _parse_csv(args.speeds)
    angles = _parse_csv(args.angles)

    cfg = load_config_from_yaml(args.config)

    # ── VisualAlignConfig (marker + camera-AGV transform) ─────────────
    va: VisualAlignConfig | None = None
    for task in getattr(cfg, "tasks", []) or []:
        vc = getattr(task, "visual_align_config", None)
        if vc is not None:
            va = vc
            break
    if va is None:
        logger.warning(
            "No visual_align task found in %s — using default VisualAlignConfig "
            "(marker_size / marker_id / camera offsets may NOT match production!)",
            args.config,
        )
        va = VisualAlignConfig()

    host = args.host or cfg.agv_config.host

    print("=" * 78)
    print("AGV turn overshoot diagnostic")
    print("=" * 78)
    print(f"  config       : {args.config}")
    print(f"  AGV host     : {host}")
    print(f"  marker_id    : {va.marker_id}  size={va.marker_size}m  family={va.marker_family}")
    print(f"  speeds (deg/s): {speeds}")
    print(f"  angles (deg) : {angles}")
    print(f"  reps/cell    : {args.reps}   settle={args.settle}s")
    print("-" * 78)
    print("This MOVES the AGV (rotation only). Ensure the area is clear.")
    if not args.yes:
        input("Press Enter to start (Ctrl-C to abort)... ")

    # ── connect hardware ──────────────────────────────────────────────
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()

    agv = SeerAGVController(
        host=host,
        port=cfg.agv_config.port,
        connection_timeout=cfg.agv_config.connection_timeout,
        read_timeout=cfg.agv_config.read_timeout,
        auto_reconnect=cfg.agv_config.auto_reconnect,
    )
    if not agv.connect():
        print("ERROR: AGV connection failed")
        return 1

    detector = _get_detector(va.marker_family)

    try:
        if not find_marker(robot, agv, va, detector, args.center_speed):
            print("ERROR: could not find the marker. Aborting (no motion issued).")
            return 1
        print("Marker located. Beginning sweep...\n")

        results = []

        for speed in speeds:
            print(f"\n--- turn_speed = {speed:.0f} deg/s ---")
            for angle in angles:
                for rep in range(args.reps):
                    # re-center before each turn so the test starts near 0°
                    if not center_marker(robot, agv, va, detector, args.center_speed):
                        print("  [re-center failed: marker lost]")
                        find_marker(robot, agv, va, detector, args.center_speed)
                        continue

                    before = measure_bearing(robot, va, detector)
                    if before is None:
                        print("  [before measure failed: marker lost]")
                        continue

                    turn_and_wait(agv, angle, speed, args.settle)

                    after = measure_bearing(robot, va, detector)
                    if after is None:
                        print(f"  cmd {angle:+6.1f}°  [after measure failed: marker lost]")
                        find_marker(robot, agv, va, detector, args.center_speed)
                        continue

                    actual = before - after
                    ratio = actual / angle if angle != 0 else float("nan")
                    results.append({
                        "speed": speed, "command": angle, "actual": actual,
                        "ratio": ratio, "before": before, "after": after,
                    })
                    print(
                        f"  cmd {angle:+6.1f}° → actual {actual:+7.1f}°  "
                        f"ratio {ratio:5.2f}   (before {before:+6.1f}° after {after:+6.1f}°)"
                    )

        _summarize(results)

    finally:
        for obj, name in ((agv, "AGV"), (robot, "robot")):
            try:
                disc = getattr(obj, "disconnect", None)
                if disc is not None:
                    disc()
            except Exception:
                pass

    return 0


def _summarize(results: list[dict]) -> None:
    if not results:
        print("\nNo measurements collected.")
        return

    by_speed = defaultdict(list)
    for r in results:
        by_speed[r["speed"]].append(r)

    print("\n" + "=" * 78)
    print("SUMMARY (grouped by turn speed)")
    print("=" * 78)
    hdr = (
        f"{'speed':>7} {'n':>3} {'mean|cmd|°':>10} {'mean|act|°':>11} "
        f"{'ratio mean':>11} {'ratio std':>10} {'mean resid°':>11} {'sign-flips':>10}"
    )
    print(hdr)
    print("-" * 78)
    for speed in sorted(by_speed, reverse=True):
        rows = by_speed[speed]
        n = len(rows)
        mean_cmd = float(np.mean([abs(r["command"]) for r in rows]))
        mean_act = float(np.mean([abs(r["actual"]) for r in rows]))
        ratios = [r["ratio"] for r in rows]
        mean_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))
        mean_resid = float(np.mean([abs(r["actual"] - r["command"]) for r in rows]))
        flips = sum(1 for r in rows if r["actual"] * r["command"] < 0)
        print(
            f"{speed:>6.0f} {n:>3} {mean_cmd:>10.2f} {mean_act:>11.2f} "
            f"{mean_ratio:>11.2f} {std_ratio:>10.2f} {mean_resid:>11.2f} {flips:>10}"
        )
    print("=" * 78)
    print("Reading: ratio mean ≈ 1.0 = accurate;  > 1.0 = overshoot.")
    print("Pick the FASTEST speed whose ratio ≈ 1.0 with small std (and no sign-flips).")


if __name__ == "__main__":
    sys.exit(main())
