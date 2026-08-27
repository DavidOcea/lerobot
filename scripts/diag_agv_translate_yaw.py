#!/usr/bin/env python3
"""
AGV translate-yaw + small-turn diagnostic.

Purpose
-------
The production visual-align loop oscillates: every "turn + drive forward"
iteration flips the marker from one side to the other (iter0 dtheta ~ +3.3°
-> iter1 ~ -6.5°). Two competing explanations remain after the turn-only
diagnostic (``diag_agv_turn_overshoot.py``) showed large turns are accurate
(~1.14x):

  1. The **forward translate** itself yaws the AGV several degrees (the
     marker's lateral offset shifts during a "straight" drive), or
  2. **Small turns** (1-2°, the size the production loop actually commands)
     overshoot far more than the 5-10° turns the turn diagnostic tested.

This script isolates both:

  A. FORWARD test  — center the marker, command ``translate(dist, vx, mode)``
     with NO turn, and measure the marker's full AGV-frame position before and
     after.  A straight drive should keep ``dy_agv`` constant; any change in
     ``dy_agv`` over the forward distance is an implied yaw:
        yaw_deg = atan2(dy_after - dy_before, dx_before - dx_after).
     Sweeps mode 0 and 1 so we can tell whether localization mode changes it.

  B. SMALL-TURN test — center the marker, command ``turn(angle, mode=0)`` for
     angles 1°, 1.5°, 2°, 3°, and measure actual/command ratio (same method as
     the turn diagnostic).  If the ratio balloons at small angles, that's the
     real root cause.

Run ON the robot (needs head camera + AGV over TCP).

WARNING: this MOVES the AGV.  Clear area + emergency stop reachable.
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
    _marker_to_agv_xy,
    detect_marker,
)

DEG_TO_RAD = math.pi / 180.0
HEAD_CAM_KEY = "head_cam"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("diag_translate_yaw")


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


def measure_marker_xy(robot, va, detector):
    """Marker position in AGV ground frame (dx, dy). None if not found."""
    bgr = _grab_bgr(robot)
    if bgr is None:
        return None
    marker = detect_marker(bgr, va, detector)
    if marker is None:
        return None
    dx, dy = _marker_to_agv_xy(marker["tvec"], va)
    return dx, dy


def measure_bearing(robot, va, detector):
    """Bearing to marker in degrees (+ = left/CCW). None if not found."""
    xy = measure_marker_xy(robot, va, detector)
    if xy is None:
        return None
    dx, dy = xy
    return math.atan2(dy, dx) * 180.0 / math.pi


def turn_and_wait(agv, angle_deg, speed_deg_s, settle, mode=0):
    vw = speed_deg_s * DEG_TO_RAD
    if angle_deg < 0:
        vw = -vw
    agv.turn(angle=abs(angle_deg) * DEG_TO_RAD, vw=vw, mode=mode)
    agv.wait_for_turn_complete(timeout=10.0)
    time.sleep(settle)


def translate_and_wait(agv, dist, vx, mode):
    agv.translate(dist=dist, vx=vx, mode=mode)
    agv.wait_for_translate_complete(timeout=15.0)
    time.sleep(0.5)


def find_marker(robot, agv, va, detector, search_speed):
    for _ in range(3):
        if measure_marker_xy(robot, va, detector) is not None:
            return True
        time.sleep(0.3)
    step = va.search_turn_step or 10.0
    max_turn = va.search_max_turn or 90.0
    turned = 0.0
    while turned < max_turn:
        turn_and_wait(agv, step, search_speed, 0.5)
        turned += step
        if measure_marker_xy(robot, va, detector) is not None:
            logger.warning("Marker found after %.0f° search", turned)
            return True
    return False


def center_marker(robot, agv, va, detector, center_speed, tol=1.0, max_iters=5):
    """Center the marker near 0° bearing (also pulls to a sane distance)."""
    for _ in range(max_iters):
        xy = measure_marker_xy(robot, va, detector)
        if xy is None:
            return False
        d = math.atan2(xy[1], xy[0]) * 180.0 / math.pi
        if abs(d) <= tol:
            return True
        turn_and_wait(agv, d, center_speed, 0.6)
    return True


# ── main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AGV translate-yaw + small-turn diagnostic")
    p.add_argument(
        "--config", default="configs/visual_align_test.yaml",
        help="orchestrator YAML with a visual_align task",
    )
    p.add_argument("--host", default=None, help="override AGV IP from YAML")
    p.add_argument(
        "--fwd-dists", default="0.05,0.10",
        help="comma-separated forward distances (m) for the translate test",
    )
    p.add_argument(
        "--fwd-modes", default="0,1",
        help="comma-separated translate modes to sweep (0=odometry, 1=localization)",
    )
    p.add_argument("--fwd-speed", type=float, default=0.15, help="translate speed m/s")
    p.add_argument(
        "--turn-angles", default="1,1.5,2,3",
        help="comma-separated small turn angles (deg) for the turn test",
    )
    p.add_argument("--turn-speed", type=float, default=15.0, help="turn speed deg/s")
    p.add_argument("--reps", type=int, default=3, help="repeats per cell")
    p.add_argument("--center-speed", type=float, default=3.0, help="deg/s while centering")
    p.add_argument("--skip-fwd", action="store_true", help="skip the forward-translate test")
    p.add_argument("--skip-turn", action="store_true", help="skip the small-turn test")
    p.add_argument("--yes", action="store_true", help="skip the start confirmation prompt")
    return p.parse_args()


def _parse_csv(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def _load_va(args) -> tuple[VisualAlignConfig, object]:
    cfg = load_config_from_yaml(args.config)
    va = None
    for task in getattr(cfg, "tasks", []) or []:
        vc = getattr(task, "visual_align_config", None)
        if vc is not None:
            va = vc
            break
    if va is None:
        logger.warning("No visual_align task in %s — using default VisualAlignConfig", args.config)
        va = VisualAlignConfig()
    return va, cfg


def _run_forward_test(robot, agv, va, detector, args, fwd_dists, fwd_modes) -> list[dict]:
    results = []
    print("\n" + "=" * 78)
    print("A. FORWARD-TRANSLATE TEST (marker lateral shift during straight drive)")
    print("=" * 78)
    for mode in fwd_modes:
        for dist in fwd_dists:
            for rep in range(args.reps):
                if not center_marker(robot, agv, va, detector, args.center_speed):
                    print("  [re-center failed: marker lost]")
                    find_marker(robot, agv, va, detector, args.center_speed)
                    continue
                before = measure_marker_xy(robot, va, detector)
                if before is None:
                    print("  [before measure failed]")
                    continue

                translate_and_wait(agv, dist, args.fwd_speed, mode)

                after = measure_marker_xy(robot, va, detector)
                if after is None:
                    print(f"  mode={mode} dist={dist}m  [after measure failed: marker lost]")
                    find_marker(robot, agv, va, detector, args.center_speed)
                    continue

                fwd_delta = before[0] - after[0]   # how much closer the marker got
                lat_delta = after[1] - before[1]   # lateral shift (0 = straight)
                yaw_deg = math.atan2(lat_delta, max(fwd_delta, 1e-6)) * 180.0 / math.pi
                results.append({
                    "mode": mode, "dist": dist, "fwd_delta": fwd_delta,
                    "lat_delta": lat_delta, "yaw_deg": yaw_deg,
                })
                print(
                    f"  mode={mode} dist={dist:4.2f}m  fwd_delta={fwd_delta:+5.3f}m "
                    f"lat_delta={lat_delta:+5.3f}m  → implied yaw {yaw_deg:+6.2f}°"
                )
    return results


def _run_turn_test(robot, agv, va, detector, args, turn_angles) -> list[dict]:
    results = []
    print("\n" + "=" * 78)
    print("B. SMALL-TURN TEST (actual/command ratio at production-sized angles)")
    print("=" * 78)
    for angle in turn_angles:
        for rep in range(args.reps):
            if not center_marker(robot, agv, va, detector, args.center_speed):
                print("  [re-center failed]")
                find_marker(robot, agv, va, detector, args.center_speed)
                continue
            before = measure_bearing(robot, va, detector)
            if before is None:
                print("  [before measure failed]")
                continue
            turn_and_wait(agv, angle, args.turn_speed, 0.8, mode=0)
            after = measure_bearing(robot, va, detector)
            if after is None:
                print(f"  cmd {angle:+5.1f}°  [after measure failed: marker lost]")
                find_marker(robot, agv, va, detector, args.center_speed)
                continue
            actual = before - after
            ratio = actual / angle if angle != 0 else float("nan")
            results.append({
                "command": angle, "actual": actual, "ratio": ratio,
                "before": before, "after": after,
            })
            print(
                f"  cmd {angle:+5.1f}° → actual {actual:+6.1f}°  ratio {ratio:5.2f}  "
                f"(before {before:+6.1f}° after {after:+6.1f}°)"
            )
    return results


def main() -> int:
    args = parse_args()
    fwd_dists = _parse_csv(args.fwd_dists)
    fwd_modes = _parse_csv(args.fwd_modes, cast=int)
    turn_angles = _parse_csv(args.turn_angles)

    va, cfg = _load_va(args)
    host = args.host or cfg.agv_config.host

    print("=" * 78)
    print("AGV translate-yaw + small-turn diagnostic")
    print("=" * 78)
    print(f"  config        : {args.config}")
    print(f"  AGV host      : {host}")
    print(f"  marker_id     : {va.marker_id}  size={va.marker_size}m")
    print(f"  fwd dists (m) : {fwd_dists}   modes={fwd_modes}")
    print(f"  turn angles(°) : {turn_angles}  speed={args.turn_speed}°/s")
    print(f"  reps/cell     : {args.reps}")
    print("-" * 78)
    print("This MOVES the AGV (small translate + small turn). Clear area required.")
    if not args.yes:
        input("Press Enter to start (Ctrl-C to abort)... ")

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
        print("Marker located. Beginning tests...\n")

        if not args.skip_fwd:
            fwd_results = _run_forward_test(robot, agv, va, detector, args,
                                            fwd_dists, fwd_modes)
            _summarize_fwd(fwd_results)

        if not args.skip_turn:
            turn_results = _run_turn_test(robot, agv, va, detector, args, turn_angles)
            _summarize_turn(turn_results)

    finally:
        for obj, name in ((agv, "AGV"), (robot, "robot")):
            try:
                disc = getattr(obj, "disconnect", None)
                if disc is not None:
                    disc()
            except Exception:
                pass

    return 0


def _summarize_fwd(results: list[dict]) -> None:
    if not results:
        print("\nNo forward-test measurements.")
        return
    print("\n" + "-" * 78)
    print("FORWARD TEST SUMMARY (implied yaw during straight drive)")
    print("-" * 78)
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    for mode in sorted(by_mode):
        rows = by_mode[mode]
        yaws = [r["yaw_deg"] for r in rows]
        lats = [abs(r["lat_delta"]) for r in rows]
        print(
            f"  mode={mode}: n={len(rows)}  "
            f"mean|yaw|={float(np.mean(np.abs(yaws))):5.2f}°  "
            f"max|yaw|={float(np.max(np.abs(yaws))):5.2f}°  "
            f"mean|lat_shift|={float(np.mean(lats)):5.3f}m"
        )
    print("-" * 78)
    print("Reading: |yaw| ≈ 0 = translate stays straight. A few degrees of |yaw|")
    print("per 0.05-0.10m of drive = the translate itself is yawing the AGV.")


def _summarize_turn(results: list[dict]) -> None:
    if not results:
        print("\nNo turn-test measurements.")
        return
    print("\n" + "-" * 78)
    print("SMALL-TURN TEST SUMMARY")
    print("-" * 78)
    by_angle = defaultdict(list)
    for r in results:
        by_angle[abs(r["command"])].append(r)
    for angle in sorted(by_angle):
        rows = by_angle[angle]
        ratios = [r["ratio"] for r in rows]
        print(
            f"  |cmd|={angle:4.1f}°: n={len(rows)}  ratio mean={float(np.mean(ratios)):5.2f} "
            f"std={float(np.std(ratios)):5.2f}  mean|resid|={float(np.mean([abs(r['actual']-r['command']) for r in rows])):5.2f}°"
        )
    print("-" * 78)
    print("Reading: ratio >> 1.5 at small angles = small turns overshoot far more")
    print("than the 5-10° turns the earlier diagnostic measured.")


if __name__ == "__main__":
    sys.exit(main())
