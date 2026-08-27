#!/usr/bin/env python3
"""
AGV self-reported motion accuracy diagnostic (no camera, no tag).

Why
---
The visual-align loop oscillates ~10° every "turn + drive forward" step, and
neither translate_mode=1 nor turn_mode=1 fixed it. We need to know whether the
AGV *physically* moves the amount we command, WITHOUT involving the camera
(which may itself be miscalibrated via camera_offset_yaw).

This script reads the AGV's OWN localization (API_TASK_STATUS_QUERY -> x, y,
angle) before/after each commanded turn and translate, so actual motion is
compared to commanded motion directly.

  A. TURN test      — command turn(angle, mode), measure actual heading change.
                      actual ≈ command  -> turn is honest; the ~10° swing the
                      camera sees is a camera/measurement problem.
                      actual >> command -> the AGV overshoots (small-angle).

  B. TRANSLATE test — command translate(dist forward), decompose the actual
                      map-frame displacement into forward vs lateral (projected
                      onto the AGV heading). lateral ≈ 0 -> "forward" is really
                      forward. |lateral| ~ |forward| -> the translate drifts
                      sideways (direction error).

Run on the robot with clear floor space: turns are in place; each translate
moves ~0.1-0.2 m forward (plus any lateral drift). NO camera, NO tag needed,
so there is no "tag too far / too close" concern.

WARNING: this MOVES the AGV. Clear area + emergency stop reachable.
The AGV must be localized (loc_state healthy) for get_position() to be valid.
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

from lerobot.tasks.config import load_config_from_yaml
from lerobot.robots.agv.seer_agv_controller import SeerAGVController

DEG_TO_RAD = math.pi / 180.0

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("diag_self_report")


def _wrap_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _read_pose(agv: SeerAGVController) -> tuple[float, float, float]:
    """Return (x, y, theta) in map frame from the AGV's own localization."""
    pos = agv.get_position()
    return pos.x, pos.y, pos.theta


def turn_and_wait(agv, angle_deg, speed_deg_s, settle, mode):
    vw = speed_deg_s * DEG_TO_RAD
    if angle_deg < 0:
        vw = -vw
    agv.turn(angle=abs(angle_deg) * DEG_TO_RAD, vw=vw, mode=mode)
    agv.wait_for_turn_complete(timeout=10.0)
    time.sleep(settle)


def translate_and_wait(agv, dist, vx, mode, settle):
    agv.translate(dist=dist, vx=vx, mode=mode)
    agv.wait_for_translate_complete(timeout=15.0)
    time.sleep(settle)


# ── A. TURN test ─────────────────────────────────────────────────────────

def run_turn_test(agv, args, angles, modes) -> list[dict]:
    results = []
    print("\n" + "=" * 78)
    print("A. TURN TEST (AGV self-reported heading change vs commanded)")
    print("=" * 78)
    for mode in modes:
        for angle in angles:
            for rep in range(args.reps):
                x0, y0, th0 = _read_pose(agv)
                turn_and_wait(agv, angle, args.turn_speed, args.settle, mode)
                x1, y1, th1 = _read_pose(agv)
                actual = _wrap_180(math.degrees(th1 - th0))
                ratio = actual / angle if angle != 0 else float("nan")
                results.append({
                    "mode": mode, "command": angle, "actual": actual,
                    "ratio": ratio,
                })
                print(
                    f"  mode={mode} cmd={angle:+5.1f}° -> actual {actual:+6.1f}°  "
                    f"ratio {ratio:5.2f}  (theta {math.degrees(th0):+7.1f}° -> "
                    f"{math.degrees(th1):+7.1f}°)"
                )
    return results


def _summarize_turn(results: list[dict]) -> None:
    if not results:
        print("\nNo turn measurements.")
        return
    print("\n" + "-" * 78)
    print("TURN SUMMARY (actual/command ratio; 1.0 = honest turn)")
    print("-" * 78)
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    for mode in sorted(by_mode):
        for angle in sorted({abs(r["command"]) for r in by_mode[mode]}):
            rows = [r for r in by_mode[mode] if abs(r["command"]) == angle]
            ratios = [r["ratio"] for r in rows]
            act = [r["actual"] for r in rows]
            print(
                f"  mode={mode} |cmd|={angle:4.1f}°: n={len(rows)}  "
                f"ratio mean={float(np.mean(ratios)):5.2f} std={float(np.std(ratios)):5.2f}  "
                f"mean|actual|={float(np.mean(np.abs(act))):5.1f}°"
            )
    print("-" * 78)
    print("Reading: ratio ~1.0 = turn honest (the camera's ~10° swing is a")
    print("camera/measurement problem). ratio >> 1.5 at |cmd|<=2° = small-angle")
    print("turn overshoot (the AGV itself is over-rotating).")


# ── B. TRANSLATE test ────────────────────────────────────────────────────

def run_translate_test(agv, args, dist, modes) -> list[dict]:
    results = []
    print("\n" + "=" * 78)
    print("B. TRANSLATE TEST (AGV self-reported displacement vs commanded forward)")
    print("=" * 78)
    for mode in modes:
        for rep in range(args.reps):
            x0, y0, th0 = _read_pose(agv)
            translate_and_wait(agv, dist, args.fwd_speed, mode, args.settle)
            x1, y1, th1 = _read_pose(agv)
            dx = x1 - x0
            dy = y1 - y0
            # Project map-frame displacement onto the AGV heading (body frame).
            fwd = dx * math.cos(th0) + dy * math.sin(th0)
            lat = -dx * math.sin(th0) + dy * math.cos(th0)
            drift_deg = math.atan2(lat, fwd) * 180.0 / math.pi
            results.append({
                "mode": mode, "fwd": fwd, "lat": lat, "drift_deg": drift_deg,
            })
            print(
                f"  mode={mode} cmd={dist:4.2f}m -> fwd={fwd:+5.3f}m "
                f"lat={lat:+5.3f}m  drift {drift_deg:+6.1f}°  "
                f"(heading {math.degrees(th0):+7.1f}°)"
            )
    return results


def _summarize_translate(results: list[dict]) -> None:
    if not results:
        print("\nNo translate measurements.")
        return
    print("\n" + "-" * 78)
    print("TRANSLATE SUMMARY (lateral drift during 'straight' forward drive)")
    print("-" * 78)
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    for mode in sorted(by_mode):
        rows = by_mode[mode]
        lats = [abs(r["lat"]) for r in rows]
        fwds = [r["fwd"] for r in rows]
        print(
            f"  mode={mode}: n={len(rows)}  mean fwd={float(np.mean(fwds)):5.3f}m  "
            f"mean|lat|={float(np.mean(lats)):5.3f}m  "
            f"mean|drift|={float(np.mean(np.abs([r['drift_deg'] for r in rows]))):5.1f}°"
        )
    print("-" * 78)
    print("Reading: |lat| ~ 0 = 'forward' is really forward. |lat| ~ |fwd| =")
    print("translate drifts ~45° sideways (direction error in the drive).")


# ── main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AGV self-reported motion accuracy diagnostic")
    p.add_argument(
        "--config", default="../agv_class_online_test.yaml",
        help="orchestrator YAML with a top-level agv_config (host/port). "
             "Default = production config; the in-repo test config points at a "
             "different robot (192.168.2.210), so do NOT use it for the real AGV.",
    )
    p.add_argument("--host", default=None, help="override AGV IP from YAML")
    p.add_argument(
        "--turn-angles", default="1,1.5,2,3,5,10",
        help="comma-separated turn angles (deg) to command",
    )
    p.add_argument(
        "--turn-modes", default="0,1",
        help="comma-separated turn modes to sweep (0=odometry, 1=localization)",
    )
    p.add_argument("--turn-speed", type=float, default=15.0, help="turn speed deg/s")
    p.add_argument(
        "--fwd-dist", type=float, default=0.10,
        help="forward distance (m) per translate rep",
    )
    p.add_argument(
        "--fwd-modes", default="0,1",
        help="comma-separated translate modes to sweep",
    )
    p.add_argument("--fwd-speed", type=float, default=0.15, help="translate speed m/s")
    p.add_argument("--reps", type=int, default=3, help="repeats per cell")
    p.add_argument("--settle", type=float, default=0.5, help="settle time (s) after motion")
    p.add_argument("--skip-turn", action="store_true", help="skip the turn test")
    p.add_argument("--skip-translate", action="store_true", help="skip the translate test")
    p.add_argument("--yes", action="store_true", help="skip the start confirmation prompt")
    return p.parse_args()


def _parse_csv(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def main() -> int:
    args = parse_args()
    turn_angles = _parse_csv(args.turn_angles)
    turn_modes = _parse_csv(args.turn_modes, cast=int)
    fwd_modes = _parse_csv(args.fwd_modes, cast=int)

    cfg = load_config_from_yaml(args.config)
    host = args.host or cfg.agv_config.host

    print("=" * 78)
    print("AGV self-reported motion accuracy diagnostic")
    print("=" * 78)
    print(f"  config       : {args.config}")
    print(f"  AGV host     : {host}  port={cfg.agv_config.port}")
    print(f"  turn angles  : {turn_angles}  modes={turn_modes}  speed={args.turn_speed}°/s")
    print(f"  translate    : {args.fwd_dist}m  modes={fwd_modes}  speed={args.fwd_speed}m/s")
    print(f"  reps/cell    : {args.reps}")
    print("-" * 78)
    print("This MOVES the AGV (in-place turns + short forward drives).")
    print("Clear area required. NO camera / tag needed.")
    if not args.yes:
        input("Press Enter to start (Ctrl-C to abort)... ")

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

    try:
        x, y, th = _read_pose(agv)
        print(f"Initial pose: x={x:.3f} y={y:.3f} theta={math.degrees(th):.1f}°\n")

        if not args.skip_turn:
            _summarize_turn(run_turn_test(agv, args, turn_angles, turn_modes))
        if not args.skip_translate:
            _summarize_translate(run_translate_test(agv, args, args.fwd_dist, fwd_modes))
    finally:
        try:
            agv.disconnect()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
