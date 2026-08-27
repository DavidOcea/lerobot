#!/usr/bin/env python3
"""
Camera measurement stability diagnostic (static + single-turn cross-check).

Why
---
The visual-align loop oscillates ~10° every "turn + drive forward" step, but
the AGV self-report diagnostic (diag_agv_self_report.py, 0420.log) proved the
AGV *motion* is accurate (translate ~2 mm lateral drift, turn ~+1° overshoot).
A fixed calibration constant (camera_offset_yaw / pitch / marker_size) cannot
cause a turn-to-turn oscillation either: a constant rotation cancels when you
look at the *change* between two measurements.

So the ~9° extra swing must come from the camera's per-measurement lateral
readout changing too much. Two candidates remain:

  (a) solvePnP lateral instability — the flat tag is viewed at a steep pitch
      (+ maybe yaw), where the lateral position (tvec[0]) solution is noisy
      while depth (z_cam) and vertical (y_cam) stay stable.
  (b) the camera is not rigid — it rotates relative to the AGV during a turn
      (loose/flexing mount, robot sway).

This script separates them with a THREE-way cross-check, using a clean,
calibration-independent reference: the marker's PIXEL CENTER (u, v) + camera
matrix K gives a bearing ``atan2(cx - u, fx)`` that does NOT go through
solvePnP's tvec at all.

  A. STATIC test — hold still, read the marker ~N times. Report spread of:
       x_cam       (solvePnP tvec[0], the suspect)
       z_cam       (solvePnP tvec[2], depth — expected smooth)
       pixel_bearing (pixel center + K — the clean reference)
       dtheta      (solvePnP -> _marker_to_agv_xy, what the loop actually uses)
     Large x_cam std while pixel_bearing is quiet -> (a) solvePnP lateral noise.

  B. TURN cross-check — command a small turn, compare the AGV's OWN heading
     change (get_position) against two bearing changes:
       cam_dbearing   = dtheta change (solvePnP path)
       pixel_dbearing = pixel-bearing change (pixel-center path)
     A rigid camera + accurate solvePnP must satisfy, for an AGV heading
     change of +Δ (CCW/left):
         pixel_dbearing ≈ cam_dbearing ≈ -Δ
     so define residuals:
         residual_pixel = pixel_dbearing + agv_dheading
         residual_cam   = cam_dbearing   + agv_dheading
     - residual_pixel ≈ 0  AND  residual_cam ≈ 0      -> both fine
     - residual_pixel ≈ 0  BUT  residual_cam is large -> solvePnP lateral
       unstable (pixel data is fine; fix the bearing computation, not the AGV)
     - BOTH residuals large (and similar)             -> the camera moved
       relative to the AGV (non-rigid mount)

Run ON the robot (needs head camera + AGV over TCP), tag in view.

WARNING: Part B MOVES the AGV a few degrees. Clear area + emergency stop.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

import numpy as np

from lerobot.tasks.config import VisualAlignConfig, load_config_from_yaml
from lerobot.robots import make_robot_from_config
from lerobot.robots.agv.seer_agv_controller import SeerAGVController
from lerobot.agent.visual_align import (
    _get_camera_params,
    _get_detector,
    _marker_to_agv_xy,
    detect_marker,
)

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
HEAD_CAM_KEY = "head_cam"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("diag_camera_stability")


# ── camera + marker helpers ─────────────────────────────────────────────

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


def _pixel_center(corners) -> tuple[float, float]:
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    u, v = c.mean(axis=0)
    return float(u), float(v)


def measure_marker(robot, va, detector):
    """Return dict {tvec, pixel_u, pixel_v, dtheta} or None if marker lost.

    dtheta uses the SAME path as the production loop (solvePnP tvec ->
    _marker_to_agv_xy -> atan2). pixel_u/v are the tag center in pixels.
    """
    bgr = _grab_bgr(robot)
    if bgr is None:
        return None
    marker = detect_marker(bgr, va, detector)
    if marker is None:
        return None
    tvec = np.asarray(marker["tvec"], dtype=float)
    u, v = _pixel_center(marker["corners"])
    dx, dy = _marker_to_agv_xy(tvec, va)
    dtheta = math.atan2(dy, dx) * RAD_TO_DEG
    return {
        "tvec": tvec,
        "x_cam": float(tvec[0]),
        "y_cam": float(tvec[1]),
        "z_cam": float(tvec[2]),
        "pixel_u": u,
        "pixel_v": v,
        "dtheta": dtheta,
    }


def pixel_bearing(u, va) -> float:
    """Bearing from pixel center + intrinsics, + = left of optical axis (deg).

    Independent of solvePnP: uses only the tag's pixel center and fx/cx.
    """
    K, _ = _get_camera_params(va)
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    return math.atan2(cx - u, fx) * RAD_TO_DEG


def turn_and_wait(agv, angle_deg, speed_deg_s, settle, mode=0):
    vw = speed_deg_s * DEG_TO_RAD
    if angle_deg < 0:
        vw = -vw
    agv.turn(angle=abs(angle_deg) * DEG_TO_RAD, vw=vw, mode=mode)
    agv.wait_for_turn_complete(timeout=10.0)
    time.sleep(settle)


def _read_heading(agv) -> float:
    return float(agv.get_position().theta)  # radians, map frame


def _angle_diff_deg(a_rad, b_rad) -> float:
    """Wrapped (a - b) in degrees."""
    return math.atan2(math.sin(a_rad - b_rad), math.cos(a_rad - b_rad)) * RAD_TO_DEG


def find_marker(robot, agv, va, detector, search_speed):
    for _ in range(3):
        if measure_marker(robot, va, detector) is not None:
            return True
        time.sleep(0.3)
    step = va.search_turn_step or 10.0
    max_turn = va.search_max_turn or 90.0
    turned = 0.0
    while turned < max_turn:
        turn_and_wait(agv, step, search_speed, 0.5)
        turned += step
        if measure_marker(robot, va, detector) is not None:
            logger.warning("Marker found after %.0f° search", turned)
            return True
    return False


# ── A. STATIC test ──────────────────────────────────────────────────────

def run_static_test(robot, va, detector, args) -> None:
    print("\n" + "=" * 78)
    print("A. STATIC TEST (hold still; is the lateral readout noisy at rest?)")
    print("=" * 78)
    xs, zs, ysv, dts, pbs = [], [], [], [], []
    for i in range(args.static_reads):
        m = measure_marker(robot, va, detector)
        if m is None:
            print(f"  [{i:2d}] marker lost")
            continue
        pb = pixel_bearing(m["pixel_u"], va)
        xs.append(m["x_cam"]); zs.append(m["z_cam"]); ysv.append(m["y_cam"])
        dts.append(m["dtheta"]); pbs.append(pb)
        print(
            f"  [{i:2d}] x_cam={m['x_cam']:+7.4f}  z_cam={m['z_cam']:7.4f}  "
            f"px_bearing={pb:+7.3f}°  dtheta={m['dtheta']:+7.3f}°"
        )
        time.sleep(args.static_interval)

    if not xs:
        print("\n  No valid reads — marker not in view.")
        return

    xa, za, yva, dta, pba = map(np.array, (xs, zs, ysv, dts, pbs))
    print("-" * 78)
    print(f"  x_cam        mean={xa.mean():+.4f}  std={xa.std():.4f} m  "
          f"(std {xa.std()*100:.2f} cm)")
    print(f"  z_cam        mean={za.mean():.4f}  std={za.std():.4f} m")
    print(f"  y_cam        mean={yva.mean():.4f}  std={yva.std():.4f} m")
    print(f"  dtheta       mean={dta.mean():+.3f}  std={dta.std():.3f}°")
    print(f"  px_bearing   mean={pba.mean():+.3f}  std={pba.std():.3f}°")
    print("-" * 78)

    x_std_cm = xa.std() * 100.0
    pb_std = pba.std()
    if x_std_cm < 0.5 and pb_std < 0.5:
        print("Reading: quiet at rest (x_cam < 0.5 cm, px_bearing < 0.5°). The")
        print("lateral readout is stable when still -> check Part B for what")
        print("happens DURING a turn.")
    elif x_std_cm >= 1.5 and pb_std < 0.5:
        print("Reading: x_cam is noisy (~%.1f cm) but pixel bearing is quiet"
              % x_std_cm)
        print("(~%.1f°). The pixel data is clean; solvePnP's lateral (tvec[0]) is"
              % pb_std)
        print("the noise source -> (a) solvePnP lateral instability.")
    else:
        print("Reading: both x_cam (~%.1f cm) and px_bearing (~%.1f°) jitter."
              % (x_std_cm, pb_std))
        print("Could be detection/illumination noise; retry with better lighting")


# ── B. TURN cross-check ─────────────────────────────────────────────────

def run_turn_test(robot, agv, va, detector, args) -> None:
    print("\n" + "=" * 78)
    print("B. TURN CROSS-CHECK (does the camera rotate more than the AGV?)")
    print("=" * 78)
    fx = float(_get_camera_params(va)[0][0, 0])
    print(f"  turn {args.turn_angle:+}° per rep, {args.reps} reps, "
          f"fx={fx:.0f}px (from config)")
    print("-" * 78)
    print("  rep  agv_dh    cam_dbear  px_dbear   res_cam  res_pixel  verdict")

    for rep in range(args.reps):
        direction = 1.0 if rep % 2 == 0 else -1.0
        ang = direction * args.turn_angle

        before = measure_marker(robot, va, detector)
        h0 = _read_heading(agv)
        if before is None:
            print(f"  {rep:3d}  [before measure failed: marker lost]")
            continue

        turn_and_wait(agv, ang, args.turn_speed, args.settle)

        after = measure_marker(robot, va, detector)
        h1 = _read_heading(agv)
        if after is None:
            print(f"  {rep:3d}  [after measure failed: marker lost]")
            continue

        agv_dh = _angle_diff_deg(h1, h0)  # + = CCW/left
        cam_db = after["dtheta"] - before["dtheta"]
        pix_db = (pixel_bearing(after["pixel_u"], va)
                  - pixel_bearing(before["pixel_u"], va))

        res_cam = cam_db + agv_dh
        res_pix = pix_db + agv_dh

        # Verdict
        if abs(res_pix) < 1.0 and abs(res_cam) < 1.0:
            verdict = "both OK"
        elif abs(res_pix) < 1.0 and abs(res_cam) >= 3.0:
            verdict = "solvePnP lateral unstable"
        elif abs(res_pix) >= 3.0 and abs(res_cam) >= 3.0:
            verdict = "camera moved (non-rigid)"
        else:
            verdict = "borderline"

        print(
            f"  {rep:3d}  {agv_dh:+7.2f}  {cam_db:+9.2f}  {pix_db:+9.2f}  "
            f"{res_cam:+8.2f}  {res_pix:+8.2f}  {verdict}"
        )

    print("-" * 78)
    print("Reading: a rigid camera + accurate solvePnP gives res_cam ≈ 0 AND")
    print("res_pixel ≈ 0 (the bearing change = -agv_dheading).")
    print("  res_pixel ≈ 0 but |res_cam| large  -> solvePnP lateral instability")
    print("      (fix the bearing math, not the AGV / not the mount).")
    print("  both |res| large and similar        -> camera moved relative to AGV")
    print("      (fix the mount mechanically).")


# ── C. TRANSLATE cross-check ────────────────────────────────────────────

def run_translate_test(robot, agv, va, detector, args) -> None:
    print("\n" + "=" * 78)
    print("C. TRANSLATE CROSS-CHECK (does 'straight' forward yaw the AGV?)")
    print("=" * 78)
    print(f"  forward {args.fwd_dist:.3f}m  modes={args.fwd_modes}  reps={args.reps}")
    print("-" * 78)
    print("  mode  agv_dhead  agv_lat    cam_dbear  px_dbear  x_cam_delta  verdict")

    for mode in args.fwd_modes:
        for rep in range(args.reps):
            before = measure_marker(robot, va, detector)
            p0 = agv.get_position()
            if before is None:
                print(f"  {mode:4d}  [before measure failed: marker lost]")
                continue

            agv.translate(dist=args.fwd_dist, vx=args.fwd_speed, mode=mode)
            agv.wait_for_translate_complete(timeout=15.0)
            time.sleep(args.settle)

            after = measure_marker(robot, va, detector)
            p1 = agv.get_position()
            if after is None:
                print(f"  {mode:4d}  [after measure failed: marker lost]")
                continue

            # AGV self-report: heading change + lateral displacement (body frame)
            agv_dh = _angle_diff_deg(p1.theta, p0.theta)
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            lat = -dx * math.sin(p0.theta) + dy * math.cos(p0.theta)

            cam_db = after["dtheta"] - before["dtheta"]
            pix_db = (pixel_bearing(after["pixel_u"], va)
                      - pixel_bearing(before["pixel_u"], va))
            xcam_d = after["x_cam"] - before["x_cam"]

            if abs(agv_dh) < 1.0:
                verdict = "translate clean"
            elif abs(agv_dh) < 3.0:
                verdict = "mild yaw"
            else:
                verdict = "LARGE YAW <- root cause"

            print(
                f"  {mode:4d}  {agv_dh:+8.2f}  {lat:+8.4f}  {cam_db:+9.2f}  "
                f"{pix_db:+9.2f}  {xcam_d:+10.4f}  {verdict}"
            )

    print("-" * 78)
    print("Reading: agv_dhead = the AGV's OWN heading change during the forward")
    print("drive — the one measurement the self-report test read but never printed.")
    print("  |agv_dhead| large (several deg) -> the translate YAWS the AGV; the")
    print("      pitched-down camera sees that yaw as a big lateral marker swing.")
    print("  |agv_dhead| ~ 0 -> translate is straight; look at control/reference.")


# ── D. COMBINED turn+forward (production sequence) ──────────────────────

def run_combined_test(robot, agv, va, detector, args) -> None:
    print("\n" + "=" * 78)
    print("D. COMBINED TURN+FORWARD (production Phase-1 sequence, sub-step reads)")
    print("=" * 78)
    print(f"  turn toward marker by {args.turn_angle}° then forward {args.fwd_dist:.3f}m, "
          f"{args.reps} reps")
    print("-" * 78)
    print("  rep  step          x_cam     z_cam    dtheta    px_bear")

    def _row(tag: str, m) -> None:
        if m is None:
            print(f"  {rep:3d}  {tag:12s}  [marker lost]")
        else:
            pb = pixel_bearing(m["pixel_u"], va)
            print(f"  {rep:3d}  {tag:12s}  {m['x_cam']:+7.4f}  {m['z_cam']:7.4f}  "
                  f"{m['dtheta']:+8.2f}  {pb:+8.2f}")

    for rep in range(args.reps):
        before = measure_marker(robot, va, detector)
        if before is None:
            print(f"  {rep:3d}  [before: marker lost]")
            continue
        _row("before", before)

        # Turn TOWARD the marker (same direction the production loop would).
        turn_deg = math.copysign(args.turn_angle, before["dtheta"])
        turn_and_wait(agv, turn_deg, args.turn_speed, args.settle)
        mid = measure_marker(robot, va, detector)
        _row("after-turn", mid)

        agv.translate(dist=args.fwd_dist, vx=args.fwd_speed, mode=args.fwd_modes[0])
        agv.wait_for_translate_complete(timeout=15.0)
        time.sleep(args.settle)
        after = measure_marker(robot, va, detector)
        _row("after-fwd", after)

        if mid is not None:
            print(f"  {rep:3d}  turn Δx_cam={mid['x_cam'] - before['x_cam']:+.4f}  "
                  f"turn Δdtheta={mid['dtheta'] - before['dtheta']:+.2f}°")
        if after is not None and mid is not None:
            print(f"  {rep:3d}  fwd  Δx_cam={after['x_cam'] - mid['x_cam']:+.4f}  "
                  f"fwd  Δdtheta={after['dtheta'] - mid['dtheta']:+.2f}°")

    print("-" * 78)
    print("Reading: if x_cam swings ~10 cm here, it reproduces production, and the")
    print("sub-step rows show WHICH step causes it:")
    print("  large 'turn Δx_cam' for a small turn -> the AGV over-rotates (yaw)")
    print("  large 'fwd Δx_cam' -> the forward drive curves/yaws after the turn")
    print("  px_bear stays put while x_cam/dtheta swing -> solvePnP artifact")


# ── E. TURN-ANGLE -> FORWARD-YAW sweep (proportional or fixed?) ─────────

def run_forward_yaw_test(robot, agv, va, detector, args) -> None:
    print("\n" + "=" * 78)
    print("E. TURN-ANGLE -> FORWARD-YAW SWEEP (is the yaw proportional to turn?)")
    print("=" * 78)
    print(f"  forward {args.fwd_dist:.3f}m fixed; turn angles {args.turn_angles}°  "
          f"reps={args.reps}")
    print("-" * 78)
    print("  rep  turn°   turn Δdθ    fwd Δdθ    fwd Δx_cam")

    for ang in args.turn_angles:
        for rep in range(args.reps):
            before = measure_marker(robot, va, detector)
            if before is None:
                print(f"  {rep:3d}  {ang:5.1f}  [before: marker lost]")
                continue
            # turn toward the marker by `ang` (production behaviour)
            turn_deg = math.copysign(ang, before["dtheta"])
            turn_and_wait(agv, turn_deg, args.turn_speed, args.settle)
            mid = measure_marker(robot, va, detector)

            agv.translate(dist=args.fwd_dist, vx=args.fwd_speed, mode=args.fwd_modes[0])
            agv.wait_for_translate_complete(timeout=15.0)
            time.sleep(args.settle)
            after = measure_marker(robot, va, detector)

            if mid is None or after is None:
                print(f"  {rep:3d}  {ang:5.1f}  [measure failed: marker lost]")
                continue
            turn_dd = mid["dtheta"] - before["dtheta"]
            fwd_dd = after["dtheta"] - mid["dtheta"]
            fwd_dx = after["x_cam"] - mid["x_cam"]
            print(f"  {rep:3d}  {ang:5.1f}  {turn_dd:+8.2f}  {fwd_dd:+8.2f}  {fwd_dx:+8.4f}")

    print("-" * 78)
    print("Reading:")
    print("  fwd Δdθ scales ~linearly with turn° -> yaw is PROPORTIONAL (k = slope);")
    print("      fix = scale the turn command by 1/(1+k) so turn+fwd_yaw = target.")
    print("  fwd Δdθ ~constant across turn° -> yaw is a FIXED offset;")
    print("      fix = subtract that constant from the turn command.")
    print("  Either way the fix is a one-line turn-angle adjustment (no vy needed).")


# ── main ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera stability diagnostic (static + single-turn cross-check)"
    )
    p.add_argument(
        "--config", default="../agv_class_online_test.yaml",
        help="orchestrator YAML with a visual_align task + robot_config. "
             "Default = production config; the in-repo test config points at a "
             "different robot (192.168.2.210), so do NOT use it for the real AGV.",
    )
    p.add_argument("--host", default=None, help="override AGV IP from YAML")
    p.add_argument("--static-reads", type=int, default=30, help="reads in Part A")
    p.add_argument("--static-interval", type=float, default=0.15, help="s between static reads")
    p.add_argument("--turn-angle", type=float, default=2.0, help="turn magnitude (deg) per rep")
    p.add_argument("--turn-speed", type=float, default=10.0, help="turn speed deg/s")
    p.add_argument("--reps", type=int, default=3, help="turn reps (alternating direction)")
    p.add_argument("--settle", type=float, default=0.8, help="settle time (s) after motion")
    p.add_argument("--search-speed", type=float, default=10.0, help="deg/s while searching")
    p.add_argument("--fwd-dist", type=float, default=0.075,
                   help="forward distance (m) per translate rep (Part C)")
    p.add_argument("--fwd-speed", type=float, default=0.15, help="translate speed m/s (Part C)")
    p.add_argument("--fwd-modes", default="1",
                   help="comma-separated translate modes to sweep, 0=odometry 1=localization")
    p.add_argument("--turn-angles", default="0.5,1,2,3",
                   help="comma-separated turn angles (deg) to sweep in Part E")
    p.add_argument("--skip-static", action="store_true", help="skip Part A")
    p.add_argument("--skip-turn", action="store_true", help="skip Part B")
    p.add_argument("--skip-translate", action="store_true", help="skip Part C")
    p.add_argument("--skip-combined", action="store_true", help="skip Part D")
    p.add_argument("--skip-lateral", action="store_true", help="skip Part E")
    p.add_argument("--yes", action="store_true", help="skip the start confirmation prompt")
    return p.parse_args()


def _parse_csv(s: str, cast=float):
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


def main() -> int:
    args = parse_args()
    args.fwd_modes = _parse_csv(args.fwd_modes, cast=int)
    args.turn_angles = _parse_csv(args.turn_angles, cast=float)
    va, cfg = _load_va(args)
    host = args.host or cfg.agv_config.host

    print("=" * 78)
    print("Camera measurement stability diagnostic")
    print("=" * 78)
    print(f"  config       : {args.config}")
    print(f"  AGV host     : {host}")
    print(f"  marker_id    : {va.marker_id}  size={va.marker_size}m")
    print(f"  static reads : {args.static_reads}")
    print(f"  turn         : ±{args.turn_angle}°  speed={args.turn_speed}°/s  reps={args.reps}")
    print(f"  translate    : {args.fwd_dist:.3f}m  modes={args.fwd_modes}  "
          f"speed={args.fwd_speed}m/s")
    print("-" * 78)
    print("Part A holds still; Parts B/C MOVE the AGV (turn + forward).")
    print("Clear area required. Tag must be in view.")
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
        if not find_marker(robot, agv, va, detector, args.search_speed):
            print("ERROR: could not find the marker. Aborting (no motion issued).")
            return 1
        print("Marker located. Beginning tests...\n")

        if not args.skip_static:
            run_static_test(robot, va, detector, args)
        if not args.skip_turn:
            run_turn_test(robot, agv, va, detector, args)
        if not args.skip_translate:
            run_translate_test(robot, agv, va, detector, args)
        if not args.skip_combined:
            run_combined_test(robot, agv, va, detector, args)
        if not args.skip_lateral:
            run_forward_yaw_test(robot, agv, va, detector, args)

    finally:
        for obj, name in ((agv, "AGV"), (robot, "robot")):
            try:
                disc = getattr(obj, "disconnect", None)
                if disc is not None:
                    disc()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
