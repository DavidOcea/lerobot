#!/usr/bin/env python3
"""Camera calibration script for head camera.

Uses a standard chessboard pattern to compute the camera intrinsic matrix
and distortion coefficients via OpenCV's calibrateCamera().  Outputs values
ready to paste into any YAML's visual_align_config section.

Usage:
    # Interactive mode (recommended): shows live preview, press SPACE to
    # capture a calibration frame.  Frame is saved automatically when
    # chessboard corners are detected successfully.
    python -m lerobot.scripts.calibrate_camera \
        --robot.type supre_robot_follower \
        --robot.cameras.head_cam.type opencv \
        --robot.cameras.head_cam.index 0 \
        --robot.cameras.head_cam.width 640 \
        --robot.cameras.head_cam.height 480 \
        --robot.cameras.head_cam.fps 30 \
        --chessboard-rows 8 \
        --chessboard-cols 6 \
        --chessboard-size 0.025 \
        --min-samples 20

    # Non-interactive / headless (auto-capture one frame per second):
    # Add --auto-collect and a capture interval.
"""

import argparse
import time

import cv2
import numpy as np

from lerobot.robots import make_robot_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate head camera with chessboard")
    p.add_argument("--robot.type", type=str, required=True)
    p.add_argument("--robot.cameras.head_cam.type", type=str, default="opencv")
    p.add_argument("--robot.cameras.head_cam.index", type=int, default=0)
    p.add_argument("--robot.cameras.head_cam.width", type=int, default=640)
    p.add_argument("--robot.cameras.head_cam.height", type=int, default=480)
    p.add_argument("--robot.cameras.head_cam.fps", type=int, default=30)
    p.add_argument("--chessboard-rows", type=int, default=8,
                   help="Number of INNER corners along the rows of the chessboard")
    p.add_argument("--chessboard-cols", type=int, default=6,
                   help="Number of INNER corners along the columns of the chessboard")
    p.add_argument("--chessboard-size", type=float, default=0.025,
                   help="Physical size of one chessboard square (meters)")
    p.add_argument("--min-samples", type=int, default=20,
                   help="Minimum number of successful captures before running calibration")
    p.add_argument("--auto-collect", action="store_true",
                   help="Auto-capture mode: capture one frame per interval, no keyboard needed")
    p.add_argument("--capture-interval", type=float, default=1.0,
                   help="Seconds between auto-captures")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Reconstruct robot config from CLI args (same format as run_task_agent) ──
    robot_cfg = {
        "type": getattr(args, "robot.type"),
        "cameras": {
            "head_cam": {
                "type": getattr(args, "robot.cameras.head_cam.type"),
                "index": getattr(args, "robot.cameras.head_cam.index"),
                "width": getattr(args, "robot.cameras.head_cam.width"),
                "height": getattr(args, "robot.cameras.head_cam.height"),
                "fps": getattr(args, "robot.cameras.head_cam.fps"),
            },
        },
    }

    # ── Chessboard params ─────────────────────────────────────────────
    board_rows = args.chessboard_rows
    board_cols = args.chessboard_cols
    board_size = args.chessboard_size   # meters per square
    board_shape = (board_rows, board_cols)
    min_samples = args.min_samples

    img_w = getattr(args, "robot.cameras.head_cam.width")
    img_h = getattr(args, "robot.cameras.head_cam.height")

    # ── Connect to robot ──────────────────────────────────────────────
    print("Connecting to robot...")
    from lerobot.robots.config import RobotConfig
    import draccus
    robot_config = draccus.parse(RobotConfig, robot_cfg)
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("Robot connected.")

    # ── Calibration data collection ───────────────────────────────────
    # 3D object points for one chessboard view
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_rows, 0:board_cols].T.reshape(-1, 2)
    objp *= board_size

    obj_points = []  # 3D world points (accumulated)
    img_points = []  # 2D image points (accumulated)

    print()
    print("=" * 60)
    print("   CAMERA CALIBRATION — Chessboard Pattern")
    print("=" * 60)
    print(f"  Board:   {board_rows}×{board_cols} inner corners, {board_size*1000:.0f}mm squares")
    print(f"  Target:  {min_samples}+ good captures from different angles")
    print()
    if args.auto_collect:
        print(f"  Mode:    AUTO — capturing every {args.capture_interval:.1f}s")
        print(f"           Move the chessboard around slowly between captures.")
    else:
        print("  Mode:    INTERACTIVE — press SPACE to capture, ESC to finish")
        print("           Hold chessboard from different angles/distances.")
        print("           Frame saved only when chessboard corners are detected.")
    print("=" * 60)
    print()

    # Auto-collect does not need a window
    if not args.auto_collect:
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 960, 720)

    last_capture_time = 0.0
    while True:
        obs = robot.get_observation()
        img = obs.get("images", {}).get("head_cam")
        if img is None:
            print("ERROR: no head_cam image")
            time.sleep(0.1)
            continue

        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        found, corners = cv2.findChessboardCorners(gray, board_shape, None)

        display = bgr.copy()
        if found:
            # Sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, board_shape, corners_sub, found)

        # Overlay status
        status = f"Captured: {len(obj_points)}/{min_samples}"
        color = (0, 255, 0) if len(obj_points) >= min_samples else (0, 165, 255)
        cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if found:
            cv2.putText(display, "CHESSBOARD DETECTED", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "no chessboard", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── Capture decision ──────────────────────────────────────────
        should_capture = False
        if args.auto_collect:
            now = time.time()
            if found and now - last_capture_time > args.capture_interval:
                should_capture = True
                last_capture_time = now
                time.sleep(0.2)  # brief settle
        else:
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32 and found:  # SPACE + detected
                should_capture = True

        if should_capture and found:
            img_points.append(corners_sub)
            obj_points.append(objp)
            n = len(obj_points)
            print(f"  [{n:2d}] Captured — corners detected, {min_samples - n} more to go")

            # Auto-calibrate once we have enough samples
            if n >= min_samples:
                print()
                print(f"  Reached {n} samples — running calibration...")
                break

    if not args.auto_collect:
        cv2.destroyAllWindows()

    robot.disconnect()

    # ── Run calibration ────────────────────────────────────────────────
    if len(obj_points) < 3:
        print(f"ERROR: need at least 3 good captures, got {len(obj_points)}")
        return

    print()
    print("=" * 60)
    print("   CALIBRATION RESULT")
    print("=" * 60)

    ret, K, d, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (img_w, img_h), None, None,
    )

    # Per-view reprojection error
    total_err = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, d)
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_err += err
    mean_err = total_err / len(obj_points)

    print(f"  RMS reprojection error:  {ret:.3f} px")
    print(f"  Mean per-view error:      {mean_err:.3f} px")
    print()
    print("  Camera matrix (3×3):")
    print(f"    fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print()
    print("  Distortion coefficients (k1,k2,p1,p2,k3):")
    d_flat = d.flatten()
    print(f"    {[f'{x:.6f}' for x in d_flat]}")
    print()

    # ── Output ready to paste into YAML ────────────────────────────────
    print("─" * 60)
    print("  Paste this into your YAML visual_align_config:")
    print("─" * 60)
    print(f"  camera_matrix: [{K[0,0]:.3f}, 0.0, {K[0,2]:.3f}, 0.0, {K[1,1]:.3f}, {K[1,2]:.3f}, 0.0, 0.0, 1.0]")
    print(f"  dist_coeffs: [{d_flat[0]:.6f}, {d_flat[1]:.6f}, {d_flat[2]:.6f}, {d_flat[3]:.6f}, {d_flat[4]:.6f}]")
    print("─" * 60)
    print()

    # Quality assessment
    print("  Quality assessment:")
    if ret < 0.3:
        print("    ✓ Excellent (<0.3 px RMS)")
    elif ret < 0.5:
        print("    ✓ Good (0.3-0.5 px RMS)")
    elif ret < 1.0:
        print("    △ Acceptable (0.5-1.0 px RMS) — consider more varied angles")
    else:
        print("    ✗ Poor (>1.0 px RMS) — re-calibrate with more samples and varied poses")

    if K[0,0] < 200 or K[0,0] > 2000:
        print(f"    ⚠  fx={K[0,0]:.0f} looks unusual for a standard webcam")
    if abs(K[0,2] - img_w / 2) > img_w * 0.15:
        print(f"    ⚠  cx={K[0,2]:.0f} deviates from image center ({img_w/2:.0f})")
    print()


if __name__ == "__main__":
    main()
