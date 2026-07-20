#!/usr/bin/env python3
"""Camera calibration script for head camera.

Supports two board types:
  - chessboard  (standard black-white checkerboard)
  - charuco     (ChArUco board = chessboard + ArUco markers)

Computes the camera intrinsic matrix and distortion coefficients via
OpenCV.  Outputs values ready to paste into any YAML's visual_align_config.

ChArUco is recommended: partial visibility → more edge samples → better
distortion estimation.

Usage (ChArUco, recommended):
    python -m lerobot.scripts.calibrate_camera_v1 \\
        --robot-type supre_robot_follower \\
        --cam-type opencv \\
        --cam-index 0 \\
        --cam-width 640 \\
        --cam-height 480 \\
        --cam-fps 30 \\
        --board-type charuco \\
        --charuco-squares-x 5 \\
        --charuco-squares-y 7 \\
        --charuco-square-length 0.030 \\
        --charuco-marker-length 0.024 \\
        --charuco-marker-family 6x6 \\
        --min-samples 30

Usage (chessboard):
    python -m lerobot.scripts.calibrate_camera_v1 \\
        --robot-type supre_robot_follower \\
        ...
        --board-type chessboard \\
        --chessboard-rows 7 --chessboard-cols 5 --chessboard-size 0.030 \\
        --min-samples 25
"""

import argparse
import time

import cv2
import numpy as np


_ARUCO_DICT_MAP = {
    "4x4": cv2.aruco.DICT_4X4_50,
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5": cv2.aruco.DICT_5X5_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6": cv2.aruco.DICT_6X6_250,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "7x7": cv2.aruco.DICT_7X7_50,
}


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate head camera")
    # Robot
    p.add_argument("--robot-type", type=str, default="supre_robot_follower")
    p.add_argument("--cam-type", type=str, default="opencv")
    p.add_argument("--cam-index", type=int, default=0)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)

    # Board
    p.add_argument("--board-type", choices=["chessboard", "charuco"],
                   default="charuco")

    # Chessboard
    p.add_argument("--chessboard-rows", type=int, default=7)
    p.add_argument("--chessboard-cols", type=int, default=5)
    p.add_argument("--chessboard-size", type=float, default=0.030)

    # ChArUco
    p.add_argument("--charuco-squares-x", type=int, default=5,
                   help="Number of chessboard squares along X")
    p.add_argument("--charuco-squares-y", type=int, default=7,
                   help="Number of chessboard squares along Y")
    p.add_argument("--charuco-square-length", type=float, default=0.030,
                   help="Physical size of one chessboard square in meters")
    p.add_argument("--charuco-marker-length", type=float, default=0.024,
                   help="Physical side length of one ArUco marker in meters")
    p.add_argument("--charuco-marker-family", type=str, default="4x4",
                   help="ArUco dictionary (4x4, 5x5, 6x6, 7x7)")

    # Collection
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--auto-collect", action="store_true")
    p.add_argument("--capture-interval", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    board_type = args.board_type
    min_samples = args.min_samples
    img_w = args.cam_width
    img_h = args.cam_height

    # ── Board setup ──────────────────────────────────────────────────
    if board_type == "charuco":
        squares_x = args.charuco_squares_x
        squares_y = args.charuco_squares_y
        sq_len = args.charuco_square_length
        mk_len = args.charuco_marker_length

        dict_name = args.charuco_marker_family
        dict_id = _ARUCO_DICT_MAP.get(dict_name)
        if dict_id is None:
            print(f"ERROR: unknown marker family '{dict_name}'")
            print(f"  Supported: {list(_ARUCO_DICT_MAP.keys())}")
            return
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), sq_len, mk_len, dictionary)
        detector = cv2.aruco.CharucoDetector(board)

        board_desc = (
            f"ChArUco {squares_x}×{squares_y} squares, {sq_len*1000:.0f}mm squares, "
            f"{mk_len*1000:.0f}mm markers, {dict_name}"
        )
    else:
        board_rows = args.chessboard_rows
        board_cols = args.chessboard_cols
        board_size = args.chessboard_size
        board_shape = (board_rows, board_cols)

        objp = np.zeros((board_rows * board_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_rows, 0:board_cols].T.reshape(-1, 2)
        objp *= board_size

        board_desc = (
            f"Chessboard {board_rows}×{board_cols} inner corners, "
            f"{board_size*1000:.0f}mm squares"
        )

    # ── Open camera ─────────────────────────────────────────────────
    print("Connecting to robot...")
    import cv2
    cam_idx = args.cam_index
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {cam_idx}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    print(f"Camera connected: index={cam_idx} {args.cam_width}x{args.cam_height}")

    def _grab_frame():
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    print("Robot connected.")

    # ── Header ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("   CAMERA CALIBRATION")
    print("=" * 60)
    print(f"  Board:   {board_desc}")
    print(f"  Target:  {min_samples}+ good captures from different angles")
    print()
    if args.auto_collect:
        print(f"  Mode:    AUTO — capturing every {args.capture_interval:.1f}s")
        print("           Move the board around slowly between captures.")
    else:
        print("  Mode:    INTERACTIVE:")
        print("           SPACE = capture (only when board is green)")
        print("           ESC   = finish & calibrate early")
    print()
    if board_type == "charuco":
        print("  💡 ChArUco: board can be partially visible — push it")
        print("     to the very edge of the frame for best results.")
    print("=" * 60)
    print()

    # ── Collection ────────────────────────────────────────────────────
    if not args.auto_collect:
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 960, 720)

    if board_type == "charuco":
        all_charuco_corners = []
        all_charuco_ids = []
    else:
        obj_points = []
        img_points = []

    last_capture_time = 0.0
    n_captured = 0

    while True:
        img = _grab_frame()
        if img is None:
            print("ERROR: no image from camera")
            time.sleep(0.1)
            continue

        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        display = bgr.copy()

        if board_type == "charuco":
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            found = (charuco_corners is not None and len(charuco_corners) >= 4)

            if marker_corners is not None and marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if found:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
        else:
            found, corners = cv2.findChessboardCorners(gray, board_shape, None)
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_sub = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, board_shape, corners_sub, found)

        # Overlay
        n = n_captured
        status = f"Captured: {n}/{min_samples}"
        color = (0, 255, 0) if n >= min_samples else (0, 165, 255)
        cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        label = "BOARD DETECTED" if found else "no board"
        label_color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(display, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        # Capture decision
        should_capture = False
        if args.auto_collect:
            now = time.time()
            if found and now - last_capture_time > args.capture_interval:
                should_capture = True
                last_capture_time = now
                time.sleep(0.2)
        else:
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                print(f"\n  ESC pressed — calibrating with {n} samples...")
                break
            if key == 32 and found:  # SPACE
                should_capture = True

        if should_capture and found:
            if board_type == "charuco":
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            else:
                img_points.append(corners_sub)
                obj_points.append(objp)
            n_captured += 1
            remaining = max(0, min_samples - n_captured)
            print(f"  [{n_captured:2d}] Captured — {remaining} more to go")
            if n_captured >= min_samples:
                print(f"\n  Reached {n_captured} samples — running calibration...")
                break

    if not args.auto_collect:
        cv2.destroyAllWindows()
    cap.release()

    # ── Calibration ───────────────────────────────────────────────────
    if board_type == "charuco":
        n_views = len(all_charuco_corners)
        if n_views < 3:
            print(f"ERROR: need at least 3 good captures, got {n_views}")
            return
    else:
        n_views = len(obj_points)
        if n_views < 3:
            print(f"ERROR: need at least 3 good captures, got {n_views}")
            return

    print()
    print("=" * 60)
    print("   CALIBRATION RESULT")
    print("=" * 60)

    if board_type == "charuco":
        ret, K, d, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board,
            (img_w, img_h), None, None,
        )
        # Per-view reprojection error for ChArUco
        total_err = 0.0
        n_pts = 0
        for i in range(n_views):
            if rvecs is not None and tvecs is not None:
                obj_pts = board.getChessboardCorners()[all_charuco_ids[i].flatten()]
                img_pts = all_charuco_corners[i]
                projected, _ = cv2.projectPoints(obj_pts.astype(np.float32), rvecs[i], tvecs[i], K, d)
                err = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
                total_err += err
                n_pts += 1
        mean_err = total_err / max(n_pts, 1)
    else:
        ret, K, d, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (img_w, img_h), None, None,
        )
        total_err = 0.0
        for i in range(n_views):
            projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, d)
            err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
            total_err += err
        mean_err = total_err / n_views

    d_flat = d.flatten()

    print(f"  Views:                    {n_views}")
    print(f"  RMS reprojection error:   {ret:.3f} px")
    print(f"  Mean per-view error:      {mean_err:.3f} px")
    print()
    print("  Camera matrix (3×3):")
    print(f"    fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print()
    print("  Distortion coefficients (k1,k2,p1,p2,k3):")
    print(f"    {[f'{x:.6f}' for x in d_flat]}")
    print()

    # ── YAML-ready output ─────────────────────────────────────────────
    print("─" * 60)
    print("  Paste this into your YAML visual_align_config:")
    print("─" * 60)
    print(f"  camera_matrix: [{K[0,0]:.3f}, 0.0, {K[0,2]:.3f}, 0.0, {K[1,1]:.3f}, {K[1,2]:.3f}, 0.0, 0.0, 1.0]")
    print(f"  dist_coeffs: [{d_flat[0]:.6f}, {d_flat[1]:.6f}, {d_flat[2]:.6f}, {d_flat[3]:.6f}, {d_flat[4]:.6f}]")
    print("─" * 60)
    print()

    # Quality
    print("  Quality assessment:")
    if ret < 0.3:
        print("    ✓ Excellent (<0.3 px RMS)")
    elif ret < 0.5:
        print("    ✓ Good (0.3-0.5 px RMS)")
    elif ret < 1.0:
        print("    △ Acceptable (0.5-1.0 px RMS)")
    else:
        print("    ✗ Poor (>1.0 px RMS) — re-calibrate with more varied poses")

    if K[0,0] < 200 or K[0,0] > 2000:
        print(f"    ⚠  fx={K[0,0]:.0f} looks unusual for a standard webcam")
    if abs(K[0,2] - img_w / 2) > img_w * 0.15:
        print(f"    ⚠  cx={K[0,2]:.0f} deviates from image center ({img_w/2:.0f})")
    print()


if __name__ == "__main__":
    main()
