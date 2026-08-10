"""Visual alignment module — AprilTag-guided AGV fine positioning.

Uses an AprilTag marker detected by the head camera to compute the
relative pose between the AGV and the target, then iteratively
aligns the AGV via "turn + translate forward/backward" steps until
convergence.  The Seer AGV does not support lateral (vy) movement
on its differential-drive chassis, so we always rotate to face the
marker first, then drive straight toward it.

Typical usage (inside orchestrator):
    result = execute_visual_align(
        robot=robot,
        agv_controller=agv_controller,
        config=VisualAlignConfig(marker_id=0, marker_size=0.10),
        logger=logging.getLogger(__name__),
    )
"""

import logging
import math
import time

import cv2
import numpy as np

from lerobot.tasks.config import VisualAlignConfig

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Default estimated camera intrinsics for 640x480 OpenCV camera.
# fx ≈ fy ≈ width * 0.83 ≈ 530 for typical webcam at 640x480.
# These are reasonable defaults; replace with calibrated values for
# higher accuracy.
DEFAULT_CAMERA_MATRIX = np.array(
    [[530.0, 0.0, 320.0],
     [0.0, 530.0, 240.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DEFAULT_DIST_COEFFS = np.zeros((5,), dtype=np.float64)

# AprilTag dictionary name → cv2.aruco constant mapping
_TAG_DICT_MAP = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


def _get_detector(family: str = "tag36h11") -> cv2.aruco.ArucoDetector:
    """Create an AprilTag detector for the given tag family."""
    dict_id = _TAG_DICT_MAP.get(family, cv2.aruco.DICT_APRILTAG_36h11)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def _get_camera_params(config: VisualAlignConfig):
    """Return (camera_matrix, dist_coeffs) from config or defaults."""
    if config.camera_matrix is not None:
        K = np.array(config.camera_matrix, dtype=np.float64).reshape(3, 3)
    else:
        K = DEFAULT_CAMERA_MATRIX

    if config.dist_coeffs is not None:
        d = np.array(config.dist_coeffs, dtype=np.float64)
    else:
        d = DEFAULT_DIST_COEFFS

    return K, d


def detect_marker(
    image: np.ndarray,
    config: VisualAlignConfig,
    detector: cv2.aruco.ArucoDetector | None = None,
    target_id: int | None = None,
):
    """Detect AprilTag markers in an image.

    Returns the matching marker detection (or None).
    Each detection is a dict with: id, corners, rvec, tvec.

    Args:
        target_id: specific marker ID to search for.
                   None = use config.marker_id.
    """
    if detector is None:
        detector = _get_detector(config.marker_family)

    corners, ids, rejected = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        return None

    K, d = _get_camera_params(config)
    look_for = target_id if target_id is not None else config.marker_id
    # Use tag_1_size for tag_1 marker if configured
    msize = config.marker_size
    if config.tag_1_id is not None and look_for == config.tag_1_id and config.tag_1_size is not None:
        msize = config.tag_1_size

    half = msize / 2.0
    obj_points = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    for i, marker_id in enumerate(ids.flatten()):
        if look_for is None or marker_id == look_for:
            img_points = corners[i].reshape(4, 2).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, K, d,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue
            return {
                "id": int(marker_id),
                "corners": corners[i],
                "rvec": rvec.flatten(),
                "tvec": tvec.flatten(),
            }

    return None


def _marker_to_agv_xy(
    tvec: np.ndarray,
    config: VisualAlignConfig,
) -> tuple[float, float]:
    """Transform marker position from camera frame to AGV ground-plane frame.

    Steps:
      1. Pitch correction — project tilted optical axis onto ground plane.
      2. Yaw + offset — rotate and translate from camera to AGV center.

    Returns (dx_agv, dy_agv):
      dx_agv: forward distance from AGV center to marker ground projection (m)
      dy_agv: leftward distance from AGV center to marker ground projection (m)
    """
    x_cam = tvec[0]  # rightward
    y_cam = tvec[1]  # downward
    z_cam = tvec[2]  # along optical axis

    # Pitch correction: project onto horizontal plane
    pitch_rad = config.camera_offset_pitch * DEG_TO_RAD
    z_horiz = math.cos(pitch_rad) * z_cam - math.sin(pitch_rad) * y_cam
    x_horiz = x_cam  # pitch around x-axis, lateral unchanged

    # Rotate + translate to AGV frame
    offset_yaw_rad = config.camera_offset_yaw * DEG_TO_RAD
    cos_yaw = math.cos(offset_yaw_rad)
    sin_yaw = math.sin(offset_yaw_rad)
    dx_agv = z_horiz * cos_yaw - x_horiz * sin_yaw + config.camera_offset_x
    dy_agv = -z_horiz * sin_yaw - x_horiz * cos_yaw + config.camera_offset_y

    return dx_agv, dy_agv


def compute_agv_movement(
    tvec: np.ndarray,
    rvec: np.ndarray,
    config: VisualAlignConfig,
):
    """Compute AGV movement from marker pose in camera frame.

    Since vy doesn't work on the Seer AGV, we use a "rotate + forward"
    strategy:
      1. Compute the angle to the marker → AGV must turn this angle
         to face the marker.
      2. After turning, compute the remaining forward distance → AGV
         drives straight toward the marker.

    Returns (dtheta_deg, forward_dist_m):
      dtheta_deg: angle AGV must turn (positive = left/CCW)
      forward_dist_m: distance AGV must move forward after turning
    """
    dx_agv, dy_agv = _marker_to_agv_xy(tvec, config)

    # Angle to face the marker
    dtheta_rad = math.atan2(dy_agv, dx_agv)
    dtheta_deg = dtheta_rad * RAD_TO_DEG

    # After turning by dtheta, the marker will be straight ahead.
    # Ground-plane distance minus the desired approach distance.
    total_dist = math.sqrt(dx_agv**2 + dy_agv**2)
    forward_dist = total_dist - config.approach_distance

    return dtheta_deg, forward_dist


def save_reference_pose(path: str, tvec: np.ndarray, rvec: np.ndarray,
                       tag_1_tvec: np.ndarray | None = None,
                       tag_1_rvec: np.ndarray | None = None):
    """Save reference camera pose(s) to JSON.  Dual-tag mode if tag_1_tvec set."""
    import json
    data = {
        "tvec": tvec.tolist(),
        "rvec": rvec.tolist(),
    }
    if tag_1_tvec is not None:
        data["tag_1_tvec"] = tag_1_tvec.tolist()
        data["tag_1_rvec"] = (tag_1_rvec if tag_1_rvec is not None else np.zeros(3)).tolist()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_reference_pose(path: str) -> tuple[np.ndarray, ...]:
    """Load reference camera pose from JSON.
    Returns (tvec, rvec, tag_1_tvec|None, tag_1_rvec|None).
    """
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    t1_t = np.array(data["tag_1_tvec"]) if "tag_1_tvec" in data else None
    t1_r = np.array(data["tag_1_rvec"]) if "tag_1_rvec" in data else None
    return np.array(data["tvec"]), np.array(data["rvec"]), t1_t, t1_r


def compute_alignment_to_reference(
    tvec_cur: np.ndarray,
    rvec_cur: np.ndarray,
    ref_tvec: np.ndarray,
    ref_rvec: np.ndarray,
    config: VisualAlignConfig,
) -> tuple[float, float]:
    """Compute AGV movement to align current view to reference view.

    Stable "face marker + match distance" strategy:
      1. Turn to face the marker directly (atan2 in AGV frame).
      2. Match the reference radial distance to the marker.

    This is more stable than full 2D position matching because it
    avoids amplifying solvePnP lateral noise.  The AGV ends up facing
    the marker head-on at the reference distance — sufficient for
    most docking / pick-and-place scenarios.

    Returns (dtheta_deg, forward_dist_m).
    """
    # Current and reference marker positions in AGV frame
    cur_x, cur_y = _marker_to_agv_xy(tvec_cur, config)
    ref_x, ref_y = _marker_to_agv_xy(ref_tvec, config)

    cur_dist = math.sqrt(cur_x**2 + cur_y**2)
    ref_dist = math.sqrt(ref_x**2 + ref_y**2)

    # Turn to face the marker
    dtheta_rad = math.atan2(cur_y, cur_x)
    dtheta_deg = dtheta_rad * RAD_TO_DEG

    # Match reference distance + fine-tune
    forward_dist = cur_dist - ref_dist + config.distance_offset

    return dtheta_deg, forward_dist


def _log_reference_diagnostics(
    tvec_cur, tvec_ref, config, logger,
):
    """Log diagnostics for debugging reference alignment."""
    cur_x, cur_y = _marker_to_agv_xy(tvec_cur, config)
    ref_x, ref_y = _marker_to_agv_xy(tvec_ref, config)
    dx = cur_x - ref_x
    dy = cur_y - ref_y
    logger.warning(
        f"  [diag] cur=({cur_x:.3f}, {cur_y:.3f})m ref=({ref_x:.3f}, {ref_y:.3f})m "
        f"Δ=({dx:.3f}, {dy:.3f})m dist={math.sqrt(dx**2+dy**2):.3f}m"
    )


def capture_reference_pose(
    robot,
    config: VisualAlignConfig,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Capture current camera pose relative to AprilTag as reference.

    Takes one photo, detects the target marker, and returns (tvec, rvec).
    Returns None if marker not found.

    The caller should save the result with save_reference_pose().
    """
    detector = _get_detector(config.marker_family)
    obs = robot.get_observation()
    img = obs.get("images", {}).get("head_cam")
    if img is None:
        logger.error("capture_reference_pose: no head_cam image")
        return None
    bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)

    marker = detect_marker(bgr, config, detector)
    if marker is None:
        logger.error(
            f"capture_reference_pose: marker ID={config.marker_id} not found"
        )
        return None

    logger.warning(
        f"Reference pose captured: marker ID={marker['id']}, "
        f"tvec=[{marker['tvec'][0]:.3f}, {marker['tvec'][1]:.3f}, {marker['tvec'][2]:.3f}]"
    )
    return marker["tvec"].copy(), marker["rvec"].copy()


def search_marker(
    robot,
    agv_controller,
    config: VisualAlignConfig,
    logger: logging.Logger,
) -> tuple[dict | None, int, float]:
    """Search for the target AprilTag by rotating the AGV.

    Rotates AGV left (CCW) in small steps until the marker is found
    in the head camera.

    Returns:
        (marker, attempts_used, total_turned_deg) — marker is None if not found.
    """
    detector = _get_detector(config.marker_family)
    total_turned = 0.0

    for attempt in range(config.search_max_attempts):
        # Capture image from head camera
        obs = robot.get_observation()
        images = obs.get("images", {})
        img = images.get("head_cam")
        if img is None:
            logger.warning(f"Search attempt {attempt}: no head_cam image available")
            return None, attempt, total_turned

        # Convert to BGR if needed (OpenCV detection needs BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Assume image is already in displayable format; OpenCV works with BGR
            bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
        else:
            logger.warning(f"Search attempt {attempt}: unexpected image shape {img.shape}")
            return None, attempt, total_turned

        marker = detect_marker(bgr, config, detector)
        if marker is not None:
            logger.warning(
                f"Search: found marker ID={marker['id']} at attempt {attempt} "
                f"(total search turn: {total_turned:.1f}°)"
            )
            return marker, attempt, total_turned

        # Marker not found → rotate AGV one step
        step_deg = config.search_turn_step
        total_turned += step_deg
        if total_turned > config.search_max_turn:
            logger.warning(
                f"Search: exceeded max turn ({config.search_max_turn}°) "
                f"without finding marker"
            )
            return None, attempt, total_turned

        logger.warning(f"Search: turning {step_deg}° (cumulative {total_turned:.1f}°)")
        agv_controller.turn(
            angle=step_deg * DEG_TO_RAD,
            vw=config.turn_speed * DEG_TO_RAD,
            mode=0,
        )
        agv_controller.wait_for_turn_complete(timeout=5.0)
        time.sleep(0.3)  # Brief settle time after AGV stops

    logger.warning("Search: marker not found after all attempts")
    return None, config.search_max_attempts, total_turned


def execute_visual_align(
    robot,
    agv_controller,
    config: VisualAlignConfig,
    logger: logging.Logger,
) -> tuple[bool, str, dict | None]:
    """Execute closed-loop visual alignment using AprilTag.

    Two modes:
      - approach_distance mode (default): align to a fixed distance in front of tag.
      - reference_pose mode: align to a saved reference camera pose.

    Flow:
      1. Search for marker (rotate AGV if not immediately visible)
      2. [Reference mode] Load reference pose
      3. For each iteration:
         a. Detect marker → compute (turn_angle, forward_distance)
         b. Check convergence (within tolerance)
         c. Turn AGV, then drive forward/backward
      4. Return (success, message, trace_dict)

    Args:
        robot: SupreRobotFollower instance (provides get_observation)
        agv_controller: SeerAGVController instance (provides turn/translate)
        config: VisualAlignConfig parameters
        logger: Logger instance

    Returns:
        (success, message, trace) — trace is a dict of execution metrics,
        or None on early failure (marker not found, etc.).
    """
    import time as _time
    _t_start = _time.time()

    # ── Trace accumulator ───────────────────────────────────────────
    # Built incrementally during execution; returned as the 3rd element.
    # Phase 1 fields:
    #   raw_dtheta_initial (float) — first-iteration raw dtheta before gain.
    #     Captures the station's typical initial angular deviation.  Used
    #     by Idea 3 (per-station operating-range learning) to decide
    #     whether a station needs a custom gain schedule or a retreat
    #     before alignment.
    #   per_iteration_log (list[dict]) — per-iteration record of raw
    #     angles/distances before and after gain, so later analysis can
    #     tell which gain step was too aggressive or too conservative.
    #     Fields: iteration, raw_dtheta, gain, applied_dtheta,
    #             raw_forward, applied_forward, capped (bool).
    #   phase1_iterations, phase1_converged, convergence_reason,
    #   final_dtheta_deg, final_forward_m, gain_sequence,
    #   oscillation_detected, mode
    # Phase 2 fields (reference mode only):
    #   phase2_ran, heading_mode, phase2_iterations, phase2_converged,
    #   final_dheading_deg, phase2_gain_sequence, phase2_oscillation,
    #   lateral_err_m, lateral_correction_applied
    # Search fields:
    #   search_attempts, search_total_turned_deg
    trace: dict[str, Any] = {}
    _gain_seq: list[float] = []
    _osc_detected = False
    _iter_log: list[dict[str, Any]] = []   # per-iteration trace (Idea 3)
    detector = _get_detector(config.marker_family)

    # Step 0: Search for marker
    marker, search_attempts, search_total_turned = search_marker(robot, agv_controller, config, logger)
    if marker is None:
        return False, f"Marker ID={config.marker_id} not found during search", None

    logger.warning(f"Marker found: ID={marker['id']}, proceeding to alignment loop")

    # Load reference pose if using reference alignment mode
    ref_tvec = None
    ref_rvec = None
    use_reference = config.reference_pose_path is not None
    if use_reference:
        try:
            ref_tvec, ref_rvec, ref_tag1_tvec, ref_tag1_rvec = load_reference_pose(config.reference_pose_path)
            logger.warning(
                f"Reference pose loaded from {config.reference_pose_path}: "
                f"tvec={ref_tvec}, rvec={ref_rvec}"
            )
            ref_x, ref_y = _marker_to_agv_xy(ref_tvec, config)
            ref_dist = math.sqrt(ref_x**2 + ref_y**2)
            logger.warning(
                f"  [diag] reference: ground_dist={ref_dist:.3f}m"
            )
        except Exception as e:
            return False, f"Failed to load reference pose: {e}", None

    # Step 1: Closed-loop alignment
    aligned = False
    last_raw = 0.0  # track previous raw_dtheta for oscillation detection
    for iteration in range(config.max_iterations):
        # Capture fresh image
        obs = robot.get_observation()
        img = obs.get("images", {}).get("head_cam")
        if img is None:
            return False, f"Iteration {iteration}: head_cam image unavailable", None
        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)

        # Detect marker
        marker = detect_marker(bgr, config, detector)
        if marker is None:
            return False, f"Iteration {iteration}: marker lost (no detection)", None

        # Compute required AGV movement
        if use_reference:
            dtheta_deg, forward_dist = compute_alignment_to_reference(
                marker["tvec"], marker["rvec"], ref_tvec, ref_rvec, config,
            )
            mode_label = "[ref]"
            _log_reference_diagnostics(marker["tvec"], ref_tvec, config, logger)
        else:
            dtheta_deg, forward_dist = compute_agv_movement(
                marker["tvec"], marker["rvec"], config,
            )
            mode_label = ""

        logger.warning(
            f"Iteration {iteration}{mode_label}: marker at "
            f"tvec=[{marker['tvec'][0]:.3f}, {marker['tvec'][1]:.3f}, {marker['tvec'][2]:.3f}]m "
            f"→ dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m"
        )

        # Check convergence (including dead zone for noise immunity)
        converged_angle = abs(dtheta_deg) < config.angle_tolerance
        converged_pos = abs(forward_dist) < config.position_tolerance
        # Dead zone: sub-mm / sub-0.5° corrections are below AGV precision;
        # executing them only adds noise.  Treat as converged.
        in_dead_zone = abs(dtheta_deg) < 0.5 and abs(forward_dist) < 0.005
        if converged_angle and converged_pos or in_dead_zone:
            why = "dead zone" if in_dead_zone else "within tolerance"
            logger.warning(
                f"Alignment converged at iteration {iteration} ({why}): "
                f"dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m"
            )
            aligned = True
            trace["phase1_iterations"] = iteration + 1
            trace["phase1_converged"] = True
            trace["convergence_reason"] = why
            trace["final_dtheta_deg"] = dtheta_deg
            trace["final_forward_m"] = forward_dist
            break

        # ── Decaying gain + oscillation damping ─────────────────────
        if config.warm_gain_sequence is not None:
            gains = config.warm_gain_sequence
        else:
            gains = [0.8, 0.6, 0.5, 0.4]
        gain = gains[iteration] if iteration < len(gains) else gains[-1]
        _gain_seq.append(gain)  # record before clamping

        # Save raw values before gain for the stuck check below
        raw_dtheta = dtheta_deg
        raw_forward = forward_dist

        # Oscillation detection: if dtheta flipped sign AND the amplitude
        # is above 3° (solvePnP noise floor), the AGV overshot.
        if iteration > 0 and raw_dtheta * last_raw < 0 and abs(raw_dtheta) > 3.0:
            gain = min(gain, 0.3)
            _osc_detected = True
            logger.warning(
                f"  oscillation detected (prev={last_raw:.1f}° cur={raw_dtheta:.1f}°) "
                f"→ gain clamped to {gain}"
            )

        # Once the raw correction is small, skip the gain entirely so
        # the AGV can make one clean final move.  Damped micro-steps
        # are often under-executed by the AGV (<5mm command → 0mm actual)
        # which wastes iterations.
        if abs(raw_dtheta) < 1.5 and abs(raw_forward) < 0.05:
            gain = 1.0
            logger.warning(
                f"  gain=1.0 (final undamped step: "
                f"dtheta={raw_dtheta:.1f}°, forward={raw_forward:.3f}m)"
            )

        if gain < 1.0:
            logger.warning(
                f"  gain={gain:.1f} (raw: dtheta={raw_dtheta:.1f}°, "
                f"forward={raw_forward:.3f}m)"
            )
        dtheta_deg = raw_dtheta * gain
        forward_dist = raw_forward * gain
        last_raw = raw_dtheta  # store RAW for next oscillation comparison

        # ── Per-iteration trace (Idea 3) ─────────────────────────────
        # Capture raw & applied values so later analysis can determine
        # which gain step was too aggressive / too conservative for this
        # specific station.
        if iteration == 0:
            trace["raw_dtheta_initial"] = round(raw_dtheta, 2)
        _iter_log.append({
            "iteration": iteration,
            "raw_dtheta": round(raw_dtheta, 2),
            "gain": round(gain, 3),
            "applied_dtheta": round(dtheta_deg, 2),
            "raw_forward": round(raw_forward, 4),
            "applied_forward": round(forward_dist, 4),
            "capped": False,  # set True below if FOV cap engages
        })

        # ── Stuck check: corrections below AGV execution thresholds ──
        # Only accept if raw values are within a loose envelope —
        # otherwise a ~1.2cm residual × 0.4 gain ≈ 5mm would falsely
        # trigger convergence when AGV can't execute the tiny correction.
        raw_close = (abs(raw_dtheta) < config.angle_tolerance * 2
                     and abs(raw_forward) < config.position_tolerance * 3)
        if abs(dtheta_deg) < 1.0 and abs(forward_dist) < 0.005 and raw_close:
            logger.warning(
                f"Alignment converged at iteration {iteration} (below exec threshold): "
                f"dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m "
                f"(raw: {raw_dtheta:.1f}° {raw_forward:.3f}m)"
            )
            aligned = True
            trace["phase1_iterations"] = iteration + 1
            trace["phase1_converged"] = True
            trace["convergence_reason"] = "exec_threshold"
            trace["final_dtheta_deg"] = dtheta_deg
            trace["final_forward_m"] = forward_dist
            break

        # ── Per-step safety cap (keep marker in camera FOV) ───────────
        # Turning too far in one step pushes the marker out of the
        # camera's field of view (~78° HFOV for fx=392@640px).  Cap
        # each iteration's dtheta at ~1/3 half-FOV so the marker
        # stays visible between iterations.
        max_turn = 15.0  # degrees
        max_fwd  = 0.30  # meters
        did_cap = False
        if abs(dtheta_deg) > max_turn:
            # Scale forward proportionally (shorten turn → shorter path)
            ratio = max_turn / abs(dtheta_deg)
            logger.warning(
                f"  capping turn {dtheta_deg:.1f}° → {dtheta_deg*ratio:.1f}° "
                f"(keep marker in FOV)"
            )
            dtheta_deg = dtheta_deg * ratio
            forward_dist *= ratio
            did_cap = True
        if abs(forward_dist) > max_fwd:
            logger.warning(
                f"  capping forward {forward_dist:.3f}m → "
                f"{math.copysign(max_fwd, forward_dist):.3f}m"
            )
            forward_dist = math.copysign(max_fwd, forward_dist)
            did_cap = True
        if did_cap and _iter_log:
            _iter_log[-1]["capped"] = True

        # Step 1a: Turn AGV
        if abs(dtheta_deg) > 0.5:  # skip turns < 0.5° (below AGV precision)
            turn_deg = dtheta_deg
            vw = config.turn_speed * DEG_TO_RAD
            if turn_deg < 0:
                vw = -vw
            logger.warning(f"  Turning {turn_deg:.2f}°")
            agv_controller.turn(
                angle=abs(turn_deg) * DEG_TO_RAD,
                vw=vw,
                mode=0,
            )
            agv_controller.wait_for_turn_complete(timeout=10.0)
            time.sleep(0.3)

        # Step 1b: Drive forward/backward
        if abs(forward_dist) > 0.005:  # skip moves < 5mm
            if forward_dist > 0:
                vx = config.translate_speed
                direction = "forward"
            else:
                vx = -config.translate_speed
                direction = "backward"
            dist = abs(forward_dist)
            logger.warning(f"  Driving {direction} {dist:.3f}m at {abs(vx):.2f}m/s")
            ok = agv_controller.translate(
                dist=dist,
                vx=vx,
                mode=0,
            )
            if not ok:
                logger.error("  Translate command failed!")
            else:
                agv_controller.wait_for_translate_complete(timeout=10.0)
                time.sleep(0.3)

    # ── Finalize Phase 1 trace ──────────────────────────────────────
    # Populated at convergence break points above; here we fill in the
    # remaining fields for the case where the loop exhausted.
    if "phase1_iterations" not in trace:
        # Loop exhausted without explicit convergence
        trace["phase1_iterations"] = config.max_iterations
        trace["phase1_converged"] = False
        trace["convergence_reason"] = "loop_exhausted"
        trace["final_dtheta_deg"] = dtheta_deg if "dtheta_deg" in dir() else 999.0
        trace["final_forward_m"] = forward_dist if "forward_dist" in dir() else 999.0
    trace["gain_sequence"] = _gain_seq
    trace["oscillation_detected"] = _osc_detected
    trace["mode"] = "reference" if use_reference else "approach"
    trace["per_iteration_log"] = _iter_log
    trace["search"] = {
        "attempts": search_attempts + 1 if marker else search_attempts,
        "total_turned_deg": search_total_turned,
    }

    # ── Phase 2: heading alignment ───────────────────────────────────
    # Single-tag: heading correction from ref_y offset.
    # Dual-tag: absolute workstation heading from tag_0→tag_1 vector,
    #   immune to workstation rotation.
    if use_reference and aligned:
        dual_tag = (config.tag_1_id is not None and ref_tag1_tvec is not None)

        if dual_tag:
            # ── Dual-tag heading with convergence loop ────────────────
            # Each iteration: (1) correct heading from tag_0→tag_1 vector
            #                 (2) correct lateral offset via mini Phase 1
            #                 (3) re-check heading
            _p2_gain_seq: list[float] = []
            _p2_osc_detected = False
            _p2_lateral_err_m = 0.0
            _p2_lateral_applied = False
            dh_prev = None
            for dh_iter in range(config.max_iterations):
                obs = robot.get_observation()
                img = obs.get("images", {}).get("head_cam")
                if img is None:
                    logger.warning("Phase 2: head_cam unavailable")
                    break
                bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)

                # ── Step A: heading correction ────────────────────────
                m0 = detect_marker(bgr, config, detector,
                                   target_id=config.marker_id)
                m1 = detect_marker(bgr, config, detector,
                                   target_id=config.tag_1_id)
                if m0 is None or m1 is None:
                    logger.warning(
                        f"Phase 2 iter {dh_iter}: "
                        f"{'tag_0' if m0 is None else ''}"
                        f"{' ' if m0 is None and m1 is None else ''}"
                        f"{'tag_1' if m1 is None else ''} missing"
                    )
                    break

                x0, y0 = _marker_to_agv_xy(m0["tvec"], config)
                x1, y1 = _marker_to_agv_xy(m1["tvec"], config)
                cur_heading = math.atan2(y1 - y0, x1 - x0)

                x0r, y0r = _marker_to_agv_xy(ref_tvec, config)
                x1r, y1r = _marker_to_agv_xy(ref_tag1_tvec, config)
                ref_heading = math.atan2(y1r - y0r, x1r - x0r)

                dheading_deg = (cur_heading - ref_heading) * RAD_TO_DEG
                logger.warning(
                    f"Phase 2 iter {dh_iter}: cur={cur_heading*RAD_TO_DEG:.1f}° "
                    f"ref={ref_heading*RAD_TO_DEG:.1f}° "
                    f"→ correction={dheading_deg:.1f}°"
                )
                if abs(dheading_deg) <= 1.0:
                    logger.warning("Phase 2 dual-tag: heading converged")
                    trace["phase2"] = {
                        "ran": True,
                        "heading_mode": "dual_tag",
                        "iterations": dh_iter + 1,
                        "converged": True,
                        "final_dheading_deg": dheading_deg,
                        "gain_sequence": _p2_gain_seq,
                        "oscillation_detected": _p2_osc_detected,
                        "lateral_err_m": _p2_lateral_err_m,
                        "lateral_correction_applied": _p2_lateral_applied,
                    }
                    trace["duration_s"] = time.time() - t_start
                    return True, "Aligned (dual-tag, heading OK)", trace

                # ── Heading gain + two-way overshoot detection ─────
                raw_dheading = dheading_deg  # save for oscillation detection
                # First iteration: no gain — let the AGV make a clean
                # turn so the overshoot bias (±1.5°) works in our favour.
                # Subsequent iterations: decaying gain to damp overshoot.
                if dh_iter == 0:
                    gain_p2 = 1.0
                else:
                    gains_p2 = [0.7, 0.5, 0.4]
                    gain_p2 = gains_p2[dh_iter - 1] if dh_iter - 1 < len(gains_p2) else 0.4

                if dh_prev is not None:
                    if raw_dheading * dh_prev < 0:
                        # Direction flipped — classical oscillation
                        gain_p2 = min(gain_p2, 0.3)
                        _p2_osc_detected = True
                        logger.warning(
                            f"  oscillation detected → gain clamped to {gain_p2}"
                        )
                    elif abs(raw_dheading) >= abs(dh_prev) * 0.95:
                        # Error barely shrinking (or growing) — overshoot
                        gain_p2 = min(gain_p2, 0.4)
                        logger.warning(
                            f"  overshoot (not converging) → gain clamped to {gain_p2}"
                        )

                dheading_deg = raw_dheading * gain_p2
                _p2_gain_seq.append(gain_p2)
                if gain_p2 < 1.0:
                    logger.warning(
                        f"  gain={gain_p2:.1f} (raw={raw_dheading:.1f}° → cmd={dheading_deg:.1f}°)"
                    )

                vw_s = 1.0 if dheading_deg > 0 else -1.0
                agv_controller.turn(
                    angle=abs(dheading_deg) * DEG_TO_RAD,
                    vw=config.turn_speed * DEG_TO_RAD * vw_s,
                    mode=0,
                )
                agv_controller.wait_for_turn_complete(timeout=10.0)
                time.sleep(0.3)
                dh_prev = raw_dheading  # store RAW for next comparison

            # ── Step B: one-shot lateral correction (after heading settled) ──
            # Moved outside the heading loop so the lateral turn doesn't
            # undo Step A's work.  Compute the lateral→angle correction
            # once and apply it after heading has converged or loop exhausted.
            obs = robot.get_observation()
            img = obs.get("images", {}).get("head_cam")
            if img is not None:
                bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
                m0 = detect_marker(bgr, config, detector,
                                   target_id=config.marker_id)
                if m0 is not None:
                    cur_x, cur_y = _marker_to_agv_xy(m0["tvec"], config)
                    dy = cur_y - ref_y
                    lateral_err = abs(dy)
                    logger.warning(
                        f"  lateral (post-heading): cur=({cur_x:.3f},{cur_y:.3f}) "
                        f"ref=({ref_x:.3f},{ref_y:.3f}) dy={dy*100:.1f}cm"
                    )
                    if lateral_err >= 0.01:
                        _p2_lateral_err_m = float(lateral_err)
                        _p2_lateral_applied = True
                        dtheta_lat, dforward_lat = compute_alignment_to_reference(
                            m0["tvec"], m0["rvec"], ref_tvec, ref_rvec, config,
                        )
                        if abs(dtheta_lat) > 1.0:
                            logger.warning(f"  lateral turn: {dtheta_lat:.1f}°")
                            vw = config.turn_speed * DEG_TO_RAD
                            if dtheta_lat > 0:
                                vw = abs(vw)
                            else:
                                vw = -abs(vw)
                            agv_controller.turn(
                                angle=abs(dtheta_lat) * DEG_TO_RAD,
                                vw=vw, mode=0,
                            )
                            agv_controller.wait_for_turn_complete(timeout=10.0)
                            time.sleep(0.3)
                        if abs(dforward_lat) > 0.005:
                            dforward_lat = math.copysign(min(abs(dforward_lat), 0.15), dforward_lat)
                            logger.warning(f"  lateral drive: {dforward_lat:.3f}m")
                            vx = config.translate_speed
                            if dforward_lat < 0:
                                vx = -vx
                            agv_controller.translate(
                                dist=abs(dforward_lat), vx=vx, mode=0,
                            )
                            agv_controller.wait_for_translate_complete(timeout=10.0)
                            time.sleep(0.3)

            # Exhausted iterations without convergence
            msg = (f"Aligned (dual-tag, heading leftover {dheading_deg:.1f}°)"
                   if abs(dheading_deg) <= 3.0
                   else f"Dual-tag heading not converged: {dheading_deg:.1f}°")
            trace["phase2"] = {
                "ran": True,
                "heading_mode": "dual_tag",
                "iterations": config.max_iterations,
                "converged": abs(dheading_deg) <= 3.0,
                "final_dheading_deg": dheading_deg,
                "gain_sequence": _p2_gain_seq,
                "oscillation_detected": _p2_osc_detected,
                "lateral_err_m": _p2_lateral_err_m,
                "lateral_correction_applied": _p2_lateral_applied,
            }
            trace["duration_s"] = time.time() - t_start
            return abs(dheading_deg) <= 3.0, msg, trace

        # ── Single-tag heading (original logic) ─────────────────────
        ref_x, ref_y = _marker_to_agv_xy(ref_tvec, config)
        heading_correction = -math.atan2(ref_y, ref_x) * RAD_TO_DEG
        if abs(heading_correction) > 0.5:
            logger.warning(
                f"Phase 2 heading correction: {heading_correction:.1f}° "
                f"(ref lateral offset: {ref_y*100:.0f}cm)"
            )
            vw_sign = 1.0 if heading_correction > 0 else -1.0
            agv_controller.turn(
                angle=abs(heading_correction) * DEG_TO_RAD,
                vw=config.turn_speed * DEG_TO_RAD * vw_sign,
                mode=0,
            )
            agv_controller.wait_for_turn_complete(timeout=10.0)
            time.sleep(0.3)
            return True, f"Aligned (dist + heading, correction {heading_correction:.1f}°)", {
                **trace,
                "phase2": {
                    "ran": True,
                    "heading_mode": "single_tag",
                    "iterations": 1,
                    "converged": True,
                    "final_dheading_deg": heading_correction,
                    "gain_sequence": [1.0],
                    "oscillation_detected": False,
                    "lateral_err_m": abs(ref_y),
                    "lateral_correction_applied": True,
                },
                "duration_s": time.time() - t_start,
            }
        else:
            logger.warning(
                f"Phase 2: heading correction {heading_correction:.1f}° < 0.5° — skipped"
            )
            # Heading already fine — alignment succeeded
            trace["phase2"] = {
                "ran": True,
                "heading_mode": "single_tag",
                "iterations": 1,
                "converged": True,
                "final_dheading_deg": heading_correction,
                "gain_sequence": [1.0],
                "oscillation_detected": False,
                "lateral_err_m": abs(ref_y),
                "lateral_correction_applied": False,
            }
            trace["duration_s"] = time.time() - t_start
            return True, "Aligned (dist only, heading already OK)", trace

    # Did not converge within max_iterations (Phase 1 exhausted, no Phase 2)
    logger.warning(
        f"Alignment did not converge after {config.max_iterations} iterations "
        f"(last: dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m)"
    )
    trace["duration_s"] = time.time() - t_start
    return False, f"Did not converge after {config.max_iterations} iterations", trace