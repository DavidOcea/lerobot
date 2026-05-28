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
):
    """Detect AprilTag markers in an image.

    Returns the first matching marker detection (or None).
    Each detection is a dict with: id, corners, rvec, tvec.
    """
    if detector is None:
        detector = _get_detector(config.marker_family)

    corners, ids, rejected = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        return None

    K, d = _get_camera_params(config)
    marker_size_m = config.marker_size

    # Estimate pose using solvePnP (estimatePoseSingleMarkers removed in OpenCV 4.12+).
    # AprilTag corner order: [top-left, top-right, bottom-right, bottom-left]
    # The 3D object points are the marker's corners in its own coordinate frame
    # (origin at marker center, marker_size side length, z=0 plane).
    half = marker_size_m / 2.0
    obj_points = np.array([
        [-half, half, 0.0],   # top-left
        [half, half, 0.0],    # top-right
        [half, -half, 0.0],   # bottom-right
        [-half, -half, 0.0],  # bottom-left
    ], dtype=np.float32)

    # Find the target marker (or first visible one if marker_id is None)
    for i, marker_id in enumerate(ids.flatten()):
        if config.marker_id is None or marker_id == config.marker_id:
            # solvePnP for this marker's corners
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
    z_horiz = math.cos(pitch_rad) * z_cam + math.sin(pitch_rad) * y_cam
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


def save_reference_pose(path: str, tvec: np.ndarray, rvec: np.ndarray):
    """Save a reference camera pose (relative to AprilTag) to JSON."""
    import json
    data = {
        "tvec": tvec.tolist(),
        "rvec": rvec.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_reference_pose(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load reference camera pose from JSON. Returns (tvec, rvec)."""
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array(data["tvec"]), np.array(data["rvec"])


def compute_alignment_to_reference(
    tvec_cur: np.ndarray,
    rvec_cur: np.ndarray,
    ref_tvec: np.ndarray,
    ref_rvec: np.ndarray,
    config: VisualAlignConfig,
) -> tuple[float, float]:
    """Compute AGV movement to align current view to reference view.

    Both current and reference marker poses are in camera frame (solvePnP output).
    Uses the same camera→AGV transform as compute_agv_movement, but the target
    is the reference pose instead of a fixed approach_distance.

    Strategy for differential-drive AGV (no lateral movement):
      1. Turn to face the marker (same as approach_distance mode).
      2. Drive forward/backward to match the reference distance.

    Returns (dtheta_deg, forward_dist_m).
    """
    # Current marker position in AGV frame
    cur_x, cur_y = _marker_to_agv_xy(tvec_cur, config)
    # Reference marker position in AGV frame
    ref_x, ref_y = _marker_to_agv_xy(ref_tvec, config)

    # Distance from AGV to marker (ground-plane projection)
    cur_dist = math.sqrt(cur_x**2 + cur_y**2)
    ref_dist = math.sqrt(ref_x**2 + ref_y**2)

    # Same strategy as compute_agv_movement: turn to face the marker,
    # then drive forward/backward to match the reference distance.
    # Using atan2(cur_y, cur_x) avoids the sign-reversal bug when
    # ref_x < cur_x (which happens when reference was closer to tag).
    dtheta_rad = math.atan2(cur_y, cur_x)
    dtheta_deg = dtheta_rad * RAD_TO_DEG
    forward_dist = cur_dist - ref_dist  # +forward to get closer, -backward if overshot

    return dtheta_deg, forward_dist


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
) -> dict | None:
    """Search for the target AprilTag by rotating the AGV.

    Rotates AGV left (CCW) in small steps until the marker is found
    in the head camera.  Returns the marker detection dict, or None
    if not found after exhausting search_max_attempts.

    Note: search rotation is cumulative, but the subsequent visual_align
    loop will re-align the AGV to the correct heading, so we don't need
    to undo the search rotation.
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
            return None

        # Convert to BGR if needed (OpenCV detection needs BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Assume image is already in displayable format; OpenCV works with BGR
            bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
        else:
            logger.warning(f"Search attempt {attempt}: unexpected image shape {img.shape}")
            return None

        marker = detect_marker(bgr, config, detector)
        if marker is not None:
            logger.warning(
                f"Search: found marker ID={marker['id']} at attempt {attempt} "
                f"(total search turn: {total_turned:.1f}°)"
            )
            return marker

        # Marker not found → rotate AGV one step
        step_deg = config.search_turn_step
        total_turned += step_deg
        if total_turned > config.search_max_turn:
            logger.warning(
                f"Search: exceeded max turn ({config.search_max_turn}°) "
                f"without finding marker"
            )
            return None

        logger.warning(f"Search: turning {step_deg}° (cumulative {total_turned:.1f}°)")
        agv_controller.turn(
            angle=step_deg * DEG_TO_RAD,
            vw=config.turn_speed * DEG_TO_RAD,
            mode=0,
        )
        agv_controller.wait_for_turn_complete(timeout=5.0)
        time.sleep(0.3)  # Brief settle time after AGV stops

    logger.warning("Search: marker not found after all attempts")
    return None


def execute_visual_align(
    robot,
    agv_controller,
    config: VisualAlignConfig,
    logger: logging.Logger,
) -> tuple[bool, str]:
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
      4. Return (success, message)

    Args:
        robot: SupreRobotFollower instance (provides get_observation)
        agv_controller: SeerAGVController instance (provides turn/translate)
        config: VisualAlignConfig parameters
        logger: Logger instance

    Returns:
        (success: bool, message: str)
    """
    detector = _get_detector(config.marker_family)

    # Step 0: Search for marker
    marker = search_marker(robot, agv_controller, config, logger)
    if marker is None:
        return False, f"Marker ID={config.marker_id} not found during search"

    logger.warning(f"Marker found: ID={marker['id']}, proceeding to alignment loop")

    # Load reference pose if using reference alignment mode
    ref_tvec = None
    ref_rvec = None
    use_reference = config.reference_pose_path is not None
    if use_reference:
        try:
            ref_tvec, ref_rvec = load_reference_pose(config.reference_pose_path)
            logger.warning(
                f"Reference pose loaded from {config.reference_pose_path}: "
                f"tvec={ref_tvec}, rvec={ref_rvec}"
            )
        except Exception as e:
            return False, f"Failed to load reference pose: {e}"

    # Step 1: Closed-loop alignment
    for iteration in range(config.max_iterations):
        # Capture fresh image
        obs = robot.get_observation()
        img = obs.get("images", {}).get("head_cam")
        if img is None:
            return False, f"Iteration {iteration}: head_cam image unavailable"
        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)

        # Detect marker
        marker = detect_marker(bgr, config, detector)
        if marker is None:
            return False, f"Iteration {iteration}: marker lost (no detection)"

        # Compute required AGV movement
        if use_reference:
            dtheta_deg, forward_dist = compute_alignment_to_reference(
                marker["tvec"], marker["rvec"], ref_tvec, ref_rvec, config,
            )
            mode_label = "[ref]"
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

        # Check convergence
        converged_angle = abs(dtheta_deg) < config.angle_tolerance
        converged_pos = abs(forward_dist) < config.position_tolerance
        if converged_angle and converged_pos:
            logger.warning(f"Alignment converged at iteration {iteration}")
            return True, f"Aligned after {iteration + 1} iterations"

        # Step 1a: Turn AGV to face marker
        if abs(dtheta_deg) > config.angle_tolerance:
            turn_deg = dtheta_deg  # positive = CCW (left turn)
            # Seer AGV: angle=absolute magnitude, vw sign controls direction
            # vw > 0 = CCW (left), vw < 0 = CW (right)
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
        if abs(forward_dist) > config.position_tolerance:
            # Seer AGV translate: dist=absolute magnitude, vx sign controls direction
            # vx > 0 = forward, vx < 0 = backward
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

    # Did not converge within max_iterations
    logger.warning(
        f"Alignment did not converge after {config.max_iterations} iterations "
        f"(last: dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m)"
    )
    return False, f"Did not converge after {config.max_iterations} iterations"