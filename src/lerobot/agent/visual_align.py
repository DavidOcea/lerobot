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

    # Estimate pose for all detected markers
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size_m, K, d,
    )

    # Find the target marker (or first visible one if marker_id is None)
    for i, marker_id in enumerate(ids.flatten()):
        if config.marker_id is None or marker_id == config.marker_id:
            return {
                "id": int(marker_id),
                "corners": corners[i],
                "rvec": rvecs[i][0],
                "tvec": tvecs[i][0],
            }

    return None


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

    Camera coordinate convention (OpenCV):
      x: right, y: down, z: forward (into the scene)

    We want the AGV to stop at config.approach_distance meters in
    front of the marker, not right on top of it.

    Returns (dtheta_deg, forward_dist_m):
      dtheta_deg: angle AGV must turn (positive = left/CCW)
      forward_dist_m: distance AGV must move forward after turning
    """
    # Marker position in camera frame
    x_cam = tvec[0]  # rightward offset
    y_cam = tvec[1]  # downward offset (not used for ground-plane alignment)
    z_cam = tvec[2]  # forward distance

    # Apply camera-AGV offset (currently (0,0,0), so no change)
    offset_yaw_rad = config.camera_offset_yaw * DEG_TO_RAD
    offset_x = config.camera_offset_x
    offset_y = config.camera_offset_y

    # Rotate marker position from camera frame to AGV frame.
    # If camera faces same direction as AGV (offset_yaw=0):
    #   AGV forward = camera z, AGV left = -camera x
    cos_yaw = math.cos(offset_yaw_rad)
    sin_yaw = math.sin(offset_yaw_rad)
    dx_agv = z_cam * cos_yaw - x_cam * sin_yaw + offset_x  # AGV forward direction
    dy_agv = -z_cam * sin_yaw - x_cam * cos_yaw + offset_y  # AGV left direction

    # Angle to face the marker: atan2(left_offset, forward_offset)
    dtheta_rad = math.atan2(dy_agv, dx_agv)
    dtheta_deg = dtheta_rad * RAD_TO_DEG

    # After turning by dtheta, the marker will be straight ahead.
    # Total distance = sqrt(dx^2 + dy^2), minus the desired approach distance.
    total_dist = math.sqrt(dx_agv**2 + dy_agv**2)
    forward_dist = total_dist - config.approach_distance

    # Don't drive backward past the marker — if we're already closer
    # than approach_distance, just adjust angle.
    if forward_dist < 0:
        forward_dist = 0.0

    return dtheta_deg, forward_dist


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
            logger.info(
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

        logger.info(f"Search: turning {step_deg}° (cumulative {total_turned:.1f}°)")
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

    Flow:
      1. Search for marker (rotate AGV if not immediately visible)
      2. For each iteration:
         a. Detect marker → compute (turn_angle, forward_distance)
         b. Check convergence (within tolerance)
         c. Turn AGV to face marker
         d. Drive forward/backward toward marker
      3. Return (success, message)

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

    logger.info(f"Marker found: ID={marker['id']}, proceeding to alignment loop")

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
        dtheta_deg, forward_dist = compute_agv_movement(
            marker["tvec"], marker["rvec"], config,
        )

        logger.info(
            f"Iteration {iteration}: marker at "
            f"tvec=[{marker['tvec'][0]:.3f}, {marker['tvec'][1]:.3f}, {marker['tvec'][2]:.3f}]m "
            f"→ dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m"
        )

        # Check convergence
        converged_angle = abs(dtheta_deg) < config.angle_tolerance
        converged_pos = forward_dist < config.position_tolerance
        if converged_angle and converged_pos:
            logger.info(f"Alignment converged at iteration {iteration}")
            return True, f"Aligned after {iteration + 1} iterations"

        # Step 1a: Turn AGV to face marker
        if abs(dtheta_deg) > config.angle_tolerance:
            turn_deg = dtheta_deg  # positive = CCW (left turn)
            logger.info(f"  Turning {turn_deg:.2f}°")
            agv_controller.turn(
                angle=abs(turn_deg) * DEG_TO_RAD,
                vw=config.turn_speed * DEG_TO_RAD,
                mode=0,
            )
            agv_controller.wait_for_turn_complete(timeout=10.0)
            time.sleep(0.3)

        # Step 1b: Drive forward/backward
        if forward_dist > config.position_tolerance:
            # Forward: positive vx
            vx = config.translate_speed
            logger.info(f"  Driving forward {forward_dist:.3f}m at {vx:.2f}m/s")
            agv_controller.translate(
                dist=forward_dist,
                vx=vx,
                mode=0,
            )
            agv_controller.wait_for_translate_complete(timeout=10.0)
            time.sleep(0.3)

    # Did not converge within max_iterations
    logger.warning(
        f"Alignment did not converge after {config.max_iterations} iterations "
        f"(last: dtheta={dtheta_deg:.2f}°, forward={forward_dist:.3f}m)"
    )
    return False, f"Did not converge after {config.max_iterations} iterations"