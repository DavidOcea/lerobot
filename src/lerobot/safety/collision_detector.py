"""
Collision detection based on torque/force sensing.

This module implements collision detection by monitoring joint torque anomalies.
The detector uses a base torque calibration and identifies significant deviations
that indicate physical contact with obstacles.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import CollisionConfig

logger = logging.getLogger(__name__)


@dataclass
class CollisionResult:
    """Result of collision detection check."""

    is_detected: bool = False
    affected_joints: dict[str, float] = field(default_factory=dict)  # joint -> anomaly value
    severity: str = "none"  # "none", "low", "medium", "high"
    timestamp: float = field(default_factory=time.time)
    raw_torques: dict[str, float] = field(default_factory=dict)
    estimated_inertial_torques: dict[str, float] = field(default_factory=dict)


@dataclass
class CollisionEvent:
    """Represents a collision event with full context."""

    timestamp: float
    affected_joints: dict[str, float]
    severity: str
    observation: dict[str, Any]
    action: dict[str, Any] | None = None
    estimated_inertial_torques: dict[str, float] = field(default_factory=dict)


class CollisionDetector:
    """Detects collisions based on torque sensing anomalies.

    The detector works by:
    1. Calibrating base torques (expected torques during normal operation)
    2. Monitoring current torques for significant deviations
    3. Compensating for inertial effects during motion
    4. Using a detection window to filter false positives

    Usage:
        detector = CollisionDetector(config)
        detector.calibrate_base_torques(robot)
        # In control loop:
        result = detector.check_collision(observation, action)
        if result.is_detected:
            handle_collision(result)
    """

    def __init__(self, config: CollisionConfig):
        """Initialize the collision detector.

        Args:
            config: Collision detection configuration.
        """
        self.config = config

        # Calibration data
        self.base_torques: dict[str, float] = {}
        self.base_torque_stds: dict[str, float] = {}
        self.is_calibrated = False

        # Detection buffer for window-based confirmation
        self.detection_buffer: deque[dict[str, float]] = deque(maxlen=config.detection_window)

        # Velocity tracking for inertia compensation
        self._prev_positions: dict[str, float] = {}
        self._prev_timestamp: float | None = None

        # Statistics
        self._total_checks = 0
        self._collision_count = 0

    def calibrate_base_torques(
        self, robot, num_samples: int | None = None, verbose: bool = True
    ) -> dict[str, float]:
        """Calibrate base torques by sampling robot at rest.

        Collects multiple samples to establish the expected torque baseline
        for each joint when the robot is in a safe, resting state.

        Args:
            robot: Robot instance with get_observation() method.
            num_samples: Number of samples to collect (uses config if None).
            verbose: Whether to log calibration progress.

        Returns:
            Dictionary of calibrated base torques per joint.
        """
        if num_samples is None:
            num_samples = self.config.calibration_samples

        if verbose:
            logger.info(f"Calibrating base torques with {num_samples} samples...")

        torque_samples: dict[str, list[float]] = {}

        for i in range(num_samples):
            observation = robot.get_observation()

            # Extract force/torque values
            for key, value in observation.items():
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    if joint_name not in torque_samples:
                        torque_samples[joint_name] = []
                    torque_samples[joint_name].append(float(value))

            if verbose and (i + 1) % 20 == 0:
                logger.info(f"Calibration progress: {i + 1}/{num_samples}")

            time.sleep(0.01)  # Small delay between samples

        # Compute statistics
        self.base_torques = {}
        self.base_torque_stds = {}

        for joint_name, samples in torque_samples.items():
            samples_array = np.array(samples)
            mean_torque = float(np.mean(samples_array))
            std_torque = float(np.std(samples_array))

            # Check if calibration is stable
            if std_torque > self.config.calibration_threshold:
                logger.warning(
                    f"Joint {joint_name} shows high torque variance during "
                    f"calibration (std={std_torque:.3f} Nm). "
                    f"Robot may not be in a stable resting state."
                )

            self.base_torques[joint_name] = mean_torque
            self.base_torque_stds[joint_name] = std_torque

        self.is_calibrated = True

        if verbose:
            logger.info(f"Calibration complete. Base torques: {self.base_torques}")

        return self.base_torques

    def check_collision(
        self, observation: dict[str, Any], action: dict[str, Any] | None = None
    ) -> CollisionResult:
        """Check if a collision has occurred based on torque anomalies.

        Args:
            observation: Current observation dict containing {joint}.force values.
            action: Current action dict containing {joint}.pos values (for inertia compensation).

        Returns:
            CollisionResult with detection status and details.
        """
        self._total_checks += 1
        timestamp = time.time()

        # Get current timestamp if not provided
        current_time = timestamp

        # Extract torque data
        current_torques = {k: v for k, v in observation.items() if ".force" in k}

        # Get current positions for velocity estimation
        current_positions = {k: v for k, v in observation.items() if ".pos" in k}

        # Initialize result
        result = CollisionResult(timestamp=timestamp, raw_torques=current_torques)

        if not self.is_calibrated:
            logger.warning("Collision detector not calibrated. Skipping detection.")
            return result

        # Estimate and subtract inertial torques
        estimated_inertial = {}
        if self.config.velocity_compensation and action is not None:
            estimated_inertial = self._estimate_inertial_torques(
                current_positions, action, current_time
            )

        result.estimated_inertial_torques = estimated_inertial

        # Check each joint for torque anomaly
        anomalies = {}
        for joint_force_key, torque_value in current_torques.items():
            joint_name = joint_force_key.replace(".force", "")

            # Get base torque for this joint
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Get inertial compensation
            inertial_comp = estimated_inertial.get(joint_name, 0.0)

            # Compute anomaly (deviation from baseline)
            anomaly = abs(torque_value - base_torque)
            if self.config.velocity_compensation:
                # Subtract estimated inertial component
                anomaly = max(0, anomaly - inertial_comp)

            # Get threshold for this joint
            threshold = self.config.joint_specific_thresholds.get(
                joint_name, self.config.collision_threshold
            )

            # Check if anomaly exceeds threshold
            if anomaly > threshold:
                anomalies[joint_name] = anomaly

            # Check absolute torque limit
            if abs(torque_value) > self.config.max_torque_limit:
                anomalies[joint_name] = float("inf")  # Force immediate detection

        result.affected_joints = anomalies

        # Determine severity
        if anomalies:
            max_anomaly = max(anomalies.values())
            if max_anomaly == float("inf"):
                result.severity = "high"
            elif max_anomaly > self.config.collision_threshold * 2:
                result.severity = "high"
            elif max_anomaly > self.config.collision_threshold * 1.5:
                result.severity = "medium"
            else:
                result.severity = "low"

        # Window-based confirmation
        if anomalies:
            self.detection_buffer.append(anomalies)

            # Check if we have consistent detections
            if len(self.detection_buffer) >= self.config.detection_window:
                # Count how many times each joint was detected
                joint_detection_counts: dict[str, int] = {}
                for detection in self.detection_buffer:
                    for joint_name in detection:
                        joint_detection_counts[joint_name] = (
                            joint_detection_counts.get(joint_name, 0) + 1
                        )

                # Trigger collision if any joint is detected in enough windows
                confirmed_joints = {
                    jn: anomalies.get(jn, 0)
                    for jn, count in joint_detection_counts.items()
                    if count >= self.config.detection_window // 2
                }

                if confirmed_joints:
                    result.is_detected = True
                    result.affected_joints = confirmed_joints
                    self._collision_count += 1
                    logger.warning(
                        f"Collision detected! Affected joints: {confirmed_joints}, "
                        f"Severity: {result.severity}"
                    )
        else:
            # Clear buffer if no anomaly
            if len(self.detection_buffer) > 0:
                self.detection_buffer.clear()

        return result

    def _estimate_inertial_torques(
        self, current_positions: dict[str, float], action: dict[str, Any], current_time: float
    ) -> dict[str, float]:
        """Estimate inertial torque components for velocity compensation.

        Args:
            current_positions: Current joint positions.
            action: Target action positions.
            current_time: Current timestamp.

        Returns:
            Dictionary of estimated inertial torques per joint.
        """
        estimated: dict[str, float] = {}

        if self._prev_timestamp is None:
            # First call - just store state
            for key, value in current_positions.items():
                joint_name = key.replace(".pos", "")
                self._prev_positions[joint_name] = float(value)
            self._prev_timestamp = current_time
            return estimated

        # Compute time delta
        dt = current_time - self._prev_timestamp
        if dt <= 0:
            return estimated

        # Estimate velocity and acceleration
        for key, current_pos in current_positions.items():
            joint_name = key.replace(".pos", "")

            if joint_name not in self._prev_positions:
                continue

            prev_pos = self._prev_positions[joint_name]

            # Estimate velocity
            velocity = (current_pos - prev_pos) / dt

            # Get joint inertia
            inertia = self.config.joint_inertia.get(joint_name, 0.1)

            # Estimate inertial torque = I * alpha (approximate)
            # We use velocity as a proxy for acceleration in this simplified model
            inertial_torque = abs(velocity * inertia * self.config.inertia_compensation_factor)

            estimated[joint_name] = inertial_torque

            # Update previous position
            self._prev_positions[joint_name] = float(current_pos)

        self._prev_timestamp = current_time

        return estimated

    def reset(self):
        """Reset detector state (clear buffer and statistics).

        Useful after recovery operations.
        """
        self.detection_buffer.clear()
        self._prev_positions.clear()
        self._prev_timestamp = None

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with detection statistics.
        """
        return {
            "total_checks": self._total_checks,
            "collision_count": self._collision_count,
            "collision_rate": (
                self._collision_count / self._total_checks if self._total_checks > 0 else 0
            ),
            "is_calibrated": self.is_calibrated,
            "base_torques": self.base_torques.copy(),
        }

    def set_base_torque(self, joint_name: str, torque: float):
        """Manually set base torque for a specific joint.

        Useful for fine-tuning after initial calibration.

        Args:
            joint_name: Name of the joint.
            torque: Base torque value in Nm.
        """
        self.base_torques[joint_name] = torque
        self.is_calibrated = True
