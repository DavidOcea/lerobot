"""
Adaptive Collision Threshold System

This module provides adaptive threshold adjustment based on robot motion state,
significantly reducing false positives while maintaining high sensitivity.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from lerobot.safety.collision_detector import CollisionConfig, CollisionDetector, CollisionResult

logger = logging.getLogger(__name__)


class MotionPhase(Enum):
    """Robot motion phases for adaptive thresholds."""

    STATIC = "static"  # Robot not moving
    ACCELERATING = "accelerating"  # Starting motion
    CRUISING = "cruising"  # Constant velocity
    DECELERATING = "decelerating"  # Slowing down
    FINE_MOTION = "fine_motion"  # Precision movements
    GRIPPER_ACTIVE = "gripper_active"  # Gripper operation


@dataclass
class AdaptiveThresholdConfig(CollisionConfig):
    """Configuration for adaptive collision threshold system."""

    # Base thresholds (from CollisionConfig)
    collision_threshold: float = 0.8

    # Motion state detection
    velocity_threshold: float = 0.01  # rad/s, below this is static
    acceleration_threshold: float = 0.1  # rad/s², below this is cruising
    fine_motion_threshold: float = 0.05  # rad/s, fine motion threshold

    # Phase-specific threshold multipliers
    static_multiplier: float = 0.5  # More sensitive when static
    accelerating_multiplier: float = 1.5  # Less sensitive during acceleration
    cruising_multiplier: float = 1.2  # Slightly less sensitive at constant speed
    decelerating_multiplier: float = 1.0  # Normal sensitivity
    fine_motion_multiplier: float = 0.7  # More sensitive for precision work
    gripper_multiplier: float = 1.3  # Less sensitive during gripper operation

    # Threshold transition smoothing
    enable_smoothing: bool = True
    smoothing_factor: float = 0.3  # 0-1, lower = smoother transitions

    # Joint-specific adjustments
    enable_joint_specific_adaptation: bool = True
    proximal_joint_multiplier: float = 1.2  # Large joints (near base)
    distal_joint_multiplier: float = 0.8  # Small joints (near end-effector)

    # Load compensation
    enable_load_compensation: bool = True
    gravity_compensation_factor: float = 0.3
    inertial_compensation_factor: float = 0.5

    # Minimum and maximum threshold bounds
    min_threshold_ratio: float = 0.3  # Minimum 30% of base threshold
    max_threshold_ratio: float = 3.0  # Maximum 300% of base threshold


class AdaptiveCollisionDetector(CollisionDetector):
    """Collision detector with adaptive threshold adjustment.

    This detector dynamically adjusts collision thresholds based on:
    1. Current motion phase (static, accelerating, cruising, decelerating)
    2. Joint position (proximal vs distal)
    3. Load conditions (gravity, inertia)
    4. Gripper state

    This significantly reduces false positives during normal motion
    while maintaining high sensitivity for actual collisions.

    Usage:
        detector = AdaptiveCollisionDetector(config)
        detector.calibrate_base_torques(robot)
        # In control loop:
        result = detector.check_collision(observation, action)
        if result.is_detected:
            handle_collision(result)
    """

    def __init__(self, config: AdaptiveThresholdConfig):
        super().__init__(config)
        self.config: AdaptiveThresholdConfig = config

        # Motion tracking
        self._previous_positions: dict[str, float] = {}
        self._previous_velocities: dict[str, float] = {}
        self._previous_accelerations: dict[str, float] = {}
        self._current_velocities: dict[str, float] = {}
        self._current_accelerations: dict[str, float] = {}

        # State tracking
        self._motion_phase: MotionPhase = MotionPhase.STATIC
        self._previous_phase: MotionPhase = MotionPhase.STATIC
        self._phase_start_time: float = time.time()

        # Current thresholds (smoothed)
        self._current_thresholds: dict[str, float] = {}

        # Gripper state tracking
        self._gripper_active: bool = False
        self._gripper_forces: deque = deque(maxlen=10)

        # Statistics
        self._phase_counts = {phase: 0 for phase in MotionPhase}
        self._threshold_adjustments: deque = deque(maxlen=100)

        # Timestamp tracking
        self._last_timestamp: float | None = None
        self._dt: float = 0.02  # Initial estimate

    def check_collision(
        self, observation: dict[str, Any], action: dict[str, Any] | None = None
    ) -> CollisionResult:
        """Enhanced collision check with adaptive thresholds.

        Args:
            observation: Current observation dict.
            action: Current action dict.

        Returns:
            CollisionResult with detection status and adaptive threshold info.
        """
        self._total_checks += 1
        timestamp = time.time()

        # Update timing
        if self._last_timestamp is not None:
            self._dt = timestamp - self._last_timestamp
        self._last_timestamp = timestamp

        # Extract current data
        current_forces = {k: v for k, v in observation.items() if ".force" in k}
        current_positions = {k: v for k, v in observation.items() if ".pos" in k}

        # Estimate motion state
        self._update_motion_state(current_positions)

        # Detect gripper activity
        self._update_gripper_state(observation)

        # Determine motion phase
        self._motion_phase = self._determine_motion_phase()

        # Update phase statistics
        self._phase_counts[self._motion_phase] += 1

        # Calculate adaptive thresholds
        adaptive_thresholds = self._calculate_adaptive_thresholds(current_forces)

        # Store thresholds for analysis
        for joint_name, threshold in adaptive_thresholds.items():
            self._current_thresholds[joint_name] = threshold

        # Initialize result
        result = CollisionResult(timestamp=timestamp, raw_torques=current_forces)

        if not self.is_calibrated:
            logger.warning("Collision detector not calibrated. Skipping detection.")
            return result

        # Check each joint with adaptive threshold
        anomalies = {}
        for joint_force_key, torque_value in current_forces.items():
            joint_name = joint_force_key.replace(".force", "")

            # Get adaptive threshold for this joint
            threshold = adaptive_thresholds.get(joint_name, self.config.collision_threshold)

            # Get base torque
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Compute anomaly
            anomaly = abs(torque_value - base_torque)

            # Check against adaptive threshold
            if anomaly > threshold:
                anomalies[joint_name] = anomaly

            # Check absolute limit (use base threshold multiplier for safety)
            abs_limit = self.config.max_torque_limit
            if abs(torque_value) > abs_limit:
                anomalies[joint_name] = float("inf")

        result.affected_joints = anomalies

        # Determine detection
        if anomalies:
            result.is_detected = True
            result.severity = self._determine_severity(anomalies)
            self._collision_count += 1

            # Log adaptive detection details
            self._log_adaptive_detection(anomalies, adaptive_thresholds)

        # Store previous values
        self._previous_positions = current_positions.copy()
        self._previous_phase = self._motion_phase

        return result

    def _update_motion_state(self, current_positions: dict[str, float]):
        """Update velocity and acceleration estimates."""
        if not self._previous_positions:
            return

        for pos_key, current_pos in current_positions.items():
            joint_name = pos_key.replace(".pos", "")

            if pos_key in self._previous_positions or joint_name in self._previous_positions:
                prev_pos = self._previous_positions.get(joint_name, current_pos)

                # Calculate velocity
                velocity = (current_pos - prev_pos) / self._dt if self._dt > 0 else 0.0
                self._current_velocities[joint_name] = velocity

                # Calculate acceleration
                if joint_name in self._previous_velocities:
                    prev_velocity = self._previous_velocities[joint_name]
                    acceleration = (velocity - prev_velocity) / self._dt if self._dt > 0 else 0.0
                    self._current_accelerations[joint_name] = acceleration

        # Store current as previous for next iteration
        self._previous_velocities = self._current_velocities.copy()
        self._previous_accelerations = self._current_accelerations.copy()

    def _update_gripper_state(self, observation: dict[str, Any]):
        """Detect if gripper is active."""
        gripper_forces = []

        # Check gripper joints (typically _joint_7)
        for key, value in observation.items():
            if ".force" in key and ("joint_7" in key or "gripper" in key.lower()):
                gripper_forces.append(abs(float(value)))

        if gripper_forces:
            self._gripper_forces.append(sum(gripper_forces) / len(gripper_forces))

            # Gripper is active if force is elevated
            if len(self._gripper_forces) >= 3:
                avg_gripper_force = sum(self._gripper_forces) / len(self._gripper_forces)
                self._gripper_active = avg_gripper_force > 0.2  # Threshold for gripper activity

    def _determine_motion_phase(self) -> MotionPhase:
        """Determine current motion phase based on velocity and acceleration."""
        if not self._current_velocities:
            return MotionPhase.STATIC

        # Calculate aggregate motion metrics
        velocities = list(self._current_velocities.values())
        accelerations = list(self._current_accelerations.values())

        if not velocities:
            return MotionPhase.STATIC

        avg_velocity = np.mean(np.abs(velocities))
        max_velocity = np.max(np.abs(velocities))

        avg_acceleration = np.mean(np.abs(accelerations)) if accelerations else 0.0

        # Determine phase
        if max_velocity < self.config.velocity_threshold:
            return MotionPhase.STATIC
        elif avg_velocity < self.config.fine_motion_threshold:
            return MotionPhase.FINE_MOTION
        elif avg_acceleration > self.config.acceleration_threshold:
            # Determine if accelerating or decelerating
            if self._is_accelerating():
                return MotionPhase.ACCELERATING
            else:
                return MotionPhase.DECELERATING
        else:
            return MotionPhase.CRUISING

    def _is_accelerating(self) -> bool:
        """Check if robot is generally accelerating or decelerating."""
        if not self._current_accelerations:
            return False

        # Check if most joints are accelerating in same direction as velocity
        accelerating_count = 0
        decelerating_count = 0

        for joint_name, accel in self._current_accelerations.items():
            if joint_name in self._current_velocities:
                vel = self._current_velocities[joint_name]
                # Same sign = accelerating, opposite sign = decelerating
                if vel * accel > 0:
                    accelerating_count += 1
                else:
                    decelerating_count += 1

        return accelerating_count > decelerating_count

    def _calculate_adaptive_thresholds(self, current_forces: dict[str, float]) -> dict[str, float]:
        """Calculate adaptive thresholds for each joint.

        Args:
            current_forces: Current force readings.

        Returns:
            Dictionary of joint names to adaptive thresholds.
        """
        adaptive_thresholds = {}

        # Get phase multiplier
        phase_multiplier = self._get_phase_multiplier()

        for force_key in current_forces.keys():
            joint_name = force_key.replace(".force", "")

            # Start with base threshold
            base_threshold = self.config.joint_specific_thresholds.get(
                joint_name, self.config.collision_threshold
            )

            # Apply phase multiplier
            adaptive_threshold = base_threshold * phase_multiplier

            # Apply joint-specific adjustment
            if self.config.enable_joint_specific_adaptation:
                joint_multiplier = self._get_joint_multiplier(joint_name)
                adaptive_threshold *= joint_multiplier

            # Apply load compensation
            if self.config.enable_load_compensation:
                load_adjustment = self._calculate_load_adjustment(joint_name)
                adaptive_threshold += load_adjustment

            # Apply gripper adjustment
            if self._gripper_active and "joint_7" not in joint_name.lower():
                # Other joints can have higher threshold when gripper is active
                adaptive_threshold *= self.config.gripper_multiplier

            # Apply bounds
            min_threshold = base_threshold * self.config.min_threshold_ratio
            max_threshold = base_threshold * self.config.max_threshold_ratio
            adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)

            # Apply smoothing if enabled and we have previous threshold
            if self.config.enable_smoothing and joint_name in self._current_thresholds:
                prev_threshold = self._current_thresholds[joint_name]
                smooth_factor = self.config.smoothing_factor
                adaptive_threshold = (
                    smooth_factor * adaptive_threshold +
                    (1 - smooth_factor) * prev_threshold
                )

            adaptive_thresholds[joint_name] = adaptive_threshold

        # Track threshold adjustments
        if adaptive_thresholds:
            avg_multiplier = np.mean([
                t / self.config.collision_threshold
                for t in adaptive_thresholds.values()
            ])
            self._threshold_adjustments.append(avg_multiplier)

        return adaptive_thresholds

    def _get_phase_multiplier(self) -> float:
        """Get threshold multiplier for current motion phase."""
        if self._gripper_active and self._motion_phase in [
            MotionPhase.STATIC, MotionPhase.FINE_MOTION
        ]:
            # During gripper operation in static/fine motion, be more perceptive
            return min(
                self.config.static_multiplier,
                self.config.fine_motion_multiplier
            )

        multipliers = {
            MotionPhase.STATIC: self.config.static_multiplier,
            MotionPhase.ACCELERATING: self.config.accelerating_multiplier,
            MotionPhase.CRUISING: self.config.cruising_multiplier,
            MotionPhase.DECELERATING: self.config.decelerating_multiplier,
            MotionPhase.FINE_MOTION: self.config.fine_motion_multiplier,
            MotionPhase.GRIPPER_ACTIVE: self.config.gripper_multiplier,
        }

        return multipliers.get(self._motion_phase, 1.0)

    def _get_joint_multiplier(self, joint_name: str) -> float:
        """Get joint-specific threshold multiplier based on joint type."""
        # Proximal joints (near base) handle more load
        proximal_joints = ["joint_1", "joint_2", "trunk"]
        # Distal joints (near end-effector) are more sensitive
        distal_joints = ["joint_5", "joint_6", "joint_7"]

        joint_lower = joint_name.lower()

        if any(pj in joint_lower for pj in proximal_joints):
            return self.config.proximal_joint_multiplier
        elif any(dj in joint_lower for dj in distal_joints):
            return self.config.distal_joint_multiplier
        else:
            return 1.0

    def _calculate_load_adjustment(self, joint_name: str) -> float:
        """Calculate additional threshold based on current load.

        Args:
            joint_name: Name of the joint.

        Returns:
            Additional threshold to add (Nm).
        """
        adjustment = 0.0

        # Gravity compensation based on position
        if joint_name in self._previous_positions:
            # Simplified: joints supporting more weight (lower joints) get more allowance
            if "joint_1" in joint_name or "joint_2" in joint_name or "trunk" in joint_name.lower():
                adjustment += self.config.gravity_compensation_factor * 0.5

        # Inertial compensation based on acceleration
        if joint_name in self._current_accelerations:
            accel = self._current_accelerations[joint_name]
            adjustment += abs(accel) * self.config.inertial_compensation_factor

        return adjustment

    def _determine_severity(self, anomalies: dict[str, float]) -> str:
        """Determine collision severity considering adaptive context."""
        max_anomaly = max(
            v if v != float("inf") else 999
            for v in anomalies.values()
        )

        # Get average threshold multiplier for context
        if self._threshold_adjustments:
            avg_multiplier = np.mean(list(self._threshold_adjustments))
        else:
            avg_multiplier = 1.0

        # Adjust severity based on how lenient our thresholds were
        normalized_anomaly = max_anomaly / avg_multiplier

        if max_anomaly == float("inf") or normalized_anomaly > self.config.collision_threshold * 3:
            return "critical"
        elif normalized_anomaly > self.config.collision_threshold * 2:
            return "high"
        elif normalized_anomaly > self.config.collision_threshold * 1.5:
            return "medium"
        else:
            return "low"

    def _log_adaptive_detection(
        self,
        anomalies: dict[str, float],
        adaptive_thresholds: dict[str, float],
    ):
        """Log adaptive detection details."""
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning(f"🔴 ADAPTIVE COLLISION DETECTED - Phase: {self._motion_phase.value.upper()}")
        logger.warning("=" * 80)

        # Sort anomalies by severity
        sorted_anomalies = sorted(
            anomalies.items(),
            key=lambda x: abs(x[1]) if x[1] != float("inf") else 999,
            reverse=True
        )

        for joint_name, anomaly_value in sorted_anomalies[:5]:
            force_key = f"{joint_name}.force"

            # Get current force
            current_force = 0.0
            if hasattr(self, '_current_forces'):
                current_force = self._current_forces.get(force_key, 0.0)

            # Get base torque
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Get adaptive threshold
            adaptive_threshold = adaptive_thresholds.get(joint_name, self.config.collision_threshold)

            # Calculate multiplier
            base_threshold = self.config.joint_specific_thresholds.get(
                joint_name, self.config.collision_threshold
            )
            multiplier = adaptive_threshold / base_threshold if base_threshold > 0 else 1.0

            logger.warning(f"   📍 {joint_name}")
            logger.warning(f"      Anomaly:       {anomaly_value if anomaly_value != float('inf') else '∞':.3f} Nm")
            logger.warning(f"      Adaptive Th:   {adaptive_threshold:.3f} Nm (×{multiplier:.2f})")
            logger.warning(f"      Base Th:       {base_threshold:.3f} Nm")
            logger.warning(f"      Motion Phase:  {self._motion_phase.value}")

            # Show motion state for this joint
            if joint_name in self._current_velocities:
                logger.warning(f"      Velocity:      {self._current_velocities[joint_name]:.3f} rad/s")
            if joint_name in self._current_accelerations:
                logger.warning(f"      Acceleration:  {self._current_accelerations[joint_name]:.3f} rad/s²")

        logger.warning("=" * 80)
        logger.warning("")

    def reset(self):
        """Reset detector state."""
        super().reset()
        self._previous_positions.clear()
        self._previous_velocities.clear()
        self._previous_accelerations.clear()
        self._current_velocities.clear()
        self._current_accelerations.clear()
        self._current_thresholds.clear()
        self._gripper_forces.clear()
        self._motion_phase = MotionPhase.STATIC

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics."""
        base_stats = super().get_statistics()

        phase_stats = {
            f"phase_{phase.value}": count
            for phase, count in self._phase_counts.items()
        }

        threshold_stats = {}
        if self._threshold_adjustments:
            threshold_stats = {
                "avg_threshold_multiplier": float(np.mean(list(self._threshold_adjustments))),
                "min_threshold_multiplier": float(np.min(list(self._threshold_adjustments))),
                "max_threshold_multiplier": float(np.max(list(self._threshold_adjustments))),
            }

        return {
            **base_stats,
            **phase_stats,
            **threshold_stats,
            "current_phase": self._motion_phase.value,
            "gripper_active": self._gripper_active,
        }

    def get_current_thresholds(self) -> dict[str, float]:
        """Get current adaptive thresholds for all joints.

        Returns:
            Dictionary of joint names to current thresholds.
        """
        return self._current_thresholds.copy()

    def get_motion_phase(self) -> MotionPhase:
        """Get current motion phase."""
        return self._motion_phase


def create_adaptive_collision_config(
    collision_threshold: float = 0.8,
    **kwargs
) -> AdaptiveThresholdConfig:
    """Create an adaptive collision configuration.

    Args:
        collision_threshold: Base threshold for collision detection.
        **kwargs: Additional configuration parameters.

    Returns:
        AdaptiveThresholdConfig instance.
    """
    return AdaptiveThresholdConfig(
        collision_threshold=collision_threshold,
        enable_smoothing=True,
        smoothing_factor=0.3,
        enable_joint_specific_adaptation=True,
        enable_load_compensation=True,
        detection_window=1,  # Immediate response
        velocity_compensation=True,
        **kwargs
    )
