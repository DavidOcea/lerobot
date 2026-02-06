"""
Enhanced Collision Detection with Multiple Detection Strategies

This module provides advanced collision detection with:
1. Torque anomaly detection (baseline)
2. Force rate-of-change detection (sudden spikes)
3. Absolute torque limit checking
4. Surface proximity detection (based on force patterns)
5. Immediate detection mode for high-severity collisions
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.safety.collision_detector import (
    CollisionConfig,
    CollisionDetector,
    CollisionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCollisionConfig(CollisionConfig):
    """Enhanced collision configuration with additional detection modes."""

    # Force rate-of-change detection
    enable_rate_detection: bool = True
    force_rate_threshold: float = 2.0  # Nm per step
    force_rate_window: int = 1  # Immediate detection

    # Immediate detection for high torque
    immediate_threshold: float = 4.0  # Nm - trigger immediately
    immediate_absolute_limit: float = 6.0  # Nm - hard limit

    # Surface proximity detection
    enable_surface_detection: bool = True
    surface_force_threshold: float = 1.5  # Nm - indicates contact with surface
    surface_consecutive_frames: int = 2  # Frames to confirm surface contact

    # Multiple joint simultaneous detection
    multi_joint_threshold: int = 3  # Number of joints over threshold
    multi_joint_threshold_per_joint: float = 0.5  # Lower threshold for multi-joint

    # Detection strategy
    detection_mode: str = "immediate"  # "immediate" or "windowed"

    # Gripper-specific settings
    gripper_force_scale: float = 5.0  # Scale factor to convert gripper 0-1 force to Nm equivalent
    gripper_collision_threshold: float = 0.3  # Normalized threshold for gripper force
    gripper_rate_threshold: float = 0.2  # Rate threshold for gripper force (per step)


class EnhancedCollisionDetector(CollisionDetector):
    """Enhanced collision detector with multiple detection strategies.

    This detector uses:
    1. Baseline torque anomaly detection
    2. Force rate-of-change (sudden spikes)
    3. Absolute torque limits
    4. Multi-joint simultaneous detection
    5. Surface contact pattern detection
    """

    def __init__(self, config: EnhancedCollisionConfig):
        super().__init__(config)
        self.config: EnhancedCollisionConfig = config

        # Force rate tracking
        self._prev_forces: dict[str, float] = {}
        self._force_rates: dict[str, float] = {}

        # Surface contact tracking
        self._surface_contact_frames: int = 0

        # Statistics
        self._rate_detections = 0
        self._surface_detections = 0
        self._immediate_detections = 0

    def check_collision(
        self, observation: dict[str, Any], action: dict[str, Any] | None = None
    ) -> CollisionResult:
        """Enhanced collision check with multiple detection strategies.

        Args:
            observation: Current observation dict containing {joint}.force values.
            action: Current action dict containing {joint}.pos values.

        Returns:
            CollisionResult with detection status and details.
        """
        self._total_checks += 1
        timestamp = time.time()

        # Extract torque data with gripper force scaling
        current_torques = {}
        current_positions = {k: v for k, v in observation.items() if ".pos" in k}

        for key, value in observation.items():
            if ".force" in key:
                joint_name = key.replace(".force", "")
                # Check if this is a gripper joint (joint_7)
                if joint_name.endswith("_joint_7"):
                    # Scale gripper force to Nm equivalent
                    # Gripper force is 0-1 normalized, scale to Nm range
                    current_torques[key] = value * self.config.gripper_force_scale
                else:
                    current_torques[key] = value

        # Initialize result
        result = CollisionResult(timestamp=timestamp, raw_torques=current_torques)

        if not self.is_calibrated:
            logger.warning("Collision detector not calibrated. Skipping detection.")
            return result

        # Run all detection strategies
        anomalies = {}

        # Strategy 1: Baseline torque anomaly
        baseline_anomalies = self._check_baseline_anomalies(
            current_torques, current_positions, action, timestamp
        )
        anomalies.update(baseline_anomalies)

        # Strategy 2: Force rate-of-change detection (with gripper-specific handling)
        if self.config.enable_rate_detection:
            rate_anomalies = self._check_force_rate_changes(current_torques, observation)
            anomalies.update(rate_anomalies)

        # Strategy 3: Immediate high-torque detection
        immediate_anomalies = self._check_immediate_dangers(current_torques, observation)
        anomalies.update(immediate_anomalies)

        # Strategy 4: Multi-joint detection
        multi_joint_info = self._check_multi_joint_collision(anomalies)
        if multi_joint_info["detected"]:
            for joint_name in multi_joint_info["joints"]:
                anomalies[joint_name] = anomalies.get(joint_name, 0.5)

        # Strategy 5: Surface contact detection
        if self.config.enable_surface_detection:
            surface_detected = self._check_surface_contact(current_torques)
            if surface_detected:
                self._surface_contact_frames += 1
                if self._surface_contact_frames >= self.config.surface_consecutive_frames:
                    # Surface contact confirmed - slow down or stop
                    result.surface_contact = True
                    for joint_name in current_torques.keys():
                        if joint_name not in anomalies:
                            anomalies[joint_name] = 0.1  # Low severity but detected

        # Set affected joints and detection status
        result.affected_joints = anomalies

        # Determine detection mode
        if self.config.detection_mode == "immediate":
            # Trigger immediately if any anomaly detected
            if anomalies:
                result.is_detected = True
                result.detection_strategy = self._get_detection_strategy(anomalies)
        else:
            # Use window-based confirmation
            if anomalies:
                self.detection_buffer.append(anomalies)
                if len(self.detection_buffer) >= self.config.detection_window:
                    confirmed_anomalies = self._confirm_with_window()
                    if confirmed_anomalies:
                        result.is_detected = True
                        result.affected_joints = confirmed_anomalies
                        result.detection_strategy = self._get_detection_strategy(confirmed_anomalies)

        # Determine severity
        if result.is_detected:
            max_anomaly = max(
                v if v != float("inf") else 999
                for v in result.affected_joints.values()
            )
            if max_anomaly > self.config.immediate_threshold or max_anomaly == 999:
                result.severity = "critical"
            elif max_anomaly > self.config.collision_threshold * 2:
                result.severity = "high"
            elif max_anomaly > self.config.collision_threshold * 1.5:
                result.severity = "medium"
            else:
                result.severity = "low"

            # Log detailed force analysis for collision debugging
            self._log_collision_analysis(
                current_torques, observation, anomalies, result.severity, result.detection_strategy
            )

        # Optional: Periodic force status logging (every 100 checks) for monitoring
        if self._total_checks % 100 == 0 and self.is_calibrated:
            self._log_force_status(current_torques, observation)

        # Update previous forces for rate detection (store original values)
        for key, value in observation.items():
            if ".force" in key:
                self._prev_forces[key] = value

        return result

    def _check_baseline_anomalies(
        self,
        current_torques: dict[str, float],
        current_positions: dict[str, float],
        action: dict[str, Any] | None,
        timestamp: float,
    ) -> dict[str, float]:
        """Check baseline torque anomalies using parent class logic."""
        anomalies = {}

        # Estimate and subtract inertial torques
        estimated_inertial = {}
        if self.config.velocity_compensation and action is not None:
            estimated_inertial = self._estimate_inertial_torques(
                current_positions, action, timestamp
            )

        # Check each joint
        for joint_force_key, torque_value in current_torques.items():
            joint_name = joint_force_key.replace(".force", "")

            # Get base torque
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Get inertial compensation
            inertial_comp = estimated_inertial.get(joint_name, 0.0)

            # Compute anomaly
            anomaly = abs(torque_value - base_torque)
            if self.config.velocity_compensation:
                anomaly = max(0, anomaly - inertial_comp)

            # Get threshold for this joint
            threshold = self.config.joint_specific_thresholds.get(
                joint_name, self.config.collision_threshold
            )

            if anomaly > threshold:
                anomalies[joint_name] = anomaly

            # Check absolute limit
            if abs(torque_value) > self.config.max_torque_limit:
                anomalies[joint_name] = float("inf")

        return anomalies

    def _check_force_rate_changes(
        self, current_torques: dict[str, float], raw_observation: dict[str, Any]
    ) -> dict[str, float]:
        """Check for sudden force changes (rate-of-change detection).

        Args:
            current_torques: Current torque values (scaled for grippers)
            raw_observation: Raw observation with original force values

        Returns:
            Dictionary of joints with excessive force rate changes
        """
        anomalies = {}

        for key, value in raw_observation.items():
            if ".force" in key:
                joint_name = key.replace(".force", "")

                # Need previous force to compute rate
                prev_key = f"{joint_name}.force"
                if prev_key not in self._prev_forces:
                    continue

                prev_force = self._prev_forces[prev_key]
                force_delta = abs(value - prev_force)

                # Check if this is a gripper joint
                if joint_name.endswith("_joint_7"):
                    # Use gripper-specific rate threshold
                    if force_delta > self.config.gripper_rate_threshold:
                        # Scale to Nm equivalent for anomaly reporting
                        scaled_delta = force_delta * self.config.gripper_force_scale
                        anomalies[joint_name] = scaled_delta
                        self._rate_detections += 1

                        if force_delta > self.config.gripper_rate_threshold * 3:
                            # Very high rate - critical
                            anomalies[joint_name] = float("inf")
                            self._immediate_detections += 1
                else:
                    # Use arm motor rate threshold
                    if force_delta > self.config.force_rate_threshold:
                        anomalies[joint_name] = force_delta
                        self._rate_detections += 1

                        if force_delta > self.config.force_rate_threshold * 3:
                            # Very high rate - critical
                            anomalies[joint_name] = float("inf")
                            self._immediate_detections += 1

        return anomalies

    def _check_immediate_dangers(
        self, current_torques: dict[str, float], raw_observation: dict[str, Any]
    ) -> dict[str, float]:
        """Check for immediate dangerous conditions.

        Args:
            current_torques: Current torque values (scaled for grippers)
            raw_observation: Raw observation with original force values

        Returns:
            Dictionary of joints with dangerous conditions
        """
        anomalies = {}

        for key, value in raw_observation.items():
            if ".force" in key:
                joint_name = key.replace(".force", "")

                # Check if this is a gripper joint
                if joint_name.endswith("_joint_7"):
                    # Check gripper absolute force against threshold
                    if value > self.config.gripper_collision_threshold:
                        # Scale to Nm equivalent
                        scaled_value = value * self.config.gripper_force_scale
                        anomalies[joint_name] = scaled_value

                        # Very high gripper force - critical
                        if value > self.config.gripper_collision_threshold * 2:
                            anomalies[joint_name] = float("inf")
                else:
                    # Check arm motor absolute torque
                    if abs(value) > self.config.immediate_absolute_limit:
                        anomalies[joint_name] = float("inf")

                    # Check against immediate threshold
                    if abs(value) > self.config.immediate_threshold:
                        anomalies[joint_name] = abs(value)

        return anomalies

    def _check_multi_joint_collision(self, anomalies: dict[str, float]) -> dict:
        """Check if multiple joints are experiencing high forces simultaneously.

        Args:
            anomalies: Current anomaly readings per joint

        Returns:
            Dictionary with detection status and affected joints
        """
        joints_over_threshold = []

        for joint_name, anomaly in anomalies.items():
            if anomaly > self.config.multi_joint_threshold_per_joint:
                joints_over_threshold.append(joint_name)

        result = {
            "detected": len(joints_over_threshold) >= self.config.multi_joint_threshold,
            "joints": joints_over_threshold,
            "count": len(joints_over_threshold),
        }

        return result

    def _check_surface_contact(self, current_torques: dict[str, float]) -> bool:
        """Detect if robot is in contact with a surface.

        Surface contact is indicated by:
        - Multiple joints showing consistent elevated force
        - Forces are relatively stable (not spiking)

        Args:
            current_torques: Current torque values

        Returns:
            True if surface contact is detected
        """
        elevated_count = 0
        total_force = 0

        for joint_force_key, torque_value in current_torques.items():
            joint_name = joint_force_key.replace(".force", "")

            # Get base torque
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Check if force is elevated but not spiking
            anomaly = abs(torque_value - base_torque)
            if (
                self.config.surface_force_threshold * 0.5
                < anomaly
                < self.config.surface_force_threshold * 2
            ):
                elevated_count += 1
                total_force += anomaly

        # Surface contact requires multiple joints with elevated forces
        return elevated_count >= 4

    def _confirm_with_window(self) -> dict[str, float] | None:
        """Confirm collision using sliding window."""
        # Count detections per joint
        joint_counts: dict[str, int] = {}
        joint_max_anomalies: dict[str, float] = {}

        for detection in self.detection_buffer:
            for joint_name, anomaly in detection.items():
                joint_counts[joint_name] = joint_counts.get(joint_name, 0) + 1
                joint_max_anomalies[joint_name] = max(
                    joint_max_anomalies.get(joint_name, 0), anomaly
                )

        # Confirm joints that appear in enough windows
        confirmed = {}
        for joint_name, count in joint_counts.items():
            if count >= self.config.detection_window:
                confirmed[joint_name] = joint_max_anomalies[joint_name]

        return confirmed if confirmed else None

    def _get_detection_strategy(self, anomalies: dict[str, float]) -> str:
        """Identify which detection strategy triggered the collision."""
        has_inf = any(v == float("inf") for v in anomalies.values())

        if has_inf:
            return "absolute_limit"
        elif self._rate_detections > 0:
            return "force_rate"
        elif self._surface_contact_frames >= self.config.surface_consecutive_frames:
            return "surface_contact"
        else:
            return "baseline_anomaly"

    def _log_collision_analysis(
        self,
        current_torques: dict[str, float],
        raw_observation: dict[str, Any],
        anomalies: dict[str, float],
        severity: str,
        strategy: str,
    ):
        """Log detailed force analysis when collision is detected.

        Shows raw force, base torque, anomaly, and threshold for each affected joint.
        """
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning(f"🔴 COLLISION DETECTED - {severity.upper()} SEVERITY")
        logger.warning(f"   Strategy: {strategy}")
        logger.warning("=" * 80)

        # Sort anomalies by value (descending)
        sorted_anomalies = sorted(anomalies.items(), key=lambda x: abs(x[1]) if x[1] != float("inf") else 999, reverse=True)

        for joint_force_key, anomaly_value in sorted_anomalies[:10]:  # Show top 10
            joint_name = joint_force_key.replace(".force", "")

            # Get raw force (from original observation, not scaled)
            raw_force = raw_observation.get(joint_force_key, 0.0)

            # Get base torque
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Get threshold
            threshold = self.config.joint_specific_thresholds.get(
                joint_name, self.config.collision_threshold
            )

            # Calculate delta (actual change from baseline)
            delta = raw_force - base_torque

            # Format output
            anomaly_display = "∞" if anomaly_value == float("inf") else f"{anomaly_value:.3f}"

            logger.warning(f"   📍 {joint_name}")
            logger.warning(f"      Raw Force:    {raw_force:+7.3f} Nm")
            logger.warning(f"      Base Torque:  {base_torque:+7.3f} Nm")
            logger.warning(f"      Delta (Δ):    {delta:+7.3f} Nm")
            logger.warning(f"      Anomaly:      {anomaly_display:>7} Nm")
            logger.warning(f"      Threshold:    {threshold:.3f} Nm")

            # Show rate if available
            if joint_force_key in self._prev_forces:
                force_rate = abs(raw_force - self._prev_forces[joint_force_key])
                logger.warning(f"      Rate (Δ/step): {force_rate:.3f} Nm")

        logger.warning("=" * 80)
        logger.warning("")

    def _log_force_status(
        self,
        current_torques: dict[str, float],
        raw_observation: dict[str, Any],
    ):
        """Log periodic force status for monitoring (non-collision state).

        Shows joints with highest force deviations from baseline.
        """
        deviations = []

        for joint_force_key, _ in current_torques.items():
            joint_name = joint_force_key.replace(".force", "")

            # Get raw force
            raw_force = raw_observation.get(joint_force_key, 0.0)

            # Get base torque
            base_torque = self.base_torques.get(joint_name, 0.0)

            # Calculate deviation
            deviation = abs(raw_force - base_torque)

            # Get threshold
            threshold = self.config.joint_specific_thresholds.get(
                joint_name, self.config.collision_threshold
            )

            # Calculate percentage of threshold
            threshold_pct = (deviation / threshold * 100) if threshold > 0 else 0

            deviations.append({
                "joint": joint_name,
                "raw": raw_force,
                "base": base_torque,
                "deviation": deviation,
                "threshold": threshold,
                "threshold_pct": threshold_pct,
            })

        # Sort by deviation (descending) and show top 5
        deviations.sort(key=lambda x: x["deviation"], reverse=True)

        logger.debug("")
        logger.debug("─" * 60)
        logger.debug(f"Force Status Check #{self._total_checks}")
        logger.debug("─" * 60)

        for d in deviations[:5]:
            if d["threshold_pct"] > 20:  # Only show if > 20% of threshold
                logger.debug(
                    f"   {d['joint']:25} | "
                    f"Raw: {d['raw']:+6.2f} | "
                    f"Base: {d['base']:+6.2f} | "
                    f"Δ: {d['deviation']:.2f} Nm ({d['threshold_pct']:.0f}% of threshold)"
                )

        logger.debug("─" * 60)

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about collision detection."""
        return {
            "total_checks": self._total_checks,
            "total_collisions": self._collision_count,
            "rate_detections": self._rate_detections,
            "surface_detections": self._surface_detections,
            "immediate_detections": self._immediate_detections,
            "detection_buffer_size": len(self.detection_buffer),
            "calibrated": self.is_calibrated,
            "base_torques": self.base_torques if self.is_calibrated else {},
        }


def create_enhanced_collision_config(
    collision_threshold: float = 0.8,  # More sensitive
    detection_window: int = 1,  # Immediate response
    **kwargs
) -> EnhancedCollisionConfig:
    """Create an enhanced collision configuration optimized for safety.

    Args:
        collision_threshold: Base threshold for torque anomaly (Nm)
        detection_window: Number of consecutive detections required
        **kwargs: Additional configuration parameters

    Returns:
        EnhancedCollisionConfig instance
    """
    return EnhancedCollisionConfig(
        collision_threshold=collision_threshold,
        detection_window=detection_window,
        enable_rate_detection=True,
        force_rate_threshold=1.5,
        force_rate_window=1,
        immediate_threshold=3.0,
        immediate_absolute_limit=5.0,
        enable_surface_detection=True,
        surface_force_threshold=1.2,
        surface_consecutive_frames=3,
        multi_joint_threshold=3,
        multi_joint_threshold_per_joint=0.4,
        detection_mode="immediate",
        adaptive_mode=True,
        velocity_compensation=True,
        # Gripper-specific settings
        gripper_force_scale=5.0,  # Convert 0-1 gripper force to Nm equivalent
        gripper_collision_threshold=0.3,  # Normalized threshold for gripper
        gripper_rate_threshold=0.15,  # Rate threshold for gripper (more sensitive)
        **kwargs,
    )
