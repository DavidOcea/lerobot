"""
Temporal Collision Detector with Multi-Timeframe Analysis

This module provides advanced collision detection using temporal patterns
and multi-timeframe analysis for improved sensitivity and reduced false alarms.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.safety.collision_detector import CollisionConfig, CollisionDetector, CollisionResult

logger = logging.getLogger(__name__)


@dataclass
class TemporalCollisionConfig(CollisionConfig):
    """Configuration for temporal collision detection."""

    # Inherit from CollisionConfig and add temporal-specific parameters
    collision_threshold: float = 0.8

    # Temporal window settings
    temporal_window_size: int = 10  # Number of samples to analyze
    accumulated_threshold: float = 0.4  # Lower threshold for accumulated detection

    # Force gradient detection
    enable_gradient_detection: bool = True
    gradient_threshold: float = 0.3  # Nm per step (sudden contact)
    gradient_window: int = 2  # Steps to check for gradient

    # Oscillation detection (collision bounce-back pattern)
    enable_oscillation_detection: bool = True
    oscillation_threshold: float = 0.5  # Nm
    oscillation_min_cycles: int = 2
    oscillation_window: int = 8

    # Persistent deviation detection (continuous contact)
    enable_persistent_detection: bool = True
    persistent_threshold: float = 0.3  # Nm
    persistent_min_duration: int = 5  # Minimum consecutive steps

    # Trend analysis
    enable_trend_detection: bool = True
    trend_threshold: float = 0.1  # Nm per step (consistent increase)
    trend_window: int = 5

    # Multi-joint correlation
    enable_correlation_detection: bool = True
    correlation_threshold: float = 0.7  # Correlation coefficient
    min_simultaneous_joints: int = 3


class TemporalCollisionDetector(CollisionDetector):
    """Enhanced collision detector with temporal pattern analysis.

    This detector extends the base CollisionDetector with:
    1. Accumulated deviation detection (small but persistent forces)
    2. Force gradient spike detection (sudden contact events)
    3. Oscillation detection (collision bounce-back patterns)
    4. Persistent deviation detection (continuous contact)
    5. Trend analysis (gradual force increase)
    6. Multi-joint correlation detection

    Usage:
        detector = TemporalCollisionDetector(config)
        detector.calibrate_base_torques(robot)
        # In control loop:
        result = detector.check_collision(observation, action)
        if result.is_detected:
            handle_collision(result)
    """

    def __init__(self, config: TemporalCollisionConfig):
        super().__init__(config)
        self.config: TemporalCollisionConfig = config

        # Temporal data storage
        self._force_history: deque[dict[str, float]] = deque(maxlen=config.temporal_window_size)
        self._position_history: deque[dict[str, float]] = deque(maxlen=config.temporal_window_size)
        self._anomaly_history: deque[dict[str, float]] = deque(maxlen=config.temporal_window_size)

        # Detection states
        self._persistent_counter: dict[str, int] = {}  # Track persistent deviations per joint
        self._oscillation_detector = OscillationDetector(config.oscillation_window)
        self._trend_analyzer = TrendAnalyzer(config.trend_window)

        # Statistics
        self._temporal_detections = 0
        self._gradient_detections = 0
        self._oscillation_detections = 0
        self._persistent_detections = 0
        self._trend_detections = 0
        self._correlation_detections = 0

    def check_collision(
        self, observation: dict[str, Any], action: dict[str, Any] | None = None
    ) -> CollisionResult:
        """Enhanced collision check with temporal pattern analysis.

        Args:
            observation: Current observation dict.
            action: Current action dict.

        Returns:
            CollisionResult with detection status and temporal analysis details.
        """
        self._total_checks += 1
        timestamp = time.time()

        # Extract current data
        current_forces = {k: v for k, v in observation.items() if ".force" in k}
        current_positions = {k: v for k, v in observation.items() if ".pos" in k}

        # Update history
        self._force_history.append(current_forces.copy())
        self._position_history.append(current_positions.copy())

        # Initialize result
        result = CollisionResult(timestamp=timestamp, raw_torques=current_forces)

        if not self.is_calibrated:
            logger.warning("Collision detector not calibrated. Skipping detection.")
            return result

        # Run base collision detection
        base_result = super().check_collision(observation, action)

        # Run temporal detection strategies
        temporal_anomalies = {}

        # Strategy 1: Accumulated deviation detection
        if len(self._force_history) >= self.config.temporal_window_size:
            accumulated = self._check_accumulated_deviation()
            temporal_anomalies.update(accumulated)

        # Strategy 2: Force gradient detection
        if self.config.enable_gradient_detection and len(self._force_history) >= 2:
            gradient_anomalies = self._check_force_gradient()
            temporal_anomalies.update(gradient_anomalies)

        # Strategy 3: Oscillation detection
        if self.config.enable_oscillation_detection and len(self._force_history) >= self.config.oscillation_window:
            oscillation_anomalies = self._check_oscillation()
            temporal_anomalies.update(oscillation_anomalies)

        # Strategy 4: Persistent deviation detection
        if self.config.enable_persistent_detection:
            persistent_anomalies = self._check_persistent_deviation(current_forces)
            temporal_anomalies.update(persistent_anomalies)

        # Strategy 5: Trend detection
        if self.config.enable_trend_detection and len(self._force_history) >= self.config.trend_window:
            trend_anomalies = self._check_trend()
            temporal_anomalies.update(trend_anomalies)

        # Strategy 6: Multi-joint correlation detection
        if self.config.enable_correlation_detection and len(self._force_history) >= 3:
            correlation_anomalies = self._check_multi_joint_correlation()
            temporal_anomalies.update(correlation_anomalies)

        # Combine with base detection
        all_anomalies = base_result.affected_joints.copy()
        all_anomalies.update(temporal_anomalies)

        result.affected_joints = all_anomalies

        # Determine detection
        if all_anomalies:
            result.is_detected = True

            # Determine severity with temporal context
            result.severity = self._determine_severity_with_temporal(
                all_anomalies, temporal_anomalies
            )

            # Log temporal analysis details
            self._log_temporal_analysis(temporal_anomalies, result.severity)

            self._collision_count += 1

        return result

    def _check_accumulated_deviation(self) -> dict[str, float]:
        """Check for accumulated small deviations over time.

        Small forces that persist over time can indicate contact even if
        individual readings don't exceed threshold.
        """
        anomalies = {}

        if len(self._force_history) < self.config.temporal_window_size:
            return anomalies

        # Get joint names
        joint_names = set()
        for forces in self._force_history:
            for key in forces.keys():
                joint_name = key.replace(".force", "")
                joint_names.add(joint_name)

        for joint_name in joint_names:
            force_key = f"{joint_name}.force"

            # Calculate accumulated deviation
            accumulated_deviation = 0.0
            valid_samples = 0

            for forces in self._force_history:
                if force_key in forces:
                    force = forces[force_key]
                    base = self.base_torques.get(joint_name, 0.0)
                    deviation = abs(force - base)
                    accumulated_deviation += deviation
                    valid_samples += 1

            if valid_samples < self.config.temporal_window_size:
                continue

            avg_deviation = accumulated_deviation / valid_samples

            # Check against accumulated threshold
            if avg_deviation > self.config.accumulated_threshold:
                anomalies[joint_name] = avg_deviation
                self._temporal_detections += 1

        return anomalies

    def _check_force_gradient(self) -> dict[str, float]:
        """Check for sudden force changes (gradients).

        High gradient indicates sudden contact or impact.
        """
        anomalies = {}

        if len(self._force_history) < self.config.gradient_window + 1:
            return anomalies

        # Get recent forces
        recent_forces = list(self._force_history)[-self.config.gradient_window - 1:]

        for joint_force_key in recent_forces[0].keys():
            joint_name = joint_force_key.replace(".force", "")

            # Calculate maximum gradient over the window
            max_gradient = 0.0
            for i in range(len(recent_forces) - 1):
                if joint_force_key in recent_forces[i] and joint_force_key in recent_forces[i + 1]:
                    force1 = recent_forces[i][joint_force_key]
                    force2 = recent_forces[i + 1][joint_force_key]
                    gradient = abs(force2 - force1)
                    max_gradient = max(max_gradient, gradient)

            if max_gradient > self.config.gradient_threshold:
                anomalies[joint_name] = max_gradient
                self._gradient_detections += 1

        return anomalies

    def _check_oscillation(self) -> dict[str, float]:
        """Check for oscillation patterns (collision bounce-back).

        Oscillation indicates the robot is bouncing off an obstacle.
        """
        anomalies = {}

        if len(self._force_history) < self.config.oscillation_window:
            return anomalies

        # Get joint names
        joint_names = set()
        for forces in self._force_history:
            for key in forces.keys():
                joint_names.add(key.replace(".force", ""))

        for joint_name in joint_names:
            force_key = f"{joint_name}.force"

            # Extract force history for this joint
            force_values = []
            for forces in list(self._force_history)[-self.config.oscillation_window:]:
                if force_key in forces:
                    force_values.append(forces[force_key])

            if len(force_values) < self.config.oscillation_window:
                continue

            # Detect oscillations
            oscillation_info = self._oscillation_detector.detect_oscillation(
                np.array(force_values)
            )

            if oscillation_info["is_oscillating"]:
                anomalies[joint_name] = oscillation_info["amplitude"]
                self._oscillation_detections += 1

        return anomalies

    def _check_persistent_deviation(self, current_forces: dict[str, float]) -> dict[str, float]:
        """Check for persistent deviations (continuous contact).

        Forces that consistently stay above a lower threshold.
        """
        anomalies = {}

        for force_key, force_value in current_forces.items():
            joint_name = force_key.replace(".force", "")

            base = self.base_torques.get(joint_name, 0.0)
            deviation = abs(force_value - base)

            if deviation > self.config.persistent_threshold:
                # Increment counter
                self._persistent_counter[joint_name] = (
                    self._persistent_counter.get(joint_name, 0) + 1
                )

                # Check if persistent for long enough
                if self._persistent_counter[joint_name] >= self.config.persistent_min_duration:
                    anomalies[joint_name] = deviation
                    self._persistent_detections += 1
            else:
                # Reset counter
                self._persistent_counter[joint_name] = 0

        return anomalies

    def _check_trend(self) -> dict[str, float]:
        """Check for consistent increasing force trends.

        Gradual force increase can indicate approaching obstacle.
        """
        anomalies = {}

        if len(self._force_history) < self.config.trend_window:
            return anomalies

        # Get joint names
        joint_names = set()
        for forces in self._force_history:
            for key in forces.keys():
                joint_names.add(key.replace(".force", ""))

        for joint_name in joint_names:
            force_key = f"{joint_name}.force"

            # Extract force history
            force_values = []
            for forces in list(self._force_history)[-self.config.trend_window:]:
                if force_key in forces:
                    force_values.append(forces[force_key])

            if len(force_values) < self.config.trend_window:
                continue

            # Analyze trend
            trend_info = self._trend_analyzer.analyze_trend(np.array(force_values))

            if abs(trend_info["slope"]) > self.config.trend_threshold:
                # Force is consistently increasing
                anomalies[joint_name] = abs(trend_info["total_change"])
                self._trend_detections += 1

        return anomalies

    def _check_multi_joint_correlation(self) -> dict[str, float]:
        """Check for correlated force changes across multiple joints.

        Simultaneous force changes in multiple joints indicate collision.
        """
        anomalies = {}

        if len(self._force_history) < 3:
            return anomalies

        # Get all joint names
        joint_names = set()
        for forces in self._force_history:
            for key in forces.keys():
                joint_names.add(key.replace(".force", ""))

        # Calculate correlation matrix for recent forces
        force_matrix = []
        valid_joints = []

        for joint_name in joint_names:
            force_key = f"{joint_name}.force"
            force_values = []
            for forces in list(self._force_history):
                if force_key in forces:
                    force_values.append(forces[force_key])

            if len(force_values) >= len(self._force_history) // 2:
                force_matrix.append(force_values)
                valid_joints.append(joint_name)

        if len(valid_joints) < self.config.min_simultaneous_joints:
            return anomalies

        # Calculate correlations
        force_array = np.array(force_matrix)
        if force_array.shape[1] < 2:
            return anomalies

        correlations = np.corrcoef(force_array)

        # Find joints with high correlation
        highly_correlated = set()
        for i in range(len(valid_joints)):
            for j in range(i + 1, len(valid_joints)):
                if not np.isnan(correlations[i, j]) and abs(correlations[i, j]) > self.config.correlation_threshold:
                    highly_correlated.add(valid_joints[i])
                    highly_correlated.add(valid_joints[j])

        # If enough joints are correlated, flag all of them
        if len(highly_correlated) >= self.config.min_simultaneous_joints:
            for joint_name in highly_correlated:
                force_key = f"{joint_name}.force"
                current_force = self._force_history[-1].get(force_key, 0.0)
                base = self.base_torques.get(joint_name, 0.0)
                anomalies[joint_name] = abs(current_force - base)
            self._correlation_detections += 1

        return anomalies

    def _determine_severity_with_temporal(
        self,
        anomalies: dict[str, float],
        temporal_anomalies: dict[str, float],
    ) -> str:
        """Determine severity considering temporal context."""
        max_anomaly = max(
            v if v != float("inf") else 999
            for v in anomalies.values()
        )

        # Base severity on anomaly magnitude
        if max_anomaly == 999 or max_anomaly > self.config.immediate_threshold if hasattr(self.config, 'immediate_threshold') else max_anomaly > 4.0:
            return "critical"
        elif max_anomaly > self.config.collision_threshold * 2:
            return "high"
        elif max_anomaly > self.config.collision_threshold * 1.5:
            return "medium"
        else:
            # For lower anomalies, check if detected by multiple temporal strategies
            detection_count = len(temporal_anomalies)
            if detection_count >= 3:
                return "medium"  # Multiple temporal detections
            else:
                return "low"

    def _log_temporal_analysis(
        self,
        temporal_anomalies: dict[str, float],
        severity: str,
    ):
        """Log detailed temporal analysis when collision is detected."""
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning(f"🔴 TEMPORAL COLLISION DETECTED - {severity.upper()} SEVERITY")
        logger.warning("=" * 80)

        # Group anomalies by detection type
        detection_types = {
            "Accumulated Deviation": [],
            "Force Gradient": [],
            "Oscillation": [],
            "Persistent Deviation": [],
            "Trend": [],
            "Correlation": [],
        }

        # Sort anomalies by value
        sorted_anomalies = sorted(
            temporal_anomalies.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for joint_name, value in sorted_anomalies:
            # Determine detection type based on value characteristics
            if joint_name in self._persistent_counter and self._persistent_counter[joint_name] > 0:
                detection_types["Persistent Deviation"].append((joint_name, value))
            else:
                detection_types["Accumulated Deviation"].append((joint_name, value))

        # Log each detection type
        for det_type, joints in detection_types.items():
            if joints:
                logger.warning(f"\n   [{det_type}]")
                for joint_name, value in joints[:5]:  # Top 5 per type
                    logger.warning(f"      📍 {joint_name}: {value:.3f} Nm")

        logger.warning("=" * 80)
        logger.warning("")

    def reset(self):
        """Reset detector state."""
        super().reset()
        self._force_history.clear()
        self._position_history.clear()
        self._anomaly_history.clear()
        self._persistent_counter.clear()
        self._oscillation_detector.reset()
        self._trend_analyzer.reset()

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics."""
        base_stats = super().get_statistics()
        temporal_stats = {
            "temporal_detections": self._temporal_detections,
            "gradient_detections": self._gradient_detections,
            "oscillation_detections": self._oscillation_detections,
            "persistent_detections": self._persistent_detections,
            "trend_detections": self._trend_detections,
            "correlation_detections": self._correlation_detections,
            "history_size": len(self._force_history),
        }
        return {**base_stats, **temporal_stats}


class OscillationDetector:
    """Detects oscillation patterns in force signals."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size

    def detect_oscillation(self, force_values: np.ndarray) -> dict:
        """Detect if force signal is oscillating.

        Oscillation is characterized by:
        - Multiple direction changes (zero crossings)
        - Consistent amplitude
        - Periodic pattern

        Args:
            force_values: Array of force values.

        Returns:
            Dict with oscillation detection results.
        """
        if len(force_values) < 4:
            return {"is_oscillating": False, "amplitude": 0.0, "cycles": 0}

        # Calculate mean and remove it
        mean_val = np.mean(force_values)
        centered = force_values - mean_val

        # Count zero crossings
        zero_crossings = 0
        for i in range(len(centered) - 1):
            if centered[i] * centered[i + 1] < 0:
                zero_crossings += 1

        # Calculate amplitude (peak-to-peak)
        amplitude = np.max(force_values) - np.min(force_values)

        # Oscillation criteria
        min_cycles = 2  # At least 2 direction changes (1 full cycle)
        min_amplitude = 0.3  # Minimum oscillation amplitude

        is_oscillating = (
            zero_crossings >= min_cycles * 2 and
            amplitude > min_amplitude
        )

        return {
            "is_oscillating": is_oscillating,
            "amplitude": amplitude,
            "cycles": zero_crossings // 2,
            "zero_crossings": zero_crossings,
        }

    def reset(self):
        """Reset detector state."""
        pass


class TrendAnalyzer:
    """Analyzes trends in force signals."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def analyze_trend(self, force_values: np.ndarray) -> dict:
        """Analyze trend in force signal.

        Args:
            force_values: Array of force values.

        Returns:
            Dict with trend analysis results.
        """
        if len(force_values) < 2:
            return {"slope": 0.0, "total_change": 0.0, "trend": "stable"}

        # Linear regression to find slope
        x = np.arange(len(force_values))
        y = force_values

        # Calculate slope using least squares
        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)

        # Calculate total change
        total_change = force_values[-1] - force_values[0]

        # Determine trend direction
        if abs(slope) < 0.05:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "slope": slope,
            "total_change": total_change,
            "trend": trend,
            "r_squared": self._calculate_r_squared(x, y, slope),
        }

    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float) -> float:
        """Calculate R-squared for the trend fit."""
        y_mean = np.mean(y)
        y_pred = slope * (x - np.mean(x)) + y_mean

        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def reset(self):
        """Reset analyzer state."""
        pass


def create_temporal_collision_config(
    collision_threshold: float = 0.8,
    temporal_window_size: int = 10,
    **kwargs
) -> TemporalCollisionConfig:
    """Create a temporal collision configuration.

    Args:
        collision_threshold: Base threshold for collision detection.
        temporal_window_size: Number of samples for temporal analysis.
        **kwargs: Additional configuration parameters.

    Returns:
        TemporalCollisionConfig instance.
    """
    return TemporalCollisionConfig(
        collision_threshold=collision_threshold,
        temporal_window_size=temporal_window_size,
        enable_gradient_detection=True,
        gradient_threshold=0.3,
        enable_oscillation_detection=True,
        enable_persistent_detection=True,
        enable_trend_detection=True,
        enable_correlation_detection=True,
        detection_window=1,  # Immediate response
        velocity_compensation=True,
        **kwargs
    )
