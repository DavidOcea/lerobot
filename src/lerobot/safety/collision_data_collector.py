"""
Collision Data Collector for Training Learned Collision Detection Models

This module collects robot operation data with collision labels for training
machine learning models for collision detection.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CollisionDataSample:
    """A single sample of robot data with collision label."""

    timestamp: float
    observation: dict[str, Any]
    action: dict[str, Any] | None
    is_collision: bool = False
    collision_severity: str = "none"  # "none", "low", "medium", "high", "critical"
    collision_joints: list[str] = field(default_factory=list)

    # Additional context
    task_name: str = ""
    phase: str = ""  # "approach", "grasp", "transport", "release"
    environment_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "observation": self._sanitize_for_json(self.observation),
            "action": self._sanitize_for_json(self.action) if self.action else None,
            "is_collision": self.is_collision,
            "collision_severity": self.collision_severity,
            "collision_joints": self.collision_joints,
            "task_name": self.task_name,
            "phase": self.phase,
            "environment_info": self.environment_info,
        }

    @staticmethod
    def _sanitize_for_json(data: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to JSON-compatible types."""
        if isinstance(data, dict):
            return {k: CollisionDataSample._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [CollisionDataSample._sanitize_for_json(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        else:
            return str(data)


class CollisionDataCollector:
    """Collects robot operation data for collision detection training.

    Usage:
        collector = CollisionDataCollector(output_dir="./collision_data")
        collector.start_recording(task="pick_place")
        # ... run robot operations ...
        collector.mark_collision(severity="high", joints=["left_arm_joint_3"])
        # ... continue operations ...
        collector.stop_recording()
    """

    def __init__(
        self,
        output_dir: str | Path,
        auto_label_collision: bool = True,
        max_samples_per_file: int = 10000,
    ):
        """Initialize the data collector.

        Args:
            output_dir: Directory to save collected data.
            auto_label_collision: Automatically label collisions from collision_detector.
            max_samples_per_file: Maximum samples before creating a new file.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_label_collision = auto_label_collision
        self.max_samples_per_file = max_samples_per_file

        # State
        self._is_recording = False
        self._current_samples: list[CollisionDataSample] = []
        self._session_start_time: float | None = None
        self._collision_detector = None

        # Current recording info
        self._current_task: str = ""
        self._current_phase: str = ""
        self._manual_collision_label: bool = False
        self._manual_severity: str = "none"

    def set_collision_detector(self, collision_detector):
        """Set the collision detector for automatic labeling."""
        self._collision_detector = collision_detector

    def start_recording(self, task: str = "unknown", phase: str = ""):
        """Start recording robot data.

        Args:
            task: Name of the task being performed.
            phase: Current phase of the task (approach, grasp, etc.).
        """
        self._is_recording = True
        self._session_start_time = time.time()
        self._current_task = task
        self._current_phase = phase
        self._current_samples.clear()

        logger.info(f"Started recording collision data: task={task}, phase={phase}")

    def stop_recording(self) -> str:
        """Stop recording and save the collected data.

        Returns:
            Path to the saved data file.
        """
        if not self._is_recording:
            logger.warning("No active recording to stop")
            return ""

        self._is_recording = False
        output_path = self._save_samples()

        logger.info(f"Stopped recording. Saved {len(self._current_samples)} samples to {output_path}")

        return str(output_path)

    def record_sample(
        self,
        observation: dict[str, Any],
        action: dict[str, Any] | None = None,
        is_collision: bool | None = None,
        collision_severity: str | None = None,
        collision_joints: list[str] | None = None,
    ):
        """Record a single sample.

        Args:
            observation: Current observation dict.
            action: Current action dict.
            is_collision: Manual collision label (overrides auto-detection).
            collision_severity: Manual severity label.
            collision_joints: Manual list of affected joints.
        """
        if not self._is_recording:
            return

        # Determine collision label
        if is_collision is not None:
            # Manual label takes precedence
            final_is_collision = is_collision
            final_severity = collision_severity or self._manual_severity
            final_joints = collision_joints or []
        elif self.auto_label_collision and self._collision_detector is not None:
            # Auto-detect using collision detector
            result = self._collision_detector.check_collision(observation, action)
            final_is_collision = result.is_detected
            final_severity = result.severity
            final_joints = list(result.affected_joints.keys())
        else:
            # No collision
            final_is_collision = self._manual_collision_label
            final_severity = self._manual_severity
            final_joints = []

        sample = CollisionDataSample(
            timestamp=time.time(),
            observation=observation,
            action=action,
            is_collision=final_is_collision,
            collision_severity=final_severity,
            collision_joints=final_joints,
            task_name=self._current_task,
            phase=self._current_phase,
        )

        self._current_samples.append(sample)

        # Auto-save if we have enough samples
        if len(self._current_samples) >= self.max_samples_per_file:
            self._save_samples()
            self._current_samples.clear()

    def mark_collision(self, severity: str = "medium", joints: list[str] | None = None):
        """Manually mark that a collision occurred.

        This affects subsequent samples until unmark_collision() is called.

        Args:
            severity: Collision severity level.
            joints: List of affected joint names.
        """
        self._manual_collision_label = True
        self._manual_severity = severity
        logger.info(f"Manual collision mark: severity={severity}, joints={joints}")

    def unmark_collision(self):
        """Clear manual collision marking."""
        self._manual_collision_label = False
        self._manual_severity = "none"
        logger.info("Cleared manual collision mark")

    def set_phase(self, phase: str):
        """Update the current task phase."""
        self._current_phase = phase

    def _save_samples(self) -> Path:
        """Save current samples to a file."""
        if not self._current_samples:
            return None

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"collision_data_{timestamp}.jsonl"
        output_path = self.output_dir / filename

        # Save as JSONL (one JSON object per line)
        with open(output_path, "w") as f:
            for sample in self._current_samples:
                json.dump(sample.to_dict(), f)
                f.write("\n")

        return output_path

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about collected data."""
        collision_count = sum(1 for s in self._current_samples if s.is_collision)
        severity_counts = {}
        for sample in self._current_samples:
            severity_counts[sample.collision_severity] = severity_counts.get(sample.collision_severity, 0) + 1

        return {
            "total_samples": len(self._current_samples),
            "collision_samples": collision_count,
            "normal_samples": len(self._current_samples) - collision_count,
            "severity_distribution": severity_counts,
            "is_recording": self._is_recording,
            "current_task": self._current_task,
            "current_phase": self._current_phase,
        }


class FeatureExtractor:
    """Extract features from observation/action for collision detection model.

    Features include:
    - Raw force values
    - Force rate of change
    - Force variance over time
    - Position and velocity
    - Deviation from base force
    - Joint correlations
    """

    def __init__(self, history_length: int = 10):
        """Initialize feature extractor.

        Args:
            history_length: Number of past samples to use for temporal features.
        """
        self.history_length = history_length
        self._history: list[dict[str, Any]] = []

    def extract_features(
        self,
        observation: dict[str, Any],
        action: dict[str, Any] | None = None,
        base_torques: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Extract feature vector from observation.

        Args:
            observation: Current observation dict.
            action: Current action dict.
            base_torques: Calibrated base torques for deviation calculation.

        Returns:
            Feature vector as numpy array.
        """
        features = []

        # Extract force and position data
        forces = {}
        positions = {}
        for key, value in observation.items():
            if ".force" in key:
                joint_name = key.replace(".force", "")
                forces[joint_name] = float(value)
            elif ".pos" in key:
                joint_name = key.replace(".pos", "")
                positions[joint_name] = float(value)

        # 1. Raw force values (normalized)
        force_values = list(forces.values())
        features.extend(force_values)

        # 2. Force deviations from baseline
        if base_torques:
            deviations = []
            for joint_name, force in forces.items():
                base = base_torques.get(joint_name, 0.0)
                deviations.append(abs(force - base))
            features.extend(deviations)
        else:
            features.extend([0.0] * len(forces))

        # 3. Force rate of change
        if len(self._history) > 0:
            prev_forces = {}
            for key, value in self._history[-1].items():
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    prev_forces[joint_name] = float(value)

            rates = []
            for joint_name, force in forces.items():
                prev_force = prev_forces.get(joint_name, force)
                rates.append(abs(force - prev_force))
            features.extend(rates)
        else:
            features.extend([0.0] * len(forces))

        # 4. Force variance over history
        if len(self._history) >= 3:
            variance_features = self._compute_force_variance()
            features.extend(variance_features)
        else:
            features.extend([0.0] * len(forces))

        # 5. Position values (normalized)
        pos_values = list(positions.values())
        features.extend(pos_values)

        # 6. Joint correlation features (cross-joint force patterns)
        if len(forces) >= 2:
            correlation_features = self._compute_force_correlation(forces)
            features.extend(correlation_features)
        else:
            features.extend([0.0] * 4)

        # Update history
        self._history.append(observation.copy())
        if len(self._history) > self.history_length:
            self._history.pop(0)

        return np.array(features, dtype=np.float32)

    def _compute_force_variance(self) -> list[float]:
        """Compute force variance over recent history."""
        variances = {}

        # Collect force history per joint
        force_history = {}
        for obs in self._history:
            for key, value in obs.items():
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    if joint_name not in force_history:
                        force_history[joint_name] = []
                    force_history[joint_name].append(float(value))

        # Compute variance for each joint
        for joint_name, values in force_history.items():
            if len(values) >= 2:
                variances[joint_name] = float(np.var(values))
            else:
                variances[joint_name] = 0.0

        # Return as list (sorted by joint name for consistency)
        return [variances.get(joint, 0.0) for joint in sorted(variances.keys())]

    def _compute_force_correlation(self, current_forces: dict[str, float]) -> list[float]:
        """Compute cross-joint force correlation features."""
        # Simple correlation-like features
        force_list = list(current_forces.values())

        if len(force_list) < 2:
            return [0.0] * 4

        features = []

        # 1. Max/Min ratio (indicates imbalance)
        max_force = max(abs(f) for f in force_list)
        min_force = min(abs(f) for f in force_list) if force_list else 0.001
        features.append(max_force / (min_force + 0.001))

        # 2. Sum of absolute forces
        features.append(sum(abs(f) for f in force_list))

        # 3. Standard deviation of forces
        features.append(float(np.std(force_list)))

        # 4. Number of joints with high force (> threshold)
        high_threshold = 0.5
        features.append(sum(1 for f in force_list if abs(f) > high_threshold))

        return features

    def get_feature_dim(self, num_joints: int) -> int:
        """Get the total feature dimension for a given number of joints."""
        return (
            num_joints  # Raw forces
            + num_joints  # Deviations from baseline
            + num_joints  # Force rates
            + num_joints  # Force variance
            + num_joints  # Positions
            + 4  # Correlation features
        )

    def reset(self):
        """Reset the history buffer."""
        self._history.clear()
