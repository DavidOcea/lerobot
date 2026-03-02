"""
Task completion detection based on multiple sensor modalities.

This module implements detection of task completion using:
- Position-based detection (joint positions at target)
- Force-based detection (grip confirmation)
- Stability-based detection (state stability over time)
- Composite detection (combining multiple criteria)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import CompletionCriteria

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of completion detection check."""

    is_completed: bool = False
    confidence: float = 0.0  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    satisfied_conditions: list[str] = field(default_factory=list)
    unsatisfied_conditions: list[str] = field(default_factory=list)


class TaskCompletionDetector:
    """Detects task completion based on sensor data.

    The detector supports multiple criteria types:
    - "position": Checks if joints reach target positions
    - "force": Checks if force exceeds threshold (for grip confirmation)
    - "stability": Checks if state remains stable over a time window
    - "composite": Combines multiple conditions with AND logic

    Usage:
        detector = TaskCompletionDetector(criteria)
        result = detector.detect(observation, action_history)
        if result.is_completed:
            logger.info("Task completed!")
    """

    def __init__(self, criteria: CompletionCriteria):
        """Initialize the completion detector.

        Args:
            criteria: Completion criteria configuration.
        """
        self.criteria = criteria

        # State tracking for stability detection
        self._position_buffer: deque[dict[str, float]] = deque(maxlen=100)
        self._force_buffer: deque[dict[str, float]] = deque(maxlen=100)
        self._start_time: float | None = None

        # Detection count
        self._detection_count = 0
        self._total_checks = 0

    def detect(
        self, observation: dict[str, Any], action_history: list[dict[str, Any]]
    ) -> DetectionResult:
        """Check if task is complete based on current observation.

        Args:
            observation: Current observation dict.
            action_history: Recent action history for context.

        Returns:
            DetectionResult with completion status.
        """
        self._total_checks += 1
        result = DetectionResult(timestamp=time.time())

        # Debug: Check what keys are available in observation
        if self._total_checks == 1:
            force_keys = [k for k in observation.keys() if ".force" in k]
            print(f"[CompletionDetector] First check - available force keys: {force_keys}")
            logger.info(f"[CompletionDetector] First check - available force keys: {force_keys}")

        # Strong logging: print to ensure output is visible
        print(f"[CompletionDetector] Check #{self._total_checks}, criteria_type={self.criteria.type}, buffer_size={len(self._position_buffer)}")
        logger.info(f"[CompletionDetector] Check #{self._total_checks}, buffer_size={len(self._position_buffer)}, type={self.criteria.type}")

        # Update buffers
        self._update_buffers(observation)

        if self._start_time is None:
            self._start_time = time.time()

        # Dispatch based on criteria type
        if self.criteria.type == "position":
            result = self._check_position_criteria(observation)
        elif self.criteria.type == "force":
            result = self._check_force_criteria(observation)
        elif self.criteria.type == "stability":
            result = self._check_stability(observation, action_history)
        elif self.criteria.type == "composite":
            result = self._check_composite_criteria(observation, action_history)
        else:
            logger.warning(f"Unknown criteria type: {self.criteria.type}")
            return result

        # Log composite condition details every 5 checks
        if self.criteria.type == "composite" and self._total_checks % 5 == 0:
            if result.satisfied_conditions:
                logger.info(f"[CompletionDetector] Satisfied conditions: {result.satisfied_conditions}")
            if result.unsatisfied_conditions:
                logger.info(f"[CompletionDetector] Unsatisfied conditions: {result.unsatisfied_conditions}")

        # Update detection count
        if result.is_completed:
            self._detection_count += 1
            logger.info(f"[CompletionDetector] Task completed! check={self._total_checks}, confidence={result.confidence:.2f}, details={result.details}")
        elif self._total_checks % 10 == 0:
            # Log progress every 10 checks when not completed
            logger.debug(f"[CompletionDetector] Not yet completed, confidence={result.confidence:.2f}, is_completed={result.is_completed}")

        return result

    def _check_position_criteria(self, observation: dict[str, Any]) -> DetectionResult:
        """Check if joint positions match target values.

        Args:
            observation: Current observation.

        Returns:
            DetectionResult with position-based completion status.
        """
        result = DetectionResult()

        if not self.criteria.target_joint_positions:
            return result

        all_satisfied = True
        satisfied = []
        unsatisfied = []

        for joint_name, target_pos in self.criteria.target_joint_positions.items():
            # Get current position
            current_pos = observation.get(f"{joint_name}.pos")
            if current_pos is None:
                logger.warning(f"Position not found for joint: {joint_name}")
                all_satisfied = False
                unsatisfied.append(f"{joint_name}: not found")
                continue

            # Check if within tolerance
            error = abs(current_pos - target_pos)
            is_satisfied = error <= self.criteria.position_tolerance

            if is_satisfied:
                satisfied.append(f"{joint_name}: {error:.4f} rad")
            else:
                all_satisfied = False
                unsatisfied.append(f"{joint_name}: {error:.4f} rad (max: {self.criteria.position_tolerance})")

        result.is_completed = all_satisfied
        result.confidence = 1.0 if all_satisfied else 0.0
        result.satisfied_conditions = satisfied
        result.unsatisfied_conditions = unsatisfied
        result.details = {
            "type": "position",
            "target_positions": self.criteria.target_joint_positions,
            "tolerance": self.criteria.position_tolerance,
        }

        return result

    def _check_force_criteria(self, observation: dict[str, Any]) -> DetectionResult:
        """Check if force exceeds threshold (for grip confirmation).

        Args:
            observation: Current observation.

        Returns:
            DetectionResult with force-based completion status.
        """
        result = DetectionResult()

        # Determine which joint to monitor
        joint_name = self.criteria.joint_name
        if joint_name is None:
            # Try to find a gripper joint
            for key in observation.keys():
                if "gripper" in key.lower() or "joint_7" in key:
                    joint_name = key.replace(".force", "").replace(".pos", "")
                    break

        if joint_name is None:
            logger.warning("No joint specified for force criteria")
            return result

        # Get force value
        force_key = f"{joint_name}.force"
        force = observation.get(force_key)
        if force is None:
            logger.warning(f"Force not found for joint: {joint_name}")
            return result

        # Check if force exceeds threshold
        # We use absolute value since force direction doesn't matter
        force_magnitude = abs(force)
        is_satisfied = force_magnitude >= self.criteria.force_threshold

        result.is_completed = is_satisfied
        result.confidence = min(1.0, force_magnitude / self.criteria.force_threshold)
        result.satisfied_conditions = (
            [f"{joint_name}: {force_magnitude:.3f} Nm"] if is_satisfied else []
        )
        result.unsatisfied_conditions = (
            [] if is_satisfied else [f"{joint_name}: {force_magnitude:.3f} Nm (min: {self.criteria.force_threshold})"]
        )
        result.details = {
            "type": "force",
            "joint_name": joint_name,
            "force_magnitude": force_magnitude,
            "threshold": self.criteria.force_threshold,
        }

        return result

    def _check_stability(
        self, observation: dict[str, Any], action_history: list[dict[str, Any]]
    ) -> DetectionResult:
        """Check if state has remained stable over a time window.

        Args:
            observation: Current observation.
            action_history: Recent action history.

        Returns:
            DetectionResult with stability-based completion status.
        """
        result = DetectionResult()

        if len(self._position_buffer) < self.criteria.stability_window:
            result.details = {
                "type": "stability",
                "buffer_size": len(self._position_buffer),
                "required_size": self.criteria.stability_window,
                "status": "collecting_data",
            }
            return result

        # Get recent positions
        recent_positions = list(self._position_buffer)[-self.criteria.stability_window:]

        # Check variance for each joint
        stable_joints = []
        unstable_joints = []

        all_stable = True
        joint_variances: dict[str, float] = {}

        # Get all joint names from the first sample
        joint_names = set()
        for sample in recent_positions:
            joint_names.update(sample.keys())

        for joint_name in joint_names:
            positions = [sample.get(joint_name, 0) for sample in recent_positions]

            # Compute variance
            variance = float(np.var(positions)) if len(positions) > 1 else 0.0
            joint_variances[joint_name] = variance

            is_stable = variance <= (self.criteria.stability_tolerance**2)

            if is_stable:
                stable_joints.append(f"{joint_name}: var={variance:.6f}")
            else:
                all_stable = False
                unstable_joints.append(
                    f"{joint_name}: var={variance:.6f} (max: {self.criteria.stability_tolerance**2:.6f})"
                )

        result.is_completed = all_stable
        result.confidence = 1.0 if all_stable else 0.5
        result.satisfied_conditions = stable_joints
        result.unsatisfied_conditions = unstable_joints
        result.details = {
            "type": "stability",
            "window": self.criteria.stability_window,
            "tolerance": self.criteria.stability_tolerance,
            "joint_variances": joint_variances,
        }

        return result

    def _check_composite_criteria(
        self, observation: dict[str, Any], action_history: list[dict[str, Any]]
    ) -> DetectionResult:
        """Check multiple conditions with AND logic.

        All conditions must be satisfied for completion.

        Args:
            observation: Current observation.
            action_history: Recent action history.

        Returns:
            DetectionResult with composite completion status.
        """
        result = DetectionResult()
        all_satisfied = True
        all_confidences = []

        for condition in self.criteria.conditions:
            condition_type = condition.get("type", "position")

            # Direct check without creating temporary detector
            # This ensures we use the main detector's buffer for stability checks
            if condition_type == "force":
                condition_result = self._check_force_criteria_for_condition(observation, condition)
            elif condition_type == "stability":
                condition_result = self._check_stability_for_condition(observation, action_history, condition)
            elif condition_type == "position":
                condition_result = self._check_position_criteria_for_condition(observation, condition)
            else:
                logger.warning(f"Unknown condition type in composite: {condition_type}")
                continue

            if condition_result.is_completed:
                result.satisfied_conditions.extend(
                    [f"[{condition_type}] {c}" for c in condition_result.satisfied_conditions]
                )
            else:
                all_satisfied = False
                result.unsatisfied_conditions.extend(
                    [f"[{condition_type}] {c}" for c in condition_result.unsatisfied_conditions]
                )

            all_confidences.append(condition_result.confidence)

        result.is_completed = all_satisfied
        # For composite criteria, confidence is the minimum of all condition confidences
        # This ensures that if any condition has low confidence, overall confidence is low
        result.confidence = min(all_confidences) if all_confidences else 0.0
        result.details = {
            "type": "composite",
            "num_conditions": len(self.criteria.conditions),
            "num_satisfied": len(result.satisfied_conditions),
            "min_confidence": result.confidence,
        }

        return result

    def _check_force_criteria_for_condition(
        self, observation: dict[str, Any], condition: dict[str, Any]
    ) -> DetectionResult:
        """Check force criteria with parameters from condition dict.

        Args:
            observation: Current observation.
            condition: Condition dict with parameters.

        Returns:
            DetectionResult with force-based completion status.
        """
        result = DetectionResult()

        # Get parameters from condition
        joint_name = condition.get("joint_name")
        force_threshold = condition.get("force_threshold", 0.5)

        # Determine which joint to monitor
        if joint_name is None:
            # Try to find a gripper joint
            for key in observation.keys():
                if "gripper" in key.lower() or "joint_7" in key:
                    joint_name = key.replace(".force", "").replace(".pos", "")
                    break

        if joint_name is None:
            logger.warning("No joint specified for force criteria")
            return result

        # Get force value
        force_key = f"{joint_name}.force"
        force = observation.get(force_key)
        if force is None:
            logger.warning(f"Force not found for joint: {joint_name}")
            return result

        # Check if force exceeds threshold
        force_magnitude = abs(force)
        is_satisfied = force_magnitude >= force_threshold

        result.is_completed = is_satisfied
        result.confidence = min(1.0, force_magnitude / force_threshold)
        result.satisfied_conditions = (
            [f"{joint_name}: {force_magnitude:.3f} Nm"] if is_satisfied else []
        )
        result.unsatisfied_conditions = (
            [f"{joint_name}: {force_magnitude:.3f} Nm (threshold: {force_threshold})"]
            if not is_satisfied
            else []
        )
        result.details = {
            "type": "force",
            "joint_name": joint_name,
            "force_threshold": force_threshold,
            "current_force": force_magnitude,
        }

        return result

    def _check_stability_for_condition(
        self, observation: dict[str, Any], action_history: list[dict[str, Any]], condition: dict[str, Any]
    ) -> DetectionResult:
        """Check stability criteria with parameters from condition dict.

        Uses the main detector's position buffer for history.

        Args:
            observation: Current observation.
            action_history: Recent action history.
            condition: Condition dict with parameters.

        Returns:
            DetectionResult with stability-based completion status.
        """
        result = DetectionResult()

        # Get parameters from condition
        stability_window = condition.get("stability_window", 10)
        stability_tolerance = condition.get("stability_tolerance", 0.005)

        # Check if buffer has enough data
        if len(self._position_buffer) < stability_window:
            result.details = {
                "type": "stability",
                "buffer_size": len(self._position_buffer),
                "required_size": stability_window,
                "status": "collecting_data",
            }
            result.is_completed = False
            result.confidence = len(self._position_buffer) / stability_window  # Progress towards having enough data
            result.unsatisfied_conditions = [
                f"collecting_data: {len(self._position_buffer)}/{stability_window} samples"
            ]
            return result

        # Get recent positions from main buffer
        recent_positions = list(self._position_buffer)[-stability_window:]

        # Check variance for each joint
        stable_joints = []
        unstable_joints = []

        all_stable = True
        joint_variances: dict[str, float] = {}

        # Get all joint names from the first sample
        joint_names = set()
        for sample in recent_positions:
            joint_names.update(sample.keys())

        for joint_name in joint_names:
            positions = [sample.get(joint_name, 0) for sample in recent_positions]

            # Compute variance
            variance = float(np.var(positions)) if len(positions) > 1 else 0.0
            joint_variances[joint_name] = variance

            is_stable = variance <= (stability_tolerance**2)

            if is_stable:
                stable_joints.append(f"{joint_name}: var={variance:.6f}")
            else:
                all_stable = False
                unstable_joints.append(
                    f"{joint_name}: var={variance:.6f} (max: {stability_tolerance**2:.6f})"
                )

        result.is_completed = all_stable
        result.confidence = 1.0 if all_stable else 0.5
        result.satisfied_conditions = stable_joints
        result.unsatisfied_conditions = unstable_joints
        result.details = {
            "type": "stability",
            "window": stability_window,
            "tolerance": stability_tolerance,
            "joint_variances": joint_variances,
        }

        return result

    def _check_position_criteria_for_condition(
        self, observation: dict[str, Any], condition: dict[str, Any]
    ) -> DetectionResult:
        """Check position criteria with parameters from condition dict.

        Args:
            observation: Current observation.
            condition: Condition dict with parameters.

        Returns:
            DetectionResult with position-based completion status.
        """
        result = DetectionResult()

        # Get parameters from condition
        target_joint_positions = condition.get("target_joint_positions", {})
        position_tolerance = condition.get("position_tolerance", 0.01)

        if not target_joint_positions:
            return result

        all_satisfied = True
        satisfied = []
        unsatisfied = []

        for joint_name, target_pos in target_joint_positions.items():
            # Get current position
            current_pos = observation.get(f"{joint_name}.pos")
            if current_pos is None:
                logger.warning(f"Position not found for joint: {joint_name}")
                all_satisfied = False
                unsatisfied.append(f"{joint_name}: not found")
                continue

            # Check if within tolerance
            error = abs(current_pos - target_pos)
            is_satisfied = error <= position_tolerance

            if is_satisfied:
                satisfied.append(f"{joint_name}: {error:.4f} rad")
            else:
                all_satisfied = False
                unsatisfied.append(f"{joint_name}: {error:.4f} rad (max: {position_tolerance})")

        result.is_completed = all_satisfied
        result.confidence = 1.0 if all_satisfied else 0.0
        result.satisfied_conditions = satisfied
        result.unsatisfied_conditions = unsatisfied
        result.details = {
            "type": "position",
            "target_positions": target_joint_positions,
            "tolerance": position_tolerance,
        }

        return result

    def _update_buffers(self, observation: dict[str, Any]):
        """Update internal buffers with new observation data."""
        # Extract positions
        positions = {k: v for k, v in observation.items() if ".pos" in k}
        if positions:
            self._position_buffer.append(positions)

        # Extract forces
        forces = {k: v for k, v in observation.items() if ".force" in k}
        if forces:
            self._force_buffer.append(forces)

    def reset(self):
        """Reset detector state."""
        self._position_buffer.clear()
        self._force_buffer.clear()
        self._start_time = None
        self._detection_count = 0
        self._total_checks = 0

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_checks": self._total_checks,
            "detection_count": self._detection_count,
            "detection_rate": (
                self._detection_count / self._total_checks if self._total_checks > 0 else 0
            ),
            "buffer_size": len(self._position_buffer),
        }
