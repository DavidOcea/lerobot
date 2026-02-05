"""
Adaptive Task Scheduler with Force Feedback and Collision Avoidance

This module provides enhanced task execution with:
1. Force-based grasp detection
2. Adaptive speed control based on force feedback
3. Improved collision detection with per-joint thresholds
4. Smart recovery strategies
"""

import logging
from typing import TYPE_CHECKING, Any

from .config import TaskConfig, TaskResult, TaskStatus

if TYPE_CHECKING:
    from .task_scheduler import LocalTaskScheduler

logger = logging.getLogger(__name__)


class AdaptiveTaskScheduler:
    """Enhanced task scheduler with force feedback and adaptive control.

    Features:
    - Force-based grasp detection
    - Adaptive speed control (slow down when high force detected)
    - Per-joint collision thresholds
    - Smart collision recovery
    """

    def __init__(
        self,
        scheduler: "LocalTaskScheduler",
        gripper_config: dict[str, Any] | None = None,
    ):
        """Initialize the adaptive scheduler.

        Args:
            scheduler: Base LocalTaskScheduler instance
            gripper_config: Configuration for gripper force feedback
        """
        self.scheduler = scheduler
        self.gripper_config = gripper_config or {}

        # Grasp detection state
        self.grasp_detected = False
        self.grasp_stable_frames = 0
        self.current_gripper_force = 0.0

        # Adaptive speed control
        self.current_speed_factor = 1.0
        self.high_force_detected = False

        # Statistics
        self.total_collisions = 0
        self.grasp_attempts = 0
        self.successful_grasps = 0

    def execute_task_adaptive(
        self,
        task: TaskConfig,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> TaskResult:
        """Execute task with adaptive force feedback control.

        Args:
            task: Task configuration
            collision_detector: Optional collision detector
            collision_handler: Optional collision handler
            state_monitor: Optional state monitor

        Returns:
            TaskResult with execution outcome
        """
        from time import time

        start_time = time.time()
        timeout = start_time + task.max_duration

        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.PENDING,
            start_time=start_time,
        )

        # Reset state
        self.scheduler.policy_executor.reset()
        self._reset_grasp_state()

        # Get control frequency
        control_dt = 1.0 / self.scheduler.robot.config.control_frequency

        try:
            last_action = None
            last_observation = None

            while time.time() < timeout:
                loop_start = time.time()

                # Get current observation
                try:
                    observation = self.scheduler.robot.get_observation()
                except Exception as e:
                    result.status = TaskStatus.FATAL_FAILURE
                    result.error_message = f"Failed to get observation: {e}"
                    return result

                last_observation = observation

                # Update monitor
                if state_monitor is not None:
                    state_monitor.update(observation, last_action)

                # Check collision with per-joint thresholds
                if collision_detector is not None:
                    collision_result = self._check_collision_with_thresholds(
                        observation, last_action, collision_detector
                    )
                    if collision_result.is_detected:
                        self.total_collisions += 1
                        logger.warning(
                            f"Collision detected! Affected joints: {collision_result.affected_joints}"
                        )

                        # Smart recovery based on collision severity
                        if collision_result.severity == "high":
                            # High severity - full retreat
                            result.collision_detected = True
                            if self._recover_from_collision(
                                collision_result, observation, collision_handler
                            ):
                                continue
                            else:
                                result.status = TaskStatus.FATAL_FAILURE
                                result.error_message = "Severe collision - cannot continue"
                                return result
                        else:
                            # Low/medium severity - pause and continue
                            time.sleep(0.2)
                            continue

                # Check grasp force feedback
                self._update_gripper_force(observation)

                # Adaptive speed control
                speed_factor = self._compute_adaptive_speed_factor()
                if speed_factor != self.current_speed_factor:
                    self.current_speed_factor = speed_factor
                    logger.debug(f"Speed factor adjusted to {speed_factor:.2f}")

                # Check task completion with force feedback
                if self._check_task_completion(observation, task):
                    result.status = TaskStatus.COMPLETED
                    result.success = True
                    result.completion_confidence = 1.0
                    result.final_observation = observation
                    logger.info(f"Task {task.name} completed successfully")
                    return result

                # Get action from policy
                action = self.scheduler.policy_executor.get_action(observation)
                if action is None:
                    result.error_message = "Failed to get action from policy"
                    return result

                # Apply adaptive speed control
                action = self._apply_speed_control(action, speed_factor)

                # Send action to robot
                try:
                    sent_action = self.scheduler.robot.send_action(action)
                    self.scheduler.action_history.append(sent_action)
                    self.scheduler.observation_history.append(observation)
                    last_action = sent_action
                except Exception as e:
                    result.error_message = f"Failed to send action: {e}"
                    return result

                # Maintain control frequency
                loop_time = time.time() - loop_start
                sleep_time = control_dt - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Timeout
            result.status = TaskStatus.FAILED
            result.error_message = f"Task timeout after {task.max_duration}s"

        except Exception as e:
            result.status = TaskStatus.FATAL_FAILURE
            result.error_message = f"Exception during execution: {e}"
            logger.exception("Exception during adaptive task execution")

        result.end_time = time.time()
        return result

    def _check_collision_with_thresholds(
        self,
        observation: dict[str, Any],
        action: dict[str, Any] | None,
        collision_detector,
    ) -> Any:
        """Check collision with per-joint threshold support.

        This extends the basic collision detection to use joint-specific thresholds
        for more sensitive detection on fragile joints.
        """
        # Get base collision result
        result = collision_detector.check_collision(observation, action)

        # Enhance with per-joint severity assessment
        if result.is_detected and hasattr(collision_detector.config, 'joint_specific_thresholds'):
            for joint_name, anomaly in result.affected_joints.items():
                # Get joint-specific threshold
                joint_threshold = collision_detector.config.joint_specific_thresholds.get(
                    joint_name, collision_detector.config.collision_threshold
                )

                # Determine severity based on how much we exceeded threshold
                excess_ratio = anomaly / joint_threshold if joint_threshold > 0 else float('inf')

                if excess_ratio > 3.0:
                    result.severity = "high"
                elif excess_ratio > 1.5:
                    result.severity = "medium"
                else:
                    result.severity = "low"

        return result

    def _update_gripper_force(self, observation: dict[str, Any]) -> None:
        """Update gripper force state from observation.

        Args:
            observation: Current observation dict with force data
        """
        # Extract gripper forces (both left and right)
        left_gripper_force = observation.get("left_arm_joint_7.force", 0.0)
        right_gripper_force = observation.get("right_arm_joint_7.force", 0.0)

        # Track the higher force (assuming one gripper is active)
        self.current_gripper_force = max(left_gripper_force, right_gripper_force)

        # Check grasp stability
        grasp_threshold = self.gripper_config.get("grasp_force_threshold", 0.8)
        stable_frames = self.gripper_config.get("grasp_force_stable_frames", 3)

        if self.current_gripper_force > grasp_threshold:
            self.grasp_stable_frames += 1
            if self.grasp_stable_frames >= stable_frames:
                self.grasp_detected = True
        else:
            self.grasp_stable_frames = 0
            self.grasp_detected = False

    def _compute_adaptive_speed_factor(self) -> float:
        """Compute adaptive speed factor based on force feedback.

        Returns:
            Speed factor between 0.1 (very slow) and 1.0 (normal speed)
        """
        if not self.gripper_config.get("enable_adaptive_speed", True):
            return 1.0

        high_force_threshold = self.gripper_config.get("high_force_threshold", 1.5)
        slow_speed_factor = self.gripper_config.get("slow_speed_factor", 0.3)

        # Reduce speed when approaching objects (high force detected)
        if self.current_gripper_force > high_force_threshold:
            self.high_force_detected = True
            # Gradually reduce speed based on force level
            force_ratio = min(self.current_gripper_force / high_force_threshold, 2.0)
            speed = max(slow_speed_factor, 1.0 - (force_ratio - 1.0) * 0.5)
            return speed
        else:
            self.high_force_detected = False
            return 1.0

    def _apply_speed_control(
        self, action: dict[str, float], speed_factor: float
    ) -> dict[str, float]:
        """Apply adaptive speed control to action.

        Instead of directly sending the policy's target position,
        we compute an intermediate target that's a small step towards
        the goal based on current position and speed factor.

        Args:
            action: Target action from policy
            speed_factor: Speed factor (0.1 to 1.0)

        Returns:
            Modified action with speed control applied
        """
        if speed_factor >= 1.0:
            return action  # No speed reduction needed

        # Get current position
        try:
            current_positions = self.scheduler.robot.get_current_position()
        except Exception as e:
            logger.warning(f"Failed to get current position for speed control: {e}")
            return action

        # Compute intermediate target
        modified_action = {}
        max_step = self.scheduler.robot.config.max_relative_joint_move * speed_factor

        for joint_name in self.scheduler.robot.observation_joint_names:
            key = f"{joint_name}.pos"
            if key not in action:
                continue

            current_pos = current_positions.get(joint_name, 0.0)
            target_pos = action[key]

            # Limit the step size
            diff = target_pos - current_pos
            if abs(diff) > max_step:
                limited_diff = max_step if diff > 0 else -max_step
                modified_action[key] = current_pos + limited_diff
            else:
                modified_action[key] = target_pos

        return modified_action

    def _check_task_completion(
        self, observation: dict[str, Any], task: TaskConfig
    ) -> bool:
        """Check if task is completed using force feedback.

        Args:
            observation: Current observation
            task: Task configuration

        Returns:
            True if task is completed
        """
        # Use completion detector if available
        if self.scheduler.completion_detector is not None:
            detection = self.scheduler.completion_detector.detect(
                observation, list(self.scheduler.action_history)
            )
            if detection.is_completed and detection.confidence > 0.7:
                return True

        # Force-based completion detection for grasping tasks
        if "grasp" in task.name.lower() or "pick" in task.name.lower():
            # Check if stable grasp is detected
            return self.grasp_detected

        return False

    def _recover_from_collision(
        self,
        collision_result,
        observation: dict[str, Any],
        collision_handler,
    ) -> bool:
        """Recover from collision with smart strategy.

        Args:
            collision_result: Collision detection result
            observation: Current observation
            collision_handler: Collision handler instance

        Returns:
            True if recovery successful, False if should abort
        """
        try:
            if collision_handler is not None:
                handler_result = collision_handler.handle_collision(
                    collision_result, observation
                )
                return handler_result.can_continue
        except Exception as e:
            logger.error(f"Collision handler failed: {e}")

        # Fallback: simple retreat
        try:
            # Move back slightly
            retreat_distance = 0.02  # Small retreat
            current_pos = self.scheduler.robot.get_current_position()

            retreat_action = {}
            for joint_name, pos in current_pos.items():
                # Move in opposite direction of recent movement
                retreat_action[f"{joint_name}.pos"] = pos

            self.scheduler.robot.send_action(retreat_action)
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"Retreat failed: {e}")
            return False

    def _reset_grasp_state(self) -> None:
        """Reset grasp detection state."""
        self.grasp_detected = False
        self.grasp_stable_frames = 0
        self.current_gripper_force = 0.0
        self.current_speed_factor = 1.0
        self.high_force_detected = False

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_collisions": self.total_collisions,
            "grasp_attempts": self.grasp_attempts,
            "successful_grasps": self.successful_grasps,
            "grasp_success_rate": (
                self.successful_grasps / self.grasp_attempts
                if self.grasp_attempts > 0
                else 0
            ),
            "current_gripper_force": self.current_gripper_force,
            "grasp_detected": self.grasp_detected,
            "current_speed_factor": self.current_speed_factor,
        }
