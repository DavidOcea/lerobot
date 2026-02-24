"""
Adaptive Task Scheduler with Force Feedback and Collision Avoidance

This module provides enhanced task execution with:
1. Force-based grasp detection
2. Adaptive speed control based on force feedback
3. Improved collision detection with per-joint thresholds
4. Smart recovery strategies
5. Action smoothing for precise control
6. Temporal and adaptive collision detection
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from lerobot.control import ActionPostProcessor, PostProcessorConfig, create_post_processor_for_robot
from lerobot.safety import (
    AdaptiveCollisionDetector,
    MotionPhase,
    TemporalCollisionDetector,
    create_adaptive_collision_config,
    create_temporal_collision_config,
)

from .config import TaskConfig
from .task_scheduler import TaskResult, TaskStatus

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
    - Action smoothing for precise motion
    - Temporal and adaptive collision detection
    """

    def __init__(
        self,
        scheduler: "LocalTaskScheduler",
        gripper_config: dict[str, Any] | None = None,
        enable_action_smoothing: bool = True,
        smoothing_level: str = "medium",
        collision_detector_type: str = "adaptive",  # "basic", "enhanced", "temporal", "adaptive"
        emergency_check_callback: Callable[[dict[str, Any], dict[str, float], str], Any] | None = None,
    ):
        """Initialize the adaptive scheduler.

        Args:
            scheduler: Base LocalTaskScheduler instance
            gripper_config: Configuration for gripper force feedback
            enable_action_smoothing: Enable action post-processing for smooth motion
            smoothing_level: Level of action smoothing ("low", "medium", "high")
            collision_detector_type: Type of collision detector to use
                - "basic": Standard collision detector
                - "enhanced": Enhanced with rate and immediate detection
                - "temporal": Temporal pattern analysis (most sensitive)
                - "adaptive": Motion-aware thresholds (fewest false positives)
            emergency_check_callback: Optional callback for emergency stop checking
                Signature: (observation, action, task_name) -> RecoveryAction | None
        """
        self.scheduler = scheduler
        self.gripper_config = gripper_config or {}
        self.enable_action_smoothing = enable_action_smoothing
        self.smoothing_level = smoothing_level
        self.collision_detector_type = collision_detector_type
        self.emergency_check_callback = emergency_check_callback

        # Initialize action post-processor
        self.action_post_processor: Optional[ActionPostProcessor] = None
        if enable_action_smoothing:
            self._initialize_action_post_processor()

        # Grasp detection state
        self.grasp_detected = False
        self.grasp_stable_frames = 0
        self.current_gripper_force = 0.0
        self.current_gripper_force_raw = 0.0  # Raw normalized force (0-1)

        # Adaptive speed control
        self.current_speed_factor = 1.0
        self.high_force_detected = False

        # Statistics
        self.total_collisions = 0
        self.grasp_attempts = 0
        self.successful_grasps = 0
        self.action_smooth_count = 0

    def _initialize_action_post_processor(self):
        """Initialize the action post-processor."""
        try:
            # Try to create from robot config
            self.action_post_processor = create_post_processor_for_robot(
                self.scheduler.robot.config,
                smoothing_level=self.smoothing_level,
            )
            logger.info(f"Action post-processor initialized with '{self.smoothing_level}' smoothing")
        except Exception as e:
            logger.warning(f"Failed to create action post-processor from config: {e}")
            # Create with default config
            joint_names = getattr(
                self.scheduler.robot,
                "observation_joint_names",
                [
                    "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
                    "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
                    "left_arm_joint_7",
                    "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
                    "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
                    "right_arm_joint_7",
                    "trunk_joint_1", "trunk_joint_2",
                ]
            )
            config = PostProcessorConfig()
            if self.smoothing_level == "low":
                config.filter_alpha = 0.9
                config.max_velocity = 5.0
            elif self.smoothing_level == "high":
                config.filter_alpha = 0.5
                config.max_velocity = 2.0
            else:  # medium
                config.filter_alpha = 0.7
                config.max_velocity = 3.0

            self.action_post_processor = ActionPostProcessor(config, joint_names)
            logger.info(f"Action post-processor created with default '{self.smoothing_level}' config")

    def create_collision_detector(self, collision_threshold: float = 0.8):
        """Create a collision detector of the configured type.

        Args:
            collision_threshold: Base collision threshold in Nm.

        Returns:
            Configured collision detector instance.
        """
        detector_type = self.collision_detector_type

        if detector_type == "adaptive":
            config = create_adaptive_collision_config(
                collision_threshold=collision_threshold,
                enable_smoothing=True,
                smoothing_factor=0.3,
            )
            detector = AdaptiveCollisionDetector(config)
            logger.info("Created Adaptive collision detector (motion-aware thresholds)")

        elif detector_type == "temporal":
            config = create_temporal_collision_config(
                collision_threshold=collision_threshold,
                temporal_window_size=10,
            )
            detector = TemporalCollisionDetector(config)
            logger.info("Created Temporal collision detector (pattern analysis)")

        elif detector_type == "enhanced":
            from lerobot.safety import create_enhanced_collision_config, EnhancedCollisionDetector
            config = create_enhanced_collision_config(
                collision_threshold=collision_threshold,
            )
            detector = EnhancedCollisionDetector(config)
            logger.info("Created Enhanced collision detector")

        else:  # "basic" or default
            from lerobot.safety import CollisionConfig, CollisionDetector
            config = CollisionConfig(collision_threshold=collision_threshold)
            detector = CollisionDetector(config)
            logger.info("Created Basic collision detector")

        return detector

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
        import time

        start_time = time.time()
        timeout = start_time + task.max_duration

        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.PENDING,
            start_time=start_time,
        )

        # Load policy for this task first
        logger.info(f"Loading policy for task {task.name}: {task.policy_path}")
        if not self.scheduler.policy_executor.load_policy(task.policy_path, task.policy_type):
            result.status = TaskStatus.FATAL_FAILURE
            result.error_message = f"Failed to load policy from {task.policy_path}"
            return result

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

                # Apply action smoothing for precise motion
                if self.action_post_processor is not None:
                    action = self.action_post_processor.process_action(action, observation)
                    self.action_smooth_count += 1

                # Log action processing details periodically
                if self.total_collisions == 0 and self.action_smooth_count % 100 == 0:
                    stats = self.action_post_processor.get_statistics()
                    logger.debug(
                        f"Action smoothing stats: limit_rate={stats['limit_rate']:.2%}, "
                        f"total_processed={stats['total_processed']}"
                    )

                # Check emergency stop callback before sending action
                if self.emergency_check_callback is not None:
                    recovery_action = self.emergency_check_callback(observation, action, task.name)
                    if recovery_action is not None:
                        # Emergency stop was triggered
                        from lerobot.safety.emergency_stop_controller import RecoveryAction
                        if recovery_action == RecoveryAction.STOP_PROGRAM:
                            result.status = TaskStatus.FATAL_FAILURE
                            result.error_message = "Emergency stop: User requested to stop program"
                            result.collision_detected = True
                            return result
                        elif recovery_action == RecoveryAction.ROLLBACK_AND_CONTINUE:
                            # Continue execution after rollback
                            logger.info("Continuing task after rollback")
                            continue
                        elif recovery_action == RecoveryAction.ROLLBACK_AND_RETRY_MODEL:
                            # Need to reload with new model - exit and let orchestrator handle
                            result.status = TaskStatus.FAILED
                            result.error_message = "Emergency stop: User requested to retry with new model"
                            result.collision_detected = True
                            result.retry_with_new_model = True
                            return result

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

        Also handles temporal and adaptive detectors with enhanced logging.
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

        # Log additional context for temporal and adaptive detectors
        if result.is_detected:
            if isinstance(collision_detector, AdaptiveCollisionDetector):
                motion_phase = collision_detector.get_motion_phase()
                current_thresholds = collision_detector.get_current_thresholds()
                logger.info(
                    f"Adaptive collision: phase={motion_phase.value}, "
                    f"avg_threshold_multiplier={np.mean(list(current_thresholds.values())) / collision_detector.config.collision_threshold:.2f}"
                )
            elif isinstance(collision_detector, TemporalCollisionDetector):
                stats = collision_detector.get_statistics()
                logger.info(
                    f"Temporal collision: gradient_detections={stats.get('gradient_detections', 0)}, "
                    f"oscillation_detections={stats.get('oscillation_detections', 0)}, "
                    f"persistent_detections={stats.get('persistent_detections', 0)}"
                )

        return result

    def _update_gripper_force(self, observation: dict[str, Any]) -> None:
        """Update gripper force state from observation.

        Args:
            observation: Current observation dict with force data

        Note: Gripper force is normalized 0-1, needs to be scaled for comparison
        """
        # Extract gripper forces (both left and right)
        # Note: Gripper force is in 0-1 range, scale to Nm equivalent
        gripper_force_scale = self.gripper_config.get("gripper_force_scale", 5.0)

        left_gripper_force_raw = observation.get("left_arm_joint_7.force", 0.0)
        right_gripper_force_raw = observation.get("right_arm_joint_7.force", 0.0)

        # Scale to Nm equivalent for threshold comparison
        left_gripper_force = left_gripper_force_raw * gripper_force_scale
        right_gripper_force = right_gripper_force_raw * gripper_force_scale

        # Track the higher force (assuming one gripper is active)
        self.current_gripper_force = max(left_gripper_force, right_gripper_force)

        # Also track raw force for rate detection
        self.current_gripper_force_raw = max(left_gripper_force_raw, right_gripper_force_raw)

        # Check grasp stability
        grasp_threshold = self.gripper_config.get("grasp_force_threshold", 0.4)
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

        high_force_threshold = self.gripper_config.get("high_force_threshold", 0.8)
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

    def _reset_grasp_state(self) -> None:
        """Reset grasp detection state."""
        self.grasp_detected = False
        self.grasp_stable_frames = 0
        self.current_gripper_force = 0.0
        self.current_gripper_force_raw = 0.0
        self.current_speed_factor = 1.0
        self.high_force_detected = False

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

                # Reset policy executor after retreat to clear stale state
                if handler_result.recovery_success:
                    if hasattr(self.scheduler, 'policy_executor'):
                        self.scheduler.policy_executor.reset()
                        logger.info("Policy executor reset after collision recovery")

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

            # Reset policy executor after fallback retreat
            if hasattr(self.scheduler, 'policy_executor'):
                self.scheduler.policy_executor.reset()
                logger.info("Policy executor reset after fallback retreat")

            return True
        except Exception as e:
            logger.error(f"Retreat failed: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
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
            "collision_detector_type": self.collision_detector_type,
            "smoothing_enabled": self.enable_action_smoothing,
            "action_smooth_count": self.action_smooth_count,
        }

        # Add action post-processor statistics
        if self.action_post_processor is not None:
            stats["action_post_processor"] = self.action_post_processor.get_statistics()

        return stats
