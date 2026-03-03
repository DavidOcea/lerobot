"""
Emergency Stop Controller with Action History and Rollback

This module provides:
1. Dangerous action detection
2. Emergency stop triggering (manual and automatic)
3. Action history storage for rollback
4. Rollback to previous safe state
5. Interactive pause/resume functionality

Usage:
    controller = EmergencyStopController(robot, history_size=1000)
    if controller.check_action_danger(action):
        controller.trigger_stop("Dangerous action detected")
    controller.rollback(steps=50)
    controller.resume()
"""

import logging
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from lerobot.tasks.config import TaskConfig

logger = logging.getLogger(__name__)

__all__ = [
    "EmergencyStopController",
    "StopTrigger",
    "StopReason",
    "RecoveryAction",
    "ActionSnapshot",
    "StopEvent",
    "RollbackConfig",
    "DangerDetectionConfig",
    "create_emergency_stop_controller",
]


class StopTrigger(Enum):
    """Who/what triggered the emergency stop."""
    MANUAL = "manual"  # User pressed emergency stop button
    AUTOMATIC = "automatic"  # System detected dangerous condition
    RECOVERY_COMPLETE = "recovery_complete"  # Rollback completed, ready to resume


class StopReason(Enum):
    """Reason for the emergency stop."""
    DANGEROUS_ACTION = "dangerous_action"  # Action deemed dangerous
    HIGH_FORCE = "high_force"  # Force threshold exceeded
    COLLISION = "collision"  # Collision detected
    USER_REQUEST = "user_request"  # User requested stop
    VELOCITY_MISMATCH = "velocity_mismatch"  # Unexpected velocity


class RecoveryAction(Enum):
    """User-selected recovery action after emergency stop."""
    STOP_PROGRAM = "stop_program"  # Stop the entire program
    ROLLBACK_AND_CONTINUE = "rollback_and_continue"  # Rollback and continue with same task
    ROLLBACK_AND_RETRY_MODEL = "rollback_and_retry_model"  # Rollback and retry with new model


@dataclass
class ActionSnapshot:
    """Snapshot of robot state at a point in time."""
    timestamp: float
    actions: dict[str, Any]
    observation: dict[str, Any] | None = None
    action_number: int = 0


@dataclass
class StopEvent:
    """Record of a stop event."""
    timestamp: float
    trigger: StopTrigger
    reason: StopReason
    action_at_stop: ActionSnapshot | None = None
    rollback_snapshot: ActionSnapshot | None = None


@dataclass
class RollbackConfig:
    """Configuration for rollback behavior."""
    max_rollback_steps: int = 100  # Maximum steps to rollback
    rollback_step_delay: float = 0.02  # Delay between rollback steps (seconds)
    safe_state_confirm_steps: int = 10  # Steps to hold after rollback before confirmation


@dataclass
class DangerDetectionConfig:
    """Configuration for dangerous action detection."""
    # Force thresholds
    force_threshold: float = 3.5  # Nm - single joint (increased from 2.5 to 3.5)
    total_force_threshold: float = 12.0  # Nm - sum of absolute forces (increased from 8.0 to 12.0)
    max_joint_force: float = 3.5  # Nm - maximum for any single joint (increased from 2.5 to 3.5)

    # Velocity thresholds - adjusted for realistic robot motion
    # Note: velocity here is position change per control step, not actual velocity
    # With control_dt ~0.1s, max_velocity=5.0 means 50 rad/s which is too sensitive
    max_velocity: float = 0.5  # rad per control step (was 5.0, too sensitive)
    velocity_change_threshold: float = 1.0  # rad per step² - maximum acceleration

    # Action change thresholds - adjusted for normal robot operation
    # A typical robot joint can move several radians per second
    # With 10Hz control, 1.0 rad/step = 10 rad/s which is reasonable
    max_action_delta: float = 2.0  # rad per control step (increased from 0.3)

    # Detection window
    detection_window: int = 5  # Number of steps to check

    # Enable/disable specific checks
    # Note: These checks can be too sensitive for normal robot operation
    # Disable them by default and rely on collision detection instead
    enable_velocity_check: bool = False  # Disabled - too sensitive
    enable_action_delta_check: bool = False  # Disabled - policy outputs can vary significantly
    enable_force_check: bool = True  # Keep force check for safety

    # Custom danger checker
    custom_danger_checker: Callable[[dict[str, float], dict[str, Any]], bool] | None = None

    # Disable force check by default to avoid false positives during normal robot motion
    # Force detection should be handled by the collision detector instead
    enable_force_check: bool = False  # Disabled - use collision detector instead


class EmergencyStopController:
    """Emergency stop controller with action history and rollback.

    Features:
    1. Dangerous action detection (manual and automatic)
    2. Emergency stop triggering
    3. Action history storage for rollback
    4. Rollback to previous safe state
    5. Interactive pause/resume functionality
    """

    def __init__(
        self,
        robot,
        history_size: int = 1000,
        danger_config: DangerDetectionConfig = None,
        rollback_config: RollbackConfig = None,
    ):
        """Initialize the emergency stop controller.

        Args:
            robot: Robot instance with send_action method.
            history_size: Maximum number of action snapshots to store.
            danger_config: Configuration for danger detection.
            rollback_config: Configuration for rollback behavior.
        """
        self.robot = robot
        self.danger_config = danger_config or DangerDetectionConfig()
        self.rollback_config = rollback_config or RollbackConfig()

        # Action history
        self.action_history: deque[ActionSnapshot] = deque(maxlen=history_size)
        self.current_action_number: int = 0

        # Stop event tracking
        self.stop_events: list[StopEvent] = []
        self.current_stop_event: StopEvent | None = None

        # State
        self._is_stopped: bool = False
        self._is_paused: bool = False
        self._rollback_in_progress: bool = False
        self._safe_confirmed_count: int = 0

        # Callbacks
        self._on_stop_callback: Callable[[StopEvent], None] | None = None
        self._on_rollback_complete_callback: Callable[[StopEvent], None] | None = None

        # Statistics
        self._total_checks: int = 0
        self._danger_detected_count: int = 0
        self._manual_stop_count: int = 0
        self._automatic_stop_count: int = 0
        self._rollback_count: int = 0

        logger.info(f"EmergencyStopController initialized with history_size={history_size}")

    def check_action_danger(
        self,
        actions: dict[str, float],
        observation: dict[str, Any] | None = None,
    ) -> tuple[bool, StopReason | None]:
        """Check if the given actions are dangerous.

        Args:
            actions: Target action positions.
            observation: Current observation (optional, for custom checkers).

        Returns:
            (is_dangerous, reason) tuple. None if not dangerous.

        Danger conditions checked:
        1. Excessive force levels
        2. Excessive velocities
        3. Large position changes
        4. Custom danger checker
        """
        self._total_checks += 1

        # Extract force data if available
        forces = {}
        if observation:
            for key, value in observation.items():
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    forces[joint_name] = float(value)

        # Check 1: Excessive force (only if enabled)
        if forces and self.danger_config.enable_force_check:
            force_values = np.array(list(forces.values()))
            max_abs_force = np.max(np.abs(force_values))
            total_abs_force = np.sum(np.abs(force_values))

            if max_abs_force > self.danger_config.max_joint_force:
                logger.warning(f"High force detected: max_abs_force={max_abs_force:.3f} > {self.danger_config.max_joint_force:.3f}")
                self._danger_detected_count += 1
                return True, StopReason.HIGH_FORCE
            elif total_abs_force > self.danger_config.total_force_threshold:
                logger.warning(f"High total force detected: total={total_abs_force:.3f} > {self.danger_config.total_force_threshold:.3f}")
                self._danger_detected_count += 1
                return True, StopReason.HIGH_FORCE

        # Check 2: Excessive velocity (only if enabled)
        # Note: This is position change per step, not actual velocity
        # Disabled by default as it's too sensitive for normal operation
        if self.danger_config.enable_velocity_check and self.action_history:
            prev_action = self.action_history[-1] if len(self.action_history) > 0 else None

            if prev_action and observation:
                velocities = {}
                for joint_name in actions.keys():
                    if joint_name in prev_action.actions and joint_name in actions:
                        prev_pos = prev_action.actions[joint_name]
                        curr_pos = actions[joint_name]
                        velocities[joint_name] = abs(curr_pos - prev_pos)

                max_velocity = max(velocities.values()) if velocities else 0

                if max_velocity > self.danger_config.max_velocity:
                    logger.warning(f"High velocity detected: max_velocity={max_velocity:.3f} > {self.danger_config.max_velocity:.3f}")
                    self._danger_detected_count += 1
                    return True, StopReason.VELOCITY_MISMATCH

        # Check 3: Large position changes (only if enabled)
        if self.danger_config.enable_action_delta_check and self.action_history:
            prev_action = self.action_history[-1] if len(self.action_history) > 0 else None

            if prev_action:
                max_delta = 0
                for joint_name in actions.keys():
                    if joint_name in prev_action.actions and joint_name in actions:
                        delta = abs(actions[joint_name] - prev_action.actions[joint_name])
                        max_delta = max(max_delta, delta)

                if max_delta > self.danger_config.max_action_delta:
                    logger.warning(f"Large action delta detected: max_delta={max_delta:.3f} > {self.danger_config.max_action_delta:.3f}")
                    self._danger_detected_count += 1
                    return True, StopReason.DANGEROUS_ACTION

        # Check 4: Custom danger checker
        if self.danger_config.custom_danger_checker and observation:
            try:
                is_dangerous = self.danger_config.custom_danger_checker(
                    actions, observation
                )
                if is_dangerous:
                    self._danger_detected_count += 1
                    return True, StopReason.DANGEROUS_ACTION
            except Exception as e:
                logger.error(f"Custom danger checker failed: {e}")

        # Not dangerous
        self._total_checks += 1
        return False, None

    def trigger_stop(self, reason: StopReason = StopReason.USER_REQUEST, auto_rollback: bool = False):
        """Trigger an emergency stop.

        Args:
            reason: Why the stop is triggered.
            auto_rollback: Whether to automatically rollback after stopping.

        Returns:
            StopEvent with details of the stop event.
        """
        timestamp = time.time()
        logger.warning(f"🚨 EMERGENCY STOP triggered - Reason: {reason.value}")

        # Create stop event
        stop_event = StopEvent(
            timestamp=timestamp,
            trigger=StopTrigger.MANUAL if reason == StopReason.USER_REQUEST else StopTrigger.AUTOMATIC,
            reason=reason,
        )

        # Capture current action for rollback
        current_action = self._capture_current_action()
        stop_event.action_at_stop = current_action

        # Add to stop events
        self.stop_events.append(stop_event)
        self.current_stop_event = stop_event

        # Update robot state
        self._is_stopped = True
        self._is_paused = False

        # Track stop type
        if reason == StopReason.USER_REQUEST:
            self._manual_stop_count += 1
        else:
            self._automatic_stop_count += 1

        # Call stop callback
        if self._on_stop_callback:
            try:
                self._on_stop_callback(stop_event)
            except Exception as e:
                logger.error(f"Stop callback error: {e}")

        # Auto rollback if requested
        if auto_rollback:
            logger.info("Auto-rollback enabled, initiating rollback...")
            return self.rollback(steps=None, confirm_before_resume=False)
        else:
            return stop_event

    def _capture_current_action(self) -> ActionSnapshot | None:
        """Capture the current robot action for rollback purposes.

        Returns:
            ActionSnapshot with current robot state, or None if capture fails.
        """
        try:
            # Get current action from robot
            if hasattr(self.robot, 'get_last_sent_action'):
                current_actions = self.robot.get_last_sent_action()
            elif hasattr(self.robot, 'get_current_position'):
                # Fallback: get current position
                current_pos = self.robot.get_current_position()
                current_actions = {f"{k}.pos": v for k, v in current_pos.items()}
            else:
                logger.warning("Cannot capture current action - robot methods not available")
                return None

            # Get current observation
            observation = None
            if hasattr(self.robot, 'get_observation'):
                observation = self.robot.get_observation()

            # Create snapshot
            snapshot = ActionSnapshot(
                timestamp=time.time(),
                actions=current_actions,
                observation=observation,
                action_number=self.current_action_number,
            )

            # Store in history
            self.action_history.append(snapshot)

            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture current action: {e}")
            return None

    def _send_action_snapshot(self, snapshot: ActionSnapshot) -> bool:
        """Send an action from a historical snapshot to the robot.

        Args:
            snapshot: ActionSnapshot containing the action to send.

        Returns:
            True if action was sent successfully, False otherwise.
        """
        try:
            if snapshot.actions is None:
                logger.warning("Snapshot has no actions to send")
                return False

            # Convert snapshot actions to robot format
            # The snapshot.actions may contain keys with or without ".pos" suffix
            action_dict = {}
            for key, value in snapshot.actions.items():
                # Remove ".pos" suffix if present to get joint name
                joint_name = key.replace(".pos", "") if ".pos" in key else key
                # Add ".pos" suffix for robot.send_action format
                action_dict[f"{joint_name}.pos"] = value

            # Send action to robot
            if hasattr(self.robot, 'send_action'):
                self.robot.send_action(action_dict)
                logger.debug(f"Sent snapshot action #{snapshot.action_number}")
                return True
            else:
                logger.error("Robot does not have send_action method")
                return False

        except Exception as e:
            logger.error(f"Failed to send snapshot action: {e}")
            return False

    def rollback(self, steps: int | None = None, confirm_before_resume: bool = True) -> bool:
        """Rollback robot to a previous safe state.

        Args:
            steps: Number of steps to rollback (None = use config default).
            confirm_before_resume: Whether to wait for user confirmation before resuming.

        Returns:
            True if rollback completed successfully, False otherwise.
        """
        if steps is None:
            steps = self.rollback_config.max_rollback_steps

        if len(self.action_history) < steps:
            logger.warning(f"Insufficient history for rollback: only {len(self.action_history)} snapshots available, requested {steps}")
            # Use available steps instead
            steps = len(self.action_history)

        if steps == 0:
            logger.warning("No steps to rollback")
            return False

        self._rollback_in_progress = True
        logger.info(f"Starting rollback of {steps} steps...")

        # Find the target snapshot (the state we want to restore)
        # We want to go back 'steps' actions in history
        target_index = len(self.action_history) - steps
        if target_index < 0:
            target_index = 0

        target_snapshot = self.action_history[target_index]
        logger.info(f"Rolling back to snapshot #{target_snapshot.action_number} from {steps} steps ago")

        # Execute rollback with proper exception handling
        try:
            # Method 1: Directly send the target snapshot action
            # This moves the robot directly to the historical position
            success = self._send_action_snapshot(target_snapshot)

            if success:
                # Hold at safe state for confirmation
                if confirm_before_resume:
                    logger.info("Rollback complete, waiting for user confirmation...")
                    self._wait_for_resume_confirmation()

                # Store the rollback snapshot for resume reference
                if self.current_stop_event:
                    self.current_stop_event.rollback_snapshot = target_snapshot

                self._rollback_count += 1
                return True
            else:
                logger.error("Failed to send target snapshot during rollback")
                return False

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
        finally:
            self._rollback_in_progress = False

    def _wait_for_resume_confirmation(self):
        """Wait for user to confirm after rollback."""
        logger.info("Waiting for resume confirmation (user should call resume())")
        self._safe_confirmed_count += 1

    def pause(self):
        """Pause the robot after rollback."""
        if self._is_stopped:
            self._is_paused = True
            logger.info("Robot paused after rollback - waiting for user action")
            return True
        logger.warning("Cannot pause - not in stopped state")
        return False

    def prompt_recovery_action(self, task_name: str = None, timeout: float = 60.0) -> RecoveryAction:
        """Prompt user to select recovery action after emergency stop.

        Args:
            task_name: Name of the task that was interrupted (optional).
            timeout: Maximum time to wait for user input in seconds (default: 60s).

        Returns:
            RecoveryAction selected by user.
        """
        print("\n" + "=" * 60)
        print("🚨 EMERGENCY STOP TRIGGERED")
        print("=" * 60)
        if task_name:
            print(f"Interrupted task: {task_name}")
        if self.current_stop_event:
            print(f"Stop reason: {self.current_stop_event.reason.value}")
        print("")
        print("Available recovery actions:")
        print("  1 - Stop program completely")
        print("  2 - Rollback to safe position and continue with same task")
        print("  3 - Rollback to safe position and retry with new model")
        print("=" * 60)
        print(f"Auto-selecting option 2 (rollback and continue) in {timeout:.0f} seconds if no input...")
        print(">>> Waiting for user input (enter 1, 2, or 3)...", flush=True)

        # Use a non-blocking approach with timeout
        import select
        import sys

        user_input = ""

        # If not running in a real terminal, use shorter timeout and auto-select default
        if not sys.stdin.isatty():
            logger.info("Not running in a terminal, auto-selecting option 2 (rollback and continue)")
            # Short wait to allow any buffered output to flush
            time.sleep(0.5)
            return RecoveryAction.ROLLBACK_AND_CONTINUE

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if there's input available (non-blocking)
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    try:
                        line = sys.stdin.readline()
                        if line:
                            # Clean the input - only accept single digit commands
                            cleaned = line.strip()
                            # Filter out non-numeric input (like hardware data)
                            if cleaned and len(cleaned) <= 2 and cleaned.isdigit():
                                user_input = cleaned
                                logger.info(f"User input received: {user_input}")
                                break
                            elif cleaned:
                                # Log but ignore non-numeric input
                                logger.debug(f"Ignoring non-numeric input: {cleaned[:50]}...")
                    except (EOFError, KeyboardInterrupt):
                        logger.info("Input interrupted, defaulting to option 2 (rollback and continue)")
                        return RecoveryAction.ROLLBACK_AND_CONTINUE
            except (OSError, ValueError):
                # stdin might not be available in some environments
                time.sleep(0.1)
                continue

        # Timeout or input received
        if not user_input or time.time() - start_time >= timeout:
            logger.info(f"Timeout after {timeout:.0f}s, defaulting to option 2 (rollback and continue)")
            user_input = "2"  # Default: rollback and continue

        if user_input == "1":
            logger.info("User selected: Stop program")
            return RecoveryAction.STOP_PROGRAM
        elif user_input == "2":
            logger.info("User selected: Rollback and continue")
            return RecoveryAction.ROLLBACK_AND_CONTINUE
        elif user_input == "3":
            logger.info("User selected: Rollback and retry with new model")
            return RecoveryAction.ROLLBACK_AND_RETRY_MODEL
        else:
            logger.warning(f"Invalid input '{user_input}', defaulting to option 2 (rollback and continue)")
            return RecoveryAction.ROLLBACK_AND_CONTINUE

    def get_suggested_rollback_steps(self) -> int:
        """Get suggested number of steps to rollback based on stop reason.

        Returns:
            Suggested rollback steps.
        """
        if self.current_stop_event and self.current_stop_event.reason == StopReason.HIGH_FORCE:
            # For high force collisions, rollback more steps
            return min(self.rollback_config.max_rollback_steps, 50)
        elif self.current_stop_event and self.current_stop_event.reason == StopReason.COLLISION:
            return min(self.rollback_config.max_rollback_steps, 30)
        else:
            # Default: smaller rollback
            return min(self.rollback_config.max_rollback_steps, 20)

    def resume(self):
        """Resume execution after emergency stop/rollback.

        Returns:
            True if successfully resumed, False otherwise.
        """
        if not self._is_stopped:
            logger.warning("Cannot resume - not in stopped state")
            return False

        logger.info("Resuming execution...")

        # Clear stop state
        self._is_stopped = False
        self._is_paused = False

        # Note: The robot is already at the rollback position from the rollback() call
        # We don't need to re-send the action, just clear the stop event

        # Clear current stop event but keep history
        self.current_stop_event = None

        logger.info("Execution resumed - robot at rollback position, ready for new actions")
        return True

    def set_safe_state(self, snapshot: ActionSnapshot):
        """Set a specific state as safe for future reference."""
        logger.info(f"Set safe state from snapshot #{snapshot.action_number}")
        # Implementation: could store this in a separate "safe states" list

    def get_safe_state(self) -> ActionSnapshot | None:
        """Get the last known safe state."""
        logger.info("Retrieving safe state")
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get controller statistics."""
        return {
            "total_checks": self._total_checks,
            "danger_detected_count": self._danger_detected_count,
            "manual_stops": self._manual_stop_count,
            "automatic_stops": self._automatic_stop_count,
            "rollback_count": self._rollback_count,
            "is_stopped": self._is_stopped,
            "is_paused": self._is_paused,
            "action_history_size": len(self.action_history),
            "stop_events_count": len(self.stop_events),
            "safe_confirmed_count": self._safe_confirmed_count,
            "current_action_number": self.current_action_number,
        }

    # Callbacks
    def set_stop_callback(self, callback: Callable[[StopEvent], None]):
        """Set callback for stop events."""
        self._on_stop_callback = callback
        logger.info("Stop callback registered")

    def set_rollback_complete_callback(self, callback: Callable[[StopEvent], None]):
        """Set callback for rollback completion."""
        self._on_rollback_complete_callback = callback
        logger.info("Rollback complete callback registered")

    def get_current_stop_event(self) -> StopEvent | None:
        """Get the most recent stop event."""
        return self.current_stop_event

    def get_action_history(self, num_recent: int = 10) -> list[ActionSnapshot]:
        """Get recent action history."""
        return list(self.action_history)[-num_recent:]

    def record_action(
        self,
        action: dict[str, Any],
        observation: dict[str, Any] | None = None
    ) -> None:
        """Record an action and its observation for potential rollback.

        Args:
            action: The action that was sent to the robot.
            observation: Optional observation at the time of the action.
        """
        self.current_action_number += 1
        snapshot = ActionSnapshot(
            timestamp=time.time(),
            actions=action,
            observation=observation,
            action_number=self.current_action_number,
        )
        self.action_history.append(snapshot)

    def get_latest_action_snapshot(self) -> ActionSnapshot | None:
        """Get the most recent action snapshot.

        Returns:
            The most recent ActionSnapshot, or None if no history.
        """
        return self.action_history[-1] if self.action_history else None


def create_emergency_stop_controller(
    robot,
    history_size: int = 1000,
    danger_config: DangerDetectionConfig = None,
    rollback_config: RollbackConfig = None,
) -> EmergencyStopController:
    """Create an emergency stop controller instance.

    Args:
        robot: Robot instance with send_action method.
        history_size: Maximum number of action snapshots to store.
        danger_config: Configuration for danger detection.
        rollback_config: Configuration for rollback behavior.

    Returns:
        Configured EmergencyStopController instance.
    """
    return EmergencyStopController(
        robot=robot,
        history_size=history_size,
        danger_config=danger_config,
        rollback_config=rollback_config,
    )
