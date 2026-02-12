"""
Manual Intervention Controller for Robot Task Execution

This module provides human intervention capabilities during task execution:
1. Pause robot motion on human request
2. Keep program running (not terminate)
3. Allow user to choose: rollback, skip, continue, select new task
4. Resume robot motion after intervention

Usage:
    controller = ManualInterventionController(robot)
    if controller.check_intervention_needed(observation, action):
        controller.interrupt_robot()
        user_choice = controller.prompt_intervention_menu()
        controller.handle_user_choice(user_choice)
"""


import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from lerobot.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


class InterventionReason(Enum):
    """Reason for human intervention."""
    IMMINENT_COLLISION = "imminent_collision"  # Human detects imminent collision
    UNSAFE_MOTION = "unsafe_motion"  # Motion looks unsafe
    WRONG_TASK = "wrong_task"  # Wrong task being executed
    MANUAL_PAUSE = "manual_pause"  # Manual pause request


class InterventionAction(Enum):
    """User action after intervention."""
    ROLLBACK = "rollback"  # Rollback to previous safe state
    SKIP_TASK = "skip_task"  # Skip current task
    CONTINUE_TASK = "continue"  # Continue with current task
    SELECT_NEW_TASK = "select_new"  # Select a different task
    EXIT_PROGRAM = "exit"  # Exit the program


@dataclass
class InterventionState:
    """State of the intervention controller."""
    is_paused: bool = False  # Robot is paused
    waiting_for_user: bool = False  # Waiting for user input
    intervention_count: int = 0  # Total interventions
    last_intervention_reason: InterventionReason | None = None  # Last reason
    last_intervention_time: float | None = None  # Timestamp of last intervention


@dataclass
class InterventionConfig:
    """Configuration for intervention behavior."""
    enable_force_detection: bool = True  # Auto-detect high force
    force_threshold: float = 2.5  # Force threshold (Nm)
    force_detection_window: int = 3  # Steps above threshold

    enable_velocity_detection: bool = True
    max_velocity: float = 5.0  # Max velocity (rad/s)

    rollback_steps: int = 50  # Default rollback steps
    rollback_step_delay: float = 0.033  # Delay between rollback steps (30Hz)

    enable_manual_pause: bool = True  # Allow manual pause (Ctrl+C)


class ManualInterventionController:
    """Manual intervention controller for human-in-the-loop robot control.

    This controller allows human intervention during task execution WITHOUT
    terminating the program. The robot can be paused, rolled back,
    or the user can select a different task to execute.

    Key features:
    - Pause robot motion immediately
    - Maintain program execution state
    - Interactive menu for user decisions
    - Rollback to previous safe state
    - Skip to next task
    - Select new task
    """

    def __init__(
        self,
        robot,
        config: InterventionConfig | None = None,
    ):
        """Initialize the manual intervention controller.

        Args:
            robot: Robot instance with send_action method.
            config: Intervention configuration (optional).
        """
        self.robot = robot
        self.config = config or InterventionConfig()

        # State tracking
        self.state = InterventionState(
            is_paused=False,
            waiting_for_user=False,
            intervention_count=0,
            last_intervention_reason=None,
            last_intervention_time=None,
        )

        # Action history for rollback
        self.action_history: deque = deque(maxlen=1000)
        self.action_number: int = 0

        # Task list for user selection
        self.available_tasks: list[TaskConfig] = []

        # Current task being executed
        self.current_task: TaskConfig | None = None

        logger.info("ManualInterventionController initialized")

    def set_tasks(self, tasks: list[TaskConfig]) -> None:
        """Set available tasks for user selection.

        Args:
            tasks: List of available TaskConfigs.
        """
        self.available_tasks = tasks
        logger.info(f"Set {len(tasks)} tasks for intervention selection")

    def check_intervention_needed(
        self,
        observation: dict[str, Any],
        action: dict[str, float],
    ) -> tuple[bool, InterventionReason | None]:
        """Check if human intervention is needed.

        Args:
            observation: Current observation from robot.
            action: Current action being executed.

        Returns:
            (needs_intervention, reason) tuple.
        """
        # Check 1: Force detection
        if self.config.enable_force_detection:
            is_high_force, reason = self._check_force_threshold(observation)
            if is_high_force:
                self.state.intervention_count += 1
                return True, reason

        # Check 2: Velocity detection
        if self.config.enable_velocity_detection:
            is_high_velocity, reason = self._check_velocity_threshold(observation, action)
            if is_high_velocity:
                self.state.intervention_count += 1
                return True, reason

        # No intervention needed
        return False, None

    def _check_force_threshold(
        self,
        observation: dict[str, Any],
    ) -> tuple[bool, InterventionReason]:
        """Check if force exceeds threshold."""
        forces = []
        for key, value in observation.items():
            if ".force" in key:
                try:
                    forces.append(abs(float(value)))
                except (ValueError, TypeError):
                    pass

        if not forces:
            return False, None

        max_force = max(forces)
        total_force = sum(forces)

        if max_force > self.config.force_threshold:
            logger.warning(f"High force detected: {max_force:.2f} Nm > {self.config.force_threshold} Nm")
            return True, InterventionReason.IMMINENT_COLLISION

        if total_force > (self.config.force_threshold * 2):
            logger.warning(f"High total force: {total_force:.2f} Nm")
            return True, InterventionReason.IMMINENT_COLLISION

        return False, None

    def _check_velocity_threshold(
        self,
        observation: dict[str, Any],
        action: dict[str, float],
    ) -> tuple[bool, InterventionReason]:
        """Check if velocity exceeds threshold."""
        velocities = []
        for key, value in observation.items():
            if ".velocity" in key:
                try:
                    velocities.append(abs(float(value)))
                except (ValueError, TypeError):
                    pass

        if not velocities:
            return False, None

        max_velocity = max(velocities)
        if max_velocity > self.config.max_velocity:
            logger.warning(f"High velocity detected: {max_velocity:.2f} rad/s > {self.config.max_velocity} rad/s")
            return True, InterventionReason.UNSAFE_MOTION

        return False, None

    def interrupt_robot(self) -> None:
        """Immediately pause/stop the robot.

        This stops robot motion but keeps the program running.
        """
        if self.state.is_paused:
            logger.warning("Robot already paused")
            return

        logger.warning("🤚️ INTERRUPTING ROBOT - Human intervention triggered")

        # Try to stop the robot
        if hasattr(self.robot, 'stop'):
            try:
                self.robot.stop()
                logger.info("Robot.stop() called")
            except Exception as e:
                logger.error(f"Error calling robot.stop(): {e}")

        # Try to send zero actions
        if hasattr(self.robot, 'send_action'):
            zero_action = {}
            for key in action.keys() if action else []:
                zero_action[key] = 0.0
            try:
                self.robot.send_action(zero_action)
                logger.info("Sent zero action to stop robot")
            except Exception as e:
                logger.error(f"Error sending zero action: {e}")

        # Update state
        self.state.is_paused = True
        self.state.waiting_for_user = True
        self.state.last_intervention_time = time.time()

        logger.info("Robot paused, waiting for user input...")

    def prompt_intervention_menu(
        self,
        reason: InterventionReason,
        observation: dict[str, Any] | None = None,
    ) -> InterventionAction:
        """Display interactive menu and get user choice.

        Args:
            reason: Reason for the intervention.
            observation: Current observation (for display).

        Returns:
            User's selected action.
        """
        print("\n" + "=" * 70)
        print("🤚️ HUMAN INTERVENTION REQUIRED")
        print("=" * 70)
        print(f"Reason: {reason.value}")
        print(f"Current task: {self.current_task.name if self.current_task else 'None'}")

        # Show current state if available
        if observation:
            forces = []
            for key, value in observation.items():
                if ".force" in key:
                    try:
                        forces.append(f"{key}: {abs(float(value)):.2f} Nm")
                    except (ValueError, TypeError):
                        pass
            if forces:
                print(f"\nCurrent forces:")
                for f in forces[:5]:  # Show first 5
                    print(f"  {f}")
                if len(forces) > 5:
                    print(f"  ... ({len(forces) - 5} more)")

        print("\n" + "-" * 70)
        print("Available actions:")
        print("  1. Rollback and RETRY")
        print("  2. Skip current task")
        print("  3. Continue execution (CAUTION)")
        print("  4. Select different task")
        print("  0. Exit program")
        print("-" * 70)

        while True:
            try:
                user_input = input("Select action [0-4]: ").strip()

                if not user_input:
                    user_input = "1"  # Default: rollback

                # Parse input
                if user_input == "0":
                    return InterventionAction.EXIT_PROGRAM
                elif user_input == "1":
                    return InterventionAction.ROLLBACK
                elif user_input == "2":
                    return InterventionAction.SKIP_TASK
                elif user_input == "3":
                    return InterventionAction.CONTINUE_TASK
                elif user_input == "4":
                    return self._prompt_task_selection()
                else:
                    print("Invalid input, please enter 0-4")

            except (KeyboardInterrupt, EOFError):
                print("\nUser cancelled, defaulting to ROLLBACK")
                return InterventionAction.ROLLBACK

    def _prompt_task_selection(self) -> InterventionAction:
        """Prompt user to select a new task."""
        if not self.available_tasks:
            print("No tasks available!")
            return InterventionAction.CONTINUE_TASK

        print("\nAvailable tasks:")
        for i, task in enumerate(self.available_tasks):
            current = " <- CURRENT" if task == self.current_task else ""
            print(f"  {i+1}. {task.name}{current}")

        while True:
            try:
                user_input = input(f"Select task [1-{len(self.available_tasks)}]: ").strip()

                if not user_input:
                    print("Invalid input, please try again")
                    continue

                try:
                    task_index = int(user_input) - 1
                    if 0 <= task_index < len(self.available_tasks):
                        self._selected_task_index = task_index
                        return InterventionAction.SELECT_NEW_TASK
                    else:
                        print(f"Invalid task number, please enter 1-{len(self.available_tasks)}")
                except ValueError:
                    print("Please enter a valid number")

            except (KeyboardInterrupt, EOFError):
                print("\nCancelled, returning to main menu")
                return InterventionAction.CONTINUE_TASK

    def handle_user_choice(
        self,
        choice: InterventionAction,
    ) -> dict[str, Any]:
        """Handle user's intervention choice.

        Args:
            choice: User's selected action.

        Returns:
            Result dictionary with action to take.
        """
        result = {
            "action": choice.value,
            "resume_execution": False,
            "new_task_index": None,
            "skip_task": False,
        }

        if choice == InterventionAction.EXIT_PROGRAM:
            logger.info("User chose to exit program")
            result["exit_program"] = True

        elif choice == InterventionAction.ROLLBACK:
            logger.info("User chose to ROLLBACK and retry")
            success = self._execute_rollback()
            if success:
                result["resume_execution"] = True

        elif choice == InterventionAction.SKIP_TASK:
            logger.info("User chose to SKIP current task")
            result["skip_task"] = True
            result["resume_execution"] = True

        elif choice == InterventionAction.CONTINUE_TASK:
            logger.info("User chose to CONTINUE (CAUTION)")
            result["resume_execution"] = True

        elif choice == InterventionAction.SELECT_NEW_TASK:
            if hasattr(self, '_selected_task_index'):
                logger.info(f"User selected task index: {self._selected_task_index}")
                result["new_task_index"] = self._selected_task_index
                result["resume_execution"] = True

        return result

    def _execute_rollback(self) -> bool:
        """Execute rollback to previous safe state.

        Returns:
            True if rollback successful, False otherwise.
        """
        rollback_steps = min(self.config.rollback_steps, len(self.action_history))

        if rollback_steps < 5:
            logger.warning(f"Insufficient history for rollback: only {len(self.action_history)} snapshots")
            return False

        logger.info(f"Rolling back {rollback_steps} steps...")

        try:
            # Rollback by replaying previous actions in reverse
            for i in range(rollback_steps):
                if len(self.action_history) > 0:
                    snapshot = self.action_history.pop()  # Remove from history
                    # Replay the action
                    if hasattr(self.robot, 'send_action'):
                        self.robot.send_action(snapshot.actions)
                        logger.debug(f"Rollback step {i+1}/{rollback_steps}: action #{snapshot.action_number}")

                    # Delay between steps
                    if i < rollback_steps - 1:
                        time.sleep(self.config.rollback_step_delay)

            logger.info("Rollback complete")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def resume_execution(self) -> bool:
        """Resume robot execution after intervention.

        Returns:
            True if resume successful, False otherwise.
        """
        if not self.state.is_paused:
            logger.warning("Robot not paused, nothing to resume")
            return True  # Not paused, so "running"

        logger.info("Resuming robot execution...")

        # Try to resume robot
        if hasattr(self.robot, 'resume'):
            try:
                self.robot.resume()
                logger.info("Robot.resume() called")
            except Exception as e:
                logger.error(f"Error calling robot.resume(): {e}")

        # Update state
        self.state.is_paused = False
        self.state.waiting_for_user = False

        logger.info("Robot execution resumed")
        return True

    def record_action(self, action: dict[str, float], observation: dict[str, Any] | None = None) -> None:
        """Record an action for potential rollback.

        Args:
            action: The action being executed.
            observation: Current observation (optional).
        """
        self.action_number += 1

        # Store snapshot
        from lerobot.safety.emergency_stop_controller import ActionSnapshot
        snapshot = ActionSnapshot(
            timestamp=time.time(),
            actions=action.copy(),
            observation=observation,
            action_number=self.action_number,
        )
        self.action_history.append(snapshot)

    def get_statistics(self) -> dict[str, Any]:
        """Get controller statistics."""
        return {
            "is_paused": self.state.is_paused,
            "waiting_for_user": self.state.waiting_for_user,
            "intervention_count": self.state.intervention_count,
            "last_intervention_reason": self.state.last_intervention_reason.value if self.state.last_intervention_reason else None,
            "last_intervention_time": self.state.last_intervention_time,
            "action_history_size": len(self.action_history),
        }


def create_manual_intervention_controller(
    robot,
    config: InterventionConfig | None = None,
) -> ManualInterventionController:
    """Create a manual intervention controller instance.

    Args:
        robot: Robot instance with send_action method.
        config: Intervention configuration (optional).

    Returns:
        Configured ManualInterventionController instance.
    """
    return ManualInterventionController(
        robot=robot,
        config=config,
    )
