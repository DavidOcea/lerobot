"""
Task scheduler for multi-step robotic task execution.

This module provides:
- Task sequencing and state management
- Policy switching between tasks
- Task execution with retry logic
- Integration with collision detection and completion detection
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .completion_detector import CompletionCriteria, DetectionResult, TaskCompletionDetector
from .config import TaskConfig

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot
    from lerobot.scripts.server.robot_client import RobotClient

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Possible states for a task during execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    FATAL_FAILURE = "fatal_failure"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a single task execution."""

    task_name: str
    status: TaskStatus
    attempts: int = 0
    duration: float = 0.0
    success: bool = False
    error_message: str = ""
    completion_confidence: float = 0.0
    final_observation: dict[str, Any] = field(default_factory=dict)
    collision_detected: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def __post_init__(self):
        if self.end_time == 0:
            self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class ExecutionSummary:
    """Summary of complete task sequence execution."""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    total_duration: float
    task_results: list[TaskResult]
    overall_success: bool
    collision_count: int = 0
    total_retries: int = 0


class TaskScheduler:
    """Manages task execution sequence and policy switching.

    The scheduler coordinates:
    - Loading and switching between ACT policies
    - Executing tasks with timeout and retry logic
    - Monitoring for task completion
    - Handling collision events

    Usage:
        scheduler = TaskScheduler(tasks, robot, policy_client)
        summary = scheduler.execute_sequence()
        for result in summary.task_results:
            print(f"{result.task_name}: {result.status}")
    """

    def __init__(
        self,
        tasks: list[TaskConfig],
        robot: "Robot | RobotClient",
        policy_client: "RobotClient | None" = None,
        completion_detector: TaskCompletionDetector | None = None,
    ):
        """Initialize the task scheduler.

        Args:
            tasks: List of task configurations to execute.
            robot: Robot instance or RobotClient for executing actions.
            policy_client: gRPC policy client for getting actions (can be same as robot if RobotClient).
            completion_detector: Optional completion detector for early termination.
        """
        self.tasks = tasks
        self.robot = robot
        self.policy_client = policy_client if policy_client is not None else robot
        self.completion_detector = completion_detector

        # Execution state
        self.current_policy_path: str | None = None
        self.action_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=100)

        # Statistics
        self._total_task_executions = 0

    def execute_sequence(
        self,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> ExecutionSummary:
        """Execute all tasks in sequence.

        Args:
            collision_detector: Optional collision detector for safety monitoring.
            collision_handler: Optional collision handler for recovery.
            state_monitor: Optional state monitor for logging.

        Returns:
            ExecutionSummary with results for all tasks.
        """
        start_time = time.time()
        task_results: list[TaskResult] = []

        collision_count = 0
        total_retries = 0

        logger.info(f"Starting task sequence with {len(self.tasks)} tasks")

        for i, task in enumerate(self.tasks):
            if not task.enabled:
                logger.info(f"Skipping disabled task: {task.name}")
                task_results.append(
                    TaskResult(
                        task_name=task.name,
                        status=TaskStatus.SKIPPED,
                        attempts=0,
                        success=False,
                    )
                )
                continue

            logger.info(f"Executing task {i + 1}/{len(self.tasks)}: {task.name}")

            # Execute task with safety
            result = self.execute_task_with_safety(
                task,
                collision_detector=collision_detector,
                collision_handler=collision_handler,
                state_monitor=state_monitor,
            )

            task_results.append(result)
            total_retries += result.attempts - 1  # Subtract 1 for initial attempt

            if result.collision_detected:
                collision_count += 1

            # Check for fatal failure
            if result.status == TaskStatus.FATAL_FAILURE:
                logger.error(f"Fatal failure in task {task.name}, stopping execution")
                break

        # Compute summary
        end_time = time.time()
        completed = sum(1 for r in task_results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in task_results if r.status == TaskStatus.FAILED)
        skipped = sum(1 for r in task_results if r.status == TaskStatus.SKIPPED)

        summary = ExecutionSummary(
            total_tasks=len(self.tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            total_duration=end_time - start_time,
            task_results=task_results,
            overall_success=(failed == 0 and completed > 0),
            collision_count=collision_count,
            total_retries=total_retries,
        )

        logger.info(
            f"Task sequence complete: {completed}/{len(self.tasks)} tasks completed, "
            f"{failed} failed, {skipped} skipped"
        )

        return summary

    def execute_task_with_safety(
        self,
        task: TaskConfig,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> TaskResult:
        """Execute a single task with safety monitoring and retry logic.

        Args:
            task: Task configuration to execute.
            collision_detector: Optional collision detector.
            collision_handler: Optional collision handler.
            state_monitor: Optional state monitor.

        Returns:
            TaskResult with execution outcome.
        """
        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.PENDING,
            start_time=time.time(),
        )

        for attempt in range(task.max_retries):
            result.attempts = attempt + 1

            logger.info(f"Attempt {attempt + 1}/{task.max_retries} for task: {task.name}")

            # Switch to task's policy
            if not self.switch_policy(task.policy_path, task.policy_type):
                result.status = TaskStatus.FATAL_FAILURE
                result.error_message = "Failed to switch policy"
                return result

            # Execute single attempt
            attempt_result = self.execute_single_attempt(
                task,
                collision_detector=collision_detector,
                collision_handler=collision_handler,
                state_monitor=state_monitor,
            )

            # Update result with attempt outcome
            result.collision_detected = attempt_result.get("collision_detected", False)
            result.final_observation = attempt_result.get("final_observation", {})
            result.completion_confidence = attempt_result.get("completion_confidence", 0.0)

            if attempt_result.get("success", False):
                result.status = TaskStatus.COMPLETED
                result.success = True
                logger.info(f"Task {task.name} completed successfully")
                return result

            # Check if retry is appropriate
            if attempt_result.get("fatal", False):
                result.status = TaskStatus.FATAL_FAILURE
                result.error_message = attempt_result.get("error", "Fatal error")
                return result

            if attempt < task.max_retries - 1:
                logger.warning(
                    f"Task {task.name} attempt {attempt + 1} failed: "
                    f"{attempt_result.get('error', 'Unknown error')}. Retrying..."
                )
                time.sleep(1.0)  # Brief pause before retry

        # All retries exhausted
        result.status = TaskStatus.FAILED
        result.error_message = f"Failed after {task.max_retries} attempts"
        result.end_time = time.time()

        return result

    def execute_single_attempt(
        self,
        task: TaskConfig,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> dict[str, Any]:
        """Execute a single attempt of a task.

        Args:
            task: Task configuration.
            collision_detector: Optional collision detector.
            collision_handler: Optional collision handler.
            state_monitor: Optional state monitor.

        Returns:
            Dictionary with attempt result.
        """
        start_time = time.time()
        timeout = start_time + task.max_duration

        result = {
            "success": False,
            "fatal": False,
            "collision_detected": False,
            "error": "",
            "final_observation": {},
            "completion_confidence": 0.0,
        }

        # Clear action queue
        self._clear_action_queue()

        try:
            while time.time() < timeout:
                # Get current observation
                try:
                    # Use RobotClient's method if available, otherwise use robot's method
                    if hasattr(self.robot, "get_latest_observation"):
                        observation = self.robot.get_latest_observation()
                    elif hasattr(self.robot, "robot") and hasattr(self.robot.robot, "get_observation"):
                        observation = self.robot.robot.get_observation()
                    else:
                        observation = self.robot.get_observation()
                except Exception as e:
                    result["fatal"] = True
                    result["error"] = f"Failed to get observation: {e}"
                    return result

                # Update monitor
                if state_monitor is not None:
                    state_monitor.update(observation, {})

                # Check for collision
                if collision_detector is not None:
                    collision_result = collision_detector.check_collision(observation)
                    if collision_result.is_detected:
                        result["collision_detected"] = True

                        if collision_handler is not None:
                            handler_result = collision_handler.handle_collision(
                                collision_result, observation
                            )

                            if not handler_result.can_continue:
                                result["fatal"] = True
                                result["error"] = "Collision - cannot continue"
                                return result

                        # Short pause after collision
                        time.sleep(0.5)
                        continue

                # Check for completion
                if self.completion_detector is not None:
                    detection = self.completion_detector.detect(observation, list(self.action_history))
                    if detection.is_completed:
                        result["success"] = True
                        result["completion_confidence"] = detection.confidence
                        result["final_observation"] = observation
                        return result

                # Request and wait for action
                action = self._get_next_action(observation)
                if action is None:
                    result["error"] = "Failed to get action from policy"
                    return result

                # Send action to robot
                try:
                    self.robot.send_action(action)
                    self.action_history.append(action)
                    self.observation_history.append(observation)
                except Exception as e:
                    result["error"] = f"Failed to send action: {e}"
                    return result

                # Maintain control frequency
                time.sleep(0.01)  # ~100Hz control loop

            # Timeout
            result["error"] = f"Task timeout after {task.max_duration}s"

        except Exception as e:
            result["fatal"] = True
            result["error"] = f"Exception during execution: {e}"
            logger.exception("Exception during task execution")

        return result

    def switch_policy(self, policy_path: str, policy_type: str = "act") -> bool:
        """Switch to a different policy.

        Args:
            policy_path: Path to the policy model.
            policy_type: Type of policy (e.g., "act", "diffusion").

        Returns:
            True if switch was successful, False otherwise.
        """
        if self.current_policy_path == policy_path:
            return True  # Already loaded

        try:
            # Import here to avoid circular imports
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

            # Load policy based on type
            if policy_type == "act":
                policy = ACTPolicy.from_pretrained(policy_path)
            elif policy_type == "diffusion":
                policy = DiffusionPolicy.from_pretrained(policy_path)
            else:
                logger.error(f"Unknown policy type: {policy_type}")
                return False

            # Send policy to server via gRPC
            self.policy_client.send_policy_instructions(
                policy_path=policy_path,
                policy_type=policy_type,
            )

            self.current_policy_path = policy_path
            logger.info(f"Switched to policy: {policy_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch policy: {e}")
            return False

    def _get_next_action(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """Get the next action from the policy server.

        Args:
            observation: Current observation dict.

        Returns:
            Action dict or None if failed.
        """
        try:
            # Send observation
            self.policy_client.send_observation(observation)

            # Get action (blocking)
            action = self.policy_client.get_action(timeout=5.0)

            return action

        except Exception as e:
            logger.error(f"Failed to get action: {e}")
            return None

    def _clear_action_queue(self):
        """Clear any pending actions in the queue."""
        self.action_history.clear()
        self.observation_history.clear()

    def get_current_state(self) -> dict[str, Any]:
        """Get the current scheduler state."""
        return {
            "current_policy": self.current_policy_path,
            "action_queue_size": len(self.action_history),
            "total_executions": self._total_task_executions,
        }


class LocalTaskScheduler:
    """Task scheduler for local execution mode (no Policy Server needed).

    This scheduler executes policies locally on the same machine as the robot,
    eliminating network latency and the need for a separate Policy Server process.

    Usage:
        scheduler = LocalTaskScheduler(
            tasks=tasks,
            robot=robot,
            policy_executor=local_executor,
        )
        summary = scheduler.execute_sequence()
    """

    def __init__(
        self,
        tasks: list[TaskConfig],
        robot: "Robot",
        policy_executor: "LocalPolicyExecutor",
        completion_detector: TaskCompletionDetector | None = None,
    ):
        """Initialize the local task scheduler.

        Args:
            tasks: List of task configurations to execute.
            robot: Robot instance for executing actions.
            policy_executor: LocalPolicyExecutor for inference.
            completion_detector: Optional completion detector for early termination.
        """
        from .local_policy_executor import LocalPolicyExecutor

        self.tasks = tasks
        self.robot = robot
        self.policy_executor: LocalPolicyExecutor = policy_executor
        self.completion_detector = completion_detector

        # Execution state
        self.current_policy_path: str | None = None
        self.action_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.observation_history: deque[dict[str, Any]] = deque(maxlen=100)

        # Statistics
        self._total_task_executions = 0

    def execute_sequence(
        self,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> ExecutionSummary:
        """Execute all tasks in sequence.

        Args:
            collision_detector: Optional collision detector for safety monitoring.
            collision_handler: Optional collision handler for recovery.
            state_monitor: Optional state monitor for logging.

        Returns:
            ExecutionSummary with results for all tasks.
        """
        start_time = time.time()
        task_results: list[TaskResult] = []

        collision_count = 0
        total_retries = 0

        logger.info(f"Starting task sequence with {len(self.tasks)} tasks")

        for i, task in enumerate(self.tasks):
            if not task.enabled:
                logger.info(f"Skipping disabled task: {task.name}")
                task_results.append(
                    TaskResult(
                        task_name=task.name,
                        status=TaskStatus.SKIPPED,
                        attempts=0,
                        success=False,
                    )
                )
                continue

            logger.info(f"Executing task {i + 1}/{len(self.tasks)}: {task.name}")

            # Set completion detector
            self.completion_detector = completion_detector

            # Execute task with safety
            result = self.execute_task_with_safety(
                task,
                collision_detector=collision_detector,
                collision_handler=collision_handler,
                state_monitor=state_monitor,
            )

            task_results.append(result)
            total_retries += result.attempts - 1

            if result.collision_detected:
                collision_count += 1

            # Check for fatal failure
            if result.status == TaskStatus.FATAL_FAILURE:
                logger.error(f"Fatal failure in task {task.name}, stopping execution")
                break

        # Compute summary
        end_time = time.time()
        completed = sum(1 for r in task_results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in task_results if r.status == TaskStatus.FAILED)
        skipped = sum(1 for r in task_results if r.status == TaskStatus.SKIPPED)

        summary = ExecutionSummary(
            total_tasks=len(self.tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            total_duration=end_time - start_time,
            task_results=task_results,
            overall_success=(failed == 0 and completed > 0),
            collision_count=collision_count,
            total_retries=total_retries,
        )

        logger.info(
            f"Task sequence complete: {completed}/{len(self.tasks)} tasks completed, "
            f"{failed} failed, {skipped} skipped"
        )

        return summary

    def execute_task_with_safety(
        self,
        task: TaskConfig,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> TaskResult:
        """Execute a single task with safety monitoring and retry logic.

        Args:
            task: Task configuration to execute.
            collision_detector: Optional collision detector.
            collision_handler: Optional collision handler.
            state_monitor: Optional state monitor.

        Returns:
            TaskResult with execution outcome.
        """
        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.PENDING,
            start_time=time.time(),
        )

        for attempt in range(task.max_retries):
            result.attempts = attempt + 1

            logger.info(f"Attempt {attempt + 1}/{task.max_retries} for task: {task.name}")

            # Load policy for this task
            if not self._load_policy_for_task(task):
                result.status = TaskStatus.FATAL_FAILURE
                result.error_message = "Failed to load policy"
                return result

            # Execute single attempt
            attempt_result = self.execute_single_attempt(
                task,
                collision_detector=collision_detector,
                collision_handler=collision_handler,
                state_monitor=state_monitor,
            )

            # Update result
            result.collision_detected = attempt_result.get("collision_detected", False)
            result.final_observation = attempt_result.get("final_observation", {})
            result.completion_confidence = attempt_result.get("completion_confidence", 0.0)

            if attempt_result.get("success", False):
                result.status = TaskStatus.COMPLETED
                result.success = True
                logger.info(f"Task {task.name} completed successfully")
                return result

            # Check if retry is appropriate
            if attempt_result.get("fatal", False):
                result.status = TaskStatus.FATAL_FAILURE
                result.error_message = attempt_result.get("error", "Fatal error")
                return result

            if attempt < task.max_retries - 1:
                logger.warning(
                    f"Task {task.name} attempt {attempt + 1} failed: "
                    f"{attempt_result.get('error', 'Unknown error')}. Retrying..."
                )
                time.sleep(1.0)

        # All retries exhausted
        result.status = TaskStatus.FAILED
        result.error_message = f"Failed after {task.max_retries} attempts"
        result.end_time = time.time()

        return result

    def execute_single_attempt(
        self,
        task: TaskConfig,
        collision_detector=None,
        collision_handler=None,
        state_monitor=None,
    ) -> dict[str, Any]:
        """Execute a single attempt of a task using action chunks.

        This method uses get_action_chunk() to retrieve a full sequence of actions
        from the policy, then executes them sequentially. This is more efficient and
        produces smoother motion than getting actions one at a time.

        Args:
            task: Task configuration.
            collision_detector: Optional collision detector.
            collision_handler: Optional collision handler.
            state_monitor: Optional state monitor.

        Returns:
            Dictionary with attempt result.
        """
        start_time = time.time()
        timeout = start_time + task.max_duration

        result = {
            "success": False,
            "fatal": False,
            "collision_detected": False,
            "error": "",
            "final_observation": {},
            "completion_confidence": 0.0,
        }

        # Reset executor state
        self.policy_executor.reset()

        # Control frequency from config
        control_dt = 1.0 / 30.0  # 30 Hz
        if hasattr(task, "control_frequency"):
            control_dt = 1.0 / task.control_frequency

        # Track if we need a new action chunk
        action_chunk: list[dict[str, float]] | None = None
        chunk_index = 0

        try:
            # Initialize with empty action for first collision check
            last_action = None

            while time.time() < timeout:
                loop_start = time.time()

                # Get current observation
                try:
                    observation = self.robot.get_observation()
                except Exception as e:
                    result["fatal"] = True
                    result["error"] = f"Failed to get observation: {e}"
                    return result

                # Update monitor
                if state_monitor is not None:
                    state_monitor.update(observation, last_action)

                # Check for collision using last action (for inertia compensation)
                if collision_detector is not None:
                    collision_result = collision_detector.check_collision(observation, last_action)
                    if collision_result.is_detected:
                        result["collision_detected"] = True
                        logger.warning(f"Collision detected: {collision_result.affected_joints}")

                        if collision_handler is not None:
                            handler_result = collision_handler.handle_collision(
                                collision_result, observation
                            )

                            if not handler_result.can_continue:
                                result["fatal"] = True
                                result["error"] = "Collision - cannot continue"
                                return result

                        # Reset action chunk after collision to get fresh trajectory
                        action_chunk = None
                        last_action = None
                        time.sleep(0.5)
                        continue

                # Check for completion
                if self.completion_detector is not None:
                    detection = self.completion_detector.detect(
                        observation, list(self.action_history)
                    )
                    if detection.is_completed:
                        result["success"] = True
                        result["completion_confidence"] = detection.confidence
                        result["final_observation"] = observation
                        logger.info(f"Task completed with confidence: {detection.confidence}")
                        return result

                # Get new action chunk if needed
                if action_chunk is None or chunk_index >= len(action_chunk):
                    action_chunk = self.policy_executor.get_action_chunk(observation)
                    if action_chunk is None or len(action_chunk) == 0:
                        result["error"] = "Failed to get action chunk from policy"
                        return result
                    chunk_index = 0
                    logger.debug(f"Got new action chunk with {len(action_chunk)} actions")

                # Get current action from chunk
                action = action_chunk[chunk_index]
                chunk_index += 1
                last_action = action

                # Send action to robot
                try:
                    self.robot.send_action(action)
                    self.action_history.append(action)
                    self.observation_history.append(observation)
                except Exception as e:
                    result["error"] = f"Failed to send action: {e}"
                    return result

                # Maintain control frequency
                loop_time = time.time() - loop_start
                sleep_time = control_dt - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Timeout
            result["error"] = f"Task timeout after {task.max_duration}s"

        except Exception as e:
            result["fatal"] = True
            result["error"] = f"Exception during execution: {e}"
            logger.exception("Exception during task execution")

        return result

    def _load_policy_for_task(self, task: TaskConfig) -> bool:
        """Load the policy for a specific task.

        Args:
            task: Task configuration containing policy path and type.

        Returns:
            True if loading was successful, False otherwise.
        """
        policy_path = task.policy_path
        policy_type = task.policy_type

        if self.current_policy_path == policy_path:
            return True  # Already loaded

        logger.info(f"Loading policy for task {task.name}: {policy_path}")

        return self.policy_executor.load_policy(policy_path, policy_type)

    def get_current_state(self) -> dict[str, Any]:
        """Get the current scheduler state."""
        return {
            "current_policy": self.current_policy_path,
            "action_queue_size": len(self.action_history),
            "total_executions": self._total_task_executions,
            "executor_info": self.policy_executor.get_info() if self.policy_executor else None,
        }
