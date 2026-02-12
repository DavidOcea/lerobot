"""
Interactive Task Selector for User-Guided Task Execution

This module provides interactive prompting and task queue management,
allowing users to:
1. Select next task interactively
2. Continue with automatic task sequence
3. Insert custom tasks dynamically

Usage:
    selector = InteractiveTaskSelector()
    selector.set_task_queue([task1, task2, task3])
    next_task = selector.prompt_next_task()
    if next_task:
        execute(next_task)
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from lerobot.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Task execution mode."""
    AUTOMATIC = "automatic"  # Execute tasks in original order
    INTERACTIVE = "interactive"  # Prompt before each task


@dataclass
class TaskSelection:
    """Result of task selection prompt."""
    selected_task: str | None = None  # None means continue with automatic
    custom_task_name: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.AUTOMATIC
    exit_requested: bool = False


class InteractiveTaskSelector:
    """Interactive task selector with user prompting.

    Features:
    - Prompt before each task for user selection
    - Support for automatic task sequence execution
    - Dynamic task queue insertion
    - Custom task creation
    - Exit handler integration
    """

    def __init__(
        self,
        tasks: list[TaskConfig],
        exit_handler: Callable[[], bool] | None = None,
    ):
        """Initialize the interactive task selector.

        Args:
            tasks: List of configured tasks.
            exit_handler: Function to call when user requests exit.
        """
        self.config_tasks = tasks
        self._exit_handler = exit_handler
        self._current_task_index: int = 0
        self._execution_mode: ExecutionMode = ExecutionMode.AUTOMATIC
        self._is_paused: bool = False

        # Task queue for dynamic task management
        self.task_queue: deque[TaskConfig] = deque(tasks)

        # State tracking
        self._total_executed: int = 0
        self._task_execution_counts: dict[str, int] = {}

        logger.info("InteractiveTaskSelector initialized")

    def set_task_queue(self, tasks: list[TaskConfig]) -> None:
        """Set a new task queue.

        Args:
            tasks: List of TaskConfigs to set as the task queue.
        """
        self.config_tasks = tasks
        self.task_queue = deque(tasks)
        logger.info(f"Task queue updated to {len(tasks)} tasks")

    def get_next_task(self) -> TaskConfig | None:
        """Get the next task from the queue without prompting.

        Returns:
            Next task or None if no tasks remaining.
        """
        if self._current_task_index < len(self.config_tasks):
            return self.config_tasks[self._current_task_index]
        return None

    def advance_task_index(self) -> None:
        """Advance to the next task in the sequence."""
        self._current_task_index += 1
        self._total_executed += 1

    def prompt_next_task(self, force_mode: ExecutionMode = ExecutionMode.INTERACTIVE) -> TaskSelection:
        """Prompt user for next task selection.

        Args:
            force_mode: Force interactive prompting even if in automatic mode.

        Returns:
            TaskSelection with user's choice.
        """
        # Check if we have tasks configured
        if not self.config_tasks:
            logger.warning("No tasks configured, returning empty selection")
            return TaskSelection(
                execution_mode=ExecutionMode.AUTOMATIC,
                selected_task=None,
                exit_requested=False
            )

        # In automatic mode, just return next task
        if self._execution_mode == ExecutionMode.AUTOMATIC and force_mode != ExecutionMode.INTERACTIVE:
            return TaskSelection(
                execution_mode=ExecutionMode.AUTOMATIC,
                selected_task=None,
                exit_requested=False
            )

        # Get current task info
        if self._current_task_index < len(self.config_tasks):
            current_task = self.config_tasks[self._current_task_index]
        else:
            current_task = None

        # Build prompt
        prompt_lines = self._build_task_prompt(current_task)

        # Display prompt and get user response
        logger.info("\n" + "\n".join(prompt_lines))
        print("\n" + "\n".join(prompt_lines))

        # Get user selection
        return self._get_user_selection()

    def _build_task_prompt(self, current_task: TaskConfig | None) -> list[str]:
        """Build the interactive prompt for user."""
        lines = [
            "=" * 60,
            "INTERACTIVE TASK SELECTION",
            "=" * 60,
        ]

        if current_task:
            lines.extend([
                f"Next task: {current_task.name}",
                f"Policy: {current_task.policy_path}",
                "",
            ])

        lines.extend([
            "Options:",
            "  1 - Execute next task in sequence",
            "  2 - Create custom task",
            "  3 - Toggle automatic/interactive mode",
            "  0 - Exit",
            "",
        ])

        # Show task list
        lines.append("Available tasks:")
        for i, task in enumerate(self.config_tasks):
            current = " <- NEXT" if i == self._current_task_index else ""
            lines.append(f"  {i+1}. {task.name}{current}")

        lines.extend([
            "",
            f"Current mode: {self._execution_mode.value}",
            "=" * 60,
        ])

        return lines

    def _get_user_selection(self) -> TaskSelection:
        """Get user selection from input.

        For programmatic use, this can be overridden.
        """
        try:
            user_input = input(">>> ").strip()

            # Handle empty input (default = option 1)
            if not user_input:
                logger.info("Empty input, defaulting to option 1 (next task)")
                user_input = "1"

            # Parse selection
            return self._parse_user_input(user_input)

        except KeyboardInterrupt:
            logger.info("User interrupted with Ctrl+C")
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                exit_requested=True
            )
        except EOFError:
            logger.info("End of input stream")
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                exit_requested=True
            )

    def _parse_user_input(self, user_input: str) -> TaskSelection:
        """Parse user input and return selection."""
        input_lower = user_input.lower().strip()

        # Handle direct task name input
        task_names = {task.name.lower(): task for task in self.config_tasks}

        if input_lower in task_names:
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                selected_task=input_lower,
            )

        # Handle numeric options
        if input_lower == "1":
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                selected_task=None,  # Next task
            )
        elif input_lower == "2":
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                custom_task_name="",  # Prompt for task name
            )
        elif input_lower == "3":
            # Toggle mode
            new_mode = (
                ExecutionMode.AUTOMATIC
                if self._execution_mode == ExecutionMode.INTERACTIVE
                else ExecutionMode.INTERACTIVE
            )
            logger.info(f"Switching execution mode: {self._execution_mode} -> {new_mode}")
            self._execution_mode = new_mode
            return TaskSelection(
                execution_mode=new_mode,
                selected_task=None,
            )
        elif input_lower == "0":
            # Exit request
            if self._exit_handler:
                result = self._exit_handler()
                logger.info("Exit handler called, shutting down...")
            else:
                logger.warning("Exit requested but no handler configured")
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                exit_requested=True
            )

        # Unknown input
        logger.warning(f"Unknown user input: {user_input}")
        return TaskSelection(
            execution_mode=ExecutionMode.INTERACTIVE,
            selected_task=None,  # Stay in interactive mode
        )

    def _process_selection(
        self,
        selection: TaskSelection,
        current_task: TaskConfig | None,
    ) -> TaskSelection:
        """Process user selection and update state."""
        # Handle exit request
        if selection.exit_requested:
            logger.info("Exit requested, stopping task execution")
            return selection

        # Handle mode switch
        if selection.execution_mode != self._execution_mode:
            self._execution_mode = selection.execution_mode
            logger.info(f"Execution mode changed to: {selection.execution_mode}")

        # Handle custom task creation
        if selection.custom_task_name is not None:
            return self._create_and_enqueue_custom_task(selection.custom_task_name)

        # Handle next task selection
        if selection.selected_task:
            # User selected a specific task
            task_name_lower = selection.selected_task.lower()
            task_names = {task.name.lower(): task for task in self.config_tasks}

            if task_name_lower in task_names:
                for i, task in enumerate(self.config_tasks):
                    if task.name.lower() == task_name_lower:
                        self._current_task_index = i
                        logger.info(f"Selected task: {task.name}")
                        break
            else:
                logger.warning(f"Task '{selection.selected_task}' not found in configuration")

        return selection

    def _create_and_enqueue_custom_task(self, task_name: str) -> TaskSelection:
        """Create a custom task and add to queue.

        Args:
            task_name: Name of the custom task.
        """
        # Import CameraConfig for creating empty cameras list
        from lerobot.tasks.config import CameraConfig

        # Create a simple custom task config
        custom_task = TaskConfig(
            name=task_name,
            policy_path="",  # Empty for custom task
            max_duration=300.0,  # 5 minutes default
            max_retries=3,
            cameras=[],  # Empty list for custom task
            enabled=True,
        )

        # Add to queue
        self.task_queue.append(custom_task)
        self.config_tasks.append(custom_task)
        logger.info(f"Created custom task '{task_name}' and added to queue")

        return TaskSelection(
            execution_mode=ExecutionMode.INTERACTIVE,
            selected_task=task_name,
        )

    def skip_current_task(self):
        """Skip the current task and move to next one.

        Used when user wants to skip the current task execution.
        """
        if self._current_task_index < len(self.config_tasks):
            logger.info(f"Skipping task: {self.config_tasks[self._current_task_index].name}")
            self._current_task_index += 1
        else:
            logger.warning("No current task to skip")

    def set_execution_mode(self, mode: ExecutionMode):
        """Set the task execution mode.

        Args:
            mode: AUTOMATIC or INTERACTIVE.
        """
        self._execution_mode = mode
        logger.info(f"Execution mode set to: {mode}")

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue and execution status.

        Returns:
            Dictionary with queue information.
        """
        return {
            "queue_size": len(self.task_queue),
            "current_task_index": self._current_task_index,
            "current_task": (
                self.config_tasks[self._current_task_index].name
                if self._current_task_index < len(self.config_tasks)
                else None
            ),
            "execution_mode": self._execution_mode,
            "is_paused": self._is_paused,
            "total_executed": self._total_executed,
            "task_counts": self._task_execution_counts.copy(),
        }

    def reset(self):
        """Reset task selector state."""
        self._current_task_index = 0
        self.task_queue.clear()
        self.task_queue.extend(self.config_tasks)
        self._total_executed = 0
        self._task_execution_counts.clear()
        self._execution_mode = ExecutionMode.AUTOMATIC
        self._is_paused = False

        logger.info("InteractiveTaskSelector reset")


def create_interactive_selector(
    tasks: list[TaskConfig],
    exit_handler: Callable[[], bool] | None = None,
) -> InteractiveTaskSelector:
    """Create an interactive task selector instance.

    Args:
        tasks: List of configured tasks.
        exit_handler: Optional handler for exit events.

    Returns:
        Configured InteractiveTaskSelector instance.
    """
    return InteractiveTaskSelector(
        tasks=tasks,
        exit_handler=exit_handler,
    )
