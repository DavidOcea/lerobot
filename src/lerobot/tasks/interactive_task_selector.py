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
import os
import select
import sys
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
    reset_requested: bool = False  # User requested robot reset to zero position


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
        monitor_collector=None,
        robot=None,
    ):
        """Initialize the interactive task selector.

        Args:
            tasks: List of configured tasks.
            exit_handler: Function to call when user requests exit.
            monitor_collector: Optional MonitorCollector for dashboard button integration.
            robot: Optional Robot instance (reserved for future use).
        """
        self.config_tasks = tasks
        self._exit_handler = exit_handler
        self._monitor_collector = monitor_collector
        self._robot = robot
        self._current_task_index: int = 0
        self._execution_mode: ExecutionMode = ExecutionMode.AUTOMATIC
        self._is_paused: bool = False

        # Task queue for dynamic task management
        self.task_queue: deque[TaskConfig] = deque(tasks)

        # State tracking
        self._total_executed: int = 0
        self._task_execution_counts: dict[str, int] = {}

        logger.info("InteractiveTaskSelector initialized")

    @property
    def current_task_index(self) -> int:
        """Get the current task index."""
        return self._current_task_index

    @current_task_index.setter
    def current_task_index(self, value: int):
        """Set the current task index."""
        self._current_task_index = max(0, min(value, len(self.config_tasks)))
        logger.debug(f"Task index set to {self._current_task_index}")

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
        prompt_data = self._build_dashboard_prompt(current_task)

        # Display prompt and get user response
        logger.info("\n" + "\n".join(prompt_lines))
        print("\n" + "\n".join(prompt_lines))

        # Get user selection (supports both terminal and dashboard frontend)
        return self._get_user_selection(prompt_data=prompt_data)

    def _build_task_prompt(self, current_task: TaskConfig | None) -> list[str]:
        """Build the interactive prompt for user."""
        lines = [
            "=" * 60,
            "交互式任务选择",
            "=" * 60,
        ]

        if current_task:
            lines.extend([
                f"下一个任务: {current_task.name}",
                f"策略路径: {current_task.policy_path}",
                "",
            ])

        lines.extend([
            "选项:",
            "  1 - 执行下一个任务",
            "  2 - 选择或创建自定义任务",
            "  3 - 切换自动/交互模式",
            "  R - 复位机器人关节到零位 (平滑复位)",
            "  0 - 退出",
            "",
        ])

        # Show task list with execution status
        lines.append("可用任务:")
        for i, task in enumerate(self.config_tasks):
            current = " ← 下一个" if i == self._current_task_index else ""

            # Add execution status
            status = ""
            if task.name in self._task_execution_counts:
                count = self._task_execution_counts[task.name]
                status = f" (已执行 {count} 次)"

            lines.append(f"  {i+1}. {task.name}{status}{current}")

        mode_str = "自动" if self._execution_mode.value == "automatic" else "交互"
        lines.extend([
            "",
            f"当前模式: {mode_str}",
            "=" * 60,
        ])

        return lines

    def _build_dashboard_prompt(self, current_task: TaskConfig | None) -> dict:
        """Build structured prompt data for the dashboard frontend buttons."""
        options = [
            {"key": "1", "label": "执行下一个任务"},
            {"key": "2", "label": "选择或创建自定义任务"},
            {"key": "3", "label": "切换自动/交互模式"},
            {"key": "r", "label": "复位机器人关节到零位"},
            {"key": "0", "label": "退出"},
        ]
        task_name = current_task.name if current_task else "(无)"
        mode_str = "自动" if self._execution_mode.value == "automatic" else "交互"
        return {
            "type": "task_selection",
            "message": f"下一个任务: {task_name} | 模式: {mode_str}",
            "options": options,
        }

    def _get_user_selection(self, prompt_data: dict | None = None) -> TaskSelection:
        """Get user selection from input.

        For programmatic use, this can be overridden.
        """
        try:
            # Use _get_input helper to avoid hardware device interference.
            # When monitor_collector is present, this also supports
            # dashboard frontend buttons via select() multiplexing.
            user_input = self._get_input(">>> ", prompt_data=prompt_data).strip()

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

    def _get_input(self, prompt: str = "", prompt_data: dict | None = None) -> str:
        """Get user input from terminal AND dashboard frontend via select().

        When monitor_collector is available, publishes prompt_data to the
        dashboard (so it renders buttons) and uses select() to wait on both
        /dev/tty and the command pipe simultaneously.  Terminal input and
        frontend button clicks are both supported without mode switching.

        Args:
            prompt: Prompt string for the terminal.
            prompt_data: Structured prompt for the dashboard frontend.
                {"type": str, "message": str, "options": [...], "timeout": float, "timeout_default": str}

        Returns:
            User input string (from terminal or frontend).
        """
        mc = self._monitor_collector

        # ── Dashboard-integrated path ──
        if mc is not None and prompt_data is not None:
            mc.set_pending_prompt(prompt_data)
            try:
                tty = open('/dev/tty', 'r')
            except (OSError, IOError):
                mc.clear_pending_prompt()
                return input(prompt).strip()

            pipe_fd = mc.command_pipe_r
            original_stdin = sys.stdin
            try:
                sys.stdin = tty
                if prompt:
                    sys.stdout.write(prompt)
                    sys.stdout.flush()

                while True:
                    # Block until terminal input OR frontend button click.
                    # No timeout needed: the self-pipe trick handles dashboard wakeup.
                    readable, _, _ = select.select([tty, pipe_fd], [], [])
                    for fd in readable:
                        if fd == tty.fileno():
                            line = tty.readline().strip()
                            return line
                        elif fd == pipe_fd:
                            os.read(pipe_fd, 256)  # drain pipe byte
                            cmd = mc.get_last_command()
                            if cmd:
                                sys.stdout.write(f"\n[frontend] {cmd}\n")
                                sys.stdout.flush()
                                return cmd
            finally:
                tty.close()
                sys.stdin = original_stdin
                mc.clear_pending_prompt()

        # ── Terminal-only fallback (no MonitorCollector) ──
        original_stdin = sys.stdin
        try:
            try:
                tty = open('/dev/tty', 'r')
                sys.stdin = tty
                if prompt:
                    sys.stdout.write(prompt)
                    sys.stdout.flush()
                user_input = tty.readline().strip()
                tty.close()
                return user_input
            except (OSError, IOError):
                return input(prompt).strip()
        finally:
            sys.stdin = original_stdin

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
            # Create custom task or select existing task
            # Show sub-menu for task selection
            print("\n" + "=" * 60)
            print("任务选择")
            print("=" * 60)
            print("选项:")
            print("  1 - 从已有任务中选择")
            print("  2 - 创建新的自定义任务")
            print("  0 - 取消")
            print("=" * 60)

            try:
                choice = self._get_input(
                    "请选择 (1/2/0): ",
                    prompt_data={
                        "type": "task_selection",
                        "message": "任务选择",
                        "options": [
                            {"key": "1", "label": "从已有任务中选择"},
                            {"key": "2", "label": "创建新的自定义任务"},
                            {"key": "0", "label": "取消"},
                        ],
                    },
                ).strip()

                if choice == "1":
                    # Select from existing tasks
                    print("\n可用任务:")
                    task_names = {task.name.lower(): task for task in self.config_tasks}

                    for i, task in enumerate(self.config_tasks):
                        # Show execution status
                        status_info = ""
                        if task.name in self._task_execution_counts:
                            count = self._task_execution_counts[task.name]
                            status_info = f" [已执行 {count} 次]"
                        current = " ← 下一个" if i == self._current_task_index else ""
                        print(f"  {i+1}. {task.name}{status_info}{current}")

                    task_input = self._get_input(
                        "输入任务编号或名称: ",
                        prompt_data={
                            "type": "task_selection",
                            "message": "选择要执行的任务",
                            "options": [
                                {"key": str(i + 1), "label": task.name}
                                for i, task in enumerate(self.config_tasks)
                            ],
                        },
                    ).strip()

                    # Try to parse as number first
                    try:
                        task_idx = int(task_input) - 1
                        if 0 <= task_idx < len(self.config_tasks):
                            selected_task = self.config_tasks[task_idx]
                            logger.info(f"User selected existing task: {selected_task.name}")
                            return TaskSelection(
                                execution_mode=ExecutionMode.INTERACTIVE,
                                selected_task=selected_task.name,
                            )
                        else:
                            print(f"无效的任务编号: {task_input}")
                            return TaskSelection(
                                execution_mode=ExecutionMode.INTERACTIVE,
                                selected_task=None,  # Stay in interactive mode
                            )
                    except ValueError:
                        # Not a number, treat as task name
                        task_name_lower = task_input.lower()
                        if task_name_lower in task_names:
                            logger.info(f"User selected existing task: {task_input}")
                            return TaskSelection(
                                execution_mode=ExecutionMode.INTERACTIVE,
                                selected_task=task_input,
                            )
                        else:
                            print(f"未找到任务: '{task_input}'")
                            return TaskSelection(
                                execution_mode=ExecutionMode.INTERACTIVE,
                                selected_task=None,  # Stay in interactive mode
                            )

                elif choice == "2":
                    # Create new custom task
                    custom_name = self._get_input(
                        "输入新任务名称: ",
                        prompt_data={
                            "type": "task_selection",
                            "message": "输入新任务名称 (请在终端输入)",
                            "options": [
                                {"key": "", "label": "取消 (空输入)"},
                            ],
                        },
                    ).strip()
                    if custom_name:
                        logger.info(f"User creating custom task: {custom_name}")
                        return TaskSelection(
                            execution_mode=ExecutionMode.INTERACTIVE,
                            custom_task_name=custom_name,
                        )
                    else:
                        print("任务名称不能为空")
                        return TaskSelection(
                            execution_mode=ExecutionMode.INTERACTIVE,
                            selected_task=None,  # Stay in interactive mode
                        )

                elif choice == "0":
                    # Cancel
                    return TaskSelection(
                        execution_mode=ExecutionMode.INTERACTIVE,
                        selected_task=None,  # Stay in interactive mode
                    )

                else:
                    print(f"无效的选择: {choice}")
                    return TaskSelection(
                        execution_mode=ExecutionMode.INTERACTIVE,
                        selected_task=None,  # Stay in interactive mode
                    )

            except (KeyboardInterrupt, EOFError):
                logger.info("Input interrupted")
                return TaskSelection(
                    execution_mode=ExecutionMode.INTERACTIVE,
                    selected_task=None,  # Stay in interactive mode
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
        elif input_lower == "r" or input_lower == "reset":
            # Reset robot joints to zero position
            logger.info("User requested robot reset to zero position")
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                reset_requested=True,
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
        print(f"未知输入: {user_input}")
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
        # Validate task name
        if not task_name or not task_name.strip():
            logger.warning("Empty task name provided, ignoring custom task creation")
            return TaskSelection(
                execution_mode=ExecutionMode.INTERACTIVE,
                selected_task=None,
            )

        task_name = task_name.strip()

        # Check if task already exists
        for existing_task in self.config_tasks:
            if existing_task.name == task_name:
                logger.info(f"Task '{task_name}' already exists, selecting it instead of creating duplicate")
                # Return selection for the existing task
                return TaskSelection(
                    execution_mode=ExecutionMode.INTERACTIVE,
                    selected_task=task_name,
                )

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

        # Add to queue and config
        self.task_queue.append(custom_task)
        self.config_tasks.append(custom_task)
        logger.info(f"Created custom task '{task_name}' and added to queue")

        # Return selection for the newly created task
        return TaskSelection(
            execution_mode=ExecutionMode.INTERACTIVE,
            selected_task=task_name,
        )

    def record_task_execution(self, task_name: str):
        """Record that a task has been executed.

        Args:
            task_name: Name of the task that was executed.
        """
        if task_name not in self._task_execution_counts:
            self._task_execution_counts[task_name] = 0
        self._task_execution_counts[task_name] += 1
        logger.info(f"Task '{task_name}' execution recorded (count: {self._task_execution_counts[task_name]})")

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
    monitor_collector=None,
) -> InteractiveTaskSelector:
    """Create an interactive task selector instance.

    Args:
        tasks: List of configured tasks.
        exit_handler: Optional handler for exit events.
        monitor_collector: Optional MonitorCollector for dashboard button integration.

    Returns:
        Configured InteractiveTaskSelector instance.
    """
    return InteractiveTaskSelector(
        tasks=tasks,
        exit_handler=exit_handler,
        monitor_collector=monitor_collector,
    )
