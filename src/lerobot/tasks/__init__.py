"""
Task module for robot task execution system.

This module provides task scheduling, execution, and completion detection
for multi-step robotic operations using ACT policies.
"""

from .completion_detector import (
    CompletionCriteria,
    DetectionResult,
    TaskCompletionDetector,
)
from .config import (
    CameraConfig,
    OrchestratorConfig,
    TaskConfig,
    load_config_from_yaml,
)
# Import RobotConfig directly from robots module to avoid conflicts
from lerobot.robots.config import RobotConfig
from .local_policy_executor import LocalPolicyExecutor
from .adaptive_scheduler import AdaptiveTaskScheduler
from .task_scheduler import (
    ExecutionSummary,
    LocalTaskScheduler,
    TaskResult,
    TaskScheduler,
    TaskStatus,
)

__all__ = [
    # Completion detection
    "CompletionCriteria",
    "DetectionResult",
    "TaskCompletionDetector",
    # Configuration
    "CameraConfig",
    "OrchestratorConfig",
    "RobotConfig",
    "TaskConfig",
    "load_config_from_yaml",
    # Policy execution
    "LocalPolicyExecutor",
    # Adaptive scheduling
    "AdaptiveTaskScheduler",
    # Scheduling
    "ExecutionSummary",
    "LocalTaskScheduler",
    "TaskResult",
    "TaskScheduler",
    "TaskStatus",
]
