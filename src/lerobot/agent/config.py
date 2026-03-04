"""
Configuration for the task agent orchestrator.

Re-exports the main OrchestratorConfig from the tasks module
for convenience and backward compatibility.
"""

from typing import Any

from dataclasses import dataclass, field

import draccus

from lerobot.tasks.config import (
    CollisionConfig,
    MonitoringConfig,
    OrchestratorConfig as TasksOrchestratorConfig,
    RobotConfig,
    TaskConfig,
)


@dataclass
class OrchestratorConfig(TasksOrchestratorConfig):
    """Extended orchestrator configuration with additional agent-specific settings.

    This extends the base OrchestratorConfig from the tasks module with
    agent-specific settings for policy server connection and execution behavior.
    """

    # Execution mode settings
    use_local_execution: bool = True  # True=local (no Policy Server), False=remote (via gRPC)
    policy_device: str = "cuda"  # Device for local policy execution

    # Policy server settings (only used when use_local_execution=False)
    policy_server_host: str = "localhost"
    policy_server_port: int = 50051
    policy_connection_timeout: float = 10.0  # Seconds

    # New feature settings
    enable_interactive_mode: bool = False  # Enable interactive task selection before each task
    enable_emergency_stop: bool = True  # Enable emergency stop with action rollback
    emergency_history_size: int = 1000  # Number of actions to store for rollback
    auto_rollback_on_stop: bool = True  # Automatically rollback after emergency stop

    # Execution settings
    auto_start: bool = True  # Automatically start execution on init
    continue_on_collision: bool = True  # Continue task sequence after collision
    max_total_collisions: int = 10  # Maximum collisions before abort

    # Task settings overrides
    override_max_retries: int | None = None  # Override task-specific retry counts
    override_max_duration: float | None = None  # Override task-specific durations

    # Adaptive execution settings
    enable_adaptive_scheduler: bool = True  # Use AdaptiveTaskScheduler for better performance
    gripper_config: dict | None = None  # Configuration for gripper force feedback

    # Robot reset settings
    reset_duration: float = 3.0  # Time in seconds for smooth reset to zero position
    reset_positions: dict[str, float] = field(default_factory=dict)  # Manual reset positions per joint (e.g., {"right_arm_joint_7": 0.5})

    # Debug settings
    debug_mode: bool = False
    save_observations: bool = False
    save_actions: bool = False
    save_dir: str = "/tmp/task_agent_debug"
