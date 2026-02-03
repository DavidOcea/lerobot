"""
Configuration classes for the task execution system.

This module defines dataclasses and configuration loaders for:
- TaskConfig: Individual task configuration
- CompletionCriteria: Criteria for detecting task completion
- OrchestratorConfig: Overall orchestrator configuration
- RobotConfig: Robot-specific configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import yaml

# Import from specialized modules
from lerobot.safety.config import CollisionConfig
from lerobot.monitoring.config import MonitoringConfig

# Import RobotConfig from robots module
from lerobot.robots.config import RobotConfig


@dataclass
class CompletionCriteria:
    """Criteria for detecting task completion.

    Supports multiple detection types:
    - "position": Check if joint positions reach target values
    - "force": Check if force exceeds threshold (for grip confirmation)
    - "stability": Check if state remains stable over time window
    - "composite": Combine multiple conditions with AND logic
    """

    type: str = "position"  # "position", "force", "stability", "composite"
    # Position-based criteria
    target_joint_positions: dict[str, float] = field(default_factory=dict)
    position_tolerance: float = 0.01  # radians for joints
    # Force-based criteria
    force_threshold: float = 0.5  # Newton-meters
    joint_name: str | None = None  # Specific joint to monitor (e.g., gripper)
    # Stability-based criteria
    stability_window: int = 10  # Number of steps to check
    stability_tolerance: float = 0.005  # Max variation within window
    # Composite criteria
    conditions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    name: str  # Camera key name (e.g., "head_cam")
    type: str = "opencv"  # Camera type: "opencv", "realsense", etc.
    index: int = 0  # Camera device index
    width: int = 640  # Image width
    height: int = 480  # Image height
    fps: int = 30  # Frames per second


@dataclass
class TaskConfig:
    """Configuration for a single task in the execution sequence."""

    name: str
    policy_path: str
    policy_type: str = "act"
    max_duration: float = 30.0  # Maximum execution time in seconds
    max_retries: int = 3  # Maximum number of retry attempts
    completion_criteria: CompletionCriteria = field(default_factory=CompletionCriteria)
    enabled: bool = True  # Allow disabling specific tasks
    cameras: list[CameraConfig] = field(default_factory=list)  # Active cameras for this task


@dataclass
class OrchestratorConfig:
    """Complete configuration for the task agent orchestrator.

    This configuration brings together all subsystems:
    - Task sequence and scheduling
    - Robot interface
    - Collision detection and safety
    - Monitoring and logging
    """

    tasks: list[TaskConfig] = field(default_factory=list)
    robot_config: RobotConfig = field(default_factory=RobotConfig)
    collision_config: CollisionConfig = field(default_factory=CollisionConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Execution settings
    environment_dt: float = 1.0 / 30.0  # Control loop timestep
    observation_timeout: float = 5.0  # Seconds to wait for observations
    action_timeout: float = 5.0  # Seconds to wait for actions


def load_config_from_yaml(config_path: str | Path) -> OrchestratorConfig:
    """Load orchestrator configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        OrchestratorConfig: Loaded configuration object.

    Example YAML structure:
        tasks:
          - name: "pick_short_workpiece"
            policy_path: "/path/to/act_model_short_piece"
            policy_type: "act"
            max_duration: 30.0
            max_retries: 3
            completion_criteria:
              type: "composite"
              conditions:
                - type: "force"
                  joint_name: "left_arm_joint_7"
                  force_threshold: 0.5
                - type: "stability"
                  window: 10
                  stability_tolerance: 0.005

        robot_config:
          type: "supre_robot_follower"
          camera_enabled: true
          force_sensing_enabled: true

        collision_config:
          collision_threshold: 2.0
          detection_window: 5
          adaptive_mode: true
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Parse tasks
    tasks = []
    for task_dict in config_dict.get("tasks", []):
        # Parse completion criteria if present
        criteria_dict = task_dict.get("completion_criteria", {})
        if criteria_dict:
            # Convert conditions list to CompletionCriteria objects
            if "conditions" in criteria_dict:
                conditions = []
                for cond in criteria_dict["conditions"]:
                    conditions.append(cond)
                criteria_dict["conditions"] = conditions

            completion_criteria = CompletionCriteria(**criteria_dict)
        else:
            completion_criteria = CompletionCriteria()

        # Parse cameras if present
        cameras_list = task_dict.get("cameras", [])
        cameras = []
        for cam_dict in cameras_list:
            cameras.append(CameraConfig(**cam_dict))

        # Create TaskConfig
        task_dict["completion_criteria"] = completion_criteria
        task_dict["cameras"] = cameras
        tasks.append(TaskConfig(**task_dict))

    # Parse robot config using draccus to handle polymorphic types
    robot_config_dict = config_dict.get("robot_config", {})
    if robot_config_dict:
        # Import robot config modules to register their configs
        # Use direct import to avoid triggering hardware dependencies
        import sys
        import importlib.util

        # Get the robot type from the config
        robot_type = robot_config_dict.get("type", "supre_robot_follower")

        # Try to get the config class - it should already be registered by other imports
        try:
            robot_config_class = RobotConfig.get_choice_class(robot_type)
            # Remove 'type' from dict as it's handled by draccus
            robot_config = robot_config_class(**{k: v for k, v in robot_config_dict.items() if k != "type"})
        except Exception as e:
            # If class not found, try loading the config module
            config_module_paths = {
                "supre_robot_follower": Path(__file__).parent.parent / "robots" / "supre_robot_follower" / "supre_robot_follower_config.py",
                "mock_robot": Path(__file__).parent.parent.parent.parent / "tests" / "mocks" / "mock_robot.py",
            }

            config_module_path = config_module_paths.get(robot_type)
            if config_module_path and config_module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    f"{robot_type}_config",
                    config_module_path
                )
                if spec and spec.loader:
                    config_module = importlib.util.module_from_spec(spec)
                    # Insert into sys.modules before loading
                    module_name = f"lerobot.robots.{robot_type}" if robot_type != "mock_robot" else "tests.mocks.mock_robot"
                    sys.modules[module_name] = config_module
                    spec.loader.exec_module(config_module)

            # Try again
            try:
                robot_config_class = RobotConfig.get_choice_class(robot_type)
                robot_config = robot_config_class(**{k: v for k, v in robot_config_dict.items() if k != "type"})
            except Exception as e2:
                # Fallback: if we can't load the specific robot config, raise an error
                # Using base RobotConfig would fail later when accessing config.type
                import logging
                logging.getLogger(__name__).error(
                    f"Failed to load robot config class for type '{robot_type}': {e2}"
                )
                raise ValueError(
                    f"Cannot load robot configuration for type '{robot_type}'. "
                    f"Please ensure the robot type is registered and the config module is accessible."
                ) from e2
    else:
        # No robot config provided - raise an error
        raise ValueError(
            "No robot_config provided in configuration. "
            "Please specify a robot_config section with a valid 'type' field."
        )

    # Parse collision config
    collision_config_dict = config_dict.get("collision_config", {})
    collision_config = CollisionConfig(**collision_config_dict)

    # Parse monitoring config
    monitoring_config_dict = config_dict.get("monitoring_config", {})
    monitoring_config = MonitoringConfig(**monitoring_config_dict)

    # Create main config
    config = OrchestratorConfig(
        tasks=tasks,
        robot_config=robot_config,
        collision_config=collision_config,
        monitoring_config=monitoring_config,
        environment_dt=config_dict.get("environment_dt", 1.0 / 30.0),
        observation_timeout=config_dict.get("observation_timeout", 5.0),
        action_timeout=config_dict.get("action_timeout", 5.0),
    )

    return config


# Note: draccus.register_choice_type is not available in current versions
# The dataclasses work without explicit registration when using YAML loading
