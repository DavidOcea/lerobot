"""
Configuration classes for the task execution system.

This module defines dataclasses and configuration loaders for:
- TaskConfig: Individual task configuration
- CompletionCriteria: Criteria for detecting task completion
- OrchestratorConfig: Overall orchestrator configuration
- RobotConfig: Robot-specific configuration
- AGVTaskConfig: AGV navigation task configuration (NEW)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import draccus
import yaml

# Import from specialized modules
from lerobot.safety.config import CollisionConfig
from lerobot.monitoring.config import MonitoringConfig

# Import RobotConfig from robots module
from lerobot.robots.config import RobotConfig

# Import camera config from lerobot.cameras for robot configuration
# Use alias to avoid conflict with local CameraConfig class
from lerobot.cameras.configs import CameraConfig as RobotCameraConfig


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
class AGVTaskConfig:
    """AGV移动任务配置.

    用于定义AGV导航任务，支持站点导航和坐标导航两种模式。
    """

    # 目标定义 (二选一)
    target_station: str | None = None  # 目标站点ID
    target_position: tuple[float, float, float] | None = None  # (x, y, theta) 米/弧度

    # 执行参数
    max_duration: float = 60.0  # 最大执行时间 (秒)
    wait_for_arrival: bool = True  # 是否等待到达完成
    arrival_timeout: float = 60.0  # 到达等待超时 (秒)

    # 到达判定
    arrival_tolerance: float = 0.3  # 距离容差 (米)
    station_match_required: bool = True  # 必须站点ID匹配才算到达

    # 安全检查
    check_arm_safe_position: bool = True  # 移动前检查机械臂是否在安全位置
    arm_safe_positions: dict[str, float] = field(default_factory=dict)  # 安全位置阈值

    # 错误处理
    retry_on_timeout: bool = True  # 超时是否重试
    retry_count: int = 2  # 重试次数
    emergency_stop_on_error: bool = True  # 异常时急停

    def validate(self) -> bool:
        """验证配置有效性."""
        if self.target_station is None and self.target_position is None:
            raise ValueError("AGVTaskConfig: Must specify either target_station or target_position")
        if self.max_duration <= 0:
            raise ValueError("AGVTaskConfig: max_duration must be positive")
        if self.arrival_timeout <= 0:
            raise ValueError("AGVTaskConfig: arrival_timeout must be positive")
        return True


@dataclass
class TaskConfig:
    """Configuration for a single task in the execution sequence.

    Supports multiple task types:
    - "policy": Execute a learned policy (ACT, etc.)
    - "agv": Execute AGV navigation task
    """

    name: str

    # 任务类型 (NEW)
    task_type: Literal["policy", "agv", "position"] = "policy"

    # Policy任务字段 (现有)
    policy_path: str | None = None
    policy_type: str = "act"
    max_duration: float = 30.0  # Maximum execution time in seconds
    max_retries: int = 3  # Maximum number of retry attempts
    completion_criteria: CompletionCriteria = field(default_factory=CompletionCriteria)
    enabled: bool = True  # Allow disabling specific tasks
    cameras: list[CameraConfig] = field(default_factory=list)  # Active cameras for this task

    # AGV任务字段 (NEW)
    agv_config: AGVTaskConfig | None = None

    def validate(self) -> bool:
        """验证任务配置."""
        if self.task_type == "policy":
            if self.policy_path is None:
                raise ValueError(f"Task '{self.name}': policy_path required for policy tasks")
        elif self.task_type == "agv":
            if self.agv_config is None:
                raise ValueError(f"Task '{self.name}': agv_config required for agv tasks")
            self.agv_config.validate()
        elif self.task_type == "position":
            if not self.completion_criteria.target_joint_positions:
                raise ValueError(f"Task '{self.name}': target_joint_positions required for position tasks")
        return True


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

    # Reset settings
    reset_duration: float = 3.0  # Time in seconds for smooth reset to zero position
    reset_positions: dict[str, float] = field(default_factory=dict)  # Manual reset positions per joint (e.g., {"left_arm_joint_1": 0.0, ...}). If empty, use 0.0 for all joints.


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
        # Get task type (default to "policy")
        task_type = task_dict.get("task_type", "policy")

        # Parse completion criteria if present (for policy tasks)
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

        # Parse AGV config if present (for AGV tasks)
        agv_config_dict = task_dict.get("agv_config", {})
        agv_config = None
        if agv_config_dict:
            # Parse target_position if it's a list
            if "target_position" in agv_config_dict:
                pos = agv_config_dict["target_position"]
                if isinstance(pos, list) and len(pos) >= 2:
                    # Convert list to tuple
                    theta = pos[2] if len(pos) >= 3 else 0.0
                    agv_config_dict["target_position"] = (float(pos[0]), float(pos[1]), float(theta))

            agv_config = AGVTaskConfig(**agv_config_dict)

        # Build TaskConfig with proper fields based on task_type
        # Remove fields that don't belong to TaskConfig direct attributes
        task_kwargs = {
            "name": task_dict["name"],
            "task_type": task_type,
            "completion_criteria": completion_criteria,
            "cameras": cameras,
            "agv_config": agv_config,
        }

        # Add policy-specific fields
        if task_type == "policy":
            task_kwargs["policy_path"] = task_dict.get("policy_path")
            task_kwargs["policy_type"] = task_dict.get("policy_type", "act")
            task_kwargs["max_duration"] = task_dict.get("max_duration", 30.0)
            task_kwargs["max_retries"] = task_dict.get("max_retries", 3)
            task_kwargs["enabled"] = task_dict.get("enabled", True)

        # Add AGV-specific fields (some overlap with policy)
        elif task_type == "agv":
            task_kwargs["max_duration"] = task_dict.get("max_duration", 60.0)
            task_kwargs["max_retries"] = task_dict.get("max_retries", 2)
            task_kwargs["enabled"] = task_dict.get("enabled", True)

        # Add position-specific fields (for direct joint position movement)
        elif task_type == "position":
            task_kwargs["max_duration"] = task_dict.get("max_duration", 10.0)
            task_kwargs["max_retries"] = task_dict.get("max_retries", 1)
            task_kwargs["enabled"] = task_dict.get("enabled", True)

        tasks.append(TaskConfig(**task_kwargs))

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
            robot_config_dict_clean = {k: v for k, v in robot_config_dict.items() if k != "type"}

            # Parse camera configurations if present
            if "cameras" in robot_config_dict_clean and robot_config_dict_clean["cameras"]:
                cameras = {}
                for cam_name, cam_dict in robot_config_dict_clean["cameras"].items():
                    if isinstance(cam_dict, dict):
                        # Convert YAML camera config to CameraConfig
                        # The 'type' field is used to select the subclass (e.g., "opencv" -> OpenCVCameraConfig)
                        cam_type = cam_dict.pop("type", "opencv")  # Remove 'type' from kwargs

                        # Map common field names
                        if "index" in cam_dict:
                            cam_dict["index_or_path"] = cam_dict.pop("index")

                        # Import the correct camera config class based on type
                        if cam_type == "opencv":
                            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
                            cameras[cam_name] = OpenCVCameraConfig(**cam_dict)
                        elif cam_type == "intelrealsense":
                            from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
                            cameras[cam_name] = RealSenseCameraConfig(**cam_dict)
                        else:
                            raise ValueError(f"Unknown camera type: {cam_type}")
                    else:
                        cameras[cam_name] = cam_dict
                robot_config_dict_clean["cameras"] = cameras

            robot_config = robot_config_class(**robot_config_dict_clean)
        except Exception as e:
            # If class not found, try loading the config module
            # Only load the config module, not the full package (to avoid hardware dependencies)
            config_module_paths = {
                "supre_robot_follower": Path(__file__).parent.parent / "robots" / "supre_robot_follower" / "supre_robot_follower_config.py",
                "mock_robot": Path(__file__).parent.parent.parent.parent / "tests" / "mocks" / "mock_robot.py",
            }

            config_module_path = config_module_paths.get(robot_type)
            if config_module_path and config_module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    f"lerobot.robots.{robot_type}.{robot_type}_config",
                    config_module_path
                )
                if spec and spec.loader:
                    config_module = importlib.util.module_from_spec(spec)
                    # For mock_robot, set the module name correctly
                    if robot_type == "mock_robot":
                        sys.modules["tests.mocks.mock_robot"] = config_module
                    # For other robots, use a unique name that won't interfere with normal imports
                    else:
                        sys.modules[f"lerobot.robots.{robot_type}.{robot_type}_config"] = config_module
                    spec.loader.exec_module(config_module)

            # Try again
            try:
                robot_config_class = RobotConfig.get_choice_class(robot_type)
                # Remove 'type' from dict as it's handled by draccus
                robot_config_dict_clean = {k: v for k, v in robot_config_dict.items() if k != "type"}

                # Parse camera configurations if present
                if "cameras" in robot_config_dict_clean and robot_config_dict_clean["cameras"]:
                    cameras = {}
                    for cam_name, cam_dict in robot_config_dict_clean["cameras"].items():
                        if isinstance(cam_dict, dict):
                            # Convert YAML camera config to CameraConfig
                            # The 'type' field is used to select the subclass (e.g., "opencv" -> OpenCVCameraConfig)
                            cam_type = cam_dict.pop("type", "opencv")  # Remove 'type' from kwargs

                            # Map common field names
                            if "index" in cam_dict:
                                cam_dict["index_or_path"] = cam_dict.pop("index")

                            # Import the correct camera config class based on type
                            if cam_type == "opencv":
                                from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
                                cameras[cam_name] = OpenCVCameraConfig(**cam_dict)
                            elif cam_type == "intelrealsense":
                                from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
                                cameras[cam_name] = RealSenseCameraConfig(**cam_dict)
                            else:
                                raise ValueError(f"Unknown camera type: {cam_type}")
                        else:
                            cameras[cam_name] = cam_dict
                    robot_config_dict_clean["cameras"] = cameras

                robot_config = robot_config_class(**robot_config_dict_clean)
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

    # Create main config - use the extended orchestrator config if available
    try:
        # Try to import the extended config from agent module
        from lerobot.agent.config import OrchestratorConfig as AgentOrchestratorConfig
        config = AgentOrchestratorConfig(
            tasks=tasks,
            robot_config=robot_config,
            collision_config=collision_config,
            monitoring_config=monitoring_config,
            environment_dt=config_dict.get("environment_dt", 1.0 / 30.0),
            observation_timeout=config_dict.get("observation_timeout", 5.0),
            action_timeout=config_dict.get("action_timeout", 5.0),
            reset_duration=config_dict.get("reset_duration", 3.0),
            reset_positions=config_dict.get("reset_positions", {}),
        )
    except ImportError:
        # Fall back to base config
        config = OrchestratorConfig(
            tasks=tasks,
            robot_config=robot_config,
            collision_config=collision_config,
            monitoring_config=monitoring_config,
            environment_dt=config_dict.get("environment_dt", 1.0 / 30.0),
            observation_timeout=config_dict.get("observation_timeout", 5.0),
            action_timeout=config_dict.get("action_timeout", 5.0),
            reset_duration=config_dict.get("reset_duration", 3.0),
            reset_positions=config_dict.get("reset_positions", {}),
        )

    return config


# Note: draccus.register_choice_type is not available in current versions
# The dataclasses work without explicit registration when using YAML loading
