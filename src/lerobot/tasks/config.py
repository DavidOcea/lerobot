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
    stability_tolerance: float = 0.05  # Max variation within window (rad, ~3° for physical arms)
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

    用于定义AGV导航任务，支持三种导航模式:
    1. 站点导航: target_station (导航到指定站点ID)
    2. 坐标导航: target_position (导航到指定坐标)
    3. 相对移动: translate_dist / turn_angle (平移/转动指定距离/角度)
    三种模式互斥，只能指定一种。
    """

    # ===== 模式1: 站点导航 =====
    target_station: str | None = None  # 目标站点ID

    # ===== 模式2: 坐标导航 =====
    target_position: tuple[float, float, float] | None = None  # (x, y, theta) 米/弧度

    # ===== 模式3: 相对移动 =====
    translate_dist: float | None = None  # 平移距离绝对值 (米), 方向由vx/vy决定
    translate_vx: float | None = None  # X方向速度 (米/秒), 正=前进, 负=后退
    translate_vy: float | None = None  # Y方向速度 (米/秒), 正=左移, 负=右移
    translate_mode: int = 0  # 平移模式 (0=默认)
    turn_angle: float | None = None  # 转动角度 (度), 正=逆时针, 负=顺时针
    turn_vw: float | None = None  # 转动角速度 (度/秒)
    turn_mode: int = 0  # 转动模式 (0=默认)

    # ===== 执行参数 =====
    max_duration: float = 60.0  # 最大执行时间 (秒)
    wait_for_arrival: bool = True  # 是否等待到达完成
    arrival_timeout: float = 60.0  # 到达等待超时 (秒)

    # ===== 到达判定 =====
    arrival_tolerance: float = 0.3  # 距离容差 (米)
    station_match_required: bool = True  # 必须站点ID匹配才算到达

    # ===== 安全检查 =====
    check_arm_safe_position: bool = True  # 移动前检查机械臂是否在安全位置
    arm_safe_positions: dict[str, float] = field(default_factory=dict)  # 安全偏差阈值
    arm_home_positions: dict[str, float] = field(default_factory=dict)  # 安全基准位置 (默认用硬编码 home)

    # ===== 错误处理 =====
    retry_on_timeout: bool = True  # 超时是否重试
    retry_count: int = 2  # 重试次数
    emergency_stop_on_error: bool = True  # 异常时急停

    # ===== 角度修正 =====
    angle_correction: bool = False  # 平移后自动修正角度偏差
    angle_correction_tolerance: float = 0.03  # 角度修正阈值 (弧度, ~1.7°)
    angle_correction_reference: str = "initial"  # 修正目标: "initial"(保持初始角) | "station"(对齐站点theta)

    def validate(self) -> bool:
        """验证配置有效性.

        三种导航模式互斥: target_station, target_position, translate_dist/turn_angle
        只能指定一种。
        """
        # Determine which navigation mode is specified
        has_station = self.target_station is not None
        has_position = self.target_position is not None
        has_translate = self.translate_dist is not None
        has_turn = self.turn_angle is not None
        modes_count = sum([has_station, has_position, has_translate, has_turn])

        if modes_count == 0:
            raise ValueError(
                "AGVTaskConfig: Must specify one of: target_station, "
                "target_position, translate_dist, or turn_angle"
            )
        if modes_count > 1:
            raise ValueError(
                f"AGVTaskConfig: Navigation modes are mutually exclusive. "
                f"Found {modes_count} modes specified, but only 1 allowed. "
                f"Choose one: target_station={self.target_station}, "
                f"target_position={self.target_position}, "
                f"translate_dist={self.translate_dist}, "
                f"turn_angle={self.turn_angle}"
            )

        # Mode-specific validation
        if has_translate:
            if self.translate_dist <= 0:
                raise ValueError(f"AGVTaskConfig: translate_dist must be positive (absolute value), got {self.translate_dist}")
        if has_turn:
            if self.turn_angle == 0:
                raise ValueError("AGVTaskConfig: turn_angle cannot be zero")

        # Common validation
        if self.max_duration <= 0:
            raise ValueError("AGVTaskConfig: max_duration must be positive")
        if self.arrival_timeout <= 0:
            raise ValueError("AGVTaskConfig: arrival_timeout must be positive")
        return True


@dataclass
class PositionSequenceStep:
    """A single step within a position_sequence task.

    Each step moves the robot to a target joint position using smooth interpolation.
    Position can be specified as:
    - A dict of {joint_name: position_in_degrees} (inline)
    - A string referencing a named_positions key (resolved during parsing)
    """

    name: str = ""
    position: dict[str, float] = field(default_factory=dict)
    max_duration: float = 10.0
    position_tolerance: float = 3.0


@dataclass
class VisualAlignConfig:
    """视觉精调任务配置 — AprilTag 标记引导 AGV 微调对准.

    流程: 检测 AprilTag → 解算位姿 → 先转向对准 → 再前进/后退 → 闭环验证收敛.
    由于 Seer AGV 的 vy (左移) 不生效, 只使用 "转向+前后平移" 策略.
    """

    # ===== 标记配置 =====
    marker_id: int | None = None  # 目标 AprilTag ID (None=搜索第一个可见标记)
    marker_size: float = 0.10  # 标记物理边长 (米), 位姿解算必须知道
    marker_family: str = "tag36h11"  # AprilTag 家族 (tag36h11, tag25h9, etc.)

    # ===== 搜索配置 =====
    search_turn_step: float = 10.0  # 搜索时每步转动角度 (度), 左转搜索
    search_max_turn: float = 90.0  # 最大总搜索转动角度 (度)
    search_max_attempts: int = 9  # 最大搜索步数 (90/10=9)

    # ===== 精度配置 =====
    position_tolerance: float = 0.02  # 位置收敛阈值 (米, 2cm)
    angle_tolerance: float = 2.0  # 角度收敛阈值 (度)
    max_iterations: int = 3  # 闭环最大迭代次数

    # ===== AGV 微调速度 =====
    translate_speed: float = 0.15  # 前进/后退微调速度 (m/s)
    turn_speed: float = 15.0  # 转动微调速度 (度/s)
    approach_distance: float = 0.50  # 目标接近距离 (米), 停在标记前方此距离处

    # ===== 参考照片对齐 (可选) =====
    # 设置后跳过 approach_distance, 用参考帧的位姿作为对齐目标
    reference_pose_path: str | None = None  # 参考位姿 JSON 文件路径
    distance_offset: float = 0.0  # 参考位姿距离修正 (米), 正值=多前进, 负值=多后退

    # ===== 相机-AGV 坐标变换 =====
    # 相机相对于 AGV 中心的偏移 (先用 (0,0,0) 近似, 闭环收敛弥补误差)
    camera_offset_x: float = 0.0  # 相机在 AGV 前后方向的偏移 (米)
    camera_offset_y: float = 0.0  # 相机在 AGV 左右方向的偏移 (米)
    camera_offset_yaw: float = 0.0  # 相机朝向相对 AGV 朝向的偏转 (度)
    camera_offset_pitch: float = 0.0  # 相机俯仰角 (度, 正值=俯视朝下)
    # pitch=0: 相机水平朝前, z_cam 即地面水平距离 (原默认行为)
    # pitch>0: 相机俯视, z_cam 是沿光轴距离, 需投影到地面水平面

    # ===== 相机内参 (可选, None=用默认估算值) =====
    camera_matrix: list | None = None  # 3x3 内参矩阵 (fx, fy, cx, cy), flat list
    dist_coeffs: list | None = None  # 畸变系数 (k1, k2, p1, p2, k3)

    # ===== 安全配置 =====
    check_arm_safe_position: bool = True  # AGV 微调前检查机械臂安全位置
    arm_safe_positions: dict[str, float] = field(default_factory=dict)  # 安全偏差阈值
    arm_home_positions: dict[str, float] = field(default_factory=dict)  # 安全基准位置 (可选)


@dataclass
class ClassifyConfig:
    """Configuration for workpiece classification tasks.

    Supports pluggable methods via the ``method`` field. Each method
    has its own set of parameters as nested fields.
    """

    method: str = "apriltag"  # classification method: "apriltag", "color", "model"

    # --- AprilTag method ---
    marker_id_map: dict[int, str] = field(default_factory=dict)
    # e.g. {1: "workpiece_type_1", 2: "workpiece_type_2"}
    marker_family: str = "tag36h11"
    marker_size: float = 0.05  # smaller tags for workpieces

    # --- Common ---
    default_label: str = "unknown"
    default_next_task: str = ""  # fallback if classification fails

    # --- YOLO method ---
    model_path: str = ""  # ONNX model path (e.g. best.onnx)
    classes: list = field(default_factory=lambda: ["short", "long"])  # cls_id → label
    conf_threshold: float = 0.50  # minimum YOLO confidence

    # --- Label counter (for alternating placements) ---
    label_counter_enable: bool = False  # enable label counting (e.g. long→long_1, long_2)
    counter_keywords: list = field(default_factory=list)  # which labels to count
    counter_modulo: int = 0  # if > 0, wrap count: 1→2→...→counter_modulo→1→... (0=no wrap)

    # --- No-detection retry ---
    retry_on_no_detect: bool = False  # wait and retry if nothing found
    retry_wait_seconds: float = 2.0   # seconds between retries
    retry_max_attempts: int = 10      # max retries before giving up
    retry_command: str = ""  # shell command run before each retry (e.g. TTS)
    recovery_task: str = ""  # task name to jump to after retries exhausted (""→FAIL)

    # ===== 安全配置 =====
    check_arm_safe_position: bool = True  # AGV 微调前检查机械臂安全位置
    arm_safe_positions: dict[str, float] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a single task in the execution sequence.

    Supports multiple task types:
    - "policy": Execute a learned policy (ACT, etc.)
    - "agv": Execute AGV navigation task
    - "position": Move robot to a single target joint position
    - "position_sequence": Move robot through a sequence of positions (sub-steps)
    - "visual_align": Use camera + AprilTag to guide AGV fine alignment
    """

    name: str

    # 任务类型
    task_type: Literal["policy", "agv", "position", "position_sequence", "visual_align", "classify", "system_command"] = "policy"

    # Policy任务字段 (现有)
    policy_path: str | None = None
    policy_type: str = "act"
    max_duration: float = 30.0  # Maximum execution time in seconds
    max_retries: int = 3  # Maximum number of retry attempts
    completion_criteria: CompletionCriteria = field(default_factory=CompletionCriteria)
    enabled: bool = True  # Allow disabling specific tasks
    cameras: list[CameraConfig] = field(default_factory=list)  # Active cameras for this task
    cycle_end: bool = False  # When True, orchestrator ends the current cycle after this task

    # AGV任务字段
    agv_config: AGVTaskConfig | None = None

    # Position sequence任务字段
    steps: list[PositionSequenceStep] = field(default_factory=list)

    # Visual align任务字段
    visual_align_config: VisualAlignConfig | None = None

    # Classify任务字段 (workpiece identification for conditional branching)
    classify_config: ClassifyConfig | None = None
    # Conditional branch routing: {label: next_task_name, ...}
    next_tasks: dict[str, str] = field(default_factory=dict)
    # system_command task field
    command: str = ""  # shell command to run (for system_command task_type)

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
        elif self.task_type == "position_sequence":
            if not self.steps:
                raise ValueError(f"Task '{self.name}': steps required for position_sequence tasks")
        elif self.task_type == "visual_align":
            if self.visual_align_config is None:
                raise ValueError(f"Task '{self.name}': visual_align_config required for visual_align tasks")
        elif self.task_type == "classify":
            if self.classify_config is None:
                raise ValueError(f"Task '{self.name}': classify_config required for classify tasks")
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

    # Named positions for reuse across tasks (optional)
    named_positions: dict[str, dict[str, float]] = field(default_factory=dict)

    # Execution settings
    environment_dt: float = 1.0 / 30.0  # Control loop timestep
    observation_timeout: float = 5.0  # Seconds to wait for observations
    action_timeout: float = 5.0  # Seconds to wait for actions
    speed_multiplier: float = 1.0  # Global speed scaling (1.0=normal, 2.0=2x). Clamped to [0.25, 4.0].

    # Reset settings
    reset_duration: float = 3.0  # Time in seconds for smooth reset to zero position
    reset_positions: dict[str, float] = field(default_factory=dict)  # Manual reset positions per joint (e.g., {"left_arm_joint_1": 0.0, ...}). If empty, use 0.0 for all joints.


def parse_task_dict(
    task_dict: dict[str, Any],
    named_positions: dict[str, dict[str, float]] | None = None,
    default_arm_safe_positions: dict[str, Any] | None = None,
    default_arm_home_positions: dict[str, Any] | None = None,
) -> TaskConfig:
    """Parse a single task dict into a fully-formed TaskConfig.

    Extracted from load_config_from_yaml so builtin skills can reuse the same
    parsing pipeline (AGV config, completion criteria, position_sequence steps,
    named position resolution).

    Args:
        task_dict: Raw task entry from YAML.
        named_positions: Optional mapping of position name → joint dict, used to
            resolve ``position: home`` shorthand references.
        default_arm_safe_positions: Optional default arm-safe positions inherited
        default_arm_home_positions: Optional default arm-home positions inherited
            by AGV tasks that don't specify their own.

    Returns:
        A fully-populated TaskConfig ready for execution.
    """
    named_positions = named_positions or {}
    default_arm_safe_positions = default_arm_safe_positions or {}
    default_arm_home_positions = default_arm_home_positions or {}

    task_type = task_dict.get("task_type", "policy")

    # Parse completion criteria if present
    criteria_dict = task_dict.get("completion_criteria", {})
    if criteria_dict:
        if "conditions" in criteria_dict:
            conditions = []
            for cond in criteria_dict["conditions"]:
                conditions.append(cond)
            criteria_dict["conditions"] = conditions

        # Resolve position: home shorthand
        if "position" in criteria_dict and isinstance(criteria_dict["position"], str):
            pos_name = criteria_dict["position"]
            if named_positions and pos_name in named_positions:
                criteria_dict["target_joint_positions"] = named_positions[pos_name].copy()
            else:
                raise ValueError(f"Unknown named position: '{pos_name}'")
            del criteria_dict["position"]
        elif "position" in criteria_dict and isinstance(criteria_dict["position"], dict):
            criteria_dict["target_joint_positions"] = criteria_dict["position"]
            del criteria_dict["position"]

        completion_criteria = CompletionCriteria(**criteria_dict)
    else:
        completion_criteria = CompletionCriteria()

    # Resolve string position references in completion_criteria
    if named_positions:
        if isinstance(completion_criteria.target_joint_positions, str):
            pos_name = completion_criteria.target_joint_positions
            if pos_name in named_positions:
                completion_criteria.target_joint_positions = named_positions[pos_name].copy()
            else:
                raise ValueError(f"Unknown named position: '{pos_name}'")

    # Parse cameras
    cameras_list = task_dict.get("cameras", [])
    cameras = [CameraConfig(**cam_dict) for cam_dict in cameras_list]

    # Parse AGV config
    agv_config_dict = task_dict.get("agv_config", {})
    agv_config = None
    if agv_config_dict:
        if default_arm_safe_positions and "arm_safe_positions" not in agv_config_dict:
            agv_config_dict["arm_safe_positions"] = default_arm_safe_positions
        if default_arm_home_positions and "arm_home_positions" not in agv_config_dict:
            agv_config_dict["arm_home_positions"] = default_arm_home_positions

        # Convert target_position list → tuple
        if "target_position" in agv_config_dict:
            pos = agv_config_dict["target_position"]
            if isinstance(pos, list) and len(pos) >= 2:
                theta = pos[2] if len(pos) >= 3 else 0.0
                agv_config_dict["target_position"] = (float(pos[0]), float(pos[1]), float(theta))

        agv_config = AGVTaskConfig(**agv_config_dict)

    # Base kwargs shared by all task types
    task_kwargs: dict[str, Any] = {
        "name": task_dict["name"],
        "task_type": task_type,
        "completion_criteria": completion_criteria,
        "cameras": cameras,
        "agv_config": agv_config,
        "cycle_end": task_dict.get("cycle_end", False),
    }

    # Task-type-specific fields
    if task_type == "policy":
        task_kwargs["policy_path"] = task_dict.get("policy_path")
        task_kwargs["policy_type"] = task_dict.get("policy_type", "act")
        task_kwargs["max_duration"] = task_dict.get("max_duration", 30.0)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 3)
        task_kwargs["enabled"] = task_dict.get("enabled", True)

    elif task_type == "agv":
        task_kwargs["max_duration"] = task_dict.get("max_duration", 60.0)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 2)
        task_kwargs["enabled"] = task_dict.get("enabled", True)

    elif task_type == "position":
        task_kwargs["max_duration"] = task_dict.get("max_duration", 10.0)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 1)
        task_kwargs["enabled"] = task_dict.get("enabled", True)

    elif task_type == "position_sequence":
        steps = []
        for step_dict in task_dict.get("steps", []):
            pos = step_dict.get("position", {})
            if isinstance(pos, str):
                if pos in named_positions:
                    pos = named_positions[pos].copy()
                else:
                    raise ValueError(f"Unknown named position: '{pos}'")
            steps.append(PositionSequenceStep(
                name=step_dict.get("name", ""),
                position=pos,
                max_duration=step_dict.get("max_duration", 10.0),
                position_tolerance=step_dict.get("position_tolerance", 3.0),
            ))
        task_kwargs["steps"] = steps
        total_steps_duration = sum(s.max_duration for s in steps)
        task_kwargs["max_duration"] = task_dict.get("max_duration", total_steps_duration)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 1)
        task_kwargs["enabled"] = task_dict.get("enabled", True)

    elif task_type == "visual_align":
        va_config_dict = task_dict.get("visual_align_config", {})
        if default_arm_safe_positions and "arm_safe_positions" not in va_config_dict:
            va_config_dict["arm_safe_positions"] = default_arm_safe_positions
        if default_arm_home_positions and "arm_home_positions" not in va_config_dict:
            va_config_dict["arm_home_positions"] = default_arm_home_positions
        task_kwargs["visual_align_config"] = VisualAlignConfig(**va_config_dict)
        task_kwargs["max_duration"] = task_dict.get("max_duration", 30.0)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 2)
        task_kwargs["enabled"] = task_dict.get("enabled", True)

    elif task_type == "classify":
        cc_dict = task_dict.get("classify_config", {})
        task_kwargs["classify_config"] = ClassifyConfig(**cc_dict)
        task_kwargs["max_duration"] = task_dict.get("max_duration", 10.0)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 1)
        task_kwargs["enabled"] = task_dict.get("enabled", True)
        # Conditional branch routing
        task_kwargs["next_tasks"] = task_dict.get("next_tasks", {})

    elif task_type == "system_command":
        task_kwargs["command"] = task_dict.get("command", "")
        task_kwargs["max_duration"] = task_dict.get("max_duration", 5.0)
        task_kwargs["max_retries"] = task_dict.get("max_retries", 1)
        task_kwargs["enabled"] = task_dict.get("enabled", True)

    return TaskConfig(**task_kwargs)


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

    # Parse named_positions first (needed for position reference resolution)
    named_positions = config_dict.get("named_positions", {})

    # Parse global agv_config for default_arm_safe_positions
    global_agv_config_dict = config_dict.get("agv_config", {})
    default_arm_safe_positions = global_agv_config_dict.get("default_arm_safe_positions", {})
    default_arm_home_positions = global_agv_config_dict.get("default_arm_home_positions", {})

    # Parse tasks
    tasks = []
    for task_dict in config_dict.get("tasks", []):
        tc = parse_task_dict(task_dict, named_positions, default_arm_safe_positions,
                             default_arm_home_positions)
        tasks.append(tc)

    # Parse robot config using draccus to handle polymorphic types
    robot_config_dict = config_dict.get("robot_config", {})
    # Extract action smoothing fields from robot_config (they belong to orchestrator, not robot)
    robot_enable_smoothing = robot_config_dict.pop("enable_action_smoothing", None)
    robot_smoothing_level = robot_config_dict.pop("action_smoothing_level", None)
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

    # Parse AGV global config
    from lerobot.agent.config import AGVGlobalConfig
    agv_config_dict = config_dict.get("agv_config", {})
    agv_global_config = AGVGlobalConfig(**agv_config_dict) if agv_config_dict else AGVGlobalConfig()

    # Create main config - use the extended orchestrator config if available
    try:
        # Try to import the extended config from agent module
        from lerobot.agent.config import OrchestratorConfig as AgentOrchestratorConfig
        config = AgentOrchestratorConfig(
            tasks=tasks,
            robot_config=robot_config,
            collision_config=collision_config,
            monitoring_config=monitoring_config,
            named_positions=named_positions,
            environment_dt=config_dict.get("environment_dt", 1.0 / 30.0),
            observation_timeout=config_dict.get("observation_timeout", 5.0),
            action_timeout=config_dict.get("action_timeout", 5.0),
            reset_duration=config_dict.get("reset_duration", 3.0),
            reset_positions=config_dict.get("reset_positions", {}),
            agv_config=agv_global_config,
            max_cycles=config_dict.get("max_cycles", 1),
            cycle_delay=config_dict.get("cycle_delay", 2.0),
            enable_cycle_prompt=config_dict.get("enable_cycle_prompt", True),
            enable_interactive_mode=config_dict.get("enable_interactive_mode", False),
            enable_action_smoothing=(
                robot_enable_smoothing if robot_enable_smoothing is not None
                else config_dict.get("enable_action_smoothing", True)
            ),
            action_smoothing_level=(
                robot_smoothing_level if robot_smoothing_level is not None
                else config_dict.get("action_smoothing_level", "medium")
            ),
            enable_monitoring_dashboard=config_dict.get("enable_monitoring_dashboard", False),
            monitoring_dashboard_port=config_dict.get("monitoring_dashboard_port", 8080),
        )
    except ImportError:
        # Fall back to base config
        config = OrchestratorConfig(
            tasks=tasks,
            robot_config=robot_config,
            collision_config=collision_config,
            monitoring_config=monitoring_config,
            named_positions=named_positions,
            environment_dt=config_dict.get("environment_dt", 1.0 / 30.0),
            observation_timeout=config_dict.get("observation_timeout", 5.0),
            action_timeout=config_dict.get("action_timeout", 5.0),
            reset_duration=config_dict.get("reset_duration", 3.0),
            reset_positions=config_dict.get("reset_positions", {}),
        )

    return config


# Note: draccus.register_choice_type is not available in current versions
# The dataclasses work without explicit registration when using YAML loading
