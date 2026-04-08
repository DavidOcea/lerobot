from dataclasses import dataclass, field
from pathlib import Path
from lerobot.cameras import CameraConfig

from lerobot.teleoperators.config import TeleoperatorConfig

_DEFAULT_JOINT_CONFIG_PATH = "config_supre_robot_joint.yaml"

@TeleoperatorConfig.register_subclass("supre_robot_leader")
@dataclass
class SupreRobotLeaderConfig(TeleoperatorConfig):
    """Configuration for the SupreRobot Leader (teleoperation device)."""
    joint_config_file: str = _DEFAULT_JOINT_CONFIG_PATH
    joint_direction: list = field(default_factory=lambda: [-1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1])

    # ==================== 力反馈配置（安全版本）====================
    # 核心安全原则：只有在用户移动时才提供阻力，静止时力矩为零

    # 力反馈开关
    enable_force_feedback: bool = False  # 默认关闭，确认安全后再启用

    # ===== 阻尼参数 =====
    damping_gain: float = 0.5             # 阻尼增益 (Nm/Nm)
    max_damping_torque: float = 0.3       # 最大阻尼力矩限制 (Nm)，安全起见设小
    rated_torque: float = 2.0             # 电机额定力矩 (Nm)
    force_filter_alpha: float = 0.5       # 低通滤波系数 (0-1)

    # ===== 安全参数（关键！）=====
    # 速度死区：小于此速度不发力矩（静止时不发力矩，防止失控）
    velocity_deadband: float = 2.0        # 速度死区 (°/s)

    # 速度安全阈值：超过此速度立即切断力矩（防止失控）
    max_velocity_threshold: float = 100.0  # 最大速度阈值 (°/s)

    # 速度调制系数：速度越快，阻尼越强
    velocity_scale: float = 30.0          # 速度调制系数 (°/s)

    # 力矩安全限制
    torque_safety_margin: float = 0.1     # 力矩安全余量 (Nm)