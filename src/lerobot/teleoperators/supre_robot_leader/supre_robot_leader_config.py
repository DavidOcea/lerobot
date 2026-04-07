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

    # ==================== 力反馈配置 ====================
    # 用于将 Follower 的力数据转换为 Leader 的阻尼力矩

    # 力反馈开关
    enable_force_feedback: bool = True

    # 阻尼参数
    damping_gain: float = 0.3           # 阻尼增益 (Nm/Nm)
    max_damping_torque: float = 0.5     # 最大阻尼力矩 (Nm)

    # 电机参数
    rated_torque: float = 2.0           # 电机额定力矩 (Nm)

    # 力信号滤波
    force_filter_alpha: float = 0.7     # 低通滤波系数 (0-1)