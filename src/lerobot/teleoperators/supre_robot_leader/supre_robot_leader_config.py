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

    # ==================== 力反馈配置（抖动提示模式）====================
    # 当 Follower 遇到阻力时，Leader 产生抖动提示用户

    # 力反馈开关
    enable_force_feedback: bool = False  # 默认关闭，确认安全后再启用

    # ===== 抖动提示参数 =====
    force_threshold: float = 0.3         # 触发抖动的力阈值 (Nm)
    vibration_amplitude: float = 0.15    # 抖动幅度 (Nm)
    vibration_duration: float = 0.3      # 抖动持续时间 (s)
    vibration_frequency: float = 20.0    # 抖动频率 (Hz)
    force_debounce_count: int = 3        # 防抖计数

    # 电机参数
    rated_torque: float = 2.0            # 电机额定力矩 (Nm)