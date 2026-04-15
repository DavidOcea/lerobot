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

    # 速度读取开关（默认关闭以优化性能）
    enable_velocity_read: bool = False

    # ==================== 力反馈配置（声音提示模式）====================
    # 当 Follower 遇到阻力时，播放声音提示用户
    # 阻力越大，声音越急促

    # 力反馈开关
    enable_force_feedback: bool = False  # 默认关闭

    # ===== 力检测参数 =====
    force_threshold: float = 0.3         # 触发声音的力阈值 (Nm)
    force_debounce_count: int = 3        # 防抖计数

    # ===== 声音参数 =====
    min_beep_interval: float = 0.1       # 最小蜂鸣间隔 (s) - 最大阻力时
    max_beep_interval: float = 1.0       # 最大蜂鸣间隔 (s) - 最小阻力时
    max_force_for_sound: float = 1.0     # 最大力参考值 (Nm)