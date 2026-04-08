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
    # WARNING: CST mode causes motors to actively move, not provide damping!
    # This feature is DANGEROUS and should remain DISABLED until proper implementation.
    # The CST (Cyclic Synchronous Torque) mode makes motors output torque and move actively,
    # which is the opposite of what we want (damping/resistance feeling).

    # Force feedback is DISABLED by default for safety
    enable_force_feedback: bool = False  # DANGER: Set to True only after proper implementation

    # The following parameters are kept for future reference but will not be used
    # when enable_force_feedback is False

    # 阻尼参数
    damping_gain: float = 0.3           # 阻尼增益 (Nm/Nm)
    max_damping_torque: float = 0.5     # 最大阻尼力矩 (Nm)

    # 电机参数
    rated_torque: float = 2.0           # 电机额定力矩 (Nm)

    # 力信号滤波
    force_filter_alpha: float = 0.7     # 低通滤波系数 (0-1)