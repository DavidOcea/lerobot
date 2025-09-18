from dataclasses import dataclass, field
from pathlib import Path
from lerobot.cameras import CameraConfig

from lerobot.teleoperators.config import TeleoperatorConfig

_DEFAULT_JOINT_CONFIG_PATH = "config_supre_robot_joint.yaml"
    
@TeleoperatorConfig.register_subclass("supre_robot_leader")
@dataclass
class SupreRobotLeaderConfig(TeleoperatorConfig):
    """Configuration for the SupreRobot."""
    joint_config_file: str = _DEFAULT_JOINT_CONFIG_PATH
    joint_direction: list= field(default_factory=lambda: [-1, -1, 1, 1, 1, -1,1, -1, -1, 1, 1, 1, -1,1])