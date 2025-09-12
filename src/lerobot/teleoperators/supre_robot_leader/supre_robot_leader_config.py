from dataclasses import dataclass, field
from pathlib import Path
from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "supre_robot_config.yaml"
    
@RobotConfig.register_subclass("supre_robot_leader")
@dataclass
class SupreRobotLeaderConfig(RobotConfig):
    """Configuration for the SupreRobot."""
    config_path: str = _DEFAULT_CONFIG_PATH