from dataclasses import dataclass, field
from pathlib import Path
from lerobot.cameras import CameraConfig

from lerobot.teleoperators.config import TeleoperatorConfig

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "supre_robot_config.yaml"
    
@TeleoperatorConfig.register_subclass("supre_robot_leader")
@dataclass
class SupreRobotLeaderConfig(TeleoperatorConfig):
    """Configuration for the SupreRobot."""
    config_path: str = _DEFAULT_CONFIG_PATH