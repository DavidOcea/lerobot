from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

    
@dataclass
class SupreRobotLeaderConfig(RobotConfig):
    """Configuration for the SupreRobot."""
    config_path: str = "supre_robot_config.yaml"