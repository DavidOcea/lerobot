from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Import supre_robot modules to trigger registration
from . import supre_robot_follower

# ROS2 follower is optional - only import if ROS2 is available
try:
    from . import ros2_follower
except ImportError:
    # ROS2 not installed, ros2_follower module unavailable
    pass
