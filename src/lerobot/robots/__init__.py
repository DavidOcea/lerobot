from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Import supre_robot modules to trigger registration.
# Hardware (eu_motor_py, jodell_gripper_py) only available on the robot —
# skip on training machines where these C++ modules are not installed.
try:
    from . import supre_robot_follower
except ImportError:
    pass

# ROS2 follower is optional - only import if ROS2 is available
try:
    from . import ros2_follower
except ImportError:
    # ROS2 not installed, ros2_follower module unavailable
    pass
