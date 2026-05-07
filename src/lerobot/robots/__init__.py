from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Import robot config modules to register their subclasses with draccus.
# For hardware-dependent robots (supre_robot_follower), we use importlib to
# load the config file directly, bypassing the package __init__.py which
# imports the main class and its hardware dependencies (eu_motor_py, etc.).
# This ensures the config class is registered with draccus even when hardware
# is not available, so --robot.type= choices work in CLI scripts.
import importlib.util
import sys
from pathlib import Path

_robots_dir = Path(__file__).parent


def _register_config(config_path: Path, module_name: str):
    """Load a robot config module to register its @register_subclass decorator.

    Uses importlib to load just the config file without triggering the
    package __init__.py, avoiding hardware dependency imports.
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(config_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except Exception:
        pass  # Config may not be available in this environment


# Register SupreRobotFollowerConfig (bypasses __init__.py → hardware deps)
_register_config(
    _robots_dir / "supre_robot_follower" / "supre_robot_follower_config.py",
    "lerobot.robots.supre_robot_follower.supre_robot_follower_config",
)

# Register ROS2 follower configs (no hardware deps, direct import is fine)
try:
    from lerobot.robots.ros2_follower.config_ros2_follower import ROS2FollowerConfig
    from lerobot.robots.ros2_follower.config_ros2_dual_follower import ROS2DualFollowerConfig
except ImportError:
    pass