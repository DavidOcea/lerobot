# Hardware (eu_motor_py, jodell_gripper_py) only available on the robot —
# skip on training machines where these C++ modules are not installed.
try:
    from .supre_robot_hardware_manager import SupreRobotHardwareManager
except ImportError:
    pass