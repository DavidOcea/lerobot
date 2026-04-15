from dataclasses import dataclass, field
from pathlib import Path
from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig
# 1. 定义与 SupreRobot 相关的配置类
@dataclass
class MotorCalibration:
    joint_name:str
    min_position:float
    max_position:float


_DEFAULT_JOINT_CONFIG_PATH = "config_supre_robot_joint.yaml"

@RobotConfig.register_subclass("supre_robot_follower")
@dataclass
class SupreRobotFollowerConfig(RobotConfig):
    """Configuration for the SupreRobot."""
    joint_config_file: str = _DEFAULT_JOINT_CONFIG_PATH
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_direction: list= field(default_factory=lambda: [-1, -1, 1, 1, 1, -1,1, -1, -1, 1, 1, 1, -1,1])
    max_relative_joint_move: float = 15.0 #30.0
    prometheus_port: int | None = 8000
    control_frequency: int = 30
    # 速度读取开关（默认关闭以优化性能）
    enable_velocity_read: bool = False
    calibration:list[MotorCalibration] = field(default_factory=lambda: [
        MotorCalibration(
            joint_name="left_arm_joint_1",
            min_position=-160.0,
            max_position=160.0,
        ),
        MotorCalibration(
            joint_name="left_arm_joint_2",
            min_position=-90.0,
            max_position=0.0,
        ), 
        MotorCalibration(
            joint_name="left_arm_joint_3",
            min_position=-150.0, #-60.0
            max_position=150.0, #60.0
        ),
        MotorCalibration(
            joint_name="left_arm_joint_4",
            min_position=-90.0,
            max_position=0.0,
        ),
        MotorCalibration(
            joint_name="left_arm_joint_5",
            min_position=-150.0, #-60.0
            max_position=150.0, #60.0
        ),
        MotorCalibration(
            joint_name="left_arm_joint_6",
            min_position=-90.0,
            max_position=90.0,
        ),
        MotorCalibration(
            joint_name="left_arm_joint_7",
            min_position=0.0,
            max_position=1.0,
        ),        
        MotorCalibration(
            joint_name="right_arm_joint_1",
            min_position=-160.0,
            max_position=160.0,
        ),
        MotorCalibration(
            joint_name="right_arm_joint_2",
            min_position=0.0,
            max_position=90.0,
        ),
        MotorCalibration(
            joint_name="right_arm_joint_3",
            min_position=-150.0, #-60.0
            max_position=150.0, #-60.0
        ),
        MotorCalibration(
            joint_name="right_arm_joint_4",
            min_position=0.0,
            max_position=90.0,
        ),
        MotorCalibration(
            joint_name="right_arm_joint_5",
            min_position=-150.0, #-60.0
            max_position=150.0, #-60.0
        ),
        MotorCalibration(
            joint_name="right_arm_joint_6",
            min_position=-90.0,
            max_position=90.0,
        ),
        MotorCalibration(
            joint_name="right_arm_joint_7",
            min_position=0.0,
            max_position=1.0,
        ),           
    ])    