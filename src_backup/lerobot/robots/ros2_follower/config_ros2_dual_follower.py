# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@dataclass
class MotorCalibration:
    joint_name:str
    min_position:float
    max_position:float
@RobotConfig.register_subclass("ros2_dual_follower")
@dataclass
class ROS2DualFollowerConfig(RobotConfig):
    """
    Configuration for the ROS2 Follower Robot (Right Arm).
    """

    name: str = "ros2_dual_follower"
    topic_joint_states: str = "/supre_robot_follower/joint_states"
    topic_joint_positions_right: str = "/supre_robot_follower/right_arm_controller/commands"
    topic_joint_positions_left: str = "/supre_robot_follower/left_arm_controller/commands"
    joint_name_prefix:str = "follower_"
    prometheus_port: int | None = 8000
    joint_names: list[str] = field(default_factory=lambda:[
        'left_arm_joint_1',
        'left_arm_joint_2',
        'left_arm_joint_3',
        'left_arm_joint_4',
        'left_arm_joint_5',
        'left_arm_joint_6',     
        'left_arm_joint_7',
        'right_arm_joint_1',
        'right_arm_joint_2',
        'right_arm_joint_3',
        'right_arm_joint_4',
        'right_arm_joint_5',
        'right_arm_joint_6',
        'right_arm_joint_7',
    ])
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_direction: list= field(default_factory=lambda: [-1, -1, 1, 1, 1, -1,1, -1, -1, 1, 1, 1, -1,1])
    max_relative_joint_move: float = 30.0
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