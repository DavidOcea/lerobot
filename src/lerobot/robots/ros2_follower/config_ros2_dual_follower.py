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

@RobotConfig.register_subclass("ros2_dual_follower")
@dataclass
class ROS2DualFollowerConfig(RobotConfig):
    """
    Configuration for the ROS2 Follower Robot (Right Arm).
    """

    name: str = "ros2_dual_follower"
    num_joints: int = 7  # Default to 7, adjust as needed
    topic_joint_states: str = "/supre_robot_follower/joint_states"
    topic_joint_positions_right: str = "/supre_robot_follower/right_arm_controller/commands"
    topic_joint_positions_left: str = "/supre_robot_follower/left_arm_controller/commands"
    joint_name_prefix:str = "follower_"
    joint_names: list[str] = field(default_factory=lambda:[
        'left_arm_joint_1',
        'left_arm_joint_2',
        'left_arm_joint_3',
        'left_arm_joint_4',
        'left_arm_joint_5',
        'left_arm_joint_6',
        'right_arm_joint_1',
        'right_arm_joint_2',
        'right_arm_joint_3',
        'right_arm_joint_4',
        'right_arm_joint_5',
        'right_arm_joint_6',
    ])
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_direction: list= field(default_factory=lambda: [-1, -1, 1, 1, 1, 1,-1, -1, 1, 1, 1, 1])
    max_relative_joint_move: float = 5.0
