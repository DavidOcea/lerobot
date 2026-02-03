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

@RobotConfig.register_subclass("ros2_follower")
@dataclass
class ROS2FollowerConfig(RobotConfig):
    """
    Configuration for the ROS2 Follower Robot (Right Arm).
    """

    name: str = "ros2_follower"
    num_joints: int = 7  # Default to 7, adjust as needed
    topic_joint_states: str = "/joint_states"
    topic_joint_trajectory: str = "/right_arm_controller/follow_joint_trajectory"
    topic_joint_positions: str = "/right_arm_controller/commands"
    topic_gripper = '/misumi_gripper_controller/commands'
    topic_gripper_joint_positions = '/misumi_gripper/joint_states'
    joint_name_prefix:str = "right_arm_joint_"
    observation_joint_name_prefix: str = "arm_joint_"
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_direction: list= field(default_factory=lambda: [1, -1, 1, 1, 1, 1, 1])
