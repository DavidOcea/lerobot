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

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("ros2_dual_leader")
@dataclass
class ROS2DualLeaderConfig(TeleoperatorConfig):
    """
    Configuration for the ROS2 Leader Teleoperator (Both Arms).
    """

    name: str = "ros2_dual_leader"
    topic_joint_states: str = "/supre_robot_leader/joint_states"
    joint_name_prefix: str = "leader_"
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
