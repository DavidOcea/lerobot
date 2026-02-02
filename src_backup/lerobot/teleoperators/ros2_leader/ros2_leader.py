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

import threading
import time
from typing import Any, ClassVar, Type

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ..teleoperator import Teleoperator
from lerobot.teleoperators.ros2_leader.config_ros2_leader import ROS2LeaderConfig
from lerobot.utils.shared_ros2_manager import SharedROS2Manager
from functools import cached_property

class ROS2RobotLeader(Teleoperator):
    """
    The "Leader" teleoperator, representing the right arm.
    It reads its own joint states and provides them as actions for the follower.
    """

    config_class: ClassVar[Type[ROS2LeaderConfig]] = ROS2LeaderConfig
    name: str = "ros2_leader"

    def __init__(self, config: ROS2LeaderConfig):
        super().__init__(config)
        self.config = config
        self._ros_node: Node | None = None
        self._ros_thread: threading.Thread | None = None
        self._joint_state: JointState | None = None
        self._lock = threading.Lock()

        # IMPORTANT: Joint names are now constructed from the config's prefix.
        # Verify that `joint_name_prefix` and `num_joints` in your config match the robot.
        self.joint_names = [f"{self.config.joint_name_prefix}{i+1}" for i in range(self.config.num_joints)]
        self.observation_joint_names = [f"{self.config.observation_joint_name_prefix}{i+1}" for i in range(self.config.num_joints)]
    def _joint_state_callback(self, msg: JointState):
        # Prepare lists to hold the filtered data
        filtered_names = []
        filtered_positions = []
        filtered_velocities = []
        
        # Check if the incoming message has position and velocity data
        has_positions = len(msg.position) == len(msg.name)
        has_velocities = len(msg.velocity) == len(msg.name)
        
        #print(f"Received joint states: {msg.name}")
        # Iterate through the received message and pick out the joints we care about
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                filtered_names.append(name)
                if has_positions:
                    filtered_positions.append(msg.position[i])
                if has_velocities:
                    filtered_velocities.append(msg.velocity[i])
        
        # --- Validation Step ---
        # Only accept the message if it contains ALL the joints we need.
        # This prevents storing a partial state.
        if len(filtered_names) == len(self.joint_names):
            # Create a new, clean JointState message
            filtered_msg = JointState()
            filtered_msg.header = msg.header
            filtered_msg.name = filtered_names
            filtered_msg.position = filtered_positions
            filtered_msg.velocity = filtered_velocities
            with self._lock:
                self._joint_state = filtered_msg
        else: 
            print(f"Leader joint states are missing some joints. Ignoring this message. joint_names: {self.joint_names} filterred_names: {filtered_names}")

    @property
    def is_connected(self) -> bool:
        # 更稳健的检查：节点存在且rclpy仍在运行
        return self._ros_node is not None and rclpy.ok()

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            print("Leader teleoperator is already connected.")
            return
        # --- FIX: Call this BEFORE creating any ROS2 objects ---
        SharedROS2Manager.ensure_initialized()
        # 3. 不再手动初始化 rclpy 或创建线程
        # 节点创建保持不变
        self._ros_node = Node(f"{self.name}_teleop_interface_{id(self)}")
        self._ros_node.create_subscription(
            JointState, self.config.topic_joint_states, self._joint_state_callback, 10
        )

        # 将节点添加到共享管理器，由它负责启动和管理执行器
        SharedROS2Manager.add_node(self._ros_node)

        print("Waiting for the first joint state message from the leader (right arm)...")
        # 因为共享执行器已在后台运行，这个循环现在可以正常工作了
        start_time = time.time()
        while self._joint_state is None:
            if time.time() - start_time > 5: # 增加5秒超时
                raise RuntimeError("Failed to receive joint state for leader within 5 seconds.")
            time.sleep(0.1)
        print("Leader teleoperator connected.")

    def disconnect(self):
        if self.is_connected and self._ros_node is not None:
            # 4. 通知共享管理器移除此节点
            # 管理器会在最后一个节点被移除时自动关闭执行器
            SharedROS2Manager.remove_node(self._ros_node)

            # 销毁节点本身
            self._ros_node.destroy_node()
            self._ros_node = None
            print("Leader teleoperator disconnected.")

    def get_action(self) -> dict[str, Any]:
        """
        Reads the leader's (left arm) current state and formats it as an action
        for the follower (right arm).
        """
        if not self.is_connected:
            raise RuntimeError("Leader teleoperator is not connected.")

        with self._lock:
            if self._joint_state is None:
                raise RuntimeError("Leader joint states are not being received.")
            state = self._joint_state

        pos_map = dict(zip(state.name, state.position))
        action_value = {}
        for i in range(self.config.num_joints):
            joint_name = self.joint_names[i]
            observation_joint_name = self.observation_joint_names[i]
            action_value[observation_joint_name] = pos_map[joint_name]
        # The action for the follower is the position of the leader's joints.
        action = {f"{m}.pos":v for m,v in action_value.items()}
        return action


    def configure(self) -> None:
        pass
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.observation_joint_names}
    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
    
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError
    