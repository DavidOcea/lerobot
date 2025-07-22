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

    def _joint_state_callback(self, msg: JointState):
        with self._lock:
            self._joint_state = msg

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
        Reads the leader's (right arm) current state and formats it as an action
        for the follower (left arm).
        """
        if not self.is_connected:
            raise RuntimeError("Leader teleoperator is not connected.")

        with self._lock:
            if self._joint_state is None:
                raise RuntimeError("Leader joint states are not being received.")
            state = self._joint_state

        pos_map = dict(zip(state.name, state.position))

        # The action for the follower is the position of the leader's joints.
        return {
            "joint_positions": np.array([pos_map.get(name, 0.0) for name in self.joint_names]),
        }


    def configure(self) -> None:
        pass
    @property
    def action_features(self) -> dict:
        return {
            "joint_positions": (self.config.num_joints,),
        }
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
    