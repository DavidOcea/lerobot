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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ..robot import Robot
from lerobot.robots.ros2_follower.config_ros2_follower import ROS2FollowerConfig


class ROS2RobotFollower(Robot):
    """
    The "Follower" robot, representing the left arm.
    It receives actions and applies them, and provides its own state as an observation.
    """

    config_class: ClassVar[Type[ROS2FollowerConfig]] = ROS2FollowerConfig
    name: str = "ros2_follower"

    def __init__(self, config: ROS2FollowerConfig):
        super().__init__(config)
        self.config = config
        self._ros_node: Node | None = None
        self._ros_thread: threading.Thread | None = None
        self._publisher = None
        self._joint_state: JointState | None = None
        self._lock = threading.Lock()

        # IMPORTANT: Joint names are now constructed from the config's prefix.
        # Verify that `joint_name_prefix` and `num_joints` in your config match the robot.
        self.joint_names = [f"{self.config.joint_name_prefix}{i+1}" for i in range(self.config.num_joints)]

    def _ros_spin(self):
        rclpy.spin(self._ros_node)

    def _joint_state_callback(self, msg: JointState):
        with self._lock:
            self._joint_state = msg

    @property
    def observation_features(self) -> dict:
        return {
            "joint_positions": (self.config.num_joints,),
            "joint_velocities": (self.config.num_joints,),
        }

    @property
    def action_features(self) -> dict:
        return {
            "joint_positions": (self.config.num_joints,),
        }

    @property
    def is_connected(self) -> bool:
        return self._ros_node is not None and self._publisher is not None

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            print("Follower robot is already connected.")
            return

        try:
            rclpy.init()
        except RuntimeError:
            pass  # Already initialized

        self._ros_node = Node(f"{self.name}_robot_interface")
        self._publisher = self._ros_node.create_publisher(
            JointTrajectory, self.config.topic_joint_trajectory, 10
        )
        self._ros_node.create_subscription(
            JointState, self.config.topic_joint_states, self._joint_state_callback, 10
        )

        self._ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._ros_thread.start()

        print("Waiting for the first joint state message from the follower (left arm)...")
        while self._joint_state is None:
            time.sleep(0.1)
        print("Follower robot connected.")

        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self):
        print("Follower calibration is assumed to be handled by the robot's internal systems.")
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("Follower robot is not connected.")

        with self._lock:
            if self._joint_state is None:
                raise RuntimeError("Follower joint states are not being received.")
            state = self._joint_state

        pos_map = dict(zip(state.name, state.position))
        vel_map = dict(zip(state.name, state.velocity))

        return {
            "joint_positions": np.array([pos_map.get(name, 0.0) for name in self.joint_names]),
            "joint_velocities": np.array([vel_map.get(name, 0.0) for name in self.joint_names]),
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("Follower robot is not connected.")

        target_positions = action.get("joint_positions")
        if target_positions is None:
            raise ValueError("Action dictionary must contain 'joint_positions'.")
        return {"sent_joint_positions": target_positions}

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in target_positions]
        point.time_from_start.sec = 1  # Move in 1 second, adjust if needed
        traj_msg.points.append(point)

        #self._publisher.publish(traj_msg)
        return {"sent_joint_positions": target_positions}

    def disconnect(self):
        if self.is_connected:
            self._ros_node.destroy_node()
            # Do not shutdown rclpy here, as the leader might still be using it
            self._ros_node = None
            self._publisher = None
            print("Follower robot disconnected.")

    def home(self, **kwargs):
        """Return the robot to a pre-defined home position."""
        # Implement the logic to send the robot to a home position if required.
        print("Homing sequence not implemented for ROS2RobotFollower.")
        pass

    def configure(self) -> None:
        pass