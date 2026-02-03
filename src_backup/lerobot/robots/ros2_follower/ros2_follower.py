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

import logging

import threading
import time
from typing import Any, ClassVar, Type

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Float64MultiArray
from rclpy.publisher import Publisher


from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from lerobot.robots.ros2_follower.config_ros2_follower import ROS2FollowerConfig
from lerobot.utils.shared_ros2_manager import SharedROS2Manager
from functools import cached_property
import wandb

logger = logging.getLogger(__name__)

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
        self._action_client: ActionClient | None = None        
        self._joint_state: JointState | None = None
        self._lock = threading.Lock()

        self._publisher: Publisher | None = None
        self._gripper_publisher: Publisher |None = None
        # IMPORTANT: Joint names are now constructed from the config's prefix.
        # Verify that `joint_name_prefix` and `num_joints` in your config match the robot.
        self.joint_names = [f"{self.config.joint_name_prefix}{i+1}" for i in range(self.config.num_joints)]
        self.observation_joint_names = [f"{self.config.observation_joint_name_prefix}{i+1}" for i in range(self.config.num_joints)]
        self.cameras = make_cameras_from_configs(config.cameras)
        self.joint_direction_map = {f"{self.observation_joint_names[i]}.pos": config.joint_direction[i] for i in range(len(self.observation_joint_names))}
      
        ### WANDB MODIFICATION START ###
        # 2. 在初始化时启动 wandb run
        try:
            wandb.init(
                project="gpttest", 
                name=f"{self.name}-run-{int(time.time())}",
                config={},
                reinit=True
            )
            # 定义 "timestamp" 为自定义 x 轴
            wandb.define_metric("timestamp")
            # 将所有 "action/" 开头的指标的 x 轴都设置为 "timestamp"
            wandb.define_metric("action/*", step_metric="timestamp")
            print("Weights & Biases initialized for ROS2RobotFollower, using 'timestamp' as metric.")
        except Exception as e:
            print(f"Could not initialize Weights & Biases. Error: {e}")
            wandb.init(mode="disabled")
        ### WANDB MODIFICATION END ###
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
            print(f"Follower joint states are missing some joints. Ignoring this message. joint_names: {self.joint_names} filterred_names: {filtered_names}")



    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        #return self._ros_node is not None and self._action_client is not None and self._action_client.server_is_ready() and all(cam.is_connected for cam in self.cameras.values())
        return self._ros_node is not None and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            print("Follower robot is already connected.")
            return

        # --- FIX: Call this BEFORE creating any ROS2 objects ---
        SharedROS2Manager.ensure_initialized()
        # Create the node but DO NOT spin it here
        self._ros_node = Node(f"{self.name}_robot_interface_{id(self)}")
        # --- CREATE ACTION CLIENT INSTEAD OF PUBLISHER ---
        # The topic_joint_trajectory from the config now refers to the action name
        #self._action_client = ActionClient(
        #    self._ros_node, FollowJointTrajectory, self.config.topic_joint_trajectory
        #)

        self._ros_node.create_subscription(
            JointState, self.config.topic_joint_states, self._joint_state_callback, 10
        )

        # Add the node to our shared manager, which handles the executor
        SharedROS2Manager.add_node(self._ros_node)

        # Wait for the action server to be available
        #print(f"Waiting for '{self.config.topic_joint_trajectory}' action server...")
        #if not self._action_client.wait_for_server(timeout_sec=5.0):
        #    self._ros_node.get_logger().error("Action server not available after waiting")
        #    self.disconnect()
        #    raise RuntimeError("FollowJointTrajectory action server not available.")
        #print("Action server found.")

        # 1. 创建一个发布者
        # -----------------
        # 主题名称遵循 ros2_control 的标准格式：/<控制器名称>/commands
        # 消息类型必须是 Float64MultiArray
        # QoS 配置文件的大小设为10，这是通用标准。
        topic_name = self.config.topic_joint_positions
        self.publisher_ = self._ros_node.create_publisher(Float64MultiArray, topic_name, 10)


        self._ros_node.get_logger().info('Waiting for subscriber to connect...')
        while self.publisher_.get_subscription_count() == 0:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1) # 短暂spin来处理事件
        print("Waiting for the first joint state message from the follower (left arm)...")
        # The while loop now works because the shared executor is spinning in a background thread
        start_time = time.time()
        while self._joint_state is None:
            if time.time() - start_time > 5: # Add a timeout
                raise RuntimeError("Failed to receive joint state for follower within 5 seconds.")
            time.sleep(0.1)
        print("Follower robot connected.")

        if calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def disconnect(self):
        if self.is_connected and self._ros_node is not None:
            # Action client doesn't need explicit destruction like a publisher
            self._action_client = None
            # Tell the manager to remove our node. It will handle shutdown if we are the last one.
            SharedROS2Manager.remove_node(self._ros_node)
            self._ros_node.destroy_node() # Still need to destroy the node itself
            self._ros_node = None
            print("Follower robot disconnected.")

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

        action_value = {}
        for i in range(self.config.num_joints):
            joint_name = self.joint_names[i]
            observation_joint_name = self.observation_joint_names[i]
            action_value[observation_joint_name] = pos_map[joint_name]    

        obs_dict = {f"{motor}.pos": val for motor, val in action_value.items()}
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        return obs_dict



    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("Follower robot is not connected.")

        if action is None:
            raise ValueError("Action dictionary must contain 'joint_positions'.")
        # modify the action by joint_direction_map
        action = {key: val * self.joint_direction_map[key] for key, val in action.items()}

        action_pos = {key.removesuffix(".pos"): val for key, val in action.items()}
        sorted_items = sorted(action_pos.items(), key=lambda item: int(item[0].split('_')[-1]))
        # sorted_items is now a list of (key, value) tuples, sorted correctly.
        #print(f"Received action: {sorted_items}")



        ### WANDB MODIFICATION START ###
        # 4. 在发送动作时，使用时间戳记录 action 数据
        current_timestamp = time.time()

        # 准备要记录的数据，键名使用 'action/' 前缀进行分组
        log_data = {f"action/{key}": value for key, value in sorted_items}
        
        # 将时间戳本身也添加到 log_data 中，这是定义 x 轴的关键
        log_data["timestamp"] = current_timestamp
        
        wandb.log(log_data)
        ### WANDB MODIFICATION END ###

        # Extract just the values from the sorted list
        target_positions = [value for key, value in sorted_items]
        #print(f"Sending action: {target_positions}")
        self.send_target_position(target_positions)
        return action

    def send_target_position(self, target_positions):
        """
        发送一组目标关节位置。

        Args:
            target_positions (list of float): 一个包含所有关节目标位置的列表。
                                              列表的顺序必须与控制器配置文件中的关节顺序严格一致。
        """
        # 2. 创建消息对象
        # -----------------
        msg = Float64MultiArray()

        # 3. 填充消息数据
        # -----------------
        # 将 Python 列表赋值给消息的 data 字段。
        msg.data = target_positions

        # 4. 发布消息
        # -------------
        self.publisher_.publish(msg)
        self._ros_node.get_logger().info(f'已发布目标位置: {msg.data}')

    def send_target_trajectory(self, target_positions: list[Any],action_client:ActionClient):
        # --- NEW ACTION CLIENT LOGIC ---
        goal_msg = FollowJointTrajectory.Goal()
        
        # Build the trajectory
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        #print(f"Joint names: {traj.joint_names} target_positions: {target_positions}")
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in target_positions]
        # Set a duration for the movement. Make this configurable.
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000
        traj.points.append(point)
        
        goal_msg.trajectory = traj
        
        self._ros_node.get_logger().info('Sending goal to action server...')
        
        # Send the goal asynchronously.
        # In a teleop loop, we don't wait for the result. We send a new goal
        # on the next tick, which will preempt the old one.
        action_client.send_goal_async(goal_msg)
    def home(self, **kwargs):
        """Return the robot to a pre-defined home position."""
        # Implement the logic to send the robot to a home position if required.
        print("Homing sequence not implemented for ROS2RobotFollower.")
        pass

    def configure(self) -> None:
        pass

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }    
    

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.observation_joint_names}
