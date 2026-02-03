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
from lerobot.robots.ros2_follower.config_ros2_dual_follower import ROS2DualFollowerConfig
from lerobot.utils.shared_ros2_manager import SharedROS2Manager
from functools import cached_property
from ..utils import ensure_safe_goal_position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from lerobot.utils.prometheus_manager import prometheus_manager

import wandb

logger = logging.getLogger(__name__)

class ROS2DualRobotFollower(Robot):
    """
    The "Follower" robot, representing the left arm.
    It receives actions and applies them, and provides its own state as an observation.
    """

    config_class: ClassVar[Type[ROS2DualFollowerConfig]] = ROS2DualFollowerConfig
    name: str = "ros2_dual_follower"

    def __init__(self, config: ROS2DualFollowerConfig):
        super().__init__(config)
        self.config = config
        self._ros_node: Node | None = None
        self._ros_thread: threading.Thread | None = None
        self._action_client: ActionClient | None = None        
        self._joint_state: JointState | None = None
        self._lock = threading.Lock()

        self._left_arm_publisher: Publisher | None = None
        self._right_arm_publisher: Publisher | None = None
        # IMPORTANT: Joint names are now constructed from the config's prefix.
        # Verify that `joint_name_prefix` and `num_joints` in your config match the robot.
        self.joint_names = [f"{self.config.joint_name_prefix}{name}" for name in self.config.joint_names]
        self.observation_joint_names = self.config.joint_names
        self.cameras = make_cameras_from_configs(config.cameras)
        self.joint_direction_map = {f"{self.observation_joint_names[i]}.pos": config.joint_direction[i] for i in range(len(self.observation_joint_names))}
      
        # 将 calibration 列表转换为一个字典以便快速查找
        # key: joint_name, value: MotorCalibration object
        self.calibration_limits = {cal.joint_name: cal for cal in self.config.calibration}
        # 增加一个检查，确保所有在 joint_names 中的关节都有对应的 calibration 设置
        for joint_name in self.observation_joint_names:
            if joint_name not in self.calibration_limits:
                raise ValueError(f"Missing calibration data for joint '{joint_name}' in config.")
        ### WANDB MODIFICATION START ###
        # 2. 在初始化时启动 wandb run
        '''
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
        '''
        self.prometheus_port = getattr(config, 'prometheus_port', None)
        self.joint_position_gauge = None
        if self.prometheus_port is not None:
            self.joint_position_gauge = prometheus_manager.get_gauge('joint_position')        
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

        qos_profile_controller_compatible = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # 对于指令，通常只关心最新的
        )
        topic_name = self.config.topic_joint_positions_left
        self.left_arm_publisher_ = self._ros_node.create_publisher(Float64MultiArray, 
                                                                   topic_name,
                                                                   qos_profile=qos_profile_controller_compatible
                                                                   )


        self._ros_node.get_logger().info('Waiting for left arm subscriber to connect...')
        while self.left_arm_publisher_ .get_subscription_count() == 0:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1) # 短暂spin来处理事件

        topic_name = self.config.topic_joint_positions_right
        self.right_arm_publisher_ = self._ros_node.create_publisher(Float64MultiArray, 
                                                                    topic_name, 
                                                                    qos_profile=qos_profile_controller_compatible 
                                                                    )


        self._ros_node.get_logger().info('Waiting for right arm subscriber to connect...')
        while self.right_arm_publisher_ .get_subscription_count() == 0:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1) # 短暂spin来处理事件

        print("Waiting for the first joint state message from the follower ...")        
        # The while loop now works because the shared executor is spinning in a background thread
        start_time = time.time()
        while self._joint_state is None:
            if time.time() - start_time > 50: # Add a timeout
                raise RuntimeError("Failed to receive joint state for follower within 50 seconds.")
            time.sleep(0.1)
        print("Follower robot connected.")

        if calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        if self.prometheus_port is not None:
            prometheus_manager.start_server(self.prometheus_port)

    def disconnect(self):
        if self._ros_node is not None:
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
        for i in range(len(self.joint_names)):
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


    def get_current_position(self) -> dict[str, float]:
        with self._lock:
            if self._joint_state is None:
                raise RuntimeError("Leader joint states are not being received.")
            state = self._joint_state

        pos_map = dict(zip(state.name, state.position))
        action_value = {}
        for i in range(len(self.joint_names)):
            joint_name = self.joint_names[i]
            observation_joint_name = self.observation_joint_names[i]
            action_value[observation_joint_name] = pos_map[joint_name]
        return action_value
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._ros_node.get_logger().info(f"Sending action: {action}")
        if not self.is_connected:
            raise RuntimeError("Follower robot is not connected.")

        if action is None:
            raise ValueError("Action dictionary must contain 'joint_positions'.")
        # modify the action by joint_direction_map
        action = {key: val * self.joint_direction_map[key] for key, val in action.items()}

        action_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        ensure_safe = False
        if ensure_safe:
            # 1. --- GET CURRENT STATE (Now much cleaner!) ---
            present_positions_map = self.get_current_position()
            # 2. --- PREPARE DATA FOR SAFETY CHECK ---
            # Create the `goal_present_pos` dictionary required by the safety function.
            # This part is now simpler because both dicts use observation_joint_names as keys.
            goal_present_pos = {}
            for obs_name in self.observation_joint_names:
                try:
                    goal_pos = action_pos[obs_name]
                    present_pos = present_positions_map[obs_name]
                    goal_present_pos[obs_name] = (goal_pos, present_pos)
                except KeyError as e:
                    raise ValueError(f"Could not find required joint '{e}' in action or current state.")
            
            # 3. --- APPLY THE SAFETY FUNCTION ---
            # Call `ensure_safe_goal_position` to get the clamped goal positions.
            # (Remember to add `max_relative_joint_move` to your config class)
            safe_goal_positions_map = ensure_safe_goal_position(
                goal_present_pos,
                self.config.max_relative_joint_move 
            )

        if ensure_safe:
            # 4. --- USE THE SAFE GOAL POSITIONS ---
            # Reconstruct the target_positions list using the SAFE values,
            # ensuring the correct order.
            target_positions = [safe_goal_positions_map[name] for name in self.observation_joint_names]
                
            # Create `sorted_items` for logging purposes, using the safe values.
            sorted_items = list(zip(self.observation_joint_names, target_positions))   
        else:                  
            # sorted_items is now a list of (key, value) tuples, sorted correctly.
            try:
                # Create the list of target positions by iterating through the canonical joint names
                target_positions = [action_pos[name] for name in self.observation_joint_names]
                
                # Create `sorted_items` for logging purposes, ensuring it has the same correct order.
                sorted_items = list(zip(self.observation_joint_names, target_positions))
                
            except KeyError as e:
                # This error handling is crucial. It tells you if the received action is missing a joint.
                raise ValueError(f"Action dictionary is missing a required joint: {e}. Provided joints: {list(action_pos.keys())}") from e

        final_clamped_positions = []
        warnings = {}

        # 我们需要按顺序遍历关节，以保持 target_positions 列表的顺序
        for i, joint_name in enumerate(self.observation_joint_names):
            # 获取当前关节的目标位置
            target_pos = target_positions[i]
            
            # 从我们预处理好的字典中查找限制
            limits = self.calibration_limits[joint_name]
            
            # 执行钳位操作
            clamped_pos = max(limits.min_position, min(target_pos, limits.max_position))
            
            # 如果发生了钳位，记录下来以便发出警告
            if abs(clamped_pos - target_pos) > 1e-4:
                warnings[joint_name] = {
                    "original": target_pos,
                    "clamped": clamped_pos,
                    "limits": (limits.min_position, limits.max_position)
                }
            
            final_clamped_positions.append(clamped_pos)
        
        # 如果有任何关节被限制了，打印一条总的警告信息
        if warnings:
            # 可以在这里使用 logging.warning 来代替 print
            self._ros_node.get_logger().warn(
                "One or more joint positions were clamped to their absolute limits:"
            )
            for joint, data in warnings.items():
                self._ros_node.get_logger().warn(
                    f"  - Joint '{joint}': commanded {data['original']:.4f}, "
                    f"clamped to {data['clamped']:.4f} (limits: {data['limits']})",
                    throttle_duration_sec=5 # 5秒内不重复打印相同的警告
                )
        
        # 使用经过两层安全检查后的最终位置
        final_target_positions = final_clamped_positions

        if self.joint_position_gauge:
            for joint_name, position in zip(self.joint_names, final_target_positions):
                # 使用 'leader' 作为 robot_name
                self.joint_position_gauge.labels(
                    robot_name='follower', 
                    joint_name=joint_name,
                    joint_id=self.observation_joint_names[self.joint_names.index(joint_name)]
                ).set(position)

        # 同时更新 sorted_items 以便 wandb 记录正确的值
        sorted_items = list(zip(self.observation_joint_names, final_target_positions))

        ### WANDB MODIFICATION START ###
        # 4. 在发送动作时，使用时间戳记录 action 数据
        current_timestamp = time.time()

        # 准备要记录的数据，键名使用 'action/' 前缀进行分组
        log_data = {f"action/{key}": value for key, value in sorted_items}
        
        # 将时间戳本身也添加到 log_data 中，这是定义 x 轴的关键
        log_data["timestamp"] = current_timestamp
        
        #wandb.log(log_data)
        ### WANDB MODIFICATION END ###

        # Extract just the values from the sorted list
        #print(f"Sending action: {target_positions}")
        self.send_target_position(final_target_positions)
        # 首先，创建一个包含最终执行值的字典 (key: 'left_arm_joint_1', value: final_pos)
        final_action = {
            f"{name}.pos": pos 
            for name, pos in zip(self.observation_joint_names, final_target_positions)
        }
        
        #self._ros_node.get_logger().info(f"Returning final clamped action: {final_action}")
        return final_action

    def send_target_position(self, target_positions):
        """
        Splits the combined target position list and sends commands to the left and right arm publishers.

        Args:
            target_positions (list of float): A list containing target positions for all joints,
                                              ordered as [left_arm_joints..., right_arm_joints...].
        """
        self._ros_node.get_logger().info(f"Sending target positions:{target_positions}")
        if self.left_arm_publisher_ is None or self.right_arm_publisher_ is None:
            self._ros_node.get_logger().error("Arm publishers are not initialized!")
            return

        # Determine the split point based on the number of joints in the left arm
        num_left_joints = 7
        
        # Check if the received action has the correct total number of joints
        if len(target_positions) != len(self.observation_joint_names):
             self._ros_node.get_logger().error(
                f"Incorrect number of target positions received. Expected {len(self.observation_joint_names)}, "
                f"but got {len(target_positions)}. Aborting send."
            )
             return

        # Split the list into left and right arm positions
        left_positions = target_positions[:num_left_joints]
        right_positions = target_positions[num_left_joints:]

        # Create and publish the message for the left arm
        left_msg = Float64MultiArray()
        left_msg.data = left_positions
        result = self.left_arm_publisher_.publish(left_msg)
        #self._ros_node.get_logger().info(f'Published to left arm: {left_msg.data} result: {result}')

        # Create and publish the message for the right arm
        right_msg = Float64MultiArray()
        right_msg.data = right_positions
        result = self.right_arm_publisher_.publish(right_msg)
        #self._ros_node.get_logger().info(f'Published to right arm: {right_msg.data} result: {result}')

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
        
        #self._ros_node.get_logger().info('Sending goal to action server...')
        
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
