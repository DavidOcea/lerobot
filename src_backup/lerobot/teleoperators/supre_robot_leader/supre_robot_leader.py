# supre_robot.py

import time
import math
from typing import Any, Dict, List, Optional, Tuple, Type
from pathlib import Path

import yaml
import numpy as np
import dataclasses

# 导入我们之前设计的硬件管理器
from lerobot.robots.supre_robot import SupreRobotHardwareManager
# from eyou_hardware import EyouMotorHardware  # Manager will import these
# from gripper_hardware import JodellGripperHardware # Manager will import these
from ..teleoperator import Teleoperator
from .supre_robot_leader_config import SupreRobotLeaderConfig
from functools import cached_property
from lerobot.utils.prometheus_manager import prometheus_manager

# 2. 实现 Robot 接口
class SupreRobotLeader(Teleoperator):
    """
    A LeRobot-compatible class for the dual-arm robot controlled by
    Eyou motors and Jodell grippers.
    """
    # 设置 LeRobot 要求的类属性
    config_class = SupreRobotLeaderConfig
    name = "supre_robot_leader"

    def __init__(self, config: SupreRobotLeaderConfig):
        super().__init__(config)
        self.config = config
        self._hardware_manager: Optional[SupreRobotHardwareManager] = None
        self._is_connected_flag = False
        
        # 为了让 observation_features 和 action_features 可以在 connect() 之前被调用，
        # 我们需要提前加载关节顺序。
        config.joint_config_path = str(Path(__file__).resolve().parent/config.joint_config_file)
        try:
            with open(config.joint_config_path, 'r') as f:
                robot_yaml_config = yaml.safe_load(f)
            self._joint_order = robot_yaml_config["joint_order"]
            self.num_joints = len(self._joint_order)
            self.observation_joint_names = self._joint_order
            
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Failed to load joint_order from '{config.joint_config_path}': {e}")
        self.joint_direction_map = {f"{self.observation_joint_names[i]}.pos": config.joint_direction[i] for i in range(len(self.observation_joint_names))}        
        self.prometheus_port = getattr(config, 'prometheus_port', None)
        self.joint_position_gauge = None
        if self.prometheus_port is not None:
            # 从管理器获取共享的 Gauge 对象
            self.joint_position_gauge = prometheus_manager.get_gauge('joint_position')

    @property
    def is_connected(self) -> bool:
        """返回机器人是否已连接。"""
        return self._is_connected_flag

    def connect(self, calibrate: bool = True) -> None:
        """建立与机器人的通信。"""
        if self.is_connected:
            print("Robot is already connected.")
            return

        print(f"Connecting to {self.name} using config '{self.config.joint_config_path}'...")
        self._hardware_manager = SupreRobotHardwareManager(config_path=self.config.joint_config_path)
        
        try:
            if not self._hardware_manager.init():
                self._hardware_manager = None
                raise RuntimeError("Failed to initialize hardware manager.")

            if not self._hardware_manager.activate():
                self._hardware_manager = None
                raise RuntimeError("Failed to activate hardware.")

            self._is_connected_flag = True
            print("Robot connected successfully.")
            
            if calibrate:
                self.calibrate()

        except Exception as e:
            print(f"Failed to connect: {e}")
            self._hardware_manager = None
            self._is_connected_flag = False
            raise e

    @property
    def is_calibrated(self) -> bool:
        """
        对于我们的硬件，只要连接成功并读取到初始位置，就认为它是“已校准”的。
        """
        return self.is_connected

    def calibrate(self) -> None:
        """
        我们的硬件（绝对编码器）不需要显式的校准程序。
        这个方法可以是一个空操作。
        """
        if not self.is_connected:
            raise RuntimeError("Cannot calibrate while disconnected.")
        print("Hardware does not require an explicit calibration step. Skipping.")
        pass

    def configure(self) -> None:
        """
        所有配置都在硬件管理器的 init() 和 activate() 步骤中完成。
        这个方法可以是一个空操作。
        """
        if not self.is_connected:
            raise RuntimeError("Cannot configure while disconnected.")
        print("Hardware is already configured on connect. Skipping.")
        pass
    def get_action(self) -> dict[str, Any]:
        """
        Reads the leader's (left arm) current state and formats it as an action
        for the follower (right arm).
        """
        if not self.is_connected:
            raise RuntimeError("Leader teleoperator is not connected.")

        positions = self._hardware_manager.read()

        pos_map = dict(zip(self.observation_joint_names, positions))
        action_value = {}
        for i in range(len(self.observation_joint_names)):
            observation_joint_name = self.observation_joint_names[i]
            action_value[observation_joint_name] = pos_map[observation_joint_name]

            if self.joint_position_gauge:
                    # 使用 'leader' 作为 robot_name
                self.joint_position_gauge.labels(
                    robot_name='leader', 
                    joint_name=observation_joint_name,
                    joint_id=observation_joint_name
                ).set(pos_map[observation_joint_name])

            if observation_joint_name=="left_arm_joint_7" or observation_joint_name=="right_arm_joint_7":
                #convert to gripper position(0-90 to 0-40)
                action_value[observation_joint_name] = self.convert_gripper_position(action_value[observation_joint_name])
        # The action for the follower is the position of the leader's joints.
        action = {f"{m}.pos":v for m,v in action_value.items()}
                # modify the action by joint_direction_map
        action = {key: val * self.joint_direction_map[key] for key, val in action.items()}
        #self._ros_node.get_logger().info(f"get action: {action}")
        return action

    def convert_gripper_position(self,joint_position: float) -> float:
        """
        将关节角度 (度) 线性映射到夹爪开合宽度 (毫米)。
    
        该函数会先将输入关节角度限制在定义的范围内 (0-90度)，
        然后再执行线性转换。
    
        :param joint_position: 输入的关节角度 (单位: 度)。
        :return: 对应的夹爪开合宽度 (单位: 毫米)。
        """
        joint_position = abs(joint_position)
        # 1. 定义映射范围常量，清晰明了
        JOINT_MIN_DEG = 0.0
        JOINT_MAX_DEG = 60.0
        GRIPPER_MIN_MM = 0.0
        GRIPPER_MAX_MM = 1.0
    
        # 2. 纯 Python 实现的边界限制 (clamping)
        #    先用 max 保证不低于最小值，再用 min 保证不高于最大值。
        clamped_joint_pos = max(JOINT_MIN_DEG, min(joint_position, JOINT_MAX_DEG))
    
        clamped_joint_pos = JOINT_MAX_DEG - clamped_joint_pos#逆向
        # 3. 执行线性插值
        input_range = JOINT_MAX_DEG - JOINT_MIN_DEG
        output_range = GRIPPER_MAX_MM - GRIPPER_MIN_MM
    
        # 避免除以零
        if input_range == 0:
            return GRIPPER_MIN_MM
    
        scale = (clamped_joint_pos - JOINT_MIN_DEG) / input_range
        gripper_position = GRIPPER_MIN_MM + (scale * output_range)
    
        return gripper_position
    def disconnect(self) -> None:
        """断开与机器人的连接。"""
        if not self.is_connected:
            print("Robot is already disconnected.")
            return
        
        print("Disconnecting from robot...")
        try:
            if self._hardware_manager:
                self._hardware_manager.deactivate()
        except Exception as e:
            print(f"An error occurred during deactivation: {e}")
        finally:
            self._hardware_manager = None
            self._is_connected_flag = False
            print("Robot disconnected.")

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft    
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.observation_joint_names}    