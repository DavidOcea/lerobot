# supre_robot.py

import time
import math
import subprocess
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


class SupreRobotLeaderBusCompat:
    """
    Compatibility layer that provides MotorsBus-like interface for SupreRobotLeader.

    This allows SupreRobotLeader to work with code that expects robot.bus interface,
    such as ResetWrapper in gym_manipulator.py.

    Similar to SupreRobotBusCompat in supre_robot_follower.py.
    """

    def __init__(self, leader: "SupreRobotLeader"):
        self._leader = leader

    @property
    def motors(self) -> dict:
        """Return dict of motor names (for interface compatibility)."""
        if not self._leader._hardware_manager:
            return {}
        return {name: True for name in self._leader.observation_joint_names}

    def sync_read(self, data_type: str) -> dict:
        """Read data from all motors synchronously."""
        if not self._leader._hardware_manager:
            raise RuntimeError("Hardware not connected")

        positions, forces = self._leader._hardware_manager.read()

        if data_type == "Present_Position":
            return {f"{name}.pos": positions[i] for i, name in enumerate(self._leader.observation_joint_names)}
        elif data_type == "Present_Current":
            return {f"{name}.force": forces[i] for i, name in enumerate(self._leader.observation_joint_names)}
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def sync_write(self, data_type: str, values: dict) -> None:
        """Write data to motors synchronously."""
        if not self._leader._hardware_manager:
            raise RuntimeError("Hardware not connected")

        if data_type == "Goal_Position":
            target_positions = [
                values.get(f"{name}.pos", values.get(name, 0.0))
                for name in self._leader.observation_joint_names
            ]
            self._leader._hardware_manager.write(target_positions)

        elif data_type == "Torque_Enable":
            # SupreRobotLeader motors stay enabled for teleoperation
            # Compatibility stub - no action needed
            pass

        else:
            raise ValueError(f"Unsupported sync_write data_type: {data_type}")


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
        self._bus_compat: Optional[SupreRobotLeaderBusCompat] = None  # Bus compatibility layer

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

        # ==================== 力反馈状态 ====================
        # 声音提示模式：当 Follower 遇到阻力时，播放声音提示用户
        # 阻力越大，声音越急促

        # 声音状态
        self._last_sound_time = 0.0           # 上次播放声音的时间
        self._force_exceed_count = 0          # 力超限计数（用于防抖）

        # 统计信息
        self._feedback_count = 0

        # ==================== 力反馈配置缓存 ====================
        # 性能优化：将 getattr 调用移到初始化时，避免每帧调用
        self._force_feedback_enabled = getattr(config, 'enable_force_feedback', False)
        self._force_threshold = getattr(config, 'force_threshold', 0.3)
        self._force_debounce_count = getattr(config, 'force_debounce_count', 3)
        self._min_beep_interval = getattr(config, 'min_beep_interval', 0.1)
        self._max_beep_interval = getattr(config, 'max_beep_interval', 1.0)
        self._max_force_for_sound = getattr(config, 'max_force_for_sound', 1.0)

    @property
    def is_connected(self) -> bool:
        """返回机器人是否已连接。"""
        return self._is_connected_flag

    @property
    def bus(self) -> SupreRobotLeaderBusCompat:
        """MotorsBus compatibility layer for ResetWrapper and other components."""
        if self._bus_compat is None:
            self._bus_compat = SupreRobotLeaderBusCompat(self)
        return self._bus_compat

    def connect(self, calibrate: bool = True) -> None:
        """建立与机器人的通信。"""
        if self.is_connected:
            print("Robot is already connected.")
            return

        print(f"Connecting to {self.name} using config '{self.config.joint_config_path}'...")
        self._hardware_manager = SupreRobotHardwareManager(
            config_path=self.config.joint_config_path,
            enable_velocity_read=self.config.enable_velocity_read
        )
        
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

        hd_readings = self._hardware_manager.read()
        positions = hd_readings[0]
        forces = hd_readings[1]
        
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
        """
        声音提示模式：当 Follower 遇到阻力时，播放声音提示用户。

        阻力越大，声音越急促（间隔越短）。
        这种方式比力矩反馈更安全：
        - 不会让电机自己转动
        - 用户能明确感知阻力大小
        - 防止过度操作导致电机堵转

        Args:
            feedback: 力反馈字典，格式为 {"joint_name.force": force_value_in_Nm}
        """
        if not self.is_connected:
            return

        # 检查是否启用力反馈（使用缓存的配置值）
        if not self._force_feedback_enabled:
            return

        current_time = time.perf_counter()

        # ===== 1. 检测 Follower 是否遇到阻力 =====
        max_force = 0.0
        max_force_joint = ""
        for key, value in feedback.items():
            if key.endswith('.force'):
                force_abs = abs(value)
                if force_abs > max_force:
                    max_force = force_abs
                    max_force_joint = key

        # 防抖处理：连续多次检测到超限才触发
        if max_force > self._force_threshold:
            self._force_exceed_count += 1
        else:
            self._force_exceed_count = 0

        # ===== 2. 播放声音提示 =====
        if self._force_exceed_count >= self._force_debounce_count:
            # 根据力的大小计算蜂鸣间隔
            # 力越大，间隔越短（声音越急促）
            force_ratio = min(max_force / self._max_force_for_sound, 1.0)  # 归一化到 0-1
            beep_interval = self._max_beep_interval - force_ratio * (self._max_beep_interval - self._min_beep_interval)

            # 检查是否到了播放时间
            if current_time - self._last_sound_time >= beep_interval:
                self._last_sound_time = current_time

                # 使用异步方式播放声音，避免阻塞控制循环
                try:
                    # Popen 是非阻塞的，会在后台播放声音
                    subprocess.Popen(
                        ['paplay', '/usr/share/sounds/freedesktop/stereo/message.oga'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    # 如果 paplay 失败，使用终端蜂鸣（也是非阻塞的）
                    try:
                        print('\a', end='', flush=True)
                    except:
                        pass

                # 打印调试信息
                print(f"[ALERT] Force: {max_force:.2f} Nm at {max_force_joint}, interval: {beep_interval:.2f}s")

        # ===== 3. 统计与调试 =====
        self._feedback_count += 1
        if self._feedback_count % 100 == 0 and max_force > self._force_threshold:
            print(f"Force check: max_force={max_force:.3f} Nm, threshold={self._force_threshold} Nm")

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.observation_joint_names}    