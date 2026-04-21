# supre_robot.py

import logging
import time
import math
from typing import Any, Dict, List, Optional, Tuple, Type
from pathlib import Path
import threading
import queue
import os

import yaml
import numpy as np
import dataclasses
# 导入我们之前设计的硬件管理器
from lerobot.robots.supre_robot import SupreRobotHardwareManager
# from eyou_hardware import EyouMotorHardware  # Manager will import these
# from gripper_hardware import JodellGripperHardware # Manager will import these
from ..robot import Robot
from .supre_robot_follower_config import SupreRobotFollowerConfig
from ..utils import ensure_safe_goal_position
from functools import cached_property
from lerobot.utils.prometheus_manager import prometheus_manager
import logging
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.monitor_utils import monitor_performance

#logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


# ==================== MotorsBus 兼容层 ====================
# RobotEnv 等组件期望 robot.bus.motors 和 bus.sync_read/write 接口
# SupreRobotFollower 使用 SupreRobotHardwareManager，需要适配

class SupreRobotBusCompat:
    """
    MotorsBus 接口兼容层。

    将 SupreRobotHardwareManager 的接口适配为标准 MotorsBus 接口，
    供 RobotEnv 等使用。

    主要方法：
    - motors: 返回关节名 dict（兼容 MotorsBus.motors）
    - sync_read(data_type): 同步读取数据
    - sync_write(data_type, values): 同步写入数据
    """

    def __init__(self, robot: "SupreRobotFollower"):
        self._robot = robot

    @property
    def motors(self) -> dict:
        """
        返回关节名字典，模拟 MotorsBus.motors。

        MotorsBus.motors 格式: {"shoulder_pan": Motor(...), ...}
        这里简化为返回关节名作为 key（RobotEnv 只用 key 列表）。
        """
        # 返回 {joint_name: True} 格式，RobotEnv 只需要 key 来迭代
        return {name: True for name in self._robot.observation_joint_names}

    def sync_read(self, data_type: str) -> dict:
        """
        同步读取指定类型的数据。

        Args:
            data_type: 数据类型，如 "Present_Position", "Present_Current"

        Returns:
            dict: {motor_name: value} 格式
        """
        if not self._robot.is_connected:
            raise RuntimeError("Robot is not connected.")

        # 从 hardware_manager 读取
        positions, forces = self._robot._hardware_manager.read()

        if data_type == "Present_Position":
            # 返回位置数据
            return {
                name: positions[i]
                for i, name in enumerate(self._robot.observation_joint_names)
            }
        elif data_type == "Present_Current":
            # 返回力/电流数据（RobotEnv 用于观察）
            return {
                name: forces[i]
                for i, name in enumerate(self._robot.observation_joint_names)
            }
        else:
            logging.warning(f"Unknown data_type '{data_type}', returning empty dict")
            return {}

    def sync_write(self, data_type: str, values: dict) -> None:
        """
        同步写入指定类型的数据。

        Args:
            data_type: 数据类型，如 "Goal_Position"
            values: {motor_name: value} 格式的字典
        """
        if not self._robot.is_connected:
            raise RuntimeError("Robot is not connected.")

        if data_type == "Goal_Position":
            # 将 dict 转换为列表，按 joint_order 顺序
            target_positions = [
                values.get(name, 0.0)
                for name in self._robot.observation_joint_names
            ]
            self._robot._hardware_manager.write(target_positions)
        elif data_type == "Torque_Enable":
            # 使能/失能电机，SupreRobotHardwareManager 在 activate 时已处理
            logging.debug(f"Torque_Enable write ignored (handled by hardware manager): {values}")
        else:
            logging.warning(f"Unknown sync_write data_type '{data_type}'")

    def write(self, data_type: str, motor_name: str, value: Any) -> None:
        """
        写入单个电机的数据。

        Args:
            data_type: 数据类型
            motor_name: 电机名
            value: 值
        """
        # 对于 PID 参数等，SupreRobot 使用不同的配置方式
        # 这里提供兼容接口但可能不实际生效
        logging.debug(f"Single motor write: {data_type} -> {motor_name} = {value} (compat mode)")


# 2. 实现 Robot 接口

class SupreRobotFollower(Robot):
    """
    A LeRobot-compatible class for the dual-arm robot controlled by
    Eyou motors and Jodell grippers.
    """
    # 设置 LeRobot 要求的类属性
    config_class = SupreRobotFollowerConfig
    name = "supre_robot_follower"

    def __init__(self, config: SupreRobotFollowerConfig):
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

        self.cameras = make_cameras_from_configs(config.cameras)
      
        # 将 calibration 列表转换为一个字典以便快速查找
        # key: joint_name, value: MotorCalibration object
        self.calibration_limits = {cal.joint_name: cal for cal in self.config.calibration}
        # 增加一个检查，确保所有在 joint_names 中的关节都有对应的 calibration 设置
        for joint_name in self.observation_joint_names:
            if joint_name not in self.calibration_limits:
                raise ValueError(f"Missing calibration data for joint '{joint_name}' in config.")

        self.prometheus_port = getattr(config, 'prometheus_port', None)
        self.joint_position_gauge = None
        if self.prometheus_port is not None:
            # 从管理器获取共享的 Gauge 对象
            self.joint_position_gauge = prometheus_manager.get_gauge('joint_position')

        self._use_interpolation = os.getenv('SUPRE_ROBOT_INTERPOLATION_ENABLED', 'false').lower() == 'true'

        if self._use_interpolation:
            logger.info("Interpolation mode is ENABLED via environment variable.")
            # 使用 maxsize=1 的队列，它天然只保存最新的目标
        else:
            logger.info("Interpolation mode is DISABLED. Using direct command sending.")

        # ==================== 力反馈缓存 ====================
        # 缓存最新的力数据，避免 get_force_feedback() 重复调用硬件读取
        self._cached_forces: Optional[List[float]] = None

        # ==================== MotorsBus 兼容层 ====================
        # RobotEnv 等组件使用 robot.bus 接口，这里提供兼容
        self._bus_compat: Optional[SupreRobotBusCompat] = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._force_ft}
        # return {**self._motors_ft, **self._cameras_ft}
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def bus(self) -> SupreRobotBusCompat:
        """
        MotorsBus 兼容层接口。

        RobotEnv 等组件期望 robot.bus.motors 和 bus.sync_read/write。
        返回 SupreRobotBusCompat 对象，提供这些接口。
        """
        if self._bus_compat is None:
            self._bus_compat = SupreRobotBusCompat(self)
        return self._bus_compat

    @property
    def is_connected(self) -> bool:
        """返回机器人是否已连接。"""
        return self._is_connected_flag

    def connect(self, calibrate: bool = True) -> None:
        """建立与机器人的通信。"""
        if self.is_connected:
            logging.info("Robot is already connected.")
            return

        logging.info(f"Connecting to {self.name} using config '{self.config.joint_config_path}'...")
        self._hardware_manager = SupreRobotHardwareManager(
            config_path=self.config.joint_config_path,
            control_frequency=self.config.control_frequency,
            use_interpolation=self._use_interpolation,
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
            logging.info("Robot connected successfully.")

            if calibrate:
                self.calibrate()

            for cam in self.cameras.values():
                cam.connect()

        except Exception as e:
            logging.error(f"Failed to connect: {e}")
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
        logging.debug("Hardware does not require an explicit calibration step. Skipping.")
        pass

    def configure(self) -> None:
        """
        所有配置都在硬件管理器的 init() 和 activate() 步骤中完成。
        这个方法可以是一个空操作。
        """
        if not self.is_connected:
            raise RuntimeError("Cannot configure while disconnected.")
        logging.debug("Hardware is already configured on connect. Skipping.")
        pass

    def reset(self) -> None:
        """
        重置机器人状态，用于 Episode 开始时。

        清除缓存数据，刷新状态。物理复位（移动到初始位置）由外部调用
        execute_trajectory 或 send_action 完成。

        这个方法是 Robot 接口的一部分，RobotEnv.reset() 会调用它。
        """
        if not self.is_connected:
            raise RuntimeError("Cannot reset while disconnected.")

        # 清除缓存的力数据
        self._cached_forces = None

        # 刷新硬件状态（可选，确保状态同步）
        positions, forces = self._hardware_manager.read()
        self._cached_forces = forces

        logging.debug("SupreRobotFollower reset completed.")

    @monitor_performance
    def get_observation(self) -> dict[str, Any]:
        """从机器人获取当前观测值。"""
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        hd_readings = self._hardware_manager.read()
        positions = hd_readings[0]
        forces = hd_readings[1]

        # 缓存力数据供 get_force_feedback() 使用，避免重复硬件读取
        self._cached_forces = forces

        # 移除 logging.debug 以减少每帧开销
        # obs_dict = {f"{self.observation_joint_names[i]}.pos": positions[i] for i in range(len(self.observation_joint_names))}
        obs_dict = {}
        for i in range(len(self.observation_joint_names)):
            joint_name = self.observation_joint_names[i]
            # 添加关节位置
            obs_dict[f"{joint_name}.pos"] = positions[i]
            # 添加关节力/力矩
            obs_dict[f"{joint_name}.force"] = forces[i]

        # 移除 logging.debug 以减少每帧开销
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            # 移除 logger.debug 以减少每帧开销
        return obs_dict

    def get_current_position(self) -> dict[str, float]:
        """获取机器人的当前位置。"""
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        positions = self._hardware_manager.read()[0]

        # 移除 logging.debug 以减少每帧开销
        return {self.observation_joint_names[i]: positions[i] for i in range(len(self.observation_joint_names))}

    def _prepare_and_clamp_action(self, action: dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        if action is None:
            raise ValueError("Action dictionary must contain 'joint_positions'.")

        action_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        ensure_safe = True
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
            logger.warning(
                "One or more joint positions were clamped to their absolute limits:"
            )
        
        # 使用经过两层安全检查后的最终位置
        final_target_positions = final_clamped_positions

        if self.joint_position_gauge:
            for joint_name, position in zip(self.observation_joint_names, final_target_positions):
                # 使用 'leader' 作为 robot_name
                self.joint_position_gauge.labels(
                    robot_name='follower', 
                    joint_name=joint_name,
                    joint_id=joint_name,
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

        # 首先，创建一个包含最终执行值的字典 (key: 'left_arm_joint_1', value: final_pos)
        final_action = {
            f"{name}.pos": pos 
            for name, pos in zip(self.observation_joint_names, final_target_positions)
        }
        
        return final_clamped_positions, final_action    
    @monitor_performance
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """向机器人发送动作指令。"""
        if not self.is_connected:
            raise RuntimeError("Follower robot is not connected.")
        logger.debug(f"Sending action: {action}")
        # 1. 调用辅助方法来完成所有的计算和安全检查
        final_target_positions, final_action_dict = self._prepare_and_clamp_action(action)
        # 移除 logging.debug 以减少每帧开销
        # 2. 将计算结果发送到硬件
        # 2. 根据是否启用插值，选择不同的发送方式

        # --- 直接发送逻辑 ---
        self.send_target_position(final_target_positions)


        return final_action_dict
        
    def send_target_position(self, target_positions: list[float]) -> None:
        """将目标位置发送给机器人。"""
        self._hardware_manager.write(target_positions)
    def disconnect(self) -> None:
        """断开与机器人的连接。"""
        if not self.is_connected:
            logging.info("Robot is already disconnected.")
            return

        logging.info("Disconnecting from robot...")

        try:
            if self._hardware_manager:
                self._hardware_manager.deactivate()
        except Exception as e:
            logging.error(f"An error occurred during deactivation: {e}")
        finally:
            self._hardware_manager = None
            self._is_connected_flag = False
            logging.info("Robot disconnected.")

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }    
    

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.observation_joint_names}   

    @property
    def _force_ft(self) -> dict[str, type]:
        return {f"{motor}.force": float for motor in self.observation_joint_names}

    def get_force_feedback(self) -> dict[str, float]:
        """
        获取当前关节力/力矩数据，用于力反馈功能。
        使用缓存的力数据，避免重复硬件读取。

        Returns:
            dict[str, float]: 力数据字典，格式为 {"joint_name.force": force_value_in_Nm}
        """
        if not self.is_connected:
            raise RuntimeError("Follower robot is not connected.")

        # 使用缓存的力数据（由 get_observation() 更新），避免重复CAN读取
        if self._cached_forces is None:
            # 如果没有缓存（如首次调用或未执行过 get_observation），则读取一次
            hd_readings = self._hardware_manager.read()
            forces = hd_readings[1]
        else:
            forces = self._cached_forces

        force_feedback = {}
        for i, joint_name in enumerate(self.observation_joint_names):
            force_feedback[f"{joint_name}.force"] = forces[i]

        return force_feedback   
        
    def execute_trajectory(self, goal_action: dict[str, Any], duration: float = 1.0) -> None:
        """
        通过线性插值，在给定的时间内平滑地将机器人移动到目标位置。
        这是一个阻塞式方法，直到轨迹完成。

        :param goal_action: 包含最终目标关节位置的字典。
        :param duration: 完成移动所需的总时间（秒）。
        """
        """
        通过线性插值，在给定的时间内平滑地将机器人移动到目标位置。
        这是一个阻塞式方法，直到轨迹完成。

        :param goal_action: 包含最终目标关节位置的字典。
        :param duration: 完成移动所需的总时间（秒）。
        """
        if not self.is_connected:
            raise RuntimeError("Cannot execute trajectory while disconnected.")
            
        if duration <= 0:
            self.send_action(goal_action)
            return

        # --- 1. 获取轨迹的起点和终点 (无硬件副作用) ---
        
        # 起点: 机器人的当前位置
        start_positions_map = self.get_current_position()
        start_positions = np.array([start_positions_map[name] for name in self.observation_joint_names])
        
        # 终点: 调用辅助方法计算最终钳位后的目标位置，但 *不发送*
        final_target_positions, _ = self._prepare_and_clamp_action(goal_action)
        end_positions = np.array(final_target_positions)

        # --- 2. 计算插值参数 ---
        control_period = 1.0 / self.config.control_frequency
        num_steps = int(duration / control_period)
        if num_steps < 2:
            self.send_target_position(end_positions.tolist())
            time.sleep(duration)
            return

        # --- 3. 执行高频插值控制循环 ---
        # print(f"Executing trajectory over {duration:.2f}s in {num_steps} steps.")
        
        for i in range(num_steps):
            step_start_time = time.perf_counter()
            
            alpha = (i + 1) / num_steps
            interpolated_positions = start_positions + alpha * (end_positions - start_positions)
            
            # 直接调用最底层的发送方法，跳过 send_action 的重复检查
            self.send_target_position(interpolated_positions.tolist())
            
            elapsed_time = time.perf_counter() - step_start_time
            sleep_time = control_period - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)