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

        # ==================== 力反馈状态 ====================
        # 用于将 Follower 的力数据转换为 Leader 的阻尼力矩

        # CST 模式状态
        self._cst_mode_enabled = False
        self._force_feedback_failed = False  # 标志：CST 配置是否失败过

        # 力滤波缓冲
        self._filtered_forces: Dict[str, float] = {}

        # 统计信息
        self._feedback_count = 0

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

        # 如果在 CST 模式，先切回 CSP 模式
        if self._cst_mode_enabled:
            self._disable_cst_mode()

        try:
            if self._hardware_manager:
                self._hardware_manager.deactivate()
        except Exception as e:
            print(f"An error occurred during deactivation: {e}")
        finally:
            self._hardware_manager = None
            self._is_connected_flag = False
            self._cst_mode_enabled = False
            self._filtered_forces.clear()
            print("Robot disconnected.")

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        安全的力反馈：只有在用户移动时才提供阻力。

        核心安全原则：
        1. 静止时力矩为零 → 电机不会自己动
        2. 速度过快切断力矩 → 防止失控
        3. 力矩有限制 → 即使出错也不会产生危险运动

        Args:
            feedback: 力反馈字典，格式为 {"joint_name.force": force_value_in_Nm}
        """
        if not self.is_connected:
            raise RuntimeError("Leader teleoperator is not connected.")

        # 检查是否启用力反馈
        if not getattr(self.config, 'enable_force_feedback', False):
            return

        # 如果之前失败过，直接返回
        if self._force_feedback_failed:
            return

        # 首次调用时启用 CST 模式
        if not self._cst_mode_enabled:
            if not self._enable_cst_mode():
                print("Warning: Failed to enable CST mode, force feedback disabled for this session")
                self._force_feedback_failed = True
                return

        # ===== 安全力反馈核心逻辑 =====

        # 1. 先读取硬件状态（必须先调用 read() 更新速度数据）
        try:
            self._hardware_manager.read()
        except Exception as e:
            print(f"Warning: Failed to read hardware state: {e}")
            return

        # 2. 读取 Leader 当前速度（用于安全检查）
        try:
            velocities = self._hardware_manager.read_velocities()
        except Exception as e:
            print(f"Warning: Failed to read velocities: {e}")
            return

        # 2. 获取配置参数（必须在使用前定义）
        damping_gain = getattr(self.config, 'damping_gain', 0.5)
        max_damping = getattr(self.config, 'max_damping_torque', 0.3)
        filter_alpha = getattr(self.config, 'force_filter_alpha', 0.5)
        rated_torque = getattr(self.config, 'rated_torque', 2.0)
        velocity_deadband = getattr(self.config, 'velocity_deadband', 2.0)
        max_velocity_threshold = getattr(self.config, 'max_velocity_threshold', 100.0)
        velocity_scale = getattr(self.config, 'velocity_scale', 30.0)
        torque_safety_margin = getattr(self.config, 'torque_safety_margin', 0.1)

        # 调试：打印速度信息（每100次循环打印一次）
        if self._feedback_count % 100 == 0:
            max_vel = max(abs(v) for v in velocities) if velocities else 0
            print(f"DEBUG: Max velocity: {max_vel:.2f} °/s, deadband: {velocity_deadband} °/s")

        # 3. 计算每个关节的阻尼力矩
        torques_to_send = [0.0] * self.num_joints
        safety_cut_count = 0
        active_joint_count = 0

        for i, joint_name in enumerate(self.observation_joint_names):
            velocity = velocities[i] if i < len(velocities) else 0.0
            velocity_abs = abs(velocity)
            force_key = f"{joint_name}.force"

            # ===== 安全检查：速度只作为触发条件 =====

            # 检查1: 速度过快，切断力矩
            if velocity_abs > max_velocity_threshold:
                torques_to_send[i] = 0.0
                safety_cut_count += 1
                if joint_name in self._filtered_forces:
                    del self._filtered_forces[joint_name]
                continue

            # 检查2: 静止时不发力矩（核心安全机制！）
            if velocity_abs < velocity_deadband:
                torques_to_send[i] = 0.0
                if joint_name in self._filtered_forces:
                    del self._filtered_forces[joint_name]
                continue

            # 用户正在移动，提供阻尼
            active_joint_count += 1

            # 获取 Follower 的力数据
            raw_force = feedback.get(force_key, 0.0)

            # 简单低通滤波
            if joint_name in self._filtered_forces:
                filtered_force = (
                    filter_alpha * raw_force
                    + (1 - filter_alpha) * self._filtered_forces[joint_name]
                )
            else:
                filtered_force = raw_force
            self._filtered_forces[joint_name] = filtered_force

            # ========================================
            # 核心安全逻辑：真正的阻尼
            # ========================================
            # 阻尼定义：始终与运动方向相反，永远不会主动驱动电机
            #
            # 用户正向移动 (velocity > 0) → 阻尼力矩 < 0 (阻止正向移动)
            # 用户反向移动 (velocity < 0) → 阻尼力矩 > 0 (阻止反向移动)
            #
            # Follower 力数据决定阻尼的"大小"，速度方向决定阻尼的"方向"
            # ========================================

            # 速度方向（确定阻尼方向）
            velocity_sign = 1.0 if velocity > 0 else -1.0

            # 阻尼力矩 = -速度方向 × 力大小 × 增益
            # 负号确保阻尼始终与运动方向相反
            damping_magnitude = abs(filtered_force) * damping_gain
            damping_torque = -velocity_sign * damping_magnitude

            # 限制最大阻尼力矩
            damping_torque = max(-max_damping, min(damping_torque, max_damping))

            # 应用关节方向映射
            direction = self.joint_direction_map.get(f"{joint_name}.pos", 1)
            damping_torque *= direction

            torques_to_send[i] = damping_torque

        # 4. 发送力矩指令
        try:
            self._hardware_manager.write_torques(torques_to_send, rated_torque)
        except Exception as e:
            print(f"Warning: Failed to write torques: {e}")
            return

        # 5. 统计与日志
        self._feedback_count += 1
        if self._feedback_count % 100 == 0:
            max_torque = max(abs(t) for t in torques_to_send) if torques_to_send else 0
            print(f"Force feedback: {self._feedback_count} cycles, "
                  f"active joints: {active_joint_count}, "
                  f"max torque: {max_torque:.3f} Nm, "
                  f"safety cuts: {safety_cut_count}")

    def _enable_cst_mode(self) -> bool:
        """启用 CST 力矩控制模式。"""
        print("Enabling CST (Torque) mode for force feedback...")
        try:
            if self._hardware_manager.configure_cst_mode(interpolation_period_ms=10):
                self._cst_mode_enabled = True
                print("CST mode enabled successfully.")
                return True
            else:
                print("Failed to configure CST mode.")
                return False
        except Exception as e:
            print(f"Error enabling CST mode: {e}")
            return False

    def _disable_cst_mode(self) -> bool:
        """切回 CSP 位置控制模式。"""
        print("Switching back to CSP (Position) mode...")
        try:
            if self._hardware_manager.configure_csp_mode():
                self._cst_mode_enabled = False
                print("CSP mode restored.")
                return True
            else:
                print("Failed to restore CSP mode.")
                return False
        except Exception as e:
            print(f"Error disabling CST mode: {e}")
            return False
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft    
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.observation_joint_names}    