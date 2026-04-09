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
        # 抖动提示模式：当 Follower 遇到阻力时，Leader 抖动提示

        # CST 模式状态
        self._cst_mode_enabled = False
        self._force_feedback_failed = False  # 标志：CST 配置是否失败过

        # 抖动状态
        self._vibration_active = False       # 是否正在抖动
        self._vibration_start_time = 0.0     # 抖动开始时间
        self._vibration_phase = 0            # 抖动相位
        self._last_force_check_time = 0.0    # 上次检测时间
        self._force_exceed_count = 0         # 力超限计数（用于防抖）

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
        抖动提示模式：当 Follower 遇到较大阻力时，Leader 产生短促抖动提示用户。

        这种方式比持续阻尼更安全、更直观：
        - 用户明确感知到"遇到阻力了"
        - 不会产生"泥鳅"般的不稳定感
        - 防止过度操作导致电机堵转

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

        import time
        current_time = time.perf_counter()

        # ===== 配置参数 =====
        force_threshold = getattr(self.config, 'force_threshold', 0.3)      # 触发抖动的力阈值 (Nm)
        vibration_amplitude = getattr(self.config, 'vibration_amplitude', 0.2)  # 抖动幅度 (Nm)
        vibration_duration = getattr(self.config, 'vibration_duration', 0.5)    # 抖动持续时间 (s)
        vibration_frequency = getattr(self.config, 'vibration_frequency', 15.0) # 抖动频率 (Hz)
        force_debounce_count = getattr(self.config, 'force_debounce_count', 3)  # 防抖计数
        rated_torque = getattr(self.config, 'rated_torque', 2.0)

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
        if max_force > force_threshold:
            self._force_exceed_count += 1
        else:
            self._force_exceed_count = 0

        # ===== 2. 触发抖动 =====
        if self._force_exceed_count >= force_debounce_count and not self._vibration_active:
            self._vibration_active = True
            self._vibration_start_time = current_time
            self._vibration_phase = 0
            print(f"[VIBRATION] Triggered! Max force: {max_force:.3f} Nm at {max_force_joint}")

        # ===== 3. 执行抖动 =====
        if self._vibration_active:
            elapsed = current_time - self._vibration_start_time

            if elapsed < vibration_duration:
                # 生成抖动信号：正弦波脉冲
                # 使用 sin 产生正负交替的力矩
                phase = 2 * 3.14159 * vibration_frequency * elapsed
                vibration_value = vibration_amplitude * math.sin(phase)

                # 发送到所有关节
                torques_to_send = [vibration_value] * self.num_joints

                # 应用关节方向映射
                for i, joint_name in enumerate(self.observation_joint_names):
                    direction = self.joint_direction_map.get(f"{joint_name}.pos", 1)
                    torques_to_send[i] *= direction

                try:
                    self._hardware_manager.write_torques(torques_to_send, rated_torque)
                except Exception as e:
                    print(f"Warning: Failed to send vibration: {e}")

                # 每50次循环打印一次调试信息
                self._vibration_phase += 1
                if self._vibration_phase % 50 == 0:
                    print(f"[VIBRATION] Elapsed: {elapsed:.2f}s, value: {vibration_value:.3f} Nm")
            else:
                # 抖动结束
                self._vibration_active = False
                self._force_exceed_count = 0
                print("[VIBRATION] Ended")

                # 发送零力矩
                try:
                    self._hardware_manager.write_torques([0.0] * self.num_joints, rated_torque)
                except Exception as e:
                    print(f"Warning: Failed to clear torques: {e}")

        # ===== 4. 统计与调试 =====
        self._feedback_count += 1
        if self._feedback_count % 100 == 0:
            print(f"Force check: max_force={max_force:.3f} Nm, threshold={force_threshold} Nm, "
                  f"vibration_active={self._vibration_active}")

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