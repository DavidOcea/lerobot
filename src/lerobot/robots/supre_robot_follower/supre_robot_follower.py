# supre_robot.py

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

        # Build joint_direction_map for cross-robot model deployment.
        # When model_joint_direction_override is None, all directions are 1 (identity).
        override = self.config.model_joint_direction_override
        self._joint_direction_map = {}
        for i, joint_name in enumerate(self._joint_order):
            direction = override[i] if override is not None and i < len(override) else 1
            self._joint_direction_map[joint_name] = direction

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

        # Force display control
        self._observation_count = 0
        self._force_display_interval = 50  # Print detailed force info every N observations
        self._last_forces = None  # Store previous forces for rate calculation

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._force_ft}
        # return {**self._motors_ft, **self._cameras_ft}
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

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
        self._hardware_manager = SupreRobotHardwareManager(config_path=self.config.joint_config_path,control_frequency=self.config.control_frequency,use_interpolation=self._use_interpolation)
        
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

            for cam in self.cameras.values():
                cam.connect()

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
    @monitor_performance
    def get_observation(self) -> dict[str, Any]:
        """从机器人获取当前观测值。"""
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        hd_readings = self._hardware_manager.read()
        positions = hd_readings[0]
        forces = hd_readings[1]

        # Debug: Check if forces contain gripper data
        if len(forces) == len(self.observation_joint_names):
            # Forces array matches joint names order
            logger.debug(f"Position and force arrays have length {len(positions)}")
        else:
            logger.warning(f"Mismatch: positions={len(positions)}, forces={len(forces)}")

        # Enhanced force display with analysis
        self._observation_count += 1
        self._display_force_info(forces)

        # obs_dict = {f"{self.observation_joint_names[i]}.pos": positions[i] for i in range(len(self.observation_joint_names))}
        obs_dict = {}
        for i in range(len(self.observation_joint_names)):
            joint_name = self.observation_joint_names[i]
            direction = self._joint_direction_map.get(joint_name, 1)
            # 添加关节位置
            obs_dict[f"{joint_name}.pos"] = positions[i] * direction
            # 添加关节力/力矩
            if i < len(forces):
                obs_dict[f"{joint_name}.force"] = forces[i] * direction
            else:
                # If forces array is shorter, default to 0.0
                obs_dict[f"{joint_name}.force"] = 0.0
                logger.warning(f"Force data missing for joint {joint_name}")

        # Debug: Log gripper forces for pick tasks
        if self._observation_count % 10 == 0:
            left_force = obs_dict.get("left_arm_joint_7.force", 0.0)
            right_force = obs_dict.get("right_arm_joint_7.force", 0.0)
            logger.debug(f"Gripper forces - Left: {left_force:.3f}, Right: {right_force:.3f}")

        # 添加相机图像到 'images' 子字典中
        obs_dict["images"] = {}
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict["images"][cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        return obs_dict

    def get_current_position(self) -> dict[str, float]:
        """获取机器人的当前位置。"""
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        positions = self._hardware_manager.read()[0]

        pos_dict = {f"{self.observation_joint_names[i]}": positions[i] * self._joint_direction_map.get(self.observation_joint_names[i], 1) for i in range(len(self.observation_joint_names))}
        print("current_pos: ", pos_dict)
        return {self.observation_joint_names[i]: positions[i] * self._joint_direction_map.get(self.observation_joint_names[i], 1) for i in range(len(self.observation_joint_names))}

    def _prepare_and_clamp_action(self, action: dict[str, Any], skip_safety: bool = False) -> Tuple[List[float], Dict[str, Any]]:
        if action is None:
            raise ValueError("Action dictionary must contain 'joint_positions'.")

        action_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        ensure_safe = not skip_safety
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
        print("final_target_positions: ",final_target_positions)
        # 2. 将计算结果发送到硬件
        # 2. 根据是否启用插值，选择不同的发送方式

        # --- 直接发送逻辑 ---
        self.send_target_position(final_target_positions)

        
        return final_action_dict
        
    def send_target_position(self, target_positions: list[float]) -> None:
        """将目标位置发送给机器人。"""
        hw_positions = [pos * self._joint_direction_map.get(name, 1)
                        for name, pos in zip(self.observation_joint_names, target_positions)]
        self._hardware_manager.write(hw_positions)
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

    def emergency_stop(self) -> None:
        """Emergency stop - immediately deactivate all hardware.

        This method is called when a collision is detected or when
        immediate shutdown is required. It stops all joint motion
        by deactivating the hardware manager.
        """
        if not self.is_connected:
            logger.warning("Emergency stop called but robot is not connected")
            return

        logger.warning("EMERGENCY STOP activated - deactivating hardware")

        try:
            if self._hardware_manager:
                self._hardware_manager.deactivate()
                logger.info("Hardware deactivated successfully")
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    def keepalive(self) -> None:
        """Lightweight Modbus keepalive — call during idle select() periods."""
        if self._hardware_manager:
            self._hardware_manager.keepalive()

    def get_raw_torques(self) -> dict[str, float]:
        """Get raw torque/force data from all joints.

        Returns:
            Dictionary mapping joint names to their current torque values in Nm.
        """
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        try:
            positions, forces = self._hardware_manager.read()
            return {
                joint_name: float(forces[i]) * self._joint_direction_map.get(joint_name, 1)
                for i, joint_name in enumerate(self.observation_joint_names)
            }
        except Exception as e:
            logger.error(f"Failed to get raw torques: {e}")
            return {}

    def _display_force_info(self, forces: List[float]):
        """Display force information with analysis.

        Shows raw forces, force rates, and highlights high forces.

        Args:
            forces: List of force values for each joint.
        """
        # Always print raw forces (simple format)
        if self._observation_count % self._force_display_interval == 0:
            # Detailed print every N cycles
            print("")
            print("=" * 70)
            print(f"Force Analysis - Observation #{self._observation_count}")
            print("=" * 70)

            # Group joints by type for better readability
            arm_joints = []
            gripper_joints = []
            trunk_joints = []

            for i, joint_name in enumerate(self.observation_joint_names):
                force = forces[i]
                rate = 0.0
                if self._last_forces is not None and i < len(self._last_forces):
                    rate = abs(force - self._last_forces[i])

                info = {
                    "name": joint_name,
                    "force": force,
                    "rate": rate,
                }

                if "gripper" in joint_name.lower() or joint_name.endswith("_joint_7"):
                    gripper_joints.append(info)
                elif "trunk" in joint_name.lower():
                    trunk_joints.append(info)
                else:
                    arm_joints.append(info)

            # Display each group
            for group_name, group_data in [
                ("Left Arm", [j for j in arm_joints if "left" in j["name"].lower()]),
                ("Right Arm", [j for j in arm_joints if "right" in j["name"].lower()]),
                ("Trunk", trunk_joints),
                ("Grippers", gripper_joints),
            ]:
                if not group_data:
                    continue

                print(f"\n{group_name}:")
                for info in group_data:
                    force = info["force"]
                    rate = info["rate"]

                    # Visual indicators
                    force_bar = self._get_force_bar(force)
                    rate_indicator = " 🔺" if rate > 0.1 else ""

                    print(f"  {info['name']:30} | Force: {force:+6.3f} Nm {force_bar} {rate_indicator}")
                    if rate > 0.05:
                        print(f"  {'':30} | Rate:   {rate:.3f} Nm/step")

            print("=" * 70)
            print("")
        elif self._last_forces is not None:
            # Check for sudden force spikes - alert immediately
            max_rate = 0
            max_rate_joint = None
            for i, force in enumerate(forces):
                rate = abs(force - self._last_forces[i]) if i < len(self._last_forces) else 0
                if rate > max_rate:
                    max_rate = rate
                    max_rate_joint = self.observation_joint_names[i] if i < len(self.observation_joint_names) else "unknown"

            if max_rate > 0.3:
                logger.warning(f"⚠️ Sudden force spike: {max_rate_joint} | Δ = {max_rate:.3f} Nm/step")

        # Store for next comparison
        self._last_forces = forces.copy() if forces else None

    def _get_force_bar(self, force: float, max_abs: float = 2.0) -> str:
        """Generate a visual bar for force magnitude.

        Args:
            force: The force value in Nm.
            max_abs: Maximum absolute force for full bar.

        Returns:
            String with visual bar representation.
        """
        abs_force = abs(force)
        if abs_force < 0.1:
            return "│"

        # Scale to 0-10 range
        scaled = min(int(abs_force / max_abs * 10), 10)

        if force > 0:
            bar = "▸" * scaled
        else:
            bar = "◂" * scaled

        return bar

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
        
    def execute_trajectory(self, goal_action: dict[str, Any], duration: float = 1.0, skip_safety_check: bool = False) -> None:
        """
        通过线性插值，在给定的时间内平滑地将机器人移动到目标位置。
        这是一个阻塞式方法，直到轨迹完成。

        :param goal_action: 包含最终目标关节位置的字典。
        :param duration: 完成移动所需的总时间（秒）。
        :param skip_safety_check: 是否跳过安全检查（max_relative_joint_move限制）。
                                True用于复位操作，False用于正常控制。
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
        final_target_positions, _ = self._prepare_and_clamp_action(goal_action, skip_safety=skip_safety_check)
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

    def reset_to_zero(self, duration: float = 3.0, target_positions: dict[str, float] | None = None) -> None:
        """Smoothly reset all joints to zero or target position.

        Args:
            duration: Time in seconds to complete the reset (default: 3.0s).
            target_positions: Optional dictionary mapping joint names to target positions.
                           If None, all joints reset to 0.0.
                           Example: {"left_arm_joint_1": 0.0, "right_arm_joint_7": 0.5}
        """
        if not self.is_connected:
            raise RuntimeError("Cannot reset while disconnected.")

        logger.info(f"Resetting robot joints over {duration:.2f}s...")

        # Create target action with joint positions
        if target_positions is None:
            # Default: all joints to 0.0
            target_action = {f"{name}.pos": 0.0 for name in self.observation_joint_names}
        else:
            # Use specified positions for provided joints, 0.0 for others
            target_action = {}
            for name in self.observation_joint_names:
                if name in target_positions:
                    target_action[f"{name}.pos"] = target_positions[name]
                else:
                    target_action[f"{name}.pos"] = 0.0

        # Execute smooth trajectory to target positions
        # Skip safety check to allow large movements during reset
        self.execute_trajectory(target_action, duration=duration, skip_safety_check=True)

        logger.info("Robot reset completed successfully")