# lerobot/src/lerobot/robots/sim_robot/sim_robot.py
import logging
import time
import math
import pybullet as p
import pybullet_data
import numpy as np
from functools import cached_property
from typing import Any, Dict, Tuple

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots import Robot
from .config_sim_robot import SimRobotConfig

logger = logging.getLogger(__name__)

class SimRobot(Robot):
    config_class = SimRobotConfig
    name = "sim_robot"

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True  # 仿真环境默认已校准
        self.simulator = None

        # 关节名称映射（与仿真环境一致）
        self.joint_names = [
            "shoulder_roll_right", "shoulder_lift_right", "elbow_roll_right", 
            "elbow_flex_right", "wrist_roll_right", "gripper_flex_right",
            "gripper_right_1", "gripper_right_2", "shoulder_roll_left",
            "shoulder_lift_left", "elbow_roll_left", "elbow_flex_left",
            "wrist_roll_left", "gripper_flex_left", "gripper_left_1", "gripper_left_2"
        ]

        # 相机配置（从config转换）
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征定义（遵循仓库格式）"""
        return {f"{motor}.pos": float for motor in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征定义"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征集合"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征集合"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.simulator is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 初始化仿真环境
        self.simulator = self._create_simulator()
        self._is_connected = True
        logger.info(f"{self} connected to PyBullet simulator")

        # 连接相机（仿真环境中无需实际硬件连接）
        for cam in self.cameras.values():
            cam.connect()

    def _create_simulator(self) -> Any:
        """创建并初始化仿真环境"""
        from .simulator import Simulator  # 导入用户提供的仿真器类
        return Simulator(
            headless=self.config.headless,
            is_manual=self.config.is_manual
        )

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        """仿真环境无需实际校准"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self._is_calibrated = True
        logger.info(f"{self} calibration skipped (simulation)")

    def configure(self) -> None:
        """配置仿真环境参数"""
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 获取关节状态
        joint_positions, _ = self.simulator.get_joint_states()
        obs_dict = {
            f"{name}.pos": joint_positions[i]
            for i, name in enumerate(self.joint_names)
        }

        # 获取相机图像
        images = self.simulator.get_camera_images()
        for cam_name, img in images.items():
            obs_dict[cam_name] = img

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 转换动作格式（从字典到数组）
        action_array = np.array([
            action[f"{name}.pos"] for name in self.joint_names
        ])

        # 执行仿真步骤
        _, _, done, _ = self.simulator.step(action_array)
        if done:
            logger.info("Simulation episode completed")

        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.simulator.close()
        self.simulator = None
        self._is_connected = False

        # 断开相机连接
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected from simulator")