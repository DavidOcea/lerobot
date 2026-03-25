"""
机器人接口 (Robot Interface)

定义机器人控制的抽象接口，方便测试和替换实现。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class RobotInterface(ABC):
    """机器人控制接口"""

    @abstractmethod
    def connect(self) -> bool:
        """连接机器人"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    def get_joint_states(self) -> Optional[np.ndarray]:
        """获取关节角度（度）"""
        pass

    @abstractmethod
    def get_tcp_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取TCP位姿

        Returns:
            (position, quaternion) 位置和四元数 [qx, qy, qz, qw]
        """
        pass

    @abstractmethod
    def move_to_position(self, x: float, y: float, z: float,
                         qx: float, qy: float, qz: float, qw: float) -> bool:
        """移动到指定位姿"""
        pass

    @abstractmethod
    def move_joint(self, joint_idx: int, angle_deg: float) -> bool:
        """移动单个关节"""
        pass

    @abstractmethod
    def move_joints(self, joint_angles: np.ndarray) -> bool:
        """移动所有关节"""
        pass

    @abstractmethod
    def get_gripper_state(self) -> float:
        """获取夹爪状态"""
        pass

    @abstractmethod
    def set_gripper(self, value: float) -> bool:
        """设置夹爪"""
        pass


class MockRobot(RobotInterface):
    """模拟机器人（用于测试）"""

    def __init__(self):
        self.joints = np.zeros(7)
        self.tcp_pos = np.array([0.3, 0.0, 0.5])
        self.tcp_rot = np.array([0, 0, 0, 1])
        self.gripper = 0.0

    def connect(self) -> bool:
        print("[Mock] 连接成功")
        return True

    def disconnect(self) -> bool:
        print("[Mock] 断开连接")
        return True

    def get_joint_states(self) -> Optional[np.ndarray]:
        return self.joints.copy()

    def get_tcp_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.tcp_pos.copy(), self.tcp_rot.copy()

    def move_to_position(self, x: float, y: float, z: float,
                         qx: float, qy: float, qz: float, qw: float) -> bool:
        self.tcp_pos = np.array([x, y, z])
        self.tcp_rot = np.array([qx, qy, qz, qw])
        print(f"[Mock] 移动到 ({x:.3f}, {y:.3f}, {z:.3f})")
        return True

    def move_joint(self, joint_idx: int, angle_deg: float) -> bool:
        if 0 <= joint_idx < len(self.joints):
            self.joints[joint_idx] = angle_deg
            print(f"[Mock] 关节{joint_idx} -> {angle_deg:.1f}°")
            return True
        return False

    def move_joints(self, joint_angles: np.ndarray) -> bool:
        self.joints = joint_angles.copy()
        print(f"[Mock] 移动关节: {joint_angles}")
        return True

    def get_gripper_state(self) -> float:
        return self.gripper

    def set_gripper(self, value: float) -> bool:
        self.gripper = value
        print(f"[Mock] 夹爪 -> {value:.1f}")
        return True