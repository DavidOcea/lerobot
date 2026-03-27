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

    @abstractmethod
    def set_cartesian_velocity(self, vx: float, vy: float, vz: float,
                                wx: float = 0, wy: float = 0, wz: float = 0) -> bool:
        """设置笛卡尔速度（平滑IBVS所需）

        Args:
            vx, vy, vz: 线速度 (m/s)
            wx, wy, wz: 角速度 (rad/s)
        """
        pass

    @abstractmethod
    def servo_to_position(self, x: float, y: float, z: float,
                          qx: float, qy: float, qz: float, qw: float,
                          time_ms: int = 100) -> bool:
        """伺服模式移动（平滑轨迹）

        与move_to_position的区别：
        - move_to_position: 点到点，可能有加速减速
        - servo_to_position: 流式位置，平滑轨迹，适合连续控制

        Args:
            x, y, z: 目标位置
            qx, qy, qz, qw: 目标姿态四元数
            time_ms: 到达目标的时间（毫秒）
        """
        pass

    def has_velocity_control(self) -> bool:
        """是否支持速度控制"""
        return False

    def has_servo_mode(self) -> bool:
        """是否支持伺服模式"""
        return False


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

    def set_cartesian_velocity(self, vx: float, vy: float, vz: float,
                                wx: float = 0, wy: float = 0, wz: float = 0) -> bool:
        """模拟速度控制（通过积分更新位置）"""
        dt = 0.05  # 50ms周期
        self.tcp_pos += np.array([vx, vy, vz]) * dt
        print(f"[Mock] 速度 ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s")
        return True

    def servo_to_position(self, x: float, y: float, z: float,
                          qx: float, qy: float, qz: float, qw: float,
                          time_ms: int = 100) -> bool:
        """模拟伺服模式"""
        self.tcp_pos = np.array([x, y, z])
        self.tcp_rot = np.array([qx, qy, qz, qw])
        print(f"[Mock] Servo到 ({x:.3f}, {y:.3f}, {z:.3f})")
        return True

    def has_velocity_control(self) -> bool:
        return True

    def has_servo_mode(self) -> bool:
        return True