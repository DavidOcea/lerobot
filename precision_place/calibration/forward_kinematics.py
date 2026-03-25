#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正运动学模块 (Forward Kinematics)

提供从关节角度计算末端位姿的功能。

支持两种实现：
1. 基于placo库（需要URDF文件）- 精确
2. 基于DH参数（手动配置）- 灵活

使用方法：
    # 方式1：使用URDF
    fk = ForwardKinematics.from_urdf(urdf_path, joint_names, end_effector_frame)
    pose = fk.compute(joint_angles)

    # 方式2：使用DH参数
    fk = ForwardKinematics.from_dh_params(dh_params)
    pose = fk.compute(joint_angles)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from scipy.spatial.transform import Rotation as R


@dataclass
class EndEffectorPose:
    """末端位姿"""
    # 位置 (米)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # 四元数 (qx, qy, qz, qw)
    quaternion: np.ndarray = None

    # 旋转矩阵 (3x3)
    rotation_matrix: np.ndarray = None

    # 变换矩阵 (4x4)
    transform_matrix: np.ndarray = None

    def __post_init__(self):
        if self.quaternion is None:
            self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        if self.rotation_matrix is None:
            self.rotation_matrix = np.eye(3)
        if self.transform_matrix is None:
            self.transform_matrix = np.eye(4)
            self.transform_matrix[:3, :3] = self.rotation_matrix
            self.transform_matrix[:3, 3] = [self.x, self.y, self.z]

    def get_position(self) -> np.ndarray:
        """获取位置向量"""
        return np.array([self.x, self.y, self.z])

    def get_euler_angles(self, seq: str = 'xyz') -> np.ndarray:
        """获取欧拉角 (弧度)"""
        return R.from_matrix(self.rotation_matrix).as_euler(seq)

    def get_quaternion_wxyz(self) -> np.ndarray:
        """获取四元数 (qw, qx, qy, qz)格式"""
        return np.array([self.quaternion[3], self.quaternion[0], self.quaternion[1], self.quaternion[2]])


class ForwardKinematics:
    """
    正运动学计算器

    支持多种实现后端：
    - placo: 基于URDF的精确计算
    - dh: 基于DH参数的计算
    """

    def __init__(self):
        self.backend = None
        self.joint_names = []
        self.num_joints = 0

    @classmethod
    def from_urdf(cls, urdf_path: str, joint_names: List[str],
                  end_effector_frame: str = "gripper_frame_link") -> 'ForwardKinematics':
        """
        从URDF文件创建正运动学计算器

        Args:
            urdf_path: URDF文件路径
            joint_names: 关节名称列表
            end_effector_frame: 末端执行器坐标系名称

        Returns:
            ForwardKinematics实例
        """
        fk = cls()
        fk.backend = 'placo'
        fk.joint_names = joint_names
        fk.num_joints = len(joint_names)
        fk.urdf_path = urdf_path
        fk.end_effector_frame = end_effector_frame

        try:
            from lerobot.model.kinematics import RobotKinematics
            fk._kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name=end_effector_frame,
                joint_names=joint_names
            )
            print(f"✓ 正运动学初始化成功 (placo后端)")
            print(f"  URDF: {urdf_path}")
            print(f"  末端坐标系: {end_effector_frame}")
            print(f"  关节数: {fk.num_joints}")
        except ImportError:
            raise ImportError("placo库未安装，请运行: pip install placo")
        except Exception as e:
            raise RuntimeError(f"正运动学初始化失败: {e}")

        return fk

    @classmethod
    def from_dh_params(cls, dh_params: List[Dict]) -> 'ForwardKinematics':
        """
        从DH参数创建正运动学计算器

        DH参数格式 (标准DH约定):
        {
            'a': 连杆长度 (米),
            'alpha': 连杆扭转角 (弧度),
            'd': 连杆偏距 (米),
            'theta': 关节角偏移 (弧度)
        }

        Args:
            dh_params: DH参数列表

        Returns:
            ForwardKinematics实例
        """
        fk = cls()
        fk.backend = 'dh'
        fk.dh_params = dh_params
        fk.num_joints = len(dh_params)

        print(f"✓ 正运动学初始化成功 (DH参数后端)")
        print(f"  关节数: {fk.num_joints}")

        return fk

    def compute(self, joint_angles: np.ndarray) -> EndEffectorPose:
        """
        计算正运动学

        Args:
            joint_angles: 关节角度 (度)

        Returns:
            EndEffectorPose 末端位姿
        """
        if self.backend == 'placo':
            return self._compute_placo(joint_angles)
        elif self.backend == 'dh':
            return self._compute_dh(joint_angles)
        else:
            raise ValueError("未初始化正运动学后端")

    def _compute_placo(self, joint_angles: np.ndarray) -> EndEffectorPose:
        """使用placo计算正运动学"""
        T = self._kinematics.forward_kinematics(joint_angles)

        # 提取位置
        x, y, z = T[0, 3], T[1, 3], T[2, 3]

        # 提取旋转矩阵
        R_mat = T[:3, :3]

        # 转换为四元数
        quat = R.from_matrix(R_mat).as_quat()  # [qx, qy, qz, qw]

        return EndEffectorPose(
            x=x, y=y, z=z,
            quaternion=quat,
            rotation_matrix=R_mat,
            transform_matrix=T
        )

    def _compute_dh(self, joint_angles: np.ndarray) -> EndEffectorPose:
        """使用DH参数计算正运动学"""
        # 将角度转换为弧度
        theta = np.deg2rad(joint_angles[:self.num_joints])

        # 初始化变换矩阵
        T = np.eye(4)

        for i, params in enumerate(self.dh_params):
            a = params['a']
            alpha = params['alpha']
            d = params['d']
            theta_i = theta[i] + params.get('theta', 0)

            # 标准DH变换矩阵
            ct = np.cos(theta_i)
            st = np.sin(theta_i)
            ca = np.cos(alpha)
            sa = np.sin(alpha)

            Ti = np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0,   sa,      ca,      d],
                [0,   0,       0,       1]
            ])

            T = T @ Ti

        # 提取结果
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        R_mat = T[:3, :3]
        quat = R.from_matrix(R_mat).as_quat()

        return EndEffectorPose(
            x=x, y=y, z=z,
            quaternion=quat,
            rotation_matrix=R_mat,
            transform_matrix=T
        )

    def get_flange_pose(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取法兰位姿（便捷方法）

        Args:
            joint_angles: 关节角度 (度)

        Returns:
            (position, quaternion) 位置和四元数
        """
        pose = self.compute(joint_angles)
        return pose.get_position(), pose.quaternion


# Supre机器人右臂DH参数（示例，需要根据实际机器人调整）
SUPRE_RIGHT_ARM_DH_PARAMS = [
    # Joint 1 (底座旋转)
    {'a': 0.0, 'alpha': np.pi/2, 'd': 0.0, 'theta': 0.0},
    # Joint 2 (肩部俯仰)
    {'a': 0.0, 'alpha': 0.0, 'd': 0.0, 'theta': 0.0},
    # Joint 3 (肩部侧摆)
    {'a': 0.0, 'alpha': np.pi/2, 'd': 0.0, 'theta': 0.0},
    # Joint 4 (前臂俯仰)
    {'a': 0.0, 'alpha': -np.pi/2, 'd': 0.0, 'theta': 0.0},
    # Joint 5 (腕部俯仰)
    {'a': 0.0, 'alpha': np.pi/2, 'd': 0.0, 'theta': 0.0},
    # Joint 6 (手腕旋转)
    {'a': 0.0, 'alpha': 0.0, 'd': 0.0, 'theta': 0.0},
]


def create_fk_from_urdf(urdf_path: str, arm: str = "right") -> ForwardKinematics:
    """
    从URDF创建正运动学计算器的便捷函数

    Args:
        urdf_path: URDF文件路径
        arm: 手臂选择 ("left" 或 "right")

    Returns:
        ForwardKinematics实例
    """
    # 根据手臂选择关节名称
    if arm == "right":
        joint_names = [
            "right_arm_joint_1",
            "right_arm_joint_2",
            "right_arm_joint_3",
            "right_arm_joint_4",
            "right_arm_joint_5",
            "right_arm_joint_6",
        ]
        end_effector = "right_arm_link_6"  # 或根据URDF中的实际名称
    else:
        joint_names = [
            "left_arm_joint_1",
            "left_arm_joint_2",
            "left_arm_joint_3",
            "left_arm_joint_4",
            "left_arm_joint_5",
            "left_arm_joint_6",
        ]
        end_effector = "left_arm_link_6"

    return ForwardKinematics.from_urdf(urdf_path, joint_names, end_effector)


if __name__ == "__main__":
    # 测试
    print("正运动学模块测试")

    # 使用DH参数测试
    fk = ForwardKinematics.from_dh_params(SUPRE_RIGHT_ARM_DH_PARAMS)
    joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose = fk.compute(joints)
    print(f"零位姿态: x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}")