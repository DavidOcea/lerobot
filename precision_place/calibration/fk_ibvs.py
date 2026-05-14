#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FK-IBVS 控制器 (Forward Kinematics based IBVS)

基于正运动学(FK) + 手眼标定 + 相机内参的图像视觉伺服。
与 SimpleIBVS (灵敏度标定) 的对比方案。

核心原理:
  AprilTag检测 → 反投影到世界坐标 → FK+投影计算解析雅可比 → 关节调整

优势:
  - 不需要灵敏度标定数据
  - 全工作空间有效
  - 深度通过FK直接计算，不需要tag尺寸估计
  - 雅可比在任意关节配置下数学精确
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FKIBVSController:
    """基于正运动学的IBVS控制器"""

    def __init__(self,
                 fk_solver,                    # ForwardKinematics 实例
                 T_flange_cam: np.ndarray,     # 4x4 手眼矩阵 (Flange → Camera)
                 camera_matrix: np.ndarray,    # 3x3 相机内参
                 joint_indices: List[int] = None,
                 gain: float = 0.8,
                 damping_ratio: float = 0.03,
                 jacobian_delta: float = 0.15,
                 joint_sign_corrections: Dict[int, Tuple[int, int]] = None):
        """
        Args:
            fk_solver: ForwardKinematics实例
            T_flange_cam: 4x4 外参矩阵 (Flange → Camera)
            camera_matrix: 3x3 相机内参矩阵
            joint_indices: 使用的关节索引列表 (默认右臂 j7-j12 + j14)
            gain: 控制增益
            damping_ratio: 阻尼比 (λ = error * ratio)
            jacobian_delta: 数值雅可比的关节扰动 (度)
            joint_sign_corrections: 关节方向修正 {joint_idx: (dx_sign, dy_sign)}
                用于修正URDF与真实机器人关节旋转方向的差异。
                例如 {8: (-1, 1)} 表示j8的dx灵敏度翻转、dy保持。
        """
        self.fk = fk_solver
        self.T_flange_cam = T_flange_cam
        self.T_cam_flange = np.linalg.inv(T_flange_cam)
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

        self.joint_indices = joint_indices or [7, 8, 9, 10, 11, 12, 14]
        self.gain = gain
        self.damping_ratio = damping_ratio
        self.jacobian_delta = jacobian_delta
        self.max_adjust = 2.0
        self.joint_sign_corrections = joint_sign_corrections or {}

        # 状态
        self.tag_world_pos: Optional[np.ndarray] = None  # tag在世界坐标系中的位置
        self.tag_estimated = False
        self._estimation_depth = 0.0

    # ==================== Tag世界坐标估计 ====================

    def estimate_tag_world_pos(self, joints: np.ndarray,
                                tag_center_px: Tuple[float, float],
                                depth_mm: float) -> Optional[np.ndarray]:
        """
        从单帧检测反投影tag到世界坐标。

        Args:
            joints: 当前16维关节角度 (度)
            tag_center_px: tag中心像素坐标 (u, v)
            depth_mm: 当前tag深度 (mm), 从tag尺寸估算

        Returns:
            tag世界坐标 [x, y, z] (米)
        """
        T_world_cam = self._get_camera_pose_world(joints)
        if T_world_cam is None:
            return None

        # 反投影: 像素 → 相机坐标系
        depth_m = depth_mm / 1000.0
        x_cam = (tag_center_px[0] - self.cx) * depth_m / self.fx
        y_cam = (tag_center_px[1] - self.cy) * depth_m / self.fy
        z_cam = depth_m
        P_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # 相机 → 世界
        P_world = T_world_cam @ P_cam
        pos = P_world[:3].copy()

        self.tag_world_pos = pos
        self.tag_estimated = True
        self._estimation_depth = depth_mm

        return pos

    def set_tag_world_pos(self, pos: np.ndarray):
        """直接设置tag世界坐标 (跳过反投影)"""
        self.tag_world_pos = np.asarray(pos, dtype=float)
        self.tag_estimated = True

    # ==================== FK + 投影 ====================

    def _get_camera_pose_world(self, joints: np.ndarray) -> Optional[np.ndarray]:
        """计算相机在世界坐标系中的位姿 T_world_cam (4x4)"""
        try:
            # FK: 关节 → 法兰在世界坐标系中
            ee_pose = self.fk.compute(joints)
            T_world_flange = ee_pose.transform_matrix

            # 法兰 → 相机
            T_world_cam = T_world_flange @ self.T_flange_cam
            return T_world_cam
        except Exception as e:
            print(f"[FK-IBVS] FK计算失败: {e}")
            return None

    def project_tag_to_pixel(self, joints: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        将已知tag世界坐标投影到图像平面。

        Returns:
            (pixel_uv, depth_mm) 或 (None, 0) 如果在相机后方
        """
        if self.tag_world_pos is None:
            return None, 0.0

        T_world_cam = self._get_camera_pose_world(joints)
        if T_world_cam is None:
            return None, 0.0

        T_cam_world = np.linalg.inv(T_world_cam)
        P_cam = T_cam_world @ np.append(self.tag_world_pos, 1.0)

        if P_cam[2] <= 0.001:
            return None, 0.0  # 在相机后方

        u = self.fx * P_cam[0] / P_cam[2] + self.cx
        v = self.fy * P_cam[1] / P_cam[2] + self.cy
        depth_mm = float(P_cam[2] * 1000.0)

        return np.array([u, v]), depth_mm

    def get_fk_depth(self, joints: np.ndarray) -> float:
        """通过FK直接计算当前相机-tag距离 (mm)"""
        if self.tag_world_pos is None:
            return 0.0

        T_world_cam = self._get_camera_pose_world(joints)
        if T_world_cam is None:
            return 0.0

        cam_pos_world = T_world_cam[:3, 3]
        dist = np.linalg.norm(cam_pos_world - self.tag_world_pos)
        return float(dist * 1000.0)

    # ==================== 解析雅可比 ====================

    def compute_jacobian(self, joints: np.ndarray) -> Tuple[Optional[np.ndarray], List[int], Optional[np.ndarray], float]:
        """
        通过FK+投影的数值微分计算解析雅可比 J = ∂(pixel)/∂(joint)。

        对每个关节施加微小扰动，重新计算FK和投影，得到像素变化。

        Returns:
            (J_2×N, joint_indices, base_pixel, base_depth_mm)
            或 (None, [], None, 0) 若失败
        """
        if self.tag_world_pos is None:
            return None, [], None, 0.0

        base_pixel, base_depth = self.project_tag_to_pixel(joints)
        if base_pixel is None:
            return None, [], None, 0.0

        n_joints = len(self.joint_indices)
        J = np.zeros((2, n_joints))
        active_indices = []

        for i, jidx in enumerate(self.joint_indices):
            perturbed = joints.copy().astype(float)
            perturbed[jidx] += self.jacobian_delta

            new_pixel, _ = self.project_tag_to_pixel(perturbed)
            if new_pixel is not None:
                J[:, i] = (new_pixel - base_pixel) / self.jacobian_delta
                active_indices.append(jidx)
            else:
                # 关节扰动导致tag移到相机后方 → 灵敏度近似为0
                J[:, i] = 0.0
                active_indices.append(jidx)

        # 应用关节方向修正 (修正URDF与真实机器人旋转方向的差异)
        for i, jidx in enumerate(active_indices):
            if jidx in self.joint_sign_corrections:
                dx_sign, dy_sign = self.joint_sign_corrections[jidx]
                J[0, i] *= dx_sign
                J[1, i] *= dy_sign

        return J, active_indices, base_pixel, base_depth

    # ==================== 关节调整量计算 ====================

    def compute_joint_adjustments(self,
                                   pixel_error_x: float,
                                   pixel_error_y: float,
                                   current_joints: np.ndarray,
                                   current_depth_mm: float = None) -> Dict[int, float]:
        """
        使用解析雅可比计算关节调整量。

        Args:
            pixel_error_x: X方向像素误差 (实际 - 目标)
            pixel_error_y: Y方向像素误差
            current_joints: 当前16维关节角度 (度)
            current_depth_mm: 未使用 (FK直接计算深度)

        Returns:
            {joint_idx: delta_deg} 调整量字典
        """
        J, active_indices, base_pixel, fk_depth = self.compute_jacobian(current_joints)
        if J is None:
            return {}

        error = np.array([pixel_error_x, pixel_error_y])

        # 阻尼最小二乘
        error_norm = float(np.linalg.norm(error))
        damping = max(0.3, error_norm * self.damping_ratio)

        JW = J  # 解析雅可比不需要位置权重缩放
        JW_JWT = JW @ JW.T  # 2×2

        try:
            z = np.linalg.solve(JW_JWT + damping**2 * np.eye(2), -error)
            delta = JW.T @ z
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(JW) @ (-error)

        # 应用增益
        delta = delta * self.gain

        # 裁剪
        delta = np.clip(delta, -self.max_adjust, self.max_adjust)

        adjustments = {}
        for i, jidx in enumerate(active_indices):
            if abs(delta[i]) > 0.005:
                adjustments[jidx] = float(delta[i])

        return adjustments

    # ==================== 深度辅助 ====================

    def get_depth_info(self, joints: np.ndarray) -> Dict:
        """
        获取通过FK计算的深度信息 (用于对齐判定)。

        Returns:
            {'depth_mm': float, 'estimated': bool, 'tag_world_pos': ...}
        """
        fk_depth = self.get_fk_depth(joints)
        return {
            'depth_mm': fk_depth,
            'estimated': self.tag_estimated,
            'tag_world_pos': self.tag_world_pos,
            'estimation_depth': fk_depth,
        }

    # ==================== 状态 ====================

    def reset(self):
        """重置tag世界坐标估计"""
        self.tag_world_pos = None
        self.tag_estimated = False
        self._estimation_depth = 0.0

    def is_ready(self) -> bool:
        """tag世界坐标是否已估计"""
        return self.tag_estimated

    @property
    def tag_world_pos_mm(self) -> Optional[np.ndarray]:
        """返回tag世界坐标 (mm)"""
        if self.tag_world_pos is None:
            return None
        return self.tag_world_pos * 1000.0
