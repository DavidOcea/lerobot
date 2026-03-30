#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
坐标变换模块 (Coordinate Transformer)

基于手眼标定的外参矩阵，实现精确的像素→世界坐标变换。

核心功能：
1. 像素坐标 → 相机坐标系射线
2. 相机坐标系 → 法兰坐标系
3. 法兰坐标系 → 世界坐标系（需要TCP位姿）
4. 像素偏移 → 世界坐标偏移（对齐用）

使用方法：
    from precision_place.calibration.coordinate_transform import CoordinateTransformer

    # 加载外参矩阵
    transformer = CoordinateTransformer.from_calibration_file("hand_eye_extrinsic.yaml")

    # 设置当前TCP位姿
    transformer.set_tcp_pose(position, rotation)

    # 像素偏移转世界偏移
    world_offset = transformer.pixel_offset_to_world(pixel_offset, depth)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import yaml


@dataclass
class CameraPose:
    """相机在世界坐标系中的位姿"""
    position: np.ndarray  # [x, y, z] 米
    rotation_matrix: np.ndarray  # 3x3 旋转矩阵
    transform_matrix: np.ndarray  # 4x4 齐次变换矩阵


class CoordinateTransformer:
    """
    坐标变换器

    使用手眼标定结果进行精确的坐标变换。
    """

    def __init__(self,
                 extrinsic_matrix: np.ndarray,
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray = None):
        """
        初始化坐标变换器

        Args:
            extrinsic_matrix: 4x4 外参矩阵 (Flange -> Camera)
            camera_matrix: 3x3 相机内参矩阵
            dist_coeffs: 畸变系数
        """
        self.T_flange2cam = extrinsic_matrix  # Flange -> Camera
        self.T_cam2flange = np.linalg.inv(extrinsic_matrix)  # Camera -> Flange
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        # 当前TCP位姿
        self.tcp_position: Optional[np.ndarray] = None
        self.tcp_rotation_matrix: Optional[np.ndarray] = None
        self.T_base2flange: Optional[np.ndarray] = None
        self.T_base2cam: Optional[np.ndarray] = None

    @classmethod
    def from_calibration_file(cls, filepath: str,
                              camera_matrix: np.ndarray = None) -> 'CoordinateTransformer':
        """
        从标定文件加载

        Args:
            filepath: 外参矩阵文件路径 (YAML)
            camera_matrix: 相机内参矩阵（可选，使用默认值）

        Returns:
            CoordinateTransformer实例
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"标定文件不存在: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # 加载外参矩阵
        ext_data = data['extrinsic_matrix']['data']
        extrinsic_matrix = np.array(ext_data).reshape(4, 4)

        # 使用默认相机内参或传入的参数
        if camera_matrix is None:
            # 默认值（实际使用时应标定）
            camera_matrix = np.array([
                [500.0, 0, 320.0],
                [0, 500.0, 240.0],
                [0, 0, 1]
            ], dtype=np.float64)

        print(f"✓ 加载外参矩阵: {filepath}")
        print(f"  平移: [{data['translation']['x']:.4f}, {data['translation']['y']:.4f}, {data['translation']['z']:.4f}] 米")
        print(f"  标定RMSE: {data.get('rmse_error_pixels', 'N/A')} 像素")

        return cls(extrinsic_matrix, camera_matrix)

    def set_tcp_pose(self,
                     position: np.ndarray,
                     rotation: np.ndarray,
                     rotation_format: str = "quaternion"):
        """
        设置当前TCP位姿

        Args:
            position: TCP位置 [x, y, z] 米
            rotation: TCP旋转
                - "quaternion": [qx, qy, qz, qw]
                - "euler": [roll, pitch, yaw] 弧度
                - "matrix": 3x3旋转矩阵
            rotation_format: 旋转格式
        """
        self.tcp_position = np.array(position)

        # 转换旋转格式
        if rotation_format == "quaternion":
            self.tcp_rotation_matrix = R.from_quat(rotation).as_matrix()
        elif rotation_format == "euler":
            self.tcp_rotation_matrix = R.from_euler('xyz', rotation).as_matrix()
        elif rotation_format == "matrix":
            self.tcp_rotation_matrix = np.array(rotation)
        else:
            raise ValueError(f"未知的旋转格式: {rotation_format}")

        # 计算变换矩阵
        # T_base2flange: 世界坐标系 -> 法兰坐标系
        self.T_base2flange = np.eye(4)
        self.T_base2flange[:3, :3] = self.tcp_rotation_matrix
        self.T_base2flange[:3, 3] = self.tcp_position

        # T_base2cam: 世界坐标系 -> 相机坐标系
        self.T_base2cam = self.T_base2flange @ self.T_flange2cam

    def get_camera_pose(self) -> CameraPose:
        """
        获取相机在世界坐标系中的位姿

        Returns:
            CameraPose 相机位姿
        """
        if self.T_base2cam is None:
            raise ValueError("请先调用 set_tcp_pose() 设置TCP位姿")

        # T_world2cam 的逆矩阵 = T_cam2world
        T_cam2world = np.linalg.inv(self.T_base2cam)

        return CameraPose(
            position=T_cam2world[:3, 3],
            rotation_matrix=T_cam2world[:3, :3],
            transform_matrix=T_cam2world
        )

    def pixel_to_camera_ray(self, pixel: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        像素坐标转相机坐标系中的射线

        Args:
            pixel: 像素坐标 (u, v)

        Returns:
            (ray_origin, ray_direction) 射线原点和方向（相机坐标系）
        """
        u, v = pixel

        # 相机内参
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # 归一化坐标
        x = (u - cx) / fx
        y = (v - cy) / fy

        # 射线方向（相机坐标系）
        ray_origin = np.array([0, 0, 0])  # 相机光心
        ray_direction = np.array([x, y, 1])  # 归一化方向
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        return ray_origin, ray_direction

    def pixel_to_world_at_depth(self,
                                 pixel: Tuple[float, float],
                                 depth: float) -> np.ndarray:
        """
        像素坐标转世界坐标（给定深度）

        Args:
            pixel: 像素坐标 (u, v)
            depth: 深度（米）

        Returns:
            世界坐标 [x, y, z]
        """
        if self.T_base2cam is None:
            raise ValueError("请先调用 set_tcp_pose() 设置TCP位姿")

        # 像素 -> 相机坐标射线
        ray_origin, ray_direction = self.pixel_to_camera_ray(pixel)

        # 在给定深度处的相机坐标点
        # 注意：在相机坐标系中，Z轴指向场景，所以深度是Z坐标
        point_cam = ray_origin + ray_direction * depth

        # 相机坐标 -> 世界坐标
        point_cam_homogeneous = np.append(point_cam, 1.0)
        T_cam2world = np.linalg.inv(self.T_base2cam)
        point_world = T_cam2world @ point_cam_homogeneous

        return point_world[:3]

    def pixel_offset_to_world_offset(self,
                                      pixel_offset: Tuple[float, float],
                                      depth: float) -> np.ndarray:
        """
        像素偏移转世界坐标偏移

        这是核心对齐函数！
        将相机中观察到的像素偏移转换为世界坐标系中的实际位移。

        Args:
            pixel_offset: 像素偏移 (du, dv)
            depth: 目标深度（米）

        Returns:
            世界坐标偏移 [dx, dy, dz] 米
        """
        if self.T_base2cam is None:
            raise ValueError("请先调用 set_tcp_pose() 设置TCP位姿")

        du, dv = pixel_offset

        # 相机内参
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]

        # 像素偏移 -> 相机坐标系位移
        # 在给定深度处，像素du对应相机坐标系中的dx = du * depth / fx
        dx_cam = du * depth / fx
        dy_cam = dv * depth / fy
        dz_cam = 0  # 像素偏移不影响Z

        offset_cam = np.array([dx_cam, dy_cam, dz_cam])

        # 相机坐标系偏移 -> 世界坐标系偏移
        # 旋转矩阵部分（不考虑平移，因为是偏移量）
        T_cam2world = np.linalg.inv(self.T_base2cam)
        R_cam2world = T_cam2world[:3, :3]

        offset_world = R_cam2world @ offset_cam

        return offset_world

    def world_offset_to_tcp_adjustment(self,
                                        world_offset: np.ndarray,
                                        current_tcp_position: np.ndarray = None) -> np.ndarray:
        """
        世界坐标偏移转TCP位置调整量

        Args:
            world_offset: 世界坐标偏移 [dx, dy, dz] 米
            current_tcp_position: 当前TCP位置（可选，用于验证）

        Returns:
            TCP位置调整量 [dx, dy, dz] 米
        """
        # 对于简单的对齐，TCP调整量 = 世界坐标偏移
        # （假设末端执行器沿世界坐标系移动）
        return world_offset

    def compute_alignment_adjustment(self,
                                      pixel_offset: Tuple[float, float],
                                      depth: float) -> Tuple[np.ndarray, dict]:
        """
        计算对齐调整量（完整流程）

        Args:
            pixel_offset: 像素偏移 (du, dv)
            depth: 深度（米）

        Returns:
            (tcp_adjustment, info) TCP调整量和详细信息
        """
        # 像素偏移 -> 世界偏移
        world_offset = self.pixel_offset_to_world_offset(pixel_offset, depth)

        # 世界偏移 -> TCP调整
        tcp_adjustment = self.world_offset_to_tcp_adjustment(world_offset)

        info = {
            'pixel_offset': pixel_offset,
            'depth_m': depth,
            'world_offset_m': world_offset,
            'tcp_adjustment_m': tcp_adjustment,
            'camera_pose': self.get_camera_pose() if self.T_base2cam else None
        }

        return tcp_adjustment, info

    def get_rotation_correction(self,
                                 rotation_error_deg: float,
                                 depth: float) -> Tuple[float, np.ndarray]:
        """
        计算旋转校正量

        Args:
            rotation_error_deg: 旋转误差（度）
            depth: 目标深度（米）

        Returns:
            (tcp_rotation_deg, lateral_offset) TCP旋转角度和横向偏移
        """
        # 简单处理：旋转误差直接作为TCP旋转调整
        # 更复杂的处理需要考虑旋转中心位置
        tcp_rotation_deg = -rotation_error_deg

        # 旋转时的横向偏移（取决于旋转中心）
        # 这里简化处理，假设旋转中心在TCP附近
        lateral_offset = np.array([0, 0, 0])

        return tcp_rotation_deg, lateral_offset

    @staticmethod
    def estimate_depth_from_size(pixel_size: float,
                                  known_size_m: float,
                                  focal_length_px: float) -> float:
        """
        从已知物体尺寸估计深度

        Args:
            pixel_size: 物体在图像中的像素尺寸
            known_size_m: 物体的实际尺寸（米）
            focal_length_px: 焦距（像素）

        Returns:
            估计的深度（米）
        """
        if pixel_size <= 0:
            return 0.3  # 默认深度

        depth = known_size_m * focal_length_px / pixel_size
        return depth


# ==================== 集成控制器 ====================

class AlignmentController:
    """
    对齐控制器 - 基于手眼标定

    替代原来的灵敏度方法，使用精确的坐标变换。
    """

    def __init__(self, transformer: CoordinateTransformer, robot):
        """
        初始化对齐控制器

        Args:
            transformer: 坐标变换器
            robot: 机器人控制器
        """
        self.transformer = transformer
        self.robot = robot

        # 对齐参数
        self.alignment_threshold_pixel = 5.0  # 像素
        self.alignment_threshold_mm = 1.0  # 毫米
        self.max_iterations = 20
        self.step_scale = 0.8  # 步长缩放（避免过冲）

    def align_xy(self,
                 get_pixel_offset: callable,
                 get_depth: callable,
                 get_tcp_pose: callable,
                 move_tcp: callable,
                 on_progress: callable = None) -> Tuple[bool, dict]:
        """
        XY对齐主循环

        Args:
            get_pixel_offset: 获取像素偏移的函数 () -> (du, dv)
            get_depth: 获取深度的函数 () -> depth_m
            get_tcp_pose: 获取TCP位姿的函数 () -> (position, rotation)
            move_tcp: 移动TCP的函数 (position_delta) -> success
            on_progress: 进度回调

        Returns:
            (success, info) 对齐结果
        """
        for iteration in range(self.max_iterations):
            # 1. 获取当前状态
            tcp_pos, tcp_rot = get_tcp_pose()
            self.transformer.set_tcp_pose(tcp_pos, tcp_rot)

            # 2. 获取像素偏移
            pixel_offset = get_pixel_offset()
            du, dv = pixel_offset

            # 检查是否对齐
            pixel_error = np.sqrt(du**2 + dv**2)
            if pixel_error < self.alignment_threshold_pixel:
                return True, {
                    'iterations': iteration + 1,
                    'final_pixel_error': pixel_error,
                    'converged': True
                }

            # 3. 获取深度
            depth = get_depth()

            # 4. 计算调整量
            tcp_adjustment, info = self.transformer.compute_alignment_adjustment(
                pixel_offset, depth
            )

            # 应用缩放
            tcp_adjustment = tcp_adjustment * self.step_scale

            # 检查调整量是否过小
            adjustment_mm = np.linalg.norm(tcp_adjustment) * 1000
            if adjustment_mm < self.alignment_threshold_mm:
                return True, {
                    'iterations': iteration + 1,
                    'final_pixel_error': pixel_error,
                    'converged': True,
                    'reason': 'adjustment_below_threshold'
                }

            # 5. 执行移动
            success = move_tcp(tcp_adjustment)
            if not success:
                return False, {
                    'iterations': iteration + 1,
                    'error': 'move_failed'
                }

            # 进度回调
            if on_progress:
                on_progress(iteration + 1, pixel_error, adjustment_mm)

        return False, {
            'iterations': self.max_iterations,
            'converged': False,
            'reason': 'max_iterations_reached'
        }


if __name__ == "__main__":
    # 测试
    print("坐标变换模块测试")

    # 创建模拟外参矩阵
    T = np.eye(4)
    T[:3, 3] = [0.05, 0.0, 0.1]  # 相机在法兰坐标系中偏移

    # 创建变换器
    camera_matrix = np.array([
        [500.0, 0, 320.0],
        [0, 500.0, 240.0],
        [0, 0, 1]
    ])

    transformer = CoordinateTransformer(T, camera_matrix)

    # 设置TCP位姿
    tcp_pos = np.array([0.3, 0.2, 0.5])
    tcp_rot = np.array([0, 0, 0, 1])  # 无旋转
    transformer.set_tcp_pose(tcp_pos, tcp_rot, "quaternion")

    # 测试像素偏移转换
    pixel_offset = (50, 30)  # 像素
    depth = 0.4  # 米

    world_offset = transformer.pixel_offset_to_world_offset(pixel_offset, depth)
    print(f"像素偏移 {pixel_offset} @ 深度 {depth}m")
    print(f"  -> 世界偏移: [{world_offset[0]*1000:.2f}, {world_offset[1]*1000:.2f}, {world_offset[2]*1000:.2f}] mm")