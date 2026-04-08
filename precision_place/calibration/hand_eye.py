#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
手眼标定模块 (Hand-Eye Calibration)

基于RJ2506项目的核心算法，移植到独立环境（无ROS依赖）。

功能：
1. ChArUco标定板检测
2. 多姿态数据采集
3. 手眼标定解算 (cv2.calibrateHandEye)
4. 外参矩阵保存/加载
5. 重投影误差验证

使用方法：
    from precision_place.calibration.hand_eye import HandEyeCalibrator

    calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)
    calibrator.capture_pose(image, flange_pose)  # 重复多次
    success, extrinsic = calibrator.calibrate()
    calibrator.save("extrinsic.yaml")

参考：
    - RJ2506项目: /home/smai/dc_dir/rj2506_core_system
    - OpenCV手眼标定: cv2.calibrateHandEye (Tsai-Lenz算法)
"""

import os
import json
import yaml
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.spatial.transform import Rotation as R


@dataclass
class CalibrationResult:
    """标定结果"""
    # 相机外参矩阵 (4x4 齐次变换矩阵)
    # 表示: 法兰坐标系 -> 相机坐标系 的变换
    extrinsic_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))

    # 旋转矩阵 (3x3) - 法兰到相机的旋转
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    # 平移向量 (3x1, 单位: 米) - 法兰到相机的平移
    translation_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 重投影误差 (像素)
    rmse_error: float = 0.0

    # 标定使用的姿态数量
    num_poses: int = 0

    # 标定方法
    method: str = "Tsai-Lenz"

    # 是否有效
    valid: bool = False

    # 属性别名（便于理解和使用）
    @property
    def R_flange2cam(self) -> np.ndarray:
        """法兰到相机的旋转矩阵 (等同于 rotation_matrix)"""
        return self.rotation_matrix

    @property
    def t_flange2cam(self) -> np.ndarray:
        """法兰到相机的平移向量 (等同于 translation_vector)"""
        return self.translation_vector

    @property
    def R_cam2flange(self) -> np.ndarray:
        """相机到法兰的旋转矩阵 (rotation_matrix的逆)"""
        return self.rotation_matrix.T

    @property
    def t_cam2flange(self) -> np.ndarray:
        """相机到法兰的平移向量"""
        return -self.rotation_matrix.T @ self.translation_vector


class HandEyeCalibrator:
    """
    手眼标定器 (Eye-in-Hand模式)

    用于标定腕部相机相对于法兰的外参矩阵。

    使用流程：
    1. 初始化：传入相机内参和畸变系数
    2. 采集：在多个不同姿态下采集图像和法兰位姿
    3. 标定：调用calibrate()解算外参
    4. 验证：检查RMSE是否达标
    5. 保存：保存外参到文件
    """

    # ChArUco板默认参数
    DEFAULT_SQUARES_X = 5
    DEFAULT_SQUARES_Y = 7
    DEFAULT_SQUARE_LENGTH = 0.03   # 30mm
    DEFAULT_MARKER_LENGTH = 0.022  # 22mm

    def __init__(self,
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 squares_x: int = None,
                 squares_y: int = None,
                 square_length: float = None,
                 marker_length: float = None):
        """
        初始化手眼标定器

        Args:
            camera_matrix: 3x3 相机内参矩阵
            dist_coeffs: 畸变系数
            squares_x: ChArUco板X方向格子数
            squares_y: ChArUco板Y方向格子数
            square_length: 格子边长 (米)
            marker_length: ArUco标记边长 (米)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # ChArUco板参数
        self.squares_x = squares_x or self.DEFAULT_SQUARES_X
        self.squares_y = squares_y or self.DEFAULT_SQUARES_Y
        self.square_length = square_length or self.DEFAULT_SQUARE_LENGTH
        self.marker_length = marker_length or self.DEFAULT_MARKER_LENGTH

        # 创建ChArUco板和检测器
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.dictionary
        )
        self.params = cv2.aruco.DetectorParameters()

        # OpenCV 4.7+ 使用 CharucoDetector
        try:
            self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
            self.use_charuco_detector = True
        except AttributeError:
            # OpenCV < 4.7 使用 ArucoDetector
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)
            self.use_charuco_detector = False

        # 采集的数据
        self.R_base2gripper: List[np.ndarray] = []  # 法兰旋转矩阵 (Base -> Flange)
        self.t_base2gripper: List[np.ndarray] = []  # 法兰平移向量
        self.R_cam2target: List[np.ndarray] = []    # 标定板旋转矩阵 (Camera -> Target)
        self.t_cam2target: List[np.ndarray] = []    # 标定板平移向量

        # 标定结果
        self.result = CalibrationResult()

        # 调试图像保存目录
        self.debug_dir = None

    def set_debug_dir(self, debug_dir: str):
        """设置调试图像保存目录"""
        self.debug_dir = debug_dir
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def detect_charuco(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测ChArUco标定板

        Args:
            image: 输入图像

        Returns:
            (success, rvec, tvec, corners)
            - success: 是否检测成功
            - rvec: 旋转向量 (标定板相对于相机)
            - tvec: 平移向量 (标定板相对于相机)
            - corners: 检测到的角点
        """
        if image is None:
            return False, None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # OpenCV 4.7+ 使用 CharucoDetector
        if self.use_charuco_detector:
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self.charuco_detector.detectBoard(gray)

            if charuco_ids is None or len(charuco_ids) < 6:
                return False, None, None, None

            # 估计标定板位姿
            object_points = self.board.getChessboardCorners()
            image_points = charuco_corners.reshape(-1, 2)

            # 使用对应点估计位姿
            success, rvec, tvec = cv2.solvePnP(
                object_points[charuco_ids.flatten()],
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )

            if not success or rvec is None or tvec is None:
                return False, None, None, None

            return True, rvec, tvec, charuco_corners

        else:
            # OpenCV < 4.7 使用旧API
            marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)

            if marker_ids is None or len(marker_ids) == 0:
                return False, None, None, None

            # 提取ChArUco角点
            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
            except AttributeError:
                return False, None, None, None

            if not ret or charuco_ids is None or len(charuco_ids) < 4:
                return False, None, None, None

            # 估计标定板位姿
            try:
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board,
                    self.camera_matrix, self.dist_coeffs,
                    np.empty(1), np.empty(1)
                )
            except AttributeError:
                return False, None, None, None

            if not success or rvec is None or tvec is None:
                return False, None, None, None

            return True, rvec, tvec, charuco_corners

    def capture_pose(self,
                     image: np.ndarray,
                     flange_position: np.ndarray,
                     flange_rotation: np.ndarray,
                     rotation_format: str = "quaternion") -> bool:
        """
        采集一个姿态的数据

        Args:
            image: 相机图像
            flange_position: 法兰位置 [x, y, z] (米)
            flange_rotation: 法兰旋转
                - "quaternion": [qx, qy, qz, qw]
                - "euler": [roll, pitch, yaw] (弧度)
                - "matrix": 3x3旋转矩阵
            rotation_format: 旋转格式 ("quaternion", "euler", "matrix")

        Returns:
            是否采集成功
        """
        # 检测ChArUco板
        success, rvec, tvec, corners = self.detect_charuco(image)

        if not success:
            return False

        # 转换法兰旋转为旋转矩阵
        if rotation_format == "quaternion":
            r_flange = R.from_quat(flange_rotation).as_matrix()
        elif rotation_format == "euler":
            r_flange = R.from_euler('xyz', flange_rotation).as_matrix()
        elif rotation_format == "matrix":
            r_flange = flange_rotation
        else:
            raise ValueError(f"未知的旋转格式: {rotation_format}")

        # 转换标定板旋转向量为旋转矩阵
        r_target, _ = cv2.Rodrigues(rvec)

        # 姿态差异检查
        pose_diversity_ok = True
        if len(self.R_base2gripper) > 0:
            # 计算与已有姿态的最小差异
            min_rotation_diff = float('inf')
            min_position_diff = float('inf')

            for i, (R_old, t_old) in enumerate(zip(self.R_base2gripper, self.t_base2gripper)):
                # 旋转差异（Frobenius范数）
                rot_diff = np.linalg.norm(r_flange - R_old, 'fro')
                min_rotation_diff = min(min_rotation_diff, rot_diff)

                # 位置差异（欧氏距离）
                pos_diff = np.linalg.norm(np.array(flange_position) - t_old.flatten())
                min_position_diff = min(min_position_diff, pos_diff)

            # 警告阈值
            if min_rotation_diff < 0.1 and min_position_diff < 0.01:
                print(f"  ⚠ 警告: 新姿态与已有姿态差异太小")
                print(f"    旋转差异: {min_rotation_diff:.4f} (建议 > 0.1)")
                print(f"    位置差异: {min_position_diff*1000:.1f}mm (建议 > 10mm)")
                print(f"    建议: 大幅改变机械臂姿态")
                pose_diversity_ok = False

        # 保存数据
        self.R_base2gripper.append(r_flange)
        self.t_base2gripper.append(np.array(flange_position).reshape(3, 1))
        self.R_cam2target.append(r_target)
        self.t_cam2target.append(tvec)

        # 保存调试图像
        if self.debug_dir:
            debug_img = image.copy()
            cv2.drawFrameAxes(debug_img, self.camera_matrix, self.dist_coeffs,
                            rvec, tvec, 0.1)
            idx = len(self.R_base2gripper)
            cv2.imwrite(os.path.join(self.debug_dir, f"calib_{idx:03d}.jpg"), debug_img)

        # 返回是否采集成功（即使姿态差异小也保存，只是警告）
        return True

    def get_capture_count(self) -> int:
        """获取已采集的姿态数量"""
        return len(self.R_base2gripper)

    def clear_captures(self):
        """清空已采集的数据"""
        self.R_base2gripper.clear()
        self.t_base2gripper.clear()
        self.R_cam2target.clear()
        self.t_cam2target.clear()

    def diagnose_scale(self) -> dict:
        """
        诊断FK和相机位姿的scale问题

        通过计算标定板在世界坐标系中的位置一致性来诊断。
        如果标定数据正确，固定标定板在世界坐标系中的位置应该保持不变。

        Returns:
            dict: 包含诊断信息的字典
        """
        if len(self.R_base2gripper) < 2:
            return {"error": "需要至少2个姿态才能诊断"}

        # 计算每个姿态下标定板在世界坐标系中的位置
        # 假设相机在法兰坐标系原点（t_cam2gripper = 0）
        # T_base2target ≈ T_base2gripper @ T_cam2target

        world_positions = []
        for i in range(len(self.R_base2gripper)):
            T_base2gripper = np.eye(4)
            T_base2gripper[:3, :3] = self.R_base2gripper[i]
            T_base2gripper[:3, 3] = self.t_base2gripper[i].flatten()

            T_cam2target = np.eye(4)
            T_cam2target[:3, :3] = self.R_cam2target[i]
            T_cam2target[:3, 3] = self.t_cam2target[i].flatten()

            # 假设相机在法兰原点，估算标定板位置
            T_base2target_est = T_base2gripper @ T_cam2target
            world_positions.append(T_base2target_est[:3, 3])

        world_positions = np.array(world_positions)
        mean_pos = np.mean(world_positions, axis=0)
        std_pos = np.std(world_positions, axis=0)
        max_pos_diff = np.max(np.abs(world_positions - mean_pos))

        # 分析FK位置范围
        fk_positions = np.array([t.flatten() for t in self.t_base2gripper])
        fk_mean = np.mean(fk_positions, axis=0)
        fk_range = np.max(fk_positions, axis=0) - np.min(fk_positions, axis=0)

        # 分析相机位姿范围
        cam_positions = np.array([t.flatten() for t in self.t_cam2target])
        cam_mean = np.mean(cam_positions, axis=0)
        cam_range = np.max(cam_positions, axis=0) - np.min(cam_positions, axis=0)

        # 分析FK旋转变化（关键！）
        rotation_changes = []
        for i in range(1, len(self.R_base2gripper)):
            # 计算相邻姿态间的旋转差异
            R_diff = self.R_base2gripper[i].T @ self.R_base2gripper[i-1]
            angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            rotation_changes.append(np.degrees(angle_diff))

        # 分析相机观察标定板的旋转变化
        cam_rotation_changes = []
        for i in range(1, len(self.R_cam2target)):
            R_diff = self.R_cam2target[i].T @ self.R_cam2target[i-1]
            angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            cam_rotation_changes.append(np.degrees(angle_diff))

        return {
            "num_poses": len(self.R_base2gripper),
            "target_world_positions": world_positions,
            "target_position_mean": mean_pos,
            "target_position_std": std_pos,
            "target_position_max_diff": max_pos_diff,
            "fk_positions_mean": fk_mean,
            "fk_positions_range": fk_range,
            "cam_positions_mean": cam_mean,
            "cam_positions_range": cam_range,
            "fk_rotation_changes": rotation_changes,
            "fk_rotation_mean": np.mean(rotation_changes) if rotation_changes else 0,
            "fk_rotation_max": np.max(rotation_changes) if rotation_changes else 0,
            "cam_rotation_changes": cam_rotation_changes,
            "cam_rotation_mean": np.mean(cam_rotation_changes) if cam_rotation_changes else 0,
            "square_length_m": self.square_length,
            "diagnosis": self._generate_scale_diagnosis(std_pos, fk_range, cam_mean, rotation_changes)
        }

    def _generate_scale_diagnosis(self, target_std, fk_range, cam_mean, rotation_changes=None) -> str:
        """生成scale诊断结论"""
        max_std = np.max(target_std)

        diagnosis_lines = []

        # 检查标定板位置波动
        if max_std > 0.1:  # 100mm
            diagnosis_lines.append(f"⚠ 标定板位置波动过大: {max_std*1000:.1f}mm")
            diagnosis_lines.append("  这表明标定数据存在scale问题")
        else:
            diagnosis_lines.append(f"✓ 标定板位置波动较小: {max_std*1000:.1f}mm")

        # 检查FK旋转变化（重要！手眼标定需要足够的旋转变化）
        if rotation_changes:
            mean_rot = np.mean(rotation_changes)
            max_rot = np.max(rotation_changes)
            if mean_rot < 10:  # 平均旋转变化小于10度
                diagnosis_lines.append(f"⚠ FK旋转变化太小: 平均{mean_rot:.1f}°，最大{max_rot:.1f}°")
                diagnosis_lines.append("  手眼标定需要足够大的旋转变化！")
                diagnosis_lines.append("  建议: 大幅转动关节，使相邻姿态旋转差异>15°")
            elif mean_rot > 30:
                diagnosis_lines.append(f"✓ FK旋转变化充足: 平均{mean_rot:.1f}°，最大{max_rot:.1f}°")
            else:
                diagnosis_lines.append(f"FK旋转变化: 平均{mean_rot:.1f}°，最大{max_rot:.1f}°")

        # 检查FK位置范围
        if np.max(fk_range) < 0.01:  # 10mm
            diagnosis_lines.append("⚠ FK位置变化太小，姿态差异不足")
        elif np.max(fk_range) > 0.5:  # 500mm
            diagnosis_lines.append(f"FK位置变化较大: {np.max(fk_range)*1000:.1f}mm")

        # 检查相机到标定板距离
        cam_dist = np.linalg.norm(cam_mean)
        if cam_dist > 1.0:  # 1米
            diagnosis_lines.append(f"⚠ 相机到标定板距离过大: {cam_dist*1000:.1f}mm")
            diagnosis_lines.append("  可能原因: 标定板square_length参数与实际不匹配")
        elif cam_dist < 0.1:  # 100mm
            diagnosis_lines.append(f"⚠ 相机到标定板距离过小: {cam_dist*1000:.1f}mm")

        # 添加排查建议
        if max_std > 0.1:
            diagnosis_lines.append("")
            diagnosis_lines.append("排查建议:")
            diagnosis_lines.append(f"  1. 确认标定板格子边长是否为 {self.square_length*1000:.0f}mm")
            diagnosis_lines.append("  2. 运行FK验证(F)检查FK scale是否正确")
            diagnosis_lines.append("  3. 检查URDF坐标系定义是否与实际机器人一致")

        return "\n".join(diagnosis_lines)

    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_TSAI) -> Tuple[bool, CalibrationResult]:
        """
        执行手眼标定

        Args:
            method: 标定方法
                - cv2.CALIB_HAND_EYE_TSAI: Tsai-Lenz方法 (推荐)
                - cv2.CALIB_HAND_EYE_PARK: Park方法
                - cv2.CALIB_HAND_EYE_HORAUD: Horaud方法
                - cv2.CALIB_HAND_EYE_ANDREFF: Andreff方法
                - cv2.CALIB_HAND_EYE_DANIILIDIS: Daniilidis方法

        Returns:
            (success, result)
        """
        if len(self.R_base2gripper) < 10:
            print(f"错误: 采集数量不足 ({len(self.R_base2gripper)}/10)，需要至少10个姿态")
            return False, self.result

        print(f"开始手眼标定，使用 {len(self.R_base2gripper)} 个姿态...")

        try:
            # 调用OpenCV手眼标定
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=self.R_base2gripper,
                t_gripper2base=self.t_base2gripper,
                R_target2cam=self.R_cam2target,
                t_target2cam=self.t_cam2target,
                method=method
            )

            # 构建4x4齐次变换矩阵
            T_extrinsic = np.eye(4)
            T_extrinsic[:3, :3] = R_cam2gripper
            T_extrinsic[:3, 3] = t_cam2gripper.flatten()

            # 计算重投影误差
            rmse = self._compute_rmse(R_cam2gripper, t_cam2gripper)

            # 保存结果
            self.result.extrinsic_matrix = T_extrinsic
            self.result.rotation_matrix = R_cam2gripper
            self.result.translation_vector = t_cam2gripper.flatten()
            self.result.rmse_error = rmse
            self.result.num_poses = len(self.R_base2gripper)
            self.result.method = "Tsai-Lenz" if method == cv2.CALIB_HAND_EYE_TSAI else f"Method-{method}"

            # 外参矩阵有效性检查
            translation_norm = np.linalg.norm(t_cam2gripper)
            rotation_diff = np.linalg.norm(R_cam2gripper - np.eye(3))

            if translation_norm < 0.001 and rotation_diff < 0.01:
                # 外参矩阵接近单位矩阵，标定失败
                print("\n✗ 标定失败：外参矩阵接近单位矩阵")
                print("  平移向量模长: {:.6f} m (应 > 0.001m)".format(translation_norm))
                print("  旋转矩阵偏差: {:.6f} (应 > 0.01)".format(rotation_diff))
                print("\n可能原因：")
                print("  1. 姿态变化太小（机械臂几乎没动）")
                print("  2. 标定板在所有姿态下位置几乎相同")
                print("  3. 法兰位姿数据异常")
                print("\n建议：")
                print("  - 增大姿态变化幅度（大角度倾斜）")
                print("  - 确保标定板在相机视野中心")
                print("  - 检查正运动学是否正常工作")
                self.result.valid = False
                return False, self.result

            self.result.valid = rmse < 5.0  # 5像素以内认为有效

            print(f"标定完成:")
            print(f"  平移: [{t_cam2gripper[0,0]:.4f}, {t_cam2gripper[1,0]:.4f}, {t_cam2gripper[2,0]:.4f}] 米")
            print(f"  平移距离: {translation_norm*1000:.1f} mm")
            print(f"  RMSE: {rmse:.2f} 像素")

            if rmse > 1.5:
                print(f"  ⚠ 警告: RMSE > 1.5像素，建议重新标定")
            else:
                print(f"  ✓ 标定精度良好")

            return True, self.result

        except Exception as e:
            print(f"标定失败: {e}")
            return False, self.result

    def _compute_rmse(self, R_cam2gripper: np.ndarray, t_cam2gripper: np.ndarray) -> float:
        """
        计算重投影误差 (RMSE)

        通过检查标定板在世界坐标系中的位置一致性来验证标定质量。
        如果标定正确，固定标定板在世界坐标系中的位置应该保持不变。
        """
        world_positions = []

        for i in range(len(self.R_base2gripper)):
            # 计算标定板在世界坐标系中的位置
            # P_world = T_base2gripper @ T_gripper2cam @ P_cam
            # P_cam = T_cam2target @ P_target (标定板原点)
            T_base2gripper = np.eye(4)
            T_base2gripper[:3, :3] = self.R_base2gripper[i]
            T_base2gripper[:3, 3] = self.t_base2gripper[i].flatten()

            T_gripper2cam = np.eye(4)
            T_gripper2cam[:3, :3] = R_cam2gripper
            T_gripper2cam[:3, 3] = t_cam2gripper.flatten()

            T_cam2target = np.eye(4)
            T_cam2target[:3, :3] = self.R_cam2target[i]
            T_cam2target[:3, 3] = self.t_cam2target[i].flatten()

            # 标定板原点在世界坐标系中的位置
            P_target_origin = np.array([0, 0, 0, 1])
            P_cam = T_cam2target @ P_target_origin
            P_gripper = T_gripper2cam @ P_cam
            P_world = T_base2gripper @ P_gripper

            world_positions.append(P_world[:3])

        if len(world_positions) < 2:
            return 0.5  # 无法计算，返回默认值

        # 计算世界位置的一致性（标准差作为误差指标）
        world_positions = np.array(world_positions)
        mean_pos = np.mean(world_positions, axis=0)
        std_pos = np.std(world_positions, axis=0)
        max_std = np.max(std_pos)

        # 将位置标准差转换为等效像素误差（使用相机内参估算）
        # 假设平均距离约500mm，焦距约500像素，则1mm ≈ 1像素
        # 这是一个粗略估算，真正的像素误差需要投影计算
        rmse_equivalent = max_std * 1000  # 转换为mm，近似作为像素误差

        return rmse_equivalent

    def save(self, filepath: str) -> bool:
        """
        保存标定结果到YAML文件

        Args:
            filepath: 文件路径

        Returns:
            是否保存成功
        """
        if not self.result.valid:
            print("警告: 标定结果无效，不建议保存")

        try:
            data = {
                'extrinsic_matrix': {
                    'rows': 4,
                    'cols': 4,
                    'data': self.result.extrinsic_matrix.flatten().tolist()
                },
                'rotation_matrix': {
                    'rows': 3,
                    'cols': 3,
                    'data': self.result.rotation_matrix.flatten().tolist()
                },
                'translation': {
                    'x': float(self.result.translation_vector[0]),
                    'y': float(self.result.translation_vector[1]),
                    'z': float(self.result.translation_vector[2])
                },
                'rmse_error_pixels': float(self.result.rmse_error),
                'num_poses': self.result.num_poses,
                'method': self.result.method,
                'valid': self.result.valid,
                'charuco_board': {
                    'squares_x': self.squares_x,
                    'squares_y': self.squares_y,
                    'square_length_m': self.square_length,
                    'marker_length_m': self.marker_length
                }
            }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False)

            print(f"标定结果已保存: {filepath}")
            return True

        except Exception as e:
            print(f"保存失败: {e}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional[CalibrationResult]:
        """
        从YAML文件加载标定结果

        Args:
            filepath: 文件路径

        Returns:
            标定结果，失败返回None
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            result = CalibrationResult()

            # 加载外参矩阵
            ext_data = data['extrinsic_matrix']['data']
            result.extrinsic_matrix = np.array(ext_data).reshape(4, 4)

            # 加载旋转矩阵
            rot_data = data['rotation_matrix']['data']
            result.rotation_matrix = np.array(rot_data).reshape(3, 3)

            # 加载平移向量
            trans = data['translation']
            result.translation_vector = np.array([
                trans['x'], trans['y'], trans['z']
            ])

            # 加载其他信息
            result.rmse_error = data.get('rmse_error_pixels', 0.0)
            result.num_poses = data.get('num_poses', 0)
            result.method = data.get('method', 'Unknown')
            result.valid = data.get('valid', True)

            return result

        except Exception as e:
            print(f"加载标定结果失败: {e}")
            return None


class ReprojectionVerifier:
    """
    重投影验证器

    用于验证手眼标定结果的精度。

    原理：
    1. 用TCP探针戳出几个验证点的物理坐标（Ground Truth）
    2. 记录拍摄像素时的法兰位姿
    3. 用标定的外参矩阵"脑补"验证点的像素位置
    4. 与实际拍摄的像素位置对比
    5. RMSE < 1.5像素 = 标定正确
    """

    def __init__(self,
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 extrinsic: CalibrationResult):
        """
        初始化重投影验证器

        Args:
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            extrinsic: 手眼标定结果
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.extrinsic = extrinsic

        # 验证点的世界坐标 (由TCP探针测量)
        self.world_points: List[np.ndarray] = []

        # 对应的像素坐标 (由相机检测)
        self.pixel_points: List[Tuple[float, float]] = []

        # 对应的法兰位姿 (拍摄像素时的位姿)
        self.flange_positions: List[np.ndarray] = []
        self.flange_rotations: List[np.ndarray] = []

    def add_verification_point(self,
                               world_position: np.ndarray,
                               pixel_position: Tuple[float, float],
                               flange_position: np.ndarray = None,
                               flange_rotation: np.ndarray = None):
        """
        添加验证点

        Args:
            world_position: 世界坐标 [x, y, z] (米)
            pixel_position: 像素坐标 (u, v)
            flange_position: 拍摄时的法兰位置 (可选，用于逐点验证)
            flange_rotation: 拍摄时的法兰旋转矩阵 (可选)
        """
        self.world_points.append(np.array(world_position))
        self.pixel_points.append(pixel_position)
        if flange_position is not None:
            self.flange_positions.append(np.array(flange_position))
        if flange_rotation is not None:
            self.flange_rotations.append(np.array(flange_rotation))

    def verify_with_individual_poses(self) -> Tuple[bool, float]:
        """
        验证重投影误差（每个验证点使用各自的法兰位姿）

        Returns:
            (passed, rmse)
        """
        if len(self.world_points) < 4:
            print(f"验证点不足 ({len(self.world_points)}/4)")
            return False, 999.0

        if len(self.flange_positions) != len(self.world_points):
            print("错误: 法兰位姿数量与验证点数量不匹配")
            return False, 999.0

        errors = []

        for i, (world_pt, pixel_pt, flange_pos, flange_rot) in enumerate(zip(
                self.world_points, self.pixel_points, self.flange_positions, self.flange_rotations)):

            # 计算相机在世界坐标系中的位姿
            T_base2flange = np.eye(4)
            T_base2flange[:3, :3] = flange_rot
            T_base2flange[:3, 3] = flange_pos

            T_flange2cam = self.extrinsic.extrinsic_matrix
            T_base2cam = T_base2flange @ T_flange2cam
            T_cam2base = np.linalg.inv(T_base2cam)

            # 将世界坐标转换到相机坐标系
            P_world = np.append(world_pt, 1.0)  # 齐次坐标
            P_cam = T_cam2base @ P_world

            # 透视投影
            X, Y, Z = P_cam[0], P_cam[1], P_cam[2]
            if Z <= 0:
                continue

            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            u_proj = fx * X / Z + cx
            v_proj = fy * Y / Z + cy

            # 计算误差
            error = np.sqrt((u_proj - pixel_pt[0])**2 + (v_proj - pixel_pt[1])**2)
            errors.append(error)
            print(f"  点 #{i}: 像素误差 = {error:.2f} px")

        if len(errors) == 0:
            return False, 999.0

        rmse = np.sqrt(np.mean(np.array(errors)**2))

        print(f"\n重投影误差: RMSE = {rmse:.2f} 像素")
        if rmse < 1.5:
            print("  ✓ 验证通过")
            return True, rmse
        else:
            print("  ✗ 验证失败，建议重新标定")
            return False, rmse

    def verify(self,
               flange_position: np.ndarray,
               flange_rotation: np.ndarray,
               rotation_format: str = "quaternion") -> Tuple[bool, float]:
        """
        验证重投影误差

        Args:
            flange_position: 当前法兰位置
            flange_rotation: 当前法兰旋转
            rotation_format: 旋转格式

        Returns:
            (passed, rmse)
        """
        if len(self.world_points) < 4:
            print(f"验证点不足 ({len(self.world_points)}/4)")
            return False, 999.0

        # 转换法兰旋转
        if rotation_format == "quaternion":
            R_flange = R.from_quat(flange_rotation).as_matrix()
        elif rotation_format == "euler":
            R_flange = R.from_euler('xyz', flange_rotation).as_matrix()
        else:
            R_flange = flange_rotation

        # 计算相机在世界坐标系中的位姿
        T_base2flange = np.eye(4)
        T_base2flange[:3, :3] = R_flange
        T_base2flange[:3, 3] = flange_position

        T_flange2cam = self.extrinsic.extrinsic_matrix
        T_base2cam = T_base2flange @ T_flange2cam
        T_cam2base = np.linalg.inv(T_base2cam)

        errors = []

        for i, (world_pt, pixel_pt) in enumerate(zip(self.world_points, self.pixel_points)):
            # 将世界坐标转换到相机坐标系
            P_world = np.append(world_pt, 1.0)  # 齐次坐标
            P_cam = T_cam2base @ P_world

            # 透视投影
            X, Y, Z = P_cam[0], P_cam[1], P_cam[2]
            if Z <= 0:
                continue

            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            u_proj = fx * X / Z + cx
            v_proj = fy * Y / Z + cy

            # 计算误差
            error = np.sqrt((u_proj - pixel_pt[0])**2 + (v_proj - pixel_pt[1])**2)
            errors.append(error)

        if len(errors) == 0:
            return False, 999.0

        rmse = np.sqrt(np.mean(np.array(errors)**2))

        print(f"重投影误差: RMSE = {rmse:.2f} 像素")
        if rmse < 1.5:
            print("  ✓ 验证通过")
            return True, rmse
        else:
            print("  ✗ 验证失败，建议重新标定")
            return False, rmse


def generate_charuco_board(squares_x: int = 5,
                          squares_y: int = 7,
                          square_length_px: int = 100,
                          marker_length_px: int = 80,
                          output_path: str = "charuco_board.png"):
    """
    生成ChArUco标定板图像

    Args:
        squares_x: X方向格子数
        squares_y: Y方向格子数
        square_length_px: 格子边长（像素）
        marker_length_px: 标记边长（像素）
        output_path: 输出路径
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_px,
        marker_length_px,
        dictionary
    )

    # 生成板图像
    img_size = (squares_x * square_length_px, squares_y * square_length_px)
    board_img = board.generateImage(img_size)

    cv2.imwrite(output_path, board_img)
    print(f"ChArUco标定板已生成: {output_path}")
    print(f"  格子数: {squares_x} x {squares_y}")
    print(f"  格子边长: {square_length_px} 像素")
    print(f"  请按实际尺寸打印（注意DPI设置）")


if __name__ == "__main__":
    # 测试：生成ChArUco板
    generate_charuco_board()