#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单IBVS控制器 (Simple IBVS Controller)

纯粹基于关节灵敏度的图像视觉伺服，不需要手眼标定(extrinsic matrix)或正运动学(FK)。

核心原理:
  AprilTag检测 → tag像素+深度误差 → 灵敏度雅可比(3×N) → 伪逆求解 → 关节调整量 → 机器人移动
  当 tag 居中且相机距离达标(tag尺寸)时，判定为对齐完成。

对比:
  - PBVS: 需要 extrinsic matrix + FK + coordinate transform → 误差来源多，标定困难
  - SimpleIBVS: 只需要 joint sensitivity + AprilTag → 简单直接，亚像素精度

对齐方式:
  保持AprilTag在相机视野中心，通过tag物理尺寸估算深度。
  当 tag 居中(误差<容差) 且 深度≈目标深度 时 → 对齐完成。
  排除了"tag移出视野=误判遮挡"的漏洞。

优势:
  - AprilTag亚像素检测精度(~0.1-0.5px)
  - 不依赖手眼标定，不需要FK
  - 正逻辑判定：居中+距离达标，无歧义
  - 深度通过tag尺寸+相机焦距直接计算，几何确定
  - tag自带ID和旋转信息
  - 支持3D雅可比：同时优化XY和深度
"""

import numpy as np
import cv2
import json
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from precision_place.models.calibration_data import ARM_CONFIGS, JointSensitivity, CalibrationPoint


# ==================== 关节位置贡献权重 ====================
POSITION_WEIGHTS_RIGHT = {
    7: 1.0,   # joint_1 (底座旋转)
    8: 1.0,   # joint_2 (肩部俯仰)
    9: 1.0,   # joint_3 (肘部)
    10: 0.8,  # joint_4 (前臂)
    11: 0.3,  # joint_5 (腕部俯仰)
    12: 0.3,  # joint_6 (腕部旋转)
    14: 0.6,  # trunk_1
}

POSITION_WEIGHTS_LEFT = {
    0: 1.0,   # joint_1 (底座旋转)
    1: 1.0,   # joint_2 (肩部俯仰)
    2: 1.0,   # joint_3 (肘部)
    3: 0.8,   # joint_4 (前臂)
    4: 0.3,   # joint_5 (腕部俯仰)
    5: 0.3,   # joint_6 (腕部旋转)
    14: 0.6,  # trunk_1
}


# ==================== AprilTag 检测器 ====================
class AprilTagDetector:
    """
    AprilTag检测器 (基于OpenCV内置的AprilTag_36h11字典)

    使用 OpenCV 4.7+ 的 ArUcoDetector 接口检测 AprilTag。
    提供:
      - 亚像素角点检测
      - tag ID识别
      - tag旋转角度
      - tag中心坐标
      - tag尺寸 (可用于深度估算)
    """

    # 常用tag尺寸映射 (tag_id → 物理尺寸mm)
    # 用户可以根据实际打印尺寸修改此表
    TAG_SIZE_MAP = {}

    def __init__(self, tag_family: str = "36h11",
                 target_tag_ids: List[int] = None,
                 tag_size_mm: float = 20.0):
        """
        Args:
            tag_family: AprilTag家族 ("36h11", "25h9", "16h5")
            target_tag_ids: 需要检测的tag ID列表，None表示检测所有
            tag_size_mm: tag默认物理尺寸(mm)，用于深度估算
        """
        self.tag_family = tag_family
        self.target_tag_ids = target_tag_ids
        self.tag_size_mm = tag_size_mm

        # 创建字典和检测器
        family_dict_map = {
            "36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "16h5": cv2.aruco.DICT_APRILTAG_16h5,
        }
        dict_id = family_dict_map.get(tag_family, cv2.aruco.DICT_APRILTAG_36h11)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        # 检测参数: 提高亚像素精度
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 3
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01
        # 降低误检率
        params.errorCorrectionRate = 0.6

        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的AprilTag

        Returns:
            List of tag info dicts:
              {
                'id': tag ID,
                'center': (cx, cy) 亚像素中心,
                'corners': 4个角点 [(x,y), ...],
                'rotation_deg': tag旋转角度(度),
                'size_px': tag在图像中的尺寸(px),
                'size_mm': tag物理尺寸(mm),
              }
        """
        corners, ids, rejected = self.detector.detectMarkers(image)

        if ids is None or len(ids) == 0:
            return []

        tags = []
        for i, tag_id in enumerate(ids.flatten()):
            # 过滤目标tag
            if self.target_tag_ids is not None and tag_id not in self.target_tag_ids:
                continue

            c = corners[i].reshape(4, 2)  # 4角点, 每个(x,y)

            # 亚像素中心: 4角点平均值
            cx = float(np.mean(c[:, 0]))
            cy = float(np.mean(c[:, 1]))

            # 旋转角度: 从角点0→角点1的方向
            dx = c[1][0] - c[0][0]
            dy = c[1][1] - c[0][1]
            rotation_deg = float(np.degrees(np.arctan2(dy, dx)))

            # tag像素尺寸: 对角线长度
            side1 = np.linalg.norm(c[1] - c[0])
            side2 = np.linalg.norm(c[3] - c[0])
            size_px = float((side1 + side2) / 2)

            # 物理尺寸
            size_mm = self.TAG_SIZE_MAP.get(int(tag_id), self.tag_size_mm)

            tags.append({
                'id': int(tag_id),
                'center': (cx, cy),
                'corners': [(float(p[0]), float(p[1])) for p in c],
                'rotation_deg': rotation_deg,
                'size_px': size_px,
                'size_mm': size_mm,
            })

        return tags

    def estimate_depth_mm(self, tag: Dict, camera_fx: float) -> float:
        """从tag像素尺寸估算深度(mm)"""
        if tag['size_px'] <= 0 or tag['size_mm'] <= 0:
            return 0.0
        return tag['size_mm'] * camera_fx / tag['size_px']

    def draw_tags(self, image: np.ndarray, tags: List[Dict],
                  occluded: bool = False) -> np.ndarray:
        """在图像上绘制检测到的tag"""
        display = image.copy()

        for tag in tags:
            cx, cy = tag['center']
            corners = tag['corners']
            tag_id = tag['id']

            # 绘制tag边界
            pts = np.array(corners, dtype=np.int32)
            color = (0, 255, 0) if not occluded else (0, 0, 255)
            cv2.polylines(display, [pts], True, color, 2)

            # 绘制中心点
            cv2.circle(display, (int(cx), int(cy)), 4, (255, 0, 255), -1)

            # 绘制ID
            cv2.putText(display, f"ID:{tag_id}", (int(cx)-20, int(cy)-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制旋转方向 (角点0→1)
            c0, c1 = corners[0], corners[1]
            cv2.arrowedLine(display, (int(c0[0]), int(c0[1])),
                           (int(c1[0]), int(c1[1])), (0, 200, 255), 2)

        return display


@dataclass
class IBVSResult:
    """IBVS对齐结果"""
    converged: bool = False
    iterations: int = 0
    final_error_px: float = 0.0
    final_error_mm: float = 0.0
    history: List[Dict] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class TagAlignmentState:
    """Tag对齐状态"""
    tag_visible: bool = False
    tag_center: Tuple[float, float] = (0.0, 0.0)
    tag_id: int = -1
    tag_rotation_deg: float = 0.0
    tag_size_px: float = 0.0
    # 误差
    error_x: float = 0.0
    error_y: float = 0.0
    error_rotation: float = 0.0
    error_total_px: float = 0.0
    error_total_mm: float = 0.0
    # 深度 (通过tag尺寸估算)
    depth_mm: float = 0.0
    depth_error_mm: float = 0.0
    depth_filtered: float = 0.0
    # 对齐判定
    xy_centered: bool = False           # tag居中
    depth_reached: bool = False         # 深度达标
    aligned: bool = False               # 完全对齐 (xy_centered AND depth_reached)


class SimpleIBVSController:
    """
    简单IBVS控制器 (AprilTag + 灵敏度雅可比)

    对齐方式: 保持tag在相机中心，通过tag尺寸估算深度。
    控制律: [像素误差, 深度误差] → 灵敏度雅可比(3×N) → 伪逆 → 关节调整量。
    对齐判定: tag 居中 (像素误差<容差) AND 深度≈目标深度。
    """

    def __init__(self, arm: str = "right", gain: float = 0.6,
                 pixel_tolerance: float = 3.0, max_iterations: int = 25,
                 max_single_adjust: float = 2.0,
                 tag_family: str = "36h11",
                 target_tag_ids: List[int] = None,
                 tag_size_mm: float = 20.0,
                 target_depth_mm: float = 150.0,
                 depth_tolerance_mm: float = 5.0,
                 depth_weight: float = 0.1,
                 depth_filter_window: int = 5,
                 camera_fx: float = 531.0):
        """
        Args:
            arm: 'left' or 'right'
            gain: 控制增益 (0~1)
            pixel_tolerance: 像素收敛容差 (px)
            max_iterations: 最大迭代次数
            max_single_adjust: 单次最大关节调整量 (度)
            tag_family: AprilTag家族
            target_tag_ids: 要跟踪的tag ID列表
            tag_size_mm: tag物理尺寸 (mm)
            target_depth_mm: 目标深度距离 (相机到tag的期望距离, mm)
            depth_tolerance_mm: 深度收敛容差 (mm)
            depth_weight: 深度误差在雅可比中的权重 (平衡px和mm维度)
            depth_filter_window: 深度滤波窗口大小 (帧数)
            camera_fx: 相机焦距 (像素)
        """
        self.arm = arm
        self.arm_config = ARM_CONFIGS[arm]
        self.gain = gain
        self.pixel_tolerance = pixel_tolerance
        self.max_iterations = max_iterations
        self.max_single_adjust = max_single_adjust
        self.camera_fx = camera_fx

        # AprilTag 检测器
        self.tag_detector = AprilTagDetector(
            tag_family=tag_family,
            target_tag_ids=target_tag_ids,
            tag_size_mm=tag_size_mm,
        )

        # 标定数据
        self.calibration_points: List[CalibrationPoint] = []
        self.pixel_to_mm_ratio: float = 0.5

        # 相机翻转标志
        self._camera_flip_x = False
        self._camera_flip_y = False

        # 位置权重
        if arm == 'right':
            self.position_weights = POSITION_WEIGHTS_RIGHT
            self.rotation_joint = 12
        else:
            self.position_weights = POSITION_WEIGHTS_LEFT
            self.rotation_joint = 5

        # 目标像素位置 (tag中心应该出现的位置)
        self.target_pixel_x = 320.0
        self.target_pixel_y = 240.0

        # 深度相关参数
        self.target_depth_mm = target_depth_mm
        self.depth_tolerance_mm = depth_tolerance_mm
        self.depth_weight = depth_weight
        self._depth_history = deque(maxlen=depth_filter_window)

        # 关节名称映射
        self.joint_names = {}
        for i in range(7):
            self.joint_names[i] = f"left_arm_joint_{i+1}"
        for i in range(7, 14):
            self.joint_names[i] = f"right_arm_joint_{i-6}"
        self.joint_names[14] = "trunk_joint_1"
        self.joint_names[15] = "trunk_joint_2"

    def load_calibration(self, filepath: str = None) -> bool:
        """加载标定数据"""
        if filepath is None:
            filepath = str(Path(__file__).parent.parent / "calibration_points.json")

        path = Path(filepath)
        if not path.exists():
            print(f"✗ 标定数据文件不存在: {filepath}")
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.calibration_points = []
            for cp_data in data.get('points', []):
                if cp_data.get('arm', 'right') != self.arm:
                    continue
                sensitivities = [JointSensitivity(**s) for s in cp_data.get('sensitivities', [])]
                cp = CalibrationPoint(
                    height_level=cp_data['height_level'],
                    joint_states=cp_data['joint_states'],
                    sensitivities=sensitivities,
                    pixel_to_mm=cp_data.get('pixel_to_mm', 0.5),
                    timestamp=cp_data.get('timestamp', ''),
                    arm=cp_data.get('arm', 'right'),
                    camera_name=cp_data.get('camera_name', '')
                )
                self.calibration_points.append(cp)

            ratio_data = data.get('pixel_to_mm_ratio', None)
            if ratio_data:
                self.pixel_to_mm_ratio = ratio_data
            elif self.calibration_points:
                self.pixel_to_mm_ratio = self.calibration_points[0].pixel_to_mm

            self._check_camera_flip()
            print(f"✓ 已加载 {len(self.calibration_points)} 个标定点 (arm={self.arm})")
            print(f"  pixel_to_mm_ratio = {self.pixel_to_mm_ratio:.3f}")
            return True
        except Exception as e:
            print(f"✗ 加载标定数据失败: {e}")
            return False

    def save_calibration(self, filepath: str = None) -> bool:
        """保存标定数据 (含深度灵敏度 depth_dz_per_deg)"""
        if filepath is None:
            filepath = str(Path(__file__).parent.parent / "calibration_points.json")

        path = Path(filepath)
        try:
            data = {
                'version': '2.0',
                'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
                'points': []
            }

            for cp in self.calibration_points:
                cp_data = {
                    'height_level': cp.height_level,
                    'joint_states': cp.joint_states,
                    'sensitivities': [
                        {
                            'joint_idx': s.joint_idx,
                            'joint_name': s.joint_name,
                            'pixel_dx_per_deg': s.pixel_dx_per_deg,
                            'pixel_dy_per_deg': s.pixel_dy_per_deg,
                            'depth_dz_per_deg': getattr(s, 'depth_dz_per_deg', 0.0),
                            'mm_dx_per_deg': s.mm_dx_per_deg,
                            'mm_dy_per_deg': s.mm_dy_per_deg,
                            'calibration_angles': s.calibration_angles,
                        }
                        for s in cp.sensitivities
                    ],
                    'timestamp': cp.timestamp,
                    'arm': cp.arm,
                    'camera_name': cp.camera_name,
                }
                data['points'].append(cp_data)

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            has_depth = any(
                getattr(s, 'depth_dz_per_deg', 0.0) != 0.0
                for cp in self.calibration_points
                for s in cp.sensitivities
            )
            print(f"✓ 已保存 {len(self.calibration_points)} 个标定点 → {filepath}"
                  + (" (含3D深度灵敏度)" if has_depth else " (仅2D)"))
            return True
        except Exception as e:
            print(f"✗ 保存标定数据失败: {e}")
            return False

    def _check_camera_flip(self):
        """检查相机方向翻转"""
        flip_config = self.arm_config.camera_flip
        cam_name = self.arm_config.camera_name
        if cam_name in flip_config:
            self._camera_flip_x, self._camera_flip_y = flip_config[cam_name]

    def detect_and_compute_error(self, image: np.ndarray) -> TagAlignmentState:
        """
        检测AprilTag并计算对齐误差

        误差 = tag中心 - 目标像素位置
        深度 = tag物理尺寸 × fx / tag像素尺寸
        深度误差 = 深度 - 目标深度

        对齐判定: tag居中(像素误差<容差) AND 深度≈目标深度
        """
        state = TagAlignmentState()

        tags = self.tag_detector.detect(image)

        if not tags:
            state.tag_visible = False
            return state

        # tag可见
        state.tag_visible = True

        # 取离目标位置最近的tag
        best_tag = tags[0]
        if len(tags) > 1:
            best_dist = float('inf')
            for tag in tags:
                tx, ty = tag['center']
                dist = np.sqrt((tx - self.target_pixel_x)**2 + (ty - self.target_pixel_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_tag = tag

        state.tag_center = best_tag['center']
        state.tag_id = best_tag['id']
        state.tag_rotation_deg = best_tag['rotation_deg']
        state.tag_size_px = best_tag['size_px']

        # 计算像素误差
        tx, ty = best_tag['center']
        error_x = tx - self.target_pixel_x
        error_y = ty - self.target_pixel_y

        if self._camera_flip_x:
            error_x = -error_x
        if self._camera_flip_y:
            error_y = -error_y

        state.error_x = error_x
        state.error_y = error_y
        state.error_total_px = np.sqrt(error_x**2 + error_y**2)
        mm_x = error_x * self.pixel_to_mm_ratio
        mm_y = error_y * self.pixel_to_mm_ratio
        state.error_total_mm = np.sqrt(mm_x**2 + mm_y**2)

        # 旋转误差
        state.error_rotation = best_tag['rotation_deg']

        # 深度估算 + 滑动窗口滤波
        raw_depth = self.tag_detector.estimate_depth_mm(best_tag, self.camera_fx)
        self._depth_history.append(raw_depth)
        filtered_depth = float(np.mean(self._depth_history)) if self._depth_history else raw_depth

        state.depth_mm = raw_depth
        state.depth_filtered = filtered_depth
        state.depth_error_mm = filtered_depth - self.target_depth_mm

        # 对齐判定
        state.xy_centered = state.error_total_px < self.pixel_tolerance
        state.depth_reached = abs(state.depth_error_mm) < self.depth_tolerance_mm
        state.aligned = state.xy_centered and state.depth_reached

        return state

    def get_interpolated_sensitivities(self, current_joints: np.ndarray) -> List[JointSensitivity]:
        """获取当前姿态的插值灵敏度（反距离加权，含深度维度）"""
        if not self.calibration_points:
            return []

        primary_joints = self.arm_config.primary_joints

        distances = []
        for cp in self.calibration_points:
            cp_joints = np.array(cp.joint_states)
            dist = np.linalg.norm(
                np.array([cp_joints[i] for i in primary_joints]) -
                np.array([current_joints[i] for i in primary_joints])
            )
            distances.append(dist)

        epsilon = 0.001
        weights = [1.0 / (d + epsilon) for d in distances]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        joint_indices = set()
        for cp in self.calibration_points:
            for s in cp.sensitivities:
                joint_indices.add(s.joint_idx)
        joint_indices = sorted(joint_indices)

        interpolated = []
        for jidx in joint_indices:
            dx_values, dy_values, dz_values, dx_weights, dy_weights, dz_weights = [], [], [], [], [], []
            for cp, w in zip(self.calibration_points, weights):
                for s in cp.sensitivities:
                    if s.joint_idx == jidx:
                        dx_values.append(s.pixel_dx_per_deg)
                        dy_values.append(s.pixel_dy_per_deg)
                        dz_values.append(getattr(s, 'depth_dz_per_deg', 0.0))
                        dx_weights.append(w)
                        dy_weights.append(w)
                        dz_weights.append(w)
                        break

            if dx_values:
                interp_dx = sum(v * w for v, w in zip(dx_values, dx_weights))
                interp_dy = sum(v * w for v, w in zip(dy_values, dy_weights))
                interp_dz = sum(v * w for v, w in zip(dz_values, dz_weights))

                if self._camera_flip_x:
                    interp_dx = -interp_dx
                if self._camera_flip_y:
                    interp_dy = -interp_dy

                interpolated.append(JointSensitivity(
                    joint_idx=jidx,
                    joint_name=self.joint_names.get(jidx, f"joint_{jidx}"),
                    pixel_dx_per_deg=interp_dx,
                    pixel_dy_per_deg=interp_dy,
                    depth_dz_per_deg=interp_dz,
                    mm_dx_per_deg=interp_dx * self.pixel_to_mm_ratio,
                    mm_dy_per_deg=interp_dy * self.pixel_to_mm_ratio,
                    calibration_angles=current_joints.tolist()
                ))

        return interpolated

    def compute_joint_adjustments(self, pixel_error_x: float, pixel_error_y: float,
                                  current_joints: np.ndarray,
                                  depth_error_mm: float = 0.0) -> Dict[int, float]:
        """
        核心IBVS控制律: [像素误差, 深度误差] → 关节调整量

        使用 3×N 或 2×N 雅可比 (取决于是否有深度灵敏度数据):
          J·W·Δθ = -e  →  Δθ = (J·W)^+ · (-e)

        e = [pixel_err_x, pixel_err_y, depth_err_mm * depth_weight]

        自动检测标定数据中是否有深度灵敏度，没有则回退到2×N模式。
        2×N模式仍可进行XY伺服，深度仅用于判定(不做深度伺服)。
        """
        sensitivities = self.get_interpolated_sensitivities(current_joints)
        if not sensitivities:
            print("⚠ 无标定数据，无法计算调整量")
            return {}

        joint_indices = [s.joint_idx for s in sensitivities]
        n_joints = len(joint_indices)

        # 检测是否有深度灵敏度数据
        has_depth = any(
            getattr(s, 'depth_dz_per_deg', 0.0) != 0.0 for s in sensitivities
        )

        if has_depth and abs(depth_error_mm) > 0.01:
            # 3×N 雅可比
            n_rows = 3
            J = np.zeros((n_rows, n_joints))
            for i, s in enumerate(sensitivities):
                J[0, i] = s.pixel_dx_per_deg
                J[1, i] = s.pixel_dy_per_deg
                J[2, i] = getattr(s, 'depth_dz_per_deg', 0.0)
            error = np.array([pixel_error_x, pixel_error_y, depth_error_mm * self.depth_weight])
            jacobian_type = "3×N"
        else:
            # 2×N 雅可比 (回退模式: 无深度灵敏度或深度误差很小)
            n_rows = 2
            J = np.zeros((n_rows, n_joints))
            for i, s in enumerate(sensitivities):
                J[0, i] = s.pixel_dx_per_deg
                J[1, i] = s.pixel_dy_per_deg
            error = np.array([pixel_error_x, pixel_error_y])
            jacobian_type = "2×N"

        W = np.zeros((n_joints, n_joints))
        for i, s in enumerate(sensitivities):
            W[i, i] = self.position_weights.get(s.joint_idx, 0.5)

        JW = J @ W

        try:
            JW_pinv = np.linalg.pinv(JW)
            delta_angles = JW_pinv @ (-error)
        except np.linalg.LinAlgError:
            delta_angles = self._single_joint_fallback(pixel_error_x, pixel_error_y,
                                                       sensitivities, depth_error_mm, has_depth)

        max_delta = np.max(np.abs(delta_angles))
        if max_delta > 5.0:
            delta_angles = self._single_joint_fallback(pixel_error_x, pixel_error_y,
                                                       sensitivities, depth_error_mm, has_depth)

        delta_angles = delta_angles * self.gain
        delta_angles = np.clip(delta_angles, -self.max_single_adjust, self.max_single_adjust)

        adjustments = {}
        for i, (jidx, delta) in enumerate(zip(joint_indices, delta_angles)):
            if abs(delta) > 0.01:
                adjustments[jidx] = float(delta)

        if has_depth and abs(depth_error_mm) > 0.01:
            depth_parts = [f"{jidx}:{delta:.3f}" for jidx, delta in adjustments.items()
                          if abs(getattr(
                              sensitivities[joint_indices.index(jidx)], 'depth_dz_per_deg', 0.0)
                          ) > 0.01]
            if depth_parts:
                print(f"  [{jacobian_type}] depth_err={depth_error_mm:.1f}mm, "
                      f"Z joints: {', '.join(depth_parts[:3])}")

        return adjustments

    def compute_rotation_adjustment(self, rotation_error: float) -> Dict[int, float]:
        """计算旋转误差的关节调整量"""
        if abs(rotation_error) < 2.0:
            return {}
        delta = -rotation_error * self.gain * 0.5
        delta = max(-self.max_single_adjust, min(self.max_single_adjust, delta))
        return {self.rotation_joint: delta}

    def _single_joint_fallback(self, pixel_error_x: float, pixel_error_y: float,
                               sensitivities: List[JointSensitivity],
                               depth_error_mm: float = 0.0,
                               has_depth: bool = False) -> np.ndarray:
        """单关节fallback: 选灵敏度最高的关节分别控制X/Y/Z"""
        n = len(sensitivities)
        delta = np.zeros(n)

        best_x = max(range(n), key=lambda i: abs(sensitivities[i].pixel_dx_per_deg))
        if abs(sensitivities[best_x].pixel_dx_per_deg) > 0.01:
            delta[best_x] = -pixel_error_x / sensitivities[best_x].pixel_dx_per_deg

        best_y = max(range(n), key=lambda i: abs(sensitivities[i].pixel_dy_per_deg))
        if abs(sensitivities[best_y].pixel_dy_per_deg) > 0.01:
            delta[best_y] += -pixel_error_y / sensitivities[best_y].pixel_dy_per_deg

        if has_depth and abs(depth_error_mm) > 0.01:
            best_z = max(range(n),
                        key=lambda i: abs(getattr(sensitivities[i], 'depth_dz_per_deg', 0.0)))
            dz = getattr(sensitivities[best_z], 'depth_dz_per_deg', 0.0)
            if abs(dz) > 0.001:
                delta[best_z] += -depth_error_mm * self.depth_weight / dz

        return delta

    def compute_error_mm(self, pixel_error_x: float, pixel_error_y: float) -> Tuple[float, float]:
        """像素误差 → 毫米误差"""
        return pixel_error_x * self.pixel_to_mm_ratio, pixel_error_y * self.pixel_to_mm_ratio

    def set_target_pixel(self, x: float, y: float):
        """设置目标像素位置 (tag中心应该出现的位置)"""
        self.target_pixel_x = x
        self.target_pixel_y = y
        print(f"✓ 目标像素位置: ({x:.1f}, {y:.1f})")

    def set_target_depth(self, depth_mm: float):
        """设置目标深度距离 (相机到tag的期望距离, mm)"""
        self.target_depth_mm = depth_mm
        print(f"✓ 目标深度: {depth_mm:.0f}mm")

    def set_target_from_state(self, tag_state: TagAlignmentState):
        """从当前tag状态设置目标位置和目标深度"""
        if tag_state.tag_visible:
            cx, cy = tag_state.tag_center
            self.set_target_pixel(cx, cy)
            if tag_state.depth_filtered > 0:
                self.set_target_depth(tag_state.depth_filtered)
            print(f"✓ 目标已捕获: 像素({cx:.1f}, {cy:.1f}), 深度{tag_state.depth_filtered:.0f}mm")
        else:
            print("⚠ 需要检测到tag才能设置目标")

    # ==================== AprilTag 标定 (3D灵敏度) ====================

    def calibrate_joint_with_apriltag(self, camera, get_joints_fn,
                                       move_joint_fn, joint_idx: int,
                                       move_degrees: float = 6.0,
                                       settle_time: float = 1.0) -> Tuple[bool, Optional[JointSensitivity]]:
        """
        使用 AprilTag 标定单个关节的3D灵敏度 (含深度维度)

        Args:
            camera: 相机对象 (有 read() 方法)
            get_joints_fn: () → np.ndarray, 获取当前关节状态
            move_joint_fn: (joint_idx, target_angle) → None, 移动关节并等待稳定
            joint_idx: 关节索引
            move_degrees: 移动角度 (建议6°以获得更稳定的深度变化)
            settle_time: 移动后稳定等待时间(秒)

        Returns:
            (success, JointSensitivity with depth_dz_per_deg)
        """
        joint_name = self.joint_names.get(joint_idx, f"joint_{joint_idx}")
        print(f"\n{'='*60}")
        print(f"AprilTag关节灵敏度标定: {joint_name} (3D)")
        print(f"{'='*60}")

        # 获取初始关节状态
        joints = get_joints_fn()
        if joints is None or joint_idx >= len(joints):
            print(f"✗ 无法获取关节状态")
            return False, None

        initial_angle = float(joints[joint_idx])
        target_angle = initial_angle + move_degrees

        print(f"  初始角度: {initial_angle:.2f}°")
        print(f"  目标角度: {target_angle:.2f}°")

        # Phase 1: 采集初始tag
        print("  [Phase 1] 采集初始tag位置...")
        tag_before = None
        for attempt in range(10):
            img = camera.read()
            if img is None:
                time.sleep(0.1)
                continue
            tags = self.tag_detector.detect(img)
            if tags:
                tag_before = tags[0]
                break
            time.sleep(0.1)

        if tag_before is None:
            print("✗ 未检测到AprilTag，请确认tag在相机视野内")
            return False, None

        cx_before, cy_before = tag_before['center']
        size_before = tag_before['size_px']
        depth_before = self.tag_detector.estimate_depth_mm(tag_before, self.camera_fx)
        print(f"    中心: ({cx_before:.1f}, {cy_before:.1f}), "
              f"尺寸: {size_before:.1f}px, 深度: {depth_before:.0f}mm")

        # Phase 2: 移动关节
        print(f"  [Phase 2] 移动关节 {joint_name} {move_degrees}°...")
        move_joint_fn(joint_idx, target_angle)
        time.sleep(settle_time)

        # 确认实际移动
        joints_after = get_joints_fn()
        if joints_after is None or joint_idx >= len(joints_after):
            print("✗ 无法获取移动后关节状态")
            return False, None
        actual_angle = float(joints_after[joint_idx])
        actual_move = actual_angle - initial_angle
        print(f"    实际移动: {actual_move:.2f}°")

        if abs(actual_move) < 0.1:
            print("⚠ 警告: 移动角度很小，标定可能不准确")

        # Phase 3: 采集移动后tag
        print("  [Phase 3] 采集移动后tag位置...")
        tag_after = None
        for attempt in range(10):
            img = camera.read()
            if img is None:
                time.sleep(0.1)
                continue
            tags = self.tag_detector.detect(img)
            if tags:
                tag_after = tags[0]
                break
            time.sleep(0.1)

        if tag_after is None:
            print("✗ 移动后未检测到AprilTag")
            return False, None

        cx_after, cy_after = tag_after['center']
        size_after = tag_after['size_px']
        depth_after = self.tag_detector.estimate_depth_mm(tag_after, self.camera_fx)
        print(f"    中心: ({cx_after:.1f}, {cy_after:.1f}), "
              f"尺寸: {size_after:.1f}px, 深度: {depth_after:.0f}mm")

        # 计算灵敏度
        actual_deg = abs(actual_move) if abs(actual_move) > 0.1 else move_degrees

        pixel_dx = (cx_after - cx_before) / actual_deg
        pixel_dy = (cy_after - cy_before) / actual_deg
        depth_dz = (depth_after - depth_before) / actual_deg

        # 相机翻转
        if self._camera_flip_x:
            pixel_dx = -pixel_dx
        if self._camera_flip_y:
            pixel_dy = -pixel_dy

        sensitivity = JointSensitivity(
            joint_idx=joint_idx,
            joint_name=joint_name,
            pixel_dx_per_deg=pixel_dx,
            pixel_dy_per_deg=pixel_dy,
            depth_dz_per_deg=depth_dz,
            mm_dx_per_deg=pixel_dx * self.pixel_to_mm_ratio,
            mm_dy_per_deg=pixel_dy * self.pixel_to_mm_ratio,
            calibration_angles=joints_after.tolist() if joints_after is not None else [],
        )

        print(f"\n  标定结果 (3D):")
        print(f"    像素灵敏度: X={pixel_dx:.2f} px/deg, Y={pixel_dy:.2f} px/deg")
        print(f"    深度灵敏度: Z={depth_dz:.1f} mm/deg")
        print(f"    初始深度: {depth_before:.0f}mm → 最终深度: {depth_after:.0f}mm "
              f"(Δ={depth_after-depth_before:.0f}mm)")

        return True, sensitivity

    def calibrate_all_joints_apriltag(self, camera, get_joints_fn, move_joint_fn,
                                       return_after: bool = True) -> List[JointSensitivity]:
        """
        使用 AprilTag 标定所有主要关节的3D灵敏度

        Args:
            camera: 相机对象
            get_joints_fn: 获取关节状态
            move_joint_fn: 移动关节
            return_after: 标定后是否返回初始位置

        Returns:
            List of JointSensitivity (含 depth_dz_per_deg)
        """
        primary_joints = self.arm_config.primary_joints

        # 保存初始位置
        initial_joints = get_joints_fn()

        try:
            move_deg = float(input(f"移动角度 (建议6°, 默认6.0): ").strip() or "6.0")
        except (ValueError, EOFError):
            move_deg = 6.0

        try:
            settle_time = float(input("移动后稳定时间秒 (默认1.5): ").strip() or "1.5")
        except (ValueError, EOFError):
            settle_time = 1.5

        print(f"\n将标定 {len(primary_joints)} 个关节:")
        for i, jidx in enumerate(primary_joints):
            joint_name = self.joint_names.get(jidx, f"joint_{jidx}")
            print(f"  {i+1}. [{jidx}] {joint_name}")

        sensitivities = []
        for i, jidx in enumerate(primary_joints):
            joint_name = self.joint_names.get(jidx, f"joint_{jidx}")
            print(f"\n[{i+1}/{len(primary_joints)}] 标定: {joint_name}")

            success, sens = self.calibrate_joint_with_apriltag(
                camera, get_joints_fn, move_joint_fn, jidx, move_deg, settle_time
            )
            if success and sens is not None:
                sensitivities.append(sens)

        # 返回初始位置
        if return_after and initial_joints is not None:
            print("\n返回初始位置...")
            # 逐个关节移动回初始位置
            for jidx in primary_joints:
                if jidx < len(initial_joints):
                    try:
                        move_joint_fn(jidx, float(initial_joints[jidx]))
                    except Exception:
                        pass
            time.sleep(settle_time)

        # 保存为一个 CalibrationPoint
        current_joints = get_joints_fn()
        if current_joints is not None and sensitivities:
            cal_point = CalibrationPoint(
                height_level="apriltag_3d",
                joint_states=current_joints.tolist(),
                sensitivities=sensitivities,
                pixel_to_mm=self.pixel_to_mm_ratio,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                arm=self.arm,
                camera_name=self.arm_config.camera_name,
            )
            self.calibration_points.append(cal_point)
            print(f"\n✓ 标定完成: {len(sensitivities)} 个关节 (含深度灵敏度)")

        return sensitivities

    def print_summary(self):
        """打印配置摘要"""
        has_depth = any(
            getattr(s, 'depth_dz_per_deg', 0.0) != 0.0
            for cp in self.calibration_points
            for s in cp.sensitivities
        )
        print(f"\n{'='*50}")
        print(f"SimpleIBVS 配置摘要 (arm={self.arm})")
        print(f"{'='*50}")
        print(f"  AprilTag家族: {self.tag_detector.tag_family}")
        print(f"  目标tag ID: {self.tag_detector.target_tag_ids or '全部'}")
        print(f"  tag物理尺寸: {self.tag_detector.tag_size_mm:.1f}mm")
        print(f"  目标像素位置: ({self.target_pixel_x:.1f}, {self.target_pixel_y:.1f})")
        print(f"  目标深度: {self.target_depth_mm:.0f}mm (容差: ±{self.depth_tolerance_mm:.0f}mm)")
        print(f"  深度权重: {self.depth_weight}")
        print(f"  3D雅可比: {'已启用' if has_depth else '仅2D (缺少深度灵敏度数据)'}")
        print(f"  相机焦距: {self.camera_fx:.0f}px")
        print(f"  增益: {self.gain}")
        print(f"  像素容差: {self.pixel_tolerance}px")
        print(f"  最大迭代: {self.max_iterations}")
        print(f"  pixel_to_mm_ratio: {self.pixel_to_mm_ratio:.3f}")
        print(f"  相机翻转: X={self._camera_flip_x}, Y={self._camera_flip_y}")
        print(f"  标定点数量: {len(self.calibration_points)}")

        if self.calibration_points:
            for cp in self.calibration_points:
                has_depth_cp = any(
                    getattr(s, 'depth_dz_per_deg', 0.0) != 0.0
                    for s in cp.sensitivities
                )
                depth_tag = " [3D]" if has_depth_cp else ""
                print(f"    [{cp.height_level}]{depth_tag} {len(cp.sensitivities)} 关节灵敏度数据")