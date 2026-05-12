#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单IBVS控制器 (Simple IBVS Controller)

纯粹基于关节灵敏度的图像视觉伺服，不需要手眼标定(extrinsic matrix)或正运动学(FK)。

核心原理:
  AprilTag检测 → tag像素误差 → 灵敏度雅可比(2×N) → 伪逆求解 → 关节调整量 → 机器人移动
  当机器人手遮挡住tag时，判定为对齐完成。

对比:
  - PBVS: 需要 extrinsic matrix + FK + coordinate transform → 误差来源多，标定困难
  - SimpleIBVS: 只需要 joint sensitivity + AprilTag → 简单直接，亚像素精度

对齐方式:
  不需要将工件放入卡槽。在目标位置放置AprilTag，机器人手接近并遮挡tag即可判定对齐。

优势:
  - AprilTag亚像素检测精度(~0.1-0.5px) vs 彩色圆点(~2-5px)
  - 不依赖手眼标定，不需要FK
  - 遮挡判定：手遮挡tag = 对齐完成，直觉简单
  - tag自带ID和旋转信息，无需颜色区分
"""

import numpy as np
import cv2
import json
import time
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
    # 遮挡状态
    occluded: bool = False
    occlusion_ratio: float = 0.0
    # 上一帧tag信息（用于遮挡追踪）
    prev_tag_center: Optional[Tuple[float, float]] = None


class SimpleIBVSController:
    """
    简单IBVS控制器 (AprilTag + 灵敏度雅可比)

    对齐方式: 机器人手接近并遮挡AprilTag。
    控制律: 像素误差 → 灵敏度雅可比(2×N) → 伪逆 → 关节调整量。
    """

    def __init__(self, arm: str = "right", gain: float = 0.6,
                 pixel_tolerance: float = 3.0, max_iterations: int = 25,
                 max_single_adjust: float = 2.0,
                 tag_family: str = "36h11",
                 target_tag_ids: List[int] = None,
                 tag_size_mm: float = 20.0):
        self.arm = arm
        self.arm_config = ARM_CONFIGS[arm]
        self.gain = gain
        self.pixel_tolerance = pixel_tolerance
        self.max_iterations = max_iterations
        self.max_single_adjust = max_single_adjust

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
        # 默认图像中心 (640x480)
        self.target_pixel_x = 320.0
        self.target_pixel_y = 240.0

        # 遮挡判定参数
        self.occlusion_consecutive_frames = 0
        self.occlusion_confirm_frames = 3  # 连续3帧看不到tag → 遮挡确认
        self.prev_tag_size_px = 0.0  # 上帧tag尺寸（用于尺寸缩小检测）
        self.size_reduction_threshold = 0.5  # tag尺寸缩小50% → 部分遮挡

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

    def _check_camera_flip(self):
        """检查相机方向翻转"""
        flip_config = self.arm_config.camera_flip
        cam_name = self.arm_config.camera_name
        if cam_name in flip_config:
            self._camera_flip_x, self._camera_flip_y = flip_config[cam_name]

    def detect_and_compute_error(self, image: np.ndarray,
                                 camera_fx: float = 531.0) -> TagAlignmentState:
        """
        检测AprilTag并计算对齐误差

        误差 = tag中心 - 目标像素位置

        Args:
            image: 相机图像
            camera_fx: 相机焦距(像素)，用于深度估算

        Returns:
            TagAlignmentState
        """
        state = TagAlignmentState()

        tags = self.tag_detector.detect(image)

        if not tags:
            # tag不可见 → 可能被遮挡
            state.tag_visible = False
            self.occlusion_consecutive_frames += 1

            if self.occlusion_consecutive_frames >= self.occlusion_confirm_frames:
                state.occluded = True
                print(f"  ✓ Tag被遮挡 (连续{self.occlusion_consecutive_frames}帧不可见) → 对齐!")
            return state

        # tag可见 → 重置遮挡计数
        self.occlusion_consecutive_frames = 0
        state.tag_visible = True

        # 取第一个(或最近的)tag
        # 如果有多个target tag，取离目标位置最近的
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

        # 部分遮挡检测: tag尺寸显著缩小
        if self.prev_tag_size_px > 0:
            ratio = best_tag['size_px'] / self.prev_tag_size_px
            state.occlusion_ratio = 1.0 - ratio
            if ratio < self.size_reduction_threshold:
                state.occluded = True
                print(f"  ✓ Tag部分遮挡 (尺寸缩小{state.occlusion_ratio*100:.0f}%) → 接近对齐!")
        self.prev_tag_size_px = best_tag['size_px']

        # 计算像素误差
        tx, ty = best_tag['center']
        error_x = tx - self.target_pixel_x
        error_y = ty - self.target_pixel_y

        # 相机翻转
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

        # 旋转误差: tag旋转偏离目标旋转
        # 目标旋转为0 (tag正朝上)
        state.error_rotation = best_tag['rotation_deg']

        # 深度估算
        depth_mm = self.tag_detector.estimate_depth_mm(best_tag, camera_fx)
        if depth_mm > 0:
            print(f"  Tag深度: {depth_mm:.0f}mm, 尺寸: {best_tag['size_px']:.1f}px")

        return state

    def get_interpolated_sensitivities(self, current_joints: np.ndarray) -> List[JointSensitivity]:
        """获取当前姿态的插值灵敏度（反距离加权）"""
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
            dx_values, dy_values, dx_weights, dy_weights = [], [], [], []
            for cp, w in zip(self.calibration_points, weights):
                for s in cp.sensitivities:
                    if s.joint_idx == jidx:
                        dx_values.append(s.pixel_dx_per_deg)
                        dy_values.append(s.pixel_dy_per_deg)
                        dx_weights.append(w)
                        dy_weights.append(w)
                        break

            if dx_values:
                interp_dx = sum(v * w for v, w in zip(dx_values, dx_weights))
                interp_dy = sum(v * w for v, w in zip(dy_values, dy_weights))

                if self._camera_flip_x:
                    interp_dx = -interp_dx
                if self._camera_flip_y:
                    interp_dy = -interp_dy

                interpolated.append(JointSensitivity(
                    joint_idx=jidx,
                    joint_name=self.joint_names.get(jidx, f"joint_{jidx}"),
                    pixel_dx_per_deg=interp_dx,
                    pixel_dy_per_deg=interp_dy,
                    mm_dx_per_deg=interp_dx * self.pixel_to_mm_ratio,
                    mm_dy_per_deg=interp_dy * self.pixel_to_mm_ratio,
                    calibration_angles=current_joints.tolist()
                ))

        return interpolated

    def compute_joint_adjustments(self, pixel_error_x: float, pixel_error_y: float,
                                  current_joints: np.ndarray) -> Dict[int, float]:
        """
        核心IBVS控制律: 像素误差 → 关节调整量

        J·W·Δθ = -e  →  Δθ = (J·W)^+ · (-e)
        """
        sensitivities = self.get_interpolated_sensitivities(current_joints)
        if not sensitivities:
            print("⚠ 无标定数据，无法计算调整量")
            return {}

        joint_indices = [s.joint_idx for s in sensitivities]
        n_joints = len(joint_indices)

        J = np.zeros((2, n_joints))
        W = np.zeros((n_joints, n_joints))

        for i, s in enumerate(sensitivities):
            J[0, i] = s.pixel_dx_per_deg
            J[1, i] = s.pixel_dy_per_deg
            W[i, i] = self.position_weights.get(s.joint_idx, 0.5)

        JW = J @ W
        error = np.array([pixel_error_x, pixel_error_y])

        try:
            JW_pinv = np.linalg.pinv(JW)
            delta_angles = JW_pinv @ (-error)
        except np.linalg.LinAlgError:
            delta_angles = self._single_joint_fallback(pixel_error_x, pixel_error_y, sensitivities)

        max_delta = np.max(np.abs(delta_angles))
        if max_delta > 5.0:
            delta_angles = self._single_joint_fallback(pixel_error_x, pixel_error_y, sensitivities)

        delta_angles = delta_angles * self.gain
        delta_angles = np.clip(delta_angles, -self.max_single_adjust, self.max_single_adjust)

        adjustments = {}
        for i, (jidx, delta) in enumerate(zip(joint_indices, delta_angles)):
            if abs(delta) > 0.01:
                adjustments[jidx] = float(delta)

        return adjustments

    def compute_rotation_adjustment(self, rotation_error: float) -> Dict[int, float]:
        """计算旋转误差的关节调整量"""
        if abs(rotation_error) < 2.0:
            return {}
        delta = -rotation_error * self.gain * 0.5
        delta = max(-self.max_single_adjust, min(self.max_single_adjust, delta))
        return {self.rotation_joint: delta}

    def _single_joint_fallback(self, pixel_error_x: float, pixel_error_y: float,
                               sensitivities: List[JointSensitivity]) -> np.ndarray:
        """单关节fallback: 选灵敏度最高的关节分别控制X和Y"""
        n = len(sensitivities)
        delta = np.zeros(n)

        best_x = max(range(n), key=lambda i: abs(sensitivities[i].pixel_dx_per_deg))
        if abs(sensitivities[best_x].pixel_dx_per_deg) > 0.01:
            delta[best_x] = -pixel_error_x / sensitivities[best_x].pixel_dx_per_deg

        best_y = max(range(n), key=lambda i: abs(sensitivities[i].pixel_dy_per_deg))
        if abs(sensitivities[best_y].pixel_dy_per_deg) > 0.01:
            delta[best_y] = -pixel_error_y / sensitivities[best_y].pixel_dy_per_deg

        return delta

    def compute_error_mm(self, pixel_error_x: float, pixel_error_y: float) -> Tuple[float, float]:
        """像素误差 → 毫米误差"""
        return pixel_error_x * self.pixel_to_mm_ratio, pixel_error_y * self.pixel_to_mm_ratio

    def set_target_pixel(self, x: float, y: float):
        """设置目标像素位置 (tag中心应该出现的位置)"""
        self.target_pixel_x = x
        self.target_pixel_y = y
        print(f"✓ 目标像素位置: ({x:.1f}, {y:.1f})")

    def reset_occlusion_tracking(self):
        """重置遮挡追踪状态"""
        self.occlusion_consecutive_frames = 0
        self.prev_tag_size_px = 0.0

    def print_summary(self):
        """打印配置摘要"""
        print(f"\n{'='*50}")
        print(f"SimpleIBVS 配置摘要 (arm={self.arm})")
        print(f"{'='*50}")
        print(f"  AprilTag家族: {self.tag_detector.tag_family}")
        print(f"  目标tag ID: {self.tag_detector.target_tag_ids or '全部'}")
        print(f"  tag物理尺寸: {self.tag_detector.tag_size_mm:.1f}mm")
        print(f"  目标像素位置: ({self.target_pixel_x:.1f}, {self.target_pixel_y:.1f})")
        print(f"  增益: {self.gain}")
        print(f"  像素容差: {self.pixel_tolerance}px")
        print(f"  最大迭代: {self.max_iterations}")
        print(f"  pixel_to_mm_ratio: {self.pixel_to_mm_ratio:.3f}")
        print(f"  遮挡确认帧数: {self.occlusion_confirm_frames}")
        print(f"  相机翻转: X={self._camera_flip_x}, Y={self._camera_flip_y}")
        print(f"  标定点数量: {len(self.calibration_points)}")

        if self.calibration_points:
            for cp in self.calibration_points:
                print(f"    [{cp.height_level}] {len(cp.sensitivities)} 关节灵敏度数据")