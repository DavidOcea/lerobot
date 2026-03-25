"""
双标记点精准对齐系统 - V3

功能:
1. 双标记点检测 (工件3绿 + 卡槽3红)
2. 左手/右手切换
3. 自动高度调整
4. XY对齐 + 旋转校正
5. Z轴精确控制 (双目立体视觉 + 单目尺寸估计)
6. 夹爪控制
7. 运动平滑
8. 多点标定插值 (方案A)
9. 雅可比运动学框架 (方案B预留)
10. 预设位置
11. 共享状态读取 (与示教程序协同)

标定方案:
- 方案A: 多点手动标定 + 线性插值 (已实现)
- 方案B: DH参数 + 雅可比矩阵 (框架预留)

Z轴控制:
- 双目立体视觉: 精度±0.5mm
- 单目尺寸估计: 精度±1.5mm
- 卡尔曼滤波融合
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
from pathlib import Path

# 导入共享状态模块
try:
    from precision_place.robot_status import (
        RobotStatusReader, joints_dict_to_array, JOINT_NAME_TO_INDEX
    )
    _has_status_reader = True
except ImportError:
    _has_status_reader = False
    RobotStatusReader = None

# 导入Z轴控制器
try:
    from precision_place.z_axis_controller import ZAxisController, DepthEstimate
    _has_z_controller = True
except ImportError:
    _has_z_controller = False
    ZAxisController = None


# ==================== 数据结构 ====================

@dataclass
class Marker:
    """单个标记"""
    x: float
    y: float
    color: str
    confidence: float


@dataclass
class DualMarkerState:
    """三标记状态"""
    workpiece_1: Optional[Marker] = None
    workpiece_2: Optional[Marker] = None
    workpiece_3: Optional[Marker] = None
    slot_1: Optional[Marker] = None
    slot_2: Optional[Marker] = None
    slot_3: Optional[Marker] = None
    offset_x: float = 0
    offset_y: float = 0
    rotation_error: float = 0
    workpiece_detected: bool = False
    slot_detected: bool = False
    alignment_quality: float = 0
    # P1新增：退化模式标记
    degraded_mode: bool = False  # 是否处于退化模式（标记不足）
    degraded_reason: str = ""    # 退化原因
    predicted_slot_center: Optional[Tuple[float, float]] = None  # 预测的卡槽中心

    @property
    def workpiece_markers(self) -> List[Optional[Marker]]:
        return [self.workpiece_1, self.workpiece_2, self.workpiece_3]

    @property
    def slot_markers(self) -> List[Optional[Marker]]:
        return [self.slot_1, self.slot_2, self.slot_3]

    @property
    def slot_marker_count(self) -> int:
        """检测到的卡槽标记数量"""
        return sum(1 for m in self.slot_markers if m)

    @property
    def workpiece_marker_count(self) -> int:
        """检测到的工件标记数量"""
        return sum(1 for m in self.workpiece_markers if m)


@dataclass
class ArmConfig:
    """手臂配置"""
    name: str
    camera_name: str
    camera_index: int
    # 第二相机 (用于双目Z轴控制)
    camera2_name: str = ""
    camera2_index: int = -1
    # 主要控制关节索引 (用于方案A的简化控制)
    primary_joints: List[int] = field(default_factory=list)
    gripper_idx: int = 0
    gripper_open: float = 0.0
    gripper_close: float = 50.0
    # DH参数 (方案B预留)
    dh_params: Optional[List[Dict]] = None
    # 相机方向翻转 (如果两个腕部相机安装方向相反)
    # 格式: {相机名: (x_flip, y_flip)}，True表示该相机方向与主标定相机相反
    camera_flip: Dict[str, Tuple[bool, bool]] = field(default_factory=dict)


# 手臂配置 - 7关节标定 (joint_1~6 + trunk_1)
# 重要: 如果两个腕部相机安装方向相反，需要配置camera_flip
# 当相机旋转180度安装时，X和Y方向都会翻转，需要设为(True, True)
ARM_CONFIGS = {
    'right': ArmConfig(
        name='right',
        camera_name='right_wrist',
        camera_index=6,
        camera2_name='right_wrist2',
        camera2_index=8,
        primary_joints=[7, 8, 9, 10, 11, 12, 14],  # right_arm_joint_1~6 + trunk_joint_1
        gripper_idx=13,
        gripper_open=0.0,
        gripper_close=50.0,
        dh_params=None,  # 待用户提供
        # 相机方向翻转配置:
        # 格式: (x_flip, y_flip) - True表示该相机方向与标定相机相反
        # 两个相机安装方向相反时，需要为副相机设置翻转
        camera_flip={
            'right_wrist': (False, False),    # 主相机作为参考方向
            'right_wrist2': (True, True),     # 副相机安装方向相反，需要翻转X和Y
        }
    ),
    'left': ArmConfig(
        name='left',
        camera_name='left_wrist',
        camera_index=2,
        camera2_name='left_wrist2',
        camera2_index=4,
        primary_joints=[0, 1, 2, 3, 4, 5, 14],  # left_arm_joint_1~6 + trunk_joint_1
        gripper_idx=6,
        gripper_open=0.0,
        gripper_close=50.0,
        dh_params=None,  # 待用户提供
        camera_flip={
            'left_wrist': (False, False),     # 主相机作为参考方向
            'left_wrist2': (True, True),      # 副相机安装方向相反，需要翻转X和Y
        }
    )
}


# ==================== 标定数据结构 ====================

@dataclass
class JointSensitivity:
    """单个关节的灵敏度数据"""
    joint_idx: int
    joint_name: str
    # 关节移动1度时，末端在相机中的像素变化
    pixel_dx_per_deg: float = 0.0  # X方向像素变化
    pixel_dy_per_deg: float = 0.0  # Y方向像素变化
    # 关节移动1度时，末端的实际毫米移动 (近似)
    mm_dx_per_deg: float = 0.0
    mm_dy_per_deg: float = 0.0
    # 标定时的关节角度
    calibration_angles: List[float] = field(default_factory=list)


@dataclass
class CalibrationPoint:
    """单个标定点 (特定姿态下的灵敏度)"""
    height_level: str  # 高度等级: "high", "medium", "low"
    joint_states: np.ndarray  # 标定时的所有关节状态
    sensitivities: List[JointSensitivity]  # 各关节的灵敏度
    pixel_to_mm: float = 0.5  # 该高度下的像素-毫米转换比例
    timestamp: str = ""
    arm: str = "right"  # 对应的手臂
    camera_name: str = ""  # 标定时使用的相机名称 (用于检测方向差异)


# ==================== 方案B: 雅可比运动学框架 ====================

class KinematicsModel(ABC):
    """运动学模型抽象基类 (方案B框架)"""

    @abstractmethod
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正运动学: 关节角度 -> 末端位姿

        Args:
            joint_angles: 关节角度数组

        Returns:
            末端位姿 [x, y, z, roll, pitch, yaw]
        """
        pass

    @abstractmethod
    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵

        Args:
            joint_angles: 关节角度数组

        Returns:
            6xN 雅可比矩阵 (N为关节数)
        """
        pass

    @abstractmethod
    def inverse_kinematics(self, target_pose: np.ndarray,
                          current_joints: np.ndarray) -> np.ndarray:
        """
        逆运动学: 目标位姿 -> 关节角度

        Args:
            target_pose: 目标末端位姿
            current_joints: 当前关节角度 (作为初值)

        Returns:
            目标关节角度
        """
        pass

    def compute_joint_velocities(self, joint_angles: np.ndarray,
                                 cartesian_velocity: np.ndarray) -> np.ndarray:
        """
        根据笛卡尔空间速度计算关节速度

        Args:
            joint_angles: 当前关节角度
            cartesian_velocity: 笛卡尔空间期望速度 [vx, vy, vz, wx, wy, wz]

        Returns:
            关节速度
        """
        J = self.compute_jacobian(joint_angles)
        # 使用伪逆求解
        J_pinv = np.linalg.pinv(J)
        joint_velocities = J_pinv @ cartesian_velocity
        return joint_velocities


class DHKinematicsModel(KinematicsModel):
    """基于DH参数的运动学模型 (方案B实现)"""

    def __init__(self, dh_params: List[Dict]):
        """
        Args:
            dh_params: DH参数列表, 每个元素为:
                {
                    'a': 连杆长度,
                    'alpha': 连杆扭转角,
                    'd': 连杆偏距,
                    'theta': 关节角偏移
                }
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        self._validate_dh_params()

    def _validate_dh_params(self):
        """验证DH参数有效性"""
        required_keys = ['a', 'alpha', 'd', 'theta']
        for i, params in enumerate(self.dh_params):
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"DH参数第{i}个关节缺少 '{key}'")

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """正运动学 - 待实现"""
        raise NotImplementedError("请提供DH参数后实现此方法")

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵 - 待实现"""
        raise NotImplementedError("请提供DH参数后实现此方法")

    def inverse_kinematics(self, target_pose: np.ndarray,
                          current_joints: np.ndarray) -> np.ndarray:
        """逆运动学 - 待实现"""
        raise NotImplementedError("请提供DH参数后实现此方法")


class DummyKinematicsModel(KinematicsModel):
    """空运动学模型 (方案A使用)"""

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        return np.zeros(6)

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        return np.zeros((6, len(joint_angles)))

    def inverse_kinematics(self, target_pose: np.ndarray,
                          current_joints: np.ndarray) -> np.ndarray:
        return current_joints.copy()


# ==================== 检测器 ====================

class DualPointDetector:
    """双标记点检测器"""
    
    COLOR_RANGES = {
        'green': {
            'lower': np.array([35, 70, 70]),
            'upper': np.array([85, 255, 255])
        },
        'red': {
            'lower': np.array([0, 50, 50]),
            'upper': np.array([10, 255, 255]),
            'lower2': np.array([160, 50, 50]),
            'upper2': np.array([180, 255, 255])
        },
        'blue': {
            'lower': np.array([100, 50, 50]),
            'upper': np.array([130, 255, 255])
        },
    }
    
    def __init__(self):
        self.workpiece_color = "green"
        self.slot_color = "red"
        self.min_area = 100      # 最小面积，降低以支持小标记
        self.max_area = 50000    # 最大面积，提高以支持近距离大标记

    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        self.workpiece_color = workpiece_color
        self.slot_color = slot_color

    def set_area_range(self, min_area: int, max_area: int):
        """设置标记检测的面积范围

        Args:
            min_area: 最小像素面积
            max_area: 最大像素面积
        """
        self.min_area = min_area
        self.max_area = max_area
        print(f"标记面积范围: {min_area} - {max_area} px²")

    def auto_adjust_area_range(self, marker_diameter_mm: float, distance_mm: float):
        """根据标记尺寸和距离自动调整面积范围

        Args:
            marker_diameter_mm: 标记直径
            distance_mm: 预期距离
        """
        # 计算预期像素直径 (f=311px, sensor=5.76mm)
        fx = 311.0
        sensor_width = 5.76
        pixel_diameter = fx * marker_diameter_mm / distance_mm

        # 计算面积
        radius = pixel_diameter / 2
        expected_area = 3.14159 * radius * radius

        # 设置范围 (预期面积的 0.3-3 倍)
        self.min_area = int(expected_area * 0.3)
        self.max_area = int(expected_area * 3)

        print(f"自动调整面积范围: {self.min_area} - {self.max_area} px²")
        print(f"  标记直径: {marker_diameter_mm}mm")
        print(f"  预期距离: {distance_mm}mm")
        print(f"  预期像素直径: {pixel_diameter:.0f}px")
        print(f"  预期面积: {expected_area:.0f}px²")
    
    def detect_markers_by_color(self, image: np.ndarray, color: str) -> List[Marker]:
        """检测指定颜色的所有标记"""
        if color not in self.COLOR_RANGES:
            return []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cr = self.COLOR_RANGES[color]
        
        mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
        if 'lower2' in cr:
            mask2 = cv2.inRange(hsv, cr['lower2'], cr['upper2'])
            mask = cv2.bitwise_or(mask, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:
                continue
            
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            confidence = circularity * min(1.0, area / 1000)
            markers.append(Marker(x=cx, y=cy, color=color, confidence=confidence))
        
        return markers
    
    def detect_triple_marker_state(self, image: np.ndarray,
                                    allow_degraded: bool = True) -> DualMarkerState:
        """
        检测三标记状态（工件3个，卡槽3个）

        Args:
            image: 输入图像
            allow_degraded: 是否允许退化模式（标记不足时仍尝试工作）

        Returns:
            DualMarkerState: 检测状态
        """
        state = DualMarkerState()

        # 工件标记 - 需要3个，至少1个即可工作（退化模式）
        wp_markers = self.detect_markers_by_color(image, self.workpiece_color)
        if len(wp_markers) >= 1:
            sorted_wp = sorted(wp_markers, key=lambda m: m.y)
            state.workpiece_1 = sorted_wp[0]
            if len(sorted_wp) >= 2:
                state.workpiece_2 = sorted_wp[1]
            if len(sorted_wp) >= 3:
                state.workpiece_3 = sorted_wp[2]
            state.workpiece_detected = True

        # 卡槽标记 - 需要3个，至少1个即可工作（退化模式）
        sl_markers = self.detect_markers_by_color(image, self.slot_color)
        if len(sl_markers) >= 1:
            sorted_sl = sorted(sl_markers, key=lambda m: m.y)
            state.slot_1 = sorted_sl[0]
            if len(sorted_sl) >= 2:
                state.slot_2 = sorted_sl[1]
            if len(sorted_sl) >= 3:
                state.slot_3 = sorted_sl[2]
            state.slot_detected = True

        # 检查是否处于退化模式
        if state.workpiece_detected and state.slot_detected:
            wp_count = state.workpiece_marker_count
            sl_count = state.slot_marker_count

            if wp_count < 2 or sl_count < 2:
                state.degraded_mode = True
                reasons = []
                if wp_count < 2:
                    reasons.append(f"工件标记不足({wp_count}/2)")
                if sl_count < 2:
                    reasons.append(f"卡槽标记不足({sl_count}/2)")
                state.degraded_reason = ", ".join(reasons)

            self._calculate_alignment(state)

        return state

    def detect_with_secondary_camera(self, image1: np.ndarray, image2: np.ndarray = None,
                                      flip_secondary: bool = True) -> DualMarkerState:
        """
        使用双相机检测标记（融合结果）

        当主相机卡槽标记不足时，尝试使用副相机补充

        重要：如果副相机旋转180度安装，其XY方向与主相机相反，需要翻转！

        Args:
            image1: 主相机图像
            image2: 副相机图像
            flip_secondary: 是否翻转副相机的XY坐标（副相机旋转180度时需要）

        Returns:
            融合后的检测状态
        """
        # 主相机检测
        state1 = self.detect_triple_marker_state(image1)

        if image2 is None:
            return state1

        # 如果主相机检测正常，直接返回
        if state1.slot_marker_count >= 2 and not state1.degraded_mode:
            return state1

        # 副相机检测
        state2 = self.detect_triple_marker_state(image2)

        # 尝试融合：使用检测到更多标记的相机结果
        if state2.slot_marker_count > state1.slot_marker_count:
            # 副相机卡槽标记更多，使用副相机的卡槽数据
            # 但工件标记仍使用主相机（工件在夹爪上，主相机更稳定）

            if flip_secondary:
                # 副相机旋转180度安装，需要翻转XY坐标
                # 获取图像尺寸用于翻转
                h, w = image2.shape[:2]
                center_x, center_y = w / 2, h / 2

                # 翻转标记坐标（绕中心点对称）
                def flip_marker(m):
                    if m is None:
                        return None
                    return Marker(
                        x=2 * center_x - m.x,  # X翻转
                        y=2 * center_y - m.y,  # Y翻转
                        color=m.color,
                        confidence=m.confidence
                    )

                state1.slot_1 = flip_marker(state2.slot_1)
                state1.slot_2 = flip_marker(state2.slot_2)
                state1.slot_3 = flip_marker(state2.slot_3)
            else:
                state1.slot_1 = state2.slot_1
                state1.slot_2 = state2.slot_2
                state1.slot_3 = state2.slot_3

            state1.slot_detected = state2.slot_detected

            # 重新计算对齐
            if state1.workpiece_detected and state1.slot_detected:
                self._calculate_alignment(state1)
                state1.degraded_mode = state1.slot_marker_count < 2
                if state1.degraded_mode:
                    state1.degraded_reason = f"融合后卡槽标记仍不足({state1.slot_marker_count}/2)"
                else:
                    state1.degraded_reason = f"使用副相机补充卡槽标记(已翻转坐标)"

        return state1

    def detect_dual_marker_state(self, image: np.ndarray) -> DualMarkerState:
        """检测标记状态（兼容旧接口，调用三标记检测）"""
        return self.detect_triple_marker_state(image)

    def _calculate_alignment(self, state: DualMarkerState):
        """计算对齐误差"""
        # 计算中心（使用所有检测到的标记点）
        wp_x_sum = sum(m.x for m in state.workpiece_markers if m)
        wp_y_sum = sum(m.y for m in state.workpiece_markers if m)
        wp_count = sum(1 for m in state.workpiece_markers if m)

        sl_x_sum = sum(m.x for m in state.slot_markers if m)
        sl_y_sum = sum(m.y for m in state.slot_markers if m)
        sl_count = sum(1 for m in state.slot_markers if m)

        if wp_count > 0 and sl_count > 0:
            wp_cx = wp_x_sum / wp_count
            wp_cy = wp_y_sum / wp_count
            sl_cx = sl_x_sum / sl_count
            sl_cy = sl_y_sum / sl_count

            state.offset_x = sl_cx - wp_cx
            state.offset_y = sl_cy - wp_cy

            # 计算旋转角度（使用首尾标记点）
            wp_top = state.workpiece_1
            wp_bottom = state.workpiece_3 or state.workpiece_2
            sl_top = state.slot_1
            sl_bottom = state.slot_3 or state.slot_2

            if wp_top and wp_bottom and sl_top and sl_bottom:
                wp_angle = np.degrees(np.arctan2(
                    wp_bottom.x - wp_top.x,
                    wp_bottom.y - wp_top.y
                ))
                sl_angle = np.degrees(np.arctan2(
                    sl_bottom.x - sl_top.x,
                    sl_bottom.y - sl_top.y
                ))
                state.rotation_error = wp_angle - sl_angle
                # 归一化到 [-180, 180]
                while state.rotation_error > 180:
                    state.rotation_error -= 360
                while state.rotation_error < -180:
                    state.rotation_error += 360

            # 计算检测质量
            wp_conf = sum(m.confidence for m in state.workpiece_markers if m) / max(wp_count, 1)
            sl_conf = sum(m.confidence for m in state.slot_markers if m) / max(sl_count, 1)
            state.alignment_quality = (wp_conf + sl_conf) / 2
    
    def visualize(self, image: np.ndarray, state: DualMarkerState = None, target_offset_x: float = 0.0,
                target_offset_y: float = 0.0) -> np.ndarray:
        """可视化"""
        vis = image.copy()

        if state is None:
            state = self.detect_triple_marker_state(image)

        # 工件标记
        for i, m in enumerate(state.workpiece_markers, 1):
            if m:
                cv2.circle(vis, (int(m.x), int(m.y)), 10, (0, 255, 0), 2)
                cv2.putText(vis, f"WP{i}", (int(m.x)-15, int(m.y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # 卡槽标记
        for i, m in enumerate(state.slot_markers, 1):
            if m:
                cv2.circle(vis, (int(m.x), int(m.y)), 10, (0, 0, 255), 2)
                cv2.putText(vis, f"SL{i}", (int(m.x)-15, int(m.y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        # 连线（连接所有工件标点）
        wp_valid = [m for m in state.workpiece_markers if m]
        if len(wp_valid) >= 2:
            for i in range(len(wp_valid) - 1):
                cv2.line(vis, (int(wp_valid[i].x), int(wp_valid[i].y)),
                        (int(wp_valid[i+1].x), int(wp_valid[i+1].y)), (0, 255, 0), 2)

        # 连线（连接所有卡槽标点）
        sl_valid = [m for m in state.slot_markers if m]
        if len(sl_valid) >= 2:
            for i in range(len(sl_valid) - 1):
                cv2.line(vis, (int(sl_valid[i].x), int(sl_valid[i].y)),
                        (int(sl_valid[i+1].x), int(sl_valid[i+1].y)), (0, 0, 255), 2)

        # 状态
        y = 30
        wp_count = sum(1 for m in state.workpiece_markers if m)
        sl_count = sum(1 for m in state.slot_markers if m)
        cv2.putText(vis, f"WP: {wp_count}/3", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if wp_count >= 2 else (0, 0, 255), 2)
        cv2.putText(vis, f"SL: {sl_count}/3", (10, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if sl_count >= 2 else (0, 0, 255), 2)

        if state.workpiece_detected and state.slot_detected:
            cv2.putText(vis, f"XY: ({state.offset_x:.0f}, {state.offset_y:.0f})", (10, y+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"Rot: {state.rotation_error:.1f}deg", (10, y+75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示目标偏移和误差（如果有目标偏移）
            if hasattr(self, 'target_offset_x') and (self.target_offset_x != 0 or self.target_offset_y != 0):
                error_x = state.offset_x - self.target_offset_x
                error_y = state.offset_y - self.target_offset_y
                cv2.putText(vis, f"Target: ({self.target_offset_x:.0f}, {self.target_offset_y:.0f})", (10, y+100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis, f"Error: ({error_x:.0f}, {error_y:.0f})", (10, y+125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        return vis


# ==================== 控制器 ====================

class PrecisionPlaceController:
    """
    精准放置控制器 - 多点标定版

    支持两种模式:
    - 方案A: 多点手动标定 + 插值 (默认)
    - 方案B: DH参数 + 雅可比 (需提供DH参数)

    Z轴控制:
    - 双目立体视觉 (主方法)
    - 单目尺寸估计 (备份)

    被动模式:
    - 与示教程序协同工作
    - 通过共享文件读取机器人状态
    - 不发送控制指令
    """

    def __init__(self, robot, camera, arm: str = "right", passive_mode: bool = False,
                 use_interpolation: bool = True, camera2=None, enable_z_control: bool = True):
        self.robot = robot
        self.camera = camera  # 主相机
        self.camera2 = camera2  # 第二相机 (用于双目Z轴控制)
        self.arm = arm
        self.passive_mode = passive_mode  # 被动模式：只读取，不发送动作
        self.use_interpolation = use_interpolation  # 使用加权插值 (False=最近邻)
        self.enable_z_control = enable_z_control and _has_z_controller  # 是否启用Z轴控制

        # 加载手臂配置
        self.arm_config = ARM_CONFIGS.get(arm, ARM_CONFIGS['right'])

        # 检测器
        self.detector = DualPointDetector()

        # 运动学模型 (方案B)
        self.kinematics_model: Optional[KinematicsModel] = None
        self._init_kinematics_model()

        # 共享状态读取器 (被动模式使用)
        self.status_reader: Optional[RobotStatusReader] = None
        if passive_mode and _has_status_reader:
            self.status_reader = RobotStatusReader()
            print("✓ 已启用共享状态读取 (从示教程序获取机器人状态)")

        # 方案A: 多点标定数据
        self.calibration_points: List[CalibrationPoint] = []
        self._load_calibration_points()

        # 相机方向翻转状态
        self._camera_flip_x = False
        self._camera_flip_y = False

        # 参数
        self.pixel_to_mm_ratio = 0.5  # 兼容旧标定
        self.gain = 0.6
        self.tolerance_mm = 2.0
        self.tolerance_deg = 5.0  # 旋转容差（度）
        self.max_iterations = 15
        self.settle_time = 0.2    # 减少等待时间，因为移动本身已经很慢

        # 对齐目标偏移（像素）- 当工件正确放置时，工件标点中心相对于卡槽标点中心的偏移
        self.target_offset_x = 0.0
        self.target_offset_y = 0.0

        # 设置偏移量时的关节状态（用于对齐时恢复高度）
        self._calibration_joint_states: Optional[np.ndarray] = None
        self._calibration_joint_dict: Optional[Dict[str, float]] = None

        # 设置偏移量时的目标深度（用于Z轴闭环控制）
        self._calibration_target_z: Optional[float] = None

        # 透视效应补偿参数
        self._setup_height: Optional[float] = None  # 设置偏移量时的高度
        self._reference_height: Optional[float] = None  # 设置偏移量时的高度（兼容）
        self._reference_pixel_to_mm: float = 0.5  # 设置偏移量时的像素比例

        # 相机透视方向配置
        # 相机倾斜方向决定了高度变化时卡槽在图像中的偏移方向
        # 格式: (x_direction, y_direction) 表示高度增加时卡槽在图像中的偏移方向
        # 例如: (1, 0) 表示高度增加时卡槽向X正方向偏移
        #      (-1, 0) 表示高度增加时卡槽向X负方向偏移
        #      (0, 1) 表示高度增加时卡槽向Y正方向偏移
        #
        # 如何确定：
        # 1. 设置偏移量（低位置）
        # 2. 抬高夹爪（高位置）
        # 3. 观察卡槽中心相对于工件中心的变化：
        #    - 如果卡槽向图像右边移动 → x_direction = 1
        #    - 如果卡槽向图像左边移动 → x_direction = -1
        #    - 如果卡槽向图像下方移动 → y_direction = 1
        #    - 如果卡槽向图像上方移动 → y_direction = -1
        self._camera_tilt_direction: Tuple[float, float] = (1.0, 0.0)  # 默认：X正方向

        # P1: 历史偏移记录（用于预测）
        self._historical_offset_x: List[float] = []
        self._historical_offset_y: List[float] = []
        self._max_history_length = 10

        # P1: 预测模式参数
        self.use_prediction: bool = True  # 是否使用预测位置
        self._predicted_slot_center: Optional[Tuple[float, float]] = None

        # 对齐视频显示
        self.show_alignment_video: bool = True  # 是否显示对齐视频
        self._alignment_window_name = "Alignment Monitor"

        # 遮挡恢复参数
        self.max_occlusion_frames = 5  # 连续遮挡多少帧才放弃
        self.occlusion_recovery_gain = 0.4  # 遮挡时的控制增益（更保守）

        # 旋转调整增益
        self.rotation_gain = 0.3
        self.max_rotation_adjust = 2.0  # 单次最大旋转调整（度）

        # 运动平滑参数 (全局设置，所有动作都用这个参数)
        self.smooth_steps = 30     # 分30步实现更平滑的移动
        self.smooth_delay = 0.1    # 每步0.1秒，总时间3秒

        # 高度控制 (用于粗调)
        self.height_joint_idx = self.arm_config.primary_joints[1]  # 通常joint_2影响高度

        # Z轴精确控制
        self.z_controller: Optional[ZAxisController] = None
        if self.enable_z_control:
            self.z_controller = ZAxisController(marker_diameter_mm=15.0)
            if camera2 is not None:
                self.z_controller.set_cameras(camera, camera2)
            print("✓ Z轴精确控制已启用")

        # Z轴目标深度 (mm)
        self.target_z: Optional[float] = None
        self.z_tolerance_mm = 1.0  # Z轴容差

        # 预设位置
        self.presets: Dict[str, np.ndarray] = {}
        self._load_presets()

        # 标定历史 (旧格式兼容)
        self.calibration_history: List[dict] = []
        self._load_calibration_history()

        # 关节名称映射
        self.joint_names = self._build_joint_names()

    def set_camera2(self, camera2):
        """设置第二相机"""
        self.camera2 = camera2
        if self.z_controller is not None:
            self.z_controller.set_cameras(self.camera, camera2)

    def _build_joint_names(self) -> Dict[int, str]:
        """构建关节索引到名称的映射"""
        names = {}
        # 左臂
        for i in range(7):
            names[i] = f"left_arm_joint_{i+1}"
        # 右臂
        for i in range(7, 14):
            names[i] = f"right_arm_joint_{i-6}"
        # 躯干
        names[14] = "trunk_joint_1"
        names[15] = "trunk_joint_2"
        return names

    def _init_kinematics_model(self):
        """初始化运动学模型"""
        if self.arm_config.dh_params:
            # 方案B: 使用DH参数
            self.kinematics_model = DHKinematicsModel(self.arm_config.dh_params)
            print("✓ 已加载DH运动学模型 (方案B)")
        else:
            # 方案A: 使用空模型
            self.kinematics_model = DummyKinematicsModel()
            print("使用多点标定模式 (方案A)")

    # ----------------- 手臂切换 -----------------

    def switch_arm(self, arm: str):
        """切换手臂"""
        if arm not in ARM_CONFIGS:
            print(f"不支持的手臂: {arm}")
            return False

        self.arm = arm
        self.arm_config = ARM_CONFIGS[arm]
        self.height_joint_idx = self.arm_config.primary_joints[1]
        self._init_kinematics_model()
        print(f"✓ 已切换到{arm}手")
        print(f"  相机: {self.arm_config.camera_name} (索引{self.arm_config.camera_index})")
        return True

    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        self.detector.set_marker_colors(workpiece_color, slot_color)

    # ==================== 关节状态读取 ====================

    def get_joint_states(self, max_age_ms: float = 200) -> Optional[np.ndarray]:
        """
        获取关节状态

        在被动模式下从共享文件读取，在主动模式下直接从机器人读取

        Args:
            max_age_ms: 最大数据年龄 (毫秒)，仅被动模式有效

        Returns:
            16维关节角度数组，如果失败返回 None
        """
        if self.passive_mode and self.status_reader is not None:
            # 被动模式: 从共享文件读取
            joints_dict = self.status_reader.read_joints(max_age_ms)
            if joints_dict is None:
                return None
            return joints_dict_to_array(joints_dict)
        else:
            # 主动模式: 直接从机器人读取
            if self.robot is None:
                return None
            try:
                # 使用 get_current_position() 获取关节位置
                pos_dict = self.robot.get_current_position()
                # 按照配置的 joint_order 转换为数组
                joint_order = self.robot.observation_joint_names
                joints = np.array([pos_dict.get(name, 0.0) for name in joint_order])
                if len(joints) >= 14:  # 至少需要14个关节
                    if len(joints) < 16:
                        # 补齐到16维
                        joints = np.pad(joints, (0, 16 - len(joints)))
                    return joints
                return None
            except Exception as e:
                print(f"读取关节状态失败: {e}")
                return None

    def check_shared_status(self) -> Dict[str, Any]:
        """检查共享状态 (用于调试)"""
        if self.status_reader is None:
            return {"error": "状态读取器未初始化"}
        return self.status_reader.get_status_info()

    # ==================== 方案A: 多点标定 ====================

    def calibrate_joint_sensitivity(self, joint_idx: int, move_degrees: float = 4.0) -> Tuple[bool, JointSensitivity]:
        """
        标定单个关节的灵敏度 (带实时视频显示)

        流程:
        1. 显示实时视频，用户按 Enter 采集初始图像
        2. 继续显示视频，用户移动关节
        3. 用户按 Enter 采集移动后图像
        4. 计算像素变化 -> 灵敏度

        Args:
            joint_idx: 关节索引
            move_degrees: 移动角度 (建议3-5度)

        Returns:
            (success, JointSensitivity)
        """
        print(f"\n{'='*60}")
        print(f"关节灵敏度标定: {self.joint_names.get(joint_idx, f'joint_{joint_idx}')}")
        print(f"{'='*60}")

        # 获取当前关节状态
        joints = self.get_joint_states()
        if joints is None:
            print("✗ 无法获取关节位置")
            if self.passive_mode:
                print("  请确认示教程序已启动并启用状态共享 (share_status=true)")
            return False, JointSensitivity(joint_idx, "")

        joint_name = self.joint_names.get(joint_idx, f"joint_{joint_idx}")
        initial_angle = joints[joint_idx]
        target_angle = initial_angle + move_degrees

        print(f"\n初始角度: {initial_angle:.2f}°")
        print(f"目标角度: {target_angle:.2f}° (移动约 {move_degrees}°)")
        print(f"\n[视频窗口] 按 'Enter' 采集图像，按 'q' 取消")

        window_name = f"Calibration: {joint_name}"
        img1 = None
        img2 = None
        phase = 1  # 1=采集初始图像, 2=采集移动后图像

        while True:
            # 读取图像
            frame = self.camera.read()
            if frame is None:
                continue

            # 检测标记点并可视化
            state = self.detector.detect_dual_marker_state(frame)
            vis = self.detector.visualize(frame, state)

            # 获取当前关节角度
            current_joints = self.get_joint_states()
            current_angle = current_joints[joint_idx] if current_joints is not None else 0.0

            # 叠加信息
            info_y = 30
            cv2.putText(vis, f"Joint: {joint_name}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(vis, f"Current: {current_angle:.2f} deg", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            cv2.putText(vis, f"Target: {target_angle:.2f} deg", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 显示阶段提示
            if phase == 1:
                cv2.putText(vis, "Phase 1: Press ENTER to capture initial image", (10, vis.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis, "Press 'q' to cancel", (10, vis.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            else:
                # 显示移动量
                moved = current_angle - initial_angle
                move_color = (0, 255, 0) if abs(moved) >= move_degrees * 0.8 else (0, 165, 255)
                cv2.putText(vis, f"Moved: {moved:.2f} deg", (10, info_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, move_color, 2)
                cv2.putText(vis, "Phase 2: Press ENTER to capture final image", (10, vis.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis, "Press 'q' to cancel", (10, vis.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(window_name)
                print("✗ 标定已取消")
                return False, JointSensitivity(joint_idx, joint_name)

            elif key == 13 or key == 10:  # Enter key
                if phase == 1:
                    img1 = frame.copy()
                    print(f"  ✓ 已采集初始图像 (角度: {current_angle:.2f}°)")
                    phase = 2
                    print(f"\n  请移动关节到目标位置 ({target_angle:.2f}°)")
                else:
                    img2 = frame.copy()
                    final_angle = current_angle
                    print(f"  ✓ 已采集移动后图像 (角度: {final_angle:.2f}°)")
                    break

        cv2.destroyWindow(window_name)

        # 计算实际移动角度
        actual_move = final_angle - initial_angle
        print(f"\n实际移动: {actual_move:.2f}°")

        if abs(actual_move) < 0.1:
            print("⚠ 警告: 移动角度很小，标定可能不准确")

        # 计算像素变化
        pixel_dx, pixel_dy = self._compute_pixel_shift(img1, img2)

        if pixel_dx is None:
            print("✗ 特征点匹配失败")
            return False, JointSensitivity(joint_idx, joint_name)

        # 使用实际移动角度计算灵敏度
        move_degrees_actual = abs(actual_move) if abs(actual_move) > 0.1 else move_degrees

        # 计算灵敏度
        sensitivity = JointSensitivity(
            joint_idx=joint_idx,
            joint_name=joint_name,
            pixel_dx_per_deg=pixel_dx / move_degrees_actual,
            pixel_dy_per_deg=pixel_dy / move_degrees_actual,
            mm_dx_per_deg=0.0,  # 需要pixel_to_mm_ratio
            mm_dy_per_deg=0.0,
            calibration_angles=joints.tolist()
        )

        # 更新mm灵敏度 (如果有pixel_to_mm_ratio)
        if self.pixel_to_mm_ratio > 0:
            sensitivity.mm_dx_per_deg = sensitivity.pixel_dx_per_deg * self.pixel_to_mm_ratio
            sensitivity.mm_dy_per_deg = sensitivity.pixel_dy_per_deg * self.pixel_to_mm_ratio

        print(f"\n标定结果:")
        print(f"  实际移动: {actual_move:.2f}°")
        print(f"  像素变化: ({pixel_dx:.1f}, {pixel_dy:.1f}) pixels")
        print(f"  灵敏度: X={sensitivity.pixel_dx_per_deg:.2f} px/deg, Y={sensitivity.pixel_dy_per_deg:.2f} px/deg")

        return True, sensitivity

    def calibrate_joint_sensitivity_auto(self, joint_idx: int, move_degrees: float = 4.0,
                                         settle_time: float = 1.0, show_progress: bool = True) -> Tuple[bool, JointSensitivity]:
        """
        自动标定单个关节的灵敏度（无需人工干预）

        流程:
        1. 获取当前关节角度
        2. 采集初始图像
        3. 自动移动关节指定角度
        4. 等待稳定
        5. 采集移动后图像
        6. 计算像素变化和灵敏度

        Args:
            joint_idx: 关节索引
            move_degrees: 移动角度（建议3-5度）
            settle_time: 移动后等待稳定的时间（秒）
            show_progress: 是否显示进度信息

        Returns:
            (success, JointSensitivity)
        """
        if self.passive_mode:
            print("✗ 自动标定需要独立模式（不能使用被动模式）")
            return False, JointSensitivity(joint_idx, "")

        if show_progress:
            print(f"\n{'='*60}")
            print(f"自动标定关节: {self.joint_names.get(joint_idx, f'joint_{joint_idx}')}")
            print(f"{'='*60}")

        # 1. 获取当前关节状态
        joints = self.get_joint_states()
        if joints is None:
            print("✗ 无法获取关节位置")
            return False, JointSensitivity(joint_idx, "")

        joint_name = self.joint_names.get(joint_idx, f"joint_{joint_idx}")
        initial_angle = joints[joint_idx]
        target_angle = initial_angle + move_degrees

        if show_progress:
            print(f"  初始角度: {initial_angle:.2f}°")
            print(f"  目标角度: {target_angle:.2f}° (移动 {move_degrees}°)")

        # 2. 采集初始图像（预热+多帧平均）
        if show_progress:
            print(f"  [1/4] 采集初始图像（预热+多帧平均）...")

        # 预热：丢弃前几帧
        for i in range(3):
            _ = self.camera.read()
            time.sleep(0.03)

        img1_list = []
        for i in range(5):
            img = self.camera.read()
            if img is not None:
                img1_list.append(img)
            time.sleep(0.05)

        if not img1_list:
            print("  ✗ 无法采集图像")
            return False, JointSensitivity(joint_idx, joint_name)
        img1 = img1_list[len(img1_list)//2]  # 取中间帧作为代表

        # 3. 自动移动关节
        if show_progress:
            print(f"  [2/4] 自动移动关节 {move_degrees}°...")
        target_joints = joints.copy()
        target_joints[joint_idx] = target_angle

        # 平滑移动到目标位置
        self._smooth_move_single_joint(joint_idx, target_angle, steps=10)

        # 4. 等待稳定
        if show_progress:
            print(f"  [3/4] 等待稳定 ({settle_time}秒)...")
        time.sleep(settle_time)

        # 5. 采集移动后图像（预热+多帧平均）
        if show_progress:
            print(f"  [4/4] 采集移动后图像（预热+多帧平均）...")

        # 预热：丢弃前几帧
        for i in range(3):
            _ = self.camera.read()
            time.sleep(0.03)

        img2_list = []
        for i in range(5):
            img = self.camera.read()
            if img is not None:
                img2_list.append(img)
            time.sleep(0.05)

        if not img2_list:
            print("  ✗ 无法采集图像")
            return False, JointSensitivity(joint_idx, joint_name)
        img2 = img2_list[len(img2_list)//2]  # 取中间帧作为代表

        # 获取最终角度
        final_joints = self.get_joint_states()
        final_angle = final_joints[joint_idx] if final_joints is not None else target_angle

        if show_progress:
            print(f"  最终角度: {final_angle:.2f}°")

        # 6. 计算实际移动角度
        actual_move = final_angle - initial_angle
        if show_progress:
            print(f"  实际移动: {actual_move:.2f}°")

        if abs(actual_move) < 0.5:
            print("  ⚠ 警告: 移动角度太小，标定可能不准确")

        # 7. 计算像素变化
        if show_progress:
            print(f"  计算像素变化...")
        pixel_dx, pixel_dy = self._compute_pixel_shift(img1, img2, num_samples=5)

        if pixel_dx is None:
            print("  ✗ 特征点匹配失败")
            return False, JointSensitivity(joint_idx, joint_name)

        # 8. 使用实际移动角度计算灵敏度
        move_degrees_actual = abs(actual_move) if abs(actual_move) > 0.5 else move_degrees

        # 9. 计算灵敏度
        sensitivity = JointSensitivity(
            joint_idx=joint_idx,
            joint_name=joint_name,
            pixel_dx_per_deg=pixel_dx / move_degrees_actual,
            pixel_dy_per_deg=pixel_dy / move_degrees_actual,
            mm_dx_per_deg=0.0,
            mm_dy_per_deg=0.0,
            calibration_angles=joints.tolist()
        )

        # 更新mm灵敏度
        if self.pixel_to_mm_ratio > 0:
            sensitivity.mm_dx_per_deg = sensitivity.pixel_dx_per_deg * self.pixel_to_mm_ratio
            sensitivity.mm_dy_per_deg = sensitivity.pixel_dy_per_deg * self.pixel_to_mm_ratio

        # 10. 有效性检查
        magnitude = np.sqrt(sensitivity.pixel_dx_per_deg**2 + sensitivity.pixel_dy_per_deg**2)

        if show_progress:
            print(f"\n  标定结果:")
            print(f"    实际移动: {actual_move:.2f}°")
            print(f"    像素变化: ({pixel_dx:.1f}, {pixel_dy:.1f}) pixels")
            print(f"    灵敏度: X={sensitivity.pixel_dx_per_deg:.2f} px/deg, Y={sensitivity.pixel_dy_per_deg:.2f} px/deg")
            print(f"    灵敏度幅值: {magnitude:.2f} px/deg")

        if magnitude < 0.5:
            print("\n  ⚠⚠⚠ 警告: 灵敏度异常小，可能检测失败！")
            print("        可能原因:")
            print("        1. 标记不在视野内或被遮挡")
            print("        2. 光线变化大")
            print("        3. 关节移动方向与相机视角垂直（几乎没有像素变化）")
            print("        建议: 检查标记是否可见，然后重新标定此关节")
            # 仍然返回结果，但标记为可能无效
        elif magnitude < 2.0:
            print(f"\n  ⚠ 注意: 灵敏度较小 ({magnitude:.2f} px/deg)，请确认是否正确")
        else:
            print(f"  ✓ 标定成功")

        return True, sensitivity

    def _smooth_move_single_joint(self, joint_idx: int, target_angle: float, steps: int = 10):
        """平滑移动单个关节"""
        if self.passive_mode:
            return False

        joints = self.get_joint_states()
        if joints is None or len(joints) < 16:
            return False

        initial_angle = joints[joint_idx]

        for step in range(1, steps + 1):
            alpha = step / steps
            alpha = alpha * alpha * (3 - 2 * alpha)  # ease-in-out

            current_angle = initial_angle + (target_angle - initial_angle) * alpha

            # 构建action格式
            action = {f"{name}.pos": float(joints[i]) for i, name in enumerate(self.robot.observation_joint_names)}
            action[f"{self.robot.observation_joint_names[joint_idx]}.pos"] = float(current_angle)

            self.robot.send_action(action)
            time.sleep(0.05)  # 每步50ms

        return True

    def _compute_pixel_shift(self, img1: np.ndarray, img2: np.ndarray,
                              use_marker: bool = True, num_samples: int = 5) -> Tuple[Optional[float], Optional[float]]:
        """计算图像间的像素偏移

        Args:
            img1: 第一帧图像
            img2: 第二帧图像
            use_marker: 是否使用标记检测（推荐True，更准确）
            num_samples: 采样帧数（用于平均，提高稳定性）

        Returns:
            (dx, dy) 或 (None, None)
        """
        if use_marker:
            # 使用标记检测计算偏移（更准确）
            offsets_x = []
            offsets_y = []

            for i in range(num_samples):
                state1 = self.detector.detect_dual_marker_state(img1)
                state2 = self.detector.detect_dual_marker_state(img2)

                if state1.workpiece_detected and state1.slot_detected and \
                   state2.workpiece_detected and state2.slot_detected:
                    offsets_x.append(state2.offset_x - state1.offset_x)
                    offsets_y.append(state2.offset_y - state1.offset_y)

                time.sleep(0.03)

            if len(offsets_x) < 2:
                # 标记检测失败，回退到光流法
                print(f"    [!] 标记检测失败 ({len(offsets_x)}/{num_samples})，使用光流法")
                return self._compute_pixel_shift(img1, img2, use_marker=False)

            print(f"    标记检测成功: {len(offsets_x)}/{num_samples} 次")
            # 使用中值滤波
            return float(np.median(offsets_x)), float(np.median(offsets_y))

        # 光流法（fallback）
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        pts = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
        if pts is None or len(pts) < 10:
            return None, None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None)
        if p1 is None:
            return None, None

        good = p1[st == 1] - pts[st == 1]
        # 检查是否有足够的有效匹配点
        if len(good) < 5:
            return None, None

        dx = np.mean(good, axis=0)[0]
        dy = np.mean(good, axis=0)[1]

        return float(dx), float(dy)

    def calibrate_all_joints(self, move_degrees: float = 4.0) -> bool:
        """
        标定所有主要关节 (多点标定)

        在当前姿态下标定所有primary_joints的灵敏度
        """
        print(f"\n{'#'*60}")
        print("# 多点标定 - 所有主要关节")
        print(f"{'#'*60}")

        if self.passive_mode:
            print("\n[示教模式]")
            print("请确保示教程序已启动 (./run.sh)")
            print("移动示教器对应关节，执行机器人会跟随移动")

        # 获取当前高度/姿态信息
        # 获取关节状态
        joints = self.get_joint_states()
        if joints is None:
            print("✗ 无法获取关节状态")
            return False

        # 确定高度等级
        height_level = self._estimate_height_level(joints)
        print(f"\n当前高度等级: {height_level}")

        # 创建新的标定点
        cal_point = CalibrationPoint(
            height_level=height_level,
            joint_states=joints.tolist(),
            sensitivities=[],
            pixel_to_mm=self.pixel_to_mm_ratio,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            arm=self.arm,
            camera_name=self.arm_config.camera_name  # 记录标定时的相机
        )

        # 标定每个主要关节
        primary_joints = self.arm_config.primary_joints

        print(f"\n将标定 {len(primary_joints)} 个关节:")
        for i, jidx in enumerate(primary_joints):
            print(f"  {i+1}. {self.joint_names.get(jidx, f'joint_{jidx}')} (索引{jidx})")

        input("\n按 Enter 开始标定...")

        success_count = 0
        for i, jidx in enumerate(primary_joints):
            print(f"\n{'='*40}")
            print(f"[{i+1}/{len(primary_joints)}] 标定关节 {jidx}")
            print(f"{'='*40}")

            success, sensitivity = self.calibrate_joint_sensitivity(jidx, move_degrees)

            if success:
                cal_point.sensitivities.append(sensitivity)
                success_count += 1
            else:
                print(f"✗ 关节 {jidx} 标定失败，跳过")

        # 保存标定点
        if success_count > 0:
            self.calibration_points.append(cal_point)
            self._save_calibration_points()
            print(f"\n✓ 标定完成: {success_count}/{len(primary_joints)} 个关节成功")
            print(f"✓ 已保存到高度等级 '{height_level}'")
            return True
        else:
            print(f"\n✗ 标定失败: 没有关节标定成功")
            return False

    def calibrate_all_joints_auto(self, move_degrees: float = 4.0, settle_time: float = 1.0,
                                  return_after_calib: bool = True) -> bool:
        """
        自动标定所有主要关节（无需人工干预）

        Args:
            move_degrees: 每个关节的移动角度（建议3-5度）
            settle_time: 移动后等待稳定的时间（秒）
            return_after_calib: 标定完成后是否返回初始位置

        Returns:
            是否所有关节都标定成功
        """
        if self.passive_mode:
            print("✗ 自动标定需要独立模式（不能使用被动模式）")
            return False

        print(f"\n{'#'*60}")
        print("# 自动标定 - 所有主要关节")
        print(f"{'#'*60}")

        # 保存初始位置以便后续返回
        initial_joints = self.get_joint_states()
        if initial_joints is None:
            print("✗ 无法获取关节状态")
            return False

        # 确定高度等级
        height_level = self._estimate_height_level(initial_joints)
        print(f"\n当前高度等级: {height_level}")

        # 创建新的标定点
        cal_point = CalibrationPoint(
            height_level=height_level,
            joint_states=initial_joints.tolist(),
            sensitivities=[],
            pixel_to_mm=self.pixel_to_mm_ratio,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            arm=self.arm,
            camera_name=self.arm_config.camera_name  # 记录标定时的相机
        )

        # 标定每个主要关节
        primary_joints = self.arm_config.primary_joints

        print(f"\n将自动标定 {len(primary_joints)} 个关节:")
        for i, jidx in enumerate(primary_joints):
            print(f"  {i+1}. {self.joint_names.get(jidx, f'joint_{jidx}')} (索引{jidx})")

        input("\n按 Enter 开始自动标定...")

        success_count = 0
        failed_joints = []

        for i, jidx in enumerate(primary_joints):
            print(f"\n{'='*40}")
            print(f"[{i+1}/{len(primary_joints)}] 自动标定关节 {jidx}")
            print(f"{'='*40}")

            success, sensitivity = self.calibrate_joint_sensitivity_auto(
                jidx, move_degrees, settle_time, show_progress=True
            )

            if success:
                cal_point.sensitivities.append(sensitivity)
                success_count += 1
            else:
                print(f"✗ 关节 {jidx} 标定失败，跳过")
                failed_joints.append(jidx)

        # 返回初始位置（可选）
        if return_after_calib:
            print(f"\n返回初始位置...")
            self._smooth_move_all_joints(initial_joints)

        # 保存标定点
        if success_count > 0:
            self.calibration_points.append(cal_point)
            self._save_calibration_points()

            print(f"\n{'='*60}")
            print(f"✓ 自动标定完成: {success_count}/{len(primary_joints)} 个关节成功")
            print(f"✓ 已保存到高度等级 '{height_level}'")

            if failed_joints:
                print(f"\n⚠ 以下关节标定失败: {failed_joints}")
                print(f"  可能原因:")
                print(f"    - 该关节在当前姿态下无法产生足够的图像位移")
                print(f"    - 纹理不足导致光流匹配失败")
                print(f"    - 标点移出视野")

            return True
        else:
            print(f"\n✗ 自动标定失败: 没有关节标定成功")
            return False

    def _estimate_height_level(self, joints: np.ndarray) -> str:
        """
        根据关节状态估计高度等级

        使用多关节加权评估末端高度：
        - 肩部俯仰 (joint_2): 主要影响高度
        - 肘部俯仰 (joint_3): 次要影响
        - 躯干 (trunk_1): 影响整体高度基准
        """
        if len(self.arm_config.primary_joints) < 3:
            return "medium"

        primary_joints = self.arm_config.primary_joints

        # 获取关键关节角度
        shoulder_idx = primary_joints[1]  # joint_2: 肩部俯仰
        elbow_idx = primary_joints[2]     # joint_3: 肘部俯仰
        trunk_idx = 14                    # trunk_joint_1

        shoulder_angle = joints[shoulder_idx] if shoulder_idx < len(joints) else 0
        elbow_angle = joints[elbow_idx] if elbow_idx < len(joints) else 0
        trunk_angle = joints[trunk_idx] if trunk_idx < len(joints) else 0

        # 加权计算高度分数
        # 肩部角度越大(手臂越抬高)，末端越高
        # 肘部角度影响前臂位置
        # 躯干角度影响整体高度基准
        height_score = (
            shoulder_angle * 0.50 +  # 肩部权重最大
            elbow_angle * 0.30 +     # 肘部次之
            trunk_angle * 0.20       # 躯干影响最小
        )

        # 根据分数判断高度等级
        if height_score > 35:
            return "high"
        elif height_score > 15:
            return "medium"
        else:
            return "low"

    def get_interpolated_sensitivities(self, current_joints: np.ndarray) -> List[JointSensitivity]:
        """
        根据当前关节状态获取灵敏度

        根据 use_interpolation 配置选择插值方式:
        - True: 加权插值 (平滑过渡)
        - False: 最近邻 (稳定简单)
        """
        if not self.calibration_points:
            return []

        # 只使用当前手臂的标定点
        arm_points = [cp for cp in self.calibration_points if cp.arm == self.arm]
        if not arm_points:
            arm_points = self.calibration_points

        if len(arm_points) == 1:
            sensitivities = arm_points[0].sensitivities
        elif self.use_interpolation:
            sensitivities = self._weighted_interpolation(current_joints, arm_points)
        else:
            sensitivities = self._nearest_neighbor(current_joints, arm_points)

        # 应用相机方向翻转
        return self._apply_camera_flip_to_sensitivities(sensitivities)

    def _nearest_neighbor(self, current_joints: np.ndarray,
                          arm_points: List[CalibrationPoint]) -> List[JointSensitivity]:
        """最近邻方法: 选择距离最近的标定点"""
        primary_joints = self.arm_config.primary_joints
        min_dist = float('inf')
        best_point = arm_points[0]

        for cp in arm_points:
            cp_joints = np.array(cp.joint_states)
            dist = np.linalg.norm(
                np.array([cp_joints[i] for i in primary_joints]) -
                np.array([current_joints[i] for i in primary_joints])
            )
            if dist < min_dist:
                min_dist = dist
                best_point = cp

        print(f"  最近邻: {best_point.height_level} (距离={min_dist:.2f})")
        return best_point.sensitivities

    def _weighted_interpolation(self, current_joints: np.ndarray,
                                 arm_points: List[CalibrationPoint]) -> List[JointSensitivity]:
        """加权插值方法: 对所有标定点进行距离加权平均"""
        primary_joints = self.arm_config.primary_joints

        # 计算到各标定点的距离
        distances = []
        for cp in arm_points:
            cp_joints = np.array(cp.joint_states)
            dist = np.linalg.norm(
                np.array([cp_joints[i] for i in primary_joints]) -
                np.array([current_joints[i] for i in primary_joints])
            )
            distances.append(dist)

        # 计算权重 (反距离加权)
        epsilon = 0.001
        weights = [1.0 / (d + epsilon) for d in distances]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 打印调试信息
        print(f"  插值权重: ", end="")
        for cp, w in zip(arm_points, weights):
            print(f"{cp.height_level}={w*100:.1f}% ", end="")
        print()

        # 对每个关节的灵敏度进行加权平均
        joint_indices = set()
        for cp in arm_points:
            for s in cp.sensitivities:
                joint_indices.add(s.joint_idx)
        joint_indices = sorted(joint_indices)

        interpolated = []
        for jidx in joint_indices:
            dx_values = []
            dy_values = []
            dx_weights = []
            dy_weights = []

            for cp, w in zip(arm_points, weights):
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

                interp_mm_dx = interp_dx * self.pixel_to_mm_ratio if self.pixel_to_mm_ratio > 0 else 0
                interp_mm_dy = interp_dy * self.pixel_to_mm_ratio if self.pixel_to_mm_ratio > 0 else 0

                joint_name = self.joint_names.get(jidx, f"joint_{jidx}")
                sensitivity = JointSensitivity(
                    joint_idx=jidx,
                    joint_name=joint_name,
                    pixel_dx_per_deg=interp_dx,
                    pixel_dy_per_deg=interp_dy,
                    mm_dx_per_deg=interp_mm_dx,
                    mm_dy_per_deg=interp_mm_dy,
                    calibration_angles=current_joints.tolist()
                )
                interpolated.append(sensitivity)

        return interpolated

    def _load_calibration_points(self):
        """加载多点标定数据"""
        path = Path(__file__).parent / "calibration_points.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)

            self.calibration_points = []
            for cp_data in data.get('points', []):
                sensitivities = [
                    JointSensitivity(**s) for s in cp_data.get('sensitivities', [])
                ]
                cp = CalibrationPoint(
                    height_level=cp_data['height_level'],
                    joint_states=cp_data['joint_states'],
                    sensitivities=sensitivities,
                    timestamp=cp_data.get('timestamp', ''),
                    arm=cp_data.get('arm', 'right'),
                    camera_name=cp_data.get('camera_name', '')  # 加载相机名称
                )
                self.calibration_points.append(cp)

            print(f"✓ 已加载 {len(self.calibration_points)} 个标定点")

            # 检测相机方向翻转
            self._check_camera_flip()

    def _save_calibration_points(self):
        """保存多点标定数据"""
        path = Path(__file__).parent / "calibration_points.json"

        data = {
            'version': '1.0',
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
                        'mm_dx_per_deg': s.mm_dx_per_deg,
                        'mm_dy_per_deg': s.mm_dy_per_deg,
                        'calibration_angles': s.calibration_angles
                    }
                    for s in cp.sensitivities
                ],
                'timestamp': cp.timestamp,
                'arm': cp.arm,
                'camera_name': cp.camera_name  # 保存相机名称
            }
            data['points'].append(cp_data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def show_calibration_points(self):
        """显示所有标定点"""
        if not self.calibration_points:
            print("\n暂无标定点，请先运行 calibrate_all_joints()")
            return

        print(f"\n{'='*60}")
        print(f"标定点列表 ({len(self.calibration_points)} 个)")
        print(f"{'='*60}")

        for i, cp in enumerate(self.calibration_points):
            camera_info = f" | 相机: {cp.camera_name}" if cp.camera_name else ""
            print(f"\n[{i+1}] {cp.timestamp} | {cp.arm}{camera_info} | 高度: {cp.height_level}")
            print(f"    关节数: {len(cp.sensitivities)}")
            for s in cp.sensitivities:
                print(f"      - {s.joint_name}: ({s.pixel_dx_per_deg:.2f}, {s.pixel_dy_per_deg:.2f}) px/deg")

    def _check_camera_flip(self):
        """
        检测相机方向翻转

        如果标定时的相机与当前运行的相机不同，检查是否需要翻转灵敏度符号
        """
        self._camera_flip_x = False
        self._camera_flip_y = False

        if not self.calibration_points:
            return

        # 获取标定时的相机名称
        calib_camera = self.calibration_points[0].camera_name
        current_camera = self.arm_config.camera_name

        if not calib_camera:
            print("  ⚠ 标定数据中缺少相机信息，无法检测方向翻转")
            return

        if calib_camera == current_camera:
            print(f"  ✓ 标定相机与当前相机一致: {current_camera}")
            return

        # 检查是否在 camera_flip 配置中标记为翻转
        camera_flip_config = self.arm_config.camera_flip
        if current_camera in camera_flip_config:
            flip_x, flip_y = camera_flip_config[current_camera]
            self._camera_flip_x = flip_x
            self._camera_flip_y = flip_y
            if flip_x or flip_y:
                print(f"  ⚠ 检测到相机方向翻转: {calib_camera} -> {current_camera}")
                print(f"      X翻转: {flip_x}, Y翻转: {flip_y}")
                print(f"      将自动调整灵敏度符号")
        else:
            # 相机不同但未配置翻转，发出警告
            print(f"  ⚠ 警告: 标定相机({calib_camera})与当前相机({current_camera})不同")
            print(f"      如果相机安装方向不同，可能需要重新标定或配置 camera_flip")

    def _apply_camera_flip_to_sensitivities(self, sensitivities: List[JointSensitivity]) -> List[JointSensitivity]:
        """
        应用相机方向翻转到灵敏度数据

        如果相机安装方向相反（旋转180度），需要翻转X和Y方向的灵敏度符号
        """
        if not (self._camera_flip_x or self._camera_flip_y):
            return sensitivities

        flipped = []
        for s in sensitivities:
            new_dx = -s.pixel_dx_per_deg if self._camera_flip_x else s.pixel_dx_per_deg
            new_dy = -s.pixel_dy_per_deg if self._camera_flip_y else s.pixel_dy_per_deg
            new_mm_dx = -s.mm_dx_per_deg if self._camera_flip_x else s.mm_dx_per_deg
            new_mm_dy = -s.mm_dy_per_deg if self._camera_flip_y else s.mm_dy_per_deg

            flipped_s = JointSensitivity(
                joint_idx=s.joint_idx,
                joint_name=s.joint_name,
                pixel_dx_per_deg=new_dx,
                pixel_dy_per_deg=new_dy,
                mm_dx_per_deg=new_mm_dx,
                mm_dy_per_deg=new_mm_dy,
                calibration_angles=s.calibration_angles
            )
            flipped.append(flipped_s)

        return flipped

    def _get_stable_pixel_offset(self, num_samples: int = 5, min_quality: float = 0.3,
                                   warmup_frames: int = 3) -> Optional[Tuple[float, float, float]]:
        """
        获取稳定的像素偏移（多帧平均）

        Args:
            num_samples: 采样帧数
            min_quality: 最小检测质量阈值
            warmup_frames: 预热帧数（丢弃前几帧，让相机稳定）

        Returns:
            (offset_x, offset_y, quality) 或 None（如果检测失败）
        """
        # 预热：丢弃前几帧
        for i in range(warmup_frames):
            _ = self.camera.read()
            time.sleep(0.03)

        offsets_x = []
        offsets_y = []
        qualities = []

        for i in range(num_samples):
            img = self.camera.read()
            if img is None:
                continue

            state = self.detector.detect_dual_marker_state(img)
            if state.workpiece_detected and state.slot_detected:
                # 检查检测质量
                if hasattr(state, 'alignment_quality') and state.alignment_quality < min_quality:
                    print(f"  [!] 帧{i+1}: 检测质量低 ({state.alignment_quality:.2f}), 跳过")
                    continue

                offsets_x.append(state.offset_x)
                offsets_y.append(state.offset_y)
                qualities.append(state.alignment_quality if hasattr(state, 'alignment_quality') else 1.0)

            time.sleep(0.05)

        if len(offsets_x) < 3:
            return None

        # 使用中值滤波去除异常值
        offset_x = np.median(offsets_x)
        offset_y = np.median(offsets_y)
        quality = np.mean(qualities)

        return offset_x, offset_y, quality

    def verify_sensitivity_direction(self, joint_idx: int = 8, move_deg: float = 2.0, num_samples: int = 5) -> bool:
        """
        验证灵敏度方向是否正确

        通过移动关节并观察像素变化方向来验证灵敏度符号是否正确。
        使用多帧平均提高检测稳定性。

        Args:
            joint_idx: 要测试的关节索引（默认joint_8 = right_arm_joint_2 肩部俯仰）
            move_deg: 移动角度（默认1度，减少Z方向影响）
            num_samples: 每次测量的采样帧数（用于平均）

        Returns:
            是否正确（True=方向正确，False=需要翻转）
        """
        if self.passive_mode:
            print("✗ 被动模式下无法验证")
            return False

        print("\n" + "="*60)
        print("灵敏度方向验证")
        print("="*60)
        print(f"\n测试关节: {self.joint_names.get(joint_idx, f'joint_{joint_idx}')} (索引{joint_idx})")
        print(f"移动角度: {move_deg}度")
        print(f"采样帧数: {num_samples} (用于多帧平均)")

        # 获取当前灵敏度
        current_joints = self.get_joint_states()
        if current_joints is None:
            print("✗ 无法获取关节状态")
            return False

        sensitivities = self.get_interpolated_sensitivities(current_joints)
        test_sens = None
        for s in sensitivities:
            if s.joint_idx == joint_idx:
                test_sens = s
                break

        if test_sens is None:
            print(f"✗ 未找到关节 {joint_idx} 的灵敏度数据")
            return False

        print(f"\n当前灵敏度: X={test_sens.pixel_dx_per_deg:.2f}, Y={test_sens.pixel_dy_per_deg:.2f} px/deg")

        # 采集初始图像（多帧平均）
        print("\n[1] 采集初始图像（多帧平均）...")
        result1 = self._get_stable_pixel_offset(num_samples)
        if result1 is None:
            print("✗ 初始检测不稳定，请确保标记清晰可见")
            return False

        initial_offset_x, initial_offset_y, quality1 = result1
        print(f"  初始像素偏移: X={initial_offset_x:.1f}, Y={initial_offset_y:.1f} (质量={quality1:.2f})")

        # 移动关节
        print(f"\n[2] 移动关节 {move_deg}度...")
        current_angle = current_joints[joint_idx]
        target_angle = current_angle + move_deg

        # 平滑移动
        success = self._smooth_move_single_joint_for_test(joint_idx, target_angle)
        if not success:
            print("✗ 移动失败")
            return False

        time.sleep(0.5)

        # 采集移动后图像（多帧平均）
        print("\n[3] 采集移动后图像（多帧平均）...")
        result2 = self._get_stable_pixel_offset(num_samples)
        if result2 is None:
            print("✗ 移动后检测不稳定")
            # 移回原位
            self._smooth_move_single_joint_for_test(joint_idx, current_angle)
            return False

        final_offset_x, final_offset_y, quality2 = result2
        print(f"  移动后像素偏移: X={final_offset_x:.1f}, Y={final_offset_y:.1f} (质量={quality2:.2f})")

        # 计算实际像素变化
        actual_dx = final_offset_x - initial_offset_x
        actual_dy = final_offset_y - initial_offset_y

        # 计算预期像素变化
        expected_dx = test_sens.pixel_dx_per_deg * move_deg
        expected_dy = test_sens.pixel_dy_per_deg * move_deg

        print(f"\n结果分析:")
        print(f"  实际像素变化: X={actual_dx:.1f}, Y={actual_dy:.1f}")
        print(f"  预期像素变化: X={expected_dx:.1f}, Y={expected_dy:.1f}")

        # 计算变化幅度比例（用于判断检测是否合理）
        expected_mag = np.sqrt(expected_dx**2 + expected_dy**2)
        actual_mag = np.sqrt(actual_dx**2 + actual_dy**2)

        if expected_mag > 5:  # 只有预期变化足够大时才检查比例
            ratio = actual_mag / expected_mag
            print(f"  变化幅度比例: {ratio:.2f} (实际/预期)")

            if ratio < 0.2:
                print("  [!] 警告: 实际变化过小，可能检测不稳定")
            elif ratio > 3.0:
                print("  [!] 警告: 实际变化过大，可能检测到错误标记")

        # 判断方向是否正确
        x_correct = (actual_dx * expected_dx) > 0  # 同号表示方向正确
        y_correct = (actual_dy * expected_dy) > 0

        print(f"\n方向判断:")
        print(f"  X方向: {'✓ 正确' if x_correct else '✗ 需要翻转'}")
        print(f"  Y方向: {'✓ 正确' if y_correct else '✗ 需要翻转'}")

        # 移回原位
        print(f"\n[4] 返回原位...")
        self._smooth_move_single_joint_for_test(joint_idx, current_angle)
        time.sleep(0.5)

        if not x_correct or not y_correct:
            print("\n" + "!"*60)
            print("! 警告: 灵敏度方向不正确，可能导致对齐失败")
            print("! 解决方法:")
            print("!   1. 在当前姿态下重新标定")
            print("!   2. 或在 ARM_CONFIGS 中设置 camera_flip")
            print("!"*60)
            return False

        print("\n✓ 灵敏度方向正确")
        return True

    def _smooth_move_single_joint_for_test(self, joint_idx: int, target_angle: float, steps: int = 10) -> bool:
        """平滑移动单个关节（用于测试）"""
        current_joints = self.get_joint_states()
        if current_joints is None:
            return False

        initial_angle = current_joints[joint_idx]

        for step in range(1, steps + 1):
            alpha = step / steps
            alpha = alpha * alpha * (3 - 2 * alpha)

            current_angle = initial_angle + (target_angle - initial_angle) * alpha

            # 构建action
            action = {f"{name}.pos": float(current_joints[i] if i != joint_idx else current_angle)
                      for i, name in enumerate(self.robot.observation_joint_names)
                      if i < len(current_joints)}
            self.robot.send_action(action)
            time.sleep(0.05)

        return True

    # ==================== 方案B: DH参数接口 ====================

    def set_dh_parameters(self, dh_params: List[Dict]):
        """
        设置DH参数 (方案B)

        Args:
            dh_params: DH参数列表
        """
        self.arm_config.dh_params = dh_params
        self.kinematics_model = DHKinematicsModel(dh_params)
        print("✓ 已更新DH参数")
        self._save_arm_config()

    def _save_arm_config(self):
        """保存手臂配置 (包括DH参数)"""
        # TODO: 实现配置持久化
        pass

    def set_target_offset(self, offset_x: float, offset_y: float):
        """
        设置对齐目标偏移（像素）

        当工件正确放置时，工件标点中心相对于卡槽标点中心的偏移量。

        由于腕部相机45度倾斜，高度变化会产生透视效应。
        系统会记录当前高度，对齐时自动进行透视补偿。

        Args:
            offset_x: X方向目标偏移（像素）
            offset_y: Y方向目标偏移（像素）
        """
        self.target_offset_x = offset_x
        self.target_offset_y = offset_y

        print(f"✓ 对齐目标偏移已设置: ({offset_x:.1f}, {offset_y:.1f}) 像素")

        # 记录当前高度（用于透视补偿）
        setup_height = None
        if self.z_controller is not None and self.camera2 is not None:
            image1 = self.camera.read()
            image2 = self.camera2.read()
            if image1 is not None and image2 is not None:
                estimate = self.z_controller.estimate_z(image1, image2, self.detector.workpiece_color)
                if estimate.confidence > 0.3:
                    setup_height = estimate.z
                    self._setup_height = estimate.z  # 设置偏移量时的高度
                    print(f"✓ 已记录设置高度: {estimate.z:.1f}mm（用于透视补偿）")

        # 记录当前关节状态
        current_joints = self.get_joint_states()
        if current_joints is not None:
            self._calibration_joint_states = current_joints.copy()
            if hasattr(self.robot, 'observation_joint_names'):
                self._calibration_joint_dict = {
                    name: float(current_joints[i])
                    for i, name in enumerate(self.robot.observation_joint_names)
                    if i < len(current_joints)
                }
            print(f"✓ 已记录当前关节状态")

        # 显示透视效应信息
        print("\n" + "-"*40)
        print("透视效应说明：")
        print(f"  相机角度: 约45度倾斜")
        print(f"  设置高度: {setup_height:.1f}mm" if setup_height else "  设置高度: 未知")
        print("  对齐时会根据高度差自动补偿透视效应")
        print("-"*40)

    def compute_perspective_offset(self, height_diff_mm: float, camera_angle_deg: float = 45.0) -> Tuple[float, float]:
        """
        计算透视效应导致的偏移量

        当相机有倾斜角度时，高度变化会导致卡槽在图像中的位置偏移。

        原理：
        - 相机45度向下看
        - 高度增加时，卡槽在图像中会向相机倾斜方向偏移
        - 偏移量 = 高度差 × tan(相机角度) × 像素比例

        Args:
            height_diff_mm: 高度差(mm)，正值表示对齐时比设置时高
            camera_angle_deg: 相机倾斜角度(度)，默认45度

        Returns:
            (offset_x_px, offset_y_px) 需要补偿的像素偏移
        """
        camera_angle_rad = np.deg2rad(camera_angle_deg)
        tan_angle = np.tan(camera_angle_rad)

        # 物理偏移量（沿相机倾斜方向的物理距离变化）
        physical_offset_mm = height_diff_mm * tan_angle

        # 转换为像素偏移
        pixel_offset = physical_offset_mm / self.pixel_to_mm_ratio

        # 根据相机倾斜方向确定偏移方向
        # 使用配置的方向
        x_dir, y_dir = self._camera_tilt_direction

        offset_x = pixel_offset * x_dir
        offset_y = pixel_offset * y_dir

        return offset_x, offset_y

    def calibrate_perspective_direction(self) -> Tuple[float, float]:
        """
        校准透视效应方向

        通过在两个不同高度检测偏移量来确定透视偏移方向。

        流程：
        1. 在低位置检测并记录偏移量
        2. 抬高到高位置
        3. 再次检测偏移量
        4. 计算偏移方向

        Returns:
            (x_direction, y_direction) 透视偏移方向
        """
        print("\n" + "="*50)
        print("透视效应方向校准")
        print("="*50)
        print("""
此功能将帮助确定相机倾斜导致的透视偏移方向。

操作步骤：
  1. 在低位置检测当前偏移量
  2. 抬高夹爪约100-150mm
  3. 再次检测偏移量
  4. 系统自动计算透视偏移方向
""")

        input("准备好后按 Enter 开始...")

        # 低位置检测
        print("\n[步骤1] 低位置检测")
        print("请保持夹爪在低位置（设置偏移量的高度）")
        input("准备好后按 Enter...")

        state_low = self._get_stable_pixel_offset(num_samples=5)
        if state_low is None:
            print("✗ 低位置检测失败")
            return self._camera_tilt_direction

        low_offset_x, low_offset_y, _ = state_low
        print(f"  低位置偏移: ({low_offset_x:.1f}, {low_offset_y:.1f}) px")

        # 记录低位置高度
        low_height = self.get_current_height()
        print(f"  低位置高度: {low_height:.1f}mm" if low_height else "  低位置高度: 无法获取")

        # 高位置检测
        print("\n[步骤2] 高位置检测")
        print("请抬高夹爪约100-150mm")
        input("抬高后按 Enter...")

        state_high = self._get_stable_pixel_offset(num_samples=5)
        if state_high is None:
            print("✗ 高位置检测失败")
            return self._camera_tilt_direction

        high_offset_x, high_offset_y, _ = state_high
        print(f"  高位置偏移: ({high_offset_x:.1f}, {high_offset_y:.1f}) px")

        high_height = self.get_current_height()
        print(f"  高位置高度: {high_height:.1f}mm" if high_height else "  高位置高度: 无法获取")

        # 计算偏移变化
        delta_x = high_offset_x - low_offset_x
        delta_y = high_offset_y - low_offset_y

        height_diff = (high_height - low_height) if (high_height and low_height) else 150.0

        print(f"\n[分析结果]")
        print(f"  高度变化: {height_diff:.1f}mm")
        print(f"  X偏移变化: {delta_x:.1f}px")
        print(f"  Y偏移变化: {delta_y:.1f}px")

        # 确定方向
        if abs(delta_x) > abs(delta_y):
            x_dir = 1.0 if delta_x > 0 else -1.0
            y_dir = 0.0
            print(f"  主要偏移方向: X{'正' if x_dir > 0 else '负'}方向")
        else:
            x_dir = 0.0
            y_dir = 1.0 if delta_y > 0 else -1.0
            print(f"  主要偏移方向: Y{'正' if y_dir > 0 else '负'}方向")

        # 更新配置
        self._camera_tilt_direction = (x_dir, y_dir)

        print(f"\n✓ 透视偏移方向已设置: ({x_dir:.1f}, {y_dir:.1f})")
        print("  注意: 此设置仅在当前会话有效")

        return self._camera_tilt_direction

    def get_perspective_compensated_offset(self, current_height: float = None) -> Tuple[float, float]:
        """
        获取透视补偿后的目标偏移量

        Args:
            current_height: 当前高度(mm)，None则自动获取

        Returns:
            (compensated_offset_x, compensated_offset_y) 补偿后的目标偏移
        """
        if not hasattr(self, '_setup_height') or self._setup_height is None:
            # 没有记录设置高度，不进行补偿
            return self.target_offset_x, self.target_offset_y

        if current_height is None:
            current_height = self.get_current_height()

        if current_height is None:
            # 无法获取当前高度，不进行补偿
            print("  ⚠ 无法获取当前高度，跳过透视补偿")
            return self.target_offset_x, self.target_offset_y

        # 计算高度差
        height_diff = current_height - self._setup_height

        if abs(height_diff) < 10:  # 高度差小于10mm，忽略
            return self.target_offset_x, self.target_offset_y

        # 计算透视补偿
        comp_x, comp_y = self.compute_perspective_offset(height_diff)

        compensated_x = self.target_offset_x + comp_x
        compensated_y = self.target_offset_y + comp_y

        print(f"\n[透视补偿]")
        print(f"  设置高度: {self._setup_height:.1f}mm, 当前高度: {current_height:.1f}mm")
        print(f"  高度差: {height_diff:.1f}mm")
        print(f"  透视偏移: ({comp_x:.1f}, {comp_y:.1f}) px")
        print(f"  补偿后目标: ({compensated_x:.1f}, {compensated_y:.1f}) px")

        return compensated_x, compensated_y

    def get_current_offset(self) -> Tuple[float, float]:
        """
        获取当前工件和卡槽的偏移

        Returns:
            (offset_x, offset_y) 卡槽中心 - 工件中心
        """
        image = self.camera.read()
        state = self.detector.detect_dual_marker_state(image)

        if state.workpiece_detected and state.slot_detected:
            return state.offset_x, state.offset_y
        return 0.0, 0.0

    # ==================== XY对齐 (使用标定数据) ====================

    def compute_joint_adjustments(self, pixel_error_x: float, pixel_error_y: float,
                                  current_joints: np.ndarray) -> Dict[int, float]:
        """
        根据像素误差计算关节调整量

        使用方案A (多点标定插值) 或方案B (雅可比)

        Args:
            pixel_error_x: X方向像素误差
            pixel_error_y: Y方向像素误差
            current_joints: 当前关节角度

        Returns:
            {joint_idx: adjustment_degrees}
        """
        adjustments = {}

        # 优先使用方案B (雅可比)
        if self.arm_config.dh_params and isinstance(self.kinematics_model, DHKinematicsModel):
            return self._compute_adjustments_jacobian(pixel_error_x, pixel_error_y, current_joints)

        # 使用方案A (多点标定插值)
        return self._compute_adjustments_interpolation(pixel_error_x, pixel_error_y, current_joints)

    def _compute_single_joint_with_weights(self, pixel_error_x: float, pixel_error_y: float,
                                            sensitivities: list, position_weights: dict) -> np.ndarray:
        """
        带权重的单关节方法（伪逆失败时的fallback）

        选择 灵敏度×权重 最大的关节来控制每个方向
        """
        # 计算等效灵敏度（灵敏度 × 位置权重）
        joint_indices = [s.joint_idx for s in sensitivities]

        # 找X方向最优关节
        best_x_idx = 0
        best_x_effective = 0
        for i, s in enumerate(sensitivities):
            weight = position_weights.get(s.joint_idx, 0.5)
            effective_sens = abs(s.pixel_dx_per_deg) * weight
            if effective_sens > best_x_effective:
                best_x_effective = effective_sens
                best_x_idx = i

        # 找Y方向最优关节
        best_y_idx = 0
        best_y_effective = 0
        for i, s in enumerate(sensitivities):
            weight = position_weights.get(s.joint_idx, 0.5)
            effective_sens = abs(s.pixel_dy_per_deg) * weight
            if effective_sens > best_y_effective:
                best_y_effective = effective_sens
                best_y_idx = i

        # 计算调整量
        n_joints = len(sensitivities)
        delta_angles = np.zeros(n_joints)

        if best_x_effective > 0.01:
            s = sensitivities[best_x_idx]
            delta_x = -pixel_error_x / s.pixel_dx_per_deg
            delta_angles[best_x_idx] += delta_x
            print(f"      X控制: joint_{s.joint_idx} (等效灵敏度={best_x_effective:.2f})")

        if best_y_effective > 0.01:
            s = sensitivities[best_y_idx]
            delta_y = -pixel_error_y / s.pixel_dy_per_deg
            delta_angles[best_y_idx] += delta_y
            print(f"      Y控制: joint_{s.joint_idx} (等效灵敏度={best_y_effective:.2f})")

        return delta_angles

    def _compute_adjustments_interpolation(self, pixel_error_x: float, pixel_error_y: float,
                                           current_joints: np.ndarray) -> Dict[int, float]:
        """方案A: 使用伪逆 + 位置权重计算多关节组合调整量"""
        sensitivities = self.get_interpolated_sensitivities(current_joints)

        if not sensitivities:
            # 没有标定数据，无法准确计算调整量
            print("\n" + "!"*60)
            print("! 严重警告: 无标定数据，无法准确计算关节调整量")
            print("! 建议先运行标定: 主菜单 -> 4 (标定) -> 2 或 3 (关节灵敏度标定)")
            print("! 当前返回空调整，对齐可能失败")
            print("!"*60)
            return {}

        # 调试：打印计算过程
        print(f"    像素误差: X={pixel_error_x:.1f}px, Y={pixel_error_y:.1f}px")
        print(f"    相机翻转状态: X={self._camera_flip_x}, Y={self._camera_flip_y}")

        # 打印每个关节的灵敏度
        print(f"    关节灵敏度数据:")
        for s in sensitivities:
            print(f"      joint_{s.joint_idx}: dx={s.pixel_dx_per_deg:.2f}, dy={s.pixel_dy_per_deg:.2f} px/deg")

        # ========== 位置贡献权重 ==========
        # 手腕关节(joint_11, 12)主要控制姿态，对末端位置贡献小
        # 其他手臂关节主要控制位置，贡献大
        POSITION_WEIGHTS = {
            7: 1.0,   # joint_1 (底座旋转) - 位置贡献大
            8: 1.0,   # joint_2 (肩部俯仰) - 位置贡献大
            9: 1.0,   # joint_3 (肩部侧摆) - 位置贡献大
            10: 0.8,  # joint_4 (前臂俯仰) - 位置贡献中
            11: 0.3,  # joint_5 (腕部俯仰) - 位置贡献小
            12: 0.3,  # joint_6 (腕部旋转) - 位置贡献小
            14: 0.6,  # trunk_1 (躯干旋转) - 位置贡献中
        }

        # ========== 构建雅可比矩阵 ==========
        # J: 2 x N 矩阵
        # J[0, i] = 关节i对像素X的影响 (pixel_dx_per_deg)
        # J[1, i] = 关节i对像素Y的影响 (pixel_dy_per_deg)

        joint_indices = [s.joint_idx for s in sensitivities]
        n_joints = len(joint_indices)

        J = np.zeros((2, n_joints))
        W = np.zeros((n_joints, n_joints))  # 权重对角矩阵

        for i, s in enumerate(sensitivities):
            J[0, i] = s.pixel_dx_per_deg
            J[1, i] = s.pixel_dy_per_deg
            # 获取位置权重，默认0.5
            W[i, i] = POSITION_WEIGHTS.get(s.joint_idx, 0.5)

        # ========== 伪逆求解 ==========
        # 目标：J @ delta = -error
        # 加入权重：J @ W @ delta' = -error
        # 解：delta' = (J @ W)^+ @ (-error)

        JW = J @ W  # 加权雅可比矩阵

        # 目标误差向量
        error = np.array([pixel_error_x, pixel_error_y])

        # 使用伪逆求解
        try:
            JW_pinv = np.linalg.pinv(JW)
            delta_angles = JW_pinv @ (-error)

            # 检查解是否合理（防止数值不稳定导致过大的解）
            max_delta = np.max(np.abs(delta_angles))
            if max_delta > 5.0:  # 单次调整超过5度说明数值不稳定
                print(f"    伪逆解过大 (max={max_delta:.1f}°)，使用带权重的单关节方法")
                delta_angles = self._compute_single_joint_with_weights(
                    pixel_error_x, pixel_error_y, sensitivities, POSITION_WEIGHTS
                )
        except np.linalg.LinAlgError:
            print("    伪逆求解失败，使用带权重的单关节方法")
            delta_angles = self._compute_single_joint_with_weights(
                pixel_error_x, pixel_error_y, sensitivities, POSITION_WEIGHTS
            )

        # 应用增益和限幅
        delta_angles = delta_angles * self.gain
        delta_angles = np.clip(delta_angles, -2.0, 2.0)

        # 转换为调整字典
        adjustments = {}
        print(f"    多关节组合调整:")
        for i, (joint_idx, delta) in enumerate(zip(joint_indices, delta_angles)):
            if abs(delta) > 0.01:  # 忽略微小调整
                adjustments[joint_idx] = delta
                weight = W[i, i]
                print(f"      joint_{joint_idx}: {delta:.2f}° (权重={weight})")

        # 验证预期效果
        expected_dx = sum(J[0, i] * delta_angles[i] for i in range(n_joints))
        expected_dy = sum(J[1, i] * delta_angles[i] for i in range(n_joints))
        print(f"    预期像素变化: X={expected_dx:.1f}px, Y={expected_dy:.1f}px")

        return adjustments

    def _compute_adjustments_jacobian(self, pixel_error_x: float, pixel_error_y: float,
                                      current_joints: np.ndarray) -> Dict[int, float]:
        """方案B: 使用雅可比计算调整量"""
        # TODO: 实现雅克比方法
        # 1. 获取末端位姿
        # 2. 计算雅可比矩阵
        # 3. 求解关节速度/位移

        raise NotImplementedError("雅可比方法待实现，请提供DH参数")

    def compute_rotation_adjustment(self, rotation_error: float, current_joints: np.ndarray) -> Dict[int, float]:
        """
        计算旋转调整量

        使用能产生旋转的关节（通常是手腕关节）来纠正旋转误差。

        Args:
            rotation_error: 旋转误差（度），正值为逆时针偏差
            current_joints: 当前关节角度

        Returns:
            {joint_idx: adjustment_degrees}
        """
        adjustments = {}
        abs_rot_error = abs(rotation_error)

        if abs_rot_error < self.tolerance_deg:
            return adjustments

        # 选择旋转关节：根据手臂不同，通常是某个手腕关节
        # 右臂使用 joint 12 (right_arm_joint_6 - 手腕旋转)
        # 左臂使用 joint 5 (left_arm_joint_6 - 手腕旋转)
        if self.arm == 'right':
            rotation_joint = 12
        else:
            rotation_joint = 5

        # 计算调整量（反向调整）
        delta = -rotation_error * self.rotation_gain

        # 限制单次调整量
        delta = np.clip(delta, -self.max_rotation_adjust, self.max_rotation_adjust)

        adjustments[rotation_joint] = delta
        return adjustments

    def apply_joint_adjustments(self, adjustments: Dict[int, float], rotation_adjustments: Dict[int, float] = None) -> bool:
        """应用关节调整（位置+旋转）"""
        if self.passive_mode:
            print("✗ 被动模式下无法应用关节调整")
            return False

        joints = self.get_joint_states()
        if joints is None or len(joints) < 16:
            return False

        target = joints.copy()
        for jidx, delta in adjustments.items():
            target[jidx] += delta
            print(f"[DEBUG] apply_joint_adjustments: target[{jidx}] = {joints[jidx]:.4f} + {delta:.4f} = {target[jidx]:.4f}")

        # 应用旋转调整
        if rotation_adjustments:
            for jidx, delta in rotation_adjustments.items():
                target[jidx] += delta
                print(f"[DEBUG] 旋转调整: target[{jidx}] += {delta:.4f}")

        # 平滑移动
        self._smooth_move_all_joints(target)
        return True

    def _smooth_move_all_joints(self, target_joints: np.ndarray, steps: int = None):
        """平滑移动所有关节（保持夹爪位置不变）"""
        if self.passive_mode:
            print("✗ 被动模式下无法移动关节")
            return False

        if steps is None:
            steps = self.smooth_steps

        print(f"[DEBUG] _smooth_move_all_joints: steps={steps}, smooth_delay={self.smooth_delay}")
        print(f"[DEBUG] observation_joint_names: {self.robot.observation_joint_names}")

        current_joints = self.get_joint_states()

        if current_joints is None or len(current_joints) < 16:
            print(f"[DEBUG] get_joint_states failed: current_joints={current_joints}")
            return False

        # 保存夹爪初始位置，在移动过程中保持不变
        gripper_initial_pos = current_joints[self.arm_config.gripper_idx]
        gripper_target_pos = target_joints[self.arm_config.gripper_idx]

        # DEBUG: 打印目标关节值
        print(f"[DEBUG] 开始平滑移动:")
        print(f"  current_joints[7] (right_arm_joint_1) = {current_joints[7]:.4f}")
        print(f"  target_joints[7] = {target_joints[7]:.4f}")
        print(f"  delta = {target_joints[7] - current_joints[7]:.4f}")
        print(f"  arm_config.gripper_idx = {self.arm_config.gripper_idx}")

        for step in range(1, steps + 1):
            alpha = step / steps
            alpha = alpha * alpha * (3 - 2 * alpha)  # ease-in-out

            interp = current_joints * (1 - alpha) + target_joints * alpha
            # 保持夹爪在初始位置（不被插值改变）
            interp[self.arm_config.gripper_idx] = gripper_initial_pos

            # DEBUG: 打印每步的插值值
            if step <= 3 or step == steps:
                print(f"[DEBUG] step {step}/{steps}: alpha={alpha:.4f}, interp[7]={interp[7]:.4f}")

            # 转换为正确的action格式: {'joint_name.pos': value}
            action = {f"{name}.pos": float(interp[i])
                      for i, name in enumerate(self.robot.observation_joint_names)
                      if i < len(interp)}
            self.robot.send_action(action)
            time.sleep(self.smooth_delay)

        print(f"[DEBUG] 平滑移动完成")
        return True

    # ==================== 对齐流程 ====================

    def align_xy(self, tolerance_mm: float = None, use_secondary_camera: bool = True) -> bool:
        """
        XY对齐 (使用标定数据)

        P1改进:
        - 支持退化模式（标记不足时仍尝试工作）
        - 支持副相机融合检测
        - 支持预测位置功能
        - 支持透视效应补偿

        Args:
            tolerance_mm: 目标精度
            use_secondary_camera: 是否使用副相机辅助检测

        Returns:
            是否对齐成功
        """
        if tolerance_mm is None:
            tolerance_mm = self.tolerance_mm

        # 获取透视补偿后的目标偏移
        if hasattr(self, '_use_perspective_compensation') and self._use_perspective_compensation:
            target_offset_x, target_offset_y = self.get_perspective_compensated_offset()
        else:
            target_offset_x, target_offset_y = self.target_offset_x, self.target_offset_y

        print(f"\nXY对齐 - 目标精度: {tolerance_mm}mm")
        print(f"目标偏移: ({target_offset_x:.1f}, {target_offset_y:.1f}) 像素")

        # 检查标定数据
        if not self.calibration_points:
            print("警告: 无多点标定数据，精度可能受限")
            print("建议运行 calibrate_all_joints() 进行标定")

        # 异常恢复参数
        consecutive_failures = 0
        max_consecutive_failures = self.max_occlusion_frames  # 使用配置的遮挡帧数
        recovery_attempts = 0
        max_recovery_attempts = 3  # 增加恢复尝试次数
        degraded_warnings = 0  # 退化模式警告计数

        # 视频显示窗口
        if self.show_alignment_video:
            cv2.namedWindow(self._alignment_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._alignment_window_name, 640, 480)  # 减小窗口尺寸加速显示

        # 检查标定数据是否适用于当前姿态
        if self.calibration_points:
            current_joints = self.get_joint_states()
            if current_joints is not None:
                sensitivities = self.get_interpolated_sensitivities(current_joints)
                if sensitivities:
                    # 检查插值权重，如果都很低说明姿态差异大
                    weights = [s.confidence for s in sensitivities if hasattr(s, 'confidence') and s.confidence]
                    if len(sensitivities) > 0:
                        # 简单检查：如果没有高置信度的灵敏度，给出警告
                        print(f"  检测到 {len(sensitivities)} 个关节灵敏度数据")
                        avg_weight = sum(getattr(s, 'interpolation_weight', 1.0) for s in sensitivities) / len(sensitivities) if sensitivities else 0
                        if avg_weight < 0.3:
                            print("  ⚠ 警告: 当前姿态与标定姿态差异较大，建议在当前姿态下重新标定")
                            print("  提示: 主菜单 -> 4 (标定) -> 3 (关节灵敏度标定-自动)")

        # 记录初始高度（用于XY对齐时保持高度稳定）
        initial_height = self.get_current_height()
        if initial_height is not None:
            print(f"  初始高度: {initial_height:.1f}mm (将在XY对齐时保持)")
        height_correction_interval = 3  # 每3次调整后检查高度
        height_tolerance_mm = 3.0  # 高度变化容忍阈值

        for i in range(self.max_iterations):
            print(f"\n[对齐 {i+1}/{self.max_iterations}]")

            # P1: 获取图像并尝试双相机融合
            image = self.camera.read()
            image2 = None
            if use_secondary_camera and self.camera2 is not None:
                image2 = self.camera2.read()

            # P1: 使用双相机融合检测
            if image2 is not None:
                state = self.detector.detect_with_secondary_camera(image, image2)
            else:
                state = self.detector.detect_dual_marker_state(image)

            wp = state.workpiece_marker_count
            sl = state.slot_marker_count

            # P1: 显示退化模式警告
            if state.degraded_mode:
                degraded_warnings += 1
                print(f"  ⚠ 退化模式: {state.degraded_reason}")
            else:
                degraded_warnings = 0

            print(f"  工件: {wp}/3, 卡槽: {sl}/3")

            # 视频显示（检测失败时也显示）
            if self.show_alignment_video:
                vis = self.visualize_alignment(image, state, None, 0.0, i, self.max_iterations)
                cv2.imshow(self._alignment_window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("  用户中断对齐")
                    cv2.destroyWindow(self._alignment_window_name)
                    return False

            if not state.workpiece_detected or not state.slot_detected:
                consecutive_failures += 1
                print(f"  标记不完整 ({consecutive_failures}/{max_consecutive_failures})")

                # 改进的遮挡恢复：只要有历史预测就尝试使用
                if self.use_prediction and self._has_valid_prediction():
                    print("  [遮挡恢复] 使用预测位置继续控制...")
                    predicted_state = self._create_predicted_state(state)
                    if predicted_state is not None:
                        state = predicted_state
                        print(f"  ✓ 预测偏移: ({state.offset_x:.1f}, {state.offset_y:.1f})")
                        # 不continue，继续执行控制
                    else:
                        # 预测失败，尝试物理恢复
                        if consecutive_failures >= max_consecutive_failures:
                            recovery_attempts = self._handle_detection_failure(
                                state, consecutive_failures, max_consecutive_failures,
                                recovery_attempts, max_recovery_attempts
                            )
                        if consecutive_failures >= max_consecutive_failures * 2:
                            print("  ✗ 遮挡时间过长，对齐失败")
                            cv2.destroyWindow(self._alignment_window_name)
                            return False
                        continue
                else:
                    # 无预测数据，等待恢复
                    if consecutive_failures >= max_consecutive_failures:
                        recovery_attempts = self._handle_detection_failure(
                            state, consecutive_failures, max_consecutive_failures,
                            recovery_attempts, max_recovery_attempts
                        )
                    if consecutive_failures >= max_consecutive_failures * 2:
                        print("  ✗ 持续检测失败，对齐失败")
                        cv2.destroyWindow(self._alignment_window_name)
                        return False
                    continue

            # 成功检测，重置计数器
            consecutive_failures = 0

            # P1: 记录历史偏移（用于预测）
            self._record_offset(state.offset_x, state.offset_y)

            # 计算误差：当前偏移减去目标偏移（使用透视补偿后的目标）
            current_offset_x = state.offset_x - target_offset_x
            current_offset_y = state.offset_y - target_offset_y

            mm_x = current_offset_x * self.pixel_to_mm_ratio
            mm_y = current_offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)

            print(f"  当前偏移: ({state.offset_x:.1f}, {state.offset_y:.1f}) px")
            if target_offset_x != 0 or target_offset_y != 0:
                print(f"  目标偏移: ({target_offset_x:.1f}, {target_offset_y:.1f}) px")

            # P1: 退化模式下显示精度警告
            if state.degraded_mode:
                print(f"  ⚠ 精度可能降低 (标记不足)")
            print(f"  位置误差: ({mm_x:.2f}, {mm_y:.2f})mm, 总: {error_mm:.2f}mm")
            print(f"  旋转误差: {state.rotation_error:.2f}deg")

            # 检查位置和旋转误差
            # P1: 退化模式下使用更宽松的容差
            actual_tolerance = tolerance_mm * 1.5 if state.degraded_mode else tolerance_mm
            actual_rot_tolerance = self.tolerance_deg * 1.5 if state.degraded_mode else self.tolerance_deg

            position_ok = error_mm < actual_tolerance
            rotation_ok = abs(state.rotation_error) < actual_rot_tolerance

            if position_ok and rotation_ok:
                print(f"\n✓ 对齐完成")
                print(f"  位置误差: {error_mm:.2f}mm < {actual_tolerance:.2f}mm")
                print(f"  旋转误差: {abs(state.rotation_error):.2f}deg < {actual_rot_tolerance:.2f}deg")
                if state.degraded_mode:
                    print(f"  ⚠ 本次对齐在退化模式下完成，建议检查实际效果")
                # 关闭视频窗口
                if self.show_alignment_video:
                    cv2.destroyWindow(self._alignment_window_name)
                return True

            # 获取当前关节状态
            current_joints = self.get_joint_states()

            if current_joints is None:
                print("✗ 无法获取关节状态")
                continue

            # 计算关节调整量 (使用标定数据，传入相对误差)
            adjustments = self.compute_joint_adjustments(
                current_offset_x, current_offset_y, current_joints
            )

            # 检查是否有有效的调整量
            if not adjustments and not state.degraded_mode:
                print("  ✗ 无法计算调整量，请先完成标定")
                print("  提示: 主菜单 -> 4 (标定) -> 2 或 3 (关节灵敏度标定)")
                # 继续尝试，可能退化模式能工作
                continue

            # 计算旋转调整量 (退化模式下跳过旋转校正)
            rotation_adjustments = {}
            if not state.degraded_mode and sl >= 2:
                rotation_adjustments = self.compute_rotation_adjustment(
                    state.rotation_error, current_joints
                )

            if adjustments:
                print(f"  位置调整: {adjustments}")
            if rotation_adjustments:
                print(f"  旋转调整: {rotation_adjustments}")

            # 视频显示（显示调整信息）
            if self.show_alignment_video:
                vis = self.visualize_alignment(image, state, adjustments, error_mm, i, self.max_iterations)
                cv2.imshow(self._alignment_window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("  用户中断对齐")
                    cv2.destroyWindow(self._alignment_window_name)
                    return False

            # 保存调整前的原始偏移（用于验证）
            pre_state_offset_x = state.offset_x
            pre_state_offset_y = state.offset_y

            # 应用调整 (只有当有调整量时才执行)
            min_quality = 0.2 if state.degraded_mode else 0.3  # 退化模式下降低质量阈值
            if adjustments and state.alignment_quality > min_quality:
                self.apply_joint_adjustments(adjustments, rotation_adjustments)

            time.sleep(self.settle_time)

            # 高度检查：定期检查高度变化并补偿
            if initial_height is not None and (i + 1) % height_correction_interval == 0:
                current_height = self.get_current_height()
                if current_height is not None:
                    height_change = initial_height - current_height  # 正值表示下降了
                    if abs(height_change) > height_tolerance_mm:
                        print(f"  [高度监控] 检测到高度变化: {height_change:.1f}mm")
                        # 使用XYZ协调控制恢复高度
                        # 注意：z_delta正值表示下降，所以这里用负值表示上升
                        height_correction = -height_change  # 恢复到初始高度
                        correction_adjustments = self.compute_height_correction(height_correction)
                        if correction_adjustments:
                            print(f"  [高度补偿] 执行高度恢复: {abs(height_change):.1f}mm")
                            joints = self.get_joint_states()
                            if joints is not None:
                                target = joints.copy()
                                for jidx, angle_delta in correction_adjustments.items():
                                    if jidx < len(target):
                                        target[jidx] += angle_delta
                                self.move_to_joint_positions(target)
                                time.sleep(self.settle_time)
                    else:
                        print(f"  [高度监控] 高度稳定 (变化: {height_change:.1f}mm < {height_tolerance_mm}mm)")

            # DEBUG: 验证调整效果 - 获取新的像素误差并与预期比较
            if adjustments and state.alignment_quality > min_quality:
                new_image = self.camera.read()
                new_state = self.detector.detect_dual_marker_state(new_image)
                if new_state.workpiece_detected and new_state.slot_detected:
                    # 用原始偏移计算实际变化
                    actual_dx = new_state.offset_x - pre_state_offset_x
                    actual_dy = new_state.offset_y - pre_state_offset_y
                    print(f"  [验证] 实际像素变化: X={actual_dx:.1f}px, Y={actual_dy:.1f}px")

                    # 计算新的误差（使用透视补偿后的目标）
                    new_error_x = new_state.offset_x - target_offset_x
                    new_error_y = new_state.offset_y - target_offset_y
                    new_error_mag = np.sqrt(new_error_x**2 + new_error_y**2)
                    old_error_mag = np.sqrt(current_offset_x**2 + current_offset_y**2)

                    print(f"  [验证] 误差变化: {old_error_mag:.1f}px -> {new_error_mag:.1f}px")

                    # 关键检查：误差是否变大？
                    if new_error_mag > old_error_mag + 5:  # 误差增加超过5px
                        print(f"  ⚠⚠⚠ 严重警告: 误差增大！移动方向与预期相反！")
                        print(f"         灵敏度方向可能需要翻转")
                        print(f"         建议: 标定菜单 -> 0 -> 选择翻转选项")

                    # 检测灵敏度方向是否正确
                    # 如果误差很大但移动方向与预期相反，说明灵敏度方向错误
                    if abs(current_offset_x) > 30 or abs(current_offset_y) > 30:
                        # 预期应该向误差减小的方向移动
                        # 如果实际移动方向与预期相反，警告
                        if (abs(current_offset_x) > abs(new_error_x) and
                            np.sign(actual_dx) == np.sign(current_offset_x)):
                            print(f"  ⚠ 严重警告: X方向移动与预期相反！灵敏度方向可能错误！")
                            print(f"         建议: 运行 '验证灵敏度方向' 功能 (标定菜单 -> 9)")
                        elif (abs(current_offset_y) > abs(new_error_y) and
                              np.sign(actual_dy) == np.sign(current_offset_y)):
                            print(f"  ⚠ 严重警告: Y方向移动与预期相反！灵敏度方向可能错误！")
                            print(f"         建议: 运行 '验证灵敏度方向' 功能 (标定菜单 -> 9)")
                        elif abs(actual_dx) < 2 and abs(actual_dy) < 2 and (abs(current_offset_x) > 30 or abs(current_offset_y) > 30):
                            print(f"  ⚠ 警告: 移动效果很小，建议在当前姿态重新标定")
                            print(f"         提示: 主菜单 -> 4 (标定) -> 3 (关节灵敏度标定-自动)")

        print("\n✗ 对齐未完成 (达到最大迭代次数)")
        # 关闭视频窗口
        if self.show_alignment_video:
            cv2.destroyWindow(self._alignment_window_name)
        return False

    def _handle_detection_failure(self, state: DualMarkerState, consecutive_failures: int,
                                    max_consecutive_failures: int, recovery_attempts: int,
                                    max_recovery_attempts: int) -> int:
        """处理检测失败"""
        if consecutive_failures >= max_consecutive_failures and recovery_attempts < max_recovery_attempts:
            if not self.passive_mode:
                recovery_attempts += 1
                print(f"\n  [恢复尝试 {recovery_attempts}/{max_recovery_attempts}] 调整高度...")

                if not state.slot_detected and state.workpiece_detected:
                    print("  卡槽标记丢失，尝试上升...")
                    self.raise_height(2.0)
                elif not state.workpiece_detected and state.slot_detected:
                    print("  工件标记丢失，尝试微调...")
                    self.raise_height(1.0)
                else:
                    print("  标记均丢失，尝试上升...")
                    self.raise_height(3.0)

                time.sleep(0.5)
            else:
                print("  [被动模式] 无法自动恢复，请手动调整")

            if consecutive_failures >= max_consecutive_failures * 2:
                print("\n⚠ 持续检测失败，可能原因:")
                print("  1. 标记被遮挡或移出视野")
                print("  2. 光照变化导致颜色检测失败")
                print("  3. 高度/角度不合适")
                print("  建议手动调整后重试")

        return recovery_attempts

    def _record_offset(self, offset_x: float, offset_y: float):
        """记录历史偏移（用于预测）"""
        self._historical_offset_x.append(offset_x)
        self._historical_offset_y.append(offset_y)

        if len(self._historical_offset_x) > self._max_history_length:
            self._historical_offset_x.pop(0)
            self._historical_offset_y.pop(0)

    def _has_valid_prediction(self) -> bool:
        """检查是否有有效的预测数据"""
        return len(self._historical_offset_x) >= 3

    def _create_predicted_state(self, current_state: DualMarkerState) -> Optional[DualMarkerState]:
        """
        基于历史数据创建预测状态

        当卡槽标记丢失时，使用历史偏移预测当前位置
        """
        if not self._has_valid_prediction():
            return None

        # 使用最近的偏移平均值作为预测
        recent_x = self._historical_offset_x[-5:] if len(self._historical_offset_x) >= 5 else self._historical_offset_x
        recent_y = self._historical_offset_y[-5:] if len(self._historical_offset_y) >= 5 else self._historical_offset_y

        predicted_offset_x = sum(recent_x) / len(recent_x)
        predicted_offset_y = sum(recent_y) / len(recent_y)

        # 创建预测状态
        state = DualMarkerState(
            workpiece_1=current_state.workpiece_1,
            workpiece_2=current_state.workpiece_2,
            workpiece_3=current_state.workpiece_3,
            workpiece_detected=True,
            slot_detected=True,
            offset_x=predicted_offset_x,
            offset_y=predicted_offset_y,
            rotation_error=0,  # 预测模式下跳过旋转校正
            alignment_quality=0.5,  # 中等质量
            degraded_mode=True,
            degraded_reason="使用预测偏移"
        )

        return state

    def visualize_alignment(self, image, state: DualMarkerState,
                            adjustments: Dict[int, float] = None,
                            error_mm: float = 0.0,
                            iteration: int = 0,
                            max_iterations: int = 15) -> np.ndarray:
        """
        可视化对齐过程

        显示：
        - 相机画面 + 标记检测
        - 移动方向箭头
        - 误差信息
        - 关节调整量

        Args:
            image: 原始图像
            state: 检测状态
            adjustments: 关节调整量
            error_mm: 位置误差（毫米）
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数

        Returns:
            可视化图像
        """
        # 使用detector的基础可视化
        vis = self.detector.visualize(image, state)

        h, w = vis.shape[:2]

        # 计算工件和卡槽中心点（基于已检测到的标记）
        wp_center = None
        slot_center = None

        # 计算工件中心
        wp_markers = [m for m in state.workpiece_markers if m is not None]
        if wp_markers:
            wp_center = (
                sum(m.x for m in wp_markers) / len(wp_markers),
                sum(m.y for m in wp_markers) / len(wp_markers)
            )

        # 计算卡槽中心
        sl_markers = [m for m in state.slot_markers if m is not None]
        if sl_markers:
            slot_center = (
                sum(m.x for m in sl_markers) / len(sl_markers),
                sum(m.y for m in sl_markers) / len(sl_markers)
            )

        # 绘制移动方向箭头（如果检测到两个中心）
        if wp_center and slot_center:
            # 从工件中心指向卡槽中心的箭头（表示需要移动的方向）
            # 但实际移动方向相反：需要把工件移到卡槽位置
            # 所以箭头应该从工件中心指向卡槽中心
            cv2.arrowedLine(vis,
                           (int(wp_center[0]), int(wp_center[1])),
                           (int(slot_center[0]), int(slot_center[1])),
                           (0, 255, 255), 3, tipLength=0.3)

            # 在工件中心标注 "W"
            cv2.putText(vis, "W", (int(wp_center[0])-10, int(wp_center[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 在卡槽中心标注 "S"
            cv2.putText(vis, "S", (int(slot_center[0])-10, int(slot_center[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制信息面板
        panel_height = 180
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)

        # 迭代信息
        cv2.putText(panel, f"Iteration: {iteration+1}/{max_iterations}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 标记检测信息
        wp_count = state.workpiece_marker_count
        sl_count = state.slot_marker_count
        color = (0, 255, 0) if wp_count >= 3 and sl_count >= 3 else (0, 255, 255)
        cv2.putText(panel, f"Markers: Workpiece {wp_count}/3, Slot {sl_count}/3",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 误差信息
        mm_x = state.offset_x * self.pixel_to_mm_ratio
        mm_y = state.offset_y * self.pixel_to_mm_ratio
        cv2.putText(panel, f"Error: X={mm_x:.2f}mm, Y={mm_y:.2f}mm",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Total: {error_mm:.2f}mm",
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Rotation: {state.rotation_error:.1f}deg",
                   (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 关节调整信息
        if adjustments:
            y_pos = 150
            adj_text = "Joints: " + ", ".join([f"J{k}:{v:.2f}" for k, v in adjustments.items() if abs(v) > 0.01])
            cv2.putText(panel, adj_text[:60], (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        # 退化模式提示
        if state.degraded_mode:
            cv2.putText(panel, "DEGRADED MODE", (w-180, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # 预测模式提示
        if "预测" in state.degraded_reason if state.degraded_reason else "":
            cv2.putText(panel, "PREDICTION", (w-150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 合并面板和图像
        result = np.vstack([panel, vis])

        return result

    # ==================== 兼容旧接口 ====================

    def calibrate(self, move_distance_mm: float = 10.0) -> Tuple[bool, float]:
        """像素-毫米标定 (带视频显示)

        Args:
            move_distance_mm: 移动距离，默认10mm

        Returns:
            (success, ratio) 成功与否和转换比例
        """
        print("\n" + "="*50)
        print("像素-毫米标定")
        print("="*50)
        print(f"\n移动距离: {move_distance_mm}mm")
        print("\n操作步骤:")
        print(f"  1. 按 Enter 采集初始图像")
        print(f"  2. 使用示教器沿X方向移动 {move_distance_mm}mm")
        print(f"  3. 按 Enter 采集移动后图像")
        print("\n提示: 确保标记在视野内可见")

        window_name = "Pixel-MM Calibration"
        img1 = None
        img2 = None
        phase = 1  # 1=采集初始图像, 2=采集移动后图像

        while True:
            # 读取图像
            frame = self.camera.read()
            if frame is None:
                continue

            # 检测标记并可视化
            state = self.detector.detect_dual_marker_state(frame)
            vis = self.detector.visualize(frame, state)

            # 显示阶段提示
            if phase == 1:
                cv2.putText(vis, "Phase 1: Press ENTER to capture initial image", (10, vis.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis, f"Move distance: {move_distance_mm}mm", (10, vis.shape[0] - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                cv2.putText(vis, f"Move robot {move_distance_mm}mm along X axis", (10, vis.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(vis, "Phase 2: Press ENTER to capture final image", (10, vis.shape[0] - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(vis, "Press 'q' to cancel", (10, vis.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(window_name)
                print("✗ 标定已取消")
                return False, 0

            elif key == 13 or key == 10:  # Enter key
                if phase == 1:
                    img1 = frame.copy()
                    print(f"✓ 已采集初始图像")
                    phase = 2
                    print(f"\n请沿X方向移动 {move_distance_mm}mm...")
                else:
                    img2 = frame.copy()
                    print(f"✓ 已采集移动后图像")
                    break

        cv2.destroyWindow(window_name)

        # 计算像素偏移
        print("\n计算标定参数...")
        pixel_dx, pixel_dy = self._compute_pixel_shift(img1, img2)

        if pixel_dx is None or abs(pixel_dx) < 1:
            print("✗ 像素偏移太小或特征点匹配失败")
            return False, 0

        pixel_offset = abs(pixel_dx)
        ratio = move_distance_mm / pixel_offset

        print(f"\n计算结果:")
        print(f"  X方向像素偏移: {pixel_dx:.1f} pixels")
        print(f"  Y方向像素偏移: {pixel_dy:.1f} pixels")
        print(f"  转换比例: {ratio:.4f} mm/pixel")

        self.calibration_history.append({
            'ratio': ratio,
            'pixel_offset': pixel_offset,
            'move_distance': move_distance_mm,
            'arm': self.arm,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        self._save_calibration_history()

        self.pixel_to_mm_ratio = ratio

        # 更新现有标定点的mm灵敏度
        for cp in self.calibration_points:
            for s in cp.sensitivities:
                s.mm_dx_per_deg = s.pixel_dx_per_deg * ratio
                s.mm_dy_per_deg = s.pixel_dy_per_deg * ratio
        self._save_calibration_points()

        print("\n✓ 标定完成")
        return True, ratio

    def _load_calibration_history(self):
        """加载标定历史"""
        path = Path(__file__).parent / "calibration_history.json"
        if path.exists():
            with open(path, 'r') as f:
                self.calibration_history = json.load(f)

            if self.calibration_history:
                latest = self.calibration_history[-1]
                self.pixel_to_mm_ratio = latest.get('ratio', 0.5)

    def _save_calibration_history(self):
        """保存标定历史"""
        path = Path(__file__).parent / "calibration_history.json"
        with open(path, 'w') as f:
            json.dump(self.calibration_history, f, indent=2)

    def show_calibration_history(self):
        """显示标定历史"""
        if not self.calibration_history:
            print("\n暂无标定历史")
            return

        print("\n" + "="*50)
        print("标定历史 (像素-毫米)")
        print("="*50)

        for i, cal in enumerate(self.calibration_history[-5:]):
            print(f"\n[{i+1}] {cal['timestamp']}")
            print(f"    手臂: {cal['arm']}")
            print(f"    比例: {cal['ratio']:.4f} mm/pixel")

    # ==================== 标定验证 ====================

    def verify_xy_calibration(self, joint_idx: int = None, test_delta: float = 1.0) -> Dict:
        """
        验证XY标定的灵敏度方向是否正确

        原理：移动关节后，检测像素变化方向是否与预期一致

        Args:
            joint_idx: 要测试的关节索引，None表示测试所有已标定关节
            test_delta: 测试移动角度（度）

        Returns:
            验证结果字典
        """
        print("\n" + "="*50)
        print("XY标定验证 - 灵敏度方向检查")
        print("="*50)

        results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

        # 获取当前关节状态
        current_joints = self.get_joint_states()
        if current_joints is None:
            print("✗ 无法获取关节状态")
            return results

        # 获取当前灵敏度
        sensitivities = self.get_interpolated_sensitivities(current_joints)
        if not sensitivities:
            print("✗ 无标定数据")
            return results

        # 如果指定了关节，只测试该关节
        if joint_idx is not None:
            sensitivities = [s for s in sensitivities if s.joint_idx == joint_idx]
            if not sensitivities:
                print(f"✗ 关节 {joint_idx} 无标定数据")
                return results

        # 获取初始像素偏移
        print("\n正在获取初始位置...")
        initial_state = self._get_stable_pixel_offset(num_samples=3)
        if initial_state is None:
            print("✗ 无法获取初始检测")
            return results

        initial_offset_x, initial_offset_y, initial_quality = initial_state
        print(f"  初始偏移: ({initial_offset_x:.1f}, {initial_offset_y:.1f}) px, 质量: {initial_quality:.2f}")

        for s in sensitivities:
            jidx = s.joint_idx
            print(f"\n--- 测试关节 {jidx} ---")
            print(f"  标定灵敏度: X={s.pixel_dx_per_deg:.1f}, Y={s.pixel_dy_per_deg:.1f} px/deg")

            # 正向移动
            print(f"\n  正向移动 {test_delta}°...")
            self._move_single_joint(jidx, test_delta)
            time.sleep(0.5)

            new_state_pos = self._get_stable_pixel_offset(num_samples=3)
            if new_state_pos is None:
                print("  ✗ 移动后检测失败")
                self._move_single_joint(jidx, -test_delta)  # 恢复
                results['warnings'].append(f"关节 {jidx}: 检测失败")
                continue

            pos_offset_x, pos_offset_y, _ = new_state_pos
            actual_dx_pos = pos_offset_x - initial_offset_x
            actual_dy_pos = pos_offset_y - initial_offset_y
            print(f"  实际变化: X={actual_dx_pos:.1f}, Y={actual_dy_pos:.1f} px")

            # 恢复原位
            print(f"  恢复原位...")
            self._move_single_joint(jidx, -test_delta)
            time.sleep(0.5)

            # 反向移动验证
            print(f"\n  反向移动 {test_delta}°...")
            self._move_single_joint(jidx, -test_delta)
            time.sleep(0.5)

            new_state_neg = self._get_stable_pixel_offset(num_samples=3)
            if new_state_neg is None:
                print("  ✗ 反向移动后检测失败")
                self._move_single_joint(jidx, test_delta)  # 恢复
                results['warnings'].append(f"关节 {jidx}: 反向检测失败")
                continue

            neg_offset_x, neg_offset_y, _ = new_state_neg
            actual_dx_neg = neg_offset_x - initial_offset_x
            actual_dy_neg = neg_offset_y - initial_offset_y
            print(f"  实际变化: X={actual_dx_neg:.1f}, Y={actual_dy_neg:.1f} px")

            # 恢复原位
            self._move_single_joint(jidx, test_delta)
            time.sleep(0.3)

            # 验证方向
            # 正向移动应该产生与标定灵敏度相同方向的变化
            # 反向移动应该产生相反方向的变化

            expected_dx = s.pixel_dx_per_deg * test_delta
            expected_dy = s.pixel_dy_per_deg * test_delta

            # 检查方向是否一致
            pos_x_correct = np.sign(actual_dx_pos) == np.sign(expected_dx) if abs(expected_dx) > 5 else True
            pos_y_correct = np.sign(actual_dy_pos) == np.sign(expected_dy) if abs(expected_dy) > 5 else True
            neg_x_correct = np.sign(actual_dx_neg) == -np.sign(expected_dx) if abs(expected_dx) > 5 else True
            neg_y_correct = np.sign(actual_dy_neg) == -np.sign(expected_dy) if abs(expected_dy) > 5 else True

            # 检查反向是否相反
            reverse_x_correct = np.sign(actual_dx_pos) == -np.sign(actual_dx_neg) if (abs(actual_dx_pos) > 3 and abs(actual_dx_neg) > 3) else True
            reverse_y_correct = np.sign(actual_dy_pos) == -np.sign(actual_dy_neg) if (abs(actual_dy_pos) > 3 and abs(actual_dy_neg) > 3) else True

            all_correct = pos_x_correct and pos_y_correct and neg_x_correct and neg_y_correct and reverse_x_correct and reverse_y_correct

            if all_correct:
                print(f"\n  ✓ 关节 {jidx} 灵敏度方向正确")
                results['passed'].append({
                    'joint': jidx,
                    'expected_dx': expected_dx,
                    'expected_dy': expected_dy,
                    'actual_dx_pos': actual_dx_pos,
                    'actual_dy_pos': actual_dy_pos
                })
            else:
                print(f"\n  ✗ 关节 {jidx} 灵敏度方向错误！")
                if not pos_x_correct:
                    print(f"    X正向方向错误: 预期 {expected_dx:.1f}, 实际 {actual_dx_pos:.1f}")
                if not pos_y_correct:
                    print(f"    Y正向方向错误: 预期 {expected_dy:.1f}, 实际 {actual_dy_pos:.1f}")
                if not reverse_x_correct:
                    print(f"    X正反方向不一致")
                if not reverse_y_correct:
                    print(f"    Y正反方向不一致")

                results['failed'].append({
                    'joint': jidx,
                    'expected_dx': expected_dx,
                    'expected_dy': expected_dy,
                    'actual_dx_pos': actual_dx_pos,
                    'actual_dy_pos': actual_dy_pos,
                    'suggestion': '需要翻转灵敏度方向或重新标定'
                })

        # 总结
        print("\n" + "="*50)
        print("验证结果")
        print("="*50)
        print(f"通过: {len(results['passed'])} 个关节")
        print(f"失败: {len(results['failed'])} 个关节")
        if results['warnings']:
            print(f"警告: {len(results['warnings'])} 个")

        if results['failed']:
            print("\n建议操作:")
            print("  1. 在标定菜单中选择 '翻转灵敏度方向'")
            print("  2. 或者在当前姿态重新标定")

        return results

    def verify_z_calibration(self) -> Dict:
        """
        验证Z轴标定是否正确

        通过比较双目深度估计和已知的物理参考来判断

        Returns:
            验证结果字典
        """
        print("\n" + "="*50)
        print("Z轴标定验证")
        print("="*50)

        results = {
            'passed': False,
            'depth_estimate': None,
            'confidence': 0,
            'message': ''
        }

        if self.z_controller is None or self.camera2 is None:
            print("✗ Z轴控制器或副相机未配置")
            results['message'] = 'Z轴控制器或副相机未配置'
            return results

        # 获取多帧深度估计
        print("\n正在获取深度估计...")
        estimates = []
        for i in range(5):
            image1 = self.camera.read()
            image2 = self.camera2.read()
            if image1 is not None and image2 is not None:
                estimate = self.z_controller.estimate_z(image1, image2, self.detector.workpiece_color)
                if estimate.confidence > 0.3:
                    estimates.append((estimate.z, estimate.confidence))
                    print(f"  帧 {i+1}: 深度={estimate.z:.1f}mm, 置信度={estimate.confidence:.2f}")
            time.sleep(0.1)

        if len(estimates) < 3:
            print("✗ 深度估计不稳定")
            results['message'] = '深度估计不稳定'
            return results

        # 计算平均和标准差
        depths = [e[0] for e in estimates]
        confidences = [e[1] for e in estimates]
        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        mean_confidence = np.mean(confidences)

        print(f"\n统计结果:")
        print(f"  平均深度: {mean_depth:.1f}mm")
        print(f"  标准差: {std_depth:.1f}mm")
        print(f"  平均置信度: {mean_confidence:.2f}")

        # 判断稳定性
        if std_depth < 5.0:
            print("  ✓ 深度估计稳定 (标准差 < 5mm)")
            results['passed'] = True
            results['depth_estimate'] = mean_depth
            results['confidence'] = mean_confidence
            results['message'] = f'深度估计稳定: {mean_depth:.1f}mm ± {std_depth:.1f}mm'
        else:
            print(f"  ⚠ 深度估计不稳定 (标准差 = {std_depth:.1f}mm)")
            print("  建议:")
            print("    1. 检查双相机标定是否准确")
            print("    2. 检查光照条件")
            print("    3. 确保工件颜色设置正确")
            results['message'] = f'深度估计不稳定: 标准差 {std_depth:.1f}mm'

        return results

    def verify_calibration_completeness(self) -> Dict:
        """
        检查标定完整性

        Returns:
            完整性检查结果
        """
        print("\n" + "="*50)
        print("标定完整性检查")
        print("="*50)

        results = {
            'xy_calibration': False,
            'z_calibration': False,
            'target_offset': False,
            'pixel_ratio': False,
            'issues': []
        }

        # 检查XY标定
        print("\n[1] XY标定检查")
        if self.calibration_points:
            print(f"  ✓ 有 {len(self.calibration_points)} 个标定点")
            total_sensitivities = sum(len(cp.sensitivities) for cp in self.calibration_points)
            print(f"  ✓ 共 {total_sensitivities} 条灵敏度记录")

            # 检查每个关节是否有标定
            joints_with_calibration = set()
            for cp in self.calibration_points:
                for s in cp.sensitivities:
                    joints_with_calibration.add(s.joint_idx)

            print(f"  已标定关节: {sorted(joints_with_calibration)}")

            if len(joints_with_calibration) >= 2:
                results['xy_calibration'] = True
            else:
                results['issues'].append("标定关节数不足 (需要至少2个)")
        else:
            print("  ✗ 无XY标定数据")
            results['issues'].append("无XY标定数据")

        # 检查Z标定
        print("\n[2] Z轴标定检查")
        if self.z_controller is not None:
            if hasattr(self.z_controller, 'joint_sensitivities') and self.z_controller.joint_sensitivities:
                print(f"  ✓ 有 {len(self.z_controller.joint_sensitivities)} 个Z轴灵敏度")
                results['z_calibration'] = True
            else:
                print("  ⚠ Z轴控制器存在但无灵敏度数据")
        else:
            print("  - Z轴控制器未配置")

        # 检查目标偏移
        print("\n[3] 目标偏移检查")
        if self.target_offset_x != 0 or self.target_offset_y != 0:
            print(f"  ✓ 目标偏移已设置: ({self.target_offset_x:.1f}, {self.target_offset_y:.1f}) px")
            results['target_offset'] = True
        else:
            print("  ✗ 目标偏移未设置")
            results['issues'].append("目标偏移未设置")

        # 检查像素比例
        print("\n[4] 像素-毫米比例检查")
        if self.pixel_to_mm_ratio > 0:
            print(f"  ✓ 像素比例: {self.pixel_to_mm_ratio:.3f} mm/px")
            results['pixel_ratio'] = True
        else:
            print("  ✗ 像素比例未设置")
            results['issues'].append("像素比例未设置")

        # 总结
        print("\n" + "="*50)
        all_ok = results['xy_calibration'] and results['target_offset'] and results['pixel_ratio']
        if all_ok:
            print("✓ 标定完整，可以进行对齐")
        else:
            print("✗ 标定不完整，请补充以下内容:")
            for issue in results['issues']:
                print(f"  - {issue}")

        return results

    def _move_single_joint(self, joint_idx: int, delta: float):
        """移动单个关节"""
        joints = self.get_joint_states()
        if joints is None:
            return False

        target = joints.copy()
        if joint_idx < len(target):
            target[joint_idx] += delta
            return self.move_to_joint_positions(target)
        return False

    # ==================== 预设位置 ====================

    def save_preset(self, name: str):
        """保存当前位置为预设"""
        joints = self.get_joint_states()

        if joints is None:
            print("✗ 无法获取关节位置")
            if self.passive_mode:
                print("  请确认示教程序已启动并启用状态共享")
            return False

        self.presets[name] = joints.copy()
        self._save_presets()

        print(f"✓ 预设位置已保存: {name}")
        return True

    def load_preset(self, name: str) -> bool:
        """加载预设位置 (平滑移动)"""
        if name not in self.presets:
            print(f"✗ 预设不存在: {name}")
            return False

        target_joints = self.presets[name]
        print(f"\n移动到预设位置: {name} ...")

        # 使用平滑移动
        success = self._smooth_move_all_joints(target_joints)

        if success:
            print(f"✓ 已移动到预设位置: {name}")
        else:
            print(f"✗ 移动失败")
        return success

    def _load_presets(self):
        """加载预设"""
        path = Path(__file__).parent / "presets.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.presets = {k: np.array(v) for k, v in data.items()}

    def _save_presets(self):
        """保存预设"""
        path = Path(__file__).parent / "presets.json"
        with open(path, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.presets.items()}, f, indent=2)

    def list_presets(self):
        """列出所有预设"""
        if not self.presets:
            print("\n暂无预设位置")
            return

        print("\n预设位置列表:")
        for name in self.presets:
            print(f"  - {name}")

    # ==================== 高度控制 ====================

    def raise_height(self, step: float = 2.0):
        """上升 (粗调)"""
        if self.passive_mode:
            print("✗ 被动模式下无法调整高度")
            return False

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.height_joint_idx] -= step
        action = {f"{name}.pos": float(joints[i])
                  for i, name in enumerate(self.robot.observation_joint_names)
                  if i < len(joints)}
        self.robot.send_action(action)
        time.sleep(self.settle_time)
        return True

    def lower_height(self, step: float = 2.0):
        """下降 (粗调)"""
        if self.passive_mode:
            print("✗ 被动模式下无法调整高度")
            return False

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.height_joint_idx] += step
        action = {f"{name}.pos": float(joints[i])
                  for i, name in enumerate(self.robot.observation_joint_names)
                  if i < len(joints)}
        self.robot.send_action(action)
        time.sleep(self.settle_time)
        return True

    def auto_adjust_height(self, max_attempts: int = 10, skip_if_pose_recorded: bool = True) -> bool:
        """自动调整高度

        Args:
            max_attempts: 最大尝试次数
            skip_if_pose_recorded: 如果已记录设置偏移量时的姿态，是否跳过高度调整

        Returns:
            是否高度合适
        """
        print("\n自动调整高度...")

        # 如果已经恢复了设置偏移量时的姿态，跳过高度调整
        if skip_if_pose_recorded and self._calibration_joint_states is not None:
            print("  已恢复到设置偏移量时的姿态，跳过自动高度调整")

            # 只检查检测情况，不调整高度
            time.sleep(0.2)
            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)

            wp = sum(1 for m in state.workpiece_markers if m)
            sl = sum(1 for m in state.slot_markers if m)

            print(f"  工件: {wp}/3, 卡槽: {sl}/3")

            if wp >= 2 and sl >= 2:
                print("  ✓ 检测正常")
                return True
            else:
                print("  ⚠ 检测不完整，可能原因:")
                print("    1. 标记被遮挡")
                print("    2. 相机角度问题")
                print("    3. 设置偏移量时的姿态与当前环境不匹配")
                print("  建议: 保持当前姿态继续对齐，或重新设置偏移量")
                return True  # 仍然返回True，允许继续对齐

        for i in range(max_attempts):
            print(f"\n[高度 {i+1}/{max_attempts}]")

            time.sleep(0.2)
            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)

            wp = sum(1 for m in state.workpiece_markers if m)
            sl = sum(1 for m in state.slot_markers if m)

            print(f"  工件: {wp}/3, 卡槽: {sl}/3")

            if wp >= 2 and sl >= 2:
                print("\n✓ 高度合适")
                return True

            if sl < 2:
                print("  上升中...")
                self.raise_height()
            else:
                self.raise_height(1.0)

        print("\n✗ 高度调整未完成")
        return False

    # ==================== Z轴精确控制 ====================

    def set_target_z(self, target_z: float):
        """
        设置Z轴目标深度

        Args:
            target_z: 目标深度 (mm)，即标记点到相机的距离
        """
        self.target_z = target_z
        if self.z_controller is not None:
            self.z_controller.set_target_z(target_z)
        print(f"✓ Z轴目标深度: {target_z:.1f}mm")

    def set_marker_diameter(self, diameter_mm: float):
        """
        设置标记点直径

        Args:
            diameter_mm: 标记点直径 (mm)
        """
        if self.z_controller is not None:
            self.z_controller.depth_estimator.set_marker_diameter(diameter_mm)
        print(f"✓ 标记点直径: {diameter_mm:.1f}mm")

    def calibrate_z_baseline(self, baseline_mm: float):
        """
        设置双目基线距离

        Args:
            baseline_mm: 两个相机之间的距离 (mm)
        """
        if self.z_controller is not None:
            self.z_controller.depth_estimator.set_baseline(baseline_mm)
        print(f"✓ 双目基线距离: {baseline_mm:.1f}mm")

    def estimate_current_z(self) -> Optional[float]:
        """
        估计当前Z轴深度

        Returns:
            当前深度 (mm)，失败返回 None
        """
        if self.z_controller is None:
            return None

        image1 = self.camera.read()
        image2 = None
        if self.camera2 is not None:
            image2 = self.camera2.read()

        estimate = self.z_controller.estimate_z(image1, image2, self.detector.workpiece_color)

        if estimate.confidence > 0:
            return estimate.z
        return None

    def align_z(self, tolerance_mm: float = None) -> bool:
        """
        Z轴对齐

        Args:
            tolerance_mm: Z轴容差 (mm)

        Returns:
            是否对齐成功
        """
        if self.z_controller is None:
            print("✗ Z轴控制器未启用")
            return False

        if self.target_z is None:
            print("✗ 未设置Z轴目标，请先调用 set_target_z()")
            return False

        if tolerance_mm is None:
            tolerance_mm = self.z_tolerance_mm

        if self.passive_mode:
            print("✗ 被动模式下无法执行Z轴对齐")
            return False

        print(f"\nZ轴对齐 - 目标: {self.target_z:.1f}mm, 容差: ±{tolerance_mm}mm")

        for i in range(self.max_iterations):
            print(f"\n[Z轴 {i+1}/{self.max_iterations}]")

            # 获取图像
            image1 = self.camera.read()
            image2 = None
            if self.camera2 is not None:
                image2 = self.camera2.read()

            # 估计深度
            estimate = self.z_controller.estimate_z(image1, image2, self.detector.workpiece_color)

            print(f"  深度: {estimate.z:.1f}mm ±{estimate.uncertainty:.1f}mm ({estimate.method})")

            # 计算误差
            z_error = self.z_controller.compute_z_error(estimate.z)
            print(f"  误差: {z_error:.1f}mm")

            # 检查是否对齐
            if abs(z_error) < tolerance_mm:
                print(f"\n✓ Z轴对齐完成 (误差: {z_error:.2f}mm < {tolerance_mm}mm)")
                return True

            # 计算调整量
            adjustments = self.z_controller.compute_z_adjustment()
            adj_str = ', '.join([f'{self.z_controller.JOINT_NAMES.get(k, str(k))}:{v:.2f}°'
                                 for k, v in adjustments.items()])
            print(f"  调整: {adj_str}")

            # 应用调整
            joints = self.get_joint_states()
            if joints is not None:
                for joint_idx, adjustment_deg in adjustments.items():
                    joints[joint_idx] += adjustment_deg
                action = {f"{name}.pos": float(joints[i])
                          for i, name in enumerate(self.robot.observation_joint_names)
                          if i < len(joints)}
                self.robot.send_action(action)
                time.sleep(self.settle_time)

        print("\n✗ Z轴对齐未完成 (达到最大迭代次数)")
        return False

    def align_xyz(self, tolerance_xy: float = None, tolerance_z: float = None) -> bool:
        """
        XYZ三维对齐

        Args:
            tolerance_xy: XY容差 (mm)
            tolerance_z: Z容差 (mm)

        Returns:
            是否对齐成功
        """
        if tolerance_xy is None:
            tolerance_xy = self.tolerance_mm
        if tolerance_z is None:
            tolerance_z = self.z_tolerance_mm

        print(f"\n{'='*60}")
        print(f"三维对齐 - XY容差: ±{tolerance_xy}mm, Z容差: ±{tolerance_z}mm")
        print(f"{'='*60}")

        # 先粗调高度确保能看到标记
        self.auto_adjust_height()

        # 迭代对齐
        for i in range(self.max_iterations):
            print(f"\n[三维对齐 {i+1}/{self.max_iterations}]")

            # XY对齐
            xy_ok = self._align_xy_single_iteration(tolerance_xy)

            # Z轴对齐
            z_ok = self._align_z_single_iteration(tolerance_z)

            if xy_ok and z_ok:
                print(f"\n✓ 三维对齐完成!")
                return True

            time.sleep(self.settle_time)

        print("\n✗ 三维对齐未完成")
        return False

    def _align_xy_single_iteration(self, tolerance: float) -> bool:
        """XY单次对齐迭代"""
        image = self.camera.read()
        state = self.detector.detect_dual_marker_state(image)

        if not state.workpiece_detected or not state.slot_detected:
            print("  XY: 标记不完整")
            return False

        current_offset_x = state.offset_x - self.target_offset_x
        current_offset_y = state.offset_y - self.target_offset_y

        mm_x = current_offset_x * self.pixel_to_mm_ratio
        mm_y = current_offset_y * self.pixel_to_mm_ratio
        error_mm = np.sqrt(mm_x**2 + mm_y**2)

        print(f"  XY误差: ({mm_x:.2f}, {mm_y:.2f})mm, 总: {error_mm:.2f}mm")

        if error_mm < tolerance and abs(state.rotation_error) < self.tolerance_deg:
            print(f"  XY: ✓ 对齐")
            return True

        current_joints = self.get_joint_states()
        if current_joints is not None:
            adjustments = self.compute_joint_adjustments(current_offset_x, current_offset_y, current_joints)
            rotation_adjustments = self.compute_rotation_adjustment(state.rotation_error, current_joints)
            self.apply_joint_adjustments(adjustments, rotation_adjustments)

        return False

    def _align_z_single_iteration(self, tolerance: float) -> bool:
        """Z轴单次对齐迭代"""
        if self.z_controller is None or self.target_z is None:
            return True  # Z轴未启用视为通过

        image1 = self.camera.read()
        image2 = self.camera2.read() if self.camera2 else None

        estimate = self.z_controller.estimate_z(image1, image2, self.detector.workpiece_color)

        if estimate.confidence < 0.1:
            print("  Z: 深度估计失败")
            return False

        z_error = self.z_controller.compute_z_error(estimate.z)
        print(f"  Z误差: {z_error:.1f}mm (深度: {estimate.z:.1f}mm)")

        if abs(z_error) < tolerance:
            print(f"  Z: ✓ 对齐")
            return True

        if not self.passive_mode:
            # 获取多关节调整量
            adjustments = self.z_controller.compute_z_adjustment()
            joints = self.get_joint_states()
            if joints is not None:
                # 应用所有关节调整
                for joint_idx, adjustment_deg in adjustments.items():
                    joints[joint_idx] += adjustment_deg
                action = {f"{name}.pos": float(joints[i])
                          for i, name in enumerate(self.robot.observation_joint_names)
                          if i < len(joints)}
                self.robot.send_action(action)

        return False

    # ==================== 夹爪控制 ====================

    def open_gripper(self):
        """打开夹爪"""
        if self.passive_mode:
            print("✗ 被动模式下无法控制夹爪")
            return False

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.arm_config.gripper_idx] = self.arm_config.gripper_open
        action = {f"{name}.pos": float(joints[i])
                  for i, name in enumerate(self.robot.observation_joint_names)
                  if i < len(joints)}
        self.robot.send_action(action)
        time.sleep(0.5)
        print("✓ 夹爪已打开")
        return True

    def close_gripper(self, position: float = None):
        """闭合夹爪"""
        if self.passive_mode:
            print("✗ 被动模式下无法控制夹爪")
            return False

        if position is None:
            position = self.arm_config.gripper_close

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.arm_config.gripper_idx] = position
        action = {f"{name}.pos": float(joints[i])
                  for i, name in enumerate(self.robot.observation_joint_names)
                  if i < len(joints)}
        self.robot.send_action(action)
        time.sleep(0.5)
        print("✓ 夹爪已闭合")
        return True

    # ==================== 自动放置 ====================

    def compute_xyz_coordinated_descent(self, z_delta_mm: float) -> Dict[int, float]:
        """
        计算XYZ协调下降的关节调整量

        在下降时保持XY位置不变，使用伪逆求解多关节协调。

        Args:
            z_delta_mm: Z轴变化量(mm)，正值下降

        Returns:
            {joint_idx: angle_delta_deg}
        """
        # 获取当前关节状态
        current_joints = self.get_joint_states()
        if current_joints is None:
            return {}

        # 获取XY灵敏度数据
        xy_sensitivities = self.get_interpolated_sensitivities(current_joints)
        if not xy_sensitivities:
            print("  ✗ 无XY灵敏度数据，使用单关节下降")
            return {self.height_joint_idx: z_delta_mm / 10.0}  # 假设约10mm/度

        # 获取Z灵敏度数据
        z_sensitivities = {}
        if self.z_controller is not None and hasattr(self.z_controller, 'joint_sensitivities'):
            z_sensitivities = self.z_controller.joint_sensitivities

        # 构建关节索引列表（使用XY灵敏度中有的关节）
        joint_indices = [s.joint_idx for s in xy_sensitivities]
        n_joints = len(joint_indices)

        if n_joints < 2:
            print("  ✗ 关节数不足，使用单关节下降")
            return {self.height_joint_idx: z_delta_mm / 10.0}

        # 构建雅可比矩阵 (3 x N)
        # 行1: Z灵敏度 (mm/deg)
        # 行2: X灵敏度 (pixel/deg)
        # 行3: Y灵敏度 (pixel/deg)
        J = np.zeros((3, n_joints))
        W = np.zeros((n_joints, n_joints))  # 位置权重

        # Z控制权重（影响位置能力的权重）
        Z_WEIGHTS = {
            7: 1.0,   # joint_1 (底座旋转)
            8: 0.9,   # joint_2 (肩部俯仰) - 主要影响高度
            9: 0.7,   # joint_3 (肩部侧摆)
            10: 0.6,  # joint_4 (前臂俯仰)
            11: 0.3,  # joint_5 (腕部俯仰)
            12: 0.2,  # joint_6 (手腕旋转)
        }

        for i, jidx in enumerate(joint_indices):
            # Z灵敏度
            if jidx in z_sensitivities:
                J[0, i] = z_sensitivities[jidx].mm_per_deg
            else:
                J[0, i] = 0.0  # 无Z数据时假设为0

            # XY灵敏度
            xy_sens = next((s for s in xy_sensitivities if s.joint_idx == jidx), None)
            if xy_sens:
                J[1, i] = xy_sens.pixel_dx_per_deg
                J[2, i] = xy_sens.pixel_dy_per_deg

            # 权重
            W[i, i] = Z_WEIGHTS.get(jidx, 0.5)

        # 目标向量：Z变化z_delta_mm，XY变化为0
        target = np.array([z_delta_mm, 0.0, 0.0])

        # 使用加权伪逆求解: delta = W @ J^T @ (J @ W @ J^T)^-1 @ target
        # 或简化的最小范数解: delta = J^+ @ target
        try:
            # 计算伪逆
            JW = J @ W  # 加权雅可比 (3 x N)
            JJT = JW @ JW.T  # (3 x 3)

            # 检查是否可逆
            if np.linalg.det(JJT) < 1e-10:
                raise np.linalg.LinAlgError("奇异矩阵")

            # 伪逆: J^+ = J^T @ (J @ J^T)^-1
            JW_pinv = JW.T @ np.linalg.inv(JJT)

            # 计算角度调整
            delta_angles = JW_pinv @ target

            # 检查解是否合理
            max_delta = np.max(np.abs(delta_angles))
            if max_delta > 5.0:  # 单次调整超过5度
                print(f"    协调下降解过大 (max={max_delta:.1f}°)，限制幅度")
                delta_angles = delta_angles * 5.0 / max_delta

        except np.linalg.LinAlgError:
            print("    伪逆求解失败，使用简化方法")
            # 简化方法：只用主要关节下降
            delta_angles = np.zeros(n_joints)
            for i, jidx in enumerate(joint_indices):
                if jidx == self.height_joint_idx and jidx in z_sensitivities:
                    delta_angles[i] = z_delta_mm / z_sensitivities[jidx].mm_per_deg

        # 应用限幅
        delta_angles = np.clip(delta_angles, -2.0, 2.0)

        # 转换为调整字典
        adjustments = {}
        print(f"    XYZ协调下降 (Z={z_delta_mm:.1f}mm, XY保持):")
        for i, (jidx, delta) in enumerate(zip(joint_indices, delta_angles)):
            if abs(delta) > 0.01:
                adjustments[jidx] = delta
                print(f"      joint_{jidx}: {delta:.2f}°")

        # 验证预期效果
        expected_z = sum(J[0, i] * delta_angles[i] for i in range(n_joints))
        expected_x = sum(J[1, i] * delta_angles[i] for i in range(n_joints))
        expected_y = sum(J[2, i] * delta_angles[i] for i in range(n_joints))
        print(f"    预期变化: Z={expected_z:.1f}mm, X={expected_x:.1f}px, Y={expected_y:.1f}px")

        return adjustments

    def compute_height_correction(self, z_delta_mm: float) -> Dict[int, float]:
        """
        计算高度补偿的关节调整量（在XY调整后保持高度）

        这是 compute_xyz_coordinated_descent 的反向操作：
        - 在XY调整后，高度可能发生变化
        - 使用伪逆求解关节调整量，恢复高度同时保持XY不变

        Args:
            z_delta_mm: Z轴需要补偿的量(mm)，正值表示需要上升，负值表示需要下降

        Returns:
            {joint_idx: angle_delta_deg}
        """
        return self.compute_xyz_coordinated_descent(z_delta_mm)

    def compute_perspective_correction(self, current_height: float) -> float:
        """
        计算透视效应补偿后的像素-毫米比例

        透视效应：近大远小。相机光心到目标的距离越远，同样的物理尺寸在图像上越小。
        即：高度越高，1像素对应的物理尺寸越大。

        简化模型：假设像素尺寸与距离成正比
        ratio(H) = ratio(H_ref) * H / H_ref

        Args:
            current_height: 当前高度(mm)

        Returns:
            补偿后的 pixel_to_mm 比例
        """
        if self._reference_height is None or self._reference_height <= 0:
            return self.pixel_to_mm_ratio

        # 计算补偿比例
        # 高度越高，每像素对应的物理距离越大
        correction_factor = current_height / self._reference_height
        corrected_ratio = self._reference_pixel_to_mm * correction_factor

        return corrected_ratio

    def get_current_height(self) -> Optional[float]:
        """
        获取当前高度估计（通过双目相机）

        Returns:
            高度(mm) 或 None（如果估计失败）
        """
        if self.z_controller is None or self.camera2 is None:
            return None

        image1 = self.camera.read()
        image2 = self.camera2.read()

        if image1 is None or image2 is None:
            return None

        estimate = self.z_controller.estimate_z(image1, image2, self.detector.workpiece_color)

        if estimate.confidence > 0.3:
            return estimate.z
        return None

    def _smooth_move_height(self, delta: float, steps: int = 5, use_xyz_coordination: bool = True) -> bool:
        """平滑调整高度

        Args:
            delta: 高度变化量 (正值下降，负值上升)
            steps: 插值步数
            use_xyz_coordination: 是否使用XYZ协调（下降时保持XY）
        """
        joints = self.get_joint_states()
        if joints is None or len(joints) < 16:
            return False

        if use_xyz_coordination and delta > 0:
            # 使用XYZ协调下降（保持XY不变）
            adjustments = self.compute_xyz_coordinated_descent(delta)
            if adjustments:
                # 分步执行
                target = joints.copy()
                for jidx, angle_delta in adjustments.items():
                    if jidx < len(target):
                        target[jidx] += angle_delta
                return self._smooth_move_all_joints(target, steps)

        # 回退到单关节下降
        target = joints.copy()
        target[self.height_joint_idx] += delta

        return self._smooth_move_all_joints(target, steps)

    def auto_place(self, lower_steps: int = 5, lower_step_size: float = 2.0,
                   xy_correction: bool = True, max_xy_correction: int = 3):
        """自动放置流程 (平滑移动 + 闭环XY校正)

        Args:
            lower_steps: 下降步数
            lower_step_size: 每步下降量(mm)
            xy_correction: 是否在下降过程中校正XY偏移
            max_xy_correction: 最大XY校正次数
        """
        print("\n自动放置...")

        print("\n[1/3] 平滑下降到放置高度 (XYZ协调)")
        total_delta = lower_step_size * lower_steps

        if xy_correction:
            # 分步下降 + 每步校正XY
            remaining_delta = total_delta
            for step in range(lower_steps):
                # 计算本步下降量
                step_delta = min(lower_step_size, remaining_delta)

                # XYZ协调下降
                adjustments = self.compute_xyz_coordinated_descent(step_delta)
                if adjustments:
                    joints = self.get_joint_states()
                    if joints is not None:
                        target = joints.copy()
                        for jidx, angle_delta in adjustments.items():
                            if jidx < len(target):
                                target[jidx] += angle_delta
                        self._smooth_move_all_joints(target, 2)
                else:
                    # 回退到单关节
                    self._smooth_move_height(step_delta, steps=2, use_xyz_coordination=False)

                remaining_delta -= step_delta
                time.sleep(0.3)

                # 检测并校正XY偏移（每步后）
                if step < lower_steps - 1:  # 最后一步不需要校正
                    state = self.detector.detect_dual_marker_state(self.camera.read())
                    if state.workpiece_detected and state.slot_detected:
                        xy_error = np.sqrt(state.offset_x**2 + state.offset_y**2)
                        if xy_error > 3.0:  # 像素偏移超过3px时校正
                            print(f"    步骤{step+1}: XY偏移 {xy_error:.1f}px，校正...")
                            self.align_xy_single_step(gain=0.5)
        else:
            # 原来的方式：一次性下降
            if self._smooth_move_height(total_delta, steps=lower_steps):
                print("  ✓ 下降完成")
            else:
                print("  ✗ 下降失败")
                return False

        print("  ✓ 下降完成")

        print("\n[2/3] 松开夹爪")
        self.open_gripper()

        print("\n[3/3] 平滑抬起")
        if self._smooth_move_height(-lower_step_size * 2, steps=3, use_xyz_coordination=False):
            print("  ✓ 抬起完成")
        else:
            print("  ✗ 抬起失败")

        print("\n✓ 自动放置完成")

    def align_xy_single_step(self, gain: float = None) -> bool:
        """单步XY校正（用于下降过程中的位置保持）

        Args:
            gain: 控制增益（默认使用 self.alignment_gain）

        Returns:
            是否执行了校正
        """
        if gain is None:
            gain = self.alignment_gain

        # 获取当前图像
        image = self.camera.read()
        if image is None:
            return False

        # 检测偏移
        state = self.detector.detect_dual_marker_state(image)
        if not state.workpiece_detected or not state.slot_detected:
            return False

        # 计算误差
        current_offset_x = state.offset_x - self.target_offset_x
        current_offset_y = state.offset_y - self.target_offset_y

        # 只有偏移足够大才校正
        xy_error = np.sqrt(current_offset_x**2 + current_offset_y**2)
        if xy_error < 2.0:  # 小于2像素不校正
            return False

        # 获取关节状态
        current_joints = self.get_joint_states()
        if current_joints is None:
            return False

        # 计算调整量
        adjustments = self.compute_joint_adjustments(
            current_offset_x, current_offset_y, current_joints
        )

        if not adjustments:
            return False

        # 应用增益
        adjusted = {k: v * gain for k, v in adjustments.items()}

        # 应用调整
        self.apply_joint_adjustments(adjusted, {})

        print(f"    XY校正: 偏移{xy_error:.1f}px -> 调整{adjusted}")

        return True

    def move_to_calibration_pose(self) -> bool:
        """
        移动到设置偏移量时的关节状态

        这确保对齐时的姿态与设置偏移量时一致，
        从而保证透视效应相同，提高对齐精度。

        Returns:
            是否成功移动
        """
        if self._calibration_joint_states is None:
            print("  未记录设置偏移量时的关节状态，跳过姿态恢复")
            return True

        print("\n恢复到设置偏移量时的姿态...")

        current_joints = self.get_joint_states()
        if current_joints is None:
            print("  ✗ 无法获取当前关节状态")
            return False

        # 计算差异
        diff = np.abs(current_joints - self._calibration_joint_states)
        max_diff = np.max(diff)

        if max_diff < 1.0:  # 差异小于1度，认为姿态一致
            print(f"  当前姿态与设置时一致 (最大差异 {max_diff:.2f}°)")
            return True

        print(f"  当前姿态与设置时差异: 最大 {max_diff:.2f}°")
        print("  移动到设置偏移量时的姿态...")

        # 平滑移动到目标姿态
        if self._smooth_move_to_joints(self._calibration_joint_states):
            print("  ✓ 已恢复到设置偏移量时的姿态")
            time.sleep(0.5)  # 等待稳定
            return True
        else:
            print("  ✗ 移动失败")
            return False

    def _smooth_move_to_joints(self, target_joints: np.ndarray, steps: int = 20) -> bool:
        """
        平滑移动到目标关节状态

        Args:
            target_joints: 目标关节角度数组
            steps: 移动步数

        Returns:
            是否成功
        """
        current_joints = self.get_joint_states()
        if current_joints is None:
            return False

        delta = (target_joints - current_joints) / steps

        for step in range(steps):
            alpha = (step + 1) / steps
            # 使用平滑插值
            smooth_alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
            intermediate = current_joints + (target_joints - current_joints) * smooth_alpha

            action = {f"{name}.pos": float(intermediate[i])
                      for i, name in enumerate(self.robot.observation_joint_names)
                      if i < len(intermediate)}

            self.robot.send_action(action)
            time.sleep(self.smooth_delay)

        return True

    # ==================== 完整流程 ====================

    def run_full_sequence(self, tolerance_mm: float = 2.0, auto_place: bool = False) -> bool:
        """完整对齐流程"""
        print("\n" + "="*50)
        print("开始精准对齐")
        print("="*50)

        # 步骤0: 透视效应补偿
        print("\n[步骤0] 透视效应补偿")
        current_height = self.get_current_height()
        if hasattr(self, '_setup_height') and self._setup_height is not None and current_height is not None:
            height_diff = current_height - self._setup_height
            print(f"  设置高度: {self._setup_height:.1f}mm, 当前高度: {current_height:.1f}mm")
            print(f"  高度差: {height_diff:.1f}mm")

            if abs(height_diff) > 10:
                print("  将在XY对齐时使用透视补偿")
                self._use_perspective_compensation = True
            else:
                print("  高度差较小，无需透视补偿")
                self._use_perspective_compensation = False
        else:
            print("  未记录设置高度或无法获取当前高度，跳过透视补偿")
            self._use_perspective_compensation = False

        print("\n[步骤1] 检查检测")
        height_ok = self.auto_adjust_height()

        if not height_ok:
            print("请手动调整高度")

        # 如果有设置的目标深度，执行Z轴对齐
        # 注意：默认情况下不会设置target_z，Z轴对齐被禁用
        # 原因：设置偏移量时的深度是卡槽内深度，不适合作为对齐目标
        z_ok = True
        if self.target_z is not None and self.z_controller is not None:
            print("\n[步骤2] Z轴对齐（闭环控制）")
            z_ok = self.align_z(tolerance_mm=1.0)
            if not z_ok:
                print("  ⚠ Z轴对齐未完成，继续XY对齐")
        else:
            print("\n[步骤2] Z轴对齐 - 跳过（对齐阶段禁用Z轴控制）")
            print("  提示: Z轴控制仅在放置阶段启用，避免对齐时下降到卡槽底部")

        print("\n[步骤3] XY对齐")
        align_ok = self.align_xy(tolerance_mm)

        # 最终结果
        final_ok = align_ok and z_ok

        if auto_place and final_ok:
            print("\n[步骤4] 自动放置")
            self.auto_place()

        return final_ok

    def test_detection(self):
        """测试检测（支持所有关节控制）"""
        print("\n" + "="*50)
        print("检测测试 - 完整关节控制")
        print("="*50)
        print("\n快捷键:")
        print("  [基础控制]")
        print("    q - 退出")
        print("    s - 保存当前位置为 'detection_best'")
        print("    h - 显示所有关节状态")
        print("\n  [高度/位置控制]")
        print("    r/l - 上升/下降 (joint_2 肩部俯仰, ±2°)")
        print("    w - 底座旋转 左 (joint_1, ±2°)")
        print("    a/d - 肩部前倾/后仰微调 (joint_2, ±1°)")
        print("\n  [手臂关节]")
        print("    z/x - 肘部 上/下 (joint_3, ±2°)")
        print("    f/v - 前臂 上/下 (joint_4, ±2°)")
        print("    t/g - 腕部俯仰 上/下 (joint_5, ±2°)")
        print("    y/b - 腕部旋转 左/右 (joint_6, ±2°)")
        print("\n  [腰部控制]")
        print("    u/j - 腰部旋转 左/右 (trunk_1, ±2°)")
        print("\n  [夹爪控制]")
        print("    o/c - 打开/闭合夹爪")
        print("\n  [预设位置]")
        print("    1 - 加载预设 'home'")
        print("    2 - 加载预设 'pickup'")
        print("="*50)

        # 移动步长（度）
        step_coarse = 2.0
        step_fine = 1.0

        # 关节索引映射（右手）
        joint_indices = {
            'base': 7,      # joint_1: 底座旋转
            'shoulder': 8,  # joint_2: 肩部俯仰
            'elbow': 9,     # joint_3: 肘部俯仰
            'forearm': 10,  # joint_4: 前臂俯仰
            'wrist': 11,    # joint_5: 腕部俯仰
            'wrist_rot': 12,# joint_6: 腕部旋转
            'trunk': 14,    # trunk_joint_1: 腰部旋转
        }

        # 关节名称显示
        joint_names = {
            joint_indices['base']: 'Base',
            joint_indices['shoulder']: 'Shoulder',
            joint_indices['elbow']: 'Elbow',
            joint_indices['forearm']: 'Forearm',
            joint_indices['wrist']: 'Wrist',
            joint_indices['wrist_rot']: 'WristRot',
            joint_indices['trunk']: 'Trunk',
        }

        # 显示关节状态的函数
        def display_joint_states(vis, joints):
            y = 80
            for idx, name in joint_names.items():
                if idx < len(joints):
                    text = f"{name}: {joints[idx]:.1f}°"
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y += 15

        # 移动单个关节的函数
        def move_single_joint(joint_idx, delta_deg):
            joints = self.get_joint_states()
            if joints is None or joint_idx >= len(joints):
                return False
            joints[joint_idx] += delta_deg
            action = {f"{name}.pos": float(joints[i])
                      for i, name in enumerate(self.robot.observation_joint_names)
                      if i < len(joints)}
            self.robot.send_action(action)
            time.sleep(0.1)  # 短暂延迟
            return True

        last_key_time = 0

        while True:
            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)
            vis = self.detector.visualize(image, state)

            # Z轴深度估计（如果有副相机和Z控制器）
            if self.camera2 is not None and self.z_controller is not None:
                image2 = self.camera2.read()
                if image2 is not None:
                    estimate = self.z_controller.estimate_z(image, image2, self.detector.workpiece_color)
                    if estimate.confidence > 0:
                        z_text = f"Z: {estimate.z:.1f}mm +/-{estimate.uncertainty:.1f} ({estimate.method})"
                        color = (0, 255, 255)  # 黄色
                    else:
                        z_text = f"Z: -- (no marker)"
                        color = (128, 128, 128)
                    cv2.putText(vis, z_text, (10, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 显示基本信息
            cv2.putText(vis, f"Arm: {self.arm}", (vis.shape[1]-100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cal_status = f"Cal Points: {len(self.calibration_points)}"
            cv2.putText(vis, cal_status, (vis.shape[1]-150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 获取并显示关节状态
            joints = self.get_joint_states()
            if joints is not None:
                display_joint_states(vis, joints)

            # 显示快捷键提示（简化版）
            help_text = "Q:exit S:save R/L:±height W:±base A/D:±shoulder Z/X:±elbow"
            cv2.putText(vis, help_text, (10, vis.shape[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            cv2.imshow("Detection Test", vis)

            key = cv2.waitKey(1) & 0xFF

            # 基础控制
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_preset('detection_best')
                print("\n✓ 位置已保存为 'detection_best'")
            elif key == ord('h'):
                if joints is not None:
                    print("\n当前关节状态:")
                    for i, name in enumerate(self.robot.observation_joint_names):
                        if i < len(joints):
                            print(f"  {name}: {joints[i]:.2f}°")

            # 高度控制（粗调）
            elif key == ord('r'):
                self.raise_height()
            elif key == ord('l'):
                self.lower_height()

            # 底座旋转
            elif key == ord('w'):
                move_single_joint(joint_indices['base'], -step_coarse)
                print(f"底座旋转左: {step_coarse}°")
            elif key == ord('e'):
                move_single_joint(joint_indices['base'], step_coarse)
                print(f"底座旋转右: {step_coarse}°")

            # 肩部俯仰微调
            elif key == ord('a'):
                move_single_joint(joint_indices['shoulder'], step_fine)
                print(f"肩部微调: {step_fine}°")
            elif key == ord('d'):
                move_single_joint(joint_indices['shoulder'], -step_fine)
                print(f"肩部微调: -{step_fine}°")

            # 肘部控制
            elif key == ord('z'):
                move_single_joint(joint_indices['elbow'], step_coarse)
                print(f"肘部: {step_coarse}°")
            elif key == ord('x'):
                move_single_joint(joint_indices['elbow'], -step_coarse)
                print(f"肘部: -{step_coarse}°")

            # 前臂控制 (使用 f/v 避免 c 冲突)
            elif key == ord('f'):
                move_single_joint(joint_indices['forearm'], step_coarse)
                print(f"前臂: {step_coarse}°")
            elif key == ord('v'):
                move_single_joint(joint_indices['forearm'], -step_coarse)
                print(f"前臂: -{step_coarse}°")

            # 腕部俯仰 (使用 t/g)
            elif key == ord('t'):
                move_single_joint(joint_indices['wrist'], step_coarse)
                print(f"腕部俯仰: {step_coarse}°")
            elif key == ord('g'):
                move_single_joint(joint_indices['wrist'], -step_coarse)
                print(f"腕部俯仰: -{step_coarse}°")

            # 腕部旋转 (使用 y/b)
            elif key == ord('y'):
                move_single_joint(joint_indices['wrist_rot'], step_coarse)
                print(f"腕部旋转: {step_coarse}°")
            elif key == ord('b'):
                move_single_joint(joint_indices['wrist_rot'], -step_coarse)
                print(f"腕部旋转: -{step_coarse}°")

            # 腰部控制
            elif key == ord('u'):
                move_single_joint(joint_indices['trunk'], step_coarse)
                print(f"腰部: {step_coarse}°")
            elif key == ord('j'):
                move_single_joint(joint_indices['trunk'], -step_coarse)
                print(f"腰部: -{step_coarse}°")

            # 夹爪控制
            elif key == ord('o'):
                self.open_gripper()
            elif key == ord('c'):
                self.close_gripper()

            # 预设位置
            elif key == ord('1'):
                print("\n加载预设 'home'...")
                self.load_preset('home')
            elif key == ord('2'):
                print("\n加载预设 'pickup'...")
                self.load_preset('pickup')

            last_key_time = time.time()

        cv2.destroyAllWindows()
