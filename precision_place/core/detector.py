"""
标记检测器 (Marker Detector)

检测图像中的彩色圆形标记。
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from precision_place.models.marker import Marker, DualMarkerState


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
        self.min_area = 100
        self.max_area = 50000

    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        """设置标记颜色"""
        self.workpiece_color = workpiece_color
        self.slot_color = slot_color

    def set_area_range(self, min_area: int, max_area: int):
        """设置标记检测的面积范围"""
        self.min_area = min_area
        self.max_area = max_area
        print(f"标记面积范围: {min_area} - {max_area} px²")

    def auto_adjust_area_range(self, marker_diameter_mm: float, distance_mm: float):
        """根据标记尺寸和距离自动调整面积范围"""
        fx = 311.0
        pixel_diameter = fx * marker_diameter_mm / distance_mm
        radius = pixel_diameter / 2
        expected_area = 3.14159 * radius * radius
        self.min_area = int(expected_area * 0.3)
        self.max_area = int(expected_area * 3)
        print(f"自动调整面积范围: {self.min_area} - {self.max_area} px²")

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
            allow_degraded: 是否允许退化模式

        Returns:
            DualMarkerState: 检测状态
        """
        state = DualMarkerState()

        # 工件标记
        wp_markers = self.detect_markers_by_color(image, self.workpiece_color)
        if len(wp_markers) >= 1:
            sorted_wp = sorted(wp_markers, key=lambda m: m.y)
            state.workpiece_1 = sorted_wp[0]
            if len(sorted_wp) >= 2:
                state.workpiece_2 = sorted_wp[1]
            if len(sorted_wp) >= 3:
                state.workpiece_3 = sorted_wp[2]
            state.workpiece_detected = True

        # 卡槽标记
        sl_markers = self.detect_markers_by_color(image, self.slot_color)
        if len(sl_markers) >= 1:
            sorted_sl = sorted(sl_markers, key=lambda m: m.y)
            state.slot_1 = sorted_sl[0]
            if len(sorted_sl) >= 2:
                state.slot_2 = sorted_sl[1]
            if len(sorted_sl) >= 3:
                state.slot_3 = sorted_sl[2]
            state.slot_detected = True

        # 检查退化模式
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
        """
        state1 = self.detect_triple_marker_state(image1)

        if image2 is None:
            return state1

        if state1.slot_marker_count >= 2 and not state1.degraded_mode:
            return state1

        state2 = self.detect_triple_marker_state(image2)

        if state2.slot_marker_count > state1.slot_marker_count:
            if flip_secondary:
                h, w = image2.shape[:2]
                center_x, center_y = w / 2, h / 2

                def flip_marker(m):
                    if m is None:
                        return None
                    return Marker(
                        x=2 * center_x - m.x,
                        y=2 * center_y - m.y,
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

            if state1.workpiece_detected and state1.slot_detected:
                self._calculate_alignment(state1)
                state1.degraded_mode = state1.slot_marker_count < 2

        return state1

    def detect_dual_marker_state(self, image: np.ndarray) -> DualMarkerState:
        """检测标记状态（兼容旧接口）"""
        return self.detect_triple_marker_state(image)

    def _calculate_alignment(self, state: DualMarkerState):
        """计算对齐误差"""
        wp_markers = [m for m in state.workpiece_markers if m]
        sl_markers = [m for m in state.slot_markers if m]

        if not wp_markers or not sl_markers:
            return

        # 计算中心
        wp_cx = sum(m.x for m in wp_markers) / len(wp_markers)
        wp_cy = sum(m.y for m in wp_markers) / len(wp_markers)
        sl_cx = sum(m.x for m in sl_markers) / len(sl_markers)
        sl_cy = sum(m.y for m in sl_markers) / len(sl_markers)

        state.offset_x = sl_cx - wp_cx
        state.offset_y = sl_cy - wp_cy

        # 计算旋转角度
        if len(wp_markers) >= 2 and len(sl_markers) >= 2:
            wp_top = wp_markers[0]
            wp_bottom = wp_markers[-1]
            sl_top = sl_markers[0]
            sl_bottom = sl_markers[-1]

            wp_angle = np.degrees(np.arctan2(
                wp_bottom.x - wp_top.x,
                wp_bottom.y - wp_top.y
            ))
            sl_angle = np.degrees(np.arctan2(
                sl_bottom.x - sl_top.x,
                sl_bottom.y - sl_top.y
            ))
            state.rotation_error = wp_angle - sl_angle
            while state.rotation_error > 180:
                state.rotation_error -= 360
            while state.rotation_error < -180:
                state.rotation_error += 360

        # 计算质量
        wp_conf = sum(m.confidence for m in wp_markers) / len(wp_markers)
        sl_conf = sum(m.confidence for m in sl_markers) / len(sl_markers)
        state.alignment_quality = (wp_conf + sl_conf) / 2

    def visualize(self, image: np.ndarray, state: DualMarkerState = None,
                  target_offset_x: float = 0.0, target_offset_y: float = 0.0) -> np.ndarray:
        """可视化检测结果"""
        vis = image.copy()

        if state is None:
            state = self.detect_triple_marker_state(image)

        # 工件标记（绿色）
        for i, m in enumerate(state.workpiece_markers, 1):
            if m:
                cv2.circle(vis, (int(m.x), int(m.y)), 10, (0, 255, 0), 2)
                cv2.putText(vis, f"WP{i}", (int(m.x)-15, int(m.y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 卡槽标记（红色）
        for i, m in enumerate(state.slot_markers, 1):
            if m:
                cv2.circle(vis, (int(m.x), int(m.y)), 10, (0, 0, 255), 2)
                cv2.putText(vis, f"SL{i}", (int(m.x)-15, int(m.y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 绘制中心和偏移
        if state.workpiece_detected and state.slot_detected:
            wp_center = state.workpiece_center
            sl_center = state.slot_center

            cv2.circle(vis, (int(wp_center[0]), int(wp_center[1])), 5, (0, 255, 0), -1)
            cv2.circle(vis, (int(sl_center[0]), int(sl_center[1])), 5, (0, 0, 255), -1)

            # 偏移向量
            cv2.arrowedLine(vis,
                           (int(wp_center[0]), int(wp_center[1])),
                           (int(sl_center[0]), int(sl_center[1])),
                           (255, 255, 0), 2)

            # 显示偏移信息
            cv2.putText(vis, f"Offset: ({state.offset_x:.1f}, {state.offset_y:.1f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return vis