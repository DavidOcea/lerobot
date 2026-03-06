"""
Layered Alignment - 分层对齐检测器

用于检测工件定位孔和卡槽定位销的对齐状态
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class AlignmentState:
    """对齐状态"""
    workpiece_visible: bool
    pin_visible: bool
    offset_x: float          # 像素偏移
    offset_y: float
    confidence: float


class LayeredAlignmentDetector:
    """
    分层对齐检测器
    
    同时检测：
    - 工件上的定位孔标记
    - 卡槽上的定位销/标记
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # 颜色范围 (HSV)
        self.color_ranges = {
            'green': {
                'lower': np.array([35, 50, 50]),
                'upper': np.array([85, 255, 255])
            },
            'red': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]),
                'upper2': np.array([180, 255, 255])
            },
            'blue': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'yellow': {
                'lower': np.array([20, 50, 50]),
                'upper': np.array([35, 255, 255])
            }
        }
        
        # 标记配置
        self.workpiece_marker_color = "green"
        self.slot_marker_color = "red"
        
        # 是否启用白色定位销检测
        self.pin_detector_enabled = True
        self.pin_brightness_threshold = 180
        self.pin_min_length = 20
        self.pin_max_length = 100
        
        # 检测参数
        self.min_area = 50
        self.max_area = 3000
        self.circularity_threshold = 0.6
    
    def _load_config(self, config_path: str) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        """设置标记颜色"""
        if workpiece_color not in self.color_ranges:
            raise ValueError(f"不支持的工件标记颜色: {workpiece_color}")
        if slot_color not in self.color_ranges:
            raise ValueError(f"不支持的卡槽标记颜色: {slot_color}")
        
        self.workpiece_marker_color = workpiece_color
        self.slot_marker_color = slot_color
        print(f"标记颜色设置: 工件={workpiece_color}, 卡槽={slot_color}")
    
    def detect_color_marker(self, image: np.ndarray, color: str) -> Optional[Tuple[float, float, float]]:
        """检测指定颜色的标记"""
        if color not in self.color_ranges:
            return None
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_range = self.color_ranges[color]
        mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        
        if 'lower2' in color_range:
            mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
            mask = cv2.bitwise_or(mask, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_result = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity < self.circularity_threshold:
                continue
            
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            confidence = circularity * min(1.0, area / 500)
            
            if confidence > best_score:
                best_score = confidence
                best_result = (cx, cy, confidence)
        
        return best_result
    
    def detect_white_pin(self, image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """检测白色定位销"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, bright_mask = cv2.threshold(gray, self.pin_brightness_threshold, 255, cv2.THRESH_BINARY)
        
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        enhanced = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_v)
        
        contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_result = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30 or area > 1000:
                continue
            
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect
            
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio < 2:
                continue
            
            length = max(w, h)
            if length < self.pin_min_length or length > self.pin_max_length:
                continue
            
            confidence = min(1.0, aspect_ratio / 5) * min(1.0, area / 200)
            
            if confidence > best_score:
                best_score = confidence
                best_result = (cx, cy, confidence)
        
        return best_result
    
    def detect_alignment_state(self, image: np.ndarray) -> AlignmentState:
        """检测当前对齐状态"""
        workpiece = self.detect_color_marker(image, self.workpiece_marker_color)
        
        slot = self.detect_color_marker(image, self.slot_marker_color)
        if slot is None and self.pin_detector_enabled:
            slot = self.detect_white_pin(image)
        
        state = AlignmentState(
            workpiece_visible=workpiece is not None,
            pin_visible=slot is not None,
            offset_x=0,
            offset_y=0,
            confidence=0
        )
        
        if workpiece and slot:
            state.offset_x = slot[0] - workpiece[0]
            state.offset_y = slot[1] - workpiece[1]
            state.confidence = (workpiece[2] + slot[2]) / 2
        
        return state
    
    def visualize(self, image: np.ndarray) -> np.ndarray:
        """可视化检测结果"""
        vis = image.copy()
        
        workpiece = self.detect_color_marker(image, self.workpiece_marker_color)
        slot = self.detect_color_marker(image, self.slot_marker_color)
        if slot is None and self.pin_detector_enabled:
            slot = self.detect_white_pin(image)
        
        # 画工件标记
        if workpiece:
            cx, cy, conf = workpiece
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 255, 0), 2)
            cv2.putText(vis, "Hole", (int(cx)-20, int(cy)-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 画定位销
        if slot:
            cx, cy, conf = slot
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 0, 255), 2)
            cv2.putText(vis, "Pin", (int(cx)-15, int(cy)-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 画连线
        if workpiece and slot:
            cv2.line(vis, (int(workpiece[0]), int(workpiece[1])),
                    (int(slot[0]), int(slot[1])), (255, 255, 0), 2)
            
            offset_x = slot[0] - workpiece[0]
            offset_y = slot[1] - workpiece[1]
            cv2.putText(vis, f"Offset: ({offset_x:.0f}, {offset_y:.0f})", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 状态
        y = 45
        cv2.putText(vis, f"Hole: {'OK' if workpiece else 'NO'}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if workpiece else (0, 0, 255), 2)
        cv2.putText(vis, f"Pin: {'OK' if slot else 'NO'}", (10, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if slot else (0, 0, 255), 2)
        
        return vis
