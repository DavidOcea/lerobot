"""
Pin Detector - 白色针状定位销检测器

针对低对比度场景的定位销检测
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PinPosition:
    """定位销位置信息"""
    tip_x: float          # 针尖X坐标
    tip_y: float          # 针尖Y坐标
    base_x: float         # 针根X坐标
    base_y: float         # 针根Y坐标
    angle: float          # 倾斜角度
    length: float         # 长度（像素）
    confidence: float     # 置信度


class WhitePinDetector:
    """
    白色针状定位销检测器
    
    检测策略：
    1. 亮度增强 - 提取高亮区域
    2. 形态学处理 - 增强细长特征
    3. 霍夫线检测 - 检测直线段
    4. 端点检测 - 找到针尖位置
    """
    
    def __init__(self):
        # 检测参数
        self.brightness_threshold = 180  # 亮度阈值
        self.min_pin_length = 20         # 最小针长度（像素）
        self.max_pin_length = 100        # 最大针长度
        self.hough_threshold = 30        # 霍夫变换阈值
        
    def detect_pins(self, image: np.ndarray) -> List[PinPosition]:
        """检测图像中的白色针状定位销"""
        pins1 = self._detect_by_brightness(image)
        pins2 = self._detect_by_lines(image)
        return self._merge_results(pins1, pins2)
    
    def _detect_by_brightness(self, image: np.ndarray) -> List[PinPosition]:
        """基于亮度的检测方法"""
        # 转HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # 亮度阈值
        _, mask = cv2.threshold(v_channel, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        enhanced = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)
        
        # 查找轮廓
        contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pins = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 2000:
                continue
            
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect
            
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio < 3:
                continue
            
            length = max(w, h)
            
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            max_dist = 0
            tip_idx, base_idx = 0, 1
            for i in range(4):
                for j in range(i+1, 4):
                    dist = np.linalg.norm(box[i] - box[j])
                    if dist > max_dist:
                        max_dist = dist
                        tip_idx, base_idx = i, j
            
            tip = box[tip_idx]
            base = box[base_idx]
            
            confidence = min(1.0, area / 200) * min(1.0, aspect_ratio / 5)
            
            pins.append(PinPosition(
                tip_x=float(tip[0]),
                tip_y=float(tip[1]),
                base_x=float(base[0]),
                base_y=float(base[1]),
                angle=angle,
                length=length,
                confidence=confidence
            ))
        
        return pins
    
    def _detect_by_lines(self, image: np.ndarray) -> List[PinPosition]:
        """基于霍夫线的检测方法"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_pin_length,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        pins = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length < self.min_pin_length or length > self.max_pin_length:
                continue
            
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            
            pins.append(PinPosition(
                tip_x=float(x1), tip_y=float(y1),
                base_x=float(x2), base_y=float(y2),
                angle=angle, length=length,
                confidence=0.6
            ))
        
        return pins
    
    def _merge_results(self, pins1: List[PinPosition], pins2: List[PinPosition]) -> List[PinPosition]:
        """合并两种检测方法的结果"""
        if not pins1:
            return pins2
        if not pins2:
            return pins1
        
        merged = list(pins1)
        for pin2 in pins2:
            is_duplicate = False
            for pin1 in merged:
                dist = np.sqrt((pin1.tip_x - pin2.tip_x)**2 + (pin1.tip_y - pin2.tip_y)**2)
                if dist < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(pin2)
        
        return merged
    
    def detect_pin_center(self, image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """检测定位销的中心位置"""
        pins = self.detect_pins(image)
        
        if not pins:
            return None
        
        if len(pins) == 1:
            pin = pins[0]
            cx = (pin.tip_x + pin.base_x) / 2
            cy = (pin.tip_y + pin.base_y) / 2
            return (cx, cy, pin.confidence)
        
        pin1, pin2 = pins[0], pins[1]
        cx = (pin1.tip_x + pin2.tip_x) / 2
        cy = (pin1.tip_y + pin2.tip_y) / 2
        confidence = (pin1.confidence + pin2.confidence) / 2
        
        return (cx, cy, confidence)
    
    def calculate_offset(self, image: np.ndarray, 
                        image_center: Tuple[float, float] = None) -> Tuple[float, float, float]:
        """计算当前位置相对于图像中心的偏移"""
        if image_center is None:
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
        
        result = self.detect_pin_center(image)
        
        if result is None:
            return (0, 0, 0)
        
        cx, cy, confidence = result
        offset_x = cx - image_center[0]
        offset_y = cy - image_center[1]
        
        return (offset_x, offset_y, confidence)
    
    def visualize(self, image: np.ndarray) -> np.ndarray:
        """可视化检测结果"""
        vis = image.copy()
        pins = self.detect_pins(image)
        
        for pin in pins:
            cv2.line(vis, (int(pin.tip_x), int(pin.tip_y)), 
                    (int(pin.base_x), int(pin.base_y)), (0, 255, 0), 2)
            cv2.circle(vis, (int(pin.tip_x), int(pin.tip_y)), 5, (0, 0, 255), -1)
        
        if pins:
            result = self.detect_pin_center(image)
            if result:
                cx, cy, conf = result
                cv2.circle(vis, (int(cx), int(cy)), 8, (255, 0, 0), 2)
                cv2.putText(vis, f"Conf: {conf:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def calibrate_brightness(self, image: np.ndarray) -> int:
        """交互式标定亮度阈值"""
        def nothing(x):
            pass
        
        cv2.namedWindow('Calibration')
        cv2.createTrackbar('Brightness', 'Calibration', 180, 255, nothing)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        while True:
            thresh = cv2.getTrackBarPos('Brightness', 'Calibration')
            _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            cv2.imshow('Calibration', mask)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('s'):
                self.brightness_threshold = thresh
                print(f"亮度阈值已保存: {thresh}")
                break
        
        cv2.destroyWindow('Calibration')
        return self.brightness_threshold


class DualPinSlotDetector:
    """双定位销卡槽检测器"""
    
    def __init__(self):
        self.pin_detector = WhitePinDetector()
        self.target_pin_positions = None
        
    def save_target_pins(self, image: np.ndarray):
        """保存目标位置的定位销位置"""
        pins = self.pin_detector.detect_pins(image)
        
        if len(pins) < 2:
            print(f"警告: 只检测到 {len(pins)} 个定位销")
            self.target_pin_positions = pins
        else:
            self.target_pin_positions = pins[:2]
            print(f"已保存 {len(self.target_pin_positions)} 个定位销位置")
        
        return self.target_pin_positions
    
    def calculate_offset(self, current_image: np.ndarray) -> Tuple[float, float, float]:
        """计算当前图像相对于目标位置的偏移"""
        if self.target_pin_positions is None or len(self.target_pin_positions) < 2:
            return self.pin_detector.calculate_offset(current_image)
        
        current_pins = self.pin_detector.detect_pins(current_image)
        
        if len(current_pins) < 2:
            return (0, 0, 0)
        
        target_cx = (self.target_pin_positions[0].tip_x + self.target_pin_positions[1].tip_x) / 2
        target_cy = (self.target_pin_positions[0].tip_y + self.target_pin_positions[1].tip_y) / 2
        
        current_cx = (current_pins[0].tip_x + current_pins[1].tip_x) / 2
        current_cy = (current_pins[0].tip_y + current_pins[1].tip_y) / 2
        
        offset_x = target_cx - current_cx
        offset_y = target_cy - current_cy
        
        confidence = (current_pins[0].confidence + current_pins[1].confidence) / 2
        
        return (offset_x, offset_y, confidence)
