"""
Marker Detector - 彩色标记检测器

用于检测卡槽上的彩色标记贴纸
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class MarkerPosition:
    """标记位置信息"""
    center_x: float
    center_y: float
    radius: float
    color: str
    confidence: float


class ColorMarkerDetector:
    """
    彩色标记检测器
    
    检测卡槽上的彩色圆形贴纸标记
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # 颜色范围 (HSV)
        self.color_ranges = self._get_color_ranges()
        
        # 检测参数
        marker_config = self.config.get('marker_detector', {})
        self.min_area = marker_config.get('min_area', 100)
        self.max_area = marker_config.get('max_area', 5000)
        self.circularity_threshold = marker_config.get('circularity', 0.7)
        
        # 目标标记颜色
        self.target_color = "green"  # 默认绿色
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_color_ranges(self) -> dict:
        """获取颜色范围配置"""
        marker_config = self.config.get('marker_detector', {})
        
        ranges = {}
        for color_name in ['red', 'green', 'blue']:
            if color_name in marker_config:
                color_cfg = marker_config[color_name]
                ranges[color_name] = {
                    'lower': np.array(color_cfg['lower']),
                    'upper': np.array(color_cfg['upper'])
                }
                # 红色有两组范围
                if 'lower2' in color_cfg:
                    ranges[color_name]['lower2'] = np.array(color_cfg['lower2'])
                    ranges[color_name]['upper2'] = np.array(color_cfg['upper2'])
        
        return ranges
    
    def set_target_color(self, color: str):
        """设置目标标记颜色"""
        if color not in self.color_ranges:
            raise ValueError(f"不支持的颜色: {color}, 支持: {list(self.color_ranges.keys())}")
        self.target_color = color
        print(f"目标标记颜色已设置为: {color}")
    
    def detect_marker(self, image: np.ndarray, 
                      color: str = None) -> Optional[MarkerPosition]:
        """
        检测图像中的彩色标记
        
        Args:
            image: BGR图像
            color: 目标颜色，None则使用预设颜色
        
        Returns:
            MarkerPosition 或 None
        """
        if color is None:
            color = self.target_color
        
        if color not in self.color_ranges:
            return None
        
        # 转HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        color_range = self.color_ranges[color]
        mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        
        # 红色需要两组范围
        if 'lower2' in color_range:
            mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
            mask = cv2.bitwise_or(mask, mask2)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最佳的圆形轮廓
        best_marker = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.min_area or area > self.max_area:
                continue
            
            # 计算圆度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity < self.circularity_threshold:
                continue
            
            # 计算中心和半径
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # 计算置信度
            confidence = circularity * min(1.0, area / 500)
            
            if confidence > best_score:
                best_score = confidence
                best_marker = MarkerPosition(
                    center_x=cx,
                    center_y=cy,
                    radius=radius,
                    color=color,
                    confidence=confidence
                )
        
        return best_marker
    
    def detect_all_colors(self, image: np.ndarray) -> List[MarkerPosition]:
        """检测所有支持的颜色标记"""
        markers = []
        
        for color in self.color_ranges.keys():
            marker = self.detect_marker(image, color)
            if marker is not None:
                markers.append(marker)
        
        return markers
    
    def calculate_offset(self, image: np.ndarray,
                        image_center: Tuple[float, float] = None,
                        color: str = None) -> Tuple[float, float, float]:
        """
        计算标记相对于图像中心的偏移
        
        Returns:
            (offset_x, offset_y, confidence)
        """
        if image_center is None:
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
        
        marker = self.detect_marker(image, color)
        
        if marker is None:
            return (0, 0, 0)
        
        offset_x = marker.center_x - image_center[0]
        offset_y = marker.center_y - image_center[1]
        
        return (offset_x, offset_y, marker.confidence)
    
    def visualize(self, image: np.ndarray, marker: MarkerPosition = None) -> np.ndarray:
        """可视化检测结果"""
        vis = image.copy()
        
        if marker is None:
            marker = self.detect_marker(image)
        
        if marker is None:
            cv2.putText(vis, "No marker detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis
        
        # 画标记中心
        cv2.circle(vis, (int(marker.center_x), int(marker.center_y)),
                  int(marker.radius), (0, 255, 0), 2)
        cv2.circle(vis, (int(marker.center_x), int(marker.center_y)),
                  5, (0, 0, 255), -1)
        
        # 画十字线
        cx, cy = int(marker.center_x), int(marker.center_y)
        cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        
        # 显示信息
        cv2.putText(vis, f"Color: {marker.color}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Conf: {marker.confidence:.2f}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def calibrate_color(self, image: np.ndarray) -> dict:
        """
        交互式颜色标定
        
        使用方法：
        1. 采集包含标记的图像
        2. 用鼠标框选标记区域
        3. 系统自动计算颜色范围
        """
        print("\n颜色标定工具")
        print("请在图像中框选标记区域")
        
        roi = cv2.selectROI("Select Marker", image, False)
        cv2.destroyWindow("Select Marker")
        
        if roi[2] == 0 or roi[3] == 0:
            print("未选择区域")
            return None
        
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        
        # 转HSV
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # 计算颜色范围
        h_min, s_min, v_min = hsv[:,:,0].min(), hsv[:,:,1].min(), hsv[:,:,2].min()
        h_max, s_max, v_max = hsv[:,:,0].max(), hsv[:,:,1].max(), hsv[:,:,2].max()
        
        # 添加容差
        tolerance = 20
        lower = np.array([max(0, h_min - tolerance), 
                         max(50, s_min - tolerance), 
                         max(50, v_min - tolerance)])
        upper = np.array([min(180, h_max + tolerance), 
                         255, 255])
        
        print(f"\n检测到的颜色范围 (HSV):")
        print(f"  Lower: {lower.tolist()}")
        print(f"  Upper: {upper.tolist()}")
        
        # 测试检测
        test_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        test_mask = cv2.inRange(test_hsv, lower, upper)
        
        cv2.imshow("Detection Result", test_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("Detection Result")
        
        return {'lower': lower.tolist(), 'upper': upper.tolist()}


class MarkerVisualServo:
    """
    基于标记的视觉伺服控制器
    """
    
    def __init__(self, robot, camera, arm: str = "right", config_path: str = None):
        self.robot = robot
        self.camera = camera
        self.arm = arm
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建检测器
        self.detector = ColorMarkerDetector(config_path)
        
        # 参数
        vs_config = self.config.get('visual_servo', {})
        self.pixel_to_mm_ratio = vs_config.get('pixel_to_mm_ratio', 0.5)
        self.gain = vs_config.get('gain', 0.6)
        self.tolerance_mm = vs_config.get('tolerance_mm', 2.0)
        self.max_iterations = vs_config.get('max_iterations', 15)
        self.settle_time = vs_config.get('settle_time', 0.3)
        
        # 关节灵敏度
        jc_config = self.config.get('joint_controller', {})
        self.joint_sensitivity = jc_config.get(f'{arm}_arm', {})
    
    def _load_config(self, config_path: str) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def set_pixel_to_mm_ratio(self, ratio: float):
        self.pixel_to_mm_ratio = ratio
    
    def set_target_color(self, color: str):
        self.detector.set_target_color(color)
    
    def get_current_error(self) -> Tuple[float, float, float]:
        """获取当前误差"""
        image = self.camera.read()
        return self.detector.calculate_offset(image)
    
    def servo_step(self) -> Tuple[float, float, bool]:
        """执行一步视觉伺服"""
        # 获取误差
        offset_x, offset_y, confidence = self.get_current_error()
        
        # 计算毫米误差
        mm_x = offset_x * self.pixel_to_mm_ratio
        mm_y = offset_y * self.pixel_to_mm_ratio
        error_mm = np.sqrt(mm_x**2 + mm_y**2)
        
        print(f"  像素偏移: ({offset_x:.1f}, {offset_y:.1f})")
        print(f"  毫米误差: ({mm_x:.2f}, {mm_y:.2f}), 总误差: {error_mm:.2f}mm")
        print(f"  置信度: {confidence:.2f}")
        
        converged = error_mm < self.tolerance_mm
        
        if not converged and confidence > 0.3:
            self._apply_adjustment(mm_x * self.gain, mm_y * self.gain)
        
        return error_mm, confidence, converged
    
    def _apply_adjustment(self, mm_x: float, mm_y: float):
        """应用关节调整"""
        # 获取当前关节位置
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            print("关节位置获取失败")
            return
        
        # 计算关节调整量
        if self.arm == "right":
            # 右臂关节索引: 7-12
            joints[7] -= mm_x * self.joint_sensitivity.get('joint_1', 0.15)
            joints[8] -= mm_y * self.joint_sensitivity.get('joint_2', 0.20)
        else:
            # 左臂关节索引: 0-5
            joints[0] += mm_x * self.joint_sensitivity.get('joint_1', 0.15)
            joints[1] -= mm_y * self.joint_sensitivity.get('joint_2', 0.20)
        
        # 发送命令
        self.robot.send_action({'action': joints.tolist()})
        
        # 等待稳定
        import time
        time.sleep(self.settle_time)
    
    def servo_to_target(self, tolerance_mm: float = None, 
                        max_iterations: int = None) -> bool:
        """执行视觉伺服直到达到目标"""
        if tolerance_mm is None:
            tolerance_mm = self.tolerance_mm
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        print(f"\n开始视觉伺服，目标精度: {tolerance_mm}mm")
        
        for i in range(max_iterations):
            print(f"\n[迭代 {i+1}/{max_iterations}]")
            
            error_mm, confidence, converged = self.servo_step()
            
            if converged:
                print(f"\n✓ 达到目标精度: {error_mm:.2f}mm")
                return True
        
        print(f"\n✗ 未达到目标精度")
        return False
