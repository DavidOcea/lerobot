"""
Slot Detector - 卡槽检测器

针对带定位销的卡槽进行视觉检测和定位
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class SlotPosition:
    """卡槽位置信息"""
    center_x: float      # 中心X像素坐标
    center_y: float      # 中心Y像素坐标
    rotation: float      # 旋转角度（度）
    confidence: float    # 检测置信度
    pin_positions: Optional[List[Tuple[float, float]]] = None  # 定位销位置


class SlotDetector:
    """
    卡槽检测器
    
    支持多种检测方法：
    1. 模板匹配 - 适合固定形状的卡槽
    2. 边缘检测 + 霍夫圆 - 适合带定位销的卡槽
    3. 特征点匹配 - 适合有纹理的卡槽
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # 目标模板
        self.target_template = None
        self.template_mask = None
        
        # 检测方法
        self.detection_method = "template"  # template, edge, feature
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('slot_detector', {})
        return {}
    
    def save_target_template(self, image: np.ndarray, 
                             center: Tuple[int, int] = None,
                             size: Tuple[int, int] = None) -> np.ndarray:
        """
        保存目标卡槽模板
        
        Args:
            image: 包含卡槽的图像
            center: 模板中心位置（默认图像中心）
            size: 模板大小（默认配置值）
        
        Returns:
            保存的模板图像
        """
        if center is None:
            center = (image.shape[1] // 2, image.shape[0] // 2)
        if size is None:
            size = tuple(self.config.get('template_size', [120, 120]))
        
        w, h = size
        cx, cy = center
        
        # 提取模板区域
        y1 = max(0, cy - h // 2)
        y2 = min(image.shape[0], cy + h // 2)
        x1 = max(0, cx - w // 2)
        x2 = min(image.shape[1], cx + w // 2)
        
        self.target_template = image[y1:y2, x1:x2].copy()
        
        # 创建边缘掩码（可选，用于提高匹配精度）
        gray = cv2.cvtColor(self.target_template, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        self.template_mask = edges
        
        print(f"模板已保存: 大小 {self.target_template.shape[:2]}")
        return self.target_template
    
    def load_target_template(self, path: str) -> np.ndarray:
        """加载已保存的模板"""
        self.target_template = cv2.imread(path)
        if self.target_template is None:
            raise FileNotFoundError(f"无法加载模板: {path}")
        
        gray = cv2.cvtColor(self.target_template, cv2.COLOR_BGR2GRAY)
        self.template_mask = cv2.Canny(gray, 50, 150)
        
        print(f"模板已加载: {path}")
        return self.target_template
    
    def detect_slot(self, image: np.ndarray, 
                    method: str = "auto") -> Optional[SlotPosition]:
        """
        检测卡槽位置
        
        Args:
            image: 输入图像
            method: 检测方法 ("auto", "template", "edge", "pin")
        
        Returns:
            SlotPosition 或 None
        """
        if method == "auto":
            method = self._select_best_method(image)
        
        if method == "template":
            return self._detect_by_template(image)
        elif method == "edge":
            return self._detect_by_edge(image)
        elif method == "pin":
            return self._detect_by_pins(image)
        else:
            return self._detect_by_template(image)
    
    def _select_best_method(self, image: np.ndarray) -> str:
        """自动选择最佳检测方法"""
        # 简单策略：如果有模板就用模板匹配
        if self.target_template is not None:
            return "template"
        return "edge"
    
    def _detect_by_template(self, image: np.ndarray) -> Optional[SlotPosition]:
        """模板匹配检测"""
        if self.target_template is None:
            raise ValueError("请先保存或加载目标模板")
        
        # 转灰度
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(self.target_template, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 检查置信度
        threshold = self.config.get('match_threshold', 0.7)
        if max_val < threshold:
            print(f"匹配置信度 {max_val:.2f} 低于阈值 {threshold}")
            return None
        
        # 计算模板中心在图像中的位置
        th, tw = gray_template.shape
        center_x = max_loc[0] + tw // 2
        center_y = max_loc[1] + th // 2
        
        # 计算旋转（简化：假设无旋转）
        rotation = 0.0
        
        return SlotPosition(
            center_x=center_x,
            center_y=center_y,
            rotation=rotation,
            confidence=max_val
        )
    
    def _detect_by_edge(self, image: np.ndarray) -> Optional[SlotPosition]:
        """边缘检测方法"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        threshold1 = self.config.get('canny_threshold1', 50)
        threshold2 = self.config.get('canny_threshold2', 150)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的矩形轮廓（假设是卡槽）
        max_contour = max(contours, key=cv2.contourArea)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(max_contour)
        center = rect[0]
        rotation = rect[2]
        
        # 计算置信度（基于轮廓面积）
        area = cv2.contourArea(max_contour)
        confidence = min(1.0, area / 10000)  # 归一化
        
        return SlotPosition(
            center_x=center[0],
            center_y=center[1],
            rotation=rotation,
            confidence=confidence
        )
    
    def _detect_by_pins(self, image: np.ndarray) -> Optional[SlotPosition]:
        """通过定位销检测卡槽位置"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 霍夫圆检测
        min_radius = self.config.get('pin_min_radius', 5)
        max_radius = self.config.get('pin_max_radius', 15)
        threshold = self.config.get('hough_threshold', 30)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=threshold,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is None or len(circles[0]) < 2:
            print("未检测到足够的定位销")
            return self._detect_by_edge(image)  # 回退到边缘检测
        
        circles = circles[0]
        
        # 如果检测到2个或更多圆，计算中心
        if len(circles) >= 2:
            # 取前两个圆（假设是两个定位销）
            pin1 = circles[0][:2]
            pin2 = circles[1][:2]
            
            # 计算中心点
            center_x = (pin1[0] + pin2[0]) / 2
            center_y = (pin1[1] + pin2[1]) / 2
            
            # 计算旋转角度
            dx = pin2[0] - pin1[0]
            dy = pin2[1] - pin1[1]
            rotation = np.degrees(np.arctan2(dy, dx))
            
            return SlotPosition(
                center_x=center_x,
                center_y=center_y,
                rotation=rotation,
                confidence=0.8,
                pin_positions=[(pin1[0], pin1[1]), (pin2[0], pin2[1])]
            )
        
        return None
    
    def calculate_offset(self, image: np.ndarray, 
                         image_center: Tuple[float, float] = None) -> Tuple[float, float, float]:
        """
        计算当前位置相对于目标位置的偏移
        
        Args:
            image: 当前图像
            image_center: 图像中心（默认为图像中心）
        
        Returns:
            (offset_x, offset_y, confidence): 像素偏移和置信度
        """
        if image_center is None:
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
        
        slot_pos = self.detect_slot(image)
        
        if slot_pos is None:
            return 0, 0, 0
        
        offset_x = slot_pos.center_x - image_center[0]
        offset_y = slot_pos.center_y - image_center[1]
        
        return offset_x, offset_y, slot_pos.confidence
    
    def visualize_detection(self, image: np.ndarray, 
                           slot_pos: SlotPosition = None) -> np.ndarray:
        """可视化检测结果"""
        vis_image = image.copy()
        
        if slot_pos is None:
            slot_pos = self.detect_slot(image)
        
        if slot_pos is None:
            cv2.putText(vis_image, "No detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image
        
        # 绘制中心点
        cv2.circle(vis_image, 
                  (int(slot_pos.center_x), int(slot_pos.center_y)), 
                  10, (0, 255, 0), -1)
        
        # 绘制十字线
        cv2.line(vis_image, 
                (int(slot_pos.center_x) - 20, int(slot_pos.center_y)),
                (int(slot_pos.center_x) + 20, int(slot_pos.center_y)),
                (0, 255, 0), 2)
        cv2.line(vis_image,
                (int(slot_pos.center_x), int(slot_pos.center_y) - 20),
                (int(slot_pos.center_x), int(slot_pos.center_y) + 20),
                (0, 255, 0), 2)
        
        # 绘制定位销（如果有）
        if slot_pos.pin_positions:
            for px, py in slot_pos.pin_positions:
                cv2.circle(vis_image, (int(px), int(py)), 8, (255, 0, 0), 2)
        
        # 显示信息
        text = f"Conf: {slot_pos.confidence:.2f}"
        cv2.putText(vis_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
