"""
Dual Marker Visual Servo - 双标记视觉伺服

用于同时检测工件标记和卡槽标记，实现精准对齐
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class AlignmentResult:
    """对齐结果"""
    offset_x: float          # X方向偏移（像素）
    offset_y: float          # Y方向偏移（像素）
    rotation: float          # 旋转角度（度）
    confidence: float        # 置信度
    workpiece_detected: bool # 是否检测到工件标记
    slot_detected: bool      # 是否检测到卡槽标记


class DualMarkerDetector:
    """
    双标记检测器
    
    同时检测：
    - 工件上的标记（代表定位孔位置）
    - 卡槽上的标记（代表定位销位置）
    
    计算两者的对齐偏移量
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
        self.workpiece_marker_color = "green"   # 工件标记颜色
        self.slot_marker_color = "red"          # 卡槽标记颜色
        
        # 定位孔相对于工件标记的偏移（需要标定）
        # 如果标记就贴在定位孔旁边，这个偏移为0
        self.hole_offset_from_marker = (0, 0)   # (x_mm, y_mm)
        
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
    
    def set_hole_offset(self, x_mm: float, y_mm: float):
        """设置定位孔相对于工件标记的偏移"""
        self.hole_offset_from_marker = (x_mm, y_mm)
    
    def detect_marker_by_color(self, image: np.ndarray, color: str) -> Optional[Tuple[float, float, float]]:
        """
        检测指定颜色的标记
        
        Returns:
            (center_x, center_y, confidence) 或 None
        """
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
        
        # 找最佳轮廓
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
    
    def detect_both_markers(self, image: np.ndarray) -> AlignmentResult:
        """
        检测工件和卡槽两个标记
        
        Returns:
            AlignmentResult 包含对齐信息
        """
        # 检测工件标记
        workpiece = self.detect_marker_by_color(image, self.workpiece_marker_color)
        
        # 检测卡槽标记
        slot = self.detect_marker_by_color(image, self.slot_marker_color)
        
        # 构建结果
        result = AlignmentResult(
            offset_x=0,
            offset_y=0,
            rotation=0,
            confidence=0,
            workpiece_detected=workpiece is not None,
            slot_detected=slot is not None
        )
        
        if workpiece and slot:
            # 计算偏移：工件标记位置 → 卡槽标记位置
            # 我们需要移动机器人使工件标记移动到卡槽标记位置
            result.offset_x = slot[0] - workpiece[0]  # 卡槽X - 工件X
            result.offset_y = slot[1] - workpiece[1]  # 卡槽Y - 工件Y
            result.confidence = (workpiece[2] + slot[2]) / 2
            
            print(f"  工件标记: ({workpiece[0]:.1f}, {workpiece[1]:.1f})")
            print(f"  卡槽标记: ({slot[0]:.1f}, {slot[1]:.1f})")
            print(f"  偏移: ({result.offset_x:.1f}, {result.offset_y:.1f}) 像素")
        
        return result
    
    def visualize(self, image: np.ndarray) -> np.ndarray:
        """可视化检测结果"""
        vis = image.copy()
        
        # 检测两个标记
        workpiece = self.detect_marker_by_color(image, self.workpiece_marker_color)
        slot = self.detect_marker_by_color(image, self.slot_marker_color)
        
        # 画工件标记（绿色圆圈）
        if workpiece:
            cx, cy, conf = workpiece
            cv2.circle(vis, (int(cx), int(cy)), 15, (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)
            cv2.putText(vis, "Workpiece", (int(cx)-40, int(cy)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 画卡槽标记（红色圆圈）
        if slot:
            cx, cy, conf = slot
            cv2.circle(vis, (int(cx), int(cy)), 15, (0, 0, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(vis, "Slot", (int(cx)-20, int(cy)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 如果两个都检测到，画连线
        if workpiece and slot:
            cv2.line(vis, 
                    (int(workpiece[0]), int(workpiece[1])),
                    (int(slot[0]), int(slot[1])),
                    (255, 255, 0), 2)
            
            # 显示偏移
            offset_x = slot[0] - workpiece[0]
            offset_y = slot[1] - workpiece[1]
            cv2.putText(vis, f"Offset: ({offset_x:.0f}, {offset_y:.0f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            if not workpiece:
                cv2.putText(vis, "No workpiece marker", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not slot:
                cv2.putText(vis, "No slot marker", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis


class DualMarkerVisualServo:
    """
    双标记视觉伺服控制器
    
    通过同时检测工件标记和卡槽标记，实现精准对齐
    """
    
    def __init__(self, robot, camera, arm: str = "right", config_path: str = None):
        self.robot = robot
        self.camera = camera
        self.arm = arm
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建检测器
        self.detector = DualMarkerDetector(config_path)
        
        # 参数
        vs_config = self.config.get('visual_servo', {})
        self.pixel_to_mm_ratio = vs_config.get('pixel_to_mm_ratio', 0.5)
        self.gain = vs_config.get('gain', 0.6)
        self.tolerance_mm = vs_config.get('tolerance_mm', 2.0)
        self.max_iterations = vs_config.get('max_iterations', 15)
        self.settle_time = vs_config.get('settle_time', 0.3)
        
        # 关节灵敏度
        jc_config = self.config.get('joint_controller', {})
        self.joint_sensitivity = jc_config.get(f'{arm}_arm', {
            'joint_1': 0.15,
            'joint_2': 0.20
        })
    
    def _load_config(self, config_path: str) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        """设置标记颜色"""
        self.detector.set_marker_colors(workpiece_color, slot_color)
    
    def set_pixel_to_mm_ratio(self, ratio: float):
        self.pixel_to_mm_ratio = ratio
    
    def servo_to_target(self, tolerance_mm: float = None, 
                        max_iterations: int = None) -> bool:
        """
        执行视觉伺服，将工件标记对齐到卡槽标记
        """
        if tolerance_mm is None:
            tolerance_mm = self.tolerance_mm
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        print(f"\n开始双标记视觉伺服，目标精度: {tolerance_mm}mm")
        print(f"工件标记颜色: {self.detector.workpiece_marker_color}")
        print(f"卡槽标记颜色: {self.detector.slot_marker_color}")
        
        for i in range(max_iterations):
            print(f"\n[迭代 {i+1}/{max_iterations}]")
            
            # 采集图像
            image = self.camera.read()
            
            # 检测两个标记
            result = self.detector.detect_both_markers(image)
            
            # 检查检测结果
            if not result.workpiece_detected:
                print("  警告: 未检测到工件标记")
                continue
            
            if not result.slot_detected:
                print("  警告: 未检测到卡槽标记")
                continue
            
            # 计算毫米误差
            mm_x = result.offset_x * self.pixel_to_mm_ratio
            mm_y = result.offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)
            
            print(f"  毫米误差: ({mm_x:.2f}, {mm_y:.2f}), 总误差: {error_mm:.2f}mm")
            print(f"  置信度: {result.confidence:.2f}")
            
            # 检查是否达到目标
            if error_mm < tolerance_mm:
                print(f"\n✓ 达到目标精度: {error_mm:.2f}mm < {tolerance_mm}mm")
                return True
            
            # 应用调整
            if result.confidence > 0.3:
                self._apply_adjustment(mm_x * self.gain, mm_y * self.gain)
            else:
                print("  置信度过低，跳过调整")
            
            import time
            time.sleep(self.settle_time)
        
        print(f"\n✗ 未达到目标精度")
        return False
    
    def _apply_adjustment(self, mm_x: float, mm_y: float):
        """应用关节调整"""
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            print("  关节位置获取失败")
            return
        
        mm_x = np.clip(mm_x, -3.0, 3.0)
        mm_y = np.clip(mm_y, -3.0, 3.0)
        
        if self.arm == "right":
            joints[7] += mm_x * self.joint_sensitivity.get('joint_1', 0.15)
            joints[8] += mm_y * self.joint_sensitivity.get('joint_2', 0.20)
        else:
            joints[0] -= mm_x * self.joint_sensitivity.get('joint_1', 0.15)
            joints[1] += mm_y * self.joint_sensitivity.get('joint_2', 0.20)
        
        print(f"  应用调整: ({mm_x:.2f}, {mm_y:.2f})mm")
        self.robot.send_action({'action': joints.tolist()})
    
    def test_detection(self, num_frames: int = 100):
        """测试双标记检测"""
        print("\n双标记检测测试")
        print(f"工件标记: {self.detector.workpiece_marker_color}")
        print(f"卡槽标记: {self.detector.slot_marker_color}")
        print("按 'q' 退出")
        
        cv2.namedWindow("Dual Marker Detection", cv2.WINDOW_NORMAL)
        
        for _ in range(num_frames):
            image = self.camera.read()
            vis = self.detector.visualize(image)
            
            cv2.imshow("Dual Marker Detection", vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
