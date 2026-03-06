"""
Visual Servo - 手腕相机视觉伺服控制器

通过手腕相机实现毫米级精准放置
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import yaml
from pathlib import Path
import time

from .slot_detector import SlotDetector, SlotPosition
from .joint_controller import JointController


class WristCameraVisualServo:
    """
    手腕相机视觉伺服控制器
    
    工作原理：
    1. 预先采集目标位置的模板图像
    2. 执行时，通过模板匹配找到目标位置在当前图像中的偏移
    3. 将像素偏移转换为机器人末端微调量
    4. 迭代校正直到误差满足精度要求
    """
    
    def __init__(self, robot, camera, arm: str = "right", config_path: str = None):
        """
        Args:
            robot: SupreRobotFollower 实例
            camera: 手腕相机实例
            arm: 使用的手臂 ("left" or "right")
            config_path: 配置文件路径
        """
        self.robot = robot
        self.camera = camera
        self.arm = arm
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.slot_detector = SlotDetector(config_path)
        self.joint_controller = JointController(robot, config_path, arm)
        
        # 伺服参数
        self.gain = self.config.get('gain', 0.6)
        self.tolerance_mm = self.config.get('tolerance_mm', 2.0)
        self.max_iterations = self.config.get('max_iterations', 15)
        self.settle_time = self.config.get('settle_time', 0.3)
        
        # 状态
        self.last_error = None
        self.iteration_count = 0
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "precision_config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('visual_servo', {})
        return {}
    
    def set_pixel_to_mm_ratio(self, ratio: float):
        """设置像素到毫米的转换比例"""
        self.joint_controller.set_pixel_to_mm_ratio(ratio)
    
    def save_target_template(self, image: np.ndarray = None, 
                             center: Tuple[int, int] = None) -> np.ndarray:
        """
        保存目标位置模板
        
        使用方法：
        1. 手动将机器人移动到目标放置位置
        2. 调用此方法保存手腕相机看到的模板
        """
        if image is None:
            image = self.camera.read()
        
        template = self.slot_detector.save_target_template(image, center)
        
        # 同时保存模板到文件
        save_path = Path(__file__).parent / f"template_{self.arm}.png"
        cv2.imwrite(str(save_path), template)
        print(f"模板已保存到: {save_path}")
        
        return template
    
    def load_target_template(self, path: str = None) -> np.ndarray:
        """加载目标模板"""
        if path is None:
            path = Path(__file__).parent / f"template_{self.arm}.png"
        
        return self.slot_detector.load_target_template(str(path))
    
    def get_current_error(self) -> Tuple[float, float, float]:
        """
        获取当前位置相对于目标位置的误差
        
        Returns:
            (error_x_mm, error_y_mm, confidence): 毫米误差和置信度
        """
        # 采集当前图像
        image = self.camera.read()
        
        # 计算像素偏移
        offset_x, offset_y, confidence = self.slot_detector.calculate_offset(image)
        
        # 转换为毫米
        mm_x, mm_y = self.joint_controller.pixel_offset_to_mm(offset_x, offset_y)
        
        return mm_x, mm_y, confidence
    
    def servo_step(self, gain: float = None) -> Tuple[float, float, bool]:
        """
        执行一步视觉伺服
        
        Returns:
            (error_mm, confidence, converged): 误差、置信度、是否收敛
        """
        if gain is None:
            gain = self.gain
        
        # 获取当前误差
        mm_x, mm_y, confidence = self.get_current_error()
        
        # 计算总误差
        error_mm = np.sqrt(mm_x**2 + mm_y**2)
        
        print(f"  误差: X={mm_x:.2f}mm, Y={mm_y:.2f}mm, 总误差={error_mm:.2f}mm, 置信度={confidence:.2f}")
        
        # 检查是否收敛
        converged = error_mm < self.tolerance_mm
        
        if not converged and confidence > 0.5:
            # 应用调整
            self.joint_controller.adjust_xy(
                mm_x / self.joint_controller.pixel_to_mm_ratio,
                mm_y / self.joint_controller.pixel_to_mm_ratio,
                gain=gain,
                wait_time=self.settle_time
            )
        
        self.last_error = (mm_x, mm_y, error_mm)
        
        return error_mm, confidence, converged
    
    def servo_to_target(self, tolerance_mm: float = None, 
                        max_iterations: int = None,
                        verbose: bool = True) -> bool:
        """
        执行视觉伺服，将末端移动到目标位置
        
        Args:
            tolerance_mm: 目标精度
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
        
        Returns:
            是否成功达到目标精度
        """
        if tolerance_mm is None:
            tolerance_mm = self.tolerance_mm
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"开始视觉伺服 - 目标精度: {tolerance_mm}mm")
            print(f"{'='*50}")
        
        self.iteration_count = 0
        prev_error = float('inf')
        
        for i in range(max_iterations):
            self.iteration_count = i + 1
            
            if verbose:
                print(f"\n[迭代 {i+1}/{max_iterations}]")
            
            error_mm, confidence, converged = self.servo_step()
            
            if converged:
                if verbose:
                    print(f"\n✓ 达到目标精度: {error_mm:.2f}mm < {tolerance_mm}mm")
                    print(f"  总迭代次数: {self.iteration_count}")
                return True
            
            # 检查是否发散
            if error_mm > prev_error * 1.5:
                if verbose:
                    print(f"  警告: 误差在增加，可能需要调整增益")
                # 降低增益
                self.gain *= 0.8
            
            # 检查是否停滞
            if abs(error_mm - prev_error) < 0.1:
                if verbose:
                    print(f"  警告: 收敛停滞")
                # 增加增益
                self.gain = min(1.0, self.gain * 1.2)
            
            prev_error = error_mm
        
        if verbose:
            print(f"\n✗ 未达到目标精度")
            print(f"  最终误差: {error_mm:.2f}mm")
            print(f"  总迭代次数: {self.iteration_count}")
        
        return False
    
    def calibrate_pixel_to_mm(self, move_distance_mm: float = 5.0) -> float:
        """
        标定像素到毫米的转换比例
        
        步骤：
        1. 采集当前位置图像
        2. 手动移动机器人指定距离
        3. 采集新位置图像
        4. 计算像素偏移，得到比例
        
        Returns:
            pixel_to_mm_ratio
        """
        print(f"\n{'='*50}")
        print(f"像素-毫米比例标定")
        print(f"{'='*50}")
        
        # 采集第一张图像
        print("\n1. 采集初始位置图像...")
        img1 = self.camera.read()
        
        # 提示移动
        print(f"\n2. 请手动将机器人沿X方向移动 {move_distance_mm}mm")
        print("   (可以使用示教器或手动控制)")
        input("   移动完成后按 Enter 继续...")
        
        # 采集第二张图像
        print("\n3. 采集移动后位置图像...")
        img2 = self.camera.read()
        
        # 计算像素偏移
        pixel_offset = self._compute_pixel_offset(img1, img2)
        
        # 计算比例
        ratio = move_distance_mm / pixel_offset
        self.set_pixel_to_mm_ratio(ratio)
        
        print(f"\n{'='*50}")
        print(f"标定结果:")
        print(f"  像素偏移: {pixel_offset:.1f} pixels")
        print(f"  实际移动: {move_distance_mm} mm")
        print(f"  转换比例: {ratio:.4f} mm/pixel")
        print(f"           = {1/ratio:.1f} pixel/mm")
        print(f"{'='*50}")
        
        # 保存标定结果
        self._save_calibration(ratio)
        
        return ratio
    
    def _compute_pixel_offset(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两张图像间的像素偏移"""
        # 转灰度
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 光流法
        corners = cv2.goodFeaturesToTrack(g1, maxCorners=100, 
                                          qualityLevel=0.01, minDistance=10)
        
        if corners is not None and len(corners) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, corners, None)
            
            if p1 is not None:
                good_old = corners[st == 1]
                good_new = p1[st == 1]
                
                if len(good_old) >= 5:
                    offsets = good_new - good_old
                    mean_offset = np.mean(offsets, axis=0)
                    return np.abs(mean_offset[0])
        
        # 方法2: 模板匹配（备用）
        h, w = g1.shape
        template = g1[h//3:2*h//3, w//3:2*w//3]
        result = cv2.matchTemplate(g2, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        th, tw = template.shape
        center1 = (w//2, h//2)
        center2 = (max_loc[0] + tw//2, max_loc[1] + th//2)
        
        return np.abs(center2[0] - center1[0])
    
    def _save_calibration(self, ratio: float):
        """保存标定结果"""
        import json
        calib_path = Path(__file__).parent / f"calibration_{self.arm}.json"
        
        with open(calib_path, 'w') as f:
            json.dump({
                'pixel_to_mm_ratio': ratio,
                'arm': self.arm,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"标定结果已保存到: {calib_path}")
    
    def load_calibration(self) -> Optional[float]:
        """加载标定结果"""
        import json
        calib_path = Path(__file__).parent / f"calibration_{self.arm}.json"
        
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                data = json.load(f)
                ratio = data.get('pixel_to_mm_ratio')
                if ratio:
                    self.set_pixel_to_mm_ratio(ratio)
                    print(f"已加载标定结果: 1像素 = {ratio:.4f}mm")
                    return ratio
        
        return None
