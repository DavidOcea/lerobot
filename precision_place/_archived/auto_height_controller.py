"""
Auto Height Controller - 自动高度控制器

自动调整机器人高度，确保能同时看到工件标记和定位销
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class HeightState:
    """高度状态"""
    current_height: str      # "too_low", "optimal", "too_high"
    workpiece_visible: bool
    pin_visible: bool
    recommended_action: str  # "raise", "lower", "hold"


class AutoHeightController:
    """
    自动高度控制器
    
    功能：
    1. 检测当前高度是否合适
    2. 自动调整高度直到能同时看到两个标记
    3. 对齐后自动下降到放置高度
    """
    
    def __init__(self, robot, camera, detector, arm: str = "right"):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.arm = arm
        
        # 高度调整参数
        self.raise_step = 2.0       # 上升一步的关节角度（度）
        self.lower_step = 1.5       # 下降一步的关节角度（度）
        self.settle_time = 0.3      # 稳定等待时间
        
        # 高度判断参数
        self.min_visibility_frames = 3   # 连续多少帧检测到才算可见
        self.visibility_history = {
            'workpiece': [],
            'pin': []
        }
        
        # 关节映射
        if arm == "right":
            self.height_joint_idx = 8   # 右臂关节2（肩部俯仰）
            self.height_direction = -1  # 负方向为上升
        else:
            self.height_joint_idx = 1   # 左臂关节2
            self.height_direction = -1
        
        # 状态
        self.current_joint_value = None
    
    def get_height_state(self, image: np.ndarray = None) -> HeightState:
        """
        获取当前高度状态
        
        Returns:
            HeightState 包含高度状态和建议动作
        """
        if image is None:
            image = self.camera.read()
        
        # 检测标记
        workpiece = self.detector.detect_color_marker(image, self.detector.workpiece_marker_color)
        
        slot = self.detector.detect_color_marker(image, self.detector.slot_marker_color)
        if slot is None and self.detector.pin_detector_enabled:
            slot = self.detector.detect_white_pin(image)
        
        workpiece_visible = workpiece is not None
        pin_visible = slot is not None
        
        # 更新历史
        self.visibility_history['workpiece'].append(workpiece_visible)
        self.visibility_history['pin'].append(pin_visible)
        
        # 只保留最近的帧
        max_history = self.min_visibility_frames * 2
        for key in self.visibility_history:
            if len(self.visibility_history[key]) > max_history:
                self.visibility_history[key] = self.visibility_history[key][-max_history:]
        
        # 判断稳定可见性
        stable_workpiece = self._is_stably_visible('workpiece')
        stable_pin = self._is_stably_visible('pin')
        
        # 判断高度状态
        if stable_workpiece and stable_pin:
            current_height = "optimal"
            recommended_action = "hold"
        elif stable_workpiece and not stable_pin:
            # 能看到工件但看不到定位销 -> 太低
            current_height = "too_low"
            recommended_action = "raise"
        elif not stable_workpiece and stable_pin:
            # 能看到定位销但看不到工件 -> 太高或视野问题
            current_height = "too_high"
            recommended_action = "lower"
        else:
            # 都看不到 -> 可能是视野问题
            current_height = "unknown"
            recommended_action = "raise"  # 尝试上升
        
        return HeightState(
            current_height=current_height,
            workpiece_visible=workpiece_visible,
            pin_visible=pin_visible,
            recommended_action=recommended_action
        )
    
    def _is_stably_visible(self, marker_type: str) -> bool:
        """判断标记是否稳定可见"""
        history = self.visibility_history.get(marker_type, [])
        if len(history) < self.min_visibility_frames:
            return False
        
        recent = history[-self.min_visibility_frames:]
        return all(recent)
    
    def raise_height(self, step: float = None):
        """上升"""
        if step is None:
            step = self.raise_step
        
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            print("  无法获取关节位置")
            return False
        
        joints[self.height_joint_idx] += step * self.height_direction
        
        print(f"  上升: 关节{self.height_joint_idx} 调整 {step * self.height_direction:.2f}°")
        
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(self.settle_time)
        
        return True
    
    def lower_height(self, step: float = None):
        """下降"""
        if step is None:
            step = self.lower_step
        
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            print("  无法获取关节位置")
            return False
        
        joints[self.height_joint_idx] -= step * self.height_direction
        
        print(f"  下降: 关节{self.height_joint_idx} 调整 {-step * self.height_direction:.2f}°")
        
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(self.settle_time)
        
        return True
    
    def auto_adjust_to_optimal_height(self, max_attempts: int = 10) -> bool:
        """
        自动调整到最佳高度
        
        Returns:
            是否成功调整到最佳高度
        """
        print("\n" + "="*60)
        print("自动高度调整")
        print("="*60)
        
        # 清空历史
        self.visibility_history = {'workpiece': [], 'pin': []}
        
        for attempt in range(max_attempts):
            print(f"\n[尝试 {attempt + 1}/{max_attempts}]")
            
            # 采集几帧图像稳定检测
            time.sleep(0.2)
            image = self.camera.read()
            
            state = self.get_height_state(image)
            
            print(f"  工件标记: {'可见' if state.workpiece_visible else '不可见'}")
            print(f"  定位销: {'可见' if state.pin_visible else '不可见'}")
            print(f"  高度状态: {state.current_height}")
            print(f"  建议动作: {state.recommended_action}")
            
            if state.current_height == "optimal":
                print("\n✓ 已达到最佳高度")
                return True
            
            if state.recommended_action == "raise":
                self.raise_height()
            elif state.recommended_action == "lower":
                self.lower_height()
            else:
                # 未知状态，尝试上升
                self.raise_height()
        
        print(f"\n✗ 未能自动调整到最佳高度")
        return False
    
    def descend_to_place(self, steps: int = 5, step_size: float = 2.0):
        """
        下降到放置高度
        
        Args:
            steps: 下降步数
            step_size: 每步角度
        """
        print("\n" + "="*60)
        print("自动下降到放置高度")
        print("="*60)
        
        for i in range(steps):
            print(f"\n[下降 {i+1}/{steps}]")
            self.lower_height(step_size)
            time.sleep(0.2)
        
        print("\n已到达放置高度")


class AutoHeightAlignmentController:
    """
    带自动高度调整的对齐控制器
    
    完整流程：
    1. 自动调整到最佳高度
    2. 高位置对齐
    3. 自动下降到放置高度
    """
    
    def __init__(self, robot, camera, arm: str = "right", config_path: str = None):
        self.robot = robot
        self.camera = camera
        self.arm = arm
        
        # 导入检测器
        from precision_place.layered_alignment import LayeredAlignmentDetector
        
        self.detector = LayeredAlignmentDetector(config_path)
        self.height_controller = AutoHeightController(
            robot, camera, self.detector, arm
        )
        
        # 对齐参数
        self.pixel_to_mm_ratio = 0.5
        self.gain = 0.6
        self.tolerance_mm = 2.0
        self.max_iterations = 15
        self.settle_time = 0.3
        
        # 关节灵敏度
        self.joint_sensitivity = {'joint_1': 0.15, 'joint_2': 0.20}
    
    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        self.detector.set_marker_colors(workpiece_color, slot_color)
    
    def set_pixel_to_mm_ratio(self, ratio: float):
        self.pixel_to_mm_ratio = ratio
        self.height_controller.pixel_to_mm_ratio = ratio
    
    def run_full_auto_sequence(self, tolerance_mm: float = 2.0) -> bool:
        """
        运行完全自动化的对齐放置流程
        """
        print("\n" + "#"*60)
        print("# 全自动分层对齐精准放置")
        print("#"*60)
        
        start_time = time.time()
        
        # 阶段0: 抓取确认
        print("\n[阶段0] 抓取确认")
        print("请手动完成抓取，确保工件标记可见")
        input("抓取完成后按 Enter...")
        
        # 阶段1: 移动到卡槽上方
        print("\n[阶段1] 移动到卡槽区域")
        print("请手动将机器人移动到卡槽上方大致位置")
        input("到位后按 Enter...")
        
        # 阶段2: 自动高度调整
        print("\n[阶段2] 自动高度调整")
        height_ok = self.height_controller.auto_adjust_to_optimal_height()
        
        if not height_ok:
            print("\n警告: 自动高度调整未完成")
            print("请手动调整高度直到能同时看到两个标记")
            input("调整完成后按 Enter...")
        
        # 阶段3: 高位置对齐
        print("\n[阶段3] 高位置对齐")
        align_ok = self._high_position_align(tolerance_mm)
        
        if not align_ok:
            print("\n警告: 对齐未达到精度要求")
            response = input("是否继续自动下降放置？(y/n): ").strip().lower()
            if response != 'y':
                return False
        
        # 阶段4: 自动下降
        print("\n[阶段4] 自动下降")
        response = input("是否自动下降？(y=自动/n=手动): ").strip().lower()
        
        if response == 'y':
            self.height_controller.descend_to_place(steps=5, step_size=2.0)
        else:
            print("请手动下降到放置位置")
            input("下降完成后按 Enter...")
        
        # 阶段5: 放置
        print("\n[阶段5] 放置")
        input("请松开夹爪，完成后按 Enter...")
        
        # 阶段6: 撤退
        print("\n[阶段6] 撤退")
        self.height_controller.raise_height(step=5.0)
        print("已自动抬起")
        
        elapsed = time.time() - start_time
        
        print("\n" + "#"*60)
        print(f"# 全自动流程完成!")
        print(f"# 耗时: {elapsed:.1f}秒")
        print(f"# 对齐结果: {'成功' if align_ok else '未达精度'}")
        print("#"*60)
        
        return align_ok
    
    def _high_position_align(self, tolerance_mm: float) -> bool:
        """高位置对齐"""
        for i in range(self.max_iterations):
            print(f"\n[对齐迭代 {i+1}/{self.max_iterations}]")
            
            image = self.camera.read()
            state = self.detector.detect_alignment_state(image)
            
            if not state.workpiece_visible or not state.pin_visible:
                print("  标记不可见，尝试调整高度...")
                self.height_controller.auto_adjust_to_optimal_height(max_attempts=3)
                continue
            
            mm_x = state.offset_x * self.pixel_to_mm_ratio
            mm_y = state.offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)
            
            print(f"  误差: ({mm_x:.2f}, {mm_y:.2f})mm, 总误差: {error_mm:.2f}mm")
            
            if error_mm < tolerance_mm:
                print(f"\n✓ 对齐完成: {error_mm:.2f}mm < {tolerance_mm}mm")
                return True
            
            self._apply_xy_adjustment(mm_x * self.gain, mm_y * self.gain)
            time.sleep(self.settle_time)
        
        return False
    
    def _apply_xy_adjustment(self, mm_x: float, mm_y: float):
        """应用XY调整"""
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            return
        
        mm_x = np.clip(mm_x, -3.0, 3.0)
        mm_y = np.clip(mm_y, -3.0, 3.0)
        
        if self.arm == "right":
            joints[7] += mm_x * self.joint_sensitivity['joint_1']
            joints[8] += mm_y * self.joint_sensitivity['joint_2']
        else:
            joints[0] -= mm_x * self.joint_sensitivity['joint_1']
            joints[1] += mm_y * self.joint_sensitivity['joint_2']
        
        print(f"  XY调整: ({mm_x:.2f}, {mm_y:.2f})mm")
        self.robot.send_action({'action': joints.tolist()})
    
    def test_detection(self):
        """测试检测和高度状态"""
        print("\n检测测试")
        print("按 'q' 退出, 'r' 上升, 'l' 下降")
        
        cv2.namedWindow("Auto Height Test", cv2.WINDOW_NORMAL)
        
        while True:
            image = self.camera.read()
            vis = self.detector.visualize(image)
            
            # 显示高度状态
            state = self.height_controller.get_height_state(image)
            
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (w-150, 10), (w-10, 90), (0, 0, 0), -1)
            
            color = (0, 255, 0) if state.current_height == "optimal" else (0, 0, 255)
            cv2.putText(vis, f"H: {state.current_height}", (w-140, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, f"A: {state.recommended_action}", (w-140, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis, f"P: {'Y' if state.pin_visible else 'N'}", (w-140, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Auto Height Test", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.height_controller.raise_height()
            elif key == ord('l'):
                self.height_controller.lower_height()
        
        cv2.destroyAllWindows()
