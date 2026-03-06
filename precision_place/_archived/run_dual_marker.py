#!/usr/bin/env python3
"""
Dual Marker Precision Place - 双标记精准放置

解决工件抓取位置不固定的问题
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.supre_robot_follower import SupreRobotFollower
from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from precision_place.dual_marker_servo import DualMarkerDetector, DualMarkerVisualServo


# ==================== 配置 ====================

CAMERA_INDICES = {
    'head': 0,
    'left_wrist': 2,
    'left_wrist2': 4,
    'right_wrist': 6,
    'right_wrist2': 8
}

PRIMARY_WRIST_CAM = 'right_wrist2'  # 索引8

# 标记颜色配置
WORKPIECE_MARKER_COLOR = "green"  # 工件上的标记
SLOT_MARKER_COLOR = "red"         # 卡槽上的标记


class DualMarkerPrecisionSystem:
    """双标记精准放置系统"""
    
    def __init__(self):
        self.robot = None
        self.cameras = None
        self.detector = None
        self.pixel_to_mm_ratio = 0.5
    
    def connect(self):
        """连接设备"""
        print("\n" + "="*60)
        print("连接设备...")
        print("="*60)
        
        # 机器人
        print("\n连接机器人...")
        self.robot = SupreRobotFollower(SupreRobotFollowerConfig())
        self.robot.connect()
        
        # 相机
        print("\n连接相机...")
        self.cameras = {}
        for name, idx in CAMERA_INDICES.items():
            config = OpenCVCameraConfig(index_or_path=idx, fps=30, width=640, height=480)
            self.cameras[name] = OpenCVCamera(config)
            self.cameras[name].connect()
            print(f"  {name} (索引{idx}) 已连接")
        
        # 检测器
        self.detector = DualMarkerDetector()
        self.detector.set_marker_colors(WORKPIECE_MARKER_COLOR, SLOT_MARKER_COLOR)
        
        print("\n所有设备已连接")
    
    def disconnect(self):
        """断开连接"""
        if self.robot:
            self.robot.disconnect()
        if self.cameras:
            for cam in self.cameras.values():
                cam.disconnect()
    
    def get_wrist_camera(self):
        return self.cameras[PRIMARY_WRIST_CAM]
    
    # ==================== 标定 ====================
    
    def calibrate_pixel_to_mm(self, move_distance_mm: float = 5.0):
        """标定像素到毫米比例"""
        print("\n" + "="*60)
        print("像素-毫米比例标定")
        print("="*60)
        
        camera = self.get_wrist_camera()
        
        print(f"\n1. 采集初始位置图像...")
        img1 = camera.read()
        result1 = self.detector.detect_both_markers(img1)
        
        if not result1.workpiece_detected and not result1.slot_detected:
            print("未检测到任何标记，请确保标记在视野中")
            return
        
        print(f"\n2. 请手动将机器人沿X方向移动 {move_distance_mm}mm")
        input("   移动完成后按 Enter...")
        
        print(f"\n3. 采集移动后位置图像...")
        img2 = camera.read()
        result2 = self.detector.detect_both_markers(img2)
        
        # 使用检测到的标记计算偏移
        if result1.slot_detected and result2.slot_detected:
            slot1 = self.detector.detect_marker_by_color(img1, SLOT_MARKER_COLOR)
            slot2 = self.detector.detect_marker_by_color(img2, SLOT_MARKER_COLOR)
            pixel_offset = abs(slot2[0] - slot1[0])
        elif result1.workpiece_detected and result2.workpiece_detected:
            wp1 = self.detector.detect_marker_by_color(img1, WORKPIECE_MARKER_COLOR)
            wp2 = self.detector.detect_marker_by_color(img2, WORKPIECE_MARKER_COLOR)
            pixel_offset = abs(wp2[0] - wp1[0])
        else:
            print("无法计算，使用光流法")
            pixel_offset = self._optical_flow_offset(img1, img2)
        
        if pixel_offset > 0:
            self.pixel_to_mm_ratio = move_distance_mm / pixel_offset
            print(f"\n标定结果: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            self._save_calibration()
    
    def _optical_flow_offset(self, img1, img2):
        """光流法计算偏移"""
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
        if corners is not None and len(corners) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, corners, None)
            good = p1[st==1] - corners[st==1]
            return abs(np.mean(good, axis=0)[0])
        return 10
    
    def _save_calibration(self):
        import json
        with open(Path(__file__).parent / "calibration_dual_marker.json", 'w') as f:
            json.dump({
                'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
                'workpiece_color': WORKPIECE_MARKER_COLOR,
                'slot_color': SLOT_MARKER_COLOR,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print("标定结果已保存")
    
    def load_calibration(self):
        import json
        path = Path(__file__).parent / "calibration_dual_marker.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.pixel_to_mm_ratio = data.get('pixel_to_mm_ratio', 0.5)
            print(f"已加载标定: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            return True
        return False
    
    # ==================== 检测测试 ====================
    
    def test_detection(self):
        """测试双标记检测"""
        print("\n" + "="*60)
        print("双标记检测测试")
        print("="*60)
        print(f"工件标记颜色: {WORKPIECE_MARKER_COLOR}")
        print(f"卡槽标记颜色: {SLOT_MARKER_COLOR}")
        print("按 'q' 退出, 'c' 切换颜色配置")
        
        camera = self.get_wrist_camera()
        
        while True:
            image = camera.read()
            vis = self.detector.visualize(image)
            
            cv2.imshow("Dual Marker Detection", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # 切换颜色配置
                print("\n可选颜色: green, red, blue, yellow")
                wp = input("工件标记颜色: ").strip()
                sl = input("卡槽标记颜色: ").strip()
                if wp and sl:
                    self.detector.set_marker_colors(wp, sl)
        
        cv2.destroyAllWindows()
    
    # ==================== 视觉伺服 ====================
    
    def visual_servo(self, tolerance_mm: float = 2.0):
        """双标记视觉伺服"""
        print("\n" + "="*60)
        print("双标记视觉伺服")
        print("="*60)
        
        camera = self.get_wrist_camera()
        gain = 0.6
        max_iter = 15
        
        for i in range(max_iter):
            print(f"\n[迭代 {i+1}/{max_iter}]")
            
            image = camera.read()
            result = self.detector.detect_both_markers(image)
            
            if not result.workpiece_detected:
                print("  警告: 未检测到工件标记")
                continue
            
            if not result.slot_detected:
                print("  警告: 未检测到卡槽标记")
                continue
            
            mm_x = result.offset_x * self.pixel_to_mm_ratio
            mm_y = result.offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)
            
            print(f"  误差: ({mm_x:.2f}, {mm_y:.2f})mm, 总误差: {error_mm:.2f}mm")
            
            if error_mm < tolerance_mm:
                print(f"\n✓ 达到目标精度: {error_mm:.2f}mm")
                return True
            
            self._apply_adjustment(mm_x * gain, mm_y * gain)
            time.sleep(0.3)
        
        print(f"\n✗ 未达到目标精度")
        return False
    
    def _apply_adjustment(self, mm_x: float, mm_y: float):
        """应用调整"""
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            return
        
        mm_x = np.clip(mm_x, -3.0, 3.0)
        mm_y = np.clip(mm_y, -3.0, 3.0)
        
        if 'right' in PRIMARY_WRIST_CAM:
            joints[7] += mm_x * 0.15
            joints[8] += mm_y * 0.20
        else:
            joints[0] -= mm_x * 0.15
            joints[1] += mm_y * 0.20
        
        print(f"  调整: ({mm_x:.2f}, {mm_y:.2f})mm")
        self.robot.send_action({'action': joints.tolist()})
    
    # ==================== 完整流程 ====================
    
    def run_full_sequence(self, tolerance_mm: float = 2.0):
        """完整抓放流程"""
        print("\n" + "#"*60)
        print("# 双标记精准抓放")
        print("#"*60)
        
        start = time.time()
        
        # 1. 抓取
        print("\n[步骤1] 抓取工件")
        print("请手动完成抓取，确保工件上的标记可见")
        input("抓取完成后按 Enter...")
        
        # 2. 移动到卡槽上方
        print("\n[步骤2] 移动到卡槽上方")
        print("请将机器人移动到卡槽上方 (5-10cm)")
        print("确保手腕相机能同时看到工件标记和卡槽标记")
        input("到位后按 Enter...")
        
        # 3. 视觉伺服
        print("\n[步骤3] 视觉伺服对齐")
        success = self.visual_servo(tolerance_mm)
        
        if not success:
            print("警告: 未达到精度，可手动微调")
        
        # 4. 放置
        print("\n[步骤4] 放置")
        input("请手动下降并松开夹爪，完成后按 Enter...")
        
        # 5. 撤退
        print("\n[步骤5] 撤退")
        input("请手动抬起机器人，完成后按 Enter...")
        
        elapsed = time.time() - start
        
        print("\n" + "#"*60)
        print(f"# 完成! 耗时: {elapsed:.1f}秒")
        print("#"*60)
        
        return success


def main():
    print("\n" + "="*60)
    print("双标记精准放置系统")
    print("="*60)
    print("\n原理:")
    print("  - 工件上贴一个标记 (代表定位孔位置)")
    print("  - 卡槽上贴一个标记 (代表定位销位置)")
    print("  - 视觉伺服将两个标记对齐")
    print("  - 实现定位孔精准对准定位销")
    
    system = DualMarkerPrecisionSystem()
    
    try:
        while True:
            print("\n" + "-"*40)
            print("1. 连接设备")
            print("2. 测试双标记检测")
            print("3. 标定像素-毫米比例")
            print("4. 测试视觉伺服")
            print("5. 运行完整流程")
            print("6. 连续运行 (10次)")
            print("0. 退出")
            
            choice = input("\n选项: ").strip()
            
            if choice == "1":
                system.connect()
            elif choice == "2":
                if not system.cameras:
                    system.connect()
                system.test_detection()
            elif choice == "3":
                if not system.cameras:
                    system.connect()
                system.calibrate_pixel_to_mm()
            elif choice == "4":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                print("\n请将机器人移动到卡槽上方")
                input("准备好后按 Enter...")
                system.visual_servo()
            elif choice == "5":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                system.run_full_sequence()
            elif choice == "6":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                for i in range(10):
                    print(f"\n{'='*40}")
                    print(f"第 {i+1}/10 次")
                    system.run_full_sequence()
                    if i < 9:
                        input("\n准备下一次，按 Enter...")
            elif choice == "0":
                break
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        system.disconnect()
    
    print("\n再见!")


if __name__ == "__main__":
    main()
