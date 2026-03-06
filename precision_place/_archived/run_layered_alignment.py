#!/usr/bin/env python3
"""
Run Layered Alignment - 分层对齐精准放置

解决工件遮挡定位销的问题
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

from precision_place.layered_alignment import LayeredAlignmentController


# ==================== 配置 ====================

CAMERA_INDICES = {
    'head': 0,
    'left_wrist': 2,
    'left_wrist2': 4,
    'right_wrist': 6,
    'right_wrist2': 8
}

PRIMARY_WRIST_CAM = 'right_wrist2'  # 索引8

# 标记颜色
WORKPIECE_MARKER_COLOR = "green"  # 工件定位孔标记
SLOT_MARKER_COLOR = "red"         # 卡槽定位销标记


class PrecisionPlaceSystem:
    """精准放置系统 - 分层对齐版"""
    
    def __init__(self):
        self.robot = None
        self.cameras = None
        self.controller = None
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
        print("机器人已连接")
        
        # 相机
        print("\n连接相机...")
        self.cameras = {}
        for name, idx in CAMERA_INDICES.items():
            config = OpenCVCameraConfig(index_or_path=idx, fps=30, width=640, height=480)
            self.cameras[name] = OpenCVCamera(config)
            self.cameras[name].connect()
            print(f"  {name} (索引{idx}) 已连接")
        
        # 控制器
        self.controller = LayeredAlignmentController(
            robot=self.robot,
            camera=self.cameras[PRIMARY_WRIST_CAM],
            arm="right"
        )
        self.controller.set_marker_colors(WORKPIECE_MARKER_COLOR, SLOT_MARKER_COLOR)
        
        print("\n所有设备已连接")
    
    def disconnect(self):
        """断开连接"""
        if self.robot:
            self.robot.disconnect()
        if self.cameras:
            for cam in self.cameras.values():
                cam.disconnect()
        print("已断开所有连接")
    
    def load_calibration(self):
        """加载标定"""
        import json
        path = Path(__file__).parent / "calibration_result.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.pixel_to_mm_ratio = data.get('pixel_to_mm_ratio', 0.5)
            self.controller.pixel_to_mm_ratio = self.pixel_to_mm_ratio
            print(f"已加载标定: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            return True
        return False
    
    def calibrate_pixel_to_mm(self):
        """标定像素-毫米比例"""
        print("\n" + "="*60)
        print("像素-毫米比例标定")
        print("="*60)
        
        camera = self.cameras[PRIMARY_WRIST_CAM]
        
        print("\n1. 采集初始位置图像...")
        img1 = camera.read()
        cv2.imshow("Position 1", img1)
        cv2.waitKey(500)
        cv2.destroyWindow("Position 1")
        
        print("\n2. 请手动将机器人沿X方向移动 5mm")
        input("   移动完成后按 Enter...")
        
        print("\n3. 采集移动后位置图像...")
        img2 = camera.read()
        
        # 计算像素偏移
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
        
        if corners is not None and len(corners) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, corners, None)
            good = p1[st==1] - corners[st==1]
            pixel_offset = abs(np.mean(good, axis=0)[0])
            
            self.pixel_to_mm_ratio = 5.0 / pixel_offset
            self.controller.pixel_to_mm_ratio = self.pixel_to_mm_ratio
            
            print(f"\n标定结果: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            
            # 保存
            import json
            with open(Path(__file__).parent / "calibration_result.json", 'w') as f:
                json.dump({
                    'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
    
    def test_detection(self):
        """测试检测"""
        print("\n" + "="*60)
        print("检测测试")
        print("="*60)
        print(f"工件定位孔标记: {WORKPIECE_MARKER_COLOR}")
        print(f"定位销标记: {SLOT_MARKER_COLOR}")
        print("按 'q' 退出")
        
        self.controller.test_detection()
    
    def run_high_position_align(self):
        """运行高位置对齐"""
        print("\n" + "="*60)
        print("高位置对齐测试")
        print("="*60)
        print("\n请确保:")
        print("1. 机器人高度足够（5-10cm）")
        print("2. 手腕相机能同时看到工件标记和定位销标记")
        
        input("\n准备好后按 Enter 开始...")
        
        self.controller.high_position_align(tolerance_mm=2.0)
    
    def run_full_sequence(self):
        """运行完整流程"""
        self.controller.run_full_sequence(tolerance_mm=2.0)
    
    def continuous_run(self, count: int = 10):
        """连续运行"""
        for i in range(count):
            print(f"\n{'='*60}")
            print(f"第 {i+1}/{count} 次运行")
            print('='*60)
            
            self.controller.run_full_sequence(tolerance_mm=2.0)
            
            if i < count - 1:
                input("\n准备下一次，按 Enter...")


def main():
    print("\n" + "="*60)
    print("分层对齐精准放置系统")
    print("="*60)
    print("\n方案说明:")
    print("1. 高位置对齐: 在工件遮挡定位销之前完成对齐")
    print("2. 下降放置: 对齐后直接下降放置")
    print("3. 关键: 保持足够高度，确保两个标记都可见")
    
    system = PrecisionPlaceSystem()
    
    try:
        while True:
            print("\n" + "-"*40)
            print("1. 连接设备")
            print("2. 测试检测")
            print("3. 标定像素-毫米比例")
            print("4. 测试高位置对齐")
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
                system.run_high_position_align()
            elif choice == "5":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                system.run_full_sequence()
            elif choice == "6":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                system.continuous_run(10)
            elif choice == "0":
                break
            else:
                print("无效选项")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        system.disconnect()
    
    print("\n再见!")


if __name__ == "__main__":
    main()
