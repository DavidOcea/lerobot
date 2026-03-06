#!/usr/bin/env python3
"""
Run Auto Height - 带自动高度调整的精准放置

主用相机: right_wrist_cam2 (索引8)
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

# 导入统一配置
from precision_place.config import (
    CAMERA_INDICES, PRIMARY_WRIST_CAM, PRIMARY_ARM,
    WORKPIECE_MARKER_COLOR, SLOT_MARKER_COLOR, TOLERANCE_MM
)


class AutoHeightController:
    """自动高度控制器"""
    
    def __init__(self, robot, camera, detector):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        
        self.raise_step = 2.0
        self.lower_step = 1.5
        self.settle_time = 0.3
        
        # 右臂关节2控制高度
        self.height_joint_idx = 8
        self.height_direction = -1
    
    def raise_height(self, step: float = None):
        if step is None:
            step = self.raise_step
        
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            return False
        
        joints[self.height_joint_idx] += step * self.height_direction
        print(f"  上升: {step:.1f}°")
        
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(self.settle_time)
        return True
    
    def lower_height(self, step: float = None):
        if step is None:
            step = self.lower_step
        
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            return False
        
        joints[self.height_joint_idx] -= step * self.height_direction
        print(f"  下降: {step:.1f}°")
        
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(self.settle_time)
        return True
    
    def auto_adjust_height(self, max_attempts: int = 10) -> bool:
        """自动调整到最佳高度"""
        print("\n自动调整高度...")
        
        for i in range(max_attempts):
            print(f"\n[尝试 {i+1}/{max_attempts}]")
            
            time.sleep(0.2)
            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)
            
            wp_count = sum(1 for m in [state.workpiece_top, state.workpiece_bottom] if m)
            sl_count = sum(1 for m in [state.slot_top, state.slot_bottom] if m)
            
            print(f"  工件标记: {wp_count}/2, 卡槽标记: {sl_count}/2")
            
            if wp_count >= 2 and sl_count >= 2:
                print("\n✓ 达到最佳高度")
                return True
            
            if sl_count < 2:
                print("  定位销标记不完整，尝试上升...")
                self.raise_height()
            elif wp_count < 2:
                print("  工件标记不完整")
            
            time.sleep(0.3)
        
        print("\n✗ 未能调整到最佳高度")
        return False


class AutoHeightPrecisionSystem:
    """带自动高度调整的精准放置系统"""
    
    def __init__(self):
        self.robot = None
        self.cameras = None
        self.detector = None
        self.height_controller = None
        self.pixel_to_mm_ratio = 0.5
    
    def connect(self):
        print("\n" + "="*60)
        print("连接设备...")
        print("="*60)
        
        print("\n连接机器人...")
        self.robot = SupreRobotFollower(SupreRobotFollowerConfig())
        self.robot.connect()
        print("机器人已连接")
        
        print("\n连接相机...")
        self.cameras = {}
        for name, idx in CAMERA_INDICES.items():
            try:
                config = OpenCVCameraConfig(index_or_path=idx, fps=30, width=640, height=480)
                self.cameras[name] = OpenCVCamera(config)
                self.cameras[name].connect()
                print(f"  {name} (索引{idx}) 已连接")
            except Exception as e:
                print(f"  {name} (索引{idx}) 连接失败: {e}")
        
        # 检测器
        from precision_place.dual_point_alignment import DualPointDetector
        self.detector = DualPointDetector()
        self.detector.set_marker_colors(WORKPIECE_MARKER_COLOR, SLOT_MARKER_COLOR)
        
        # 高度控制器
        self.height_controller = AutoHeightController(
            self.robot, self.cameras[PRIMARY_WRIST_CAM], self.detector
        )
        
        print(f"\n主用相机: {PRIMARY_WRIST_CAM} (索引{CAMERA_INDICES[PRIMARY_WRIST_CAM]})")
    
    def disconnect(self):
        if self.robot:
            self.robot.disconnect()
        if self.cameras:
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except:
                    pass
        print("已断开连接")
    
    def load_calibration(self):
        import json
        path = Path(__file__).parent / "calibration_result.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.pixel_to_mm_ratio = data.get('pixel_to_mm_ratio', 0.5)
            print(f"已加载标定: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            return True
        return False
    
    def test_detection(self):
        print("\n检测测试 (按 'q' 退出, 'r' 上升, 'l' 下降)")
        
        while True:
            image = self.cameras[PRIMARY_WRIST_CAM].read()
            state = self.detector.detect_dual_marker_state(image)
            vis = self.detector.visualize(image, state)
            
            cv2.imshow("Detection Test", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.height_controller.raise_height()
            elif key == ord('l'):
                self.height_controller.lower_height()
        
        cv2.destroyAllWindows()
    
    def run_full_auto(self):
        print("\n" + "#"*60)
        print("# 全自动精准放置")
        print("#"*60)
        
        start = time.time()
        
        # 1. 抓取
        print("\n[步骤1] 请手动抓取工件")
        input("完成后按 Enter...")
        
        # 2. 移动到卡槽上方
        print("\n[步骤2] 请移动到卡槽上方大致位置")
        input("完成后按 Enter...")
        
        # 3. 自动调整高度
        print("\n[步骤3] 自动调整高度")
        height_ok = self.height_controller.auto_adjust_height()
        
        if not height_ok:
            print("\n请手动调整高度")
            input("调整完成后按 Enter...")
        
        # 4. XY对齐
        print("\n[步骤4] XY对齐")
        success = self._align_xy()
        
        # 5. 自动下降
        print("\n[步骤5] 自动下降")
        response = input("自动下降？(y/n): ").strip().lower()
        if response == 'y':
            for i in range(5):
                self.height_controller.lower_height()
        
        # 6. 放置
        print("\n[步骤6] 请手动放置并抬起")
        input("完成后按 Enter...")
        
        elapsed = time.time() - start
        print("\n" + "#"*60)
        print(f"# 完成! 耗时: {elapsed:.1f}秒")
        print("#"*60)
    
    def _align_xy(self, max_iter: int = 10) -> bool:
        for i in range(max_iter):
            print(f"\n[对齐 {i+1}/{max_iter}]")
            
            image = self.cameras[PRIMARY_WRIST_CAM].read()
            state = self.detector.detect_dual_marker_state(image)
            
            if not state.workpiece_detected or not state.slot_detected:
                print("  标记不完整")
                continue
            
            mm_x = state.offset_x * self.pixel_to_mm_ratio
            mm_y = state.offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)
            
            print(f"  误差: {error_mm:.2f}mm")
            
            if error_mm < TOLERANCE_MM:
                print(f"\n✓ 对齐完成")
                return True
            
            self._apply_adjustment(mm_x * 0.6, mm_y * 0.6)
            time.sleep(0.3)
        
        return False
    
    def _apply_adjustment(self, mm_x: float, mm_y: float):
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            return
        
        mm_x = np.clip(mm_x, -3.0, 3.0)
        mm_y = np.clip(mm_y, -3.0, 3.0)
        
        joints[7] += mm_x * 0.15
        joints[8] += mm_y * 0.20
        
        print(f"  调整: ({mm_x:.2f}, {mm_y:.2f})mm")
        self.robot.send_action({'action': joints.tolist()})


def main():
    print("\n" + "="*60)
    print("自动高度精准放置系统")
    print("="*60)
    print(f"""
配置:
  - 主用相机: {PRIMARY_WRIST_CAM} (索引 {CAMERA_INDICES[PRIMARY_WRIST_CAM]})
  - 工件标记: {WORKPIECE_MARKER_COLOR}
  - 卡槽标记: {SLOT_MARKER_COLOR}
""")
    
    system = AutoHeightPrecisionSystem()
    
    try:
        while True:
            print("\n" + "-"*40)
            print("1. 连接设备")
            print("2. 测试检测")
            print("3. 标定")
            print("4. 运行全自动流程")
            print("5. 连续运行 (10次)")
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
                # 标定代码...
                print("请使用 run_dual_point_alignment.py 进行标定")
            elif choice == "4":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                system.run_full_auto()
            elif choice == "5":
                if not system.cameras:
                    system.connect()
                system.load_calibration()
                for i in range(10):
                    print(f"\n第 {i+1}/10 次")
                    system.run_full_auto()
                    if i < 9:
                        input("\n按 Enter 继续...")
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
