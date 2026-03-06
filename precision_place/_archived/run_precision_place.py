#!/usr/bin/env python3
"""
Run Precision Place - 精准放置启动脚本

完整的毫米级精准放置流程
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# 添加lerobot路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.supre_robot_follower import SupreRobotFollower
from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from precision_place.marker_detector import ColorMarkerDetector, MarkerVisualServo
from precision_place.pin_detector import WhitePinDetector


# ==================== 配置 ====================

CAMERA_INDICES = {
    'head': 0,
    'left_wrist': 2,
    'left_wrist2': 4,
    'right_wrist': 6,
    'right_wrist2': 8
}

# 精准放置使用的手腕相机
PRIMARY_WRIST_CAM = 'right_wrist'  # 索引6

# 标记颜色 (根据实际贴纸颜色修改)
MARKER_COLOR = "green"  # "red", "green", "blue"


# ==================== 初始化函数 ====================

def create_cameras():
    """创建相机实例"""
    cameras = {}
    
    for name, idx in CAMERA_INDICES.items():
        config = OpenCVCameraConfig(
            index_or_path=idx,
            fps=30,
            width=640,
            height=480
        )
        cameras[name] = OpenCVCamera(config)
    
    return cameras


def create_robot():
    """创建机器人实例"""
    config = SupreRobotFollowerConfig()
    robot = SupreRobotFollower(config)
    return robot


# ==================== 主要功能 ====================

class PrecisionPlaceSystem:
    """精准放置系统"""
    
    def __init__(self):
        self.robot = None
        self.cameras = None
        self.marker_detector = None
        self.pin_detector = None
        self.visual_servo = None
        
        # 标定参数
        self.pixel_to_mm_ratio = 0.5  # 默认值，需要标定
        
    def connect(self):
        """连接机器人和相机"""
        print("\n" + "="*60)
        print("连接设备...")
        print("="*60)
        
        # 创建并连接机器人
        print("\n连接机器人...")
        self.robot = create_robot()
        self.robot.connect()
        print("机器人已连接")
        
        # 创建并连接相机
        print("\n连接相机...")
        self.cameras = create_cameras()
        for name, camera in self.cameras.items():
            camera.connect()
            print(f"  {name} (索引{CAMERA_INDICES[name]}) 已连接")
        
        # 创建检测器
        self.marker_detector = ColorMarkerDetector()
        self.marker_detector.set_target_color(MARKER_COLOR)
        
        self.pin_detector = WhitePinDetector()
        
        print("\n所有设备已连接")
    
    def disconnect(self):
        """断开连接"""
        if self.robot:
            self.robot.disconnect()
        if self.cameras:
            for camera in self.cameras.values():
                camera.disconnect()
        print("已断开所有连接")
    
    def get_primary_camera(self):
        """获取主用手腕相机"""
        return self.cameras[PRIMARY_WRIST_CAM]
    
    # ==================== 标定功能 ====================
    
    def calibrate_pixel_to_mm(self, move_distance_mm: float = 5.0):
        """
        标定像素到毫米的转换比例
        """
        print("\n" + "="*60)
        print("像素-毫米比例标定")
        print("="*60)
        
        camera = self.get_primary_camera()
        
        # 采集第一张图像
        print(f"\n1. 采集初始位置图像...")
        img1 = camera.read()
        cv2.imshow("Position 1", img1)
        cv2.waitKey(500)
        cv2.destroyWindow("Position 1")
        
        # 提示移动
        print(f"\n2. 请手动将机器人沿X方向移动 {move_distance_mm}mm")
        input("   移动完成后按 Enter...")
        
        # 采集第二张图像
        print("\n3. 采集移动后位置图像...")
        img2 = camera.read()
        
        # 计算标记偏移
        marker1 = self.marker_detector.detect_marker(img1)
        marker2 = self.marker_detector.detect_marker(img2)
        
        if marker1 and marker2:
            pixel_offset = abs(marker2.center_x - marker1.center_x)
            self.pixel_to_mm_ratio = move_distance_mm / pixel_offset
            
            print(f"\n标定结果:")
            print(f"  像素偏移: {pixel_offset:.1f} pixels")
            print(f"  转换比例: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            print(f"           = {1/self.pixel_to_mm_ratio:.1f} pixel/mm")
            
            # 保存结果
            self._save_calibration()
        else:
            print("未检测到标记，请确保标记在视野中")
            # 使用光流法作为备选
            self._calibrate_with_optical_flow(img1, img2, move_distance_mm)
    
    def _calibrate_with_optical_flow(self, img1, img2, move_distance_mm):
        """使用光流法标定"""
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        corners = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
        if corners is not None and len(corners) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, corners, None)
            good = p1[st==1] - corners[st==1]
            pixel_offset = abs(np.mean(good, axis=0)[0])
            
            self.pixel_to_mm_ratio = move_distance_mm / pixel_offset
            print(f"\n使用光流法标定结果:")
            print(f"  转换比例: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            
            self._save_calibration()
    
    def _save_calibration(self):
        """保存标定结果"""
        import json
        calib_path = Path(__file__).parent / "calibration_result.json"
        with open(calib_path, 'w') as f:
            json.dump({
                'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"标定结果已保存: {calib_path}")
    
    def load_calibration(self):
        """加载标定结果"""
        import json
        calib_path = Path(__file__).parent / "calibration_result.json"
        
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                data = json.load(f)
                self.pixel_to_mm_ratio = data.get('pixel_to_mm_ratio', 0.5)
            print(f"已加载标定结果: {self.pixel_to_mm_ratio:.4f} mm/pixel")
            return True
        return False
    
    def calibrate_marker_color(self):
        """标定标记颜色"""
        camera = self.get_primary_camera()
        image = camera.read()
        self.marker_detector.calibrate_color(image)
    
    # ==================== 视觉伺服 ====================
    
    def visual_servo_to_target(self, tolerance_mm: float = 2.0, 
                               max_iterations: int = 15) -> bool:
        """
        视觉伺服到目标位置
        
        Args:
            tolerance_mm: 目标精度
            max_iterations: 最大迭代次数
        
        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print(f"开始视觉伺服 - 目标精度: {tolerance_mm}mm")
        print("="*60)
        
        camera = self.get_primary_camera()
        gain = 0.6
        
        for i in range(max_iterations):
            print(f"\n[迭代 {i+1}/{max_iterations}]")
            
            # 采集图像
            image = camera.read()
            
            # 检测标记
            offset_x, offset_y, confidence = self.marker_detector.calculate_offset(image)
            
            # 计算毫米误差
            mm_x = offset_x * self.pixel_to_mm_ratio
            mm_y = offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)
            
            print(f"  像素偏移: ({offset_x:.1f}, {offset_y:.1f})")
            print(f"  毫米误差: ({mm_x:.2f}, {mm_y:.2f}), 总误差: {error_mm:.2f}mm")
            print(f"  置信度: {confidence:.2f}")
            
            # 检查是否达到目标
            if error_mm < tolerance_mm:
                print(f"\n✓ 达到目标精度: {error_mm:.2f}mm < {tolerance_mm}mm")
                return True
            
            # 应用调整
            if confidence > 0.3:
                self._apply_xy_adjustment(mm_x * gain, mm_y * gain)
            else:
                print("  置信度过低，跳过本次调整")
            
            time.sleep(0.3)
        
        print(f"\n✗ 未达到目标精度")
        return False
    
    def _apply_xy_adjustment(self, mm_x: float, mm_y: float):
        """应用XY调整"""
        # 获取当前关节位置
        obs = self.robot.get_observation()
        joints = np.array(obs.get('observation.state', []))
        
        if len(joints) != 16:
            print("  关节位置获取失败")
            return
        
        # 限制调整量
        mm_x = np.clip(mm_x, -3.0, 3.0)
        mm_y = np.clip(mm_y, -3.0, 3.0)
        
        # 关节灵敏度
        sens = {'joint_1': 0.15, 'joint_2': 0.20}
        
        # 根据使用的手臂调整
        if 'right' in PRIMARY_WRIST_CAM:
            joints[7] -= mm_x * sens['joint_1']  # 右臂关节1
            joints[8] -= mm_y * sens['joint_2']  # 右臂关节2
        else:
            joints[0] += mm_x * sens['joint_1']  # 左臂关节1
            joints[1] -= mm_y * sens['joint_2']  # 左臂关节2
        
        print(f"  应用调整: ({mm_x:.2f}, {mm_y:.2f})mm")
        
        self.robot.send_action({'action': joints.tolist()})
    
    # ==================== 抓放动作 ====================
    
    def pick_manual(self):
        """手动抓取"""
        print("\n" + "="*60)
        print("手动抓取模式")
        print("="*60)
        print("\n请手动控制机器人完成抓取:")
        print("1. 移动到物体上方")
        print("2. 下降到抓取位置")
        print("3. 闭合夹爪")
        print("4. 抬起")
        input("\n抓取完成后按 Enter 继续...")
    
    def move_to_place_area_manual(self):
        """手动移动到放置区域"""
        print("\n" + "="*60)
        print("移动到放置区域")
        print("="*60)
        print("\n请手动将机器人移动到卡槽上方 (约5-10cm)")
        input("到位后按 Enter 继续...")
    
    def place_with_precision(self, tolerance_mm: float = 2.0):
        """
        精准放置
        
        包含：视觉伺服 + 下降 + 松开夹爪 + 抬起
        """
        print("\n" + "="*60)
        print("精准放置")
        print("="*60)
        
        # 1. 视觉伺服
        print("\n[步骤1] 视觉伺服精调")
        success = self.visual_servo_to_target(tolerance_mm=tolerance_mm)
        
        if not success:
            print("警告: 未能达到目标精度")
        
        # 2. 下降
        print("\n[步骤2] 下降到放置高度")
        input("请手动下降到放置高度，完成后按 Enter...")
        
        # 3. 松开夹爪
        print("\n[步骤3] 松开夹爪")
        input("请手动打开夹爪，完成后按 Enter...")
        
        # 4. 抬起
        print("\n[步骤4] 抬起")
        input("请手动抬起机器人，完成后按 Enter...")
        
        print("\n放置完成!")
        return success
    
    # ==================== 完整流程 ====================
    
    def run_full_sequence(self, tolerance_mm: float = 2.0):
        """运行完整的抓放流程"""
        print("\n" + "#"*60)
        print("# 精准抓放任务开始")
        print("#"*60)
        
        start_time = time.time()
        
        # 1. 抓取
        self.pick_manual()
        
        # 2. 移动到放置区域
        self.move_to_place_area_manual()
        
        # 3. 精准放置
        success = self.place_with_precision(tolerance_mm=tolerance_mm)
        
        elapsed = time.time() - start_time
        
        print("\n" + "#"*60)
        print(f"# 任务完成!")
        print(f"# 耗时: {elapsed:.1f}秒")
        print(f"# 结果: {'成功' if success else '未达精度'}")
        print("#"*60)
        
        return success
    
    # ==================== 测试功能 ====================
    
    def test_marker_detection(self):
        """测试标记检测"""
        print("\n" + "="*60)
        print("测试标记检测")
        print("="*60)
        
        camera = self.get_primary_camera()
        
        print("\n按 'q' 退出, 's' 保存截图")
        
        while True:
            image = camera.read()
            vis = self.marker_detector.visualize(image)
            
            cv2.imshow("Marker Detection", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("marker_detection.jpg", vis)
                print("截图已保存: marker_detection.jpg")
        
        cv2.destroyAllWindows()
    
    def test_visual_servo(self):
        """测试视觉伺服"""
        print("\n" + "="*60)
        print("测试视觉伺服")
        print("="*60)
        
        # 加载标定
        self.load_calibration()
        
        print("\n请将机器人移动到卡槽附近")
        input("准备好后按 Enter 开始视觉伺服...")
        
        self.visual_servo_to_target(tolerance_mm=2.0)


# ==================== 主菜单 ====================

def main():
    print("\n" + "="*60)
    print("精准放置系统")
    print("="*60)
    
    system = PrecisionPlaceSystem()
    
    try:
        while True:
            print("\n请选择操作:")
            print("1. 连接设备")
            print("2. 测试标记检测")
            print("3. 标定像素-毫米比例")
            print("4. 测试视觉伺服")
            print("5. 运行完整抓放流程")
            print("6. 连续运行模式 (10次)")
            print("0. 退出")
            
            choice = input("\n请输入选项: ").strip()
            
            if choice == "1":
                system.connect()
            elif choice == "2":
                if system.cameras is None:
                    system.connect()
                system.test_marker_detection()
            elif choice == "3":
                if system.cameras is None:
                    system.connect()
                system.calibrate_pixel_to_mm()
            elif choice == "4":
                if system.cameras is None:
                    system.connect()
                system.test_visual_servo()
            elif choice == "5":
                if system.cameras is None:
                    system.connect()
                system.load_calibration()
                system.run_full_sequence()
            elif choice == "6":
                if system.cameras is None:
                    system.connect()
                system.load_calibration()
                for i in range(10):
                    print(f"\n{'='*60}")
                    print(f"第 {i+1}/10 次运行")
                    print('='*60)
                    system.run_full_sequence()
                    if i < 9:
                        input("\n准备好下一次运行后按 Enter...")
            elif choice == "0":
                break
            else:
                print("无效选项")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        system.disconnect()
    
    print("\n再见!")


if __name__ == "__main__":
    main()
