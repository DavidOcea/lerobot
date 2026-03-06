#!/usr/bin/env python3
"""
精准放置系统 - 主启动脚本 (优化版)

功能:
1. 引导式操作
2. 左手/右手切换
3. 夹爪控制
4. 自动下降放置
5. 运动平滑
6. 标定验证
7. 预设位置
8. 标定历史

使用: python precision_place/run.py
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

from precision_place.dual_point_alignment import PrecisionPlaceController, ARM_CONFIGS


# ==================== 配置 ====================

CAMERA_INDICES = {
    'head': 0,
    'left_wrist': 2,
    'left_wrist2': 4,
    'right_wrist': 6,
    'right_wrist2': 8
}

WORKPIECE_COLOR = "green"
SLOT_COLOR = "red"


# ==================== 系统 ====================

class PrecisionPlaceSystem:
    """精准放置系统"""
    
    def __init__(self):
        self.robot = None
        self.cameras = {}
        self.controller = None
        self.current_arm = "right"
        self.is_first_run = not (Path(__file__).parent / "calibration_history.json").exists()
    
    def connect(self, arm: str = "right"):
        """连接设备"""
        print("\n" + "="*60)
        print("连接设备")
        print("="*60)
        
        # 机器人
        print("\n连接机器人...")
        self.robot = SupreRobotFollower(SupreRobotFollowerConfig())
        self.robot.connect()
        print("✓ 机器人已连接")
        
        # 相机
        print("\n连接相机...")
        for name, idx in CAMERA_INDICES.items():
            try:
                config = OpenCVCameraConfig(index_or_path=idx, fps=30, width=640, height=480)
                self.cameras[name] = OpenCVCamera(config)
                self.cameras[name].connect()
                print(f"  ✓ {name} (索引{idx})")
            except Exception as e:
                print(f"  ✗ {name} (索引{idx}): {e}")
        
        # 控制器
        self.current_arm = arm
        arm_config = ARM_CONFIGS.get(arm)
        
        if arm_config.camera_name in self.cameras:
            self.controller = PrecisionPlaceController(
                robot=self.robot,
                camera=self.cameras[arm_config.camera_name],
                arm=arm
            )
            self.controller.set_marker_colors(WORKPIECE_COLOR, SLOT_COLOR)
            print(f"\n✓ 主用相机: {arm_config.camera_name} (索引{arm_config.camera_index})")
            print(f"✓ 使用手臂: {arm}")
        else:
            raise RuntimeError(f"无法连接相机 {arm_config.camera_name}")
    
    def disconnect(self):
        if self.robot:
            self.robot.disconnect()
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except:
                pass
        print("已断开连接")
    
    def switch_arm(self, arm: str):
        """切换手臂"""
        if arm not in ARM_CONFIGS:
            print(f"✗ 不支持的手臂: {arm}")
            return False
        
        if not self.controller:
            print("请先连接设备")
            return False
        
        arm_config = ARM_CONFIGS[arm]
        
        if arm_config.camera_name not in self.cameras:
            print(f"✗ 相机 {arm_config.camera_name} 未连接")
            return False
        
        self.controller.switch_arm(arm)
        self.controller.camera = self.cameras[arm_config.camera_name]
        self.current_arm = arm
        
        return True
    
    # ----------------- 首次使用引导 -----------------
    
    def show_first_run_guide(self):
        """首次使用引导"""
        print("\n" + "="*60)
        print("欢迎使用精准放置系统!")
        print("="*60)
        
        print("""
检测到这是首次使用，请按以下步骤操作:

┌──────────────────────────────────────────────────────────┐
│ 步骤1: 准备标记                                          │
│   ├── 工件上贴 2个绿色圆形贴纸 (靠近定位孔)              │
│   └── 卡槽上贴 2个红色圆形贴纸 (靠近定位销)              │
│                                                          │
│ 步骤2: 连接设备                                          │
│   └── 选择菜单选项 1                                     │
│                                                          │
│ 步骤3: 测试检测                                          │
│   ├── 选择菜单选项 2                                     │
│   ├── 确认能看到 4个标记                                 │
│   └── 调整高度/光照直到检测稳定                          │
│                                                          │
│ 步骤4: 标定                                              │
│   ├── 选择菜单选项 3                                     │
│   └── 按提示精确移动机器人 5mm                           │
│                                                          │
│ 步骤5: 测试完整流程                                      │
│   └── 选择菜单选项 5                                     │
│                                                          │
│ 标记要求:                                                │
│   - 直径约 1-2cm                                         │
│   - 颜色对比明显                                          │
│   - 不能被夹爪遮挡                                        │
└──────────────────────────────────────────────────────────┘
""")
        
        input("\n按 Enter 继续...")
    
    # ----------------- 标定 -----------------

    def calibrate(self):
        """像素-毫米标定"""
        if not self.controller:
            print("请先连接设备")
            return

        self.controller.calibrate()

    def calibrate_joints(self):
        """关节灵敏度标定 (多点标定)"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*60)
        print("关节灵敏度标定")
        print("="*60)
        print("""
说明:
  此标定会记录每个关节移动1度时，相机画面移动多少像素。
  建议在3个不同高度各做一次标定:
    - 高位置 (卡槽上方约15cm)
    - 中位置 (卡槽上方约10cm)
    - 低位置 (卡槽上方约5cm)

流程:
  对于每个关节，系统会:
  1. 拍摄当前画面
  2. 提示你手动移动关节2度
  3. 拍摄移动后画面
  4. 计算灵敏度
""")

        input("\n按 Enter 开始...")

        try:
            move_deg = float(input("移动角度 (默认2度): ").strip() or "2.0")
        except:
            move_deg = 2.0

        self.controller.calibrate_all_joints(move_deg)

    def show_calibration_history(self):
        """显示标定历史"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*60)
        print("标定历史")
        print("="*60)

        # 像素-毫米标定历史
        print("\n[像素-毫米标定]")
        self.controller.show_calibration_history()

        # 关节灵敏度标定点
        print("\n[关节灵敏度标定点]")
        self.controller.show_calibration_points()
    
    # ----------------- 预设位置 -----------------
    
    def save_preset(self):
        """保存预设位置"""
        if not self.controller:
            print("请先连接设备")
            return
        
        name = input("请输入预设名称: ").strip()
        if name:
            self.controller.save_preset(name)
    
    def load_preset(self):
        """加载预设位置"""
        if not self.controller:
            print("请先连接设备")
            return
        
        self.controller.list_presets()
        name = input("请输入预设名称: ").strip()
        if name:
            self.controller.load_preset(name)
    
    # ----------------- 测试 -----------------
    
    def test_detection(self):
        """测试检测"""
        if not self.controller:
            print("请先连接设备")
            return
        
        self.controller.test_detection()
    
    # ----------------- 对齐 -----------------
    
    def run_alignment(self, auto_place: bool = False):
        """运行对齐"""
        if not self.controller:
            print("请先连接设备")
            return
        
        print("\n请将机器人移动到卡槽上方 (5-10cm)")
        input("准备好后按 Enter...")
        
        self.controller.run_full_sequence(
            tolerance_mm=2.0,
            auto_place=auto_place
        )
    
    def run_full(self):
        """完整流程"""
        if not self.controller:
            print("请先连接设备")
            return
        
        print("\n" + "#"*60)
        print("# 完整精准放置流程")
        print("#"*60)
        
        start = time.time()
        
        # 1. 抓取
        print("\n[步骤1] 抓取工件")
        input("请手动抓取，完成后按 Enter...")
        
        # 2. 移动
        print("\n[步骤2] 移动到卡槽上方")
        input("请移动到大致位置，完成后按 Enter...")
        
        # 3. 对齐
        print("\n[步骤3] 自动对齐")
        success = self.controller.run_full_sequence(tolerance_mm=2.0, auto_place=False)
        
        if not success:
            print("\n警告: 对齐未达精度")
        
        # 4. 放置
        print("\n[步骤4] 放置")
        
        # 询问是否自动放置
        auto = input("自动放置? (y/n): ").strip().lower() == 'y'
        
        if auto:
            self.controller.auto_place()
        else:
            print("请手动下降并放置")
            input("完成后按 Enter...")
        
        # 5. 撤退
        print("\n[步骤5] 撤退")
        
        if not auto:
            input("请手动抬起，完成后按 Enter...")
        
        elapsed = time.time() - start
        print("\n" + "#"*60)
        print(f"# 完成! 耗时: {elapsed:.1f}秒")
        print(f"# 对齐结果: {'成功' if success else '未达精度'}")
        print("#"*60)
        
        return success
    
    def continuous_run(self, count: int = 10):
        """连续运行"""
        results = []
        
        for i in range(count):
            print(f"\n{'='*60}")
            print(f"第 {i+1}/{count} 次")
            print('='*60)
            
            success = self.run_full()
            results.append(success)
            
            if i < count - 1:
                input("\n按 Enter 继续...")
        
        rate = sum(results) / len(results) * 100
        print(f"\n统计: 成功率 {rate:.1f}%")


# ==================== 主函数 ====================

def main():
    system = PrecisionPlaceSystem()
    
    print("\n" + "="*60)
    print("精准放置系统 (优化版)")
    print("="*60)
    
    # 首次使用引导
    if system.is_first_run:
        system.show_first_run_guide()
    
    try:
        while True:
            print("\n" + "-"*40)
            print("1. 连接设备")
            print("2. 切换左手/右手")
            print("3. 测试检测")
            print("4. 标定")
            print("5. 标定历史")
            print("6. 预设位置")
            print("7. 运行对齐")
            print("8. 完整流程")
            print("9. 连续运行 (10次)")
            print("0. 退出")
            
            choice = input("\n选项: ").strip()
            
            if choice == "1":
                # 选择手臂
                print("\n选择手臂:")
                print("  1. 右手 (默认)")
                print("  2. 左手")
                arm_choice = input("选项: ").strip()
                arm = "left" if arm_choice == "2" else "right"
                system.connect(arm)
                
            elif choice == "2":
                if not system.controller:
                    print("请先连接设备")
                    continue
                
                print("\n选择手臂:")
                print("  1. 右手")
                print("  2. 左手")
                arm_choice = input("选项: ").strip()
                arm = "left" if arm_choice == "2" else "right"
                system.switch_arm(arm)
                
            elif choice == "3":
                if not system.controller:
                    system.connect()
                system.test_detection()
                
            elif choice == "4":
                print("\n标定选项:")
                print("  1. 像素-毫米标定 (基础标定)")
                print("  2. 关节灵敏度标定 (多点标定，推荐)")
                print("  3. 运行全部标定")

                calib_choice = input("选项: ").strip()

                if calib_choice == "1":
                    if not system.controller:
                        system.connect()
                    system.calibrate()
                elif calib_choice == "2":
                    if not system.controller:
                        system.connect()
                    system.calibrate_joints()
                elif calib_choice == "3":
                    if not system.controller:
                        system.connect()
                    print("\n[1/2] 像素-毫米标定")
                    system.calibrate()
                    print("\n[2/2] 关节灵敏度标定")
                    system.calibrate_joints()
                else:
                    print("无效选项")
                
            elif choice == "5":
                system.show_calibration_history()
                
            elif choice == "6":
                print("\n预设位置:")
                print("  1. 保存当前位置")
                print("  2. 加载预设位置")
                print("  3. 列出所有预设")
                
                preset_choice = input("选项: ").strip()
                
                if preset_choice == "1":
                    if not system.controller:
                        system.connect()
                    system.save_preset()
                elif preset_choice == "2":
                    if not system.controller:
                        system.connect()
                    system.load_preset()
                elif preset_choice == "3":
                    if system.controller:
                        system.controller.list_presets()
                    else:
                        print("请先连接设备")
                        
            elif choice == "7":
                if not system.controller:
                    system.connect()
                system.run_alignment()
                
            elif choice == "8":
                if not system.controller:
                    system.connect()
                system.run_full()
                
            elif choice == "9":
                if not system.controller:
                    system.connect()
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
