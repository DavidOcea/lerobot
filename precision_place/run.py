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

# 添加 lerobot 根目录和 src 目录到 Python 路径
LEROBOT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(LEROBOT_ROOT))

from lerobot.robots.supre_robot_follower import SupreRobotFollower
from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from precision_place.dual_point_alignment import (
    PrecisionPlaceController, ARM_CONFIGS, DualPointDetector, JointSensitivity
)


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
        self.passive_mode = False  # 被动模式：与示教系统协同

    def connect(self, arm: str = "right", passive: bool = False):
        """连接设备

        Args:
            arm: 手臂选择
            passive: 被动模式，与示教系统协同工作
        """
        print("\n" + "="*60)
        print("连接设备")
        print("="*60)

        self.passive_mode = passive
        if passive:
            print("\n[被动模式] 与示教系统协同工作")
            print("  - 只读取机器人状态，不发送控制指令")
            print("  - 请确保示教程序已启动 (./run.sh)")
            print("  - 跳过机器人硬件连接，避免冲突")

        # 机器人 - 被动模式下跳过连接，避免与示教程序冲突
        if passive:
            print("\n跳过机器人连接 (被动模式)")
            self.robot = None
        else:
            print("\n连接机器人...")
            try:
                self.robot = SupreRobotFollower(SupreRobotFollowerConfig())
                self.robot.connect()
                print("✓ 机器人已连接")
            except Exception as e:
                raise

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
            # 如果机器人连接失败但相机可用，创建一个简化的控制器
            if self.robot is None:
                print(f"\n⚠ 使用相机模式（无机器人连接）")
                self.controller = self._create_camera_only_controller(arm)
            else:
                self.controller = PrecisionPlaceController(
                    robot=self.robot,
                    camera=self.cameras[arm_config.camera_name],
                    arm=arm,
                    passive_mode=passive
                )
            self.controller.set_marker_colors(WORKPIECE_COLOR, SLOT_COLOR)
            print(f"\n✓ 主用相机: {arm_config.camera_name} (索引{arm_config.camera_index})")
            print(f"✓ 使用手臂: {arm}")
            if passive:
                print("✓ 模式: 被动模式（与示教协同）")
        else:
            raise RuntimeError(f"无法连接相机 {arm_config.camera_name}")

    def _create_camera_only_controller(self, arm: str):
        """创建仅相机的控制器（用于测试或与示教系统配合）"""
        arm_config = ARM_CONFIGS.get(arm)

        # 导入共享状态模块
        try:
            from precision_place.robot_status import RobotStatusReader, joints_dict_to_array
            status_reader = RobotStatusReader()
            print("  ✓ 已加载共享状态读取器")
        except ImportError:
            status_reader = None
            print("  ⚠ 无法加载共享状态模块")

        class CameraOnlyController:
            """仅相机控制器，用于被动模式，从共享文件读取关节状态"""
            def __init__(self, camera, arm_config, status_reader):
                self.camera = camera
                self.arm_config = arm_config
                self.arm = arm
                self.passive_mode = True
                self.detector = DualPointDetector()
                self.calibration_points = []
                self.pixel_to_mm_ratio = 0.5
                self.joint_names = {}
                self.status_reader = status_reader
                # 构建关节名称
                for i in range(7):
                    self.joint_names[i] = f"left_arm_joint_{i+1}"
                for i in range(7, 14):
                    self.joint_names[i] = f"right_arm_joint_{i-6}"
                self.joint_names[14] = "trunk_joint_1"
                self.joint_names[15] = "trunk_joint_2"

            def set_marker_colors(self, wp_color, slot_color):
                self.detector.set_marker_colors(wp_color, slot_color)

            def get_joint_states(self, max_age_ms=200):
                """从共享文件读取关节状态"""
                if self.status_reader is None:
                    return None
                joints_dict = self.status_reader.read_joints(max_age_ms)
                if joints_dict is None:
                    return None
                return joints_dict_to_array(joints_dict)

            def test_detection(self):
                print("\n检测测试 (仅相机模式)")
                print("按 'q' 退出")
                while True:
                    image = self.camera.read()
                    if image is None:
                        continue
                    state = self.detector.detect_dual_marker_state(image)
                    vis = self.detector.visualize(image, state)
                    cv2.imshow("Detection Test", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()

            def calibrate_joint_sensitivity(self, joint_idx, move_degrees=2.0):
                print(f"\n{'='*60}")
                print(f"关节灵敏度标定 (共享状态模式)")
                print(f"关节: {self.joint_names.get(joint_idx, f'joint_{joint_idx}')}")
                print(f"{'='*60}")

                # 从共享文件读取初始关节状态
                joints = self.get_joint_states()
                if joints is None:
                    print("✗ 无法获取关节位置")
                    print("  请确认示教程序已启动并启用 share_status=true")
                    return False, None

                current_angle = joints[joint_idx]
                print(f"\n当前关节角度: {current_angle:.2f}°")
                print(f"\n[示教模式] 请用示教器移动关节约 {move_degrees}°")
                print(f"  目标角度: 约 {current_angle + move_degrees:.2f}°")

                input("按 Enter 开始采集初始图像...")

                img1 = self.camera.read()
                if img1 is None:
                    print("✗ 图像采集失败")
                    return False, None

                input(f"\n移动完成后按 Enter...")

                # 读取移动后的关节状态
                joints_after = self.get_joint_states()
                if joints_after is None:
                    print("✗ 无法获取移动后的关节位置")
                    return False, None

                actual_move = joints_after[joint_idx] - current_angle
                print(f"  实际移动: {actual_move:.2f}°")

                if abs(actual_move) < 0.1:
                    print(f"⚠ 警告: 移动角度很小，标定可能不准确")

                img2 = self.camera.read()
                if img2 is None:
                    print("✗ 图像采集失败")
                    return False, None

                # 计算像素变化
                g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                pts = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
                if pts is None or len(pts) < 10:
                    print("✗ 特征点匹配失败")
                    return False, None

                p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None)
                if p1 is None:
                    print("✗ 光流计算失败")
                    return False, None

                good = p1[st == 1] - pts[st == 1]
                pixel_dx = float(np.mean(good, axis=0)[0])
                pixel_dy = float(np.mean(good, axis=0)[1])

                # 使用实际移动角度
                actual_deg = abs(actual_move) if abs(actual_move) > 0.1 else move_degrees
                sensitivity = JointSensitivity(
                    joint_idx=joint_idx,
                    joint_name=self.joint_names.get(joint_idx, f"joint_{joint_idx}"),
                    pixel_dx_per_deg=pixel_dx / actual_deg,
                    pixel_dy_per_deg=pixel_dy / actual_deg
                )

                print(f"\n标定结果:")
                print(f"  实际移动: {actual_move:.2f}°")
                print(f"  像素变化: ({pixel_dx:.1f}, {pixel_dy:.1f}) pixels")
                print(f"  灵敏度: X={sensitivity.pixel_dx_per_deg:.2f} px/deg, Y={sensitivity.pixel_dy_per_deg:.2f} px/deg")

                return True, sensitivity

            def calibrate_all_joints(self, move_degrees=2.0):
                print("\n[仅相机模式] 多点标定")
                print("需要手动输入关节移动角度")
                primary_joints = self.arm_config.primary_joints

                for i, jidx in enumerate(primary_joints):
                    print(f"\n[{i+1}/{len(primary_joints)}] 标定关节 {jidx}")
                    success, sens = self.calibrate_joint_sensitivity(jidx, move_degrees)
                    if success:
                        self.calibration_points.append(sens)

                return len(self.calibration_points) > 0

            def show_calibration_points(self):
                print("\n标定点:")
                for i, cp in enumerate(self.calibration_points):
                    print(f"  [{i+1}] {cp.joint_name}: ({cp.pixel_dx_per_deg:.2f}, {cp.pixel_dy_per_deg:.2f}) px/deg")

        return CameraOnlyController(self.cameras[arm_config.camera_name], arm_config, status_reader)
    
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

        if self.passive_mode:
            print("""
[示教模式说明]
  1. 确保示教程序已启动: ./run.sh
  2. 移动示教器对应关节，执行机器人会跟随
  3. 系统会自动读取实际移动角度

  关节对应 (右手):
    关节 7 = right_arm_joint_1 (底座旋转)
    关节 8 = right_arm_joint_2 (肩部俯仰)
    关节 9 = right_arm_joint_3 (肩部侧摆)
    关节 10 = right_arm_joint_4 (肘部俯仰)
""")
        else:
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

                # 选择模式
                print("\n连接模式:")
                print("  1. 独立模式 (默认) - 直接控制机器人")
                print("  2. 示教模式 - 与示教系统协同，只读取状态")
                mode_choice = input("选项: ").strip()
                passive = (mode_choice == "2")

                if passive:
                    print("\n" + "="*50)
                    print("示教模式使用说明")
                    print("="*50)
                    print("1. 请在另一个终端启动示教程序:")
                    print("   cd /home/smai/dc_dir && ./run.sh")
                    print("2. 等待示教程序启动完成")
                    print("3. 此程序将只读取机器人状态，不发送控制指令")
                    print("4. 使用示教器移动关节进行标定")

                system.connect(arm, passive=passive)
                
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
