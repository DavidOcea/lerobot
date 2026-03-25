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
import json
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

# 尝试导入Z轴控制器
try:
    from precision_place.z_axis_controller import ZAxisController
    _has_z_controller = True
except ImportError:
    _has_z_controller = False
    ZAxisController = None


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
                config = SupreRobotFollowerConfig(
                    joint_config_file="trunk_config_supre_robot_joint.yaml"
                )
                self.robot = SupreRobotFollower(config)
                self.robot.connect()
                print("✓ 机器人已连接")
            except Exception as e:
                raise

        # 相机
        print("\n连接相机...")
        for name, idx in CAMERA_INDICES.items():
            try:
                from lerobot.cameras.opencv.configuration_opencv import ColorMode
                # 使用 BGR 格式以匹配 OpenCV 的绘图和显示要求
                config = OpenCVCameraConfig(
                    index_or_path=idx, fps=30, width=640, height=480,
                    color_mode=ColorMode.BGR
                )
                self.cameras[name] = OpenCVCamera(config)
                self.cameras[name].connect()
                print(f"  ✓ {name} (索引{idx})")
            except Exception as e:
                print(f"  ✗ {name} (索引{idx}): {e}")

        # 控制器
        self.current_arm = arm
        arm_config = ARM_CONFIGS.get(arm)

        if arm_config.camera_name in self.cameras:
            # 获取副相机（用于Z轴深度估计）
            camera2 = None
            if arm_config.camera2_name and arm_config.camera2_name in self.cameras:
                camera2 = self.cameras[arm_config.camera2_name]
                print(f"✓ 副用相机: {arm_config.camera2_name} (索引{arm_config.camera2_index})")

            # 统一使用 PrecisionPlaceController（支持被动模式和Z轴控制）
            self.controller = PrecisionPlaceController(
                robot=self.robot,  # 被动模式下为 None
                camera=self.cameras[arm_config.camera_name],
                arm=arm,
                passive_mode=passive,
                camera2=camera2
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
                self.arm = arm_config.name  # 从 arm_config 获取
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
                # 预设位置
                self.presets: Dict[str, np.ndarray] = {}
                self._load_presets()

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

            def calibrate_joint_sensitivity(self, joint_idx, move_degrees=4.0):
                """关节灵敏度标定 (带实时视频显示)"""
                joint_name = self.joint_names.get(joint_idx, f"joint_{joint_idx}")
                print(f"\n{'='*60}")
                print(f"关节灵敏度标定: {joint_name}")
                print(f"{'='*60}")

                # 从共享文件读取初始关节状态
                joints = self.get_joint_states()
                if joints is None:
                    print("✗ 无法获取关节位置")
                    print("  请确认示教程序已启动并启用 share_status=true")
                    return False, None

                initial_angle = joints[joint_idx]
                target_angle = initial_angle + move_degrees

                print(f"\n初始角度: {initial_angle:.2f}°")
                print(f"目标角度: {target_angle:.2f}° (移动约 {move_degrees}°)")
                print(f"\n[视频窗口] 按 'Enter' 采集图像，按 'q' 取消")

                window_name = f"Calibration: {joint_name}"
                img1 = None
                img2 = None
                phase = 1  # 1=采集初始图像, 2=采集移动后图像

                while True:
                    # 读取图像
                    frame = self.camera.read()
                    if frame is None:
                        continue

                    # 检测标记点并可视化
                    state = self.detector.detect_dual_marker_state(frame)
                    vis = self.detector.visualize(frame, state)

                    # 获取当前关节角度
                    current_joints = self.get_joint_states()
                    current_angle = current_joints[joint_idx] if current_joints is not None else 0.0

                    # 叠加信息 - 使用背景框提高可读性
                    # 顶部信息框
                    cv2.rectangle(vis, (5, 5), (280, 110), (0, 0, 0), -1)
                    cv2.rectangle(vis, (5, 5), (280, 110), (255, 255, 255), 1)

                    info_y = 25
                    cv2.putText(vis, f"Joint: {joint_name}", (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    info_y += 28
                    cv2.putText(vis, f"Current: {current_angle:.2f} deg", (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    info_y += 28
                    cv2.putText(vis, f"Target: +/-{move_degrees:.1f} deg", (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    info_y += 28

                    # 显示阶段提示 - 底部信息框
                    bottom_h = 70 if phase == 2 else 50
                    cv2.rectangle(vis, (5, vis.shape[0] - bottom_h - 5), (vis.shape[1] - 5, vis.shape[0] - 5), (0, 0, 0), -1)
                    cv2.rectangle(vis, (5, vis.shape[0] - bottom_h - 5), (vis.shape[1] - 5, vis.shape[0] - 5), (255, 255, 255), 1)

                    if phase == 1:
                        cv2.putText(vis, "[Phase 1] Press ENTER to capture initial image", (15, vis.shape[0] - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(vis, "Press 'q' to cancel", (15, vis.shape[0] - 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                    else:
                        # 显示移动量
                        moved = current_angle - initial_angle
                        move_color = (0, 255, 0) if abs(moved) >= move_degrees * 0.8 else (0, 165, 255)
                        status = "OK" if abs(moved) >= move_degrees * 0.8 else "Move more"
                        cv2.putText(vis, f"Moved: {moved:.2f} deg [{status}]", (15, vis.shape[0] - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, move_color, 2)
                        cv2.putText(vis, "[Phase 2] Press ENTER to capture final image", (15, vis.shape[0] - 28),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(vis, "Press 'q' to cancel", (15, vis.shape[0] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                    cv2.imshow(window_name, vis)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        cv2.destroyWindow(window_name)
                        print("✗ 标定已取消")
                        return False, None

                    elif key == 13 or key == 10:  # Enter key
                        if phase == 1:
                            img1 = frame.copy()
                            print(f"  ✓ 已采集初始图像 (角度: {current_angle:.2f}°)")
                            phase = 2
                            print(f"\n  请用示教器移动关节到目标位置 ({target_angle:.2f}°)")
                        else:
                            img2 = frame.copy()
                            final_angle = current_angle
                            print(f"  ✓ 已采集移动后图像 (角度: {final_angle:.2f}°)")
                            break

                cv2.destroyWindow(window_name)

                # 计算实际移动角度
                actual_move = final_angle - initial_angle
                print(f"\n实际移动: {actual_move:.2f}°")

                if abs(actual_move) < 0.1:
                    print("⚠ 警告: 移动角度很小，标定可能不准确")

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
                # 检查是否有足够的有效匹配点
                if len(good) < 5:
                    print("✗ 有效特征点不足")
                    return False, None

                pixel_dx = float(np.mean(good, axis=0)[0])
                pixel_dy = float(np.mean(good, axis=0)[1])

                # 使用实际移动角度
                actual_deg = abs(actual_move) if abs(actual_move) > 0.1 else move_degrees
                sensitivity = JointSensitivity(
                    joint_idx=joint_idx,
                    joint_name=joint_name,
                    pixel_dx_per_deg=pixel_dx / actual_deg,
                    pixel_dy_per_deg=pixel_dy / actual_deg
                )

                print(f"\n标定结果:")
                print(f"  实际移动: {actual_move:.2f}°")
                print(f"  像素变化: ({pixel_dx:.1f}, {pixel_dy:.1f}) pixels")
                print(f"  灵敏度: X={sensitivity.pixel_dx_per_deg:.2f} px/deg, Y={sensitivity.pixel_dy_per_deg:.2f} px/deg")

                return True, sensitivity

            def calibrate_all_joints(self, move_degrees=4.0):
                print("\n[仅相机模式] 多点标定")
                print("需要手动输入关节移动角度")
                primary_joints = self.arm_config.primary_joints

                for i, jidx in enumerate(primary_joints):
                    print(f"\n[{i+1}/{len(primary_joints)}] 标定关节 {jidx}")
                    success, sens = self.calibrate_joint_sensitivity(jidx, move_degrees)
                    if success:
                        self.calibration_points.append(sens)

                return len(self.calibration_points) > 0

            def calibrate_all_joints_auto(self, move_degrees=4.0, settle_time=0.5, return_after_calib=True):
                """自动标定（被动模式下不支持）"""
                print("\n✗ 自动标定需要独立模式（不能使用被动模式）")
                print("  请使用 '独立模式' 连接设备")
                return False

            def show_calibration_points(self):
                print("\n标定点:")
                for i, cp in enumerate(self.calibration_points):
                    print(f"  [{i+1}] {cp.joint_name}: ({cp.pixel_dx_per_deg:.2f}, {cp.pixel_dy_per_deg:.2f}) px/deg")

            def _load_presets(self):
                """加载预设"""
                path = Path(__file__).parent / "presets.json"
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            self.presets = {k: np.array(v) for k, v in data.items()}
                    except Exception as e:
                        print(f"  ⚠ 加载预设失败: {e}")
                        self.presets = {}

            def _save_presets(self):
                """保存预设"""
                path = Path(__file__).parent / "presets.json"
                try:
                    with open(path, 'w') as f:
                        json.dump({k: v.tolist() for k, v in self.presets.items()}, f, indent=2)
                except Exception as e:
                    print(f"  ⚠ 保存预设失败: {e}")

            def list_presets(self):
                """列出预设位置"""
                if not self.presets:
                    print("\n  暂无预设位置")
                    print("  提示: 选择 '保存当前位置' 添加预设")
                    return

                print("\n预设位置列表:")
                for name in self.presets:
                    print(f"  - {name}")

            def save_preset(self, name):
                """保存预设位置 (从共享状态读取当前位置)"""
                joints = self.get_joint_states()
                if joints is None:
                    print(f"  ✗ 无法获取当前位置")
                    print("  请确认示教程序已启动并启用状态共享")
                    return

                self.presets[name] = joints.copy()
                self._save_presets()
                print(f"  ✓ 预设位置已保存: {name}")

            def load_preset(self, name):
                """加载预设位置 (被动模式不支持移动)"""
                print(f"  [被动模式] 无法移动到预设位置: {name}")
                print("  请使用示教器手动移动，或使用独立模式")

            def show_calibration_history(self):
                """显示标定历史 (被动模式)"""
                print("\n[被动模式] 标定历史")
                if not self.calibration_points:
                    print("  无标定数据")
                else:
                    for i, cp in enumerate(self.calibration_points):
                        print(f"  [{i+1}] {cp.joint_name}: ({cp.pixel_dx_per_deg:.2f}, {cp.pixel_dy_per_deg:.2f}) px/deg")

            def calibrate(self, move_distance_mm: float = 10.0):
                """像素-毫米标定 (被动模式，带视频显示)"""
                print("\n" + "="*50)
                print("像素-毫米标定 (被动模式)")
                print("="*50)

                print(f"\n[说明] 请用示教器将机器人沿X方向精确移动 {move_distance_mm}mm")
                print(f"  标定结果将用于像素到毫米的转换")
                print(f"\n[重要提示]")
                print(f"  1. 请确保标记点在相机视野内")
                print(f"  2. 移动方向应沿画面水平方向(X方向)")
                print(f"  3. 移动距离建议 {move_distance_mm}mm 或更多以确保像素变化明显")

                window_name = "Pixel-MM Calibration"
                img1 = None
                img2 = None
                phase = 1

                print(f"\n[视频窗口] 按 'Enter' 采集图像，按 'q' 取消")

                while True:
                    frame = self.camera.read()
                    if frame is None:
                        continue

                    # 检测标记点并可视化
                    state = self.detector.detect_dual_marker_state(frame)
                    vis = self.detector.visualize(frame, state)

                    # 绘制移动方向指引箭头
                    h, w = vis.shape[:2]
                    arrow_y = h - 100
                    arrow_start = (w // 4, arrow_y)
                    arrow_end = (3 * w // 4, arrow_y)

                    if phase == 1:
                        # 第一阶段：采集初始图像
                        cv2.putText(vis, "Phase 1: Press ENTER to capture initial image", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(vis, "Press 'q' to cancel", (10, 55),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                        # 提示标记检测状态
                        if state.workpiece_detected or state.slot_detected:
                            cv2.putText(vis, f"Markers detected: WP={state.workpiece_marker_count}, Slot={state.slot_marker_count}",
                                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            cv2.putText(vis, "No markers detected - check camera view", (10, 80),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        # 第二阶段：移动机器人并采集最终图像
                        # 绘制大的移动方向箭头
                        cv2.arrowedLine(vis, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.1)
                        cv2.putText(vis, f"Move {move_distance_mm}mm in X direction", (arrow_start[0], arrow_y - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(vis, "-->", (w // 2 - 20, arrow_y + 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                        cv2.putText(vis, "Phase 2: Move robot, then press ENTER", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(vis, "Press ENTER to capture final image", (10, 55),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(vis, "Press 'q' to cancel", (10, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

                    cv2.imshow(window_name, vis)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        cv2.destroyWindow(window_name)
                        print("✗ 标定已取消")
                        return False, 0.0

                    elif key == 13 or key == 10:  # Enter key
                        if phase == 1:
                            img1 = frame.copy()
                            print("  ✓ 已采集初始图像")
                            phase = 2
                            print(f"\n  >>> 请用示教器将机器人沿X方向移动 {move_distance_mm}mm <<<")
                            print(f"      (画面上黄色箭头指示移动方向)")
                        else:
                            img2 = frame.copy()
                            print("  ✓ 已采集移动后图像")
                            break

                cv2.destroyWindow(window_name)

                # 计算像素变化
                g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                pts = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
                if pts is None or len(pts) < 10:
                    print("✗ 特征点匹配失败")
                    return False, 0.0

                p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None)
                if p1 is None:
                    print("✗ 光流计算失败")
                    return False, 0.0

                good = p1[st == 1] - pts[st == 1]
                # 检查是否有足够的有效匹配点
                if len(good) < 5:
                    print("✗ 有效特征点不足")
                    return False, 0.0

                pixel_dx = float(np.mean(good, axis=0)[0])

                # 计算像素-毫米比例
                if abs(pixel_dx) > 0.1:
                    pixel_to_mm = move_distance_mm / abs(pixel_dx)
                else:
                    print("✗ 像素变化太小，标定失败")
                    return False, 0.0

                self.pixel_to_mm_ratio = pixel_to_mm

                print(f"\n标定结果:")
                print(f"  移动距离: {move_distance_mm} mm")
                print(f"  像素变化: {pixel_dx:.1f} pixels")
                print(f"  像素/毫米: {pixel_to_mm:.4f} mm/pixel")
                print(f"✓ 标定完成")

                return True, pixel_to_mm

            def set_marker_colors(self, wp_color, slot_color):
                """设置标记颜色"""
                self.detector.set_marker_colors(wp_color, slot_color)

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

        # 让用户输入移动距离
        print("\n[像素-毫米标定]")
        print("  此标定需要您手动移动机器人来确定像素到毫米的转换比例")
        print("  移动距离越大，标定精度越高")

        try:
            move_mm = float(input("移动距离mm (默认10mm): ").strip() or "10.0")
        except:
            move_mm = 10.0

        self.controller.calibrate(move_mm)

    def calibrate_joints(self):
        """关节灵敏度标定 (多点标定)"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*60)
        print("关节灵敏度标定")
        print("="*60)

        # 获取当前配置的关节列表
        arm_config = ARM_CONFIGS.get(self.current_arm)
        primary_joints = arm_config.primary_joints if arm_config else []

        if self.passive_mode:
            print("""
[示教模式说明]
  1. 确保示教程序已启动: ./run.sh
  2. 移动示教器对应关节，执行机器人会跟随
  3. 系统会自动读取实际移动角度
  4. 视频窗口按 Enter 采集图像，按 q 取消
""")
            # 动态显示关节列表
            print(f"  将标定 {len(primary_joints)} 个关节 ({self.current_arm}手):")
            for i, jidx in enumerate(primary_joints):
                joint_name = self.controller.joint_names.get(jidx, f"joint_{jidx}")
                print(f"    {i+1}. 关节 {jidx} = {joint_name}")
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
  2. 提示你手动移动关节4度
  3. 拍摄移动后画面
  4. 计算灵敏度
""")
            # 动态显示关节列表
            print(f"  将标定 {len(primary_joints)} 个关节 ({self.current_arm}手):")
            for i, jidx in enumerate(primary_joints):
                joint_name = self.controller.joint_names.get(jidx, f"joint_{jidx}")
                print(f"    {i+1}. 关节 {jidx} = {joint_name}")

        input("\n按 Enter 开始...")

        try:
            move_deg = float(input("移动角度 (默认4度): ").strip() or "4.0")
        except:
            move_deg = 4.0

        self.controller.calibrate_all_joints(move_deg)

    def calibrate_joints_auto(self):
        """关节灵敏度自动标定（无需手动移动）"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*60)
        print("关节灵敏度自动标定")
        print("="*60)

        # 检查是否为被动模式
        if self.passive_mode:
            print("\n✗ 自动标定需要独立模式（不能使用被动模式）")
            print("  请选择 '独立模式' 连接设备")
            return

        # 获取当前配置的关节列表
        arm_config = ARM_CONFIGS.get(self.current_arm)
        primary_joints = arm_config.primary_joints if arm_config else []

        print("""
说明:
  自动标定会自动移动每个关节4度并采集图像。
  不需要手动操作，系统会完成以下步骤:
    1. 采集初始图像
    2. 自动移动关节
    3. 等待稳定
    4. 采集移动后图像
    5. 计算灵敏度

建议在3个不同高度各做一次标定:
    - 高位置 (卡槽上方约15cm)
    - 中位置 (卡槽上方约10cm)
    - 低位置 (卡槽上方约5cm)
""")
        print(f"  将自动标定 {len(primary_joints)} 个关节 ({self.current_arm}手):")
        for i, jidx in enumerate(primary_joints):
            joint_name = self.controller.joint_names.get(jidx, f"joint_{jidx}")
            print(f"    {i+1}. 关节 {jidx} = {joint_name}")

        input("\n按 Enter 开始自动标定...")

        try:
            move_deg = float(input("移动角度 (默认4度): ").strip() or "4.0")
        except:
            move_deg = 4.0

        settle_time = 1.0
        try:
            settle_time = float(input("稳定等待时间秒 (默认1.0): ").strip() or "1.0")
        except:
            settle_time = 1.0

        return_after = True
        ans = input("标定后返回初始位置? (y/n，默认y): ").strip().lower()
        if ans == 'n':
            return_after = False

        self.controller.calibrate_all_joints_auto(move_deg, settle_time, return_after)

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

        # Z轴标定数据
        print("\n[Z轴标定数据]")
        self._show_z_axis_calibration()

    def verify_sensitivity_direction(self):
        """验证灵敏度方向是否正确"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n此功能将移动关节并观察像素变化来验证灵敏度方向")
        print("如果方向不正确，对齐时会向错误方向移动")
        print("使用多帧平均提高检测稳定性")
        print("建议: 站立姿态下joint_8(肩部俯仰)Z变化最小，检测最稳定")

        try:
            joint_idx = int(input("测试关节索引 (默认8=right_arm_joint_2 肩部俯仰): ").strip() or "8")
            move_deg = float(input("移动角度 (默认2.0): ").strip() or "2.0")
            num_samples = int(input("采样帧数 (默认5): ").strip() or "5")
        except:
            joint_idx = 8
            move_deg = 2.0
            num_samples = 5

        self.controller.verify_sensitivity_direction(joint_idx, move_deg, num_samples)

    def flip_sensitivity_direction(self):
        """翻转灵敏度方向（修复相机方向问题）"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n翻转灵敏度方向")
        print("当两个相机安装方向相反时，灵敏度符号需要翻转")
        print("\n当前状态:")
        print(f"  X翻转: {self.controller._camera_flip_x}")
        print(f"  Y翻转: {self.controller._camera_flip_y}")

        print("\n选项:")
        print("  1. 翻转X方向")
        print("  2. 翻转Y方向")
        print("  3. 翻转X和Y方向")
        print("  4. 取消翻转")
        print("  0. 取消")

        choice = input("选项: ").strip()

        if choice == "1":
            self.controller._camera_flip_x = True
            self.controller._camera_flip_y = False
        elif choice == "2":
            self.controller._camera_flip_x = False
            self.controller._camera_flip_y = True
        elif choice == "3":
            self.controller._camera_flip_x = True
            self.controller._camera_flip_y = True
        elif choice == "4":
            self.controller._camera_flip_x = False
            self.controller._camera_flip_y = False
        else:
            print("已取消")
            return

        print(f"\n已更新: X翻转={self.controller._camera_flip_x}, Y翻转={self.controller._camera_flip_y}")
        print("注意: 此设置仅在当前会话有效")
        print("如需永久生效，请在 ARM_CONFIGS 中设置 camera_flip")

    def _show_z_axis_calibration(self):
        """显示Z轴标定数据"""
        if not _has_z_controller:
            print("  Z轴控制器未加载")
            return

        import json
        from pathlib import Path

        z_calib_path = Path(__file__).parent / "z_axis_calibration.json"
        if z_calib_path.exists():
            with open(z_calib_path, 'r') as f:
                data = json.load(f)

            baseline = data.get('baseline', 0)
            marker_diameter = data.get('marker_diameter', 15.0)

            print(f"  双目基线: {baseline:.1f} mm")
            print(f"  标记直径: {marker_diameter:.1f} mm")

            sensitivities = data.get('joint_sensitivities', {})
            if sensitivities:
                print("  Z轴关节灵敏度:")
                for k, v in sensitivities.items():
                    joint_name = v.get('joint_name', f'joint_{k}')
                    mm_per_deg = v.get('mm_per_deg', 0)
                    calib_height = v.get('calibration_height', 0)
                    print(f"    {joint_name}: {mm_per_deg:.2f} mm/deg @ {calib_height:.0f}mm")
        else:
            print("  暂无Z轴标定数据")

    # ----------------- Z轴标定 -----------------

    def calibrate_z_axis_joints(self):
        """Z轴关节灵敏度标定（手动/示教模式）"""
        if not self.controller:
            print("请先连接设备")
            return

        if not _has_z_controller:
            print("✗ Z轴控制器未加载")
            return

        print("\n" + "="*60)
        print("Z轴关节灵敏度标定")
        print("="*60)

        # Z轴控制关节 - 所有6个手臂关节
        z_joints = [7, 8, 9, 10, 11, 12]  # joint_1 到 joint_6
        joint_names = {
            7: 'right_arm_joint_1 (底座旋转)',
            8: 'right_arm_joint_2 (肩部俯仰)',
            9: 'right_arm_joint_3 (肩部侧摆)',
            10: 'right_arm_joint_4 (前臂俯仰)',
            11: 'right_arm_joint_5 (腕部俯仰)',
            12: 'right_arm_joint_6 (手腕旋转)'
        }

        print("""
Z轴控制原理:
  - 这些关节的转动会影响末端高度
  - 灵敏度 = 高度变化(mm) / 关节角度变化(deg)
  - 标定后系统可以精确控制Z轴

标定关节:
  1. joint_1 (底座旋转) - 主要影响
  2. joint_2 (肩部俯仰) - 主要影响
  3. joint_3 (肩部侧摆) - 中等影响
  4. joint_4 (前臂俯仰) - 次要影响
  5. joint_5 (腕部俯仰) - 较小影响
  6. joint_6 (手腕旋转) - 较小影响
""")

        input("\n按 Enter 开始...")

        # 创建 Z 轴控制器
        z_ctrl = ZAxisController()

        # 设置相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera1 = self.cameras.get(arm_config.camera_name)
        camera2 = self.cameras.get(arm_config.camera2_name) if arm_config.camera2_name else None

        if camera1 is None:
            print("✗ 主相机未连接")
            return

        z_ctrl.set_cameras(camera1, camera2)

        # 尝试加载已有标定
        z_ctrl.load_calibration()

        try:
            move_deg = float(input("移动角度 (默认4度): ").strip() or "4.0")
        except:
            move_deg = 4.0

        success_count = 0

        for joint_idx in z_joints:
            print(f"\n{'='*50}")
            print(f"标定关节: {joint_names.get(joint_idx, f'joint_{joint_idx}')}")
            print(f"{'='*50}")

            # 使用视频窗口进行标定
            success, sensitivity = self._calibrate_z_joint_interactive(
                z_ctrl, joint_idx, move_deg
            )

            if success and sensitivity is not None:
                z_ctrl.joint_sensitivities[joint_idx] = sensitivity
                success_count += 1
                print(f"  ✓ 灵敏度: {sensitivity.mm_per_deg:.2f} mm/deg")

        # 保存标定数据
        if success_count > 0:
            z_ctrl._save_calibration()
            print(f"\n{'='*60}")
            print(f"✓ Z轴标定完成: {success_count}/{len(z_joints)} 个关节成功")
            print(f"{'='*60}")
        else:
            print("\n✗ Z轴标定失败")

    def _calibrate_z_joint_interactive(self, z_ctrl, joint_idx, move_deg):
        """交互式Z轴关节标定（带视频显示）"""
        joint_name = z_ctrl.JOINT_NAMES.get(joint_idx, f'joint_{joint_idx}')

        print(f"\n[视频窗口] 按 Enter 采集图像，按 q 取消")

        window_name = f"Z-Axis Calibration: {joint_name}"
        phase = 1  # 1=采集初始, 2=采集移动后
        z_before = None
        z_after = None
        angle_before = None
        angle_after = None

        while True:
            # 读取图像
            frame = z_ctrl.camera1.read()
            if frame is None:
                continue

            # 估计深度
            estimate = z_ctrl.estimate_z(frame, z_ctrl.camera2.read() if z_ctrl.camera2 else None)

            # 可视化
            vis = frame.copy()
            cv2.rectangle(vis, (5, 5), (300, 100), (0, 0, 0), -1)

            y = 25
            cv2.putText(vis, f"Z-Axis: {joint_name}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25
            depth_color = (0, 255, 0) if estimate.confidence > 0.5 else (0, 165, 255)
            cv2.putText(vis, f"Depth: {estimate.z:.1f}mm +/- {estimate.uncertainty:.1f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, depth_color, 2)
            y += 25
            cv2.putText(vis, f"Method: {estimate.method}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 获取关节角度
            current_joints = self.controller.get_joint_states()
            current_angle = current_joints[joint_idx] if current_joints is not None else 0.0

            # 底部提示
            bottom_h = 70 if phase == 2 else 50
            cv2.rectangle(vis, (5, vis.shape[0] - bottom_h - 5), (vis.shape[1] - 5, vis.shape[0] - 5), (0, 0, 0), -1)

            if phase == 1:
                cv2.putText(vis, "[Phase 1] Press ENTER to capture initial depth", (15, vis.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis, "Press 'q' to cancel", (15, vis.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            else:
                moved = current_angle - angle_before if angle_before else 0
                status = "OK" if abs(moved) >= move_deg * 0.7 else "Move more"
                cv2.putText(vis, f"Moved: {moved:.2f} deg [{status}]", (15, vis.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if status == "OK" else (0, 165, 255), 2)
                cv2.putText(vis, "[Phase 2] Press ENTER to capture final depth", (15, vis.shape[0] - 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(vis, "Press 'q' to cancel", (15, vis.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(window_name)
                print("  ✗ 标定已取消")
                return False, None

            elif key == 13 or key == 10:  # Enter
                if phase == 1:
                    if estimate.confidence < 0.3:
                        print(f"  ✗ 深度估计不可靠 (置信度: {estimate.confidence:.2f})")
                        continue

                    z_before = estimate.z
                    angle_before = current_angle
                    print(f"  ✓ 初始深度: {z_before:.1f}mm, 角度: {angle_before:.2f}°")
                    phase = 2
                    print(f"\n  请用示教器移动关节约 {move_deg}°")
                else:
                    if estimate.confidence < 0.3:
                        print(f"  ✗ 深度估计不可靠 (置信度: {estimate.confidence:.2f})")
                        continue

                    z_after = estimate.z
                    angle_after = current_angle
                    print(f"  ✓ 移动后深度: {z_after:.1f}mm, 角度: {angle_after:.2f}°")
                    break

        cv2.destroyWindow(window_name)

        # 计算灵敏度
        delta_angle = angle_after - angle_before
        delta_z = z_after - z_before

        if abs(delta_angle) < 0.5:
            print("  ⚠ 角度变化太小，标定可能不准确")
            return False, None

        mm_per_deg = delta_z / delta_angle

        from precision_place.z_axis_controller import ZJointSensitivity
        sensitivity = ZJointSensitivity(
            joint_idx=joint_idx,
            joint_name=joint_name,
            mm_per_deg=mm_per_deg,
            calibration_height=z_before
        )

        print(f"\n  标定结果:")
        print(f"    角度变化: {delta_angle:.2f}°")
        print(f"    Z变化: {delta_z:.1f}mm")
        print(f"    灵敏度: {mm_per_deg:.2f} mm/deg")

        return True, sensitivity

    def calibrate_z_axis_joints_auto(self):
        """Z轴关节灵敏度自动标定（独立模式）"""
        if not self.controller:
            print("请先连接设备")
            return

        if self.passive_mode:
            print("\n✗ 自动标定需要独立模式（不能使用被动模式）")
            print("  请选择 '独立模式' 连接设备")
            return

        if not _has_z_controller:
            print("✗ Z轴控制器未加载")
            return

        if self.robot is None:
            print("✗ 需要机器人连接")
            return

        print("\n" + "="*60)
        print("Z轴关节灵敏度自动标定")
        print("="*60)

        print("""
说明:
  自动标定会自动移动每个Z轴关节并记录深度变化。
  不需要手动操作。

Z轴控制关节 (全部6个):
  - joint_1 (底座旋转): 主要影响
  - joint_2 (肩部俯仰): 主要影响
  - joint_3 (肩部侧摆): 中等影响
  - joint_4 (前臂俯仰): 次要影响
  - joint_5 (腕部俯仰): 较小影响
  - joint_6 (手腕旋转): 较小影响
""")

        input("\n按 Enter 开始自动标定...")

        try:
            move_deg = float(input("移动角度 (默认4度): ").strip() or "4.0")
        except:
            move_deg = 4.0

        # 创建 Z 轴控制器
        z_ctrl = ZAxisController()

        # 设置相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera1 = self.cameras.get(arm_config.camera_name)
        camera2 = self.cameras.get(arm_config.camera2_name) if arm_config.camera2_name else None

        if camera1 is None:
            print("✗ 主相机未连接")
            return

        z_ctrl.set_cameras(camera1, camera2)
        z_ctrl.load_calibration()

        # 执行自动标定
        success = z_ctrl.calibrate_all_z_joints_auto(self.robot, move_deg)

        if success:
            print("\n✓ Z轴自动标定完成")
        else:
            print("\n✗ Z轴自动标定失败")

    def calibrate_stereo_baseline(self):
        """双相机基线标定"""
        if not self.controller:
            print("请先连接设备")
            return

        if not _has_z_controller:
            print("✗ Z轴控制器未加载")
            return

        print("\n" + "="*60)
        print("双相机基线标定")
        print("="*60)

        # 检查副相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera1 = self.cameras.get(arm_config.camera_name)
        camera2 = self.cameras.get(arm_config.camera2_name) if arm_config.camera2_name else None

        if camera1 is None:
            print("✗ 主相机未连接")
            return

        if camera2 is None:
            print("✗ 副相机未连接")
            print(f"  当前配置: 主相机={arm_config.camera_name}, 副相机={arm_config.camera2_name}")
            return

        print(f"""
双目基线标定用于提高深度估计精度。

当前配置:
  主相机: {arm_config.camera_name}
  副相机: {arm_config.camera2_name}

标定方法:
  1. 自动标定 - 机器人移动已知距离
  2. 手动标定 - 输入已知深度
  3. 单目辅助 - 使用单目深度估计
""")

        method = input("选择方法 (1/2/3): ").strip()

        z_ctrl = ZAxisController()
        z_ctrl.set_cameras(camera1, camera2)
        z_ctrl.load_calibration()

        if method == "1":
            if self.passive_mode:
                print("\n✗ 自动标定需要独立模式")
                return
            if self.robot is None:
                print("\n✗ 需要机器人连接")
                return

            try:
                move_dist = float(input("移动距离mm (默认20): ").strip() or "20")
            except:
                move_dist = 20.0

            success, baseline = z_ctrl.calibrate_stereo_baseline_auto(self.robot, move_dist)

        elif method == "2":
            try:
                known_depth = float(input("已知深度mm (默认100): ").strip() or "100")
            except:
                known_depth = 100.0

            success, baseline = z_ctrl.calibrate_stereo_baseline_manual(known_depth)

        elif method == "3":
            success, baseline = z_ctrl.calibrate_stereo_baseline_with_depth()

        else:
            print("无效选项")
            return

        if success:
            print(f"\n✓ 基线标定完成: {baseline:.1f}mm")
        else:
            print("\n✗ 基线标定失败")
    
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

    def set_target_offset(self):
        """设置对齐目标偏移"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*50)
        print("设置对齐目标偏移")
        print("="*50)
        print("""
说明：
  对齐目标是让工件正确放入卡槽。
  由于腕部相机约45度倾斜，高度变化会产生透视效应。
  设置偏移量和对齐必须在相同高度进行！

正确操作步骤：
  1. 工件被夹爪夹住
  2. 手动将工件放入卡槽正确位置
  3. 抬起到对齐高度（约200mm，与对齐时相同高度）
  4. 确认检测正常后设置偏移量

注意：对齐时将自动恢复到此高度！
""")

        input("\n准备好后按 Enter (工件在对齐高度，位置正确)...")

        # 获取当前偏移
        offset_x, offset_y = self.controller.get_current_offset()

        if offset_x == 0 and offset_y == 0:
            print("\n⚠ 无法获取标点偏移，请确认：")
            print("  1. 工件和卡槽都能被相机看到")
            print("  2. 标点颜色正确（绿色=工件，红色=卡槽）")
            return

        print(f"\n当前偏移: ({offset_x:.1f}, {offset_y:.1f}) 像素")
        print(f"说明: 卡槽中心相对于工件中心的偏移")

        confirm = input("\n确认设置此为目标偏移? (y/n): ").strip().lower()
        if confirm == 'y':
            self.controller.set_target_offset(offset_x, offset_y)
        else:
            print("已取消")

    # ----------------- 测试 -----------------

    def test_detection(self):
        """测试检测"""
        if not self.controller:
            print("请先连接设备")
            return

        self.controller.test_detection()

    def set_marker_area_range(self):
        """设置标记检测面积范围"""
        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*50)
        print("设置标记检测面积范围")
        print("="*50)

        print("""
当标记太大或太小检测不到时，需要调整面积范围。

当前标记检测设置:
  - 面积太小会被忽略（噪声）
  - 面积太大也会被忽略（可能是其他物体）

常见问题:
  - 标记太大检测不到: 增大最大面积
  - 标记太小检测不到: 减小最小面积
""")

        print("设置方法:")
        print("  1. 根据标记尺寸自动计算")
        print("  2. 手动输入面积范围")

        choice = input("选择: ").strip()

        if choice == "1":
            try:
                diameter = float(input("标记直径: ").strip() or "20")
                distance = float(input("预期距离: ").strip() or "50")

                self.controller.detector.auto_adjust_area_range(diameter, distance)
            except:
                print("输入无效")

        elif choice == "2":
            try:
                min_area = int(input("最小面积 (默认100): ").strip() or "100")
                max_area = int(input("最大面积 (默认50000): ").strip() or "50000")

                self.controller.detector.set_area_range(min_area, max_area)
            except:
                print("输入无效")
        else:
            print("已取消")

    def configure_alignment_settings(self):
        """配置对齐参数"""
        print("\n" + "="*50)
        print("对齐参数设置")
        print("="*50)

        while True:
            # 显示当前设置
            show_video = getattr(self.controller, 'show_alignment_video', True) if self.controller else True
            gain = getattr(self.controller, 'gain', 0.6) if self.controller else 0.6
            max_iter = getattr(self.controller, 'max_iterations', 15) if self.controller else 15
            tolerance = getattr(self.controller, 'tolerance_mm', 2.0) if self.controller else 2.0

            print(f"\n当前设置:")
            print(f"  1. 视频显示: {'开启' if show_video else '关闭'}")
            print(f"  2. 控制增益: {gain}")
            print(f"  3. 最大迭代次数: {max_iter}")
            print(f"  4. 对齐精度: {tolerance}mm")
            print("  0. 返回")

            choice = input("\n选项: ").strip()

            if choice == "1":
                if self.controller:
                    self.controller.show_alignment_video = not self.controller.show_alignment_video
                    print(f"视频显示已{'开启' if self.controller.show_alignment_video else '关闭'}")
                else:
                    print("请先连接设备")
            elif choice == "2":
                if self.controller:
                    try:
                        new_gain = float(input(f"输入增益 (当前{gain}, 建议0.3-0.8): ").strip())
                        if 0.1 <= new_gain <= 1.0:
                            self.controller.gain = new_gain
                            print(f"增益已设置为 {new_gain}")
                        else:
                            print("增益应在 0.1-1.0 范围内")
                    except:
                        print("输入无效")
                else:
                    print("请先连接设备")
            elif choice == "3":
                if self.controller:
                    try:
                        new_iter = int(input(f"输入最大迭代次数 (当前{max_iter}): ").strip())
                        if 5 <= new_iter <= 50:
                            self.controller.max_iterations = new_iter
                            print(f"最大迭代次数已设置为 {new_iter}")
                        else:
                            print("迭代次数应在 5-50 范围内")
                    except:
                        print("输入无效")
                else:
                    print("请先连接设备")
            elif choice == "4":
                if self.controller:
                    try:
                        new_tol = float(input(f"输入对齐精度mm (当前{tolerance}mm): ").strip())
                        if 0.5 <= new_tol <= 10.0:
                            self.controller.tolerance_mm = new_tol
                            print(f"对齐精度已设置为 {new_tol}mm")
                        else:
                            print("精度应在 0.5-10.0mm 范围内")
                    except:
                        print("输入无效")
                else:
                    print("请先连接设备")
            elif choice == "0":
                break
            else:
                print("无效选项")

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
            print("7. 设置对齐目标偏移")
            print("8. 运行对齐")
            print("9. 完整流程")
            print("10. 连续运行 (10次)")
            print("11. 设置标记面积范围")
            print("12. 设置对齐参数 (视频显示等)")
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
                print("  2. 关节灵敏度标定 (手动移动)")
                print("  3. 关节灵敏度标定 (自动移动，推荐)")
                print("  4. 运行全部标定 (手动)")
                print("  5. 运行全部标定 (自动)")
                print("  ---")
                print("  6. Z轴关节灵敏度标定 (手动)")
                print("  7. Z轴关节灵敏度标定 (自动)")
                print("  8. 双相机基线标定")
                print("  ---")
                print("  V. 标定验证 (XY灵敏度方向)")
                print("  Z. Z轴标定验证 (深度估计稳定性)")
                print("  C. 标定完整性检查")
                print("  0. 修复灵敏度方向 (翻转X/Y)")

                calib_choice = input("选项: ").strip().upper()

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
                    system.calibrate_joints_auto()
                elif calib_choice == "4":
                    if not system.controller:
                        system.connect()
                    print("\n[1/2] 像素-毫米标定")
                    system.calibrate()
                    print("\n[2/2] 关节灵敏度标定 (手动)")
                    system.calibrate_joints()
                elif calib_choice == "5":
                    if not system.controller:
                        system.connect()
                    print("\n[1/2] 像素-毫米标定")
                    system.calibrate()
                    print("\n[2/2] 关节灵敏度标定 (自动)")
                    system.calibrate_joints_auto()
                elif calib_choice == "6":
                    if not system.controller:
                        system.connect()
                    system.calibrate_z_axis_joints()
                elif calib_choice == "7":
                    if not system.controller:
                        system.connect()
                    system.calibrate_z_axis_joints_auto()
                elif calib_choice == "8":
                    if not system.controller:
                        system.connect()
                    system.calibrate_stereo_baseline()
                elif calib_choice == "V":
                    # XY标定验证
                    if not system.controller:
                        system.connect()
                    print("\nXY标定验证选项:")
                    print("  1. 验证所有已标定关节")
                    print("  2. 验证单个关节")
                    v_choice = input("选项: ").strip()
                    if v_choice == "2":
                        try:
                            joint_idx = int(input("输入关节索引: ").strip())
                            system.controller.verify_xy_calibration(joint_idx=joint_idx)
                        except:
                            print("输入无效")
                    else:
                        system.controller.verify_xy_calibration()
                elif calib_choice == "Z":
                    # Z轴标定验证
                    if not system.controller:
                        system.connect()
                    system.controller.verify_z_calibration()
                elif calib_choice == "C":
                    # 标定完整性检查
                    if not system.controller:
                        system.connect()
                    system.controller.verify_calibration_completeness()
                elif calib_choice == "0":
                    if not system.controller:
                        system.connect()
                    system.flip_sensitivity_direction()
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
                    print("请先连接设备")
                else:
                    system.set_target_offset()

            elif choice == "8":
                if not system.controller:
                    system.connect()
                system.run_alignment()

            elif choice == "9":
                if not system.controller:
                    system.connect()
                system.run_full()

            elif choice == "10":
                if not system.controller:
                    system.connect()
                system.continuous_run(10)

            elif choice == "11":
                if not system.controller:
                    system.connect()
                system.set_marker_area_range()

            elif choice == "12":
                system.configure_alignment_settings()

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
