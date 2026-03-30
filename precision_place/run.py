#!/usr/bin/env python3
"""
精准放置系统 - 主启动脚本 (V4 重构版)

功能:
1. 手眼标定 (ChArUco板 + Tsai-Lenz算法)
2. 双标记点对齐 (工件3绿 + 卡槽3红)
3. XY对齐 (外参矩阵精确变换)
4. Z轴精确控制 (双目立体视觉)
5. 预设位置管理
6. 标定历史

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

# 新模块结构
from precision_place.models.calibration_data import ARM_CONFIGS, JointSensitivity
from precision_place.models.marker import DualMarkerState
from precision_place.core.detector import DualPointDetector

# 向后兼容：旧控制器
from precision_place.dual_point_alignment import PrecisionPlaceController

# 尝试导入Z轴控制器
try:
    from precision_place.z_axis_controller import ZAxisController
    _has_z_controller = True
except ImportError:
    _has_z_controller = False
    ZAxisController = None

# 尝试导入手眼标定模块
try:
    from precision_place.calibration.hand_eye import HandEyeCalibrator
    from precision_place.models.state import CalibrationResult
    _has_hand_eye = True
except ImportError:
    _has_hand_eye = False
    HandEyeCalibrator = None

# 尝试导入正运动学模块
try:
    from precision_place.calibration.forward_kinematics import ForwardKinematics, create_fk_from_urdf
    _has_fk = True
except ImportError:
    _has_fk = False
    ForwardKinematics = None

# 尝试导入坐标变换模块
try:
    from precision_place.calibration.coordinate_transform import CoordinateTransformer
    from precision_place.core.aligner import HandEyeAligner
    _has_coord_transform = True
except ImportError:
    _has_coord_transform = False
    CoordinateTransformer = None

# 尝试导入TCP标定模块
try:
    from precision_place.calibration.tcp_calibrator import TCPCalibrator, TCPCalibrationResult
    _has_tcp_calibrator = True
except ImportError:
    _has_tcp_calibrator = False
    TCPCalibrator = None

# 尝试导入同步捕获模块
try:
    from precision_place.calibration.sync_capture import (
        SynchronizedCapture, ContinuousCapture, CaptureResult
    )
    _has_sync_capture = True
except ImportError:
    _has_sync_capture = False
    SynchronizedCapture = None

# 尝试导入IBVS模块
try:
    from precision_place.calibration.ibvs_controller import VirtualIBVSController, IBVSAlignmentRunner
    _has_ibvs = True
except ImportError:
    _has_ibvs = False
    VirtualIBVSController = None


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
        self.hand_eye_calibrator = None  # 手眼标定器
        self.tcp_calibrator = None  # TCP标定器
        self.forward_kinematics = None  # 正运动学计算器
        self.coordinate_transformer = None  # 坐标变换器（基于手眼标定）
        self.ibvs_controller = None  # IBVS控制器
        self.urdf_path = None  # URDF文件路径
        # 目标偏移量（用于标记点有固定偏移的情况）
        self.target_offset_x = 0.0  # 像素
        self.target_offset_y = 0.0  # 像素

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

    def hand_eye_calibration(self):
        """手眼标定 (Eye-in-Hand)"""
        if not _has_hand_eye:
            print("✗ 手眼标定模块未加载")
            return

        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*60)
        print("手眼标定 (Eye-in-Hand)")
        print("="*60)
        print("""
原理：
  通过在不同姿态下观察固定标定板，计算相机相对于法兰的外参矩阵。
  外参矩阵包含了相机的精确位置和旋转角度，可从根本上解决透视补偿问题。

要求：
  1. ChArUco标定板（打印后固定在桌面上，绝对不能动！）
  2. 至少采集10个不同姿态（越多越好）
  3. 姿态差异越大越好（大角度倾斜）

操作步骤：
  1. 固定ChArUco标定板在工作台上
  2. 移动机械臂到标定板上方
  3. 确保相机能看到完整的标定板
  4. 按 'C' 键捕获当前姿态
  5. 换一个不同姿态，重复步骤4
  6. 采集足够后按 'S' 键开始标定
  7. 按 'Q' 键退出
""")

        # 检查/初始化正运动学
        if self.forward_kinematics is None:
            if self.urdf_path is None:
                print("\n需要URDF文件来计算正运动学。")
                urdf_input = input("请输入URDF文件路径 (或按Enter跳过，使用简化模式): ").strip()
                if urdf_input:
                    self.urdf_path = urdf_input

            if self.urdf_path and _has_fk:
                try:
                    self.forward_kinematics = create_fk_from_urdf(self.urdf_path, self.current_arm)
                    print(f"✓ 正运动学已初始化")
                except Exception as e:
                    print(f"⚠ 正运动学初始化失败: {e}")
                    print("  将使用简化模式（精度较低）")
                    self.forward_kinematics = None

        # 获取主相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera = self.cameras.get(arm_config.camera_name)

        if camera is None:
            print("✗ 主相机未连接")
            return

        # 相机内参（需要预先标定或使用默认值）
        # TODO: 从配置文件加载真实的相机内参
        image_width = 640
        image_height = 480
        fx = 500.0  # 焦距x (像素)
        fy = 500.0  # 焦距y (像素)
        cx = image_width / 2  # 主点x
        cy = image_height / 2  # 主点y

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(5)  # 假设无畸变

        # 创建标定器
        self.hand_eye_calibrator = HandEyeCalibrator(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs
        )

        # 设置调试目录
        debug_dir = Path(__file__).parent / "calibration_debug"
        self.hand_eye_calibrator.set_debug_dir(str(debug_dir))

        # 显示正运动学状态
        if self.forward_kinematics:
            print("\n✓ 正运动学已启用")
        else:
            print("\n⚠ 正运动学未启用，将使用简化模式")
            print("  提示: 提供URDF文件可获得更高精度")

        # 创建同步捕获器
        if _has_sync_capture:
            sync_capture = SynchronizedCapture(
                camera=camera,
                controller=self.controller,
                forward_kinematics=self.forward_kinematics,
                warmup_frames=3,
                max_sync_delay_ms=50.0
            )
            print("✓ 同步捕获已启用")
        else:
            sync_capture = None
            print("⚠ 同步捕获模块未加载，使用传统模式")

        print("\n开始采集数据...")
        print("按键: [C]捕获  [S]标定  [Q]退出")

        cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand-Eye Calibration", 800, 600)

        # 同步状态显示
        sync_delay_display = 0.0

        while True:
            # 获取图像
            image = camera.read()
            if image is None:
                continue

            display = image.copy()

            # 尝试检测ChArUco板
            success, rvec, tvec, corners = self.hand_eye_calibrator.detect_charuco(image)

            if success:
                # 绘制坐标轴
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                cv2.putText(display, "Board Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 显示采集数量
            count = self.hand_eye_calibrator.get_capture_count()
            cv2.putText(display, f"Captured: {count}/30", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "[C]apture [S]olve [Q]uit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示正运动学状态
            fk_status = "FK: ON" if self.forward_kinematics else "FK: OFF"
            cv2.putText(display, fk_status, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.forward_kinematics else (0, 0, 255), 1)

            # 显示同步延迟
            if sync_capture:
                sync_color = (0, 255, 0) if sync_delay_display < 30 else (0, 165, 255)
                cv2.putText(display, f"Sync: {sync_delay_display:.1f}ms", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_color, 1)

            cv2.imshow("Hand-Eye Calibration", display)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('c') or key == ord('C'):
                # 捕获当前姿态
                if not success:
                    print("  ⚠ 未检测到标定板，无法捕获")
                    continue

                # 使用同步捕获
                if sync_capture:
                    # 同步捕获图像和关节状态
                    capture_result = sync_capture.capture_with_verification()
                    if not capture_result.success:
                        print(f"  ✗ 同步捕获失败: {capture_result.error_message}")
                        continue

                    image = capture_result.image
                    joints = capture_result.joints
                    flange_position = capture_result.flange_position
                    flange_rotation = capture_result.flange_rotation
                    sync_delay_display = capture_result.sync_delay_ms

                    print(f"  同步延迟: {sync_delay_display:.1f}ms")

                    # 如果没有正运动学，使用默认值
                    if flange_position is None:
                        if self.forward_kinematics:
                            try:
                                pose = self.forward_kinematics.compute(joints)
                                flange_position = pose.get_position()
                                flange_rotation = pose.quaternion
                            except Exception as e:
                                print(f"  ⚠ 正运动学计算失败: {e}")
                                continue
                        else:
                            print("  ⚠ 使用简化模式，精度较低")
                            flange_position = np.array([0.0, 0.0, 0.5])
                            flange_rotation = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    # 传统模式
                    joints = self.controller.get_joint_states()
                    if joints is None:
                        print("  ⚠ 无法获取关节状态")
                        continue

                    if self.forward_kinematics:
                        try:
                            pose = self.forward_kinematics.compute(joints)
                            flange_position = pose.get_position()
                            flange_rotation = pose.quaternion
                            print(f"  法兰位姿: pos=({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})m")
                        except Exception as e:
                            print(f"  ⚠ 正运动学计算失败: {e}")
                            continue
                    else:
                        print("  ⚠ 使用简化模式，精度较低")
                        flange_position = np.array([0.0, 0.0, 0.5])
                        flange_rotation = np.array([0.0, 0.0, 0.0, 1.0])

                if self.hand_eye_calibrator.capture_pose(image, flange_position, flange_rotation):
                    print(f"  ✓ 捕获成功 ({count + 1})")
                else:
                    print("  ✗ 捕获失败")

            elif key == ord('s') or key == ord('S'):
                # 开始标定
                count = self.hand_eye_calibrator.get_capture_count()
                if count < 10:
                    print(f"  ⚠ 采集数量不足 ({count}/10)")
                    continue

                success, result = self.hand_eye_calibrator.calibrate()

                if success and result.valid:
                    # 保存结果
                    output_path = Path(__file__).parent / "hand_eye_extrinsic.yaml"
                    self.hand_eye_calibrator.save(str(output_path))
                    print(f"\n✓ 标定成功！结果已保存")
                    break
                else:
                    print(f"\n✗ 标定失败或精度不足，请重新采集")

            elif key == ord('q') or key == ord('Q'):
                print("退出手眼标定")
                break

        cv2.destroyWindow("Hand-Eye Calibration")

    def reprojection_verification(self):
        """重投影验证 - 验证手眼标定精度"""
        if not _has_hand_eye:
            print("✗ 手眼标定模块未加载")
            return

        print("\n" + "="*60)
        print("重投影验证")
        print("="*60)
        print("""
原理：
  使用已标定的外参矩阵，验证相机到法兰的变换是否正确。

步骤：
  1. 加载手眼标定结果
  2. 在不同姿态下拍摄图像
  3. 检测图像中的特征点
  4. 用外参矩阵反推特征点在世界坐标系的位置
  5. 计算重投影误差 (RMSE)

验收标准：
  - RMSE < 1.5像素 = 标定正确
  - RMSE > 1.5像素 = 需要重新标定
""")

        # 加载标定结果
        extrinsic_path = Path(__file__).parent / "hand_eye_extrinsic.yaml"
        if not extrinsic_path.exists():
            print(f"✗ 未找到手眼标定结果: {extrinsic_path}")
            print("  请先运行手眼标定 (选项 H)")
            return

        result = HandEyeCalibrator.load(str(extrinsic_path))
        if result is None or not result.valid:
            print("✗ 标定结果无效")
            return

        print(f"✓ 已加载标定结果 (RMSE: {result.rmse_error:.2f}像素)")

        if not self.controller:
            print("请先连接设备")
            return

        # 检查/初始化正运动学（方法2需要）
        if self.forward_kinematics is None:
            if self.urdf_path is None:
                print("\n重投影验证需要URDF文件来计算正运动学（方法2需要）。")
                urdf_input = input("请输入URDF文件路径 (或按Enter跳过，只能使用方法1): ").strip()
                if urdf_input:
                    self.urdf_path = urdf_input

            if self.urdf_path and _has_fk:
                try:
                    self.forward_kinematics = create_fk_from_urdf(self.urdf_path, self.current_arm)
                    print(f"✓ 正运动学已初始化")
                except Exception as e:
                    print(f"⚠ 正运动学初始化失败: {e}")
                    self.forward_kinematics = None

        # 获取相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera = self.cameras.get(arm_config.camera_name)
        if camera is None:
            print("✗ 主相机未连接")
            return

        # 相机内参
        camera_matrix = np.array([
            [500.0, 0, 320.0],
            [0, 500.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros(5)

        from precision_place.calibration.hand_eye import ReprojectionVerifier
        verifier = ReprojectionVerifier(camera_matrix, dist_coeffs, result)

        print("\n验证方法：")
        print("  方法1: 使用TCP探针戳验证点（最准确）")
        print("  方法2: 使用固定标记点（简化）")
        method = input("选择方法 (1/2): ").strip()

        if method == "1":
            # 加载TCP标定结果
            tcp_result = None
            tcp_path = Path(__file__).parent / "tcp_offset.yaml"
            if tcp_path.exists():
                from precision_place.calibration.tcp_calibrator import TCPCalibrator
                tcp_result = TCPCalibrator.load(str(tcp_path))
                if tcp_result and tcp_result.valid:
                    print(f"✓ 已加载TCP标定结果 (RMSE: {tcp_result.rmse_mm:.2f}mm)")
                    print(f"  TCP偏移: ({tcp_result.offset_x*1000:.1f}, {tcp_result.offset_y*1000:.1f}, {tcp_result.offset_z*1000:.1f}) mm")
                else:
                    print("⚠ TCP标定结果无效")
                    tcp_result = None
            else:
                print("⚠ 未找到TCP标定结果，请先运行TCP标定 (选项 T)")

            # 创建ChArUco检测器用于检测角点
            charuco_calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)

            print("\n" + "="*60)
            print("ChArUco角点验证 (推荐)")
            print("="*60)
            print("""
操作流程:
  1. 相机拍摄 → 显示ChArUco角点编号
  2. 鼠标点击要验证的角点 → 系统记录像素坐标
  3. 探针戳该角点 → 按 'P' 记录世界坐标
  4. 重复采集4+个角点 → 按 'V' 验证

优势:
  - 角点像素位置自动检测（精确）
  - 点击确认明确角点编号
  - 一个标定板可验证多个角点
""")

            verification_points = []  # [{'corner_id': int, 'pixel_pos': tuple, 'world_pos': array}, ...]
            current_corner_id = None  # 当前选中的角点编号

            cv2.namedWindow("ChArUco Verification", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ChArUco Verification", 800, 600)

            print("\n按键: [鼠标点击]选择角点 [P]记录探针位置 [V]验证 [Q]退出")
            print("提示: 先点击角点确认编号，再用探针戳该角点并按P")

            # 鼠标回调
            clicked_corner = {'id': None, 'pixel': None}

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked_corner['pixel'] = (x, y)
                    clicked_corner['id'] = None  # 需要在主循环中查找最近角点

            cv2.setMouseCallback("ChArUco Verification", mouse_callback)

            while True:
                image = camera.read()
                if image is None:
                    continue

                display = image.copy()

                # 检测ChArUco角点
                success, rvec, tvec, charuco_corners = charuco_calibrator.detect_charuco(image)

                detected_corners = []
                if success and charuco_corners is not None:
                    # 显示角点编号
                    for i, corner in enumerate(charuco_corners):
                        corner_pos = corner.flatten()
                        detected_corners.append({'id': i, 'pixel': corner_pos})

                        # 绘制角点
                        cv2.circle(display, (int(corner_pos[0]), int(corner_pos[1])), 5, (0, 255, 0), -1)
                        cv2.putText(display, str(i), (int(corner_pos[0]) + 5, int(corner_pos[1]) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # 高亮已验证的角点
                    for pt in verification_points:
                        corner_id = pt['corner_id']
                        if corner_id < len(detected_corners):
                            corner_pos = detected_corners[corner_id]['pixel']
                            if pt.get('world_pos') is not None:
                                cv2.circle(display, (int(corner_pos[0]), int(corner_pos[1])), 8, (0, 0, 255), 2)
                                cv2.putText(display, f"#{corner_id} OK", (int(corner_pos[0]) + 10, int(corner_pos[1])),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # 高亮当前选中的角点
                    if current_corner_id is not None and current_corner_id < len(detected_corners):
                        corner_pos = detected_corners[current_corner_id]['pixel']
                        cv2.circle(display, (int(corner_pos[0]), int(corner_pos[1])), 12, (255, 0, 0), 3)
                        cv2.putText(display, f"#{current_corner_id} SELECTED", (int(corner_pos[0]) - 20, int(corner_pos[1]) - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # 处理鼠标点击
                if clicked_corner['pixel'] is not None:
                    click_x, click_y = clicked_corner['pixel']
                    # 找最近的角点
                    min_dist = float('inf')
                    nearest_id = None
                    for corner in detected_corners:
                        cx, cy = corner['pixel']
                        dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
                        if dist < min_dist and dist < 30:  # 30像素阈值
                            min_dist = dist
                            nearest_id = corner['id']

                    if nearest_id is not None:
                        current_corner_id = nearest_id
                        current_pixel = detected_corners[nearest_id]['pixel']
                        print(f"\n  ✓ 选择角点 #{nearest_id}")
                        print(f"    像素位置 = ({current_pixel[0]:.1f}, {current_pixel[1]:.1f})")
                        print("    请用探针戳该角点，然后按 'P' 记录世界坐标")
                    else:
                        print("  ⚠ 未检测到角点或点击位置不接近任何角点")

                    clicked_corner['pixel'] = None  # 重置

                # 显示状态信息
                cv2.putText(display, f"Verified: {len(verification_points)}/4+", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, "[Click] select corner [P] probe [V] verify [Q] quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if tcp_result:
                    cv2.putText(display, "TCP: OK", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(display, "TCP: N/A", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

                if current_corner_id is not None:
                    cv2.putText(display, f"Corner #{current_corner_id} selected", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                else:
                    cv2.putText(display, "Click a corner to select", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("ChArUco Verification", display)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('p') or key == ord('P'):
                    # 记录探针位置（世界坐标）
                    if current_corner_id is None:
                        print("  ⚠ 请先点击选择要验证的角点")
                        continue

                    # 检查是否已验证该角点
                    existing = [pt for pt in verification_points if pt['corner_id'] == current_corner_id]
                    if existing:
                        print(f"  ⚠ 角点 #{current_corner_id} 已验证，请选择其他角点")
                        continue

                    joints = self.controller.get_joint_states()
                    if joints is None:
                        print("  ✗ 无法获取关节状态")
                        continue

                    if self.forward_kinematics:
                        pose = self.forward_kinematics.compute(joints)
                        flange_pos = pose.get_position()
                        flange_rot = pose.rotation_matrix

                        # 计算TCP位置
                        if tcp_result:
                            tcp_offset = np.array([
                                tcp_result.offset_x,
                                tcp_result.offset_y,
                                tcp_result.offset_z
                            ])
                            world_pos = flange_pos + flange_rot @ tcp_offset
                        else:
                            world_pos = flange_pos

                        # 获取当前角点的像素位置
                        if current_corner_id < len(detected_corners):
                            pixel_pos = detected_corners[current_corner_id]['pixel']
                        else:
                            print("  ✗ 角点检测丢失，请重新选择")
                            continue

                        # 添加验证点
                        verification_points.append({
                            'corner_id': current_corner_id,
                            'pixel_pos': pixel_pos,
                            'world_pos': world_pos
                        })

                        print(f"  ✓ 角点 #{current_corner_id} 验证数据已记录:")
                        print(f"    像素位置 = ({pixel_pos[0]:.1f}, {pixel_pos[1]:.1f})")
                        print(f"    世界坐标 = ({world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}) m")
                        print(f"    已采集: {len(verification_points)}/4+")

                        # 清除当前选中，准备下一个
                        current_corner_id = None
                    else:
                        print("  ✗ 正运动学未启用，无法记录TCP位置")

                elif key == ord('v') or key == ord('V'):
                    # 执行验证
                    if len(verification_points) < 4:
                        print(f"  ⚠ 验证点不足 ({len(verification_points)}/4)")
                        continue

                    print("\n执行重投影验证...")

                    # 添加验证点
                    for pt in verification_points:
                        verifier.add_verification_point(pt['world_pos'], pt['pixel_pos'])

                    # 获取当前法兰位姿
                    joints = self.controller.get_joint_states()
                    if joints is None:
                        print("✗ 无法获取关节状态")
                        continue

                    if self.forward_kinematics:
                        pose = self.forward_kinematics.compute(joints)
                        flange_pos = pose.get_position()
                        flange_rot = pose.quaternion
                    else:
                        print("⚠ 正运动学未启用，使用默认姿态")
                        flange_pos = np.array([0.0, 0.0, 0.5])
                        flange_rot = np.array([0.0, 0.0, 0.0, 1.0])

                    passed, rmse = verifier.verify(flange_pos, flange_rot)

                    print("\n" + "="*50)
                    print("验证结果")
                    print("="*50)
                    print(f"验证点数量: {len(verification_points)}")
                    print(f"RMSE误差: {rmse:.2f} 像素")

                    if passed:
                        print("\n✓ 手眼标定验证通过！")
                        print("  可以放心使用该标定结果进行精准放置。")
                    else:
                        print("\n✗ 验证失败！")
                        print("  建议重新进行手眼标定。")

                    cv2.destroyWindow("ChArUco Verification")
                    return

                elif key == ord('q') or key == ord('Q'):
                    print("退出重投影验证")
                    cv2.destroyWindow("ChArUco Verification")
                    return

        elif method == "2":
            print("\n简化验证: 检查ChArUco板在世界坐标系的位置一致性")
            print("  移动机械臂到不同姿态，标定板在世界坐标系的位置应保持不变")

            if not self.forward_kinematics:
                print("\n✗ 正运动学未初始化")
                if self.urdf_path:
                    print("  URDF加载失败，请检查文件路径和格式")
                else:
                    print("  请重新运行并输入有效的URDF路径")
                print("  或使用方法1（TCP探针验证）")
                return

            calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)
            positions_world = []  # 世界坐标系下的标定板位置

            # 获取外参矩阵
            R_flange2cam = result.R_flange2cam
            t_flange2cam = result.t_flange2cam

            print("\n按 'C' 捕获，按 'Q' 完成")
            cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)

            while True:
                image = camera.read()
                if image is None:
                    continue

                display = image.copy()
                success, rvec, tvec, corners = calibrator.detect_charuco(image)

                if success:
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    cv2.putText(display, "Board Detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(display, f"Captures: {len(positions_world)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, "[C]apture [Q]uit", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Verification", display)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('c') or key == ord('C'):
                    if not success:
                        print("  ⚠ 未检测到标定板")
                        continue

                    # 获取当前法兰位姿
                    joints = self.controller.get_joint_states()
                    if joints is None:
                        print("  ⚠ 无法获取关节状态")
                        continue

                    try:
                        pose = self.forward_kinematics.compute(joints)
                        flange_pos = pose.get_position()
                        flange_rot = pose.quaternion
                    except Exception as e:
                        print(f"  ⚠ 正运动学计算失败: {e}")
                        continue

                    # 计算标定板在世界坐标系的位置
                    # P_world = R_flange2world @ (R_cam2flange @ P_cam + t_cam2flange) + t_flange2world
                    from scipy.spatial.transform import Rotation as R

                    # 法兰到世界的变换
                    R_flange2world = R.from_quat(flange_rot).as_matrix()
                    t_flange2world = flange_pos

                    # 相机到法兰的变换（外参）
                    R_cam2flange = R_flange2cam.T  # 逆变换
                    t_cam2flange = -R_flange2cam.T @ t_flange2cam

                    # 标定板在相机坐标系的位置
                    P_cam = tvec.flatten()

                    # 标定板在法兰坐标系的位置
                    P_flange = R_cam2flange @ P_cam + t_cam2flange

                    # 标定板在世界坐标系的位置
                    P_world = R_flange2world @ P_flange + t_flange2world

                    positions_world.append(P_world)
                    print(f"  捕获: 世界坐标 = ({P_world[0]:.4f}, {P_world[1]:.4f}, {P_world[2]:.4f}) 米")

                elif key == ord('q') or key == ord('Q'):
                    break

            cv2.destroyWindow("Verification")

            if len(positions_world) >= 3:
                positions_world = np.array(positions_world)
                mean_pos = np.mean(positions_world, axis=0)
                std_pos = np.std(positions_world, axis=0)

                print(f"\n标定板在世界坐标系的位置统计:")
                print(f"  平均: ({mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}) 米")
                print(f"  标准差: ({std_pos[0]*1000:.1f}, {std_pos[1]*1000:.1f}, {std_pos[2]*1000:.1f}) 毫米")

                max_std_mm = np.max(std_pos) * 1000
                if max_std_mm < 3:  # 3mm
                    print(f"  ✓ 位置一致性良好 (最大标准差: {max_std_mm:.1f}mm)")
                elif max_std_mm < 10:  # 10mm
                    print(f"  ⚠ 位置一致性一般 (最大标准差: {max_std_mm:.1f}mm)")
                else:
                    print(f"  ✗ 位置波动较大 (最大标准差: {max_std_mm:.1f}mm)，建议重新标定")

    def tcp_calibration(self):
        """
        TCP标定 (工具中心点标定)

        使用四点法标定工具中心点相对于法兰的偏移量。
        """
        if not _has_tcp_calibrator:
            print("✗ TCP标定模块未加载")
            return

        if not self.controller:
            print("请先连接设备")
            return

        print("\n" + "="*60)
        print("TCP标定 (四点法)")
        print("="*60)
        print("""
原理：
  假设探针尖端固定在世界坐标系的某一点 P_tip，
  通过采集多个姿态下的法兰位姿，使用最小二乘法求解
  探针相对于法兰的偏置向量 t_offset。

公式：
  P_flange_i + R_i * t_offset = P_tip

操作步骤：
  1. 在工作台上固定一个尖锐靶点（如大头针）
  2. 安装探针到机械臂末端
  3. 用探针尖端精确对准靶点
  4. 按 'C' 键捕获当前姿态（保持针尖对准，变换姿态）
  5. 至少采集4个不同姿态（姿态差异越大越好）
  6. 采集足够后按 'S' 键开始标定
  7. 按 'Q' 键退出

验收标准：
  RMSE < 0.5mm = 合格
""")

        # 检查/初始化正运动学
        if self.forward_kinematics is None:
            if self.urdf_path is None:
                print("\n需要URDF文件来计算正运动学。")
                urdf_input = input("请输入URDF文件路径 (或按Enter跳过): ").strip()
                if urdf_input:
                    self.urdf_path = urdf_input

            if self.urdf_path and _has_fk:
                try:
                    self.forward_kinematics = create_fk_from_urdf(self.urdf_path, self.current_arm)
                    print(f"✓ 正运动学已初始化")
                except Exception as e:
                    print(f"✗ 正运动学初始化失败: {e}")
                    print("  TCP标定需要正运动学支持，无法继续")
                    return
            else:
                print("✗ 缺少正运动学，TCP标定无法进行")
                return

        # 创建TCP标定器
        self.tcp_calibrator = TCPCalibrator()

        # 创建同步捕获器
        if _has_sync_capture:
            # TCP标定不需要相机，但需要同步读取关节状态
            sync_capture = SynchronizedCapture(
                camera=None,  # TCP标定不需要相机
                controller=self.controller,
                forward_kinematics=self.forward_kinematics,
                warmup_frames=2,
                max_sync_delay_ms=30.0
            )
            print("✓ 同步捕获已启用")
        else:
            sync_capture = None

        print("\n准备采集数据...")
        print("按键: [C]捕获  [S]标定  [Q]退出")
        print("\n注意: 每次捕获前，请确保探针尖端精确对准靶点！")

        while True:
            # 显示当前状态
            joints = self.controller.get_joint_states()
            if joints is None:
                print("⚠ 无法获取关节状态，请重试")
                continue

            count = self.tcp_calibrator.get_capture_count()
            print(f"\r当前已捕获: {count}/4+  ", end="", flush=True)

            # 等待用户输入
            key = input("\n按键 [C]捕获 [S]标定 [Q]退出: ").strip().upper()

            if key == 'C':
                # 捕获当前姿态
                if sync_capture:
                    # 同步捕获（预热丢弃旧数据）
                    for _ in range(2):
                        joints = self.controller.get_joint_states()

                    try:
                        # 计算法兰位姿
                        pose = self.forward_kinematics.compute(joints)
                        flange_position = pose.get_position()
                        flange_rotation = pose.quaternion

                        self.tcp_calibrator.capture_pose(flange_position, flange_rotation, "quaternion")
                        count = self.tcp_calibrator.get_capture_count()
                        print(f"  ✓ 捕获成功 (同步模式)")

                        if count >= 4:
                            print(f"  已采集足够数据 ({count}个)，可以按 'S' 开始标定")

                    except Exception as e:
                        print(f"  ✗ 捕获失败: {e}")
                else:
                    try:
                        pose = self.forward_kinematics.compute(joints)
                        flange_position = pose.get_position()
                        flange_rotation = pose.quaternion

                        self.tcp_calibrator.capture_pose(flange_position, flange_rotation, "quaternion")
                        count = self.tcp_calibrator.get_capture_count()

                        if count >= 4:
                            print(f"  ✓ 已采集足够数据 ({count}个)，可以按 'S' 开始标定")

                    except Exception as e:
                        print(f"  ✗ 捕获失败: {e}")

            elif key == 'S':
                # 开始标定
                count = self.tcp_calibrator.get_capture_count()
                if count < 4:
                    print(f"  ⚠ 采集数量不足 ({count}/4)，至少需要4个姿态")
                    continue

                success, result = self.tcp_calibrator.solve()

                if success and result.valid:
                    # 保存结果
                    output_path = Path(__file__).parent / "tcp_offset.yaml"
                    self.tcp_calibrator.save(str(output_path))
                    print(f"\n✓ TCP标定成功！结果已保存到: {output_path}")
                    print(f"  TCP偏移: ({result.offset_x*1000:.2f}, {result.offset_y*1000:.2f}, {result.offset_z*1000:.2f}) mm")
                    print(f"  RMSE误差: {result.rmse_mm:.3f} mm")
                else:
                    print(f"\n✗ TCP标定失败或精度不足")
                    print("  建议: 检查探针是否牢固安装，姿态差异是否足够大")
                    retry = input("  是否重新采集? [Y/N]: ").strip().upper()
                    if retry == 'Y':
                        self.tcp_calibrator.clear_captures()
                        print("已清空采集数据，请重新开始")

            elif key == 'Q':
                print("退出TCP标定")
                break

    def load_coordinate_transformer(self) -> bool:
        """加载坐标变换器（基于手眼标定结果）"""
        if not _has_coord_transform:
            print("✗ 坐标变换模块未加载")
            return False

        extrinsic_path = Path(__file__).parent / "hand_eye_extrinsic.yaml"
        if not extrinsic_path.exists():
            print(f"✗ 未找到手眼标定结果: {extrinsic_path}")
            print("  请先运行手眼标定 (选项 H)")
            return False

        try:
            self.coordinate_transformer = CoordinateTransformer.from_calibration_file(
                str(extrinsic_path)
            )
            return True
        except Exception as e:
            print(f"✗ 加载坐标变换器失败: {e}")
            return False

    def align_with_hand_eye(self):
        """使用手眼标定进行精确对齐"""
        print("\n" + "="*60)
        print("手眼标定对齐模式")
        print("="*60)

        if not self.controller:
            print("请先连接设备")
            return

        # 加载坐标变换器
        if self.coordinate_transformer is None:
            if not self.load_coordinate_transformer():
                return

        # 检查正运动学
        if self.forward_kinematics is None:
            print("\n需要正运动学来获取TCP位姿。")
            urdf_input = input("请输入URDF文件路径 (或按Enter退出): ").strip()
            if not urdf_input:
                return
            self.urdf_path = urdf_input

            if _has_fk:
                try:
                    self.forward_kinematics = create_fk_from_urdf(self.urdf_path, self.current_arm)
                    print("✓ 正运动学已初始化")
                except Exception as e:
                    print(f"✗ 正运动学初始化失败: {e}")
                    return

        # 获取相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera = self.cameras.get(arm_config.camera_name)
        if camera is None:
            print("✗ 主相机未连接")
            return

        print("""
对齐流程：
  1. 系统检测工件和卡槽位置
  2. 计算像素偏移（考虑目标偏移量）
  3. 使用外参矩阵计算精确的世界坐标偏移
  4. 移动TCP进行对齐
  5. 重复直到对齐完成

按键说明：
  A - 开始自动对齐
  M - 单步对齐（手动确认每一步）
  T - 设置目标偏移量（当前偏移作为目标）
  C - 清除目标偏移量
  Q - 退出

当前目标偏移: ({:.1f}, {:.1f}) 像素
""").format(self.target_offset_x, self.target_offset_y)

        # 颜色配置
        workpiece_color = WORKPIECE_COLOR
        slot_color = SLOT_COLOR

        cv2.namedWindow("Hand-Eye Alignment", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand-Eye Alignment", 1000, 800)

        auto_mode = False
        alignment_active = False
        last_alignment_time = 0

        while True:
            image = camera.read()
            if image is None:
                continue

            display = image.copy()

            # 检测标记
            state = self.controller.detector.detect(image, workpiece_color, slot_color)

            # 绘制检测结果
            if state.workpiece_detected:
                for m in state.workpiece_markers:
                    if m:
                        cv2.circle(display, (int(m.x), int(m.y)), 5, (0, 255, 0), -1)
                wp_center = state.workpiece_center
                cv2.circle(display, (int(wp_center[0]), int(wp_center[1])), 8, (0, 255, 0), 2)

            if state.slot_detected:
                for m in state.slot_markers:
                    if m:
                        cv2.circle(display, (int(m.x), int(m.y)), 5, (0, 0, 255), -1)
                slot_center = state.slot_center
                cv2.circle(display, (int(slot_center[0]), int(slot_center[1])), 8, (0, 0, 255), 2)

            # 计算偏移（考虑目标偏移量）
            if state.workpiece_detected and state.slot_detected:
                # 原始偏移
                raw_offset_x = state.offset_x
                raw_offset_y = state.offset_y

                # 实际需要修正的偏移 = 当前偏移 - 目标偏移
                offset_x = raw_offset_x - self.target_offset_x
                offset_y = raw_offset_y - self.target_offset_y
                pixel_error = np.sqrt(offset_x**2 + offset_y**2)

                # 显示偏移信息
                cv2.putText(display, f"Raw: ({raw_offset_x:.1f}, {raw_offset_y:.1f}) px", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(display, f"Target: ({self.target_offset_x:.1f}, {self.target_offset_y:.1f}) px", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
                cv2.putText(display, f"Error: ({offset_x:.1f}, {offset_y:.1f}) px", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Dist: {pixel_error:.1f} px", (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 绘制偏移向量
                cv2.arrowedLine(display,
                               (int(wp_center[0]), int(wp_center[1])),
                               (int(slot_center[0]), int(slot_center[1])),
                               (255, 255, 0), 2)

                # 自动对齐
                if auto_mode and alignment_active:
                    current_time = time.time()
                    if current_time - last_alignment_time > 0.5:  # 每0.5秒执行一次
                        if pixel_error > 5:  # 大于5像素才调整
                            success = self._execute_hand_eye_alignment(offset_x, offset_y, state)
                            if success:
                                last_alignment_time = current_time
                            else:
                                alignment_active = False
                                auto_mode = False
                        else:
                            cv2.putText(display, "ALIGNED!", (10, 130),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            alignment_active = False

            # 显示状态
            status = "AUTO" if auto_mode else "MANUAL" if alignment_active else "IDLE"
            color = (0, 255, 0) if auto_mode else (0, 255, 255) if alignment_active else (128, 128, 128)
            cv2.putText(display, f"Mode: {status}", (10, display.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(display, "[A]uto [M]anual [T]arget [C]lear [Q]uit", (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Hand-Eye Alignment", display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('a') or key == ord('A'):
                auto_mode = True
                alignment_active = True
                print("开始自动对齐...")
            elif key == ord('m') or key == ord('M'):
                auto_mode = False
                alignment_active = True
                if state.workpiece_detected and state.slot_detected:
                    # 使用修正后的偏移
                    offset_x = state.offset_x - self.target_offset_x
                    offset_y = state.offset_y - self.target_offset_y
                    self._execute_hand_eye_alignment(offset_x, offset_y, state)
            elif key == ord('t') or key == ord('T'):
                # 设置目标偏移量（当前偏移作为目标）
                if state.workpiece_detected and state.slot_detected:
                    self.target_offset_x = state.offset_x
                    self.target_offset_y = state.offset_y
                    print(f"✓ 目标偏移量已设置: ({self.target_offset_x:.1f}, {self.target_offset_y:.1f}) 像素")
                    print("  对齐时将向此目标偏移量靠近")
            elif key == ord('c') or key == ord('C'):
                # 清除目标偏移量
                self.target_offset_x = 0.0
                self.target_offset_y = 0.0
                print("✓ 目标偏移量已清除")
            elif key == ord('q') or key == ord('Q'):
                break

        cv2.destroyWindow("Hand-Eye Alignment")

    def _execute_hand_eye_alignment(self, offset_x: float, offset_y: float, state) -> bool:
        """执行一次手眼标定对齐"""
        # 获取当前关节状态
        joints = self.controller.get_joint_states()
        if joints is None:
            print("  ✗ 无法获取关节状态")
            return False

        # 计算TCP位姿
        if self.forward_kinematics:
            try:
                pose = self.forward_kinematics.compute(joints)
                tcp_pos = pose.get_position()
                tcp_rot = pose.quaternion
            except Exception as e:
                print(f"  ✗ 正运动学计算失败: {e}")
                return False
        else:
            print("  ✗ 正运动学未初始化")
            return False

        # 更新坐标变换器
        self.coordinate_transformer.set_tcp_pose(tcp_pos, tcp_rot, "quaternion")

        # 估计深度（从Z坐标或双目视觉）
        depth = tcp_pos[2] - 0.1  # 简化：假设工件在TCP下方10cm
        if hasattr(self.controller, 'z_controller') and self.controller.z_controller:
            # 使用双目深度估计
            depth_estimate = self.controller.z_controller.get_depth_estimate()
            if depth_estimate and depth_estimate.valid:
                depth = depth_estimate.depth_m

        # 计算调整量
        pixel_offset = (offset_x, offset_y)
        tcp_adjustment, info = self.coordinate_transformer.compute_alignment_adjustment(
            pixel_offset, depth
        )

        # 显示调整信息
        world_offset = info['world_offset_m']
        print(f"  像素偏移: ({offset_x:.1f}, {offset_y:.1f}) @ 深度 {depth:.3f}m")
        print(f"  世界偏移: ({world_offset[0]*1000:.2f}, {world_offset[1]*1000:.2f}, {world_offset[2]*1000:.2f}) mm")

        # 缩放调整量（避免过冲）
        tcp_adjustment = tcp_adjustment * 0.8

        # 执行移动
        new_tcp_pos = tcp_pos + tcp_adjustment
        success = self.controller.move_to_position(
            new_tcp_pos[0], new_tcp_pos[1], new_tcp_pos[2],
            tcp_rot[0], tcp_rot[1], tcp_rot[2], tcp_rot[3]
        )

        if success:
            print(f"  ✓ 移动成功")
            time.sleep(0.3)  # 等待稳定
        else:
            print(f"  ✗ 移动失败")

        return success

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
  系统会记录设置时的高度，对齐时自动进行透视补偿。

操作步骤：
  1. 工件被夹爪夹住
  2. 手动将工件放入卡槽正确位置
  3. 确认检测正常后设置偏移量
  4. 系统记录当前高度，对齐时自动透视补偿

注意：对齐时可以在任意高度进行，系统会自动补偿透视效应！
""")

        input("\n准备好后按 Enter (工件在正确位置)...")

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

    def ibvs_memory_phase(self):
        """
        IBVS记忆阶段 - 采集定妆照

        在完美对齐位置抬高15cm，保存特征点的3D世界坐标。
        """
        if not _has_ibvs:
            print("✗ IBVS模块未加载")
            return False

        if not self.controller:
            print("请先连接设备")
            return False

        print("\n" + "="*60)
        print("IBVS 记忆阶段 - 采集定妆照")
        print("="*60)
        print("""
原理：
  在完美对齐位置记录特征点的3D世界坐标。
  后续盲插时，即使相机被遮挡，也能通过"脑补"虚拟像素精确对齐。

┌─────────────────────────────────────────────────────────────┐
│ 操作步骤：                                                   │
│                                                              │
│  1. 【完美对齐】手动将工件精确放入卡槽                       │
│     └─ 工件在夹爪里，完美插入卡槽                           │
│                                                              │
│  2. 【垂直抬高】切换到直线模式，沿Z轴抬高15cm                │
│     ├─ ✓ 只沿Z轴向上移动                                    │
│     ├─ ✓ XY方向不动                                         │
│     ├─ ✓ 末端执行器不旋转                                   │
│     └─ ⚠ 工件还在夹爪里，不要放下！                         │
│                                                              │
│  3. 【确认标记】确保工件标记和卡槽标记都清晰可见             │
│                                                              │
│  4. 【记忆】按 'M' 键记忆当前位置                           │
│                                                              │
│  5. 【保存】按 'S' 键保存到文件                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

为什么工件要在夹爪里？
  - 如果工件留在卡槽，会遮挡卡槽标记，无法检测
  - 工件在夹爪里，抬高后两组标记都可见

为什么只沿Z轴抬高？
  - 保证像素位置一致性
  - 简化操作，不易出错
  - 记忆的像素位置 = 完美对齐时看到的像素位置
""")

        # 检查/初始化正运动学
        if self.forward_kinematics is None:
            if self.urdf_path is None:
                print("\n需要URDF文件来计算正运动学。")
                urdf_input = input("请输入URDF文件路径: ").strip()
                if urdf_input:
                    self.urdf_path = urdf_input

            if self.urdf_path and _has_fk:
                try:
                    self.forward_kinematics = create_fk_from_urdf(self.urdf_path, self.current_arm)
                    print(f"✓ 正运动学已初始化")
                except Exception as e:
                    print(f"✗ 正运动学初始化失败: {e}")
                    return False
            else:
                print("✗ 缺少正运动学，IBVS无法工作")
                return False

        # 检查手眼标定
        if self.coordinate_transformer is None:
            if not self.load_coordinate_transformer():
                print("✗ 请先完成手眼标定")
                return False

        # 获取相机
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera = self.cameras.get(arm_config.camera_name)
        if camera is None:
            print("✗ 主相机未连接")
            return False

        # 创建IBVS控制器
        camera_matrix = np.array([
            [500.0, 0, 320.0],
            [0, 500.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float64)

        self.ibvs_controller = VirtualIBVSController(
            camera_matrix=camera_matrix,
            extrinsic_matrix=self.coordinate_transformer.T_flange2cam,
            lambda_gain=0.5,
            pixel_tolerance=3.0
        )

        # 获取深度估计器（如果有）
        depth_estimator = getattr(self.controller, 'z_controller', None)

        # 运行记忆阶段
        import cv2

        cv2.namedWindow("IBVS Memory", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IBVS Memory", 800, 600)

        memory_captured = False

        while True:
            image = camera.read()
            if image is None:
                continue

            display = image.copy()

            # 检测标记
            state = self.controller.detector.detect_dual_marker_state(image)

            # 绘制检测结果
            if state.workpiece_detected:
                for m in state.workpiece_markers:
                    if m:
                        cv2.circle(display, (int(m[0]), int(m[1])), 8, (0, 255, 0), -1)
                        cv2.circle(display, (int(m[0]), int(m[1])), 10, (255, 255, 255), 2)

            if state.slot_detected:
                for m in state.slot_markers:
                    if m:
                        cv2.circle(display, (int(m[0]), int(m[1])), 8, (0, 0, 255), -1)
                        cv2.circle(display, (int(m[0]), int(m[1])), 10, (255, 255, 255), 2)

            # 状态显示
            status = "CAPTURED" if memory_captured else "READY"
            color = (0, 255, 0) if memory_captured else (0, 255, 255)
            cv2.putText(display, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(display, "[M]emorize [Z]Lift+15cm [S]ave [L]oad [Q]uit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示标记数量
            wp_count = sum(1 for m in state.workpiece_markers if m is not None)
            slot_count = sum(1 for m in state.slot_markers if m is not None)
            cv2.putText(display, f"Markers: WP={wp_count} Slot={slot_count} (Need 4+)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示当前高度和操作提示
            joints = self.controller.get_joint_states()
            current_height = 0
            current_pose = None
            if joints is not None and self.forward_kinematics:
                try:
                    current_pose = self.forward_kinematics.compute(joints)
                    current_height = current_pose.z
                    cv2.putText(display, f"Height: {current_height*100:.1f}cm", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                except:
                    pass

            # 显示操作提示
            cv2.putText(display, "Tip: Put workpiece in slot, then press Z to lift 15cm", (10, display.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("IBVS Memory", display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('z') or key == ord('Z'):
                # 自动沿Z轴抬高15cm（XY不变）
                if current_pose is None:
                    print("\n⚠ 无法获取当前位姿")
                    continue

                print("\n正在沿Z轴抬高...")
                try:
                    # 获取当前位姿
                    current_x = current_pose.x
                    current_y = current_pose.y
                    current_z = current_pose.z
                    current_rot = current_pose.quaternion

                    # 计算目标高度（抬高15cm）
                    target_z = current_z + 0.15

                    # 调用控制器移动（只改变Z，XY和旋转不变）
                    if hasattr(self.controller, 'move_to_position'):
                        print(f"  当前高度: {current_z*100:.1f}cm")
                        print(f"  目标高度: {target_z*100:.1f}cm")

                        # 分步抬高，更安全
                        step_height = 0.03  # 每步3cm
                        total_steps = int(0.15 / step_height)
                        actual_lift = 0.0

                        for step in range(total_steps):
                            step_z = current_z + (step + 1) * step_height

                            success = self.controller.move_to_position(
                                current_x, current_y, step_z,
                                current_rot[0], current_rot[1], current_rot[2], current_rot[3]
                            )

                            if not success:
                                print(f"  ⚠ 第{step+1}步移动失败，已抬高约 {step * step_height * 100:.1f}cm")
                                break

                            time.sleep(0.3)  # 等待稳定

                            # 验证实际位置
                            new_joints = self.controller.get_joint_states()
                            if new_joints is not None:
                                try:
                                    new_pose = self.forward_kinematics.compute(new_joints)
                                    actual_z = new_pose.z
                                    actual_lift = actual_z - current_z
                                    print(f"  步骤 {step+1}/{total_steps}: 实际高度 {actual_z*100:.1f}cm")

                                    # 检查XY是否偏移
                                    xy_error = np.sqrt((new_pose.x - current_x)**2 + (new_pose.y - current_y)**2)
                                    if xy_error > 0.005:  # 5mm
                                        print(f"    ⚠ XY偏移: {xy_error*1000:.1f}mm")
                                except:
                                    pass

                        if actual_lift > 0.1:  # 至少抬高10cm
                            print(f"\n✓ 抬高完成: 实际抬高 {actual_lift*100:.1f}cm")
                            print("  如果标记可见，请按 M 记忆")
                        else:
                            print(f"\n⚠ 抬高不足 ({actual_lift*100:.1f}cm)，可能受到关节限位限制")
                            print("  建议：")
                            print("  1. 调整机器人姿态后重试")
                            print("  2. 或手动使用示教器抬高")

                    else:
                        print("\n⚠ 控制器不支持 move_to_position 方法")
                        print("  请手动使用示教器抬高：")
                        print("  1. 切换到直线模式")
                        print("  2. 只操作Z轴方向")
                        print("  3. 抬高约15cm")
                        print("  4. 注意观察是否有关节限位警告")

                except Exception as e:
                    print(f"\n✗ 抬高失败: {e}")
                    print("  可能原因：")
                    print("  - 关节到达限位")
                    print("  - 逆运动学无解（奇异点）")
                    print("  - 目标位置超出工作空间")
                    print("\n  请尝试：")
                    print("  1. 调整机器人姿态")
                    print("  2. 使用示教器手动抬高")
                    print("  3. 减小抬高距离")

            elif key == ord('m') or key == ord('M'):
                # 记忆当前位置
                if wp_count + slot_count < 4:
                    print(f"\n⚠ 标记数量不足 ({wp_count + slot_count}/4)")
                    continue

                if joints is None:
                    print("\n⚠ 无法获取关节状态")
                    continue

                try:
                    pose = self.forward_kinematics.compute(joints)
                    flange_pos = pose.get_position()
                    flange_rot = pose.quaternion

                    # 获取深度
                    depth = None
                    if depth_estimator and hasattr(depth_estimator, 'get_depth_estimate'):
                        depth_est = depth_estimator.get_depth_estimate()
                        if depth_est and depth_est.valid:
                            depth = depth_est.depth_m

                    # 收集特征点
                    workpiece_markers = [m for m in state.workpiece_markers if m is not None]
                    slot_markers = [m for m in state.slot_markers if m is not None]

                    # 记忆特征点
                    success = self.ibvs_controller.memorize_from_markers(
                        workpiece_markers=workpiece_markers,
                        slot_markers=slot_markers,
                        flange_position=flange_pos,
                        flange_rotation=flange_rot,
                        depth=depth
                    )

                    if success:
                        memory_captured = True

                except Exception as e:
                    print(f"\n✗ 记忆失败: {e}")

            elif key == ord('s') or key == ord('S'):
                # 保存记忆
                if self.ibvs_controller and self.ibvs_controller.state.memorized:
                    output_path = Path(__file__).parent / "ibvs_memory.json"
                    self.ibvs_controller.save_memory(str(output_path))
                else:
                    print("\n⚠ 请先按M记忆特征点")

            elif key == ord('l') or key == ord('L'):
                # 加载记忆
                input_path = Path(__file__).parent / "ibvs_memory.json"
                if input_path.exists():
                    if self.ibvs_controller is None:
                        self.ibvs_controller = VirtualIBVSController(
                            camera_matrix=camera_matrix,
                            extrinsic_matrix=self.coordinate_transformer.T_flange2cam
                        )
                    if self.ibvs_controller.load_memory(str(input_path)):
                        memory_captured = True
                else:
                    print("\n⚠ 未找到保存的记忆文件")

            elif key == ord('q') or key == ord('Q'):
                break

        cv2.destroyWindow("IBVS Memory")
        return memory_captured

    def ibvs_alignment_phase(self):
        """
        IBVS对齐阶段 - 盲插控制

        即使相机被遮挡，也能通过虚拟重投影精确对齐。
        """
        if not _has_ibvs:
            print("✗ IBVS模块未加载")
            return False

        if not self.controller:
            print("请先连接设备")
            return False

        if self.ibvs_controller is None or not self.ibvs_controller.state.memorized:
            # 尝试加载记忆
            input_path = Path(__file__).parent / "ibvs_memory.json"
            if input_path.exists():
                if self.ibvs_controller is None:
                    # 需要先初始化
                    if self.coordinate_transformer is None:
                        if not self.load_coordinate_transformer():
                            print("✗ 请先完成手眼标定")
                            return False

                    camera_matrix = np.array([
                        [500.0, 0, 320.0],
                        [0, 500.0, 240.0],
                        [0, 0, 1]
                    ], dtype=np.float64)

                    self.ibvs_controller = VirtualIBVSController(
                        camera_matrix=camera_matrix,
                        extrinsic_matrix=self.coordinate_transformer.T_flange2cam,
                        lambda_gain=0.5,
                        pixel_tolerance=3.0
                    )

                self.ibvs_controller.load_memory(str(input_path))
            else:
                print("✗ 请先执行IBVS记忆阶段 (选项 M)")
                return False

        if self.forward_kinematics is None:
            print("✗ 正运动学未初始化")
            return False

        print("\n" + "="*60)
        print("IBVS 对齐阶段 - 盲插控制")
        print("="*60)
        print("""
原理：
  利用记忆的3D特征点坐标，通过正运动学实时计算"虚拟像素"。
  即使相机被遮挡，也能精确对齐。

操作步骤：
  1. 移动机器人到卡槽上方
  2. 按 'A' 开始自动IBVS对齐
  3. 按 'M' 单步对齐
  4. 按 'Q' 退出

特点：
  - 不依赖实时图像
  - 控制频率可达1000Hz
  - 毫米级精度
""")

        # 获取相机（可选，用于调试显示）
        arm_config = ARM_CONFIGS.get(self.current_arm)
        camera = self.cameras.get(arm_config.camera_name)

        import cv2

        if camera:
            cv2.namedWindow("IBVS Alignment", cv2.WINDOW_NORMAL)

        success = False
        iteration = 0
        max_iterations = 500
        auto_mode = False

        while iteration < max_iterations:
            # 获取当前位姿
            joints = self.controller.get_joint_states()
            if joints is None:
                print("\n⚠ 无法获取关节状态")
                break

            try:
                pose = self.forward_kinematics.compute(joints)
                flange_pos = pose.get_position()
                flange_rot = pose.quaternion
            except Exception as e:
                print(f"\n⚠ 正运动学计算失败: {e}")
                break

            # 计算速度指令
            V_flange, info = self.ibvs_controller.calculate_velocity(flange_pos, flange_rot)

            if "error" in info:
                print(f"\n⚠ 控制错误: {info['error']}")
                break

            # 调试显示
            if camera:
                image = camera.read()
                if image is not None:
                    display = image.copy()

                    # 绘制虚拟特征点（紫色）和目标位置（绿色）
                    for i, (u, v) in enumerate(info.get("virtual_pixels", [])):
                        cv2.circle(display, (int(u), int(v)), 8, (255, 0, 255), -1)
                        if i < len(self.ibvs_controller.state.feature_points):
                            target = self.ibvs_controller.state.feature_points[i].target_pixel
                            cv2.circle(display, (int(target[0]), int(target[1])), 8, (0, 255, 0), 2)
                            cv2.line(display, (int(u), int(v)), (int(target[0]), int(target[1])), (255, 255, 0), 1)

                    # 显示误差
                    cv2.putText(display, f"Error: {info['total_error']:.2f}px", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    status = "ALIGNED!" if info['aligned'] else ("AUTO" if auto_mode else "MANUAL")
                    color = (0, 255, 0) if info['aligned'] else ((0, 255, 255) if auto_mode else (255, 255, 255))
                    cv2.putText(display, f"Status: {status}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    cv2.putText(display, "[A]uto [M]anual [Q]uit", (10, display.shape[0]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow("IBVS Alignment", display)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 0
            else:
                key = input("\n[A]uto [M]anual [Q]uit: ").strip().upper()
                if key:
                    key = ord(key)
                else:
                    key = ord('a') if auto_mode else 0

            if key == ord('a') or key == ord('A'):
                auto_mode = True
                print("开始自动IBVS对齐...")
            elif key == ord('m') or key == ord('M'):
                auto_mode = False
            elif key == ord('q') or key == ord('Q'):
                break

            # 检查是否对齐
            if info["aligned"]:
                print(f"\n✓ 对齐成功! 最终误差: {info['total_error']:.2f}px")
                success = True
                break

            # 执行移动
            if auto_mode:
                velocity_scale = 0.8
                max_velocity = 0.02  # 最大速度 20mm/s (安全限制)

                scaled_velocity = V_flange[:3] * velocity_scale

                # 速度限制（防止大幅移动）
                velocity_magnitude = np.linalg.norm(scaled_velocity)
                if velocity_magnitude > max_velocity:
                    scaled_velocity = scaled_velocity * (max_velocity / velocity_magnitude)

                # 优先级1: 速度控制（最平滑）
                if hasattr(self.controller, 'has_velocity_control') and \
                   self.controller.has_velocity_control():
                    # 直接发送速度指令，连续平滑移动
                    self.controller.set_cartesian_velocity(
                        scaled_velocity[0], scaled_velocity[1], scaled_velocity[2]
                    )
                    print(f"\r迭代 {iteration}: 误差={info['total_error']:.2f}px 速度={velocity_magnitude*1000:.1f}mm/s [速度控制]   ", end="", flush=True)
                    time.sleep(0.02)  # 50Hz控制周期

                # 优先级2: 伺服模式（平滑轨迹）
                elif hasattr(self.controller, 'has_servo_mode') and \
                     self.controller.has_servo_mode():
                    dt = 0.05
                    new_pos = flange_pos + scaled_velocity * dt
                    self.controller.servo_to_position(
                        new_pos[0], new_pos[1], new_pos[2],
                        flange_rot[0], flange_rot[1], flange_rot[2], flange_rot[3],
                        time_ms=50
                    )
                    print(f"\r迭代 {iteration}: 误差={info['total_error']:.2f}px 速度={velocity_magnitude*1000:.1f}mm/s [伺服模式]   ", end="", flush=True)
                    time.sleep(dt)

                # 优先级3: 位置增量（阶梯式，可能有小抖动）
                elif hasattr(self.controller, 'move_to_position'):
                    dt = 0.05
                    new_pos = flange_pos + scaled_velocity * dt
                    self.controller.move_to_position(
                        new_pos[0], new_pos[1], new_pos[2],
                        flange_rot[0], flange_rot[1], flange_rot[2], flange_rot[3]
                    )
                    print(f"\r迭代 {iteration}: 误差={info['total_error']:.2f}px 速度={velocity_magnitude*1000:.1f}mm/s [位置模式]   ", end="", flush=True)
                    time.sleep(dt)

                else:
                    print(f"\n✗ 控制器不支持移动方法")
                    break

                iteration += 1

        if camera:
            cv2.destroyWindow("IBVS Alignment")

        if not success:
            print(f"\n⚠ IBVS对齐结束")

        return success

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
            print("8. 运行对齐 (传统灵敏度方法)")
            print("8.5 手眼标定对齐 (推荐，精度更高)")
            print("--- IBVS 视觉伺服 (抗遮挡) ---")
            print("M. IBVS记忆 (采集定妆照)")
            print("I. IBVS对齐 (盲插控制)")
            print("---")
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
                print("  === 手眼标定 (推荐，精度更高) ===")
                print("  H. 手眼标定 (ChArUco板，一次完成)")
                print("  T. TCP标定 (探针四点法)")
                print("  R. 重投影验证 (验证手眼标定精度)")
                print("  === 传统标定 ===")
                print("  1. 像素-毫米标定 (基础标定)")
                print("  2. 关节灵敏度标定 (手动移动)")
                print("  3. 关节灵敏度标定 (自动移动)")
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
                print("  P. 透视方向校准 (相机倾斜方向)")
                print("  0. 修复灵敏度方向 (翻转X/Y)")

                calib_choice = input("选项: ").strip().upper()

                if calib_choice == "H":
                    # 手眼标定
                    if not system.controller:
                        system.connect()
                    system.hand_eye_calibration()
                elif calib_choice == "T":
                    # TCP标定
                    if not system.controller:
                        system.connect()
                    system.tcp_calibration()
                elif calib_choice == "R":
                    # 重投影验证
                    system.reprojection_verification()
                elif calib_choice == "1":
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
                elif calib_choice == "P":
                    # 透视方向校准
                    if not system.controller:
                        system.connect()
                    system.controller.calibrate_perspective_direction()
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

            elif choice == "8.5":
                # 手眼标定对齐
                if not system.controller:
                    system.connect()
                system.align_with_hand_eye()

            elif choice.upper() == "M":
                # IBVS记忆阶段
                if not system.controller:
                    system.connect()
                system.ibvs_memory_phase()

            elif choice.upper() == "I":
                # IBVS对齐阶段
                if not system.controller:
                    system.connect()
                system.ibvs_alignment_phase()

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
