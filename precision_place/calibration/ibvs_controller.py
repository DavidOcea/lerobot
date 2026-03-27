#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
虚拟IBVS控制模块 (Virtual IBVS Controller)

实现抗遮挡的图像视觉伺服控制，在相机被遮挡时仍能精确对齐。

核心原理：
  1. 高空记忆：在Z=15cm处采集"定妆照"，保存特征点的3D世界坐标
  2. 虚拟重投影：相机被遮挡时，利用正运动学"脑补"虚拟像素位置
  3. 雅可比控制：计算6DoF速度指令，驱动机械臂精确对齐

优势：
  - 相机被遮挡时仍能工作
  - 控制频率可达1000Hz（纯数学计算）
  - 毫米级精度

记忆阶段操作流程：
  ┌─────────────────────────────────────────────────────────────┐
  │ 步骤1: 完美对齐                                             │
  │   - 手动将工件精确放入卡槽                                   │
  │   - 工件在夹爪里（不要放下）                                 │
  │                                                             │
  │ 步骤2: 垂直抬高15cm                                         │
  │   - 只沿Z轴抬高，XY不动                                     │
  │   - 末端执行器不旋转                                        │
  │   - 工件还在夹爪里                                          │
  │                                                             │
  │ 步骤3: 记忆特征点                                           │
  │   - 工件标记可见 ✓                                         │
  │   - 卡槽标记可见 ✓                                         │
  │   - 按 'M' 记忆                                            │
  └─────────────────────────────────────────────────────────────┘

使用方法：
    from precision_place.calibration.ibvs_controller import VirtualIBVSController

    # 创建控制器
    controller = VirtualIBVSController(camera_matrix, extrinsic_matrix)

    # 高空记忆
    controller.memorize_features(image, flange_pose)

    # 控制循环
    while not aligned:
        velocity = controller.calculate_velocity(flange_pose)
        robot.move(velocity)

参考：
    - RJ2506项目: docs/04-Module4-Virtual-IBVS-Math.md
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from scipy.spatial.transform import Rotation as R


@dataclass
class FeaturePoint3D:
    """3D特征点"""
    # 世界坐标 (米)
    world_position: np.ndarray  # [x, y, z]
    # 目标像素坐标 (定妆照中的位置)
    target_pixel: Tuple[float, float]  # (u, v)
    # 特征类型
    feature_type: str = "marker"  # marker, corner, edge
    # 特征ID (标记编号等)
    feature_id: int = -1


@dataclass
class IBVSState:
    """IBVS状态"""
    # 是否已完成高空记忆
    memorized: bool = False
    # 记忆时的法兰位姿
    memory_flange_position: np.ndarray = None
    memory_flange_rotation: np.ndarray = None
    # 记忆时的深度
    memory_depth: float = 0.0
    # 特征点列表
    feature_points: List[FeaturePoint3D] = field(default_factory=list)
    # 当前虚拟像素
    current_virtual_pixels: List[Tuple[float, float]] = field(default_factory=list)
    # 当前误差
    current_error: np.ndarray = None
    # 对齐标志
    aligned: bool = False


class VirtualIBVSController:
    """
    虚拟IBVS控制器

    实现抗遮挡的图像视觉伺服控制。

    工作流程：
        1. memorize_features(): 高空采集"定妆照"，保存3D特征点
        2. calculate_velocity(): 盲插时计算速度指令
        3. is_aligned(): 判断是否对齐
    """

    def __init__(self,
                 camera_matrix: np.ndarray,
                 extrinsic_matrix: np.ndarray,
                 lambda_gain: float = 0.5,
                 pixel_tolerance: float = 3.0):
        """
        初始化IBVS控制器

        Args:
            camera_matrix: 3x3 相机内参矩阵
            extrinsic_matrix: 4x4 外参矩阵 (Flange -> Camera)
            lambda_gain: 控制增益 (越大越快，但可能过冲)
            pixel_tolerance: 对齐容差 (像素)
        """
        self.K = camera_matrix.copy()
        self.T_flange2cam = extrinsic_matrix.copy()
        self.T_cam2flange = np.linalg.inv(extrinsic_matrix)
        self.lambda_gain = lambda_gain
        self.pixel_tolerance = pixel_tolerance

        # IBVS状态
        self.state = IBVSState()

        # 统计信息
        self.iteration_count = 0
        self.total_error_history = []

    def memorize_features(self,
                         feature_pixels: List[Tuple[float, float]],
                         flange_position: np.ndarray,
                         flange_rotation: np.ndarray,
                         depths: List[float] = None,
                         feature_types: List[str] = None,
                         feature_ids: List[int] = None) -> bool:
        """
        高空记忆特征点

        在Z=15cm左右采集特征点，保存其3D世界坐标。
        这就是"定妆照"采集过程。

        Args:
            feature_pixels: 特征点像素坐标列表 [(u1,v1), (u2,v2), ...]
            flange_position: 法兰位置 [x, y, z] 米
            flange_rotation: 法兰旋转 (四元数 [qx, qy, qz, qw])
            depths: 各特征点深度 (米)，可选
            feature_types: 特征类型列表
            feature_ids: 特征ID列表

        Returns:
            是否记忆成功
        """
        n_features = len(feature_pixels)
        if n_features < 4:
            print(f"⚠ 特征点数量不足 ({n_features}/4)，至少需要4个")
            return False

        print(f"\n{'='*50}")
        print(f"高空记忆特征点 ({n_features}个)")
        print(f"{'='*50}")

        # 保存记忆时的法兰位姿
        self.state.memory_flange_position = np.array(flange_position)
        self.state.memory_flange_rotation = np.array(flange_rotation)

        # 构建法兰位姿矩阵
        R_flange = R.from_quat(flange_rotation).as_matrix()
        T_base2flange = np.eye(4)
        T_base2flange[:3, :3] = R_flange
        T_base2flange[:3, 3] = flange_position

        # 计算相机位姿
        T_base2cam = T_base2flange @ self.T_flange2cam
        T_cam2base = np.linalg.inv(T_base2cam)

        # 提取相机内参
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # 清空旧数据
        self.state.feature_points.clear()

        # 转换每个特征点
        for i, (u, v) in enumerate(feature_pixels):
            # 归一化坐标
            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy

            # 估计深度
            if depths and i < len(depths) and depths[i] > 0:
                depth = depths[i]
            else:
                # 使用默认深度（从TCP高度估计）
                depth = 0.3  # 默认30cm

            # 计算相机坐标系下的3D点
            point_cam = np.array([
                x_norm * depth,
                y_norm * depth,
                depth,
                1.0
            ])

            # 转换到世界坐标系
            point_world = T_cam2base @ point_cam

            # 创建特征点
            feature_type = feature_types[i] if feature_types and i < len(feature_types) else "marker"
            feature_id = feature_ids[i] if feature_ids and i < len(feature_ids) else i

            feature_point = FeaturePoint3D(
                world_position=point_world[:3],
                target_pixel=(u, v),
                feature_type=feature_type,
                feature_id=feature_id
            )
            self.state.feature_points.append(feature_point)

            print(f"  特征 {i+1}: 像素({u:.1f}, {v:.1f}) → 世界({point_world[0]:.4f}, {point_world[1]:.4f}, {point_world[2]:.4f})m")

        # 保存记忆深度
        self.state.memory_depth = np.mean([fp.world_position[2] for fp in self.state.feature_points])
        self.state.memorized = True

        print(f"\n✓ 记忆完成! 共{n_features}个特征点")
        print(f"  记忆深度: {self.state.memory_depth:.3f}m")

        return True

    def memorize_from_markers(self,
                             workpiece_markers: List,
                             slot_markers: List,
                             flange_position: np.ndarray,
                             flange_rotation: np.ndarray,
                             depth: float = None) -> bool:
        """
        从标记检测结果记忆特征点

        便捷方法：直接使用检测到的标记作为特征点。

        Args:
            workpiece_markers: 工件标记列表 [(x1,y1), (x2,y2), ...]
            slot_markers: 卡槽标记列表 [(x1,y1), (x2,y2), ...]
            flange_position: 法兰位置
            flange_rotation: 法兰旋转
            depth: 深度（统一使用）

        Returns:
            是否记忆成功
        """
        feature_pixels = []
        feature_types = []
        feature_ids = []

        # 添加工件标记
        for i, m in enumerate(workpiece_markers):
            if m is not None:
                feature_pixels.append((m[0], m[1]))
                feature_types.append("workpiece")
                feature_ids.append(i)

        # 添加卡槽标记
        for i, m in enumerate(slot_markers):
            if m is not None:
                feature_pixels.append((m[0], m[1]))
                feature_types.append("slot")
                feature_ids.append(i + len(workpiece_markers))

        if depth is not None:
            depths = [depth] * len(feature_pixels)
        else:
            depths = None

        return self.memorize_features(
            feature_pixels=feature_pixels,
            flange_position=flange_position,
            flange_rotation=flange_rotation,
            depths=depths,
            feature_types=feature_types,
            feature_ids=feature_ids
        )

    def compute_interaction_matrix(self, u: float, v: float, Z: float) -> np.ndarray:
        """
        计算单个特征点的2x6图像雅可比矩阵

        这是IBVS的核心数学公式，将像素变化映射到相机运动。

        Args:
            u, v: 归一化像素坐标 (相对于光心)
            Z: 深度 (米)

        Returns:
            2x6 雅可比矩阵 L_s
        """
        L = np.zeros((2, 6))

        # 防止除零
        if Z < 0.001:
            Z = 0.001

        # 填充雅可比矩阵
        L[0, 0] = -1.0 / Z
        L[0, 1] = 0.0
        L[0, 2] = u / Z
        L[0, 3] = u * v
        L[0, 4] = -(1.0 + u * u)
        L[0, 5] = v

        L[1, 0] = 0.0
        L[1, 1] = -1.0 / Z
        L[1, 2] = v / Z
        L[1, 3] = 1.0 + v * v
        L[1, 4] = -u * v
        L[1, 5] = -u

        return L

    def adjoint_transform(self, V_cam: np.ndarray) -> np.ndarray:
        """
        伴随矩阵变换：将相机速度映射到法兰速度

        Args:
            V_cam: 相机速度 [vx, vy, vz, wx, wy, wz]

        Returns:
            法兰速度 [vx, vy, vz, wx, wy, wz]
        """
        R_ext = self.T_cam2flange[:3, :3]
        t_ext = self.T_cam2flange[:3, 3]

        # 反对称矩阵
        t_skew = np.array([
            [0, -t_ext[2], t_ext[1]],
            [t_ext[2], 0, -t_ext[0]],
            [-t_ext[1], t_ext[0], 0]
        ])

        # 6x6伴随矩阵
        Adj = np.zeros((6, 6))
        Adj[:3, :3] = R_ext
        Adj[3:, 3:] = R_ext
        Adj[:3, 3:] = t_skew @ R_ext

        return Adj @ V_cam

    def calculate_velocity(self,
                          flange_position: np.ndarray,
                          flange_rotation: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        计算IBVS速度指令

        这是核心控制律，在盲插时调用。

        Args:
            flange_position: 当前法兰位置 [x, y, z] 米
            flange_rotation: 当前法兰旋转 (四元数)

        Returns:
            (V_flange, info) 法兰速度和详细信息
        """
        if not self.state.memorized:
            return np.zeros(6), {"error": "未完成高空记忆"}

        # 构建当前法兰位姿矩阵
        R_flange = R.from_quat(flange_rotation).as_matrix()
        T_base2flange = np.eye(4)
        T_base2flange[:3, :3] = R_flange
        T_base2flange[:3, 3] = flange_position

        # 计算当前相机位姿
        T_base2cam = T_base2flange @ self.T_flange2cam
        T_cam2base = np.linalg.inv(T_base2cam)

        # 相机内参
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # 收集所有特征点的雅可比和误差
        J_list = []
        error_list = []
        self.state.current_virtual_pixels.clear()

        for fp in self.state.feature_points:
            # 步骤1: 世界坐标 → 相机坐标
            P_world = np.append(fp.world_position, 1.0)
            P_cam = T_cam2base @ P_world

            Xc, Yc, Zc = P_cam[0], P_cam[1], P_cam[2]

            # 安全检查：深度过小表示穿模
            if Zc < 0.001:
                print("⚠ 警告: 推算深度穿模，强制刹车!")
                return np.zeros(6), {"error": "depth_violation", "Z": Zc}

            # 步骤2: 相机坐标 → 虚拟像素
            u_virtual = fx * Xc / Zc + cx
            v_virtual = fy * Yc / Zc + cy

            self.state.current_virtual_pixels.append((u_virtual, v_virtual))

            # 步骤3: 计算像素误差
            u_error = u_virtual - fp.target_pixel[0]
            v_error = v_virtual - fp.target_pixel[1]

            error_list.extend([u_error, v_error])

            # 步骤4: 计算雅可比矩阵
            # 归一化坐标
            u_norm = (u_virtual - cx) / fx
            v_norm = (v_virtual - cy) / fy

            L = self.compute_interaction_matrix(u_norm, v_norm, Zc)

            # 缩放到像素空间
            L[0, :] *= fx
            L[1, :] *= fy

            J_list.append(L)

        # 组装大矩阵
        J_full = np.vstack(J_list)  # (2N x 6)
        error_full = np.array(error_list).reshape(-1, 1)  # (2N x 1)

        # 保存当前误差
        self.state.current_error = error_full.flatten()

        # 计算伪逆
        try:
            J_pinv = np.linalg.pinv(J_full)
        except np.linalg.LinAlgError:
            print("⚠ 雅可比矩阵奇异，无法计算伪逆")
            return np.zeros(6), {"error": "singular_jacobian"}

        # 控制律: V_cam = -lambda * J^+ * e
        V_camera = -self.lambda_gain * (J_pinv @ error_full)

        # 转换到法兰坐标系
        V_flange = self.adjoint_transform(V_camera.flatten())

        # 计算总误差
        total_error = np.sqrt(np.sum(error_full**2))
        self.total_error_history.append(total_error)
        self.iteration_count += 1

        # 检查是否对齐
        self.state.aligned = total_error < self.pixel_tolerance

        # 返回信息和速度
        info = {
            "total_error": total_error,
            "aligned": self.state.aligned,
            "iteration": self.iteration_count,
            "virtual_pixels": self.state.current_virtual_pixels.copy(),
            "Z_estimates": [Zc for fp in self.state.feature_points]  # 简化
        }

        return V_flange, info

    def is_aligned(self) -> bool:
        """检查是否已对齐"""
        return self.state.aligned

    def get_current_error(self) -> float:
        """获取当前总误差 (像素)"""
        if self.state.current_error is not None:
            return np.sqrt(np.sum(self.state.current_error**2))
        return float('inf')

    def save_memory(self, filepath: str) -> bool:
        """
        保存记忆到文件

        Args:
            filepath: 保存路径

        Returns:
            是否成功
        """
        if not self.state.memorized:
            print("⚠ 没有可保存的记忆")
            return False

        try:
            data = {
                "memory_flange_position": self.state.memory_flange_position.tolist(),
                "memory_flange_rotation": self.state.memory_flange_rotation.tolist(),
                "memory_depth": float(self.state.memory_depth),
                "feature_points": [
                    {
                        "world_position": fp.world_position.tolist(),
                        "target_pixel": list(fp.target_pixel),
                        "feature_type": fp.feature_type,
                        "feature_id": fp.feature_id
                    }
                    for fp in self.state.feature_points
                ],
                "lambda_gain": self.lambda_gain,
                "pixel_tolerance": self.pixel_tolerance
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ IBVS记忆已保存: {filepath}")
            return True

        except Exception as e:
            print(f"✗ 保存失败: {e}")
            return False

    def load_memory(self, filepath: str) -> bool:
        """
        从文件加载记忆

        Args:
            filepath: 文件路径

        Returns:
            是否成功
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.state.memory_flange_position = np.array(data["memory_flange_position"])
            self.state.memory_flange_rotation = np.array(data["memory_flange_rotation"])
            self.state.memory_depth = data["memory_depth"]

            self.state.feature_points.clear()
            for fp_data in data["feature_points"]:
                self.state.feature_points.append(FeaturePoint3D(
                    world_position=np.array(fp_data["world_position"]),
                    target_pixel=tuple(fp_data["target_pixel"]),
                    feature_type=fp_data["feature_type"],
                    feature_id=fp_data["feature_id"]
                ))

            self.lambda_gain = data.get("lambda_gain", 0.5)
            self.pixel_tolerance = data.get("pixel_tolerance", 3.0)
            self.state.memorized = True

            print(f"✓ IBVS记忆已加载: {filepath}")
            print(f"  特征点数量: {len(self.state.feature_points)}")

            return True

        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return False

    def reset(self):
        """重置状态"""
        self.state = IBVSState()
        self.iteration_count = 0
        self.total_error_history.clear()
        print("✓ IBVS状态已重置")


class IBVSAlignmentRunner:
    """
    IBVS对齐运行器

    封装完整的IBVS对齐流程，方便集成到主系统。
    """

    def __init__(self,
                 ibvs_controller: VirtualIBVSController,
                 forward_kinematics,
                 robot_controller):
        """
        初始化对齐运行器

        Args:
            ibvs_controller: IBVS控制器
            forward_kinematics: 正运动学计算器
            robot_controller: 机器人控制器 (需有move_velocity或move_to_position方法)
        """
        self.ibvs = ibvs_controller
        self.fk = forward_kinematics
        self.robot = robot_controller

        # 运行参数
        self.max_iterations = 1000
        self.velocity_scale = 0.8  # 速度缩放，防止过冲

    def run_memory_phase(self, camera, detector, depth_estimator=None) -> bool:
        """
        执行记忆阶段

        Args:
            camera: 相机对象
            detector: 标记检测器
            depth_estimator: 深度估计器 (可选)

        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("IBVS 记忆阶段 - 采集定妆照")
        print("="*60)
        print("""
操作说明:
  1. 手动将工件精确放入卡槽 (完美对齐状态)
  2. 垂直抬高15cm (只沿Z轴向上)
  3. 确保所有标记在相机视野中
  4. 按 'M' 键记忆当前位置
  5. 按 'Q' 键退出

这是"定妆照"采集，将保存特征点的3D世界坐标。
""")

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
            state = detector.detect_dual_marker_state(image)

            # 绘制检测结果
            if state.workpiece_detected:
                for m in state.workpiece_markers:
                    if m:
                        cv2.circle(display, (int(m[0]), int(m[1])), 5, (0, 255, 0), -1)

            if state.slot_detected:
                for m in state.slot_markers:
                    if m:
                        cv2.circle(display, (int(m[0]), int(m[1])), 5, (0, 0, 255), -1)

            # 状态显示
            status = "CAPTURED" if memory_captured else "READY"
            color = (0, 255, 0) if memory_captured else (0, 255, 255)
            cv2.putText(display, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(display, "[M]emorize [Q]uit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示标记数量
            wp_count = sum(1 for m in state.workpiece_markers if m is not None)
            slot_count = sum(1 for m in state.slot_markers if m is not None)
            cv2.putText(display, f"Markers: WP={wp_count} Slot={slot_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("IBVS Memory", display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('m') or key == ord('M'):
                # 记忆当前位置
                if wp_count + slot_count < 4:
                    print(f"⚠ 标记数量不足 ({wp_count + slot_count}/4)")
                    continue

                # 获取当前位姿
                joints = self.robot.get_joint_states()
                if joints is None:
                    print("⚠ 无法获取关节状态")
                    continue

                try:
                    pose = self.fk.compute(joints)
                    flange_pos = pose.get_position()
                    flange_rot = pose.quaternion

                    # 获取深度
                    depth = None
                    if depth_estimator:
                        depth_est = depth_estimator.estimate_z(image)
                        if depth_est and depth_est.valid:
                            depth = depth_est.depth_m

                    # 记忆特征点
                    success = self.ibvs.memorize_from_markers(
                        workpiece_markers=[m for m in state.workpiece_markers if m is not None],
                        slot_markers=[m for m in state.slot_markers if m is not None],
                        flange_position=flange_pos,
                        flange_rotation=flange_rot,
                        depth=depth
                    )

                    if success:
                        memory_captured = True
                        # 保存到文件
                        self.ibvs.save_memory(str(Path(__file__).parent / "ibvs_memory.json"))

                except Exception as e:
                    print(f"✗ 记忆失败: {e}")

            elif key == ord('q') or key == ord('Q'):
                break

        cv2.destroyWindow("IBVS Memory")
        return memory_captured

    def run_alignment_phase(self, camera=None) -> bool:
        """
        执行对齐阶段

        即使相机被遮挡，也能通过虚拟重投影精确对齐。

        Args:
            camera: 相机对象 (可选，用于调试显示)

        Returns:
            是否对齐成功
        """
        if not self.ibvs.state.memorized:
            print("⚠ 请先执行记忆阶段")
            return False

        print("\n" + "="*60)
        print("IBVS 对齐阶段 - 盲插控制")
        print("="*60)
        print("开始IBVS控制...")

        import cv2

        if camera:
            cv2.namedWindow("IBVS Alignment", cv2.WINDOW_NORMAL)

        success = False
        iteration = 0

        while iteration < self.max_iterations:
            # 获取当前位姿
            joints = self.robot.get_joint_states()
            if joints is None:
                print("⚠ 无法获取关节状态")
                break

            try:
                pose = self.fk.compute(joints)
                flange_pos = pose.get_position()
                flange_rot = pose.quaternion
            except Exception as e:
                print(f"⚠ 正运动学计算失败: {e}")
                break

            # 计算速度指令
            V_flange, info = self.ibvs.calculate_velocity(flange_pos, flange_rot)

            if "error" in info:
                print(f"⚠ 控制错误: {info['error']}")
                break

            # 显示状态
            print(f"\r迭代 {iteration}: 误差={info['total_error']:.2f}px    ", end="", flush=True)

            # 检查是否对齐
            if info["aligned"]:
                print(f"\n✓ 对齐成功! 最终误差: {info['total_error']:.2f}px")
                success = True
                break

            # 执行移动
            # 将速度转换为位置增量
            dt = 0.05  # 控制周期
            delta_pos = V_flange[:3] * dt * self.velocity_scale

            # 计算新位置
            new_pos = flange_pos + delta_pos

            # 移动
            if hasattr(self.robot, 'move_to_position'):
                self.robot.move_to_position(
                    new_pos[0], new_pos[1], new_pos[2],
                    flange_rot[0], flange_rot[1], flange_rot[2], flange_rot[3]
                )
            else:
                print("⚠ 机器人控制器不支持移动命令")
                break

            # 调试显示
            if camera:
                image = camera.read()
                if image is not None:
                    display = image.copy()

                    # 绘制虚拟特征点
                    for i, (u, v) in enumerate(info.get("virtual_pixels", [])):
                        cv2.circle(display, (int(u), int(v)), 5, (255, 0, 255), -1)
                        # 绘制目标位置
                        if i < len(self.ibvs.state.feature_points):
                            target = self.ibvs.state.feature_points[i].target_pixel
                            cv2.circle(display, (int(target[0]), int(target[1])), 5, (0, 255, 0), 2)

                    cv2.putText(display, f"Error: {info['total_error']:.1f}px", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    cv2.imshow("IBVS Alignment", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            iteration += 1
            time.sleep(dt)

        if camera:
            cv2.destroyWindow("IBVS Alignment")

        if not success:
            print(f"\n⚠ 对齐超时或失败")

        return success


if __name__ == "__main__":
    # 测试代码
    print("虚拟IBVS控制器测试")

    # 创建模拟参数
    camera_matrix = np.array([
        [500.0, 0, 320.0],
        [0, 500.0, 240.0],
        [0, 0, 1]
    ])

    # 外参矩阵：相机在法兰下方，Z方向朝下
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])  # 相机朝下
    extrinsic_matrix[:3, 3] = [0.0, 0.0, 0.1]  # 相机在法兰下方10cm

    # 创建控制器
    controller = VirtualIBVSController(camera_matrix, extrinsic_matrix)

    # 模拟记忆 - 法兰在工件上方30cm
    feature_pixels = [(300, 200), (340, 200), (300, 240), (340, 240)]
    flange_pos = np.array([0.0, 0.0, 0.4])  # 法兰高度40cm
    flange_rot = np.array([0, 0, 0, 1])

    controller.memorize_features(feature_pixels, flange_pos, flange_rot, depths=[0.3]*4)

    # 模拟控制
    print("\n模拟控制:")
    for i in range(5):
        # 模拟微小偏移（向下移动）
        flange_pos = np.array([0.0, 0.0, 0.4 - 0.02*i])  # 逐渐下降
        V, info = controller.calculate_velocity(flange_pos, flange_rot)
        if 'total_error' in info:
            print(f"  迭代 {i}: 误差={info['total_error']:.2f}px, 速度={V[:3]}")
        else:
            print(f"  迭代 {i}: {info.get('error', '未知错误')}")

    print("\n测试完成")