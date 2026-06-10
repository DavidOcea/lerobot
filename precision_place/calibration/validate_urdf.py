#!/usr/bin/env python3
"""
URDF 逐关节验证 + 自动修正脚本

原理:
  以真实机器人为 ground truth, 逐一测试 URDF 中每个关节的 FK 预测精度。
  对每个关节: 单独转动 Δθ° → 观测 AprilTag 像素位移 → 与 URDF 预测对比。

可诊断:
  - 旋转轴方向反了  → <axis xyz> 取反
  - 旋转轴方向倾斜  → <origin rpy> 需要调整
  - link 长度错误    → magnitude_ratio = 实测位移/预测位移, 需要缩放 link

双臂交叉验证:
  先测左手(j0-j5) → 再测右手(j7-j12) → j14(trunk)被两次测量, 交叉验证

用法:
  # 在 run.py 中集成 (推荐):
  from precision_place.calibration.validate_urdf import URDFValidator
  validator = URDFValidator(fk, T_flange_cam, camera_matrix, controller, camera, urdf_path, arm="right")
  validator.validate_all()

  # FK-IBVS 对齐界面按 V 键触发
"""

import os
import copy
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import yaml


# 关节名称映射
JOINT_NAMES = {
    # 左手: joint indices 0-6
    0:  "left_arm_joint_1",
    1:  "left_arm_joint_2",
    2:  "left_arm_joint_3",
    3:  "left_arm_joint_4",
    4:  "left_arm_joint_5",
    5:  "left_arm_joint_6",
    6:  "left_arm_joint_7",
    # 右手: joint indices 7-13
    7:  "right_arm_joint_1",
    8:  "right_arm_joint_2",
    9:  "right_arm_joint_3",
    10: "right_arm_joint_4",
    11: "right_arm_joint_5",
    12: "right_arm_joint_6",
    13: "right_arm_joint_7",
    # 躯干
    14: "trunk_joint_1",
    15: "trunk_joint_2",
}

RIGHT_ARM_JOINTS = [7, 8, 9, 10, 11, 12, 14]
LEFT_ARM_JOINTS  = [0, 1, 2, 3, 4, 5, 14]


class JointValidationResult:
    """单个关节的验证结果"""
    def __init__(self, joint_idx: int):
        self.joint_idx = joint_idx
        self.joint_name = JOINT_NAMES.get(joint_idx, f"joint_{joint_idx}")
        self.delta_deg: float = 0.0
        self.pixel_before: Optional[Tuple[float, float]] = None
        self.pixel_after: Optional[Tuple[float, float]] = None
        self.measured_du: float = 0.0
        self.measured_dv: float = 0.0
        self.predicted_du: float = 0.0
        self.predicted_dv: float = 0.0
        self.direction_score: float = 0.0   # cos(实测, 预测), 1=完美, -1=反向
        self.magnitude_ratio: float = 0.0   # |实测|/|预测|, 1=完美
        self.status: str = "untested"
        self.issue: str = ""
        # 每个采样点的原始数据 (用于调试)
        self.samples: List[dict] = []


class URDFValidator:
    """URDF 逐关节验证器"""

    def __init__(self,
                 fk_solver,                       # ForwardKinematics 实例
                 T_flange_cam: np.ndarray,        # 4x4 手眼矩阵
                 camera_matrix: np.ndarray,       # 3x3 相机内参
                 controller,                      # 机器人控制器
                 camera,                          # 相机 (camera.read() → BGR)
                 urdf_path: str = None,
                 arm: str = "right",
                 joint_indices: List[int] = None):
        """
        Args:
            fk_solver: 基于 URDF 的 FK 求解器
            T_flange_cam: 手眼矩阵 (Flange→Camera)
            camera_matrix: 3x3 相机内参
            controller: 机器人 (get_joint_states, _smooth_move_all_joints)
            camera: 相机
            urdf_path: URDF 文件路径 (用于自动修正)
            arm: "left" 或 "right"
            joint_indices: 自定义测试关节, 默认根据 arm 自动选择
        """
        self.fk = fk_solver
        self.T_flange_cam = T_flange_cam
        self.T_cam_flange = np.linalg.inv(T_flange_cam)
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        self.controller = controller
        self.camera = camera
        self.urdf_path = urdf_path
        self.arm = arm

        if joint_indices is not None:
            self.joint_indices = joint_indices
        elif arm == "left":
            self.joint_indices = LEFT_ARM_JOINTS
        else:
            self.joint_indices = RIGHT_ARM_JOINTS

        from precision_place.calibration.simple_ibvs import AprilTagDetector
        self.tag_detector = AprilTagDetector(tag_family="tag36h11")

        self.results: Dict[int, JointValidationResult] = {}
        self._tag_world_pos: Optional[np.ndarray] = None

        # 备份记录
        self._backup_path: Optional[str] = None
        self._fixes_applied: List[str] = []

    # ==================== 核心验证 ====================

    def validate_all(self, delta_deg: float = 6.0, n_samples: int = 2,
                     return_after: bool = True):
        """
        逐关节验证 URDF。

        Args:
            delta_deg: 每个关节转动量 (度)
            n_samples: 每关节采样次数 (来回各一次取平均)
            return_after: 测试后回到初始位置
        """
        print(f"\n{'='*60}")
        print(f"URDF 逐关节验证 — {self.arm.upper()} ARM")
        print(f"{'='*60}")
        print(f"  URDF: {self.urdf_path}")
        print(f"  测试关节: {[JOINT_NAMES.get(j, str(j)) for j in self.joint_indices]}")
        print(f"  转动步长: {delta_deg}° × {n_samples}次")
        print(f"  ⚠  AprilTag 必须保持静止在相机视野内!")
        print(f"{'='*60}")

        initial_joints = self.controller.get_joint_states()
        if initial_joints is None:
            print("✗ 无法获取关节状态")
            return

        if not self._ensure_tag_visible():
            print("✗ AprilTag 不可见")
            return

        joints_start = initial_joints.copy()
        self._tag_world_pos = None  # 重置 tag 世界位置 (每轮重估计)

        for joint_idx in self.joint_indices:
            print(f"\n  [{JOINT_NAMES.get(joint_idx, f'j{joint_idx}')}] ", end="", flush=True)
            result = self._validate_single_joint(joint_idx, delta_deg, n_samples)
            self.results[joint_idx] = result
            # 实时输出进度
            tag = "✓" if result.status == "ok" else "✗"
            print(f"{tag} {result.status}  dir={result.direction_score:.3f}  mag={result.magnitude_ratio:.2f}")

        if return_after:
            print("\n  恢复初始位置...")
            try:
                self.controller._smooth_move_all_joints(joints_start, steps=10)
                time.sleep(0.3)
            except Exception:
                pass

        self.print_report()

    def _validate_single_joint(self, joint_idx: int, delta_deg: float,
                                n_samples: int) -> JointValidationResult:
        """验证单个关节"""
        result = JointValidationResult(joint_idx)

        if self._tag_world_pos is None:
            joints = self.controller.get_joint_states()
            pixel, depth = self._detect_tag()
            if pixel is None:
                result.status = "tag_lost"
                result.issue = "Tag not visible"
                return result
            self._tag_world_pos = self._estimate_tag_world(joints, pixel, depth)
            if self._tag_world_pos is None:
                result.status = "fk_fail"
                result.issue = "FK world estimation failed"
                return result

        measured_du_list = []
        measured_dv_list = []
        pred_du_list = []
        pred_dv_list = []

        for sample in range(n_samples):
            joints_before = self.controller.get_joint_states()
            if joints_before is None:
                continue
            pixel_before, _ = self._detect_tag()
            if pixel_before is None:
                continue

            pred_du, pred_dv = self._predict_pixel_movement(
                joints_before, joint_idx, delta_deg)
            if pred_du is None:
                continue

            # 转动
            target = joints_before.copy()
            target[joint_idx] += delta_deg
            self.controller._smooth_move_all_joints(target, steps=10)
            time.sleep(0.3)

            joints_after = self.controller.get_joint_states()
            pixel_after, _ = self._detect_tag()

            # 转回
            self.controller._smooth_move_all_joints(joints_before, steps=10)
            time.sleep(0.2)

            if pixel_after is None:
                continue

            actual_delta = joints_after[joint_idx] - joints_before[joint_idx]
            if abs(actual_delta) < 0.1:
                continue

            measured_du = (pixel_after[0] - pixel_before[0]) / actual_delta
            measured_dv = (pixel_after[1] - pixel_before[1]) / actual_delta

            measured_du_list.append(measured_du)
            measured_dv_list.append(measured_dv)
            pred_du_list.append(pred_du)
            pred_dv_list.append(pred_dv)

            result.samples.append({
                'measured': (measured_du, measured_dv),
                'predicted': (pred_du, pred_dv),
                'actual_delta': actual_delta,
            })

        if not measured_du_list:
            result.status = "no_data"
            result.issue = "All samples failed"
            return result

        result.measured_du = np.mean(measured_du_list)
        result.measured_dv = np.mean(measured_dv_list)
        result.predicted_du = np.mean(pred_du_list)
        result.predicted_dv = np.mean(pred_dv_list)
        result.delta_deg = delta_deg

        measured_vec = np.array([result.measured_du, result.measured_dv])
        predicted_vec = np.array([result.predicted_du, result.predicted_dv])
        measured_norm = np.linalg.norm(measured_vec)
        predicted_norm = np.linalg.norm(predicted_vec)

        if measured_norm > 0.1 and predicted_norm > 0.1:
            result.direction_score = np.dot(measured_vec, predicted_vec) / \
                                     (measured_norm * predicted_norm)
            result.magnitude_ratio = measured_norm / predicted_norm
        elif measured_norm < 0.1:
            result.direction_score = 0.0
            result.magnitude_ratio = 0.0
            result.status = "no_effect"
            result.issue = "joint has no effect on camera"
        else:
            result.direction_score = 0.0
            result.magnitude_ratio = float('inf')
            result.status = "urdf_zero"
            result.issue = "URDF predicts zero movement"

        if result.status == "untested":
            result.status, result.issue = self._diagnose(result)

        return result

    def _predict_pixel_movement(self, joints: np.ndarray, joint_idx: int,
                                 delta_deg: float) -> Optional[Tuple[float, float]]:
        """URDF预测: 扰动 joint_idx 后 tag 的像素位移 (per degree)"""
        if self._tag_world_pos is None:
            return None

        T_world_cam_base = self._get_camera_pose_world(joints)
        if T_world_cam_base is None:
            return None
        pixel_base = self._project(T_world_cam_base, self._tag_world_pos)
        if pixel_base is None:
            return None

        perturbed = joints.copy().astype(float)
        perturbed[joint_idx] += delta_deg
        T_world_cam_pert = self._get_camera_pose_world(perturbed)
        if T_world_cam_pert is None:
            return None
        pixel_pert = self._project(T_world_cam_pert, self._tag_world_pos)
        if pixel_pert is None:
            return None

        du = (pixel_pert[0] - pixel_base[0]) / delta_deg
        dv = (pixel_pert[1] - pixel_base[1]) / delta_deg
        return du, dv

    def _diagnose(self, result: JointValidationResult) -> Tuple[str, str]:
        """诊断关节问题"""
        ds = result.direction_score
        mr = result.magnitude_ratio

        if ds > 0.85:
            if 0.7 < mr < 1.4:
                return "ok", ""
            elif mr < 0.5:
                return "link_too_long", f"URDF link ~{1/mr:.1f}x too long"
            elif mr > 2.0:
                return "link_too_short", f"URDF link ~{mr:.1f}x too short"
            else:
                return "link_scale", f"link scale factor≈{mr:.2f}"
        elif ds < -0.85:
            return "axis_reversed", "axis direction REVERSED — negate <axis xyz>"
        elif -0.5 < ds < 0.5:
            return "axis_misaligned", f"axis direction wrong (cos={ds:.2f})"
        else:
            return "partial", f"direction partially off (cos={ds:.2f})"

    # ==================== 工具方法 ====================

    def _ensure_tag_visible(self) -> bool:
        for _ in range(5):
            frame = self.camera.read()
            if frame is not None:
                tags = self.tag_detector.detect(frame)
                if tags:
                    return True
            time.sleep(0.1)
        return False

    def _detect_tag(self) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        frame = self.camera.read()
        if frame is None:
            return None, None
        for _ in range(2):
            tags = self.tag_detector.detect(frame)
            if tags:
                tag = tags[0]
                return tag['center'], self.tag_detector.estimate_depth_mm(tag, self.fx)
        return None, None

    def _estimate_tag_world(self, joints, pixel, depth_mm):
        T_world_cam = self._get_camera_pose_world(joints)
        if T_world_cam is None:
            return None
        depth_m = depth_mm / 1000.0
        x_cam = (pixel[0] - self.cx) * depth_m / self.fx
        y_cam = (pixel[1] - self.cy) * depth_m / self.fy
        P_cam = np.array([x_cam, y_cam, depth_m, 1.0])
        return (T_world_cam @ P_cam)[:3]

    def _get_camera_pose_world(self, joints):
        try:
            ee_pose = self.fk.compute(joints)
            return ee_pose.transform_matrix @ self.T_flange_cam
        except Exception:
            return None

    def _project(self, T_world_cam, P_world):
        T_cam_world = np.linalg.inv(T_world_cam)
        P_cam = T_cam_world @ np.append(P_world, 1.0)
        if P_cam[2] <= 0.001:
            return None
        u = self.fx * P_cam[0] / P_cam[2] + self.cx
        v = self.fy * P_cam[1] / P_cam[2] + self.cy
        return np.array([u, v])

    # ==================== 报告 ====================

    def print_report(self):
        """打印验证报告 + 修正建议"""
        print(f"\n{'='*75}")
        print(f"URDF 验证报告 — {self.arm.upper()} ARM")
        print(f"{'='*75}")
        header = f"{'Joint':<20} {'Status':<18} {'Dir':<8} {'Mag':<8} {'Issue'}"
        print(header)
        print(f"{'-'*75}")

        ok_count = 0
        axis_problems = []
        link_problems = []

        for jidx in sorted(self.results.keys()):
            r = self.results[jidx]
            ds_str = f"{r.direction_score:.3f}" if abs(r.direction_score) > 0.01 else "N/A"
            mr_str = f"{r.magnitude_ratio:.2f}" if 0 < r.magnitude_ratio < 100 else "N/A"
            name = JOINT_NAMES.get(jidx, f"j{jidx}")
            print(f"{name:<20} {r.status:<18} {ds_str:<8} {mr_str:<8} {r.issue}")

            if r.status == "ok":
                ok_count += 1
            elif r.status.startswith("axis_"):
                axis_problems.append(r)
            elif r.status.startswith("link_"):
                link_problems.append(r)

        print(f"{'-'*75}")
        total = len(self.results)
        print(f"  OK: {ok_count}/{total}  "
              f"轴问题: {len(axis_problems)}  "
              f"长度问题: {len(link_problems)}")

        # 轴问题修正
        if axis_problems:
            print(f"\n── 旋转轴修正 (safe, 可自动) ──")
            for r in axis_problems:
                name = JOINT_NAMES.get(r.joint_idx, f"j{r.joint_idx}")
                if r.status == "axis_reversed":
                    print(f"  {name}: <axis xyz> 取反")
                elif r.status == "axis_misaligned":
                    print(f"  {name}: 检查 <origin rpy> 和 <axis xyz> — 方向偏离 {r.direction_score:.2f}")

        # 长度问题修正
        if link_problems:
            print(f"\n── Link 长度修正 (需确认, 可批量) ──")
            # 收集方向ok的关节的magnitude_ratio, 计算中位数作为全局缩放建议
            good_dir_ratios = []
            for jidx, r in self.results.items():
                if r.status in ("ok", "link_too_long", "link_too_short", "link_scale"):
                    if 0.1 < r.magnitude_ratio < 10:
                        good_dir_ratios.append(r.magnitude_ratio)

            median_ratio = np.median(good_dir_ratios) if good_dir_ratios else 1.0

            for r in link_problems:
                name = JOINT_NAMES.get(r.joint_idx, f"j{r.joint_idx}")
                print(f"  {name}: scale={r.magnitude_ratio:.2f} "
                      f"(URDF link 需要缩放 ~{1/r.magnitude_ratio:.2f}x)")

            if len(good_dir_ratios) >= 2:
                print(f"\n  方向正确的关节 median scale = {median_ratio:.3f}")
                print(f"  → 建议将所有 link <origin xyz> 缩放 {median_ratio:.3f}x")
                if not (0.9 < median_ratio < 1.1):
                    print(f"  → 全局缩放: 整个手臂长度 × {median_ratio:.3f}")

        # 交叉验证提示 (j14被双臂共享)
        if 14 in self.results:
            r14 = self.results[14]
            if r14.status != "ok":
                print(f"\n── j14 (trunk) 交叉验证 ──")
                print(f"  trunk_joint_1 左右臂共享, 建议用另一只手臂验证后对比")

        # 修正建议
        if axis_problems or link_problems:
            print(f"\n{'='*75}")
            print("修正操作:")
            print(f"{'='*75}")
            if axis_problems:
                print(f"  1. 修正旋转轴:       validator.fix_axes()")
            if link_problems:
                print(f"  2. 修正 link 长度:   validator.fix_link_lengths()")
            print(f"  3. 一键修正全部:     validator.auto_fix()")
            print(f"     (自动备份 → 修改 URDF → 请重新验证)")
            if self.urdf_path:
                print(f"\n  修正对象: {self.urdf_path}")
            else:
                print(f"\n  ⚠ 未设置 urdf_path, 无法自动修正")

    # ==================== 自动修正 ====================

    def _backup_urdf(self, path: Path) -> Path:
        """备份 URDF 文件, 返回备份路径"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".bak.{timestamp}")
        shutil.copy2(path, backup_path)
        self._backup_path = str(backup_path)
        print(f"✓ 已备份: {backup_path}")
        return backup_path

    def fix_axes(self, urdf_path: str = None):
        """
        修正旋转轴方向 (axis_reversed 关节)。

        对所有 direction_score < -0.85 的关节,
        将 URDF 中 <axis xyz="x y z"/> 取反。
        """
        path = Path(urdf_path or self.urdf_path)
        if not path or not path.exists():
            print("✗ URDF 文件不存在")
            return

        reversed_joints = {jidx: r for jidx, r in self.results.items()
                          if r.status == "axis_reversed"}
        if not reversed_joints:
            print("  没有需要修正的轴方向")
            return

        self._backup_urdf(path)
        tree = ET.parse(str(path))
        root = tree.getroot()

        # 建立 joint name → URDF joint element 的映射
        urdf_joints = {}
        for elem in root.iter('joint'):
            name_elem = elem.find('name')
            if name_elem is not None:
                urdf_joints[name_elem.text] = elem

        fixed = 0
        for jidx, r in reversed_joints.items():
            jname = JOINT_NAMES.get(jidx)
            if jname not in urdf_joints:
                print(f"  ⚠ {jname} 不在 URDF 中, 跳过")
                continue

            axis_elem = urdf_joints[jname].find('axis')
            if axis_elem is None:
                print(f"  ⚠ {jname} 没有 <axis> 元素, 跳过")
                continue

            xyz_str = axis_elem.get('xyz', '0 0 1')
            parts = xyz_str.split()
            if len(parts) == 3:
                flipped = f"{-float(parts[0])} {-float(parts[1])} {-float(parts[2])}"
                axis_elem.set('xyz', flipped)
                print(f"  ✓ {jname}: axis {xyz_str} → {flipped}")
                fixed += 1
                self._fixes_applied.append(f"axis_flip:{jname}")

        tree.write(str(path), xml_declaration=True, encoding='UTF-8')
        print(f"  ✓ 已修正 {fixed} 个关节轴方向")

    def fix_link_lengths(self, urdf_path: str = None):
        """
        修正 link 长度。

        收集方向正确关节的 magnitude_ratio, 取中位数作为全局缩放因子。
        将 kinematic chain 上所有 link 的 <origin xyz> 乘以该因子。

        原理:
          对于旋转关节, pixel/deg ∝ distance(joint_axis → camera)。
          如果 magnitude_ratio=0.7, 说明 URDF 的 link 比真实长 ~1/0.7=1.43x。
          将所有 link 的 xyz 偏移缩放 0.7x 来修正。
        """
        path = Path(urdf_path or self.urdf_path)
        if not path or not path.exists():
            print("✗ URDF 文件不存在")
            return

        # 只使用方向正确的关节 (axis 方向对才有意义的 magnitude_ratio)
        good_ratios = []
        for jidx, r in self.results.items():
            if r.direction_score > 0.7 and 0.1 < r.magnitude_ratio < 10:
                good_ratios.append(r.magnitude_ratio)

        if not good_ratios:
            print("  ✗ 没有方向正确的关节, 无法确定缩放因子 (先修正轴方向)")
            return

        scale = float(np.median(good_ratios))
        if 0.95 < scale < 1.05:
            print(f"  link 长度无需修正 (median scale={scale:.3f})")
            return

        # 是否仅缩放特定 arm 的 link
        if self.arm == "right":
            link_prefix = "right_"
        else:
            link_prefix = "left_"

        self._backup_urdf(path)
        tree = ET.parse(str(path))
        root = tree.getroot()
        fixed = 0

        for link_elem in root.iter('link'):
            name_elem = link_elem.find('name')
            if name_elem is None:
                continue
            # 只缩放对应手臂的 link
            if not name_elem.text.startswith(link_prefix):
                continue

            for visual in link_elem.findall('visual'):
                origin = visual.find('origin')
                if origin is not None and origin.get('xyz'):
                    xyz_str = origin.get('xyz')
                    parts = [float(x) for x in xyz_str.split()]
                    if len(parts) == 3:
                        scaled = f"{parts[0]*scale:.6f} {parts[1]*scale:.6f} {parts[2]*scale:.6f}"
                        origin.set('xyz', scaled)

            for collision in link_elem.findall('collision'):
                origin = collision.find('origin')
                if origin is not None and origin.get('xyz'):
                    xyz_str = origin.get('xyz')
                    parts = [float(x) for x in xyz_str.split()]
                    if len(parts) == 3:
                        scaled = f"{parts[0]*scale:.6f} {parts[1]*scale:.6f} {parts[2]*scale:.6f}"
                        origin.set('xyz', scaled)

            # 同时缩放 joint origin (这个更重要!)
            # joint 的 <origin> 定义了 child link 相对于 parent 的偏移

        # 缩放 joint origin
        for joint_elem in root.iter('joint'):
            name_elem = joint_elem.find('name')
            if name_elem is None:
                continue
            jname = name_elem.text

            # 判断这个 joint 是否属于当前手臂
            is_our_arm = False
            for jidx, urdf_name in JOINT_NAMES.items():
                if urdf_name == jname and jidx in self.joint_indices:
                    is_our_arm = True
                    break
            # trunk 也缩放
            if jname.startswith("trunk"):
                is_our_arm = True

            if is_our_arm:
                origin = joint_elem.find('origin')
                if origin is not None and origin.get('xyz'):
                    xyz_str = origin.get('xyz')
                    parts = [float(x) for x in xyz_str.split()]
                    if len(parts) == 3:
                        scaled = f"{parts[0]*scale:.6f} {parts[1]*scale:.6f} {parts[2]*scale:.6f}"
                        origin.set('xyz', scaled)
                        fixed += 1

        tree.write(str(path), xml_declaration=True, encoding='UTF-8')
        self._fixes_applied.append(f"link_scale:{scale:.4f}")
        print(f"  ✓ 已缩放 {fixed} 个 joint origin × {scale:.4f}")
        print(f"  (方向正确的关节 median magnitude_ratio = {scale:.4f})")
        print(f"  ⚠ 注意: 这只是近似修正, trunk 和 {self.arm} arm 的 joint origin 都被缩放了")

    def auto_fix(self, urdf_path: str = None):
        """
        一键修正: 先修轴方向, 再修 link 长度。
        自动备份, 打印修改摘要。
        """
        path = urdf_path or self.urdf_path
        print(f"\n{'='*60}")
        print("URDF 自动修正")
        print(f"{'='*60}")
        print(f"  文件: {path}")

        self._fixes_applied = []

        # 1. 修轴方向
        self.fix_axes(path)

        # 2. 修 link 长度
        self.fix_link_lengths(path)

        # 摘要
        print(f"\n{'='*60}")
        print("修正摘要")
        print(f"{'='*60}")
        for fix in self._fixes_applied:
            print(f"  - {fix}")
        if self._backup_path:
            print(f"\n  备份: {self._backup_path}")
        print(f"\n  下一步: 重新进入 FK-IBVS → 按 V 验证修正效果")

    # ==================== 交叉验证 ====================

    @staticmethod
    def cross_validate(left_results: Dict[int, JointValidationResult],
                       right_results: Dict[int, JointValidationResult]):
        """交叉验证左右臂共享关节 (j14/trunk) 的结果一致性"""
        print(f"\n{'='*60}")
        print("交叉验证: j14 (trunk_joint_1) 左右臂对比")
        print(f"{'='*60}")

        r14_l = left_results.get(14)
        r14_r = right_results.get(14)

        if r14_l is None or r14_r is None:
            print("  ⚠ 缺少一侧的 j14 数据")
            return

        print(f"  Left:  status={r14_l.status}  dir={r14_l.direction_score:.3f}  "
              f"mag={r14_l.magnitude_ratio:.2f}")
        print(f"  Right: status={r14_r.status}  dir={r14_r.direction_score:.3f}  "
              f"mag={r14_r.magnitude_ratio:.2f}")

        if r14_l.status == r14_r.status:
            if r14_l.status == "axis_reversed" and r14_r.status == "axis_reversed":
                print(f"\n  ✓ 双侧一致: trunk_joint_1 轴方向需要反转 → 高置信度")
            elif r14_l.status == "ok" and r14_r.status == "ok":
                print(f"\n  ✓ 双侧一致: trunk_joint_1 OK")
            else:
                print(f"\n  ✓ 双侧一致: {r14_l.status}")
        else:
            print(f"\n  ⚠ 双侧不一致! 左={r14_l.status} vs 右={r14_r.status}")
            print(f"     可能原因: 手眼标定精度 / 相机安装 / FK 参数不同")
            print(f"     建议: 先修正一致的轴方向, 不一致的保持原样再验证")

        # 双侧 magnitude_ratio
        if 0.1 < r14_l.magnitude_ratio < 10 and 0.1 < r14_r.magnitude_ratio < 10:
            avg_mag = (r14_l.magnitude_ratio + r14_r.magnitude_ratio) / 2
            print(f"\n  j14 magnitude_ratio 均值: {avg_mag:.3f}")
            if not (0.9 < avg_mag < 1.1):
                print(f"  → trunk link 可能需要缩放 ~{avg_mag:.3f}x")


# ==================== 便捷入口 ====================

def create_validator(arm: str = "right",
                     urdf_path: str = None,
                     controller=None,
                     camera=None,
                     T_flange_cam: np.ndarray = None,
                     camera_matrix: np.ndarray = None) -> Optional[URDFValidator]:
    """从标准路径创建验证器"""
    from precision_place.calibration.forward_kinematics import create_fk_from_urdf

    if T_flange_cam is None:
        he_path = Path(__file__).parent.parent / "hand_eye_extrinsic.yaml"
        if he_path.exists():
            with open(he_path, 'r') as f:
                he_data = yaml.safe_load(f)
            T_flange_cam = np.array(he_data['extrinsic_matrix']['data']).reshape(4, 4)

    if camera_matrix is None:
        int_path = Path(__file__).parent.parent / "camera_intrinsics.yaml"
        if int_path.exists():
            with open(int_path, 'r') as f:
                data = yaml.safe_load(f)
            camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        else:
            camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    if T_flange_cam is None:
        print("✗ 需要手眼标定文件")
        return None

    if urdf_path:
        fk = create_fk_from_urdf(urdf_path, arm)
    else:
        print("✗ 需要 URDF 文件")
        return None

    return URDFValidator(fk, T_flange_cam, camera_matrix, controller, camera,
                         urdf_path, arm=arm)


def main():
    """命令行 — 仅验证 URDF 能否加载, 不移动机器人"""
    import argparse
    parser = argparse.ArgumentParser(description="URDF 逐关节验证")
    parser.add_argument("--urdf", type=str, required=True, help="URDF 文件路径")
    parser.add_argument("--arm", type=str, default="right", help="手臂")
    args = parser.parse_args()

    from precision_place.calibration.forward_kinematics import create_fk_from_urdf

    print(f"加载 URDF: {args.urdf}")
    fk = create_fk_from_urdf(args.urdf, args.arm)
    print(f"✓ FK 求解器已创建")

    joints = np.zeros(16)
    if args.arm == "right":
        joints[0:6] = [-70, 0, 0, -25, -90, -90]
        joints[7:13] = [70, 0, 0, 25, -90, 90]

    try:
        pose = fk.compute(joints)
        print(f"  FK 测试通过")
        print(f"  末端位置: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) m")
        print(f"\nURDF 加载正常, 可在机器人端验证:")
        print(f"  validator = create_validator(urdf_path='{args.urdf}', arm='{args.arm}', ...)")
        print(f"  validator.validate_all()")
        print(f"  validator.auto_fix()  # 自动修正")
    except Exception as e:
        print(f"✗ FK 测试失败: {e}")


if __name__ == "__main__":
    main()
