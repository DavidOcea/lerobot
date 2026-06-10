#!/usr/bin/env python3
"""
URDF 逐关节验证 + 自动修正脚本

原理:
  以真实机器人为 ground truth, 逐一测试 URDF 中每个关节的 FK 预测精度。
  对每个关节: 单独转动 Δθ° → 观测 AprilTag 像素位移 → 与 URDF 预测对比。

  可诊断的问题:
    - 关节旋转轴方向反了 (URDF <axis> 需要取反)
    - 关节对相机运动贡献比例不对 (link length / gear ratio 问题)
    - 关节旋转轴方向倾斜 (轴不平移)

用法:
  # 在 run.py 中集成 (推荐, 机器人已连接):
  from precision_place.calibration.validate_urdf import URDFValidator
  validator = URDFValidator(fk, T_flange_cam, camera_matrix, controller, camera)
  validator.validate_all()

  # 也可以在 FK-IBVS 对齐界面中按 V 键触发
"""

import os
import copy
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import yaml


class JointValidationResult:
    """单个关节的验证结果"""
    def __init__(self, joint_idx: int):
        self.joint_idx = joint_idx
        self.delta_deg: float = 0.0
        self.pixel_before: Optional[Tuple[float, float]] = None
        self.pixel_after: Optional[Tuple[float, float]] = None
        self.measured_du: float = 0.0  # 实测像素位移
        self.measured_dv: float = 0.0
        self.predicted_du: float = 0.0  # URDF预测像素位移
        self.predicted_dv: float = 0.0
        self.direction_score: float = 0.0  # cos(实测, 预测), 1=完美, -1=反向
        self.magnitude_ratio: float = 0.0  # |实测|/|预测|, 1=完美
        self.status: str = "untested"
        self.issue: str = ""


class URDFValidator:
    """URDF 逐关节验证器"""

    def __init__(self,
                 fk_solver,                       # ForwardKinematics 实例 (URDF)
                 T_flange_cam: np.ndarray,        # 4x4 手眼矩阵
                 camera_matrix: np.ndarray,       # 3x3 相机内参
                 controller,                      # 机器人控制器
                 camera,                          # 相机对象 (camera.read() → BGR)
                 urdf_path: str = None,
                 joint_indices: List[int] = None):
        """
        Args:
            fk_solver: 基于 URDF 的 FK 求解器
            T_flange_cam: 手眼矩阵
            camera_matrix: 相机内参
            controller: 机器人控制器 (.get_joint_states(), ._smooth_move_all_joints())
            camera: 相机
            urdf_path: URDF 文件路径 (用于自动修正)
            joint_indices: 要测试的关节索引, 默认 j7-j12, j14
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
        self.joint_indices = joint_indices or [7, 8, 9, 10, 11, 12, 14]

        # AprilTag 检测器
        from precision_place.calibration.simple_ibvs import AprilTagDetector
        self.tag_detector = AprilTagDetector(tag_family="tag36h11")

        self.results: Dict[int, JointValidationResult] = {}
        self._tag_world_pos: Optional[np.ndarray] = None  # tag 在世界坐标系中的位置 (仅首次估计)

    # ==================== 核心验证逻辑 ====================

    def validate_all(self, delta_deg: float = 6.0, n_samples: int = 2,
                     return_after: bool = True):
        """
        逐关节验证 URDF。

        Args:
            delta_deg: 每个关节的测试转动量 (度)
            n_samples: 每个关节采样次数 (取平均)
            return_after: 测试后是否回到初始位置
        """
        print(f"\n{'='*60}")
        print(f"URDF 逐关节验证")
        print(f"{'='*60}")
        print(f"  测试关节: {self.joint_indices}")
        print(f"  转动步长: {delta_deg}° × {n_samples}次")
        print(f"  先决条件: AprilTag 必须在相机视野内且保持静止")
        print(f"{'='*60}")

        # 确保 AprilTag 可见
        initial_joints = self.controller.get_joint_states()
        if initial_joints is None:
            print("✗ 无法获取关节状态")
            return

        if not self._ensure_tag_visible():
            print("✗ AprilTag 不可见, 请放置标签到相机视野内后重试")
            return

        # 记住初始关节位置 (测试后恢复)
        joints_start = initial_joints.copy()

        for joint_idx in self.joint_indices:
            result = self._validate_single_joint(joint_idx, delta_deg, n_samples)
            self.results[joint_idx] = result

        # 可选: 恢复初始位置
        if return_after:
            print("\n  恢复初始位置...")
            try:
                self.controller._smooth_move_all_joints(joints_start, steps=10)
                time.sleep(0.3)
            except Exception:
                pass

        # 打印报告
        self.print_report()

    def _validate_single_joint(self, joint_idx: int, delta_deg: float,
                                n_samples: int) -> JointValidationResult:
        """验证单个关节"""
        result = JointValidationResult(joint_idx)

        # 确保 tag 在世界坐标系中的位置已估计
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
            # 1. 记录转动前状态
            joints_before = self.controller.get_joint_states()
            if joints_before is None:
                continue
            pixel_before, _ = self._detect_tag()
            if pixel_before is None:
                continue

            # 2. URDF 预测: FK扰动 → 投影tag → 计算预期像素位移
            pred_du, pred_dv = self._predict_pixel_movement(
                joints_before, joint_idx, delta_deg)
            if pred_du is None:
                continue

            # 3. 实际转动关节
            target = joints_before.copy()
            target[joint_idx] += delta_deg
            self.controller._smooth_move_all_joints(target, steps=10)
            time.sleep(0.3)

            # 4. 记录转动后状态
            joints_after = self.controller.get_joint_states()
            pixel_after, _ = self._detect_tag()

            # 5. 转回来 (方便下一轮采样从相同初始位置开始)
            self.controller._smooth_move_all_joints(joints_before, steps=10)
            time.sleep(0.2)

            if pixel_after is None:
                continue

            actual_delta = joints_after[joint_idx] - joints_before[joint_idx]
            if abs(actual_delta) < 0.1:
                continue

            # 实测像素位移 (归一化到 per degree)
            measured_du = (pixel_after[0] - pixel_before[0]) / actual_delta
            measured_dv = (pixel_after[1] - pixel_before[1]) / actual_delta

            measured_du_list.append(measured_du)
            measured_dv_list.append(measured_dv)
            pred_du_list.append(pred_du)
            pred_dv_list.append(pred_dv)

        if not measured_du_list:
            result.status = "no_data"
            result.issue = "All samples failed"
            return result

        # 取平均
        result.measured_du = np.mean(measured_du_list)
        result.measured_dv = np.mean(measured_dv_list)
        result.predicted_du = np.mean(pred_du_list)
        result.predicted_dv = np.mean(pred_dv_list)
        result.delta_deg = delta_deg

        # 计算方向匹配度
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
            result.issue = "joint_has_no_effect"
            result.status = "no_effect"
        else:
            result.direction_score = 0.0
            result.magnitude_ratio = float('inf')
            result.issue = "urdf_predicts_zero"
            result.status = "urdf_zero"

        # 诊断
        if result.status == "untested":
            result.status, result.issue = self._diagnose(result)

        return result

    def _predict_pixel_movement(self, joints: np.ndarray, joint_idx: int,
                                 delta_deg: float) -> Optional[Tuple[float, float]]:
        """URDF预测: 扰动关节 joint_idx 后 tag 的像素位移 (per degree)"""
        if self._tag_world_pos is None:
            return None

        # 基准: 当前关节 → 相机位姿 → 投影tag
        T_world_cam_base = self._get_camera_pose_world(joints)
        if T_world_cam_base is None:
            return None
        pixel_base = self._project(T_world_cam_base, self._tag_world_pos)
        if pixel_base is None:
            return None

        # 扰动: +delta_deg
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
        """诊断关节 URDF 问题"""
        ds = result.direction_score
        mr = result.magnitude_ratio

        if ds > 0.85:
            if 0.7 < mr < 1.4:
                return "ok", ""
            elif mr < 0.5:
                return "weak", f"URDF overestimates effect ({mr:.2f}x)"
            elif mr > 2.0:
                return "strong", f"URDF underestimates effect ({mr:.2f}x)"
            else:
                return "magnitude_off", f"magnitude ratio={mr:.2f}"
        elif ds < -0.85:
            return "axis_reversed", "URDF axis direction is REVERSED — negate <axis>"
        elif -0.5 < ds < 0.5:
            return "axis_misaligned", f"URDF axis orientation wrong (cos={ds:.2f})"
        else:
            return "partial", f"direction partially off (cos={ds:.2f})"

    # ==================== 工具方法 ====================

    def _ensure_tag_visible(self) -> bool:
        """确保 AprilTag 可见"""
        for _ in range(5):
            frame = self.camera.read()
            if frame is not None:
                tags = self.tag_detector.detect(frame)
                if tags:
                    return True
            time.sleep(0.1)
        return False

    def _detect_tag(self) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """检测 AprilTag 中心像素位置和深度"""
        frame = self.camera.read()
        if frame is None:
            return None, None
        for _ in range(2):  # 尝试2次
            tags = self.tag_detector.detect(frame)
            if tags:
                tag = tags[0]
                center = tag['center']
                depth = self.tag_detector.estimate_depth_mm(tag, self.fx)
                return center, depth
        return None, None

    def _estimate_tag_world(self, joints: np.ndarray,
                             pixel: Tuple[float, float],
                             depth_mm: float) -> Optional[np.ndarray]:
        """从像素坐标 + 深度反投影 tag 到世界坐标系"""
        T_world_cam = self._get_camera_pose_world(joints)
        if T_world_cam is None:
            return None

        depth_m = depth_mm / 1000.0
        x_cam = (pixel[0] - self.cx) * depth_m / self.fx
        y_cam = (pixel[1] - self.cy) * depth_m / self.fy
        z_cam = depth_m
        P_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        P_world = T_world_cam @ P_cam
        return P_world[:3]

    def _get_camera_pose_world(self, joints: np.ndarray) -> Optional[np.ndarray]:
        """计算相机在世界坐标系中的位姿 T_world_cam (4x4)"""
        try:
            ee_pose = self.fk.compute(joints)
            T_world_flange = ee_pose.transform_matrix
            T_world_cam = T_world_flange @ self.T_flange_cam
            return T_world_cam
        except Exception:
            return None

    def _project(self, T_world_cam: np.ndarray,
                 P_world: np.ndarray) -> Optional[np.ndarray]:
        """将世界坐标系中的点投影到像素坐标"""
        T_cam_world = np.linalg.inv(T_world_cam)
        P_cam = T_cam_world @ np.append(P_world, 1.0)
        if P_cam[2] <= 0.001:  # 在相机后方
            return None
        u = self.fx * P_cam[0] / P_cam[2] + self.cx
        v = self.fy * P_cam[1] / P_cam[2] + self.cy
        return np.array([u, v])

    # ==================== 报告 ====================

    def print_report(self):
        """打印验证报告 + URDF修正建议"""
        print(f"\n{'='*70}")
        print("URDF 验证报告")
        print(f"{'='*70}")
        print(f"{'Joint':<8} {'Status':<18} {'Direction':<10} {'MagRatio':<10} {'Issue'}")
        print(f"{'-'*70}")

        ok_count = 0
        problem_joints = []

        for jidx in sorted(self.results.keys()):
            r = self.results[jidx]
            ds_str = f"{r.direction_score:.3f}" if abs(r.direction_score) > 0.01 else "N/A"
            mr_str = f"{r.magnitude_ratio:.2f}" if 0 < r.magnitude_ratio < 100 else "N/A"
            print(f"j{jidx:<7} {r.status:<18} {ds_str:<10} {mr_str:<10} {r.issue}")

            if r.status == "ok":
                ok_count += 1
            elif r.status not in ("untested", "no_data"):
                problem_joints.append(r)

        print(f"{'-'*70}")
        print(f"  OK: {ok_count}/{len(self.results)}  问题: {len(problem_joints)}")

        if problem_joints:
            print(f"\n{'='*70}")
            print("URDF 修正建议")
            print(f"{'='*70}")
            corrections = self._suggest_urdf_corrections(problem_joints)
            for c in corrections:
                print(f"  {c}")
            print()

            if self.urdf_path:
                print(f"\n  是否自动修正 URDF? 运行:")
                print(f"    validator.apply_corrections('{self.urdf_path}')")
            else:
                print(f"  (未设置 urdf_path, 无法自动修正)")

    def _suggest_urdf_corrections(self, problem_joints: List[JointValidationResult]) -> List[str]:
        """根据诊断结果生成 URDF 修正建议"""
        suggestions = []
        for r in problem_joints:
            joint_name = f"right_arm_joint_{(r.joint_idx - 7) + 1}"
            if r.joint_idx == 14:
                joint_name = "trunk_joint_1"

            if r.status == "axis_reversed":
                suggestions.append(
                    f"  j{r.joint_idx} ({joint_name}): "
                    f"反转 <axis> — URDF中 <axis xyz=\"...\"/> 需要取反"
                )
            elif r.status == "axis_misaligned":
                suggestions.append(
                    f"  j{r.joint_idx} ({joint_name}): "
                    f"旋转轴方向不对 — 检查 <origin rpy=\"...\"/> 和 <axis xyz=\"...\"/>"
                )
            elif r.status == "weak":
                suggestions.append(
                    f"  j{r.joint_idx} ({joint_name}): "
                    f"URDF过度预测 {r.magnitude_ratio:.2f}x — 检查 link 长度或 joint 类型"
                )
            elif r.status == "strong":
                suggestions.append(
                    f"  j{r.joint_idx} ({joint_name}): "
                    f"URDF预测不足 (实测是预测的 {r.magnitude_ratio:.1f}x) — link 可能太短"
                )
            elif r.status == "no_effect":
                suggestions.append(
                    f"  j{r.joint_idx} ({joint_name}): "
                    f"实际转动对相机无影响 — 可能不是 arm 关节, 不应在 joint_indices 中"
                )
        return suggestions

    # ==================== URDF 自动修正 ====================

    def apply_corrections(self, urdf_path: str = None, backup: bool = True):
        """
        根据验证结果自动修正 URDF 文件。
        目前支持: 反转关节旋转轴方向 (axis_reversed)

        Args:
            urdf_path: URDF 文件路径, 默认用 self.urdf_path
            backup: 是否备份原文件
        """
        path = urdf_path or self.urdf_path
        if not path:
            print("✗ 未设置 URDF 路径")
            return

        path = Path(path)
        if not path.exists():
            print(f"✗ URDF 文件不存在: {path}")
            return

        # 备份
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            import shutil
            shutil.copy2(path, backup_path)
            print(f"✓ 已备份: {backup_path}")

        # 解析 + 修正
        tree = ET.parse(str(path))
        root = tree.getroot()

        axis_corrections = {}
        for jidx, r in self.results.items():
            if r.status == "axis_reversed":
                # joint_idx → URDF joint name
                if jidx in range(7, 13):  # right arm joints
                    joint_num = (jidx - 7) + 1
                    joint_name = f"right_arm_joint_{joint_num}"
                elif jidx == 14:
                    joint_name = "trunk_joint_1"
                else:
                    continue
                axis_corrections[joint_name] = True

        fixed_count = 0
        for joint_elem in root.iter('joint'):
            name_elem = joint_elem.find('name')
            if name_elem is None:
                continue
            jname = name_elem.text
            if jname in axis_corrections:
                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    xyz_str = axis_elem.get('xyz', '0 0 1')
                    parts = xyz_str.split()
                    if len(parts) == 3:
                        flipped = f"{-float(parts[0])} {-float(parts[1])} {-float(parts[2])}"
                        axis_elem.set('xyz', flipped)
                        print(f"  ✓ {jname}: axis {xyz_str} → {flipped}")
                        fixed_count += 1

        tree.write(str(path), xml_declaration=True, encoding='UTF-8')
        print(f"\n✓ URDF 已修正 ({fixed_count} 个关节), 保存到: {path}")
        print(f"  请重新加载 FK 求解器并再次验证")


# ==================== 便捷入口 ====================

def create_validator(arm: str = "right",
                     urdf_path: str = None,
                     controller=None,
                     camera=None,
                     T_flange_cam: np.ndarray = None,
                     camera_matrix: np.ndarray = None) -> Optional[URDFValidator]:
    """
    从标准路径创建验证器 (简化构造)。

    Args:
        arm: "right" 或 "left"
        urdf_path: URDF 文件路径
        controller, camera: 机器人控制器和相机 (从 PrecisionPlaceSystem 传入)
        T_flange_cam, camera_matrix: 手眼矩阵和相机内参 (如不提供则从文件加载)
    """
    from precision_place.calibration.forward_kinematics import create_fk_from_urdf

    # 加载手眼
    if T_flange_cam is None:
        he_path = Path(__file__).parent.parent / "hand_eye_extrinsic.yaml"
        if he_path.exists():
            with open(he_path, 'r') as f:
                he_data = yaml.safe_load(f)
            T_flange_cam = np.array(he_data['extrinsic_matrix']['data']).reshape(4, 4)

    # 加载相机内参
    if camera_matrix is None:
        intrinsics_path = Path(__file__).parent.parent / "camera_intrinsics.yaml"
        if intrinsics_path.exists():
            with open(intrinsics_path, 'r') as f:
                data = yaml.safe_load(f)
            camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        else:
            camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    if T_flange_cam is None:
        print("✗ 需要手眼标定文件")
        return None

    # 创建 FK
    if urdf_path:
        fk = create_fk_from_urdf(urdf_path, arm)
    else:
        print("✗ 需要 URDF 文件")
        return None

    return URDFValidator(fk, T_flange_cam, camera_matrix, controller, camera, urdf_path)


# ==================== 命令行测试 ====================

def main():
    """命令行模式 — 仅测试已保存的数据, 不移动机器人"""
    import argparse
    parser = argparse.ArgumentParser(description="URDF 逐关节验证")
    parser.add_argument("--urdf", type=str, required=True, help="URDF 文件路径")
    parser.add_argument("--arm", type=str, default="right", help="手臂")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式: 仅检查 URDF 能否加载并计算 FK")
    args = parser.parse_args()

    from precision_place.calibration.forward_kinematics import create_fk_from_urdf

    print(f"加载 URDF: {args.urdf}")
    fk = create_fk_from_urdf(args.urdf, args.arm)
    print(f"✓ FK 求解器已创建")

    # 用零位测试 FK
    joints = np.zeros(16)
    if args.arm == "right":
        # 设置一个合理的右臂初始姿态
        joints[0:6] = [-70, 0, 0, -25, -90, -90]  # left arm随便设
        joints[7] = 70   # right j1
        joints[8] = 0    # right j2
        joints[9] = 0    # right j3
        joints[10] = 25  # right j4
        joints[11] = -90 # right j5
        joints[12] = 90  # right j6

    try:
        pose = fk.compute(joints)
        print(f"  FK 测试通过")
        print(f"  末端位置: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) m")
        print(f"\nURDF 加载正常, 可在机器人端运行完整验证:")
        print(f"  validator = create_validator(urdf_path='{args.urdf}', ...)")
        print(f"  validator.validate_all()")
    except Exception as e:
        print(f"✗ FK 测试失败: {e}")


if __name__ == "__main__":
    main()
