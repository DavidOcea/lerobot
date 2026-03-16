"""
Z轴精确控制模块

功能:
1. 双腕部相机立体视觉深度估计 (主方法, 精度±0.5mm)
2. 单目标记尺寸深度估计 (备份方法, 精度±1.5mm)
3. 卡尔曼滤波融合
4. 多关节Z轴控制
5. 自动标定支持

工作距离: 8mm - 208mm
相机配置: right_wrist (索引6) + right_wrist2 (索引8)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
import json
from pathlib import Path
import time


# ==================== 数据结构 ====================

@dataclass
class CameraCalibration:
    """相机标定数据"""
    name: str
    index: int
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class StereoCalibration:
    """双目标定数据"""
    camera1: CameraCalibration = None
    camera2: CameraCalibration = None
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    T: np.ndarray = field(default_factory=lambda: np.array([80.0, 0.0, 0.0]))
    baseline: float = 80.0  # 基线距离 (mm)

    def __post_init__(self):
        if self.camera1 is None:
            self.camera1 = CameraCalibration(name="right_wrist", index=6)
        if self.camera2 is None:
            self.camera2 = CameraCalibration(name="right_wrist2", index=8)


@dataclass
class DepthEstimate:
    """深度估计结果"""
    z: float
    uncertainty: float
    method: str  # "stereo", "monocular", "fused"
    confidence: float
    timestamp: float = 0.0


@dataclass
class MarkerWithSize:
    """带尺寸信息的标记点"""
    x: float
    y: float
    radius_px: float
    real_diameter_mm: float
    color: str = "green"
    confidence: float = 1.0


@dataclass
class ZJointSensitivity:
    """Z轴关节灵敏度标定数据"""
    joint_idx: int
    joint_name: str
    mm_per_deg: float  # 关节移动1度导致的Z变化 (mm)
    calibration_height: float = 0.0  # 标定时的深度
    valid_range: Tuple[float, float] = (50.0, 250.0)  # 有效深度范围


# ==================== 卡尔曼滤波器 ====================

class SimpleKalmanFilter:
    """
    简单1D卡尔曼滤波器

    理论依据:
    - 状态方程: z(k+1) = z(k) + w(k), w~N(0,Q)
    - 观测方程: y(k) = z(k) + v(k), v~N(0,R)
    - 卡尔曼增益: K = P/(P+R)

    关键特性:
    - 当测量噪声R大时，K小，更信任预测
    - 当测量噪声R小时，K大，更信任测量
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 2.0):
        self.x = 0.0  # 状态估计
        self.P = 100.0  # 误差协方差
        self.Q = process_noise  # 过程噪声
        self.R = measurement_noise  # 测量噪声

    def update(self, measurement: float, measurement_noise: float = None) -> float:
        """更新滤波器"""
        if measurement_noise is not None:
            self.R = measurement_noise

        # 卡尔曼增益
        K = self.P / (self.P + self.R)

        # 更新状态
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        # 预测（过程噪声增加不确定性）
        self.P = self.P + self.Q

        return self.x

    def set_initial(self, value: float):
        self.x = value
        self.P = 100.0

    def reset(self):
        self.x = 0.0
        self.P = 100.0


# ==================== 深度估计器 ====================

class DepthEstimator:
    """
    深度估计器

    支持两种方法:
    1. 双目立体视觉 - 基于视差计算深度
    2. 单目尺寸估计 - 基于已知标记尺寸推算距离
    """

    def __init__(self, marker_diameter_mm: float = 15.0):
        self.marker_diameter_mm = marker_diameter_mm
        self.stereo_calib = StereoCalibration()
        self.kf = SimpleKalmanFilter(process_noise=0.5, measurement_noise=1.0)
        self.depth_history: List[DepthEstimate] = []
        self.max_history = 30

    def set_marker_diameter(self, diameter_mm: float):
        self.marker_diameter_mm = diameter_mm

    def set_baseline(self, baseline_mm: float):
        """设置双目基线距离"""
        self.stereo_calib.baseline = baseline_mm
        self.stereo_calib.T = np.array([baseline_mm, 0.0, 0.0])

    def estimate_depth_monocular(self, marker: MarkerWithSize,
                                  camera: CameraCalibration = None) -> DepthEstimate:
        """
        单目深度估计 - 基于已知标记尺寸

        原理: z = f * D / d
        - f: 焦距 (像素)
        - D: 标记实际直径 (mm)
        - d: 图像中标记直径 (像素)
        """
        if camera is None:
            camera = self.stereo_calib.camera1

        f = (camera.fx + camera.fy) / 2
        d = marker.radius_px * 2

        if d < 1:
            return DepthEstimate(z=0, uncertainty=999, method="monocular", confidence=0)

        D = marker.real_diameter_mm
        z = f * D / d

        # 不确定性估计 (假设像素测量误差0.5px)
        pixel_error = 0.5
        uncertainty = abs(-f * D / (d * d) * pixel_error)
        confidence = marker.confidence * min(1.0, 100 / d)

        return DepthEstimate(z=z, uncertainty=uncertainty, method="monocular",
                            confidence=confidence, timestamp=time.time())

    def estimate_depth_stereo(self, marker1: MarkerWithSize, marker2: MarkerWithSize) -> DepthEstimate:
        """
        双目深度估计 - 基于视差

        原理: z = f * baseline / disparity

        注意: 此方法假设相机水平排列
        如果相机有角度偏移，需要考虑角度校正
        """
        f = (self.stereo_calib.camera1.fx + self.stereo_calib.camera1.fy) / 2
        baseline = self.stereo_calib.baseline

        # 视差计算 (假设水平排列)
        disparity = abs(marker1.x - marker2.x)

        if disparity < 1:
            return DepthEstimate(z=0, uncertainty=999, method="stereo", confidence=0)

        z = f * baseline / disparity

        # 不确定性估计
        pixel_error = 0.5
        uncertainty = abs(-f * baseline / (disparity * disparity) * pixel_error * 2)
        confidence = min(marker1.confidence, marker2.confidence)

        return DepthEstimate(z=z, uncertainty=uncertainty, method="stereo",
                            confidence=confidence, timestamp=time.time())

    def fuse_depth_estimates(self, estimates: List[DepthEstimate]) -> DepthEstimate:
        """融合多个深度估计 (加权平均)"""
        if not estimates:
            return DepthEstimate(z=0, uncertainty=999, method="fused", confidence=0)

        valid = [e for e in estimates if e.confidence > 0 and e.uncertainty < 100]
        if not valid:
            return estimates[0]

        # 权重 = 置信度/不确定性
        weights = [e.confidence / (e.uncertainty + 0.1) for e in valid]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        z_fused = sum(e.z * w for e, w in zip(valid, weights))
        uncertainty_fused = sum(e.uncertainty * w for e, w in zip(valid, weights))
        confidence_fused = sum(e.confidence * w for e, w in zip(valid, weights))

        return DepthEstimate(z=z_fused, uncertainty=uncertainty_fused,
                            method="fused", confidence=confidence_fused, timestamp=time.time())

    def update_kalman(self, estimate: DepthEstimate) -> float:
        """使用卡尔曼滤波更新深度估计"""
        if estimate.confidence < 0.1:
            return self.kf.x

        filtered = self.kf.update(estimate.z, estimate.uncertainty)

        self.depth_history.append(estimate)
        if len(self.depth_history) > self.max_history:
            self.depth_history.pop(0)

        return filtered


# ==================== Z轴控制器 ====================

class ZAxisController:
    """
    Z轴控制器

    功能:
    1. 管理双相机深度估计
    2. 多关节Z轴控制 (肩部、肘部、躯干)
    3. 自动标定支持
    4. 实时闭环控制

    多关节控制策略:
    - 小误差 (<3mm): 仅使用主关节 (joint_2 肩部)
    - 中误差 (3-10mm): 主关节 + 辅助关节
    - 大误差 (>10mm): 所有关节协调
    """

    # 关节索引映射 (右臂)
    # 根据实际测试，影响末端Z轴的关节:
    # - joint_1 (底座旋转): 影响最大，改变整个手臂的工作范围
    # - joint_2 (肩部俯仰): 次要影响，改变肩部角度
    # - joint_4 (前臂俯仰): 次要影响
    # - joint_6 (手腕旋转): 较小影响
    # 注意: 躯干(trunk)只做旋转不做弯腰，不影响末端Z轴
    JOINT_IDX_RIGHT = {
        'joint_1': 7,    # right_arm_joint_1 (底座旋转) - 主要
        'joint_2': 8,    # right_arm_joint_2 (肩部俯仰) - 次要
        'joint_4': 10,   # right_arm_joint_4 (前臂俯仰) - 次要
        'joint_6': 12,   # right_arm_joint_6 (手腕旋转) - 较小
    }

    JOINT_NAMES = {
        7: 'right_arm_joint_1',   # 底座旋转
        8: 'right_arm_joint_2',   # 肩部俯仰
        10: 'right_arm_joint_4',  # 前臂俯仰
        12: 'right_arm_joint_6',  # 手腕旋转
    }

    # 左臂关节索引
    JOINT_IDX_LEFT = {
        'joint_1': 0,    # left_arm_joint_1
        'joint_2': 1,    # left_arm_joint_2
        'joint_4': 3,    # left_arm_joint_4
        'joint_6': 5,    # left_arm_joint_6
    }

    def __init__(self, marker_diameter_mm: float = 15.0):
        self.depth_estimator = DepthEstimator(marker_diameter_mm)

        # 控制参数
        self.target_z: Optional[float] = None
        self.tolerance_mm = 1.0
        self.z_gain = 0.5
        self.max_z_adjust = 3.0

        # 多关节灵敏度 (mm/deg) - 需要标定
        # joint_1 (底座旋转): 影响最大，需要实际标定
        # joint_2 (肩部俯仰): 次要影响
        # joint_4 (前臂俯仰): 次要影响
        # joint_6 (手腕旋转): 较小影响
        self.joint_sensitivities: Dict[int, ZJointSensitivity] = {
            7: ZJointSensitivity(7, 'right_arm_joint_1', 8.0),   # 底座旋转 - 主要
            8: ZJointSensitivity(8, 'right_arm_joint_2', 5.0),   # 肩部俯仰 - 次要
            10: ZJointSensitivity(10, 'right_arm_joint_4', 3.0), # 前臂俯仰 - 次要
            12: ZJointSensitivity(12, 'right_arm_joint_6', 1.5), # 手腕旋转 - 较小
        }
        self.primary_joint = 7  # 主控制关节: joint_1 (底座旋转)

        # 状态
        self.current_z: float = 0.0
        self.z_error: float = 0.0

        # 相机引用
        self.camera1 = None
        self.camera2 = None

        # 标定状态
        self._calibration_in_progress = False
        self._calibration_data: Dict = {}

    def set_cameras(self, camera1, camera2):
        self.camera1 = camera1
        self.camera2 = camera2

    def set_target_z(self, target_z: float):
        self.target_z = target_z
        print(f"✓ Z轴目标: {target_z:.1f}mm")

    def set_marker_diameter(self, diameter_mm: float):
        self.depth_estimator.set_marker_diameter(diameter_mm)

    def set_baseline(self, baseline_mm: float):
        self.depth_estimator.set_baseline(baseline_mm)

    # ==================== 标记检测 ====================

    def detect_markers_with_size(self, image: np.ndarray, color: str,
                                   min_area: int = 200, max_area: int = 5000) -> List[MarkerWithSize]:
        """检测标记点并测量尺寸"""
        COLOR_RANGES = {
            'green': {'lower': np.array([35, 70, 70]), 'upper': np.array([85, 255, 255])},
            'red': {
                'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255]),
                'lower2': np.array([160, 50, 50]), 'upper2': np.array([180, 255, 255])
            },
        }

        if color not in COLOR_RANGES:
            return []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cr = COLOR_RANGES[color]

        mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
        if 'lower2' in cr:
            mask2 = cv2.inRange(hsv, cr['lower2'], cr['upper2'])
            mask = cv2.bitwise_or(mask, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0

            if circularity < 0.5:
                continue

            confidence = circularity * min(1.0, area / 500)
            markers.append(MarkerWithSize(
                x=cx, y=cy, radius_px=radius,
                real_diameter_mm=self.depth_estimator.marker_diameter_mm,
                color=color, confidence=confidence
            ))

        return markers

    # ==================== 深度估计 ====================

    def estimate_z(self, image1: np.ndarray, image2: np.ndarray = None,
                   marker_color: str = "green") -> DepthEstimate:
        """估计Z轴深度"""
        estimates = []

        markers1 = self.detect_markers_with_size(image1, marker_color)
        if not markers1:
            return DepthEstimate(z=0, uncertainty=999, method="none", confidence=0)

        marker1 = max(markers1, key=lambda m: m.radius_px)
        estimates.append(self.depth_estimator.estimate_depth_monocular(marker1))

        if image2 is not None:
            markers2 = self.detect_markers_with_size(image2, marker_color)
            if markers2:
                marker2 = max(markers2, key=lambda m: m.radius_px)
                stereo = self.depth_estimator.estimate_depth_stereo(marker1, marker2)
                if stereo.confidence > 0:
                    estimates.append(stereo)

        fused = self.depth_estimator.fuse_depth_estimates(estimates)
        filtered_z = self.depth_estimator.update_kalman(fused)
        self.current_z = filtered_z

        return DepthEstimate(z=filtered_z, uncertainty=fused.uncertainty,
                            method=fused.method, confidence=fused.confidence,
                            timestamp=time.time())

    # ==================== 控制计算 ====================

    def compute_z_error(self, current_z: float = None) -> float:
        """计算Z轴误差"""
        if current_z is not None:
            self.current_z = current_z
        if self.target_z is None:
            return 0.0
        self.z_error = self.current_z - self.target_z
        return self.z_error

    def compute_z_adjustment(self, z_error: float = None) -> Dict[int, float]:
        """
        计算Z轴多关节调整量

        根据误差大小选择不同策略:
        - 小误差: 仅主关节
        - 中误差: 主关节为主
        - 大误差: 多关节协调

        Returns:
            {joint_idx: adjustment_deg}
        """
        if z_error is None:
            z_error = self.z_error

        adjustments = {}
        abs_error = abs(z_error)

        # 计算需要的Z变化量 (负号: 误差正=太远=需要下降)
        z_delta = -z_error * self.z_gain
        z_delta = np.clip(z_delta, -self.max_z_adjust, self.max_z_adjust)

        if abs_error < 3.0:
            # 小误差: 仅用主关节
            joint_idx = self.primary_joint
            sens = self.joint_sensitivities[joint_idx].mm_per_deg
            if abs(sens) > 0.01:
                adjustments[joint_idx] = z_delta / sens

        elif abs_error < 10.0:
            # 中误差: 主关节为主
            joint_idx = self.primary_joint
            sens = self.joint_sensitivities[joint_idx].mm_per_deg
            if abs(sens) > 0.01:
                adjustments[joint_idx] = z_delta / sens

        else:
            # 大误差: 多关节协调 (主关节80%, 辅助20%)
            main_joint = self.primary_joint
            main_sens = self.joint_sensitivities[main_joint].mm_per_deg

            if abs(main_sens) > 0.01:
                adjustments[main_joint] = (z_delta * 0.8) / main_sens

            # 辅助关节
            for jidx, sens_data in self.joint_sensitivities.items():
                if jidx != main_joint and abs(sens_data.mm_per_deg) > 0.01:
                    adjustments[jidx] = (z_delta * 0.2) / sens_data.mm_per_deg

        return adjustments

    def is_z_aligned(self, tolerance: float = None) -> bool:
        if tolerance is None:
            tolerance = self.tolerance_mm
        return abs(self.z_error) < tolerance

    # ==================== 自动标定 ====================

    def start_joint_calibration(self, joint_idx: int):
        """
        开始单个关节的Z轴灵敏度标定

        Args:
            joint_idx: 要标定的关节索引
        """
        self._calibration_in_progress = True
        self._calibration_data = {
            'joint_idx': joint_idx,
            'joint_name': self.JOINT_NAMES.get(joint_idx, f'joint_{joint_idx}'),
            'phase': 'ready',
            'z_before': None,
            'z_after': None,
            'joint_before': None,
            'joint_after': None,
        }
        print(f"\n{'='*50}")
        print(f"Z轴关节灵敏度标定: {self._calibration_data['joint_name']}")
        print(f"{'='*50}")
        print("准备移动关节并记录深度变化")

    def record_calibration_point(self, joint_states: np.ndarray, phase: str = 'before') -> bool:
        """
        记录标定点

        Args:
            joint_states: 当前关节状态
            phase: 'before' 或 'after'

        Returns:
            是否成功
        """
        if not self._calibration_in_progress:
            print("✗ 未启动标定")
            return False

        # 估计当前深度
        if self.camera1 is None:
            print("✗ 相机未设置")
            return False

        image1 = self.camera1.read()
        image2 = self.camera2.read() if self.camera2 else None

        estimate = self.estimate_z(image1, image2)

        if estimate.confidence < 0.3:
            print(f"✗ 深度估计不可靠 (置信度: {estimate.confidence:.2f})")
            return False

        joint_idx = self._calibration_data['joint_idx']

        if phase == 'before':
            self._calibration_data['z_before'] = estimate.z
            self._calibration_data['joint_before'] = joint_states[joint_idx]
            self._calibration_data['phase'] = 'moved'
            print(f"✓ 记录初始点: 深度={estimate.z:.1f}mm, 关节={joint_states[joint_idx]:.2f}°")
            print("\n请移动关节 (建议3-5度)，然后记录终点")
            return True

        elif phase == 'after':
            self._calibration_data['z_after'] = estimate.z
            self._calibration_data['joint_after'] = joint_states[joint_idx]
            return self._finish_calibration()

        return False

    def _finish_calibration(self) -> bool:
        """完成标定并计算灵敏度"""
        data = self._calibration_data

        joint_delta = data['joint_after'] - data['joint_before']
        z_delta = data['z_after'] - data['z_before']

        if abs(joint_delta) < 0.5:
            print("⚠ 关节移动量太小，标定可能不准确")
            self._calibration_in_progress = False
            return False

        mm_per_deg = z_delta / joint_delta

        # 更新灵敏度
        joint_idx = data['joint_idx']
        if joint_idx in self.joint_sensitivities:
            self.joint_sensitivities[joint_idx].mm_per_deg = mm_per_deg
            self.joint_sensitivities[joint_idx].calibration_height = data['z_before']
        else:
            self.joint_sensitivities[joint_idx] = ZJointSensitivity(
                joint_idx=joint_idx,
                joint_name=data['joint_name'],
                mm_per_deg=mm_per_deg,
                calibration_height=data['z_before']
            )

        print(f"\n✓ 标定完成:")
        print(f"  关节: {data['joint_name']}")
        print(f"  移动: {joint_delta:.2f}°")
        print(f"  Z变化: {z_delta:.1f}mm")
        print(f"  灵敏度: {mm_per_deg:.2f} mm/deg")

        self._calibration_in_progress = False
        self._save_calibration()
        return True

    def calibrate_all_z_joints_auto(self, robot, move_deg: float = 3.0) -> bool:
        """
        自动标定所有Z轴相关关节

        Args:
            robot: 机器人对象
            move_deg: 标定时关节移动角度

        Returns:
            是否成功
        """
        print(f"\n{'#'*60}")
        print("# Z轴多关节自动标定")
        print(f"{'#'*60}")

        joints_to_calibrate = [7, 8, 10, 12]  # joint_1, joint_2, joint_4, joint_6
        success_count = 0

        for joint_idx in joints_to_calibrate:
            print(f"\n--- 标定关节 {self.JOINT_NAMES.get(joint_idx, str(joint_idx))} ---")

            # 获取当前状态
            current_joints = robot.get_current_position()
            if current_joints is None:
                print("✗ 无法获取关节状态")
                continue

            # 记录初始点
            self.start_joint_calibration(joint_idx)

            joint_names = list(current_joints.keys())
            initial_angle = current_joints[joint_names[joint_idx]]

            if not self.record_calibration_point(
                np.array([current_joints[name] for name in joint_names]), 'before'
            ):
                continue

            # 移动关节
            target_angle = initial_angle + move_deg
            action = current_joints.copy()
            action[joint_names[joint_idx]] = target_angle
            robot.send_action(action)
            time.sleep(1.0)

            # 记录终点
            current_joints = robot.get_current_position()
            if not self.record_calibration_point(
                np.array([current_joints[name] for name in joint_names]), 'after'
            ):
                continue

            success_count += 1

            # 返回初始位置
            action[joint_names[joint_idx]] = initial_angle
            robot.send_action(action)
            time.sleep(0.5)

        print(f"\n{'='*60}")
        print(f"标定完成: {success_count}/{len(joints_to_calibrate)} 个关节成功")
        print(f"{'='*60}")

        return success_count > 0

    def _save_calibration(self):
        """保存标定数据"""
        calib_data = {
            'joint_sensitivities': {
                str(k): {
                    'joint_idx': v.joint_idx,
                    'joint_name': v.joint_name,
                    'mm_per_deg': v.mm_per_deg,
                    'calibration_height': v.calibration_height,
                }
                for k, v in self.joint_sensitivities.items()
            },
            'baseline': self.depth_estimator.stereo_calib.baseline,
            'marker_diameter': self.depth_estimator.marker_diameter_mm,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        path = Path(__file__).parent / "z_axis_calibration.json"
        with open(path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        print(f"✓ 标定数据已保存")

    def load_calibration(self):
        """加载标定数据"""
        path = Path(__file__).parent / "z_axis_calibration.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)

            for k, v in data.get('joint_sensitivities', {}).items():
                joint_idx = int(k)
                self.joint_sensitivities[joint_idx] = ZJointSensitivity(
                    joint_idx=v['joint_idx'],
                    joint_name=v['joint_name'],
                    mm_per_deg=v['mm_per_deg'],
                    calibration_height=v.get('calibration_height', 0.0)
                )

            if 'baseline' in data:
                self.depth_estimator.set_baseline(data['baseline'])
            if 'marker_diameter' in data:
                self.depth_estimator.set_marker_diameter(data['marker_diameter'])

            print(f"✓ 已加载Z轴标定数据")
            for jidx, sens in self.joint_sensitivities.items():
                print(f"  {sens.joint_name}: {sens.mm_per_deg:.2f} mm/deg")

    # ==================== 可视化 ====================

    def visualize_depth(self, image: np.ndarray, estimate: DepthEstimate) -> np.ndarray:
        """可视化深度估计结果"""
        vis = image.copy()

        y = 30
        cv2.putText(vis, f"Z: {estimate.z:.1f} mm", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25
        cv2.putText(vis, f"+/-{estimate.uncertainty:.1f} mm ({estimate.method})", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.target_z is not None:
            y += 20
            error = self.current_z - self.target_z
            color = (0, 255, 0) if abs(error) < self.tolerance_mm else (0, 0, 255)
            cv2.putText(vis, f"Target: {self.target_z:.1f}mm (err: {error:.1f})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis

    # ==================== 双相机基线标定 ====================

    def calibrate_stereo_baseline_auto(self, robot, move_distance_mm: float = 20.0,
                                        move_direction: str = 'x') -> Tuple[bool, float]:
        """
        自动标定双相机基线距离

        原理:
        1. 记录两个相机初始图像中的标记位置
        2. 机器人移动已知距离
        3. 记录移动后的标记位置
        4. 通过像素位移和已知距离计算等效焦距和基线

        Args:
            robot: 机器人对象，需要有 send_action 和 get_current_position 方法
            move_distance_mm: 移动距离 (mm)，建议 20-50mm
            move_direction: 移动方向 'x' 或 'y'

        Returns:
            (成功与否, 计算的基线距离)
        """
        print(f"\n{'#'*60}")
        print("# 双相机基线自动标定")
        print(f"{'#'*60}")

        if self.camera1 is None or self.camera2 is None:
            print("✗ 需要设置两个相机")
            return False, 0.0

        # 获取初始图像
        print("\n[1/5] 采集初始图像...")
        img1_before = self.camera1.read()
        img2_before = self.camera2.read()

        if img1_before is None or img2_before is None:
            print("✗ 无法读取相机图像")
            return False, 0.0

        # 检测初始标记
        marker1_before = self._detect_marker_for_calibration(img1_before)
        marker2_before = self._detect_marker_for_calibration(img2_before)

        if marker1_before is None or marker2_before is None:
            print("✗ 未能在两个相机中检测到标记")
            print("  请确保标记在两个相机视野中")
            return False, 0.0

        print(f"  相机1 标记位置: ({marker1_before.x:.1f}, {marker1_before.y:.1f})")
        print(f"  相机2 标记位置: ({marker2_before.x:.1f}, {marker2_before.y:.1f})")

        # 计算初始视差
        disparity_before = abs(marker1_before.x - marker2_before.x)
        print(f"  初始视差: {disparity_before:.1f} px")

        # 获取机器人当前位置
        print(f"\n[2/5] 准备移动机器人 {move_distance_mm}mm...")
        current_pos = robot.get_current_position()
        if current_pos is None:
            print("✗ 无法获取机器人位置")
            return False, 0.0

        # 保存当前位置用于恢复
        original_pos = current_pos.copy()

        # 执行移动 (使用 end_effector 移动或关节移动)
        try:
            # 尝试使用 end_effector 移动
            if hasattr(robot, 'move_end_effector'):
                if move_direction == 'x':
                    robot.move_end_effector(move_distance_mm, 0, 0)
                else:
                    robot.move_end_effector(0, move_distance_mm, 0)
            else:
                # 备用: 直接修改关节角度 (简化处理)
                print("  使用关节移动...")
                print(f"  请手动将机器人沿{move_direction}轴移动 {move_distance_mm}mm")
                input("  移动完成后按 Enter 继续...")
        except Exception as e:
            print(f"✗ 移动失败: {e}")
            return False, 0.0

        time.sleep(0.5)  # 等待稳定

        # 采集移动后图像
        print("\n[3/5] 采集移动后图像...")
        img1_after = self.camera1.read()
        img2_after = self.camera2.read()

        # 检测移动后标记
        marker1_after = self._detect_marker_for_calibration(img1_after)
        marker2_after = self._detect_marker_for_calibration(img2_after)

        if marker1_after is None or marker2_after is None:
            print("✗ 移动后未能在两个相机中检测到标记")
            # 尝试恢复位置
            try:
                robot.send_action(original_pos)
            except:
                pass
            return False, 0.0

        print(f"  相机1 标记位置: ({marker1_after.x:.1f}, {marker1_after.y:.1f})")
        print(f"  相机2 标记位置: ({marker2_after.x:.1f}, {marker2_after.y:.1f})")

        # 计算像素位移
        print("\n[4/5] 计算标定参数...")
        pixel_shift1 = marker1_after.x - marker1_before.x
        pixel_shift2 = marker2_after.x - marker2_before.x

        print(f"  相机1 像素位移: {pixel_shift1:.1f} px")
        print(f"  相机2 像素位移: {pixel_shift2:.1f} px")

        # 计算等效焦距 (假设移动方向与相机光轴垂直)
        avg_pixel_shift = (abs(pixel_shift1) + abs(pixel_shift2)) / 2

        if avg_pixel_shift < 5:
            print("✗ 像素位移太小，标定不准确")
            try:
                robot.send_action(original_pos)
            except:
                pass
            return False, 0.0

        # 使用单目深度估计验证当前深度
        current_depth = self.depth_estimator.estimate_depth_monocular(marker1_before)
        print(f"  估计当前深度: {current_depth.z:.1f}mm")

        # 计算等效焦距: f = pixel_shift * z / move_distance
        f_equiv = avg_pixel_shift * current_depth.z / move_distance_mm
        print(f"  等效焦距: {f_equiv:.1f} px")

        # 计算基线: baseline = disparity * z / f
        # 由于相机不是标准水平排列，使用校正因子
        # 对于 90度夹角的相机配置，需要特殊处理
        baseline = disparity_before * current_depth.z / f_equiv

        print(f"\n[5/5] 计算基线距离...")
        print(f"  视差: {disparity_before:.1f} px")
        print(f"  深度: {current_depth.z:.1f} mm")
        print(f"  计算基线: {baseline:.1f} mm")

        # 恢复机器人位置
        try:
            print("\n恢复机器人位置...")
            robot.send_action(original_pos)
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠ 恢复位置失败: {e}")

        # 更新标定数据
        if baseline > 10 and baseline < 500:  # 合理范围检查
            self.depth_estimator.set_baseline(baseline)
            self.depth_estimator.stereo_calib.camera1.fx = f_equiv
            self.depth_estimator.stereo_calib.camera1.fy = f_equiv
            self.depth_estimator.stereo_calib.camera2.fx = f_equiv
            self.depth_estimator.stereo_calib.camera2.fy = f_equiv

            self._save_calibration()

            print(f"\n{'='*60}")
            print(f"✓ 双相机基线标定完成")
            print(f"  基线距离: {baseline:.1f} mm")
            print(f"  等效焦距: {f_equiv:.1f} px")
            print(f"{'='*60}")
            return True, baseline
        else:
            print(f"✗ 计算的基线值不合理: {baseline:.1f}mm")
            return False, 0.0

    def calibrate_stereo_baseline_manual(self, known_depth_mm: float = 100.0) -> Tuple[bool, float]:
        """
        手动标定双相机基线 (使用已知深度)

        适用场景:
        - 已知标记到相机的精确距离
        - 或者使用标定板

        Args:
            known_depth_mm: 已知的标记深度 (mm)

        Returns:
            (成功与否, 计算的基线距离)
        """
        print(f"\n{'#'*60}")
        print("# 双相机基线手动标定")
        print(f"{'#'*60}")
        print(f"假设标记深度: {known_depth_mm}mm")

        if self.camera1 is None or self.camera2 is None:
            print("✗ 需要设置两个相机")
            return False, 0.0

        # 采集图像
        print("\n[1/3] 采集图像...")
        img1 = self.camera1.read()
        img2 = self.camera2.read()

        if img1 is None or img2 is None:
            print("✗ 无法读取相机图像")
            return False, 0.0

        # 检测标记
        marker1 = self._detect_marker_for_calibration(img1)
        marker2 = self._detect_marker_for_calibration(img2)

        if marker1 is None or marker2 is None:
            print("✗ 未能在两个相机中检测到标记")
            return False, 0.0

        print(f"  相机1 标记位置: ({marker1.x:.1f}, {marker1.y:.1f})")
        print(f"  相机2 标记位置: ({marker2.x:.1f}, {marker2.y:.1f})")

        # 计算视差
        disparity = abs(marker1.x - marker2.x)
        print(f"\n[2/3] 视差: {disparity:.1f} px")

        # 从标记尺寸估计焦距
        f_equiv = (marker1.radius_px * 2) * known_depth_mm / self.depth_estimator.marker_diameter_mm
        print(f"  估计焦距: {f_equiv:.1f} px")

        # 计算基线: baseline = disparity * z / f
        print(f"\n[3/3] 计算基线...")
        baseline = disparity * known_depth_mm / f_equiv

        if baseline > 10 and baseline < 500:
            self.depth_estimator.set_baseline(baseline)
            self.depth_estimator.stereo_calib.camera1.fx = f_equiv
            self.depth_estimator.stereo_calib.camera1.fy = f_equiv
            self.depth_estimator.stereo_calib.camera2.fx = f_equiv
            self.depth_estimator.stereo_calib.camera2.fy = f_equiv

            self._save_calibration()

            print(f"\n✓ 基线标定完成: {baseline:.1f}mm")
            return True, baseline
        else:
            print(f"✗ 计算的基线值不合理: {baseline:.1f}mm")
            return False, 0.0

    def calibrate_stereo_baseline_with_depth(self, reference_depth_mm: float = None) -> Tuple[bool, float]:
        """
        使用单目深度估计作为参考标定双目基线

        原理:
        - 单目尺寸估计提供初始深度
        - 使用该深度计算双目基线
        - 后续双目估计精度更高

        Args:
            reference_depth_mm: 参考深度 (mm)，None 则使用单目估计

        Returns:
            (成功与否, 计算的基线距离)
        """
        print(f"\n{'#'*60}")
        print("# 双相机基线标定 (使用深度参考)")
        print(f"{'#'*60}")

        if self.camera1 is None or self.camera2 is None:
            print("✗ 需要设置两个相机")
            return False, 0.0

        # 采集图像
        print("\n[1/3] 采集图像...")
        img1 = self.camera1.read()
        img2 = self.camera2.read()

        if img1 is None or img2 is None:
            print("✗ 无法读取相机图像")
            return False, 0.0

        # 检测标记
        marker1 = self._detect_marker_for_calibration(img1)
        marker2 = self._detect_marker_for_calibration(img2)

        if marker1 is None or marker2 is None:
            print("✗ 未能在两个相机中检测到标记")
            return False, 0.0

        # 确定参考深度
        if reference_depth_mm is None:
            # 使用单目估计
            depth_estimate = self.depth_estimator.estimate_depth_monocular(marker1)
            reference_depth_mm = depth_estimate.z
            print(f"  单目估计深度: {reference_depth_mm:.1f}mm (±{depth_estimate.uncertainty:.1f}mm)")
        else:
            print(f"  使用参考深度: {reference_depth_mm:.1f}mm")

        # 计算视差
        disparity = abs(marker1.x - marker2.x)
        print(f"\n[2/3] 视差: {disparity:.1f} px")

        # 计算等效焦距
        f_equiv = (marker1.radius_px * 2) * reference_depth_mm / self.depth_estimator.marker_diameter_mm
        print(f"  等效焦距: {f_equiv:.1f} px")

        # 计算基线
        baseline = disparity * reference_depth_mm / f_equiv
        print(f"\n[3/3] 计算基线: {baseline:.1f}mm")

        if baseline > 10 and baseline < 500:
            self.depth_estimator.set_baseline(baseline)
            self.depth_estimator.stereo_calib.camera1.fx = f_equiv
            self.depth_estimator.stereo_calib.camera1.fy = f_equiv
            self.depth_estimator.stereo_calib.camera2.fx = f_equiv
            self.depth_estimator.stereo_calib.camera2.fy = f_equiv

            self._save_calibration()

            print(f"\n✓ 基线标定完成: {baseline:.1f}mm")
            return True, baseline
        else:
            print(f"✗ 基线值不合理: {baseline:.1f}mm")
            return False, 0.0

    def _detect_marker_for_calibration(self, image: np.ndarray) -> Optional[MarkerWithSize]:
        """
        为标定检测标记

        尝试检测绿色或红色标记，返回带尺寸信息的标记
        """
        if image is None:
            return None

        # 转换颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        best_marker = None
        best_confidence = 0

        # 尝试检测绿色和红色标记
        color_ranges = {
            'green': [(35, 50, 50), (85, 255, 255)],
            'red1': [(0, 50, 50), (15, 255, 255)],
            'red2': [(165, 50, 50), (180, 255, 255)],
        }

        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            # 形态学处理
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # 忽略小区域
                    continue

                # 计算最小外接圆
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                confidence = area / (np.pi * radius * radius)  # 圆度

                if confidence > best_confidence and confidence > 0.5:
                    best_confidence = confidence
                    best_marker = MarkerWithSize(
                        x=cx,
                        y=cy,
                        radius_px=radius,
                        real_diameter_mm=self.depth_estimator.marker_diameter_mm,
                        color=color_name if 'green' in color_name else 'red',
                        confidence=confidence
                    )

        return best_marker

    def get_stereo_calibration_info(self) -> Dict:
        """获取当前双目标定信息"""
        return {
            'baseline_mm': self.depth_estimator.stereo_calib.baseline,
            'camera1_fx': self.depth_estimator.stereo_calib.camera1.fx,
            'camera1_fy': self.depth_estimator.stereo_calib.camera1.fy,
            'camera2_fx': self.depth_estimator.stereo_calib.camera2.fx,
            'camera2_fy': self.depth_estimator.stereo_calib.camera2.fy,
            'marker_diameter_mm': self.depth_estimator.marker_diameter_mm,
        }

    # ==================== P3: 相机内参自动标定 ====================

    def calibrate_camera_intrinsic_with_marker(self, num_samples: int = 10,
                                                marker_spacing_mm: float = 30.0) -> Tuple[bool, Dict]:
        """
        使用标记点进行相机内参标定 (P3)

        原理:
        - 在不同位置检测已知尺寸的标记
        - 通过标记的像素大小和实际尺寸推算焦距
        - 通过标记在图像中的位置推算主点

        注意: 此方法为简化标定，精度不如棋盘格标定
        适用于: 快速粗标定、无标定板场景

        Args:
            num_samples: 采样次数 (建议 10-20)
            marker_spacing_mm: 标记间距 (用于验证)

        Returns:
            (成功与否, 标定结果字典)
        """
        print(f"\n{'#'*60}")
        print("# P3: 相机内参自动标定")
        print(f"{'#'*60}")

        if self.camera1 is None:
            print("✗ 需要设置相机")
            return False, {}

        calibration_data = {
            'samples': [],
            'fx_estimates': [],
            'fy_estimates': [],
            'cx_estimates': [],
            'cy_estimates': [],
        }

        print(f"\n需要采集 {num_samples} 个样本")
        print("请缓慢移动相机，使标记出现在图像不同位置")
        print("按 's' 保存样本，按 'q' 完成标定，按 'ESC' 取消\n")

        sample_count = 0

        while sample_count < num_samples:
            img = self.camera1.read()
            if img is None:
                continue

            # 检测标记
            marker = self._detect_marker_for_calibration(img)
            vis = img.copy()

            if marker is not None:
                # 绘制标记
                cv2.circle(vis, (int(marker.x), int(marker.y)), int(marker.radius_px), (0, 255, 0), 2)
                cv2.circle(vis, (int(marker.x), int(marker.y)), 3, (0, 0, 255), -1)

                # 显示信息
                h, w = img.shape[:2]
                f_estimate = (marker.radius_px * 2) * 150 / self.depth_estimator.marker_diameter_mm  # 假设150mm深度

                info_text = f"Sample {sample_count}/{num_samples} | Marker: ({marker.x:.0f}, {marker.y:.0f}) | f_est: {f_estimate:.0f}"
                cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis, "Press 's' to save, 'q' to finish", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(vis, f"No marker detected ({sample_count}/{num_samples})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Camera Intrinsic Calibration", vis)
            key = cv2.waitKey(100) & 0xFF

            if key == ord('s') and marker is not None:
                sample_count += 1
                calibration_data['samples'].append({
                    'x': marker.x,
                    'y': marker.y,
                    'radius': marker.radius_px,
                    'confidence': marker.confidence,
                })
                print(f"  [{sample_count}/{num_samples}] 保存样本: ({marker.x:.1f}, {marker.y:.1f}), r={marker.radius_px:.1f}")

            elif key == ord('q') and sample_count >= 5:
                break

            elif key == 27:  # ESC
                cv2.destroyWindow("Camera Intrinsic Calibration")
                print("标定已取消")
                return False, {}

        cv2.destroyWindow("Camera Intrinsic Calibration")

        if sample_count < 5:
            print(f"✗ 样本数量不足: {sample_count}/5")
            return False, {}

        # 计算内参
        print(f"\n[{sample_count}个样本] 计算内参...")

        samples = calibration_data['samples']
        h, w = self.camera1.read().shape[:2] if self.camera1 else (480, 640)

        # 方法1: 使用标记尺寸估计焦距
        # 假设平均深度约为150mm (实际应该使用双目或已知深度)
        assumed_depth = 150.0  # mm
        fx_estimates = []
        fy_estimates = []

        for s in samples:
            # f = d * z / D, 其中 d=像素直径, z=深度, D=实际直径
            pixel_diameter = s['radius'] * 2
            f_estimate = pixel_diameter * assumed_depth / self.depth_estimator.marker_diameter_mm
            fx_estimates.append(f_estimate)
            fy_estimates.append(f_estimate)

        # 使用中位数作为焦距估计
        fx = np.median(fx_estimates)
        fy = fx  # 假设方形像素

        # 主点估计: 假设图像中心附近
        # 如果有足够的样本分布，可以使用加权平均
        cx = w / 2
        cy = h / 2

        # 如果样本分布较广，使用样本中心偏移
        x_samples = [s['x'] for s in samples]
        y_samples = [s['y'] for s in samples]
        x_std = np.std(x_samples)
        y_std = np.std(y_samples)

        if x_std > w * 0.2 and y_std > h * 0.2:  # 样本分布较广
            # 使用统计方法估计主点偏移
            cx = np.mean(x_samples)
            cy = np.mean(y_samples)
            print(f"  使用样本分布估计主点")

        # 更新相机内参
        self.depth_estimator.stereo_calib.camera1.fx = fx
        self.depth_estimator.stereo_calib.camera1.fy = fy
        self.depth_estimator.stereo_calib.camera1.cx = cx
        self.depth_estimator.stereo_calib.camera1.cy = cy

        # 同步到相机2 (假设两个相机参数相近)
        self.depth_estimator.stereo_calib.camera2.fx = fx
        self.depth_estimator.stereo_calib.camera2.fy = fy
        self.depth_estimator.stereo_calib.camera2.cx = cx
        self.depth_estimator.stereo_calib.camera2.cy = cy

        result = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'image_width': w,
            'image_height': h,
            'num_samples': sample_count,
            'assumed_depth': assumed_depth,
        }

        print(f"\n{'='*60}")
        print(f"✓ 相机内参标定完成")
        print(f"  焦距: fx={fx:.1f}, fy={fy:.1f}")
        print(f"  主点: cx={cx:.1f}, cy={cy:.1f}")
        print(f"  样本数: {sample_count}")
        print(f"{'='*60}")

        self._save_calibration()

        return True, result

    def calibrate_camera_intrinsic_opencv(self, chessboard_size: Tuple[int, int] = (9, 6),
                                           square_size_mm: float = 25.0,
                                           num_images: int = 15) -> Tuple[bool, Dict]:
        """
        使用OpenCV棋盘格标定 (P3备选方案)

        这是标准的相机标定方法，精度更高
        需要准备棋盘格标定板

        Args:
            chessboard_size: 棋盘格内角点数量 (cols, rows)
            square_size_mm: 棋盘格方格边长 (mm)
            num_images: 需要采集的图像数量

        Returns:
            (成功与否, 标定结果)
        """
        print(f"\n{'#'*60}")
        print("# P3: OpenCV 棋盘格相机标定")
        print(f"{'#'*60}")

        if self.camera1 is None:
            print("✗ 需要设置相机")
            return False, {}

        print(f"\n棋盘格参数: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
        print(f"方格大小: {square_size_mm}mm")
        print(f"需要采集 {num_images} 张图像")
        print("\n操作说明:")
        print("  1. 将棋盘格放在相机前")
        print("  2. 从不同角度和位置采集图像")
        print("  3. 按 's' 保存图像，按 'q' 开始计算")
        print("  4. 按 'ESC' 取消\n")

        # 准备物体点
        objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 1)
        objp *= square_size_mm

        objpoints = []  # 3D 点
        imgpoints = []  # 2D 点

        images_captured = 0

        while images_captured < num_images:
            img = self.camera1.read()
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            vis = img.copy()

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                cv2.drawChessboardCorners(vis, chessboard_size, corners, ret)
                cv2.putText(vis, f"Chessboard detected! Press 's' to save ({images_captured}/{num_images})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f"Show chessboard to camera ({images_captured}/{num_images})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Chessboard Calibration", vis)
            key = cv2.waitKey(100) & 0xFF

            if key == ord('s') and ret:
                images_captured += 1
                objpoints.append(objp)

                # 精细化角点
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_refined)

                print(f"  [{images_captured}/{num_images}] 棋盘格已保存")

            elif key == ord('q') and images_captured >= 5:
                break

            elif key == 27:
                cv2.destroyWindow("Chessboard Calibration")
                print("标定已取消")
                return False, {}

        cv2.destroyWindow("Chessboard Calibration")

        if len(objpoints) < 5:
            print(f"✗ 图像数量不足: {len(objpoints)}/5")
            return False, {}

        # 执行标定
        print(f"\n[{len(objpoints)}张图像] 执行标定计算...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if not ret:
            print("✗ 标定计算失败")
            return False, {}

        # 计算重投影误差
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)

        # 更新相机内参
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]

        self.depth_estimator.stereo_calib.camera1.fx = fx
        self.depth_estimator.stereo_calib.camera1.fy = fy
        self.depth_estimator.stereo_calib.camera1.cx = cx
        self.depth_estimator.stereo_calib.camera1.cy = cy
        self.depth_estimator.stereo_calib.camera1.distortion = dist.flatten()

        # 同步到相机2
        self.depth_estimator.stereo_calib.camera2.fx = fx
        self.depth_estimator.stereo_calib.camera2.fy = fy
        self.depth_estimator.stereo_calib.camera2.cx = cx
        self.depth_estimator.stereo_calib.camera2.cy = cy
        self.depth_estimator.stereo_calib.camera2.distortion = dist.flatten()

        result = {
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'distortion': dist.flatten().tolist(),
            'reprojection_error': float(mean_error),
            'image_width': gray.shape[1],
            'image_height': gray.shape[0],
        }

        print(f"\n{'='*60}")
        print(f"✓ 棋盘格标定完成")
        print(f"  焦距: fx={fx:.1f}, fy={fy:.1f}")
        print(f"  主点: cx={cx:.1f}, cy={cy:.1f}")
        print(f"  畸变系数: {dist.flatten()[:5]}")
        print(f"  重投影误差: {mean_error:.4f} px")
        print(f"{'='*60}")

        self._save_calibration()

        return True, result

    # ==================== P4: 多姿态标定插值 ====================

    def calibrate_multi_pose_sensitivity(self, robot, heights_mm: List[float] = None,
                                          move_deg: float = 3.0) -> bool:
        """
        多姿态Z轴灵敏度标定 (P4)

        在不同高度记录关节灵敏度，运行时插值获取当前姿态的灵敏度

        原理:
        - 机器人关节对Z轴的影响随姿态变化
        - 在多个高度标定，建立高度-灵敏度关系
        - 运行时根据当前深度插值

        Args:
            robot: 机器人对象
            heights_mm: 标定高度列表 (mm)，默认 [80, 120, 160, 200]
            move_deg: 标定时关节移动角度

        Returns:
            是否成功
        """
        print(f"\n{'#'*60}")
        print("# P4: 多姿态Z轴灵敏度标定")
        print(f"{'#'*60}")

        if heights_mm is None:
            heights_mm = [80, 120, 160, 200]

        if self.camera1 is None:
            print("✗ 需要设置相机")
            return False

        # 存储多姿态标定数据
        self._multi_pose_calibration = {
            'heights': [],
            'sensitivities': {},  # {joint_idx: [{height, mm_per_deg}, ...]}
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # 初始化灵敏度存储
        joints_to_calibrate = [7, 8, 10, 12]
        for joint_idx in joints_to_calibrate:
            self._multi_pose_calibration['sensitivities'][joint_idx] = []

        print(f"\n将在以下高度标定: {heights_mm}mm")
        print("请确保标记在相机视野中\n")

        success_heights = 0

        for target_height in heights_mm:
            print(f"\n{'='*50}")
            print(f"标定高度: {target_height}mm")
            print(f"{'='*50}")

            # 获取当前深度
            img1 = self.camera1.read()
            img2 = self.camera2.read() if self.camera2 else None
            estimate = self.estimate_z(img1, img2)

            if estimate.confidence < 0.3:
                print(f"✗ 深度估计不可靠 (置信度: {estimate.confidence:.2f})")
                print("  请调整机器人位置...")
                input("  调整好后按 Enter 继续...")

                # 重新获取
                img1 = self.camera1.read()
                img2 = self.camera2.read() if self.camera2 else None
                estimate = self.estimate_z(img1, img2)

                if estimate.confidence < 0.3:
                    print("  跳过此高度")
                    continue

            current_height = estimate.z
            print(f"  当前深度: {current_height:.1f}mm")

            # 标定各关节
            height_data = {'height': current_height, 'joints': {}}
            success_joints = 0

            for joint_idx in joints_to_calibrate:
                print(f"\n  --- 标定关节 {self.JOINT_NAMES.get(joint_idx, str(joint_idx))} ---")

                # 获取初始状态
                current_joints = robot.get_current_position()
                if current_joints is None:
                    continue

                joint_names = list(current_joints.keys())
                initial_angle = current_joints[joint_names[joint_idx]]

                # 记录初始深度
                img1 = self.camera1.read()
                img2 = self.camera2.read() if self.camera2 else None
                z_before = self.estimate_z(img1, img2).z

                # 移动关节
                target_angle = initial_angle + move_deg
                action = current_joints.copy()
                action[joint_names[joint_idx]] = target_angle
                robot.send_action(action)
                time.sleep(1.0)

                # 记录移动后深度
                current_joints = robot.get_current_position()
                img1 = self.camera1.read()
                img2 = self.camera2.read() if self.camera2 else None
                z_after = self.estimate_z(img1, img2).z

                # 计算灵敏度
                joint_delta = move_deg
                z_delta = z_after - z_before
                mm_per_deg = z_delta / joint_delta if abs(joint_delta) > 0.1 else 0

                print(f"    Z变化: {z_delta:.1f}mm, 灵敏度: {mm_per_deg:.2f} mm/deg")

                height_data['joints'][joint_idx] = {
                    'mm_per_deg': mm_per_deg,
                    'z_before': z_before,
                    'z_after': z_after,
                }

                self._multi_pose_calibration['sensitivities'][joint_idx].append({
                    'height': current_height,
                    'mm_per_deg': mm_per_deg,
                })

                # 恢复关节位置
                action[joint_names[joint_idx]] = initial_angle
                robot.send_action(action)
                time.sleep(0.5)

                success_joints += 1

            if success_joints > 0:
                self._multi_pose_calibration['heights'].append(current_height)
                success_heights += 1

        # 保存标定数据
        if success_heights > 0:
            self._save_multi_pose_calibration()
            print(f"\n{'='*60}")
            print(f"✓ 多姿态标定完成: {success_heights}/{len(heights_mm)} 个高度")
            self._print_multi_pose_summary()
            return True
        else:
            print("✗ 多姿态标定失败")
            return False

    def _save_multi_pose_calibration(self):
        """保存多姿态标定数据"""
        path = Path(__file__).parent / "multi_pose_calibration.json"
        with open(path, 'w') as f:
            json.dump(self._multi_pose_calibration, f, indent=2)
        print(f"✓ 多姿态标定数据已保存")

    def _load_multi_pose_calibration(self):
        """加载多姿态标定数据"""
        path = Path(__file__).parent / "multi_pose_calibration.json"
        if path.exists():
            with open(path, 'r') as f:
                self._multi_pose_calibration = json.load(f)
            return True
        return False

    def _print_multi_pose_summary(self):
        """打印多姿态标定摘要"""
        print("\n多姿态灵敏度数据:")
        for joint_idx, data_list in self._multi_pose_calibration['sensitivities'].items():
            if data_list:
                joint_name = self.JOINT_NAMES.get(int(joint_idx), f'joint_{joint_idx}')
                print(f"  {joint_name}:")
                for d in data_list:
                    print(f"    @ {d['height']:.0f}mm: {d['mm_per_deg']:.2f} mm/deg")

    def get_interpolated_sensitivity(self, joint_idx: int, current_height: float) -> float:
        """
        根据当前高度插值获取关节灵敏度 (P4核心功能)

        Args:
            joint_idx: 关节索引
            current_height: 当前深度 (mm)

        Returns:
            插值后的灵敏度 (mm/deg)
        """
        # 确保有标定数据
        if not hasattr(self, '_multi_pose_calibration') or not self._multi_pose_calibration:
            if not self._load_multi_pose_calibration():
                # 使用默认值
                if joint_idx in self.joint_sensitivities:
                    return self.joint_sensitivities[joint_idx].mm_per_deg
                return 5.0

        joint_key = str(joint_idx) if isinstance(joint_idx, int) else joint_idx
        data_list = self._multi_pose_calibration.get('sensitivities', {}).get(joint_key, [])

        if not data_list:
            # 使用默认值
            if joint_idx in self.joint_sensitivities:
                return self.joint_sensitivities[joint_idx].mm_per_deg
            return 5.0

        # 单点直接返回
        if len(data_list) == 1:
            return data_list[0]['mm_per_deg']

        # 按高度排序
        sorted_data = sorted(data_list, key=lambda x: x['height'])
        heights = [d['height'] for d in sorted_data]
        sensitivities = [d['mm_per_deg'] for d in sorted_data]

        # 线性插值
        if current_height <= heights[0]:
            return sensitivities[0]
        elif current_height >= heights[-1]:
            return sensitivities[-1]
        else:
            # 找到插值区间
            for i in range(len(heights) - 1):
                if heights[i] <= current_height <= heights[i + 1]:
                    # 线性插值
                    t = (current_height - heights[i]) / (heights[i + 1] - heights[i])
                    return sensitivities[i] + t * (sensitivities[i + 1] - sensitivities[i])

        return sensitivities[-1]

    def compute_z_adjustment_interpolated(self, z_error: float, current_height: float) -> Dict[int, float]:
        """
        使用插值灵敏度计算Z轴调整量 (P4)

        Args:
            z_error: Z轴误差 (mm)
            current_height: 当前深度 (mm)

        Returns:
            {关节索引: 角度调整量}
        """
        adjustments = {}

        # 控制策略
        abs_error = abs(z_error)

        if abs_error < 1.0:
            # 小误差，不调整
            return adjustments

        # 获取插值灵敏度
        for joint_idx in [7, 8, 10, 12]:
            mm_per_deg = self.get_interpolated_sensitivity(joint_idx, current_height)

            if abs(mm_per_deg) < 0.1:
                continue

            # 根据误差大小分配调整
            if abs_error < 3.0:
                # 小误差: 仅主关节
                if joint_idx == 7:  # 主关节
                    deg = z_error / mm_per_deg * self.z_gain
                    adjustments[joint_idx] = np.clip(deg, -self.max_z_adjust, self.max_z_adjust)
            elif abs_error < 10.0:
                # 中等误差: 主关节 + 辅助关节
                weight = 1.0 if joint_idx == 7 else (0.5 if joint_idx in [8, 10] else 0.2)
                deg = z_error / mm_per_deg * self.z_gain * weight
                adjustments[joint_idx] = np.clip(deg, -self.max_z_adjust, self.max_z_adjust)
            else:
                # 大误差: 所有关节协调
                weight = 1.0 if joint_idx == 7 else (0.6 if joint_idx in [8, 10] else 0.3)
                deg = z_error / mm_per_deg * self.z_gain * weight
                adjustments[joint_idx] = np.clip(deg, -self.max_z_adjust, self.max_z_adjust)

        return adjustments

    def get_multi_pose_calibration_info(self) -> Dict:
        """获取多姿态标定信息"""
        if not hasattr(self, '_multi_pose_calibration'):
            self._load_multi_pose_calibration()

        if not self._multi_pose_calibration:
            return {'status': 'not_calibrated'}

        return {
            'status': 'calibrated',
            'num_heights': len(self._multi_pose_calibration.get('heights', [])),
            'heights': self._multi_pose_calibration.get('heights', []),
            'timestamp': self._multi_pose_calibration.get('timestamp', ''),
        }