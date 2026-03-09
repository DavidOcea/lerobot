"""
双标记点精准对齐系统 - V2

功能:
1. 双标记点检测 (工件2绿 + 卡槽2红)
2. 左手/右手切换
3. 自动高度调整
4. XY对齐 + 旋转校正
5. 夹爪控制
6. 运动平滑
7. 多点标定插值 (方案A)
8. 雅可比运动学框架 (方案B预留)
9. 预设位置
10. 共享状态读取 (与示教程序协同)

标定方案:
- 方案A: 多点手动标定 + 线性插值 (已实现)
- 方案B: DH参数 + 雅可比矩阵 (框架预留)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
from pathlib import Path

# 导入共享状态模块
try:
    from precision_place.robot_status import (
        RobotStatusReader, joints_dict_to_array, JOINT_NAME_TO_INDEX
    )
    _has_status_reader = True
except ImportError:
    _has_status_reader = False
    RobotStatusReader = None


# ==================== 数据结构 ====================

@dataclass
class Marker:
    """单个标记"""
    x: float
    y: float
    color: str
    confidence: float


@dataclass
class DualMarkerState:
    """双标记状态"""
    workpiece_top: Optional[Marker] = None
    workpiece_bottom: Optional[Marker] = None
    slot_top: Optional[Marker] = None
    slot_bottom: Optional[Marker] = None
    offset_x: float = 0
    offset_y: float = 0
    rotation_error: float = 0
    workpiece_detected: bool = False
    slot_detected: bool = False
    alignment_quality: float = 0


@dataclass
class ArmConfig:
    """手臂配置"""
    name: str
    camera_name: str
    camera_index: int
    # 主要控制关节索引 (用于方案A的简化控制)
    primary_joints: List[int]  # 主要影响XY的关节索引列表
    gripper_idx: int
    gripper_open: float = 0.0
    gripper_close: float = 50.0
    # DH参数 (方案B预留)
    dh_params: Optional[List[Dict]] = None


# 手臂配置 - 更新为多关节控制
ARM_CONFIGS = {
    'right': ArmConfig(
        name='right',
        camera_name='right_wrist2',
        camera_index=8,
        primary_joints=[7, 8, 9, 10],  # right_arm_joint_1~4
        gripper_idx=13,
        gripper_open=0.0,
        gripper_close=50.0,
        dh_params=None  # 待用户提供
    ),
    'left': ArmConfig(
        name='left',
        camera_name='left_wrist2',
        camera_index=4,
        primary_joints=[0, 1, 2, 3],  # left_arm_joint_1~4
        gripper_idx=6,
        gripper_open=0.0,
        gripper_close=50.0,
        dh_params=None  # 待用户提供
    )
}


# ==================== 标定数据结构 ====================

@dataclass
class JointSensitivity:
    """单个关节的灵敏度数据"""
    joint_idx: int
    joint_name: str
    # 关节移动1度时，末端在相机中的像素变化
    pixel_dx_per_deg: float = 0.0  # X方向像素变化
    pixel_dy_per_deg: float = 0.0  # Y方向像素变化
    # 关节移动1度时，末端的实际毫米移动 (近似)
    mm_dx_per_deg: float = 0.0
    mm_dy_per_deg: float = 0.0
    # 标定时的关节角度
    calibration_angles: List[float] = field(default_factory=list)


@dataclass
class CalibrationPoint:
    """单个标定点 (特定姿态下的灵敏度)"""
    height_level: str  # 高度等级: "high", "medium", "low"
    joint_states: np.ndarray  # 标定时的所有关节状态
    sensitivities: List[JointSensitivity]  # 各关节的灵敏度
    pixel_to_mm: float = 0.5  # 该高度下的像素-毫米转换比例
    timestamp: str = ""
    arm: str = "right"  # 对应的手臂


# ==================== 方案B: 雅可比运动学框架 ====================

class KinematicsModel(ABC):
    """运动学模型抽象基类 (方案B框架)"""

    @abstractmethod
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正运动学: 关节角度 -> 末端位姿

        Args:
            joint_angles: 关节角度数组

        Returns:
            末端位姿 [x, y, z, roll, pitch, yaw]
        """
        pass

    @abstractmethod
    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵

        Args:
            joint_angles: 关节角度数组

        Returns:
            6xN 雅可比矩阵 (N为关节数)
        """
        pass

    @abstractmethod
    def inverse_kinematics(self, target_pose: np.ndarray,
                          current_joints: np.ndarray) -> np.ndarray:
        """
        逆运动学: 目标位姿 -> 关节角度

        Args:
            target_pose: 目标末端位姿
            current_joints: 当前关节角度 (作为初值)

        Returns:
            目标关节角度
        """
        pass

    def compute_joint_velocities(self, joint_angles: np.ndarray,
                                 cartesian_velocity: np.ndarray) -> np.ndarray:
        """
        根据笛卡尔空间速度计算关节速度

        Args:
            joint_angles: 当前关节角度
            cartesian_velocity: 笛卡尔空间期望速度 [vx, vy, vz, wx, wy, wz]

        Returns:
            关节速度
        """
        J = self.compute_jacobian(joint_angles)
        # 使用伪逆求解
        J_pinv = np.linalg.pinv(J)
        joint_velocities = J_pinv @ cartesian_velocity
        return joint_velocities


class DHKinematicsModel(KinematicsModel):
    """基于DH参数的运动学模型 (方案B实现)"""

    def __init__(self, dh_params: List[Dict]):
        """
        Args:
            dh_params: DH参数列表, 每个元素为:
                {
                    'a': 连杆长度,
                    'alpha': 连杆扭转角,
                    'd': 连杆偏距,
                    'theta': 关节角偏移
                }
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        self._validate_dh_params()

    def _validate_dh_params(self):
        """验证DH参数有效性"""
        required_keys = ['a', 'alpha', 'd', 'theta']
        for i, params in enumerate(self.dh_params):
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"DH参数第{i}个关节缺少 '{key}'")

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """正运动学 - 待实现"""
        raise NotImplementedError("请提供DH参数后实现此方法")

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵 - 待实现"""
        raise NotImplementedError("请提供DH参数后实现此方法")

    def inverse_kinematics(self, target_pose: np.ndarray,
                          current_joints: np.ndarray) -> np.ndarray:
        """逆运动学 - 待实现"""
        raise NotImplementedError("请提供DH参数后实现此方法")


class DummyKinematicsModel(KinematicsModel):
    """空运动学模型 (方案A使用)"""

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        return np.zeros(6)

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        return np.zeros((6, len(joint_angles)))

    def inverse_kinematics(self, target_pose: np.ndarray,
                          current_joints: np.ndarray) -> np.ndarray:
        return current_joints.copy()


# ==================== 检测器 ====================

class DualPointDetector:
    """双标记点检测器"""
    
    COLOR_RANGES = {
        'green': {
            'lower': np.array([35, 50, 50]),
            'upper': np.array([85, 255, 255])
        },
        'red': {
            'lower': np.array([0, 80, 80]),
            'upper': np.array([10, 255, 255]),
            'lower2': np.array([160, 80, 80]),
            'upper2': np.array([180, 255, 255])
        },
        'blue': {
            'lower': np.array([100, 50, 50]),
            'upper': np.array([130, 255, 255])
        },
    }
    
    def __init__(self):
        self.workpiece_color = "green"
        self.slot_color = "red"
        self.min_area = 200
        self.max_area = 5000
    
    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        self.workpiece_color = workpiece_color
        self.slot_color = slot_color
    
    def detect_markers_by_color(self, image: np.ndarray, color: str) -> List[Marker]:
        """检测指定颜色的所有标记"""
        if color not in self.COLOR_RANGES:
            return []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cr = self.COLOR_RANGES[color]
        
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
            if area < self.min_area or area > self.max_area:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:
                continue
            
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            confidence = circularity * min(1.0, area / 1000)
            markers.append(Marker(x=cx, y=cy, color=color, confidence=confidence))
        
        return markers
    
    def detect_dual_marker_state(self, image: np.ndarray) -> DualMarkerState:
        """检测双标记状态"""
        state = DualMarkerState()
        
        # 工件标记
        wp_markers = self.detect_markers_by_color(image, self.workpiece_color)
        if len(wp_markers) >= 2:
            sorted_wp = sorted(wp_markers, key=lambda m: m.y)
            state.workpiece_top = sorted_wp[0]
            state.workpiece_bottom = sorted_wp[-1]
            state.workpiece_detected = True
        
        # 卡槽标记
        sl_markers = self.detect_markers_by_color(image, self.slot_color)
        if len(sl_markers) >= 2:
            sorted_sl = sorted(sl_markers, key=lambda m: m.y)
            state.slot_top = sorted_sl[0]
            state.slot_bottom = sorted_sl[-1]
            state.slot_detected = True
        
        if state.workpiece_detected and state.slot_detected:
            self._calculate_alignment(state)
        
        return state
    
    def _calculate_alignment(self, state: DualMarkerState):
        """计算对齐误差"""
        wp_cx = (state.workpiece_top.x + state.workpiece_bottom.x) / 2
        wp_cy = (state.workpiece_top.y + state.workpiece_bottom.y) / 2
        sl_cx = (state.slot_top.x + state.slot_bottom.x) / 2
        sl_cy = (state.slot_top.y + state.slot_bottom.y) / 2
        
        state.offset_x = sl_cx - wp_cx
        state.offset_y = sl_cy - wp_cy
        
        wp_angle = np.degrees(np.arctan2(
            state.workpiece_bottom.x - state.workpiece_top.x,
            state.workpiece_bottom.y - state.workpiece_top.y
        ))
        sl_angle = np.degrees(np.arctan2(
            state.slot_bottom.x - state.slot_top.x,
            state.slot_bottom.y - state.slot_top.y
        ))
        state.rotation_error = wp_angle - sl_angle
        
        wp_conf = (state.workpiece_top.confidence + state.workpiece_bottom.confidence) / 2
        sl_conf = (state.slot_top.confidence + state.slot_bottom.confidence) / 2
        state.alignment_quality = (wp_conf + sl_conf) / 2
    
    def visualize(self, image: np.ndarray, state: DualMarkerState = None) -> np.ndarray:
        """可视化"""
        vis = image.copy()
        
        if state is None:
            state = self.detect_dual_marker_state(image)
        
        # 工件标记
        for m, name in [(state.workpiece_top, "WP-T"), (state.workpiece_bottom, "WP-B")]:
            if m:
                cv2.circle(vis, (int(m.x), int(m.y)), 12, (0, 255, 0), 2)
                cv2.putText(vis, name, (int(m.x)-20, int(m.y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 卡槽标记
        for m, name in [(state.slot_top, "SL-T"), (state.slot_bottom, "SL-B")]:
            if m:
                cv2.circle(vis, (int(m.x), int(m.y)), 12, (0, 0, 255), 2)
                cv2.putText(vis, name, (int(m.x)-20, int(m.y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 连线
        if state.workpiece_top and state.workpiece_bottom:
            cv2.line(vis, (int(state.workpiece_top.x), int(state.workpiece_top.y)),
                    (int(state.workpiece_bottom.x), int(state.workpiece_bottom.y)),
                    (0, 255, 0), 2)
        
        if state.slot_top and state.slot_bottom:
            cv2.line(vis, (int(state.slot_top.x), int(state.slot_top.y)),
                    (int(state.slot_bottom.x), int(state.slot_bottom.y)),
                    (0, 0, 255), 2)
        
        # 状态
        y = 30
        cv2.putText(vis, f"WP: {'OK' if state.workpiece_detected else 'NO'}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if state.workpiece_detected else (0, 0, 255), 2)
        cv2.putText(vis, f"SL: {'OK' if state.slot_detected else 'NO'}", (10, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if state.slot_detected else (0, 0, 255), 2)
        
        if state.workpiece_detected and state.slot_detected:
            cv2.putText(vis, f"XY: ({state.offset_x:.0f}, {state.offset_y:.0f})", (10, y+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"Rot: {state.rotation_error:.1f}deg", (10, y+75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis


# ==================== 控制器 ====================

class PrecisionPlaceController:
    """
    精准放置控制器 - 多点标定版

    支持两种模式:
    - 方案A: 多点手动标定 + 插值 (默认)
    - 方案B: DH参数 + 雅可比 (需提供DH参数)

    被动模式:
    - 与示教程序协同工作
    - 通过共享文件读取机器人状态
    - 不发送控制指令
    """

    def __init__(self, robot, camera, arm: str = "right", passive_mode: bool = False):
        self.robot = robot
        self.camera = camera
        self.arm = arm
        self.passive_mode = passive_mode  # 被动模式：只读取，不发送动作

        # 加载手臂配置
        self.arm_config = ARM_CONFIGS.get(arm, ARM_CONFIGS['right'])

        # 检测器
        self.detector = DualPointDetector()

        # 运动学模型 (方案B)
        self.kinematics_model: Optional[KinematicsModel] = None
        self._init_kinematics_model()

        # 共享状态读取器 (被动模式使用)
        self.status_reader: Optional[RobotStatusReader] = None
        if passive_mode and _has_status_reader:
            self.status_reader = RobotStatusReader()
            print("✓ 已启用共享状态读取 (从示教程序获取机器人状态)")

        # 方案A: 多点标定数据
        self.calibration_points: List[CalibrationPoint] = []
        self._load_calibration_points()

        # 参数
        self.pixel_to_mm_ratio = 0.5  # 兼容旧标定
        self.gain = 0.6
        self.tolerance_mm = 2.0
        self.max_iterations = 15
        self.settle_time = 0.3

        # 运动平滑参数
        self.smooth_steps = 5
        self.smooth_delay = 0.05

        # 高度控制 (用于粗调)
        self.height_joint_idx = self.arm_config.primary_joints[1]  # 通常joint_2影响高度

        # 预设位置
        self.presets: Dict[str, np.ndarray] = {}
        self._load_presets()

        # 标定历史 (旧格式兼容)
        self.calibration_history: List[dict] = []
        self._load_calibration_history()

        # 关节名称映射
        self.joint_names = self._build_joint_names()

    def _build_joint_names(self) -> Dict[int, str]:
        """构建关节索引到名称的映射"""
        names = {}
        # 左臂
        for i in range(7):
            names[i] = f"left_arm_joint_{i+1}"
        # 右臂
        for i in range(7, 14):
            names[i] = f"right_arm_joint_{i-6}"
        # 躯干
        names[14] = "trunk_joint_1"
        names[15] = "trunk_joint_2"
        return names

    def _init_kinematics_model(self):
        """初始化运动学模型"""
        if self.arm_config.dh_params:
            # 方案B: 使用DH参数
            self.kinematics_model = DHKinematicsModel(self.arm_config.dh_params)
            print("✓ 已加载DH运动学模型 (方案B)")
        else:
            # 方案A: 使用空模型
            self.kinematics_model = DummyKinematicsModel()
            print("使用多点标定模式 (方案A)")

    # ----------------- 手臂切换 -----------------

    def switch_arm(self, arm: str):
        """切换手臂"""
        if arm not in ARM_CONFIGS:
            print(f"不支持的手臂: {arm}")
            return False

        self.arm = arm
        self.arm_config = ARM_CONFIGS[arm]
        self.height_joint_idx = self.arm_config.primary_joints[1]
        self._init_kinematics_model()
        print(f"✓ 已切换到{arm}手")
        print(f"  相机: {self.arm_config.camera_name} (索引{self.arm_config.camera_index})")
        return True

    def set_marker_colors(self, workpiece_color: str, slot_color: str):
        self.detector.set_marker_colors(workpiece_color, slot_color)

    # ==================== 关节状态读取 ====================

    def get_joint_states(self, max_age_ms: float = 200) -> Optional[np.ndarray]:
        """
        获取关节状态

        在被动模式下从共享文件读取，在主动模式下直接从机器人读取

        Args:
            max_age_ms: 最大数据年龄 (毫秒)，仅被动模式有效

        Returns:
            16维关节角度数组，如果失败返回 None
        """
        if self.passive_mode and self.status_reader is not None:
            # 被动模式: 从共享文件读取
            joints_dict = self.status_reader.read_joints(max_age_ms)
            if joints_dict is None:
                return None
            return joints_dict_to_array(joints_dict)
        else:
            # 主动模式: 直接从机器人读取
            if self.robot is None:
                return None
            try:
                obs = self.robot.get_observation()
                joints = np.array(obs.get('observation.state', []))
                if len(joints) >= 14:  # 至少需要14个关节
                    if len(joints) < 16:
                        # 补齐到16维
                        joints = np.pad(joints, (0, 16 - len(joints)))
                    return joints
                return None
            except Exception as e:
                print(f"读取关节状态失败: {e}")
                return None

    def check_shared_status(self) -> Dict[str, Any]:
        """检查共享状态 (用于调试)"""
        if self.status_reader is None:
            return {"error": "状态读取器未初始化"}
        return self.status_reader.get_status_info()

    # ==================== 方案A: 多点标定 ====================

    def calibrate_joint_sensitivity(self, joint_idx: int, move_degrees: float = 2.0) -> Tuple[bool, JointSensitivity]:
        """
        标定单个关节的灵敏度

        流程:
        1. 采集当前图像
        2. 手动移动指定关节 move_degrees 度
        3. 采集移动后图像
        4. 计算像素变化 -> 灵敏度

        Args:
            joint_idx: 关节索引
            move_degrees: 移动角度 (建议1-3度)

        Returns:
            (success, JointSensitivity)
        """
        print(f"\n{'='*60}")
        print(f"关节灵敏度标定: {self.joint_names.get(joint_idx, f'joint_{joint_idx}')}")
        print(f"{'='*60}")

        # 获取当前关节状态
        joints = self.get_joint_states()
        if joints is None:
            print("✗ 无法获取关节位置")
            if self.passive_mode:
                print("  请确认示教程序已启动并启用状态共享 (share_status=true)")
            return False, JointSensitivity(joint_idx, "")

        joint_name = self.joint_names.get(joint_idx, f"joint_{joint_idx}")
        current_angle = joints[joint_idx]

        print(f"\n当前关节角度: {current_angle:.2f}°")

        if self.passive_mode:
            print(f"\n[示教模式] 请用示教器将关节 {joint_name} 移动约 {move_degrees}°")
            print(f"  目标角度: 约 {current_angle + move_degrees:.2f}°")
            print(f"  提示: 移动示教器对应关节，执行机器人会跟随移动")
        else:
            print(f"请将关节 {joint_name} 移动 {move_degrees}°")
            print(f"  目标角度: {current_angle + move_degrees:.2f}°")

        # 采集初始图像
        print("\n[1/3] 采集初始图像...")
        img1 = self.camera.read()
        if img1 is None:
            print("✗ 图像采集失败")
            return False, JointSensitivity(joint_idx, joint_name)

        # 等待用户移动
        input(f"\n[2/3] 移动完成后按 Enter...")

        # 获取移动后的关节状态，计算实际移动角度
        joints_after = self.get_joint_states()
        if joints_after is None:
            print("✗ 无法获取移动后的关节位置")
            return False, JointSensitivity(joint_idx, joint_name)

        actual_move = joints_after[joint_idx] - current_angle

        if abs(actual_move) < 0.1:
            print(f"⚠ 警告: 检测到移动角度很小 ({actual_move:.2f}°)，标定可能不准确")

        # 采集移动后图像
        print("[3/3] 采集移动后图像...")
        img2 = self.camera.read()
        if img2 is None:
            print("✗ 图像采集失败")
            return False, JointSensitivity(joint_idx, joint_name)

        # 计算像素变化
        pixel_dx, pixel_dy = self._compute_pixel_shift(img1, img2)

        if pixel_dx is None:
            print("✗ 特征点匹配失败")
            return False, JointSensitivity(joint_idx, joint_name)

        # 使用实际移动角度计算灵敏度
        if abs(actual_move) > 0.1:
            move_degrees_actual = abs(actual_move)
        else:
            move_degrees_actual = move_degrees

        # 计算灵敏度
        sensitivity = JointSensitivity(
            joint_idx=joint_idx,
            joint_name=joint_name,
            pixel_dx_per_deg=pixel_dx / move_degrees_actual,
            pixel_dy_per_deg=pixel_dy / move_degrees_actual,
            mm_dx_per_deg=0.0,  # 需要pixel_to_mm_ratio
            mm_dy_per_deg=0.0,
            calibration_angles=joints.tolist()
        )

        # 更新mm灵敏度 (如果有pixel_to_mm_ratio)
        if self.pixel_to_mm_ratio > 0:
            sensitivity.mm_dx_per_deg = sensitivity.pixel_dx_per_deg * self.pixel_to_mm_ratio
            sensitivity.mm_dy_per_deg = sensitivity.pixel_dy_per_deg * self.pixel_to_mm_ratio

        print(f"\n标定结果:")
        print(f"  实际移动: {actual_move:.2f}°")
        print(f"  像素变化: ({pixel_dx:.1f}, {pixel_dy:.1f}) pixels")
        print(f"  灵敏度: X={sensitivity.pixel_dx_per_deg:.2f} px/deg, Y={sensitivity.pixel_dy_per_deg:.2f} px/deg")

        return True, sensitivity

    def _compute_pixel_shift(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """使用光流计算图像间的像素偏移"""
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        pts = cv2.goodFeaturesToTrack(g1, 100, 0.01, 10)
        if pts is None or len(pts) < 10:
            return None, None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None)
        if p1 is None:
            return None, None

        good = p1[st == 1] - pts[st == 1]
        dx = np.mean(good, axis=0)[0]
        dy = np.mean(good, axis=0)[1]

        return float(dx), float(dy)

    def calibrate_all_joints(self, move_degrees: float = 2.0) -> bool:
        """
        标定所有主要关节 (多点标定)

        在当前姿态下标定所有primary_joints的灵敏度
        """
        print(f"\n{'#'*60}")
        print("# 多点标定 - 所有主要关节")
        print(f"{'#'*60}")

        if self.passive_mode:
            print("\n[示教模式]")
            print("请确保示教程序已启动 (./run.sh)")
            print("移动示教器对应关节，执行机器人会跟随移动")

        # 获取当前高度/姿态信息
        # 获取关节状态
        joints = self.get_joint_states()
        if joints is None:
            print("✗ 无法获取关节状态")
            return False

        # 确定高度等级
        height_level = self._estimate_height_level(joints)
        print(f"\n当前高度等级: {height_level}")

        # 创建新的标定点
        cal_point = CalibrationPoint(
            height_level=height_level,
            joint_states=joints.tolist(),
            sensitivities=[],
            pixel_to_mm=self.pixel_to_mm_ratio,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            arm=self.arm
        )

        # 标定每个主要关节
        primary_joints = self.arm_config.primary_joints

        print(f"\n将标定 {len(primary_joints)} 个关节:")
        for i, jidx in enumerate(primary_joints):
            print(f"  {i+1}. {self.joint_names.get(jidx, f'joint_{jidx}')} (索引{jidx})")

        input("\n按 Enter 开始标定...")

        success_count = 0
        for i, jidx in enumerate(primary_joints):
            print(f"\n{'='*40}")
            print(f"[{i+1}/{len(primary_joints)}] 标定关节 {jidx}")
            print(f"{'='*40}")

            success, sensitivity = self.calibrate_joint_sensitivity(jidx, move_degrees)

            if success:
                cal_point.sensitivities.append(sensitivity)
                success_count += 1
            else:
                print(f"✗ 关节 {jidx} 标定失败，跳过")

        # 保存标定点
        if success_count > 0:
            self.calibration_points.append(cal_point)
            self._save_calibration_points()
            print(f"\n✓ 标定完成: {success_count}/{len(primary_joints)} 个关节成功")
            print(f"✓ 已保存到高度等级 '{height_level}'")
            return True
        else:
            print(f"\n✗ 标定失败: 没有关节标定成功")
            return False

    def _estimate_height_level(self, joints: np.ndarray) -> str:
        """根据关节状态估计高度等级"""
        # 简单实现: 根据joint_2 (肩部俯仰) 判断
        if len(self.arm_config.primary_joints) < 2:
            return "medium"

        joint_2_idx = self.arm_config.primary_joints[1]
        angle = joints[joint_2_idx] if joint_2_idx < len(joints) else 0

        if angle > 45:
            return "high"
        elif angle > 20:
            return "medium"
        else:
            return "low"

    def get_interpolated_sensitivities(self, current_joints: np.ndarray) -> List[JointSensitivity]:
        """
        根据当前关节状态，插值获取灵敏度

        方案A的核心方法
        """
        if not self.calibration_points:
            return []

        if len(self.calibration_points) == 1:
            return self.calibration_points[0].sensitivities

        # 计算与各标定点的距离权重
        current_level = self._estimate_height_level(current_joints)
        level_order = {"high": 3, "medium": 2, "low": 1}

        # 简单实现: 找最近的标定点
        min_dist = float('inf')
        best_point = self.calibration_points[0]

        for cp in self.calibration_points:
            if cp.arm != self.arm:
                continue
            dist = np.linalg.norm(np.array(cp.joint_states) - current_joints)
            if dist < min_dist:
                min_dist = dist
                best_point = cp

        return best_point.sensitivities

    def _load_calibration_points(self):
        """加载多点标定数据"""
        path = Path(__file__).parent / "calibration_points.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)

            self.calibration_points = []
            for cp_data in data.get('points', []):
                sensitivities = [
                    JointSensitivity(**s) for s in cp_data.get('sensitivities', [])
                ]
                cp = CalibrationPoint(
                    height_level=cp_data['height_level'],
                    joint_states=cp_data['joint_states'],
                    sensitivities=sensitivities,
                    timestamp=cp_data.get('timestamp', ''),
                    arm=cp_data.get('arm', 'right')
                )
                self.calibration_points.append(cp)

            print(f"✓ 已加载 {len(self.calibration_points)} 个标定点")

    def _save_calibration_points(self):
        """保存多点标定数据"""
        path = Path(__file__).parent / "calibration_points.json"

        data = {
            'version': '1.0',
            'points': []
        }

        for cp in self.calibration_points:
            cp_data = {
                'height_level': cp.height_level,
                'joint_states': cp.joint_states,
                'sensitivities': [
                    {
                        'joint_idx': s.joint_idx,
                        'joint_name': s.joint_name,
                        'pixel_dx_per_deg': s.pixel_dx_per_deg,
                        'pixel_dy_per_deg': s.pixel_dy_per_deg,
                        'mm_dx_per_deg': s.mm_dx_per_deg,
                        'mm_dy_per_deg': s.mm_dy_per_deg,
                        'calibration_angles': s.calibration_angles
                    }
                    for s in cp.sensitivities
                ],
                'timestamp': cp.timestamp,
                'arm': cp.arm
            }
            data['points'].append(cp_data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def show_calibration_points(self):
        """显示所有标定点"""
        if not self.calibration_points:
            print("\n暂无标定点，请先运行 calibrate_all_joints()")
            return

        print(f"\n{'='*60}")
        print(f"标定点列表 ({len(self.calibration_points)} 个)")
        print(f"{'='*60}")

        for i, cp in enumerate(self.calibration_points):
            print(f"\n[{i+1}] {cp.timestamp} | {cp.arm} | 高度: {cp.height_level}")
            print(f"    关节数: {len(cp.sensitivities)}")
            for s in cp.sensitivities:
                print(f"      - {s.joint_name}: ({s.pixel_dx_per_deg:.2f}, {s.pixel_dy_per_deg:.2f}) px/deg")

    # ==================== 方案B: DH参数接口 ====================

    def set_dh_parameters(self, dh_params: List[Dict]):
        """
        设置DH参数 (方案B)

        Args:
            dh_params: DH参数列表
        """
        self.arm_config.dh_params = dh_params
        self.kinematics_model = DHKinematicsModel(dh_params)
        print("✓ 已更新DH参数")
        self._save_arm_config()

    def _save_arm_config(self):
        """保存手臂配置 (包括DH参数)"""
        # TODO: 实现配置持久化
        pass

    # ==================== XY对齐 (使用标定数据) ====================

    def compute_joint_adjustments(self, pixel_error_x: float, pixel_error_y: float,
                                  current_joints: np.ndarray) -> Dict[int, float]:
        """
        根据像素误差计算关节调整量

        使用方案A (多点标定插值) 或方案B (雅可比)

        Args:
            pixel_error_x: X方向像素误差
            pixel_error_y: Y方向像素误差
            current_joints: 当前关节角度

        Returns:
            {joint_idx: adjustment_degrees}
        """
        adjustments = {}

        # 优先使用方案B (雅可比)
        if self.arm_config.dh_params and isinstance(self.kinematics_model, DHKinematicsModel):
            return self._compute_adjustments_jacobian(pixel_error_x, pixel_error_y, current_joints)

        # 使用方案A (多点标定插值)
        return self._compute_adjustments_interpolation(pixel_error_x, pixel_error_y, current_joints)

    def _compute_adjustments_interpolation(self, pixel_error_x: float, pixel_error_y: float,
                                           current_joints: np.ndarray) -> Dict[int, float]:
        """方案A: 使用插值灵敏度计算调整量"""
        sensitivities = self.get_interpolated_sensitivities(current_joints)

        if not sensitivities:
            # 没有标定数据，使用默认值
            print("警告: 无标定数据，使用默认关节")
            return {
                self.arm_config.primary_joints[0]: pixel_error_x * 0.1,
                self.arm_config.primary_joints[1]: pixel_error_y * 0.1
            }

        adjustments = {}
        for s in sensitivities:
            # 目标: pixel_dx_per_deg * delta_deg = -pixel_error_x
            # delta_deg = -pixel_error_x / pixel_dx_per_deg
            if abs(s.pixel_dx_per_deg) > 0.1:  # 避免除零
                delta_x = -pixel_error_x / s.pixel_dx_per_deg
            else:
                delta_x = 0

            if abs(s.pixel_dy_per_deg) > 0.1:
                delta_y = -pixel_error_y / s.pixel_dy_per_deg
            else:
                delta_y = 0

            # 组合X和Y方向的调整 (简单平均，可根据实际情况优化)
            delta = (delta_x + delta_y) / 2 * self.gain

            # 限制单步调整量
            delta = np.clip(delta, -2.0, 2.0)

            adjustments[s.joint_idx] = delta

        return adjustments

    def _compute_adjustments_jacobian(self, pixel_error_x: float, pixel_error_y: float,
                                      current_joints: np.ndarray) -> Dict[int, float]:
        """方案B: 使用雅可比计算调整量"""
        # TODO: 实现雅克比方法
        # 1. 获取末端位姿
        # 2. 计算雅可比矩阵
        # 3. 求解关节速度/位移

        raise NotImplementedError("雅可比方法待实现，请提供DH参数")

    def apply_joint_adjustments(self, adjustments: Dict[int, float]) -> bool:
        """应用关节调整"""
        if self.passive_mode:
            print("✗ 被动模式下无法应用关节调整")
            return False

        joints = self.get_joint_states()
        if joints is None or len(joints) < 16:
            return False

        target = joints.copy()
        for jidx, delta in adjustments.items():
            target[jidx] += delta

        # 平滑移动
        self._smooth_move_all_joints(target)
        return True

    def _smooth_move_all_joints(self, target_joints: np.ndarray, steps: int = None):
        """平滑移动所有关节"""
        if self.passive_mode:
            print("✗ 被动模式下无法移动关节")
            return False

        if steps is None:
            steps = self.smooth_steps

        current_joints = self.get_joint_states()

        if current_joints is None or len(current_joints) < 16:
            return False

        for step in range(1, steps + 1):
            alpha = step / steps
            alpha = alpha * alpha * (3 - 2 * alpha)  # ease-in-out

            interp = current_joints * (1 - alpha) + target_joints * alpha
            self.robot.send_action({'action': interp.tolist()})
            time.sleep(self.smooth_delay)

        return True

    # ==================== 对齐流程 ====================

    def align_xy(self, tolerance_mm: float = None) -> bool:
        """XY对齐 (使用标定数据)"""
        if tolerance_mm is None:
            tolerance_mm = self.tolerance_mm

        print(f"\nXY对齐 - 目标精度: {tolerance_mm}mm")

        # 检查标定数据
        if not self.calibration_points:
            print("警告: 无多点标定数据，精度可能受限")
            print("建议运行 calibrate_all_joints() 进行标定")

        for i in range(self.max_iterations):
            print(f"\n[对齐 {i+1}/{self.max_iterations}]")

            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)

            wp = sum(1 for m in [state.workpiece_top, state.workpiece_bottom] if m)
            sl = sum(1 for m in [state.slot_top, state.slot_bottom] if m)

            print(f"  工件: {wp}/2, 卡槽: {sl}/2")

            if not state.workpiece_detected or not state.slot_detected:
                print("  标记不完整")
                continue

            # 计算误差
            mm_x = state.offset_x * self.pixel_to_mm_ratio
            mm_y = state.offset_y * self.pixel_to_mm_ratio
            error_mm = np.sqrt(mm_x**2 + mm_y**2)

            print(f"  误差: ({mm_x:.2f}, {mm_y:.2f})mm, 总: {error_mm:.2f}mm")

            if error_mm < tolerance_mm:
                print(f"\n✓ 对齐完成: {error_mm:.2f}mm < {tolerance_mm}mm")
                return True

            # 获取当前关节状态
            current_joints = self.get_joint_states()

            if current_joints is None:
                print("✗ 无法获取关节状态")
                continue

            # 计算关节调整量 (使用标定数据)
            adjustments = self.compute_joint_adjustments(
                state.offset_x, state.offset_y, current_joints
            )

            print(f"  关节调整: {adjustments}")

            # 应用调整
            if state.alignment_quality > 0.3:
                self.apply_joint_adjustments(adjustments)

            time.sleep(self.settle_time)

        print("\n✗ 对齐未完成")
        return False

    # ==================== 兼容旧接口 ====================

    def calibrate(self, move_distance_mm: float = 5.0) -> Tuple[bool, float]:
        """像素-毫米标定 (兼容旧接口)"""
        print("\n" + "="*50)
        print("像素-毫米标定")
        print("="*50)

        print("\n[1/4] 采集初始位置...")
        img1 = self.camera.read()

        print(f"\n[2/4] 请将机器人沿X方向精确移动 {move_distance_mm}mm")
        input("    移动完成后按 Enter...")

        print("\n[3/4] 采集移动后位置...")
        img2 = self.camera.read()

        print("\n[4/4] 计算标定参数...")

        pixel_dx, _ = self._compute_pixel_shift(img1, img2)

        if pixel_dx is None or abs(pixel_dx) < 1:
            print("✗ 像素偏移太小")
            return False, 0

        pixel_offset = abs(pixel_dx)
        ratio = move_distance_mm / pixel_offset

        print(f"\n计算结果:")
        print(f"  像素偏移: {pixel_offset:.1f} pixels")
        print(f"  转换比例: {ratio:.4f} mm/pixel")

        self.calibration_history.append({
            'ratio': ratio,
            'pixel_offset': pixel_offset,
            'move_distance': move_distance_mm,
            'arm': self.arm,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        self._save_calibration_history()

        self.pixel_to_mm_ratio = ratio

        # 更新现有标定点的mm灵敏度
        for cp in self.calibration_points:
            for s in cp.sensitivities:
                s.mm_dx_per_deg = s.pixel_dx_per_deg * ratio
                s.mm_dy_per_deg = s.pixel_dy_per_deg * ratio
        self._save_calibration_points()

        print("\n✓ 标定完成")
        return True, ratio

    def _load_calibration_history(self):
        """加载标定历史"""
        path = Path(__file__).parent / "calibration_history.json"
        if path.exists():
            with open(path, 'r') as f:
                self.calibration_history = json.load(f)

            if self.calibration_history:
                latest = self.calibration_history[-1]
                self.pixel_to_mm_ratio = latest.get('ratio', 0.5)

    def _save_calibration_history(self):
        """保存标定历史"""
        path = Path(__file__).parent / "calibration_history.json"
        with open(path, 'w') as f:
            json.dump(self.calibration_history, f, indent=2)

    def show_calibration_history(self):
        """显示标定历史"""
        if not self.calibration_history:
            print("\n暂无标定历史")
            return

        print("\n" + "="*50)
        print("标定历史 (像素-毫米)")
        print("="*50)

        for i, cal in enumerate(self.calibration_history[-5:]):
            print(f"\n[{i+1}] {cal['timestamp']}")
            print(f"    手臂: {cal['arm']}")
            print(f"    比例: {cal['ratio']:.4f} mm/pixel")

    # ==================== 预设位置 ====================

    def save_preset(self, name: str):
        """保存当前位置为预设"""
        joints = self.get_joint_states()

        if joints is None:
            print("✗ 无法获取关节位置")
            if self.passive_mode:
                print("  请确认示教程序已启动并启用状态共享")
            return False

        self.presets[name] = joints.copy()
        self._save_presets()

        print(f"✓ 预设位置已保存: {name}")
        return True

    def load_preset(self, name: str) -> bool:
        """加载预设位置"""
        if name not in self.presets:
            print(f"✗ 预设不存在: {name}")
            return False

        joints = self.presets[name]
        self.robot.send_action({'action': joints.tolist()})

        print(f"✓ 已移动到预设位置: {name}")
        return True

    def _load_presets(self):
        """加载预设"""
        path = Path(__file__).parent / "presets.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.presets = {k: np.array(v) for k, v in data.items()}

    def _save_presets(self):
        """保存预设"""
        path = Path(__file__).parent / "presets.json"
        with open(path, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.presets.items()}, f, indent=2)

    def list_presets(self):
        """列出所有预设"""
        if not self.presets:
            print("\n暂无预设位置")
            return

        print("\n预设位置列表:")
        for name in self.presets:
            print(f"  - {name}")

    # ==================== 高度控制 ====================

    def raise_height(self, step: float = 2.0):
        """上升 (粗调)"""
        if self.passive_mode:
            print("✗ 被动模式下无法调整高度")
            return False

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.height_joint_idx] -= step
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(self.settle_time)
        return True

    def lower_height(self, step: float = 2.0):
        """下降 (粗调)"""
        if self.passive_mode:
            print("✗ 被动模式下无法调整高度")
            return False

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.height_joint_idx] += step
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(self.settle_time)
        return True

    def auto_adjust_height(self, max_attempts: int = 10) -> bool:
        """自动调整高度"""
        print("\n自动调整高度...")

        for i in range(max_attempts):
            print(f"\n[高度 {i+1}/{max_attempts}]")

            time.sleep(0.2)
            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)

            wp = sum(1 for m in [state.workpiece_top, state.workpiece_bottom] if m)
            sl = sum(1 for m in [state.slot_top, state.slot_bottom] if m)

            print(f"  工件: {wp}/2, 卡槽: {sl}/2")

            if wp >= 2 and sl >= 2:
                print("\n✓ 高度合适")
                return True

            if sl < 2:
                print("  上升中...")
                self.raise_height()
            else:
                self.raise_height(1.0)

        print("\n✗ 高度调整未完成")
        return False

    # ==================== 夹爪控制 ====================

    def open_gripper(self):
        """打开夹爪"""
        if self.passive_mode:
            print("✗ 被动模式下无法控制夹爪")
            return False

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.arm_config.gripper_idx] = self.arm_config.gripper_open
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(0.5)
        print("✓ 夹爪已打开")
        return True

    def close_gripper(self, position: float = None):
        """闭合夹爪"""
        if self.passive_mode:
            print("✗ 被动模式下无法控制夹爪")
            return False

        if position is None:
            position = self.arm_config.gripper_close

        joints = self.get_joint_states()

        if joints is None or len(joints) < 16:
            return False

        joints[self.arm_config.gripper_idx] = position
        self.robot.send_action({'action': joints.tolist()})
        time.sleep(0.5)
        print("✓ 夹爪已闭合")
        return True

    # ==================== 自动放置 ====================

    def auto_place(self, lower_steps: int = 5, lower_step_size: float = 2.0):
        """自动放置流程"""
        print("\n自动放置...")

        print("\n[1/3] 下降到放置高度")
        for i in range(lower_steps):
            print(f"  下降 {i+1}/{lower_steps}")
            self.lower_height(lower_step_size)
            time.sleep(0.2)

        print("\n[2/3] 松开夹爪")
        self.open_gripper()

        print("\n[3/3] 抬起")
        for i in range(3):
            self.raise_height(lower_step_size * 2)
            time.sleep(0.2)

        print("\n✓ 自动放置完成")

    # ==================== 完整流程 ====================

    def run_full_sequence(self, tolerance_mm: float = 2.0, auto_place: bool = False) -> bool:
        """完整对齐流程"""
        print("\n" + "="*50)
        print("开始精准对齐")
        print("="*50)

        print("\n[步骤1] 自动高度调整")
        height_ok = self.auto_adjust_height()

        if not height_ok:
            print("请手动调整高度")

        print("\n[步骤2] XY对齐")
        align_ok = self.align_xy(tolerance_mm)

        if auto_place and align_ok:
            print("\n[步骤3] 自动放置")
            self.auto_place()

        return align_ok

    def test_detection(self):
        """测试检测"""
        print("\n检测测试")
        print("按 'q' 退出, 'r' 上升, 'l' 下降, 'o' 打开夹爪, 'c' 闭合夹爪")

        while True:
            image = self.camera.read()
            state = self.detector.detect_dual_marker_state(image)
            vis = self.detector.visualize(image, state)

            cv2.putText(vis, f"Arm: {self.arm}", (vis.shape[1]-100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示标定状态
            cal_status = f"Cal Points: {len(self.calibration_points)}"
            cv2.putText(vis, cal_status, (vis.shape[1]-150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Detection Test", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.raise_height()
            elif key == ord('l'):
                self.lower_height()
            elif key == ord('o'):
                self.open_gripper()
            elif key == ord('c'):
                self.close_gripper()

        cv2.destroyAllWindows()
