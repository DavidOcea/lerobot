"""
标定数据模型 (Calibration Data Models)

定义标定相关的数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class JointSensitivity:
    """单个关节的灵敏度数据"""
    joint_idx: int
    joint_name: str
    # 关节移动1度时，末端在相机中的像素变化
    pixel_dx_per_deg: float = 0.0
    pixel_dy_per_deg: float = 0.0
    # 关节移动1度时，末端的实际毫米移动
    mm_dx_per_deg: float = 0.0
    mm_dy_per_deg: float = 0.0
    # 标定时的关节角度
    calibration_angles: List[float] = field(default_factory=list)


@dataclass
class CalibrationPoint:
    """单个标定点（特定姿态下的灵敏度）"""
    height_level: str  # "high", "medium", "low"
    joint_states: List[float]  # 标定时的所有关节状态
    sensitivities: List[JointSensitivity]
    pixel_to_mm: float = 0.5  # 该高度下的像素-毫米转换比例
    timestamp: str = ""
    arm: str = "right"
    camera_name: str = ""


@dataclass
class ArmConfig:
    """手臂配置"""
    name: str
    camera_name: str
    camera_index: int
    # 第二相机（用于双目Z轴控制）
    camera2_name: str = ""
    camera2_index: int = -1
    # 主要控制关节索引
    primary_joints: List[int] = field(default_factory=list)
    gripper_idx: int = 0
    gripper_open: float = 0.0
    gripper_close: float = 50.0
    # DH参数（方案B预留）
    dh_params: Optional[List[Dict]] = None
    # 相机方向翻转
    camera_flip: Dict[str, Tuple[bool, bool]] = field(default_factory=dict)


# 手臂配置 - 7关节标定
# 注意：相机索引取决于实际连接的USB端口
# 使用 ls /dev/video* 或 v4l2-ctl --list-devices 查看
ARM_CONFIGS = {
    'right': ArmConfig(
        name='right',
        camera_name='right_wrist',
        camera_index=4,  # 右手相机索引
        camera2_name='',  # 已拆除
        camera2_index=-1,  # 已拆除
        primary_joints=[7, 8, 9, 10, 11, 12, 14],
        gripper_idx=13,
        gripper_open=0.0,
        gripper_close=50.0,
        camera_flip={
            'right_wrist': (False, False),
        }
    ),
    'left': ArmConfig(
        name='left',
        camera_name='left_wrist',
        camera_index=2,  # 左手相机索引
        camera2_name='',  # 已拆除
        camera2_index=-1,  # 已拆除
        primary_joints=[0, 1, 2, 3, 4, 5, 14],
        gripper_idx=6,
        gripper_open=0.0,
        gripper_close=50.0,
        camera_flip={
            'left_wrist': (False, False),
        }
    )
}