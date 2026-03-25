"""
Precision Place - 毫米级精准放置模块

使用方法:
    python precision_place/run.py

功能:
    - 手眼标定 (ChArUco板 + Tsai-Lenz算法)
    - 双标记点对齐 (工件3绿 + 卡槽3红)
    - XY对齐 (外参矩阵精确变换)
    - Z轴精确控制 (双目立体视觉 ±0.5mm)
    - 旋转对齐 (三标记姿态估计)

配置:
    - 主用相机: right_wrist (索引6)
    - 副相机: right_wrist2 (索引8)
    - 工件标记: 绿色
    - 卡槽标记: 红色
    - XY精度: ±1mm (手眼标定方法)
    - Z轴精度: ±0.5mm
"""

# 新模块结构
from precision_place.models.marker import Marker, DualMarkerState
from precision_place.models.calibration_data import (
    JointSensitivity, CalibrationPoint, ArmConfig, ARM_CONFIGS
)
from precision_place.models.state import DetectionResult, AlignmentResult

from precision_place.core.detector import DualPointDetector
from precision_place.core.aligner import HandEyeAligner, SensitivityAligner

from precision_place.calibration.hand_eye import HandEyeCalibrator
from precision_place.calibration.forward_kinematics import ForwardKinematics, create_fk_from_urdf
from precision_place.calibration.coordinate_transform import CoordinateTransformer

# 保持向后兼容
from precision_place.dual_point_alignment import PrecisionPlaceController
from precision_place.z_axis_controller import (
    ZAxisController,
    DepthEstimator,
    DepthEstimate,
    MarkerWithSize,
    CameraCalibration,
    StereoCalibration,
    ZJointSensitivity
)

__all__ = [
    # 数据模型
    'Marker',
    'DualMarkerState',
    'JointSensitivity',
    'CalibrationPoint',
    'ArmConfig',
    'ARM_CONFIGS',
    'DetectionResult',
    'AlignmentResult',

    # 核心模块
    'DualPointDetector',
    'HandEyeAligner',
    'SensitivityAligner',

    # 标定模块
    'HandEyeCalibrator',
    'ForwardKinematics',
    'create_fk_from_urdf',
    'CoordinateTransformer',

    # 控制器（向后兼容）
    'PrecisionPlaceController',

    # Z轴控制
    'ZAxisController',
    'DepthEstimator',
    'DepthEstimate',
    'MarkerWithSize',
    'CameraCalibration',
    'StereoCalibration',
    'ZJointSensitivity',
]