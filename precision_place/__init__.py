"""
Precision Place - 毫米级精准放置模块

使用方法:
    python precision_place/run.py

功能:
    - 双标记点对齐 (工件3绿 + 卡槽3红)
    - 自动高度调整
    - XY对齐
    - Z轴精确控制 (双目立体视觉 ±0.5mm)
    - 分层下降放置
    - 相机内参自动标定 (P3)
    - 多姿态标定插值 (P4)

配置:
    - 主用相机: right_wrist (索引6)
    - 副相机: right_wrist2 (索引8)
    - 工件标记: 绿色
    - 卡槽标记: 红色
    - XY精度: ±1-2mm
    - Z轴精度: ±0.5mm
"""

from .dual_point_alignment import (
    DualPointDetector,
    PrecisionPlaceController,
    DualMarkerState,
    Marker,
    ARM_CONFIGS
)

from .z_axis_controller import (
    ZAxisController,
    DepthEstimator,
    DepthEstimate,
    MarkerWithSize,
    CameraCalibration,
    StereoCalibration,
    ZJointSensitivity
)

__all__ = [
    'DualPointDetector',
    'PrecisionPlaceController',
    'DualMarkerState',
    'Marker',
    'ARM_CONFIGS',
    'ZAxisController',
    'DepthEstimator',
    'DepthEstimate',
    'MarkerWithSize',
    'CameraCalibration',
    'StereoCalibration',
    'ZJointSensitivity'
]
