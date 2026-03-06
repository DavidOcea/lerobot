"""
Precision Place - 毫米级精准放置模块

使用方法:
    python precision_place/run.py

功能:
    - 双标记点对齐 (工件2绿 + 卡槽2红)
    - 自动高度调整
    - XY对齐
    - 分层下降放置

配置:
    - 主用相机: right_wrist_cam2 (索引8)
    - 工件标记: 绿色
    - 卡槽标记: 红色
    - 精度: ±1-2mm
"""

from .dual_point_alignment import (
    DualPointDetector,
    PrecisionPlaceController,
    DualMarkerState,
    Marker
)

__all__ = [
    'DualPointDetector',
    'PrecisionPlaceController',
    'DualMarkerState',
    'Marker'
]
