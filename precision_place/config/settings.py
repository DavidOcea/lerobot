"""
配置管理 (Configuration Management)

统一管理所有配置参数。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from pathlib import Path
import yaml


@dataclass
class CameraConfig:
    """相机配置"""
    # 相机索引
    indices: Dict[str, int] = field(default_factory=lambda: {
        'head': 0,
        'left_wrist': 2,
        'left_wrist2': 4,
        'right_wrist': 6,
        'right_wrist2': 8
    })
    # 默认内参（应通过标定获取真实值）
    intrinsic_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [500.0, 0, 320.0],
        [0, 500.0, 240.0],
        [0, 0, 1]
    ], dtype=np.float64))
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    # 图像尺寸
    image_width: int = 640
    image_height: int = 480


@dataclass
class MarkerConfig:
    """标记配置"""
    workpiece_color: str = "green"
    slot_color: str = "red"
    diameter_mm: float = 15.0
    min_area: int = 50
    max_area: int = 5000
    # 颜色HSV范围
    color_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        'green': ((35, 50, 50), (85, 255, 255)),
        'red': ((0, 50, 50), (10, 255, 255)),
        'blue': ((100, 50, 50), (130, 255, 255)),
    })


@dataclass
class AlignmentConfig:
    """对齐配置"""
    tolerance_mm: float = 2.0
    tolerance_pixel: float = 5.0
    max_iterations: int = 20
    step_scale: float = 0.8  # 步长缩放，避免过冲
    settle_time: float = 0.3  # 移动后稳定时间
    # 对齐方法: "hand_eye" 或 "sensitivity"
    method: str = "hand_eye"


@dataclass
class CalibrationConfig:
    """标定配置"""
    # ChArUco板参数
    charuco_squares_x: int = 5
    charuco_squares_y: int = 7
    charuco_square_length: float = 0.03  # 米
    charuco_marker_length: float = 0.022  # 米
    # 标定要求
    min_poses: int = 10
    recommended_poses: int = 30
    rmse_threshold: float = 1.5  # 像素


@dataclass
class PrecisionPlaceConfig:
    """精准放置系统总配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # 文件路径
    calibration_file: str = "hand_eye_extrinsic.yaml"
    urdf_path: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> 'PrecisionPlaceConfig':
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        if 'camera' in data:
            cam = data['camera']
            config.camera.indices = cam.get('indices', config.camera.indices)
            config.camera.image_width = cam.get('image_width', config.camera.image_width)
            config.camera.image_height = cam.get('image_height', config.camera.image_height)

        if 'marker' in data:
            marker = data['marker']
            config.marker.workpiece_color = marker.get('workpiece_color', config.marker.workpiece_color)
            config.marker.slot_color = marker.get('slot_color', config.marker.slot_color)
            config.marker.diameter_mm = marker.get('diameter_mm', config.marker.diameter_mm)
            config.marker.min_area = marker.get('min_area', config.marker.min_area)
            config.marker.max_area = marker.get('max_area', config.marker.max_area)

        if 'alignment' in data:
            align = data['alignment']
            config.alignment.tolerance_mm = align.get('tolerance_mm', config.alignment.tolerance_mm)
            config.alignment.tolerance_pixel = align.get('tolerance_pixel', config.alignment.tolerance_pixel)
            config.alignment.max_iterations = align.get('max_iterations', config.alignment.max_iterations)
            config.alignment.step_scale = align.get('step_scale', config.alignment.step_scale)
            config.alignment.method = align.get('method', config.alignment.method)

        if 'calibration' in data:
            calib = data['calibration']
            config.calibration.min_poses = calib.get('min_poses', config.calibration.min_poses)
            config.calibration.rmse_threshold = calib.get('rmse_threshold', config.calibration.rmse_threshold)

        config.calibration_file = data.get('calibration_file', config.calibration_file)
        config.urdf_path = data.get('urdf_path', config.urdf_path)

        return config

    def save(self, path: str):
        """保存配置到YAML文件"""
        data = {
            'camera': {
                'indices': self.camera.indices,
                'image_width': self.camera.image_width,
                'image_height': self.camera.image_height,
            },
            'marker': {
                'workpiece_color': self.marker.workpiece_color,
                'slot_color': self.marker.slot_color,
                'diameter_mm': self.marker.diameter_mm,
                'min_area': self.marker.min_area,
                'max_area': self.marker.max_area,
            },
            'alignment': {
                'tolerance_mm': self.alignment.tolerance_mm,
                'tolerance_pixel': self.alignment.tolerance_pixel,
                'max_iterations': self.alignment.max_iterations,
                'step_scale': self.alignment.step_scale,
                'method': self.alignment.method,
            },
            'calibration': {
                'min_poses': self.calibration.min_poses,
                'rmse_threshold': self.calibration.rmse_threshold,
            },
            'calibration_file': self.calibration_file,
            'urdf_path': self.urdf_path,
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


def get_default_config() -> PrecisionPlaceConfig:
    """获取默认配置"""
    return PrecisionPlaceConfig()


def load_config(path: Optional[str] = None) -> PrecisionPlaceConfig:
    """加载配置（优先使用指定路径，否则使用默认配置）"""
    if path is None:
        # 尝试默认配置路径
        default_path = Path(__file__).parent.parent / "configs" / "precision_config.yaml"
        if default_path.exists():
            return PrecisionPlaceConfig.load(str(default_path))
        return get_default_config()
    return PrecisionPlaceConfig.load(path)