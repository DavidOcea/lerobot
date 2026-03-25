"""
状态数据模型 (State Data Models)

定义系统状态和结果相关的数据结构。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """检测结果"""
    success: bool
    workpiece_count: int = 0
    slot_count: int = 0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation_error: float = 0.0
    quality: float = 0.0
    message: str = ""

    @property
    def pixel_error(self) -> float:
        return (self.offset_x**2 + self.offset_y**2) ** 0.5


@dataclass
class AlignmentResult:
    """对齐结果"""
    success: bool
    iterations: int = 0
    final_pixel_error: float = 0.0
    final_world_error_mm: float = 0.0
    converged: bool = False
    message: str = ""


@dataclass
class TCPAdjustment:
    """TCP调整量"""
    dx: float = 0.0  # 米
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0  # 弧度
    dpitch: float = 0.0
    dyaw: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dz])

    @property
    def magnitude_mm(self) -> float:
        return (self.dx**2 + self.dy**2 + self.dz**2) ** 0.5 * 1000


@dataclass
class CalibrationResult:
    """标定结果"""
    # 外参矩阵 (4x4)
    extrinsic_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    # 旋转矩阵 (3x3)
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    # 平移向量 (3x1, 米)
    translation_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 重投影误差 (像素)
    rmse_error: float = 0.0
    # 标定使用的姿态数量
    num_poses: int = 0
    # 标定方法
    method: str = "Tsai-Lenz"
    # 是否有效
    valid: bool = False