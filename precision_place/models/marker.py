"""
标记数据模型 (Marker Data Models)

定义标记检测相关的数据结构。
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class Marker:
    """单个标记"""
    x: float
    y: float
    color: str
    confidence: float


@dataclass
class DualMarkerState:
    """双标记状态（工件+卡槽）"""
    workpiece_1: Optional[Marker] = None
    workpiece_2: Optional[Marker] = None
    workpiece_3: Optional[Marker] = None
    slot_1: Optional[Marker] = None
    slot_2: Optional[Marker] = None
    slot_3: Optional[Marker] = None
    offset_x: float = 0
    offset_y: float = 0
    rotation_error: float = 0
    workpiece_detected: bool = False
    slot_detected: bool = False
    alignment_quality: float = 0
    # 退化模式标记
    degraded_mode: bool = False
    degraded_reason: str = ""
    predicted_slot_center: Optional[Tuple[float, float]] = None

    @property
    def workpiece_markers(self) -> List[Optional[Marker]]:
        """获取工件标记列表"""
        return [self.workpiece_1, self.workpiece_2, self.workpiece_3]

    @property
    def slot_markers(self) -> List[Optional[Marker]]:
        """获取卡槽标记列表"""
        return [self.slot_1, self.slot_2, self.slot_3]

    @property
    def slot_marker_count(self) -> int:
        """检测到的卡槽标记数量"""
        return sum(1 for m in self.slot_markers if m)

    @property
    def workpiece_marker_count(self) -> int:
        """检测到的工件标记数量"""
        return sum(1 for m in self.workpiece_markers if m)

    @property
    def workpiece_center(self) -> Tuple[float, float]:
        """计算工件中心"""
        markers = [m for m in self.workpiece_markers if m]
        if not markers:
            return (0, 0)
        x = sum(m.x for m in markers) / len(markers)
        y = sum(m.y for m in markers) / len(markers)
        return (x, y)

    @property
    def slot_center(self) -> Tuple[float, float]:
        """计算卡槽中心"""
        markers = [m for m in self.slot_markers if m]
        if not markers:
            if self.predicted_slot_center:
                return self.predicted_slot_center
            return (0, 0)
        x = sum(m.x for m in markers) / len(markers)
        y = sum(m.y for m in markers) / len(markers)
        return (x, y)

    @property
    def pixel_error(self) -> float:
        """计算像素误差"""
        return (self.offset_x**2 + self.offset_y**2) ** 0.5

    @property
    def is_aligned(self, threshold: float = 5.0) -> bool:
        """检查是否已对齐"""
        return self.pixel_error < threshold