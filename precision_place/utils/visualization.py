"""
可视化工具 (Visualization Utilities)

提供常用的绘图和可视化功能。
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


def draw_markers(image: np.ndarray,
                 markers: List,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 radius: int = 5,
                 labels: bool = True,
                 label_prefix: str = "M") -> np.ndarray:
    """
    在图像上绘制标记点

    Args:
        image: 输入图像
        markers: 标记列表 (需要有x, y属性)
        color: 颜色 (BGR)
        radius: 圆点半径
        labels: 是否显示标签
        label_prefix: 标签前缀

    Returns:
        绘制后的图像
    """
    result = image.copy()

    for i, m in enumerate(markers):
        if m is None:
            continue

        x, y = int(m.x), int(m.y)
        cv2.circle(result, (x, y), radius, color, -1)

        if labels:
            cv2.putText(result, f"{label_prefix}{i+1}",
                       (x - 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result


def draw_offset_arrow(image: np.ndarray,
                      start: Tuple[float, float],
                      end: Tuple[float, float],
                      color: Tuple[int, int, int] = (255, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    绘制偏移箭头

    Args:
        image: 输入图像
        start: 起点 (x, y)
        end: 终点 (x, y)
        color: 颜色
        thickness: 线宽

    Returns:
        绘制后的图像
    """
    result = image.copy()
    cv2.arrowedLine(result,
                   (int(start[0]), int(start[1])),
                   (int(end[0]), int(end[1])),
                   color, thickness)
    return result


def draw_alignment_info(image: np.ndarray,
                        offset_x: float,
                        offset_y: float,
                        target_offset: Tuple[float, float] = (0, 0),
                        mode: str = "IDLE") -> np.ndarray:
    """
    绘制对齐信息

    Args:
        image: 输入图像
        offset_x: X偏移
        offset_y: Y偏移
        target_offset: 目标偏移
        mode: 当前模式

    Returns:
        绘制后的图像
    """
    result = image.copy()

    # 计算修正偏移
    error_x = offset_x - target_offset[0]
    error_y = offset_y - target_offset[1]
    error_dist = np.sqrt(error_x**2 + error_y**2)

    # 显示信息
    y_offset = 30
    cv2.putText(result, f"Raw: ({offset_x:.1f}, {offset_y:.1f}) px",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_offset += 25

    if target_offset[0] != 0 or target_offset[1] != 0:
        cv2.putText(result, f"Target: ({target_offset[0]:.1f}, {target_offset[1]:.1f}) px",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        y_offset += 20

    cv2.putText(result, f"Error: ({error_x:.1f}, {error_y:.1f}) px",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_offset += 25

    cv2.putText(result, f"Dist: {error_dist:.1f} px",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_offset += 25

    # 模式显示
    mode_colors = {
        "AUTO": (0, 255, 0),
        "MANUAL": (0, 255, 255),
        "IDLE": (128, 128, 128),
        "DEGRADED": (0, 165, 255)
    }
    mode_color = mode_colors.get(mode, (128, 128, 128))
    cv2.putText(result, f"Mode: {mode}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    return result


def create_legend(image_shape: Tuple[int, int],
                  items: List[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
    """
    创建图例

    Args:
        image_shape: 图像尺寸 (height, width)
        items: 图例项列表 [(label, color), ...]

    Returns:
        图例图像
    """
    legend = np.ones((len(items) * 25 + 10, 150, 3), dtype=np.uint8) * 255

    for i, (label, color) in enumerate(items):
        y = 20 + i * 25
        cv2.circle(legend, (15, y), 5, color, -1)
        cv2.putText(legend, label, (30, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return legend