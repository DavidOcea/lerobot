"""
对齐控制器 (Alignment Controller)

使用手眼标定或灵敏度方法进行精确对齐。
"""

import time
import numpy as np
from typing import Optional, Tuple, Callable

from precision_place.models.marker import DualMarkerState
from precision_place.models.state import AlignmentResult, TCPAdjustment
from precision_place.core.detector import DualPointDetector


class PrecisionAligner:
    """精度对齐控制器基类"""

    def __init__(self, detector: DualPointDetector):
        self.detector = detector
        self.alignment_threshold_pixel = 5.0
        self.max_iterations = 20


class HandEyeAligner(PrecisionAligner):
    """基于手眼标定的对齐控制器"""

    def __init__(self, detector: DualPointDetector, transformer):
        """
        初始化手眼对齐控制器

        Args:
            detector: 标记检测器
            transformer: CoordinateTransformer实例
        """
        super().__init__(detector)
        self.transformer = transformer
        self.step_scale = 0.8  # 步长缩放
        self.settle_time = 0.3  # 稳定时间

    def compute_adjustment(self,
                           pixel_offset: Tuple[float, float],
                           depth: float) -> Tuple[TCPAdjustment, dict]:
        """
        计算TCP调整量

        Args:
            pixel_offset: 像素偏移 (du, dv)
            depth: 深度 (米)

        Returns:
            (TCPAdjustment, info) 调整量和详细信息
        """
        tcp_adj, info = self.transformer.compute_alignment_adjustment(
            pixel_offset, depth
        )

        adjustment = TCPAdjustment(
            dx=tcp_adj[0],
            dy=tcp_adj[1],
            dz=tcp_adj[2]
        )

        return adjustment, info

    def align(self,
              get_image: Callable[[], np.ndarray],
              get_tcp_pose: Callable[[], Tuple[np.ndarray, np.ndarray]],
              get_depth: Callable[[], float],
              move_tcp: Callable[[np.ndarray], bool],
              on_progress: Callable = None) -> AlignmentResult:
        """
        执行对齐循环

        Args:
            get_image: 获取图像的函数
            get_tcp_pose: 获取TCP位姿的函数 () -> (position, quaternion)
            get_depth: 获取深度的函数 () -> depth_m
            move_tcp: 移动TCP的函数 (position_delta) -> success
            on_progress: 进度回调 (iteration, pixel_error, adjustment_mm)

        Returns:
            AlignmentResult 对齐结果
        """
        for iteration in range(self.max_iterations):
            # 1. 获取图像并检测
            image = get_image()
            if image is None:
                continue

            state = self.detector.detect_triple_marker_state(image)

            if not state.workpiece_detected or not state.slot_detected:
                continue

            pixel_error = state.pixel_error

            # 检查是否已对齐
            if pixel_error < self.alignment_threshold_pixel:
                return AlignmentResult(
                    success=True,
                    iterations=iteration + 1,
                    final_pixel_error=pixel_error,
                    converged=True,
                    message="对齐成功"
                )

            # 2. 获取TCP位姿和深度
            tcp_pos, tcp_rot = get_tcp_pose()
            self.transformer.set_tcp_pose(tcp_pos, tcp_rot, "quaternion")
            depth = get_depth()

            # 3. 计算调整量
            adjustment, info = self.compute_adjustment(
                (state.offset_x, state.offset_y),
                depth
            )

            # 应用缩放
            adjustment.dx *= self.step_scale
            adjustment.dy *= self.step_scale
            adjustment.dz *= self.step_scale

            # 检查调整量是否过小
            if adjustment.magnitude_mm < 0.5:
                return AlignmentResult(
                    success=True,
                    iterations=iteration + 1,
                    final_pixel_error=pixel_error,
                    converged=True,
                    message="调整量过小，已收敛"
                )

            # 4. 执行移动
            delta = np.array([adjustment.dx, adjustment.dy, adjustment.dz])
            success = move_tcp(delta)

            if not success:
                return AlignmentResult(
                    success=False,
                    iterations=iteration + 1,
                    message="移动失败"
                )

            # 进度回调
            if on_progress:
                on_progress(iteration + 1, pixel_error, adjustment.magnitude_mm)

            # 等待稳定
            time.sleep(self.settle_time)

        return AlignmentResult(
            success=False,
            iterations=self.max_iterations,
            converged=False,
            message="达到最大迭代次数"
        )


class SensitivityAligner(PrecisionAligner):
    """基于灵敏度的对齐控制器（传统方法）"""

    def __init__(self, detector: DualPointDetector, sensitivity_calibrator):
        """
        初始化灵敏度对齐控制器

        Args:
            detector: 标记检测器
            sensitivity_calibrator: SensitivityCalibrator实例
        """
        super().__init__(detector)
        self.calibrator = sensitivity_calibrator

    def compute_joint_adjustment(self,
                                  pixel_offset: Tuple[float, float],
                                  current_joints: np.ndarray) -> np.ndarray:
        """
        计算关节角度调整量

        Args:
            pixel_offset: 像素偏移
            current_joints: 当前关节角度

        Returns:
            关节角度调整量
        """
        sensitivities = self.calibrator.get_interpolated_sensitivities(current_joints)

        # 使用最小二乘求解关节调整量
        # pixel_offset = J * joint_delta
        # J 是灵敏度矩阵

        # 简化：只使用主要控制关节
        joint_delta = np.zeros(len(current_joints))

        # TODO: 实现完整的雅可比求解

        return joint_delta