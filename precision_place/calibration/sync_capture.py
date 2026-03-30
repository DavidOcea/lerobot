#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同步捕获模块 (Synchronized Capture)

解决图像与关节状态的时间同步问题，确保数据采集的一致性。

问题：
  - camera.read() 和 controller.get_joint_states() 是分开调用的
  - 两次调用之间存在时间差，导致数据不同步
  - 标定精度要求高时，这个差异会引入误差

解决方案：
  1. 并行读取：同时获取图像和关节状态
  2. 连续采样：丢弃旧数据，确保获取最新值
  3. 时间戳验证：记录采集时间，评估同步性
  4. 多次采样平均：减少随机噪声

使用方法：
    from precision_place.calibration.sync_capture import SynchronizedCapture

    sync = SynchronizedCapture(camera, controller, forward_kinematics)

    # 同步捕获（返回图像、关节状态、法兰位姿）
    success, image, joints, flange_pose = sync.capture()

    # 带验证的捕获
    success, image, joints, flange_pose = sync.capture_with_verification()
"""

import time
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future


@dataclass
class CaptureResult:
    """同步捕获结果"""
    success: bool = False
    image: Optional[np.ndarray] = None
    joints: Optional[np.ndarray] = None
    flange_position: Optional[np.ndarray] = None
    flange_rotation: Optional[np.ndarray] = None  # 四元数 [qx, qy, qz, qw]
    capture_time: float = 0.0  # 采集时间戳
    sync_delay_ms: float = 0.0  # 实际同步延迟（毫秒）
    error_message: str = ""


class SynchronizedCapture:
    """
    同步捕获器

    确保图像和关节状态的同步采集。

    Features:
        1. 并行读取图像和关节状态
        2. 多次采样丢弃旧数据
        3. 时间戳记录和验证
        4. 自动计算法兰位姿
    """

    def __init__(self,
                 camera: Any,
                 controller: Any,
                 forward_kinematics: Any = None,
                 warmup_frames: int = 3,
                 max_sync_delay_ms: float = 50.0):
        """
        初始化同步捕获器

        Args:
            camera: 相机对象，需有 read() 方法
            controller: 控制器对象，需有 get_joint_states() 方法
            forward_kinematics: 正运动学对象，用于计算法兰位姿
            warmup_frames: 预热帧数（丢弃旧数据）
            max_sync_delay_ms: 最大允许同步延迟（毫秒）
        """
        self.camera = camera
        self.controller = controller
        self.forward_kinematics = forward_kinematics
        self.warmup_frames = warmup_frames
        self.max_sync_delay_ms = max_sync_delay_ms

        # 线程池用于并行采集
        self._executor = ThreadPoolExecutor(max_workers=2)

        # 缓存最新数据
        self._latest_image = None
        self._latest_joints = None
        self._image_time = 0.0
        self._joints_time = 0.0

    def _read_image(self) -> Tuple[Optional[np.ndarray], float]:
        """读取图像并记录时间"""
        start_time = time.perf_counter()
        image = self.camera.read()
        end_time = time.perf_counter()
        return image, (start_time + end_time) / 2

    def _read_joints(self) -> Tuple[Optional[np.ndarray], float]:
        """读取关节状态并记录时间"""
        start_time = time.perf_counter()
        joints = self.controller.get_joint_states()
        end_time = time.perf_counter()
        return joints, (start_time + end_time) / 2

    def _compute_flange_pose(self, joints: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """计算法兰位姿"""
        if self.forward_kinematics is None:
            return None, None

        try:
            pose = self.forward_kinematics.compute(joints)
            return pose.get_position(), pose.quaternion
        except Exception as e:
            print(f"正运动学计算失败: {e}")
            return None, None

    def warmup(self):
        """
        预热相机和控制器

        丢弃旧数据，确保后续读取的是最新值
        """
        for _ in range(self.warmup_frames):
            self.camera.read()
            self.controller.get_joint_states()

    def capture(self) -> CaptureResult:
        """
        同步捕获图像和关节状态

        使用并行读取减少时间差

        Returns:
            CaptureResult: 捕获结果
        """
        result = CaptureResult()
        start_time = time.perf_counter()

        try:
            # 预热：丢弃旧数据
            self.warmup()

            # 并行读取
            image_future = self._executor.submit(self._read_image)
            joints_future = self._executor.submit(self._read_joints)

            # 等待结果
            image, image_time = image_future.result(timeout=2.0)
            joints, joints_time = joints_future.result(timeout=2.0)

            end_time = time.perf_counter()

            # 检查结果
            if image is None:
                result.error_message = "无法获取图像"
                return result

            if joints is None:
                result.error_message = "无法获取关节状态"
                if hasattr(self.controller, 'passive_mode') and self.controller.passive_mode:
                    result.error_message += "（请确认示教程序已启用 share_status=true）"
                return result

            # 计算同步延迟
            sync_delay_ms = abs(image_time - joints_time) * 1000

            # 计算法兰位姿
            flange_pos, flange_rot = self._compute_flange_pose(joints)

            # 填充结果
            result.success = True
            result.image = image
            result.joints = joints
            result.flange_position = flange_pos
            result.flange_rotation = flange_rot
            result.capture_time = (image_time + joints_time) / 2
            result.sync_delay_ms = sync_delay_ms

            if sync_delay_ms > self.max_sync_delay_ms:
                print(f"⚠ 同步延迟较大: {sync_delay_ms:.1f}ms (阈值: {self.max_sync_delay_ms}ms)")

            return result

        except Exception as e:
            result.error_message = f"捕获失败: {e}"
            return result

    def capture_multi_sample(self, num_samples: int = 3) -> CaptureResult:
        """
        多次采样取平均

        用于减少随机噪声，提高稳定性

        Args:
            num_samples: 采样次数

        Returns:
            CaptureResult: 平均后的捕获结果
        """
        results = []

        for i in range(num_samples):
            result = self.capture()
            if result.success:
                results.append(result)
            time.sleep(0.05)  # 间隔50ms

        if not results:
            return CaptureResult(success=False, error_message="所有采样失败")

        # 图像取最后一次（或中间值）
        result = CaptureResult()
        result.success = True
        result.image = results[len(results) // 2].image

        # 关节状态取平均
        joints_list = [r.joints for r in results]
        result.joints = np.mean(joints_list, axis=0)

        # 法兰位姿重新计算
        if self.forward_kinematics:
            result.flange_position, result.flange_rotation = self._compute_flange_pose(result.joints)

        # 同步延迟取平均
        result.sync_delay_ms = np.mean([r.sync_delay_ms for r in results])

        return result

    def capture_with_verification(self) -> CaptureResult:
        """
        带验证的捕获

        如果同步延迟过大，会重新捕获

        Returns:
            CaptureResult: 捕获结果
        """
        max_retries = 3
        last_error = "未知错误"

        for attempt in range(max_retries):
            result = self.capture()

            if not result.success:
                last_error = result.error_message or "捕获失败"
                print(f"  重试 {attempt + 1}/{max_retries} ({last_error})")
                time.sleep(0.1)
                continue

            if result.sync_delay_ms <= self.max_sync_delay_ms:
                return result

            last_error = f"同步延迟超标: {result.sync_delay_ms:.1f}ms"
            print(f"  重试 {attempt + 1}/{max_retries} ({last_error})")
            time.sleep(0.1)

        # 返回最后一次结果，带上正确的错误信息
        result.success = False
        result.error_message = last_error
        return result

    def close(self):
        """关闭线程池"""
        self._executor.shutdown(wait=False)


class ContinuousCapture:
    """
    连续捕获器

    后台持续采集，保证获取最新数据
    """

    def __init__(self, camera: Any, controller: Any, forward_kinematics: Any = None):
        self.camera = camera
        self.controller = controller
        self.forward_kinematics = forward_kinematics

        self._running = False
        self._thread = None

        self._latest_image = None
        self._latest_joints = None
        self._image_time = 0.0
        self._joints_time = 0.0
        self._lock = threading.Lock()

    def _capture_loop(self):
        """后台采集循环"""
        while self._running:
            try:
                # 读取图像
                image = self.camera.read()
                image_time = time.perf_counter()

                # 读取关节状态
                joints = self.controller.get_joint_states()
                joints_time = time.perf_counter()

                # 更新缓存
                with self._lock:
                    self._latest_image = image
                    self._latest_joints = joints
                    self._image_time = image_time
                    self._joints_time = joints_time

            except Exception as e:
                print(f"采集错误: {e}")

            time.sleep(0.01)  # 10Hz

    def start(self):
        """启动后台采集"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("✓ 后台采集已启动")

    def stop(self):
        """停止后台采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("✓ 后台采集已停止")

    def get_latest(self) -> CaptureResult:
        """获取最新数据"""
        result = CaptureResult()

        with self._lock:
            if self._latest_image is None or self._latest_joints is None:
                result.error_message = "无可用数据"
                return result

            result.image = self._latest_image.copy() if self._latest_image is not None else None
            result.joints = self._latest_joints.copy() if self._latest_joints is not None else None
            result.sync_delay_ms = abs(self._image_time - self._joints_time) * 1000

        # 计算法兰位姿
        if self.forward_kinematics and result.joints is not None:
            result.flange_position, result.flange_rotation = \
                self._compute_flange_pose(result.joints)

        result.success = True
        return result

    def _compute_flange_pose(self, joints):
        """计算法兰位姿"""
        try:
            pose = self.forward_kinematics.compute(joints)
            return pose.get_position(), pose.quaternion
        except:
            return None, None


# 便捷函数
def sync_capture(camera, controller, forward_kinematics=None) -> CaptureResult:
    """
    便捷同步捕获函数

    Args:
        camera: 相机对象
        controller: 控制器对象
        forward_kinematics: 正运动学对象（可选）

    Returns:
        CaptureResult: 捕获结果
    """
    sync = SynchronizedCapture(camera, controller, forward_kinematics)
    return sync.capture()


if __name__ == "__main__":
    # 测试代码
    print("同步捕获模块测试")

    class MockCamera:
        def read(self):
            time.sleep(0.01)  # 模拟读取延迟
            return np.zeros((480, 640, 3), dtype=np.uint8)

    class MockController:
        def get_joint_states(self):
            time.sleep(0.005)  # 模拟读取延迟
            return np.zeros(14)

    camera = MockCamera()
    controller = MockController()

    sync = SynchronizedCapture(camera, controller)

    for i in range(5):
        result = sync.capture()
        print(f"捕获 {i+1}: 成功={result.success}, 延迟={result.sync_delay_ms:.2f}ms")

    sync.close()
    print("测试完成")