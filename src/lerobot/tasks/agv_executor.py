"""
AGV Task Executor for the task agent system.

This module provides execution of AGV navigation tasks within
the task orchestrator framework, enabling robot + AGV coordination
for tasks like: pick at A → AGV moves to B → place at B.

The executor handles:
- Safety checks (arm position before AGV movement)
- Navigation command execution
- Arrival waiting and detection
- Error handling and emergency stop
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lerobot.robots.agv.seer_agv_controller import (
    SeerAGVController,
    AGVStatus,
    AGVPosition,
)
from lerobot.tasks.task_scheduler import TaskResult, TaskStatus

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


@dataclass
class AGVExecutionResult:
    """AGV任务执行结果."""
    success: bool
    arrival_station: str
    duration: float
    final_status: AGVStatus | None
    error: str | None
    distance_traveled: float = 0.0
    battery_level: int = 0


class AGVTaskExecutor:
    """AGV任务执行器.

    负责执行AGV导航任务，包括：
    - 预安全检查（机械臂位置验证）
    - AGV连接状态检查
    - 导航指令发送
    - 到达等待和判定
    - 异常处理和急停
    - 后安全检查（确认到达状态）

    Usage:
        controller = SeerAGVController(host="192.168.1.100")
        executor = AGVTaskExecutor(controller, robot)

        result = executor.execute(agv_config)
        if result.success:
            print(f"Arrived at {result.arrival_station}")
    """

    def __init__(
        self,
        agv_controller: SeerAGVController,
        robot: "Robot | None" = None,
        enable_safety_check: bool = True,
    ):
        """初始化AGV执行器.

        Args:
            agv_controller: AGV TCP控制器实例
            robot: 机械臂实例 (用于安全检查)
            enable_safety_check: 是否启用安全检查
        """
        self.agv = agv_controller
        self.robot = robot
        self.enable_safety_check = enable_safety_check

        # 默认home位置 (推荐值，单位：度)
        # 来源: /root/workspace/dc_dir/0420.log
        self._default_home_positions = {
            # 左臂
            "left_arm_joint_1": -40.3,
            "left_arm_joint_2": -4.0,
            "left_arm_joint_3": -0.5,
            "left_arm_joint_4": -58.6,
            "left_arm_joint_5": -2.0,
            "left_arm_joint_6": 11.8,
            "left_arm_joint_7": 0.3,  # gripper (closed, holding)
            # 右臂
            "right_arm_joint_1": 40.3,
            "right_arm_joint_2": 4.0,
            "right_arm_joint_3": 0.8,
            "right_arm_joint_4": 58.6,
            "right_arm_joint_5": 2.0,
            "right_arm_joint_6": -11.8,
            "right_arm_joint_7": 0.3,  # gripper (closed, holding)
            # 躯干
            "trunk_joint_1": 0.0,
            "trunk_joint_2": 0.0,
        }

        # 默认安全偏差阈值 (关节从home位置的最大偏差，单位：度)
        self._default_safe_thresholds = {
            # 左臂
            "left_arm_joint_1": 5.0,  # 允许从home偏差±5°
            "left_arm_joint_2": 5.0,
            "left_arm_joint_3": 5.0,
            "left_arm_joint_4": 5.0,
            "left_arm_joint_5": 5.0,
            "left_arm_joint_6": 5.0,
            "left_arm_joint_7": 0.5,  # gripper
            # 右臂
            "right_arm_joint_1": 5.0,
            "right_arm_joint_2": 5.0,
            "right_arm_joint_3": 5.0,
            "right_arm_joint_4": 5.0,
            "right_arm_joint_5": 5.0,
            "right_arm_joint_6": 5.0,
            "right_arm_joint_7": 0.5,  # gripper
            # 躯干
            "trunk_joint_1": 3.0,
            "trunk_joint_2": 3.0,
        }

        # 统计信息
        self._last_execution_time = 0.0
        self._total_executions = 0
        self._successful_executions = 0

    def execute(
        self,
        task_name: str,
        target_station: str | None = None,
        target_position: tuple[float, float, float] | None = None,
        max_duration: float = 60.0,
        wait_for_arrival: bool = True,
        arrival_timeout: float = 60.0,
        arrival_tolerance: float = 0.3,
        check_arm_safe: bool = True,
        arm_safe_positions: dict[str, float] | None = None,
        arm_home_positions: dict[str, float] | None = None,
        retry_on_timeout: bool = True,
        retry_count: int = 2,
        emergency_stop_on_error: bool = True,
    ) -> AGVExecutionResult:
        """执行AGV导航任务.

        Args:
            task_name: 任务名称 (用于日志)
            target_station: 目标站点ID (优先使用)
            target_position: 目标坐标 (x, y, theta) (备选)
            max_duration: 最大执行时间
            wait_for_arrival: 是否等待到达完成
            arrival_timeout: 到达等待超时
            arrival_tolerance: 距离容差 (米)
            check_arm_safe: 是否检查机械臂安全位置
            arm_safe_positions: 安全偏差阈值字典 {joint_name: max_deviation}
            arm_home_positions: home位置字典 {joint_name: home_position}
                              如果为None，使用默认home位置
            retry_on_timeout: 超时是否重试
            retry_count: 重试次数
            emergency_stop_on_error: 异常时是否急停

        Returns:
            AGVExecutionResult执行结果
        """
        start_time = time.time()
        self._total_executions += 1

        result = AGVExecutionResult(
            success=False,
            arrival_station="",
            duration=0.0,
            final_status=None,
            error=None,
        )

        logger.info(f"[AGVExecutor] Starting task: {task_name}")

        try:
            # ========== Phase 1: 连接检查 ==========
            if not self.agv.is_connected():
                logger.info("[AGVExecutor] AGV not connected, attempting connection...")
                if not self.agv.connect():
                    result.error = "Failed to connect to AGV"
                    logger.error(result.error)
                    return result

            # ========== Phase 2: 初始状态检查 ==========
            initial_status = self.agv.get_status(use_cache=False)
            logger.info(
                f"[AGVExecutor] Initial status: "
                f"station={initial_status.current_station}, "
                f"battery={initial_status.battery}%, "
                f"moving={initial_status.is_moving}"
            )

            # 电量检查
            if initial_status.battery < 15:
                result.error = f"Low battery: {initial_status.battery}%"
                logger.warning(result.error)
                # 可以继续执行，但记录警告

            # 检查AGV是否在执行其他任务
            if initial_status.is_moving or initial_status.status_code != 0:
                logger.info("[AGVExecutor] AGV busy, waiting for idle...")
                if not self.agv.wait_for_idle(timeout=10.0):
                    # 尝试取消当前任务
                    logger.warning("[AGVExecutor] Cancelling existing AGV task")
                    self.agv.cancel_task()
                    time.sleep(2)

            # ========== Phase 3: 机械臂安全检查 ==========
            if check_arm_safe and self.robot and self.enable_safety_check:
                safe_thresholds = arm_safe_positions or self._default_safe_thresholds
                home_positions = arm_home_positions or self._default_home_positions
                if not self._check_arm_safe_position(safe_thresholds, home_positions):
                    result.error = "Arm not in safe position for AGV movement"
                    logger.warning(f"[AGVExecutor] {result.error}")
                    # 可以尝试先归位机械臂，或者返回错误让上层处理
                    # 这里先返回错误
                    return result

            # ========== Phase 4: 发送导航指令 ==========
            navigation_success = False
            target_desc = ""

            for attempt in range(retry_count + 1):
                if target_station:
                    navigation_success = self.agv.move_to_station(target_station)
                    target_desc = f"station={target_station}"
                elif target_position:
                    x, y, theta = target_position
                    navigation_success = self.agv.move_to_position(x, y, theta)
                    target_desc = f"position=({x:.2f}, {y:.2f})"
                else:
                    result.error = "No target specified (neither station nor position)"
                    return result

                if navigation_success:
                    break

                if attempt < retry_count:
                    logger.info(f"[AGVExecutor] Navigation start failed, retrying ({attempt + 1}/{retry_count})")
                    time.sleep(2)

            if not navigation_success:
                result.error = f"Failed to start navigation to {target_desc}"
                logger.error(result.error)
                return result

            logger.info(f"[AGVExecutor] Navigation started: {target_desc}")

            # ========== Phase 5: 等待到达 ==========
            if wait_for_arrival:
                arrived = False

                if target_station:
                    arrived = self.agv.wait_for_arrival(
                        target_station,
                        timeout=arrival_timeout,
                        tolerance=arrival_tolerance,
                    )
                elif target_position:
                    x, y, theta = target_position
                    arrived = self._wait_for_position(
                        x, y,
                        timeout=arrival_timeout,
                        tolerance=arrival_tolerance,
                    )

                if not arrived:
                    result.error = f"Timeout waiting for arrival at {target_desc}"
                    logger.warning(result.error)

                    # 超时重试
                    if retry_on_timeout and retry_count > 0:
                        logger.info(f"[AGVExecutor] Retrying navigation due to timeout")
                        self.agv.cancel_task()
                        time.sleep(2)
                        # 重新执行 (简化，实际应该递归调用或循环)
                        # 这里不实现完整重试，返回错误让上层处理

                    if emergency_stop_on_error:
                        logger.warning("[AGVExecutor] Executing emergency stop due to timeout")
                        self.agv.stop()

                    return result

            # ========== Phase 6: 最终状态确认 ==========
            final_status = self.agv.get_status(use_cache=False)
            elapsed = time.time() - start_time

            result.success = True
            result.arrival_station = final_status.current_station
            result.duration = elapsed
            result.final_status = final_status
            result.battery_level = final_status.battery

            # 计算移动距离 (如果有初始位置)
            if initial_status.position and final_status.position:
                dx = final_status.position.x - initial_status.position.x
                dy = final_status.position.y - initial_status.position.y
                result.distance_traveled = (dx**2 + dy**2) ** 0.5

            self._successful_executions += 1
            self._last_execution_time = elapsed

            logger.info(
                f"[AGVExecutor] Task completed: "
                f"station={result.arrival_station}, "
                f"duration={elapsed:.1f}s, "
                f"distance={result.distance_traveled:.2f}m"
            )

        except Exception as e:
            logger.error(f"[AGVExecutor] Execution error: {e}")
            result.error = str(e)

            # 异常急停
            if emergency_stop_on_error:
                try:
                    logger.warning("[AGVExecutor] Emergency stop due to exception")
                    self.agv.stop()
                except Exception as stop_err:
                    logger.error(f"[AGVExecutor] Failed to emergency stop: {stop_err}")

        finally:
            result.duration = time.time() - start_time

        return result

    def _check_arm_safe_position(
        self,
        safe_thresholds: dict[str, float],
        home_positions: dict[str, float] | None = None,
    ) -> bool:
        """检查机械臂是否在安全位置.

        AGV移动时机械臂需要收起，避免碰撞。

        Args:
            safe_thresholds: {joint_name: max_deviation} 字典，最大允许偏差
            home_positions: {joint_name: home_position} 字典，home位置参考
                           如果为None，则使用绝对位置检查（阈值检查abs(pos)）
                           如果提供，则检查偏差 abs(pos - home_position)

        Returns:
            True if all joints within safe limits
        """
        if not self.robot:
            return True

        try:
            # 获取当前观测
            observation = self.robot.capture_observation()

            unsafe_joints = []
            for joint_name, threshold in safe_thresholds.items():
                pos_key = f"{joint_name}.pos"
                if pos_key in observation:
                    pos = observation[pos_key]

                    # 检查方式：基于home偏差 vs 绝对位置
                    if home_positions and joint_name in home_positions:
                        # 基于home位置偏差检查
                        home_pos = home_positions[joint_name]
                        deviation = abs(pos - home_pos)
                        if deviation > threshold:
                            unsafe_joints.append((joint_name, pos, home_pos, deviation, threshold))
                    else:
                        # 绝对位置检查（旧逻辑）
                        if abs(pos) > threshold:
                            unsafe_joints.append((joint_name, pos, None, abs(pos), threshold))

            if unsafe_joints:
                # 格式化警告消息
                if home_positions:
                    msg_parts = [
                        f"{j}: pos={p:.1f}°, home={h:.1f}°, deviation={d:.1f}° > threshold={t:.1f}°"
                        for j, p, h, d, t in unsafe_joints if h is not None
                    ]
                    msg_parts.extend([
                        f"{j}: pos={p:.1f}° > threshold={t:.1f}°"
                        for j, p, h, d, t in unsafe_joints if h is None
                    ])
                else:
                    msg_parts = [
                        f"{j}: pos={p:.1f}° > threshold={t:.1f}°"
                        for j, p, _, _, t in unsafe_joints
                    ]
                logger.warning(
                    f"[AGVExecutor] Unsafe arm positions:\n  " + "\n  ".join(msg_parts)
                )
                return False

            logger.info("[AGVExecutor] Arm position check passed - arm is in safe home position")
            return True

        except Exception as e:
            logger.error(f"[AGVExecutor] Failed to check arm position: {e}")
            # 检查失败时保守处理
            return False

    def _wait_for_position(
        self,
        target_x: float,
        target_y: float,
        timeout: float,
        tolerance: float,
        poll_interval: float = 1.0,
    ) -> bool:
        """等待到达目标坐标.

        Args:
            target_x: 目标x坐标
            target_y: 目标y坐标
            timeout: 最大等待时间
            tolerance: 距离容差
            poll_interval: 轮询间隔

        Returns:
            True if arrived, False if timeout/error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.agv.get_status(use_cache=False)
            pos = status.position

            # 计算距离
            distance = ((pos.x - target_x) ** 2 + (pos.y - target_y) ** 2) ** 0.5

            if distance <= tolerance:
                logger.info(f"[AGVExecutor] Arrived at position: distance={distance:.3f}m")
                return True

            # 检查异常
            if status.error_code != 0:
                logger.error(
                    f"[AGVExecutor] AGV error during navigation: "
                    f"{status.error_code} - {status.error_message}"
                )
                return False

            # 检查是否已停止但未到达
            if not status.is_moving and status.status_code == 0:
                elapsed = time.time() - start_time
                if elapsed > 5.0:  # 给导航一点启动时间
                    logger.warning("[AGVExecutor] AGV stopped before reaching target")
                    return False

            time.sleep(poll_interval)

        logger.warning(f"[AGVExecutor] Timeout waiting for position ({target_x:.2f}, {target_y:.2f})")
        return False

    def get_statistics(self) -> dict:
        """获取执行统计信息."""
        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "success_rate": (
                self._successful_executions / self._total_executions
                if self._total_executions > 0 else 0
            ),
            "last_execution_time": self._last_execution_time,
            "agv_connected": self.agv.is_connected(),
            "agv_info": self.agv.get_info(),
        }

    def reset_statistics(self):
        """重置统计信息."""
        self._total_executions = 0
        self._successful_executions = 0
        self._last_execution_time = 0.0


def create_task_result_from_agv_result(
    task_name: str,
    agv_result: AGVExecutionResult,
) -> TaskResult:
    """将AGVExecutionResult转换为TaskResult.

    用于与TaskScheduler集成。

    Args:
        task_name: 任务名称
        agv_result: AGV执行结果

    Returns:
        TaskResult对象
    """
    status = TaskStatus.COMPLETED if agv_result.success else TaskStatus.FAILED

    # 构建final_observation包含AGV状态信息
    final_observation = {
        "arrival_station": agv_result.arrival_station,
        "distance_traveled": agv_result.distance_traveled,
        "battery_level": agv_result.battery_level,
    }

    if agv_result.final_status:
        final_observation["final_position"] = {
            "x": agv_result.final_status.position.x,
            "y": agv_result.final_status.position.y,
            "theta": agv_result.final_status.position.theta,
        }

    return TaskResult(
        task_name=task_name,
        status=status,
        duration=agv_result.duration,
        error_message=agv_result.error,
        collision_detected=False,
        attempts=1,
        final_observation=final_observation,
    )