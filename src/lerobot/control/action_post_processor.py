"""
Action Post-Processor for Smooth and Precise Robot Control

This module provides post-processing for robot actions to reduce jitter,
improve position accuracy, and ensure safe motion.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PostProcessorConfig:
    """Configuration for action post-processing."""

    # Velocity limits (rad/s)
    max_velocity: float = 3.0
    max_velocity_per_joint: dict[str, float] = None

    # Acceleration limits (rad/s²)
    max_acceleration: float = 15.0
    max_acceleration_per_joint: dict[str, float] = None

    # Low-pass filter parameters
    enable_low_pass_filter: bool = True
    filter_alpha: float = 0.7  # 0-1, higher = less smoothing

    # Jerk limiting (rate of change of acceleration)
    enable_jerk_limiting: bool = True
    max_jerk: float = 50.0  # rad/s³

    # End-effector precision compensation
    enable_end_effector_compensation: bool = True
    compensation_factor: float = 0.1

    # Safety limits
    max_position_delta: float = 0.5  # Maximum position change per step

    # Motion-state adaptive filter (自适应滤波 v2)
    # 改进：渐变Alpha + 状态滞回 + 目标稳定性判定
    enable_motion_adaptive_filter: bool = True
    filter_alpha_uniform: float = 0.7       # 匀速状态: 正常平滑
    filter_alpha_accelerating: float = 0.75  # 变速状态: 略微减弱（更保守）
    filter_alpha_near_target: float = 0.85   # 到位状态: 中等减弱（不是0.95）

    # 目标稳定性判定（用于正确识别"到位"）
    target_stability_threshold: float = 0.01   # rad, 目标变化小于此值视为稳定
    target_stable_count_threshold: int = 10    # 连续N次稳定才判定到位

    # Alpha渐变（避免突变）
    alpha_smooth_rate: float = 0.02  # 每周期alpha最多变化2%

    # 状态滞回（防止震荡）
    near_target_enter_threshold: float = 0.02   # 进入到位状态的阈值
    near_target_exit_threshold: float = 0.05    # 退出到位状态的阈值

    def __post_init__(self):
        if self.max_velocity_per_joint is None:
            self.max_velocity_per_joint = {}
        if self.max_acceleration_per_joint is None:
            self.max_acceleration_per_joint = {}


class ActionPostProcessor:
    """Post-processor for robot actions to ensure smooth and precise motion.

    Features:
    - Velocity limiting
    - Acceleration limiting
    - Jerk limiting (rate of change of acceleration)
    - Motion-adaptive low-pass filtering (NEW):
        - Uniform motion: normal smoothing (alpha=0.7)
        - Accelerating motion: reduced smoothing (alpha=0.85)
        - Near target: minimal smoothing (alpha=0.95) for high precision
    - End-effector precision compensation

    Usage:
        processor = ActionPostProcessor(config)
        processed_action = processor.process_action(raw_action, observation)
    """

    def __init__(self, config: PostProcessorConfig, joint_names: list[str]):
        """Initialize the post-processor.

        Args:
            config: Configuration for post-processing.
            joint_names: List of joint names in order.
        """
        self.config = config
        self.joint_names = joint_names
        self.num_joints = len(joint_names)

        # Identify gripper joints (usually named with "_joint_7" suffix)
        self._gripper_joints = [name for name in joint_names if name.endswith("_joint_7")]

        # Set higher velocity limits for gripper joints (no velocity limiting)
        gripper_velocity_limits = {}
        for name in self._gripper_joints:
            gripper_velocity_limits[name] = float('inf')  # No velocity limit

        if gripper_velocity_limits:
            self.config.max_velocity_per_joint.update(gripper_velocity_limits)

        # State tracking
        self._previous_action: dict[str, float] | None = None
        self._previous_velocity: dict[str, float] = {}
        self._previous_acceleration: dict[str, float] = {}
        self._filtered_actions: dict[str, float] = {}

        # Motion state tracking for adaptive filter v2 (改进版)
        self._current_motion_state: str = "uniform"  # uniform/accelerating/near_target
        self._current_alpha: float = 0.7  # 当前实际使用的alpha（渐变）

        # 目标稳定性追踪（用于正确判定到位）
        self._last_raw_targets: dict[str, float] = {}  # 上一步的原始目标
        self._target_stable_count: int = 0  # 目标连续稳定计数

        # Timing
        self._last_timestamp: float | None = None
        self._dt_estimate: float = 0.02  # Initial estimate (50Hz)

        # Statistics
        self._total_processed = 0
        self._velocity_limited_count = 0
        self._acceleration_limited_count = 0
        self._jerk_limited_count = 0
        self._adaptive_filter_stats: dict[str, int] = {"uniform": 0, "accelerating": 0, "near_target": 0}

        logger.info(f"ActionPostProcessor initialized for {self.num_joints} joints")
        if self._gripper_joints:
            logger.info(f"Gripper joints (no filtering/velocity limit): {self._gripper_joints}")

    def process_action(
        self,
        raw_action: dict[str, float],
        observation: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Process a raw action through all post-processing steps.

        Args:
            raw_action: Raw action dict with joint positions.
            observation: Current observation (optional, for end-effector compensation).

        Returns:
            Processed action dict.
        """
        # Extract positions from action
        action_positions = {}
        for joint_name in self.joint_names:
            key = f"{joint_name}.pos"
            if key in raw_action:
                action_positions[joint_name] = float(raw_action[key])
            elif joint_name in raw_action:
                action_positions[joint_name] = float(raw_action[joint_name])
            else:
                # Use previous value if not present
                action_positions[joint_name] = self._previous_action.get(joint_name, 0.0)

        # Update timing
        current_time = time.time()
        if self._last_timestamp is not None:
            self._dt_estimate = current_time - self._last_timestamp
        self._last_timestamp = current_time

        # Step 1: Velocity limiting
        if self._previous_action is not None:
            action_positions = self._limit_velocity(action_positions)

        # Step 2: Acceleration limiting
        if self._previous_action is not None:
            action_positions = self._limit_acceleration(action_positions)

        # Step 3: Low-pass filtering
        if self.config.enable_low_pass_filter and self._previous_action is not None:
            action_positions = self._apply_low_pass_filter(action_positions)

        # Step 4: Jerk limiting
        if self.config.enable_jerk_limiting and self._previous_action is not None:
            action_positions = self._limit_jerk(action_positions)

        # Step 5: End-effector precision compensation
        if self.config.enable_end_effector_compensation and observation is not None:
            action_positions = self._apply_end_effector_compensation(
                action_positions, observation
            )

        # Step 6: Safety limit check
        action_positions = self._apply_safety_limits(action_positions)

        # Update state
        self._previous_action = action_positions.copy()
        self._total_processed += 1

        # Convert back to action dict format
        result = {}
        for joint_name, position in action_positions.items():
            result[f"{joint_name}.pos"] = position

        return result

    def _limit_velocity(self, action_positions: dict[str, float]) -> dict[str, float]:
        """Limit velocity to maximum allowed values.

        Gripper joints are exempt from velocity limiting for faster response.

        Args:
            action_positions: Target joint positions.

        Returns:
            Velocity-limited positions.
        """
        limited_positions = {}

        for joint_name, target_pos in action_positions.items():
            # Skip velocity limiting for gripper joints
            if joint_name in self._gripper_joints:
                limited_positions[joint_name] = target_pos
                continue

            if joint_name not in self._previous_action:
                limited_positions[joint_name] = target_pos
                continue

            current_pos = self._previous_action[joint_name]
            delta = target_pos - current_pos
            velocity = delta / self._dt_estimate

            # Get joint-specific or global limit
            max_vel = self.config.max_velocity_per_joint.get(
                joint_name, self.config.max_velocity
            )

            # Limit velocity
            if abs(velocity) > max_vel:
                max_delta = max_vel * self._dt_estimate
                limited_delta = max_delta * np.sign(delta)
                limited_positions[joint_name] = current_pos + limited_delta
                self._velocity_limited_count += 1
            else:
                limited_positions[joint_name] = target_pos

            # Store velocity for acceleration limiting
            self._previous_velocity[joint_name] = (
                limited_positions[joint_name] - current_pos
            ) / self._dt_estimate

        return limited_positions

    def _limit_acceleration(self, action_positions: dict[str, float]) -> dict[str, float]:
        """Limit acceleration to maximum allowed values.

        Args:
            action_positions: Target joint positions.

        Returns:
            Acceleration-limited positions.
        """
        limited_positions = {}

        for joint_name, target_pos in action_positions.items():
            if joint_name not in self._previous_velocity:
                limited_positions[joint_name] = target_pos
                continue

            # Calculate target velocity
            current_pos = self._previous_action.get(joint_name, target_pos)
            target_velocity = (target_pos - current_pos) / self._dt_estimate

            # Get previous velocity
            prev_velocity = self._previous_velocity.get(joint_name, 0.0)

            # Calculate acceleration
            acceleration = (target_velocity - prev_velocity) / self._dt_estimate

            # Get joint-specific or global limit
            max_acc = self.config.max_acceleration_per_joint.get(
                joint_name, self.config.max_acceleration
            )

            # Limit acceleration
            if abs(acceleration) > max_acc:
                # Calculate maximum allowed velocity change
                max_velocity_change = max_acc * self._dt_estimate
                limited_velocity = prev_velocity + max_velocity_change * np.sign(acceleration)

                # Calculate new position
                limited_positions[joint_name] = current_pos + limited_velocity * self._dt_estimate
                self._acceleration_limited_count += 1
            else:
                limited_positions[joint_name] = target_pos

            # Store acceleration for jerk limiting
            new_velocity = (limited_positions[joint_name] - current_pos) / self._dt_estimate
            self._previous_acceleration[joint_name] = (
                new_velocity - prev_velocity
            ) / self._dt_estimate

        return limited_positions

    def _detect_motion_state(self, action_positions: dict[str, float]) -> str:
        """Detect current motion state for adaptive filter (v2: 改进版).

        改进点：
        1. 基于目标稳定性判定到位（而非相对距离）
        2. 状态滞回（防止震荡）
        3. 更保守的判定逻辑

        Returns:
            Motion state: "uniform" (匀速), "accelerating" (变速), "near_target" (到位)
        """
        # 初次执行
        if self._previous_action is None:
            self._last_raw_targets = action_positions.copy()
            return "uniform"

        # === 1. 检查目标稳定性（核心改进）===
        # 判断策略输出的目标是否稳定（连续输出相近值）
        target_is_stable = True
        max_target_change = 0

        for joint_name, target_pos in action_positions.items():
            if joint_name in self._gripper_joints:
                continue

            last_target = self._last_raw_targets.get(joint_name, target_pos)
            target_change = abs(target_pos - last_target)
            max_target_change = max(max_target_change, target_change)

            if target_change > self.config.target_stability_threshold:
                target_is_stable = False

        # 更新上一步目标
        self._last_raw_targets = action_positions.copy()

        # 更新稳定计数
        if target_is_stable:
            self._target_stable_count += 1
        else:
            self._target_stable_count = 0

        # === 2. 状态滞回判定 ===
        # 计算"原始目标与当前实际位置的差距"用于判定
        max_distance_to_target = 0
        for joint_name, target_pos in action_positions.items():
            if joint_name in self._gripper_joints:
                continue
            # 使用滤波后的位置作为"当前位置"的近似
            current_pos = self._filtered_actions.get(joint_name, target_pos)
            distance = abs(target_pos - current_pos)
            max_distance_to_target = max(max_distance_to_target, distance)

        # 滞回逻辑
        current_state = self._current_motion_state

        # 到位判定：目标稳定 + 距离小
        if (max_distance_to_target < self.config.near_target_enter_threshold
            and self._target_stable_count >= self.config.target_stable_count_threshold):
            # 进入到位状态：需要目标连续稳定N次 + 距离小于进入阈值
            return "near_target"

        # 退出到位判定：距离变大
        if current_state == "near_target":
            if max_distance_to_target > self.config.near_target_exit_threshold:
                # 距离超过退出阈值，离开到位状态
                pass  # 继续判断是uniform还是accelerating
            else:
                # 在滞回区内，保持到位状态
                return "near_target"

        # === 3. 匀速 vs 变速判定 ===
        if max_target_change > 0.02:  # 目标变化大
            return "accelerating"
        else:
            return "uniform"

    def _get_smooth_alpha(self, target_alpha: float) -> float:
        """平滑过渡alpha，避免突变.

        Args:
            target_alpha: 目标alpha值

        Returns:
            平滑后的alpha值
        """
        delta = target_alpha - self._current_alpha
        if abs(delta) > self.config.alpha_smooth_rate:
            delta = self.config.alpha_smooth_rate * np.sign(delta)
        self._current_alpha += delta
        return self._current_alpha

    def _apply_low_pass_filter(self, action_positions: dict[str, float]) -> dict[str, float]:
        """Apply motion-adaptive low-pass filter (v2: 改进版).

        改进点：
        1. 基于目标稳定性的状态判定
        2. Alpha渐变（避免跳变）
        3. 更保守的参数

        状态对应滤波强度:
        - 匀速移动 (uniform): alpha=0.7, 平滑优先
        - 变速移动 (accelerating): alpha=0.75, 略微减弱
        - 抓取/放置 (near_target): alpha=0.85, 中等减弱

        Formula: filtered = alpha * raw + (1 - alpha) * previous_filtered
        Gripper joints use alpha=1.0 to disable filtering for faster response.

        Args:
            action_positions: Target joint positions.

        Returns:
            Filtered positions.
        """
        # 检测运动状态
        motion_state = self._detect_motion_state(action_positions)
        self._current_motion_state = motion_state
        self._adaptive_filter_stats[motion_state] += 1

        # 根据运动状态选择目标alpha
        if self.config.enable_motion_adaptive_filter:
            if motion_state == "near_target":
                target_alpha = self.config.filter_alpha_near_target     # 0.85
            elif motion_state == "accelerating":
                target_alpha = self.config.filter_alpha_accelerating    # 0.75
            else:  # uniform
                target_alpha = self.config.filter_alpha_uniform         # 0.7
        else:
            # 禁用自适应时，使用固定alpha
            target_alpha = self.config.filter_alpha

        # Alpha渐变（关键改进：避免突变）
        alpha = self._get_smooth_alpha(target_alpha)

        # 周期性日志（每100次）
        if self._total_processed % 100 == 0:
            logger.debug(
                f"Adaptive filter v2: state={motion_state}, alpha={alpha:.3f}, "
                f"target_alpha={target_alpha:.2f}, stable_count={self._target_stable_count}, "
                f"stats={self._adaptive_filter_stats}"
            )

        filtered_positions = {}

        for joint_name, target_pos in action_positions.items():
            # Check if this is a gripper joint (use alpha=1.0 for no filtering)
            is_gripper = joint_name in self._gripper_joints

            if joint_name in self._filtered_actions:
                if is_gripper:
                    # Gripper joints: no filtering (alpha=1.0)
                    filtered_positions[joint_name] = target_pos
                    self._filtered_actions[joint_name] = target_pos
                else:
                    # Other joints: apply motion-adaptive exponential moving average
                    filtered_pos = (
                        alpha * target_pos
                        + (1 - alpha) * self._filtered_actions[joint_name]
                    )
                    filtered_positions[joint_name] = filtered_pos
                    self._filtered_actions[joint_name] = filtered_pos
            else:
                # First time, no filtering
                filtered_positions[joint_name] = target_pos
                self._filtered_actions[joint_name] = target_pos

        return filtered_positions

    def _limit_jerk(self, action_positions: dict[str, float]) -> dict[str, float]:
        """Limit jerk (rate of change of acceleration).

        Args:
            action_positions: Target joint positions.

        Returns:
            Jerk-limited positions.
        """
        limited_positions = {}

        for joint_name, target_pos in action_positions.items():
            if joint_name not in self._previous_acceleration:
                limited_positions[joint_name] = target_pos
                continue

            # Recalculate acceleration with current target
            current_pos = self._previous_action.get(joint_name, target_pos)
            target_velocity = (target_pos - current_pos) / self._dt_estimate
            prev_velocity = self._previous_velocity.get(joint_name, 0.0)
            target_acceleration = (target_velocity - prev_velocity) / self._dt_estimate

            # Calculate jerk
            prev_acceleration = self._previous_acceleration[joint_name]
            jerk = (target_acceleration - prev_acceleration) / self._dt_estimate

            # Limit jerk
            if abs(jerk) > self.config.max_jerk:
                # Calculate maximum allowed acceleration change
                max_acceleration_change = self.config.max_jerk * self._dt_estimate
                limited_acceleration = prev_acceleration + max_acceleration_change * np.sign(jerk)

                # Back-calculate velocity and position
                limited_velocity = prev_velocity + limited_acceleration * self._dt_estimate
                limited_positions[joint_name] = current_pos + limited_velocity * self._dt_estimate
                self._jerk_limited_count += 1
            else:
                limited_positions[joint_name] = target_pos

        return limited_positions

    def _apply_end_effector_compensation(
        self,
        action_positions: dict[str, float],
        observation: dict[str, Any],
    ) -> dict[str, float]:
        """Apply end-effector precision compensation.

        This adjusts the final joint positions to compensate for
        systematic positioning errors.

        Args:
            action_positions: Target joint positions.
            observation: Current observation for context.

        Returns:
            Compensated positions.
        """
        compensated_positions = action_positions.copy()

        # Get current forces (higher force might indicate contact/obstruction)
        forces = {}
        for key, value in observation.items():
            if ".force" in key:
                joint_name = key.replace(".force", "")
                forces[joint_name] = float(value)

        # Apply small compensation for joints with high force
        # (might be fighting gravity or external forces)
        for joint_name in self.joint_names:
            if joint_name in forces:
                force = forces[joint_name]
                # Compensate in direction of force
                compensation = self.config.compensation_factor * np.sign(force) * min(abs(force), 0.5)
                if joint_name in compensated_positions:
                    compensated_positions[joint_name] += compensation

        return compensated_positions

    def _apply_safety_limits(self, action_positions: dict[str, float]) -> dict[str, float]:
        """Apply final safety limits.

        Args:
            action_positions: Target joint positions.

        Returns:
            Safety-checked positions.
        """
        safe_positions = {}

        for joint_name, target_pos in action_positions.items():
            if self._previous_action is not None:
                current_pos = self._previous_action.get(joint_name, target_pos)
                delta = abs(target_pos - current_pos)

                # Hard limit on position change
                if delta > self.config.max_position_delta:
                    max_change = self.config.max_position_delta
                    safe_positions[joint_name] = current_pos + max_change * np.sign(target_pos - current_pos)
                    logger.warning(
                        f"Safety limit applied to {joint_name}: "
                        f"delta {delta:.3f} > max {self.config.max_position_delta}"
                    )
                else:
                    safe_positions[joint_name] = target_pos
            else:
                safe_positions[joint_name] = target_pos

        return safe_positions

    def reset(self):
        """Reset the post-processor state."""
        self._previous_action = None
        self._previous_velocity.clear()
        self._previous_acceleration.clear()
        self._filtered_actions.clear()
        self._last_timestamp = None
        # Reset motion state tracking (v2)
        self._current_motion_state = "uniform"
        self._current_alpha = 0.7
        self._last_raw_targets.clear()
        self._target_stable_count = 0
        self._adaptive_filter_stats = {"uniform": 0, "accelerating": 0, "near_target": 0}
        logger.info("ActionPostProcessor reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get post-processor statistics."""
        return {
            "total_processed": self._total_processed,
            "velocity_limited_count": self._velocity_limited_count,
            "acceleration_limited_count": self._acceleration_limited_count,
            "jerk_limited_count": self._jerk_limited_count,
            "limit_rate": (
                self._velocity_limited_count / self._total_processed
                if self._total_processed > 0
                else 0
            ),
            # Motion-adaptive filter v2 statistics
            "current_motion_state": self._current_motion_state,
            "current_alpha": round(self._current_alpha, 3),
            "target_stable_count": self._target_stable_count,
            "adaptive_filter_stats": self._adaptive_filter_stats.copy(),
            "enable_motion_adaptive_filter": self.config.enable_motion_adaptive_filter,
        }

    def update_config(self, **kwargs):
        """Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")


def create_post_processor_for_robot(
    robot_config: Any,
    smoothing_level: str = "medium",
) -> ActionPostProcessor:
    """Create an action post-processor with sensible defaults for a robot.

    Args:
        robot_config: Robot configuration object.
        smoothing_level: Level of smoothing ("low", "medium", "high").

    Returns:
        Configured ActionPostProcessor instance.
    """
    # Get joint names from robot config
    if hasattr(robot_config, "joint_order"):
        joint_names = robot_config.joint_order
    elif hasattr(robot_config, "observation_joint_names"):
        joint_names = robot_config.observation_joint_names
    else:
        # Default joint names
        joint_names = [
            "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
            "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
            "left_arm_joint_7",
            "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
            "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
            "right_arm_joint_7",
            "trunk_joint_1", "trunk_joint_2",
        ]

    # Configure based on smoothing level
    if smoothing_level == "low":
        config = PostProcessorConfig(
            max_velocity=5.0,
            max_acceleration=25.0,
            filter_alpha=0.9,
            enable_jerk_limiting=False,
        )
    elif smoothing_level == "high":
        config = PostProcessorConfig(
            max_velocity=2.0,
            max_acceleration=10.0,
            filter_alpha=0.5,
            enable_jerk_limiting=True,
            max_jerk=30.0,
        )
    else:  # medium
        config = PostProcessorConfig(
            max_velocity=3.0,
            max_acceleration=15.0,
            filter_alpha=0.7,
            enable_jerk_limiting=True,
        )

    return ActionPostProcessor(config, joint_names)
