#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
动作修正模块 - 独立模块，可在 replay/eval/推理中使用

提供两种修正源：
1. LeaderCorrector - Leader 设备辅助修正（状态机设计）
2. KeyboardCorrector - 键盘辅助修正（平滑精细控制）

使用示例：
```python
from lerobot.utils.action_corrector import ActionCorrector, ActionCorrectorConfig

# 初始化修正器
corrector = ActionCorrector(config, teleop=leader_device)

# 在推理循环中
action = policy.select_action(observation)  # ACT 输出原始动作
corrected_action = corrector.correct(action, events=keyboard_events)  # 应用修正
```
"""

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LeaderAdjustConfig:
    """Leader微调配置（状态机设计，滞回机制）"""

    enable: bool = True

    # 触发参数
    trigger_threshold_deg: float = 5.0  # 触发阈值：5度（Leader变化超过此值进入微调状态）
    trigger_alpha: float = 0.3          # 触发后修正系数

    # 退出参数（滞回设计，防止频繁进出）
    exit_threshold_deg: float = 1.0     # 退出阈值：1度（Leader变化小于此值开始计数退出）
    exit_frame_count: int = 5           # 连续N帧变化<1度才退出

    # 微调期间参数
    adjust_alpha: float = 0.3           # 微调修正系数（follower = leader_delta * alpha）


@dataclass
class KeyAdjustConfig:
    """键盘微调配置（平滑精细控制）"""

    enable: bool = True

    # 平滑参数（精细控制）
    step_per_frame: float = 0.2      # 每帧调整幅度（度）- 增大以提高joint_3可见度
    max_adjustment: float = 5.0      # 单次按键最大累积调整量（度）

    # 双臂控制模式：left / right / both
    arm_control_mode: str = "both"

    # 关节方向反转（解决硬件电机安装方向不一致问题）
    joint_inverse: str | dict[str, bool] = field(default_factory=dict)

    # 按键映射
    keys_joint_1_positive: str = "w"
    keys_joint_1_negative: str = "x"
    keys_joint_3_positive: str = "a"
    keys_joint_3_negative: str = "d"
    keys_trunk_positive: str = "q"
    keys_trunk_negative: str = "e"
    key_mode_toggle: str = "k"


@dataclass
class ActionCorrectorConfig:
    """动作修正器总配置"""
    enable: bool = True

    # Leader 修正配置
    leader: LeaderAdjustConfig = field(default_factory=LeaderAdjustConfig)

    # 键盘修正配置
    keyboard: KeyAdjustConfig = field(default_factory=KeyAdjustConfig)


class LeaderCorrector:
    """Leader辅助修正器（状态机设计，累积器模式）。

    状态：
    - idle: 等待触发
    - adjusting: 微调状态（Leader连续影响动作）

    流程：
    1. 等待Leader变化 > trigger_threshold_deg → 进入微调状态
    2. 重置基准位置，从此刻起计算delta
    3. 微调期间：将修正量保存到 accumulator
    4. 连续N帧变化 < exit_threshold_deg → 退出微调状态
       - accumulator 合并到 applied_offset（保持修正）
       - accumulator 清零（下次独立计算）
    5. 总修正量 = applied_offset + accumulator，保证不跳跃

    变量说明：
    - accumulator: 当前微调期间的临时累积（触发时清零）
    - applied_offset: 退出时保存的修正量（持续保持）
    """

    def __init__(self, cfg: LeaderAdjustConfig):
        self.cfg = cfg
        self.state = "idle"
        self.trigger_baseline: dict[str, float] | None = None
        self.prev_leader_pos: dict[str, float] | None = None
        self.exit_frame_count = 0
        self.accumulator: dict[str, float] = {}        # 当前微调临时累积
        self.applied_offset: dict[str, float] = {}     # 退出时保存的修正量

    def deg_to_rad(self, deg: float) -> float:
        return deg * np.pi / 180.0

    def rad_to_deg(self, rad: float) -> float:
        return rad * 180.0 / np.pi

    def reset(self):
        """重置状态机和所有修正量"""
        self.state = "idle"
        self.trigger_baseline = None
        self.prev_leader_pos = None
        self.exit_frame_count = 0
        self.accumulator.clear()
        self.applied_offset.clear()

    def reset_accumulator(self):
        """仅重置累积器（保留 applied_offset）"""
        self.accumulator.clear()

    def get_total_offset(self) -> dict[str, float]:
        """获取总修正量（applied_offset + accumulator）"""
        total = self.applied_offset.copy()
        for joint, val in self.accumulator.items():
            total[joint] = total.get(joint, 0.0) + val
        return total

    def process(self, leader_obs: dict[str, float]) -> dict[str, float]:
        """处理Leader输入，返回累积修正量。

        Args:
            leader_obs: Leader observation (包含 .pos 的关节位置)

        Returns:
            累积修正量字典（用于应用到 action）
        """
        # 提取Leader位置
        leader_pos = {k: v for k, v in leader_obs.items() if k.endswith(".pos")}

        # === 计算变化量 ===
        if self.prev_leader_pos is not None:
            frame_delta = {}
            max_delta_rad = 0.0
            for key, value in leader_pos.items():
                delta_rad = value - self.prev_leader_pos.get(key, value)
                frame_delta[key.replace(".pos", "")] = delta_rad
                max_delta_rad = max(max_delta_rad, abs(delta_rad))
            max_delta_deg = self.rad_to_deg(max_delta_rad)
        else:
            frame_delta = {}
            max_delta_deg = 0.0

        # === 状态机逻辑 ===
        if self.state == "idle":
            if max_delta_deg > self.cfg.trigger_threshold_deg:
                self.state = "adjusting"
                self.trigger_baseline = leader_pos.copy()
                self.accumulator.clear()  # 触发时清零累积器，避免叠加旧修正量导致跳跃
                self.exit_frame_count = 0
                logging.info(f"[LeaderCorrector] 微调触发: 变化量={max_delta_deg:.1f}度")

        elif self.state == "adjusting":
            if self.trigger_baseline is not None:
                for joint_key, leader_value in leader_pos.items():
                    joint_name = joint_key.replace(".pos", "")
                    baseline = self.trigger_baseline.get(joint_key, leader_value)
                    delta_rad = leader_value - baseline
                    self.accumulator[joint_name] = self.cfg.adjust_alpha * delta_rad

            if max_delta_deg < self.cfg.exit_threshold_deg:
                self.exit_frame_count += 1
                if self.exit_frame_count >= self.cfg.exit_frame_count:
                    # 退出时：保存 accumulator 到 applied_offset，然后清零 accumulator
                    for joint, val in self.accumulator.items():
                        if abs(val) > 0.001:  # 只保存有意义的修正
                            self.applied_offset[joint] = self.applied_offset.get(joint, 0.0) + val
                            logging.debug(f"[LeaderCorrector] Saved {joint} offset={val:.3f}, total={self.applied_offset[joint]:.3f}")
                    self.accumulator.clear()  # 清零，下次触发独立计算
                    self.state = "idle"
                    self.trigger_baseline = None
                    logging.info(f"[LeaderCorrector] 微调退出: 连续{self.exit_frame_count}帧变化<{self.cfg.exit_threshold_deg}度, 修正已保存")
                    self.exit_frame_count = 0
            else:
                self.exit_frame_count = 0

        # === 更新上一帧位置 ===
        self.prev_leader_pos = leader_pos.copy()

        # 返回总修正量（applied_offset + accumulator）
        return self.get_total_offset()


class KeyboardCorrector:
    """键盘辅助修正器（平滑精细控制）。

    特点：
    - 每帧渐进调整（而非跳变）
    - 按键释放时：清零累积器，但保持已微调的位置（applied_offset）
    - 每次按键独立计算最多5度，但总修正量持续累积
    - 支持关节方向反转

    变量说明：
    - accumulator: 当前按键会话的临时累积，按键释放时清零
    - applied_offset: 已应用并保持的修正量，按键释放时合并accumulated值
    """

    # 按键到关节的映射（用于释放时处理）
    KEY_TO_JOINTS = {
        "joint_1_positive": ["left_arm_joint_1", "right_arm_joint_1"],
        "joint_1_negative": ["left_arm_joint_1", "right_arm_joint_1"],
        "joint_3_positive": ["left_arm_joint_3", "right_arm_joint_3"],
        "joint_3_negative": ["left_arm_joint_3", "right_arm_joint_3"],
        "trunk_positive": ["trunk_joint_1"],
        "trunk_negative": ["trunk_joint_1"],
    }

    def __init__(self, cfg: KeyAdjustConfig):
        self.cfg = cfg
        self.accumulator: dict[str, float] = {}      # 当前按键会话的临时累积
        self.applied_offset: dict[str, float] = {}   # 已应用并保持的修正量
        self._joint_inverse: dict[str, bool] = {}
        self._prev_key_state: dict[str, bool] = {}   # 存储上一帧按键状态
        self._parse_joint_inverse()

    def _parse_joint_inverse(self):
        """解析 joint_inverse 配置"""
        raw = self.cfg.joint_inverse
        if isinstance(raw, str):
            try:
                self._joint_inverse = ast.literal_eval(raw)
                logging.info(f"[KeyboardCorrector] joint_inverse parsed: {self._joint_inverse}")
            except (ValueError, SyntaxError):
                try:
                    self._joint_inverse = json.loads(raw)
                    logging.info(f"[KeyboardCorrector] joint_inverse parsed from JSON: {self._joint_inverse}")
                except json.JSONDecodeError as e:
                    logging.warning(f"[KeyboardCorrector] Failed to parse joint_inverse '{raw}': {e}")
                    self._joint_inverse = {}
        else:
            self._joint_inverse = raw if raw else {}

    def reset(self):
        """重置所有修正量"""
        self.accumulator.clear()
        self.applied_offset.clear()

    def get_accumulator(self) -> dict[str, float]:
        """获取当前累积修正量（用于调试）"""
        return self.accumulator.copy()

    def get_total_offset(self) -> dict[str, float]:
        """获取总修正量（applied_offset + accumulator）"""
        total = self.applied_offset.copy()
        for joint, val in self.accumulator.items():
            total[joint] = total.get(joint, 0.0) + val
        return total

    def update_accumulator(self, events: dict) -> dict[str, float]:
        """根据键盘事件更新累积器。

        每次按键独立计算：按键释放时清零对应关节的累积值，
        下次按键从0开始重新累加。

        Args:
            events: 键盘事件字典，包含:
                - arm_control_mode: "left"/"right"/"both"
                - joint_1_positive_held: bool (W键)
                - joint_1_negative_held: bool (X键)
                - joint_3_positive_held: bool (A键)
                - joint_3_negative_held: bool (D键)
                - trunk_positive_held: bool (Q键)
                - trunk_negative_held: bool (E键)

        Returns:
            更新后的累积器
        """
        mode = events.get("arm_control_mode", self.cfg.arm_control_mode)
        step = self.cfg.step_per_frame
        max_adj = self.cfg.max_adjustment

        # === 检测按键释放，保存修正并清零累积器 ===
        for key_name in self.KEY_TO_JOINTS.keys():
            event_key = f"{key_name}_held"
            current_state = events.get(event_key, False)
            prev_state = self._prev_key_state.get(event_key, False)

            # 检测从 True -> False（按键释放）
            if prev_state and not current_state:
                joints_to_process = self.KEY_TO_JOINTS[key_name]
                mode_now = events.get("arm_control_mode", mode)
                for joint in joints_to_process:
                    should_process = False
                    if joint.startswith("left_arm") and mode_now in ["both", "left"]:
                        should_process = True
                    elif joint.startswith("right_arm") and mode_now in ["both", "right"]:
                        should_process = True
                    elif joint.startswith("trunk"):
                        should_process = True

                    if should_process and joint in self.accumulator:
                        # 将当前累积值合并到 applied_offset（保持微调位置）
                        acc_val = self.accumulator[joint]
                        if abs(acc_val) > 0.001:  # 只处理有意义的修正
                            self.applied_offset[joint] = self.applied_offset.get(joint, 0.0) + acc_val
                            logging.debug(f"[KeyboardCorrector] Key released: saved {joint} offset={acc_val:.3f}, total={self.applied_offset[joint]:.3f}")
                        # 清零累积器（下次按键从0开始）
                        self.accumulator[joint] = 0.0

            # 更新按键状态记录
            self._prev_key_state[event_key] = current_state

        # === 双臂 joint_1（同向）===
        if events.get("joint_1_positive_held"):  # W键
            if mode == "both":
                acc_left = self.accumulator.get("left_arm_joint_1", 0.0)
                acc_right = self.accumulator.get("right_arm_joint_1", 0.0)
                new_left = acc_left + step
                new_right = acc_right + step
                if abs(new_left) <= max_adj:
                    self.accumulator["left_arm_joint_1"] = new_left
                if abs(new_right) <= max_adj:
                    self.accumulator["right_arm_joint_1"] = new_right
            elif mode == "left":
                acc = self.accumulator.get("left_arm_joint_1", 0.0)
                new_val = acc + step
                if abs(new_val) <= max_adj:
                    self.accumulator["left_arm_joint_1"] = new_val
            elif mode == "right":
                acc = self.accumulator.get("right_arm_joint_1", 0.0)
                new_val = acc + step
                if abs(new_val) <= max_adj:
                    self.accumulator["right_arm_joint_1"] = new_val

        if events.get("joint_1_negative_held"):  # X键
            if mode == "both":
                acc_left = self.accumulator.get("left_arm_joint_1", 0.0)
                acc_right = self.accumulator.get("right_arm_joint_1", 0.0)
                new_left = acc_left - step
                new_right = acc_right - step
                if abs(new_left) <= max_adj:
                    self.accumulator["left_arm_joint_1"] = new_left
                if abs(new_right) <= max_adj:
                    self.accumulator["right_arm_joint_1"] = new_right
            elif mode == "left":
                acc = self.accumulator.get("left_arm_joint_1", 0.0)
                new_val = acc - step
                if abs(new_val) <= max_adj:
                    self.accumulator["left_arm_joint_1"] = new_val
            elif mode == "right":
                acc = self.accumulator.get("right_arm_joint_1", 0.0)
                new_val = acc - step
                if abs(new_val) <= max_adj:
                    self.accumulator["right_arm_joint_1"] = new_val

        # === 双臂 joint_3（相对方向）===
        if events.get("joint_3_positive_held"):  # A键
            if mode == "both":
                acc_left = self.accumulator.get("left_arm_joint_3", 0.0)
                acc_right = self.accumulator.get("right_arm_joint_3", 0.0)
                new_left = acc_left + step
                new_right = acc_right - step  # 相对方向
                if abs(new_left) <= max_adj:
                    self.accumulator["left_arm_joint_3"] = new_left
                if abs(new_right) <= max_adj:
                    self.accumulator["right_arm_joint_3"] = new_right
            elif mode == "left":
                acc = self.accumulator.get("left_arm_joint_3", 0.0)
                new_val = acc + step
                if abs(new_val) <= max_adj:
                    self.accumulator["left_arm_joint_3"] = new_val
            elif mode == "right":
                acc = self.accumulator.get("right_arm_joint_3", 0.0)
                new_val = acc - step
                if abs(new_val) <= max_adj:
                    self.accumulator["right_arm_joint_3"] = new_val

        if events.get("joint_3_negative_held"):  # D键
            if mode == "both":
                acc_left = self.accumulator.get("left_arm_joint_3", 0.0)
                acc_right = self.accumulator.get("right_arm_joint_3", 0.0)
                new_left = acc_left - step
                new_right = acc_right + step  # 相对方向
                if abs(new_left) <= max_adj:
                    self.accumulator["left_arm_joint_3"] = new_left
                if abs(new_right) <= max_adj:
                    self.accumulator["right_arm_joint_3"] = new_right
            elif mode == "left":
                acc = self.accumulator.get("left_arm_joint_3", 0.0)
                new_val = acc - step
                if abs(new_val) <= max_adj:
                    self.accumulator["left_arm_joint_3"] = new_val
            elif mode == "right":
                acc = self.accumulator.get("right_arm_joint_3", 0.0)
                new_val = acc + step
                if abs(new_val) <= max_adj:
                    self.accumulator["right_arm_joint_3"] = new_val

        # === 腰部 trunk ===
        if events.get("trunk_positive_held"):  # Q键
            acc = self.accumulator.get("trunk_joint_1", 0.0)
            new_val = acc + step
            if abs(new_val) <= max_adj:
                self.accumulator["trunk_joint_1"] = new_val

        if events.get("trunk_negative_held"):  # E键
            acc = self.accumulator.get("trunk_joint_1", 0.0)
            new_val = acc - step
            if abs(new_val) <= max_adj:
                self.accumulator["trunk_joint_1"] = new_val

        return self.accumulator.copy()

    def apply_to_action(self, action: dict[str, float]) -> dict[str, float]:
        """将修正量应用到动作。

        应用 applied_offset（保持的修正）+ accumulator（当前按键累积）。

        Args:
            action: 原始动作字典（格式如 {"joint_name.pos": value}）

        Returns:
            修正后的动作字典
        """
        corrected_action = action.copy()

        # 计算总修正量
        total_offset = self.get_total_offset()

        for joint_key, adjust in total_offset.items():
            action_key = f"{joint_key}.pos"
            if action_key in corrected_action:
                # 应用方向反转
                if joint_key in self._joint_inverse and self._joint_inverse[joint_key]:
                    adjust = -adjust
                corrected_action[action_key] += adjust

        return corrected_action


class ActionCorrector:
    """动作修正器 - 统一管理多种修正源。

    可在以下场景使用：
    1. replay_record - 数据增强录制
    2. eval - ACT推理评估
    3. 实时推理 - 模型输出后修正

    使用示例：
    ```python
    corrector = ActionCorrector(config, teleop=leader_device)

    # 每帧调用
    corrected_action = corrector.correct(raw_action, events=keyboard_events)
    ```
    """

    def __init__(
        self,
        config: ActionCorrectorConfig,
        teleop: Any = None,
    ):
        self.config = config
        self.teleop = teleop

        # 初始化各修正源
        if config.leader.enable:
            self.leader_corrector = LeaderCorrector(config.leader)
            logging.info("[ActionCorrector] Leader修正已启用")
        else:
            self.leader_corrector = None

        if config.keyboard.enable:
            self.keyboard_corrector = KeyboardCorrector(config.keyboard)
            logging.info("[ActionCorrector] 键盘修正已启用")
        else:
            self.keyboard_corrector = None

    def reset(self):
        """重置所有修正器状态"""
        if self.leader_corrector:
            self.leader_corrector.reset()
        if self.keyboard_corrector:
            self.keyboard_corrector.reset()
        logging.debug("[ActionCorrector] 所有修正器已重置")

    def reset_accumulators(self):
        """仅重置累积器（保留状态）"""
        if self.leader_corrector:
            self.leader_corrector.reset_accumulator()
        if self.keyboard_corrector:
            self.keyboard_corrector.reset()
        logging.debug("[ActionCorrector] 累积器已重置")

    def correct(
        self,
        action: dict[str, float],
        events: dict = None,
        leader_obs: dict[str, float] = None,
    ) -> dict[str, float]:
        """应用修正到动作。

        Args:
            action: 原始动作（来自 ACT/数据集）
            events: 键盘事件字典（可选）
            leader_obs: Leader observation（可选，若未提供则从 teleop 获取）

        Returns:
            corrected_action: 修正后的动作
        """
        if not self.config.enable:
            return action.copy()

        corrected_action = action.copy()

        # 1. Leader 修正
        if self.leader_corrector:
            # 获取 Leader 观测
            if leader_obs is None and self.teleop is not None:
                leader_obs = self.teleop.get_action()

            if leader_obs is not None:
                leader_adjustments = self.leader_corrector.process(leader_obs)
                for joint_name, adjust in leader_adjustments.items():
                    action_key = f"{joint_name}.pos"
                    if action_key in corrected_action:
                        corrected_action[action_key] += adjust

        # 2. 键盘修正
        if self.keyboard_corrector and events:
            self.keyboard_corrector.update_accumulator(events)
            corrected_action = self.keyboard_corrector.apply_to_action(corrected_action)

        return corrected_action

    def get_leader_state(self) -> str:
        """获取 Leader 修正器当前状态"""
        if self.leader_corrector:
            return self.leader_corrector.state
        return "disabled"

    def get_accumulators(self) -> dict:
        """获取所有修正器状态"""
        accumulators = {}
        if self.leader_corrector:
            accumulators["leader"] = {
                "accumulator": self.leader_corrector.accumulator.copy(),
                "applied_offset": self.leader_corrector.applied_offset.copy(),
                "total": self.leader_corrector.get_total_offset(),
            }
        if self.keyboard_corrector:
            accumulators["keyboard"] = {
                "accumulator": self.keyboard_corrector.get_accumulator(),
                "applied_offset": self.keyboard_corrector.applied_offset.copy(),
                "total": self.keyboard_corrector.get_total_offset(),
            }
        return accumulators


# 便捷函数：创建默认配置的修正器
def create_action_corrector(
    enable_leader: bool = True,
    enable_keyboard: bool = True,
    teleop: Any = None,
    **kwargs,
) -> ActionCorrector:
    """创建动作修正器的便捷函数。

    Args:
        enable_leader: 是否启用 Leader 修正
        enable_keyboard: 是否启用键盘修正
        teleop: Leader 设备实例
        **kwargs: 其他配置参数（传递给各修正器配置）

    Returns:
        ActionCorrector 实例
    """
    leader_cfg = LeaderAdjustConfig(enable=enable_leader, **{
        k: v for k, v in kwargs.items() if k in LeaderAdjustConfig.__dataclass_fields__
    })
    keyboard_cfg = KeyAdjustConfig(enable=enable_keyboard, **{
        k: v for k, v in kwargs.items() if k in KeyAdjustConfig.__dataclass_fields__
    })
    config = ActionCorrectorConfig(
        enable=True,
        leader=leader_cfg,
        keyboard=keyboard_cfg,
    )
    return ActionCorrector(config, teleop=teleop)