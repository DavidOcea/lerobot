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
2. KeyboardCorrector - 键盘辅助修正（关节选择模式 + 动态反转）

按键控制模式（方案A改进版）：
- 数字键 1-6: 选择 joint_1 ~ joint_6
- 数字键 8-9: 选择 trunk_joint_1, trunk_joint_2
- A/D: 正向/负向调整（通用）
- J/K/L: 左臂/双臂/右臂模式
- U/O: 双臂模式下反转左臂/右臂当前选中关节
- R: 重置选择状态和反转状态（保留累积修正量）

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
    """键盘微调配置（关节选择模式 + 动态反转）

    按键映射：
    - 1-6: 选择 joint_1 ~ joint_6
    - 8: 选择 trunk_joint_1
    - 9: 选择 trunk_joint_2
    - A: 正向调整（+）
    - D: 负向调整（-）
    - J: 左臂模式
    - K: 双臂模式
    - L: 右臂模式
    - U: 反转左臂当前关节（双臂模式）
    - O: 反转右臂当前关节（双臂模式）
    - R: 重置选择状态和反转状态
    """

    enable: bool = True

    # 平滑参数（精细控制）
    step_per_frame: float = 0.2      # 每帧调整幅度（度）
    max_adjustment: float = 5.0      # 单次按键最大累积调整量（度）

    # 默认状态
    default_joint: int = 1           # 默认选中 joint_1
    default_arm_mode: str = "both"   # 默认双臂模式

    # 关节方向反转（初始配置，运行时可动态修改）
    joint_inverse: str | dict[str, bool] = field(default_factory=dict)


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
    """键盘辅助修正器（关节选择模式 + 动态反转）。

    状态管理：
    - current_joint: 当前选中的关节编号（1-6, 8-9）
    - arm_mode: 臂模式（left/both/right）
    - joint_inverse: 动态反转状态（运行时可修改）

    按键功能：
    - 数字键: 选择关节
    - A/D: 调整正向/负向
    - J/K/L: 选择臂模式
    - U/O: 反转左臂/右臂当前关节
    - R: 重置选择和反转状态

    变量说明：
    - accumulator: 当前按键会话的临时累积
    - applied_offset: 已应用并保持的修正量
    """

    # 关节编号到关节名称的映射
    JOINT_NAMES = {
        1: ["left_arm_joint_1", "right_arm_joint_1"],
        2: ["left_arm_joint_2", "right_arm_joint_2"],
        3: ["left_arm_joint_3", "right_arm_joint_3"],
        4: ["left_arm_joint_4", "right_arm_joint_4"],
        5: ["left_arm_joint_5", "right_arm_joint_5"],
        6: ["left_arm_joint_6", "right_arm_joint_6"],
        8: ["trunk_joint_1"],
        9: ["trunk_joint_2"],
    }

    # 关节显示名称
    JOINT_DISPLAY = {
        1: "Joint_1",
        2: "Joint_2",
        3: "Joint_3",
        4: "Joint_4",
        5: "Joint_5",
        6: "Joint_6",
        8: "Trunk_1",
        9: "Trunk_2",
    }

    def __init__(self, cfg: KeyAdjustConfig):
        self.cfg = cfg

        # 状态变量
        self.current_joint: int = cfg.default_joint
        self.arm_mode: str = cfg.default_arm_mode
        self._joint_inverse: dict[str, bool] = {}

        # 修正累积
        self.accumulator: dict[str, float] = {}
        self.applied_offset: dict[str, float] = {}

        # 按键状态追踪
        self._prev_key_state: dict[str, bool] = {}

        # 解析初始反转配置
        self._parse_joint_inverse()

        # 打印初始状态
        self._print_status()

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
            self._joint_inverse = raw.copy() if raw else {}

    def _print_status(self):
        """打印当前状态"""
        joint_display = self.JOINT_DISPLAY.get(self.current_joint, f"Joint_{self.current_joint}")
        mode_display = {"left": "左臂", "both": "双臂", "right": "右臂"}[self.arm_mode]

        # 获取当前关节的反转状态
        inverse_status = self._get_inverse_status_display()

        print(f"[{joint_display}|{mode_display}] {inverse_status}")

    def _get_inverse_status_display(self) -> str:
        """获取反转状态显示字符串"""
        joints = self.JOINT_NAMES.get(self.current_joint, [])

        if self.current_joint in [8, 9]:  # Trunk
            return ""

        if self.arm_mode == "both":
            left_joint = joints[0]
            right_joint = joints[1]
            left_inverse = self._joint_inverse.get(left_joint, False)
            right_inverse = self._joint_inverse.get(right_joint, False)
            return f"左臂反转:{'ON' if left_inverse else 'OFF'}, 右臂反转:{'ON' if right_inverse else 'OFF'}"
        elif self.arm_mode == "left":
            left_joint = joints[0]
            left_inverse = self._joint_inverse.get(left_joint, False)
            return f"反转:{'ON' if left_inverse else 'OFF'} (单臂模式可用A/D控制方向)"
        elif self.arm_mode == "right":
            right_joint = joints[1]
            right_inverse = self._joint_inverse.get(right_joint, False)
            return f"反转:{'ON' if right_inverse else 'OFF'} (单臂模式可用A/D控制方向)"
        return ""

    def select_joint(self, joint_num: int):
        """选择关节"""
        if joint_num not in self.JOINT_NAMES:
            logging.warning(f"[KeyboardCorrector] 无效关节编号: {joint_num}")
            return

        self.current_joint = joint_num
        self._print_status()

    def set_arm_mode(self, mode: str):
        """设置臂模式"""
        if mode not in ["left", "both", "right"]:
            logging.warning(f"[KeyboardCorrector] 无效臂模式: {mode}")
            return

        self.arm_mode = mode
        self._print_status()

    def toggle_inverse(self, arm: str):
        """切换指定臂当前选中关节的反转状态

        Args:
            arm: "left" 或 "right"
        """
        if self.current_joint in [8, 9]:  # Trunk 不需要反转
            print("[提示] 腰部关节不需要反转控制")
            return

        if self.arm_mode != "both":
            print("[提示] 单臂模式不需要反转，可用A/D控制正负方向")
            return

        joints = self.JOINT_NAMES.get(self.current_joint, [])
        if not joints:
            return

        if arm == "left":
            joint_name = joints[0]  # left_arm_joint_X
        elif arm == "right":
            joint_name = joints[1]  # right_arm_joint_X
        else:
            return

        # Toggle反转状态
        current = self._joint_inverse.get(joint_name, False)
        new_state = not current
        self._joint_inverse[joint_name] = new_state

        arm_display = "左臂" if arm == "left" else "右臂"
        print(f"[{self.JOINT_DISPLAY[self.current_joint]}|双臂] {arm_display}反转: {'ON' if current else 'OFF'} → {'ON' if new_state else 'OFF'}")

    def reset_state(self):
        """重置选择状态和反转状态（保留累积修正量）"""
        self.current_joint = self.cfg.default_joint
        self.arm_mode = self.cfg.default_arm_mode
        self._joint_inverse.clear()
        print(f"[状态重置] 关节={self.JOINT_DISPLAY[self.current_joint]}, 模式={'双臂' if self.arm_mode=='both' else self.arm_mode}, 反转:全部OFF")

    def reset(self):
        """重置累积修正量（保留选择和反转状态）"""
        self.accumulator.clear()
        self.applied_offset.clear()
        logging.debug("[KeyboardCorrector] 累积修正量已清零")

    def get_accumulator(self) -> dict[str, float]:
        """获取当前累积修正量"""
        return self.accumulator.copy()

    def get_total_offset(self) -> dict[str, float]:
        """获取总修正量（applied_offset + accumulator）"""
        total = self.applied_offset.copy()
        for joint, val in self.accumulator.items():
            total[joint] = total.get(joint, 0.0) + val
        return total

    def get_joint_inverse(self) -> dict[str, bool]:
        """获取当前反转状态"""
        return self._joint_inverse.copy()

    def update_accumulator(self, events: dict) -> dict[str, float]:
        """根据键盘事件更新累积器。

        Args:
            events: 键盘事件字典，包含:
                - joint_selected: int (1-6, 8-9) 关节选择
                - positive_held: bool (A键) 正向调整
                - negative_held: bool (D键) 负向调整
                - arm_mode_left: bool (J键)
                - arm_mode_both: bool (K键)
                - arm_mode_right: bool (L键)
                - inverse_left: bool (U键) 反转左臂
                - inverse_right: bool (O键) 反转右臂
                - reset_state: bool (R键) 重置状态

        Returns:
            更新后的累积器
        """
        step = self.cfg.step_per_frame
        max_adj = self.cfg.max_adjustment

        # === 处理关节选择 ===
        for i in [1, 2, 3, 4, 5, 6, 8, 9]:
            if events.get(f"joint_{i}_selected"):
                self.select_joint(i)

        # === 处理臂模式选择 ===
        if events.get("arm_mode_left"):
            self.set_arm_mode("left")
        if events.get("arm_mode_both"):
            self.set_arm_mode("both")
        if events.get("arm_mode_right"):
            self.set_arm_mode("right")

        # === 处理反转控制 ===
        if events.get("inverse_left"):
            self.toggle_inverse("left")
        if events.get("inverse_right"):
            self.toggle_inverse("right")

        # === 处理状态重置 ===
        if events.get("reset_state"):
            self.reset_state()

        # === 获取当前关节名称 ===
        joints = self.JOINT_NAMES.get(self.current_joint, [])
        if not joints:
            return self.accumulator.copy()

        # === 检测按键释放，保存修正 ===
        for direction in ["positive", "negative"]:
            event_key = f"{direction}_held"
            current_state = events.get(event_key, False)
            prev_state = self._prev_key_state.get(event_key, False)

            # 检测从 True -> False（按键释放）
            if prev_state and not current_state:
                for joint in joints:
                    # 根据臂模式决定处理哪些关节
                    should_process = False
                    if joint.startswith("left_arm") and self.arm_mode in ["both", "left"]:
                        should_process = True
                    elif joint.startswith("right_arm") and self.arm_mode in ["both", "right"]:
                        should_process = True
                    elif joint.startswith("trunk"):
                        should_process = True

                    if should_process and joint in self.accumulator:
                        acc_val = self.accumulator[joint]
                        if abs(acc_val) > 0.001:
                            self.applied_offset[joint] = self.applied_offset.get(joint, 0.0) + acc_val
                            logging.debug(f"[KeyboardCorrector] Key released: saved {joint} offset={acc_val:.3f}")
                        self.accumulator[joint] = 0.0

            self._prev_key_state[event_key] = current_state

        # === 处理正向调整（A键）===
        if events.get("positive_held"):
            delta = step
            for joint in joints:
                should_adjust = False
                if joint.startswith("left_arm") and self.arm_mode in ["both", "left"]:
                    should_adjust = True
                elif joint.startswith("right_arm") and self.arm_mode in ["both", "right"]:
                    should_adjust = True
                elif joint.startswith("trunk"):
                    should_adjust = True

                if should_adjust:
                    acc = self.accumulator.get(joint, 0.0)
                    # 应用反转
                    if self._joint_inverse.get(joint, False):
                        delta_adjusted = -delta
                    else:
                        delta_adjusted = delta

                    new_val = acc + delta_adjusted
                    if abs(new_val) <= max_adj:
                        self.accumulator[joint] = new_val

        # === 处理负向调整（D键）===
        if events.get("negative_held"):
            delta = -step
            for joint in joints:
                should_adjust = False
                if joint.startswith("left_arm") and self.arm_mode in ["both", "left"]:
                    should_adjust = True
                elif joint.startswith("right_arm") and self.arm_mode in ["both", "right"]:
                    should_adjust = True
                elif joint.startswith("trunk"):
                    should_adjust = True

                if should_adjust:
                    acc = self.accumulator.get(joint, 0.0)
                    # 应用反转
                    if self._joint_inverse.get(joint, False):
                        delta_adjusted = -delta  # 反转后负向变成正向
                    else:
                        delta_adjusted = delta

                    new_val = acc + delta_adjusted
                    if abs(new_val) <= max_adj:
                        self.accumulator[joint] = new_val

        return self.accumulator.copy()

    def apply_to_action(self, action: dict[str, float]) -> dict[str, float]:
        """将修正量应用到动作。

        Args:
            action: 原始动作字典（格式如 {"joint_name.pos": value}）

        Returns:
            修正后的动作字典
        """
        corrected_action = action.copy()
        total_offset = self.get_total_offset()

        for joint_key, adjust in total_offset.items():
            action_key = f"{joint_key}.pos"
            if action_key in corrected_action:
                # 注意：反转已在 update_accumulator 中处理，这里直接添加
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
            logging.info("[ActionCorrector] 键盘修正已启用（关节选择模式）")
        else:
            self.keyboard_corrector = None

    def reset(self):
        """重置累积修正量（保留选择和反转状态）"""
        if self.leader_corrector:
            self.leader_corrector.reset()
        if self.keyboard_corrector:
            self.keyboard_corrector.reset()
        logging.debug("[ActionCorrector] 累积修正量已重置")

    def reset_state(self):
        """重置选择状态和反转状态"""
        if self.keyboard_corrector:
            self.keyboard_corrector.reset_state()

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

    def get_keyboard_status(self) -> dict:
        """获取键盘修正器状态"""
        if self.keyboard_corrector:
            return {
                "current_joint": self.keyboard_corrector.current_joint,
                "arm_mode": self.keyboard_corrector.arm_mode,
                "joint_inverse": self.keyboard_corrector.get_joint_inverse(),
                "accumulator": self.keyboard_corrector.get_accumulator(),
                "applied_offset": self.keyboard_corrector.applied_offset.copy(),
            }
        return {}

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