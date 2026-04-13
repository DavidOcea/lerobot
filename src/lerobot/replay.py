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
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=2
```

Example replay with bimanual so100:
```shell
python -m lerobot.replay \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

Example replay with recording (replay_record):
```shell
python -m lerobot.replay \
    --robot.type=supre_robot_follower \
    --teleop.type=supre_robot_leader \
    --dataset.repo_id=my_dataset \
    --dataset.episode=0 \
    --dataset.enable_replay_record=true \
    --dataset.record_repo_id=my_augmented_dataset \
    --dataset.record_task="Pick and place box"
```

"""

import logging
import numpy as np
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    ros2_follower,
    supre_robot_follower
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


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
    step_per_frame: float = 0.1      # 每帧调整幅度（度）
    max_adjustment: float = 5.0      # 单次按键最大累积调整量（度）

    # 双臂控制模式：left / right / both
    arm_control_mode: str = "both"

    # 按键映射
    # W/X: 双臂joint_1 正/反（同向）
    # A/D: 双臂joint_3 正/反（相对方向）
    # Q/E: 腰部 trunk 正/反
    # K: 切换控制模式
    keys_joint_1_positive: str = "w"
    keys_joint_1_negative: str = "x"
    keys_joint_3_positive: str = "a"
    keys_joint_3_negative: str = "d"
    keys_trunk_positive: str = "q"
    keys_trunk_negative: str = "e"
    key_mode_toggle: str = "k"


@dataclass
class ReplayRecordConfig:
    """Configuration for replay with recording (data augmentation)."""

    # Enable replay_record mode (replay + noise + user intervention + save)
    enable: bool = False

    # New dataset repo_id for recorded data
    record_repo_id: str | None = None

    # Task description for recorded dataset
    record_task: str | None = None

    # Root directory for recorded dataset
    record_root: str | Path | None = None

    # Noise parameters
    noise_std: float = 0.02  # Standard deviation for action noise (rad)
    noise_seed: int | None = None  # Random seed for reproducibility

    # Leader intervention parameters (状态机设计，滞回机制)
    leader_adjust: LeaderAdjustConfig = LeaderAdjustConfig()

    # Keyboard adjustment parameters（平滑精细控制）
    key_adjust: KeyAdjustConfig = KeyAdjustConfig()

    # Timestamp tolerance for recorded dataset
    tolerance_s: float = 0.03  # Same as record.py for actual timestamps

    # Success/fail keys
    success_key_timeout: float = 5.0  # Seconds to wait for success key after episode ends


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30
    # Number of replay loops. 0 = infinite loop.
    num_loops: int = 1

    # Replay_record configuration
    replay_record: ReplayRecordConfig = ReplayRecordConfig()


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Teleoperator for Leader intervention (optional)
    teleop: TeleoperatorConfig | None = None
    # Use vocal synthesis to read events.
    play_sounds: bool = True


def add_noise_to_action(action: dict[str, float], noise_std: float, rng: np.random.Generator) -> dict[str, float]:
    """Add Gaussian noise to action values.

    Args:
        action: Original action dict with joint positions.
        noise_std: Standard deviation of noise in radians.
        rng: Random number generator.

    Returns:
        Action dict with added noise.
    """
    noisy_action = {}
    for key, value in action.items():
        # Skip gripper joints (usually end with .pos and contain "gripper" or joint_7)
        if "gripper" in key.lower() or key.endswith("_joint_7.pos"):
            noisy_action[key] = value  # No noise for gripper
        else:
            noise = rng.normal(0, noise_std)
            noisy_action[key] = value + noise
    return noisy_action


class LeaderAdjustStateMachine:
    """Leader微调状态机（滞回设计）。

    状态：
    - idle: 等待触发
    - adjusting: 微调状态（Leader连续影响Follower）

    流程：
    1. 等待Leader变化 > 5度 → 进入微调状态
    2. 重置基准位置，从此刻起计算delta
    3. 微调期间：follower += alpha * (leader_pos - trigger_baseline)
    4. 连续5帧变化 < 1度 → 退出微调状态
    """

    def __init__(self, cfg: LeaderAdjustConfig):
        self.cfg = cfg
        self.state = "idle"
        self.trigger_baseline: dict[str, float] | None = None  # 触发时的Leader位置基准
        self.prev_leader_pos: dict[str, float] | None = None   # 上一帧Leader位置（用于计算变化量）
        self.exit_frame_count = 0  # 退出计数

    def deg_to_rad(self, deg: float) -> float:
        """度转弧度"""
        return deg * np.pi / 180.0

    def rad_to_deg(self, rad: float) -> float:
        """弧度转度"""
        return rad * 180.0 / np.pi

    def process(
        self,
        leader_obs: dict[str, float],
        final_action: dict[str, float],
    ) -> str:
        """处理Leader输入，返回当前状态。

        Args:
            leader_obs: Leader observation (包含 .pos 的关节位置)
            final_action: 待修改的动作字典

        Returns:
            当前状态: "idle" 或 "adjusting"
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
            # 等待状态：检测是否触发
            if max_delta_deg > self.cfg.trigger_threshold_deg:
                # 触发：进入微调状态
                self.state = "adjusting"
                self.trigger_baseline = leader_pos.copy()  # 记录触发基准
                self.exit_frame_count = 0
                logging.info(f"Leader微调触发: 变化量={max_delta_deg:.1f}度")

        elif self.state == "adjusting":
            # 微调状态：计算相对于触发基准的偏移
            if self.trigger_baseline is not None:
                for joint_key, leader_value in leader_pos.items():
                    joint_name = joint_key.replace(".pos", "")
                    baseline = self.trigger_baseline.get(joint_key, leader_value)
                    delta_rad = leader_value - baseline  # 相对触发点的偏移

                    action_key = f"{joint_name}.pos"
                    if action_key in final_action:
                        # 应用修正
                        final_action[action_key] += self.cfg.adjust_alpha * delta_rad

            # 检测退出：变化量小于阈值
            if max_delta_deg < self.cfg.exit_threshold_deg:
                self.exit_frame_count += 1
                if self.exit_frame_count >= self.cfg.exit_frame_count:
                    # 连续N帧变化小，退出微调
                    self.state = "idle"
                    self.trigger_baseline = None
                    self.exit_frame_count = 0
                    logging.info(f"Leader微调退出: 连续{self.cfg.exit_frame_count}帧变化<{self.cfg.exit_threshold_deg}度")
            else:
                # 变化大，重置退出计数
                self.exit_frame_count = 0

        # === 更新上一帧位置 ===
        self.prev_leader_pos = leader_pos.copy()

        return self.state


def replay_record_loop(
    robot: Robot,
    dataset: LeRobotDataset,
    actions: Any,
    new_dataset: LeRobotDataset,
    teleop: Teleoperator | None,
    events: dict,
    fps: int,
    cfg: ReplayRecordConfig,
    single_task: str,
) -> bool:
    """Execute replay with recording, allowing user intervention.

    Args:
        robot: Robot instance.
        dataset: Source dataset to replay.
        actions: Action column from dataset.
        new_dataset: Dataset to record new data.
        teleop: Teleoperator for Leader intervention (optional).
        events: Keyboard events dict.
        fps: Frames per second.
        cfg: ReplayRecordConfig.
        single_task: Task description.

    Returns:
        True if episode was saved successfully, False if discarded.
    """
    rng = np.random.default_rng(cfg.noise_seed)

    # Leader微调状态机（滞回设计）
    leader_state_machine: LeaderAdjustStateMachine | None = None
    if teleop is not None and cfg.leader_adjust.enable:
        leader_state_machine = LeaderAdjustStateMachine(cfg.leader_adjust)

    # 键盘调整累积器（平滑控制）
    key_adjust_accumulator: dict[str, float] = {}

    # Episode recording state
    episode_start = time.perf_counter()
    frame_index = 0

    logging.info(f"Replay_record started with noise_std={cfg.noise_std}")
    logging.info(f"Key adjust mode: {cfg.key_adjust.arm_control_mode}, step={cfg.key_adjust.step_per_frame} deg/frame")
    if leader_state_machine:
        logging.info(f"Leader adjust: trigger={cfg.leader_adjust.trigger_threshold_deg}度, exit={cfg.leader_adjust.exit_threshold_deg}度/{cfg.leader_adjust.exit_frame_count}帧")

    for idx in range(dataset.num_frames):
        # === 1. 记录实际时间戳 ===
        actual_timestamp = time.perf_counter() - episode_start
        loop_start = time.perf_counter()

        # === 2. 获取原始动作 ===
        action_array = actions[idx]["action"]
        base_action = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            base_action[name] = float(action_array[i])

        # === 3. 添加扰动 ===
        noisy_action = add_noise_to_action(base_action, cfg.noise_std, rng)
        final_action = noisy_action.copy()

        # === 4. Leader微调（状态机设计）===
        if leader_state_machine is not None:
            leader_obs = teleop.get_observation()
            state = leader_state_machine.process(leader_obs, final_action)
            # 状态会显示在日志中

        # === 5. 按键平滑微调 ===
        if cfg.key_adjust.enable:
            apply_key_adjustment_smooth(
                final_action=final_action,
                events=events,
                key_cfg=cfg.key_adjust,
                accumulator=key_adjust_accumulator,
            )

        # === 6. 获取observation ===
        observation = robot.get_observation()

        # === 7. 执行动作 ===
        sent_action = robot.send_action(final_action)

        # === 8. 录制数据 ===
        if new_dataset is not None:
            observation_frame = build_dataset_frame(new_dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(new_dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            new_dataset.add_frame(frame, task=single_task, timestamp=actual_timestamp)

        frame_index += 1

        # === 9. 检测成功/失败按键 ===
        if events.get("mark_fail"):
            logging.info("Episode marked as FAIL, discarding...")
            new_dataset.clear_episode_buffer()
            events["mark_fail"] = False
            return False

        if events.get("exit_early"):
            events["exit_early"] = False
            break

        # === 10. 时间同步 ===
        dt_s = time.perf_counter() - loop_start
        expected_interval = 1.0 / fps
        wait_time = expected_interval - dt_s
        if wait_time > 0:
            busy_wait(wait_time)

    # === 11. Episode结束后等待成功按键 ===
    logging.info("Episode replay completed. Press 'S' to save, 'F' to discard...")

    wait_start = time.perf_counter()
    while time.perf_counter() - wait_start < cfg.success_key_timeout:
        if events.get("mark_success"):
            logging.info("Episode marked as SUCCESS, saving...")
            new_dataset.save_episode()
            events["mark_success"] = False
            return True
        if events.get("mark_fail"):
            logging.info("Episode marked as FAIL, discarding...")
            new_dataset.clear_episode_buffer()
            events["mark_fail"] = False
            return False
        time.sleep(0.1)

    # 超时未按键，默认保存
    logging.info("Timeout, auto-saving episode...")
    new_dataset.save_episode()
    return True


def init_replay_record_keyboard_listener(events: dict, key_cfg: KeyAdjustConfig) -> Any:
    """Initialize keyboard listener for replay_record mode.

    支持按住持续调整（平滑精细控制）：
    - S: 标记成功，保存
    - F: 标记失败，丢弃
    - ESC: 停止 replay_record
    - W/X: 双臂 joint_1 正/反（同向）
    - A/D: 双臂 joint_3 正/反（相对）
    - Q/E: 腰部 trunk 正/反
    - K: 切换控制模式 (left/right/both)
    """
    from pynput import keyboard

    def get_char(key) -> str | None:
        """获取按键字符"""
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            return None
        except:
            return None

    def next_mode(current: str) -> str:
        """切换控制模式"""
        modes = ["both", "left", "right"]
        idx = modes.index(current) if current in modes else 0
        return modes[(idx + 1) % len(modes)]

    def on_press(key):
        try:
            char = get_char(key)

            # === 成功/失败按键（单次触发）===
            if char == 's':
                print("S key pressed. Marking as SUCCESS...")
                events["mark_success"] = True
            elif char == 'f':
                print("F key pressed. Marking as FAIL...")
                events["mark_fail"] = True
            elif key == keyboard.Key.esc:
                print("ESC key pressed. Stopping replay_record...")
                events["stop_replay_record"] = True
                events["mark_fail"] = True

            # === 控制模式切换 ===
            elif char == key_cfg.key_mode_toggle:
                current_mode = events.get("arm_control_mode", key_cfg.arm_control_mode)
                new_mode = next_mode(current_mode)
                events["arm_control_mode"] = new_mode
                print(f"Arm control mode switched: {current_mode} -> {new_mode}")

            # === 微调按键（按住持续）===
            elif char == key_cfg.keys_joint_1_positive:  # W
                events["joint_1_positive_held"] = True
            elif char == key_cfg.keys_joint_1_negative:  # X
                events["joint_1_negative_held"] = True
            elif char == key_cfg.keys_joint_3_positive:  # A
                events["joint_3_positive_held"] = True
            elif char == key_cfg.keys_joint_3_negative:  # D
                events["joint_3_negative_held"] = True
            elif char == key_cfg.keys_trunk_positive:    # Q
                events["trunk_positive_held"] = True
            elif char == key_cfg.keys_trunk_negative:    # E
                events["trunk_negative_held"] = True

        except Exception as e:
            print(f"Error handling key press: {e}")

    def on_release(key):
        """按键释放：停止调整"""
        try:
            char = get_char(key)

            # === 微调按键释放 ===
            if char == key_cfg.keys_joint_1_positive:
                events["joint_1_positive_held"] = False
            elif char == key_cfg.keys_joint_1_negative:
                events["joint_1_negative_held"] = False
            elif char == key_cfg.keys_joint_3_positive:
                events["joint_3_positive_held"] = False
            elif char == key_cfg.keys_joint_3_negative:
                events["joint_3_negative_held"] = False
            elif char == key_cfg.keys_trunk_positive:
                events["trunk_positive_held"] = False
            elif char == key_cfg.keys_trunk_negative:
                events["trunk_negative_held"] = False

        except Exception as e:
            print(f"Error handling key release: {e}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def apply_key_adjustment_smooth(
    final_action: dict,
    events: dict,
    key_cfg: KeyAdjustConfig,
    accumulator: dict[str, float],
) -> None:
    """应用平滑精细的键盘微调。

    特点：
    - 每帧渐进调整 0.1度（而非跳变）
    - 按住持续调整，释放停止
    - 最大累积调整量 5度

    Args:
        final_action: 动作字典
        events: 按键事件
        key_cfg: 键盘配置
        accumulator: 累积调整量字典
    """
    mode = events.get("arm_control_mode", key_cfg.arm_control_mode)
    step = key_cfg.step_per_frame
    max_adj = key_cfg.max_adjustment

    # === 双臂 joint_1（同向）===
    if events.get("joint_1_positive_held"):  # W键
        if mode == "both":
            acc_left = accumulator.get("left_arm_joint_1", 0)
            acc_right = accumulator.get("right_arm_joint_1", 0)
            if abs(acc_left) < max_adj:
                accumulator["left_arm_joint_1"] = acc_left + step
            if abs(acc_right) < max_adj:
                accumulator["right_arm_joint_1"] = acc_right + step
        elif mode == "left":
            acc = accumulator.get("left_arm_joint_1", 0)
            if abs(acc) < max_adj:
                accumulator["left_arm_joint_1"] = acc + step
        elif mode == "right":
            acc = accumulator.get("right_arm_joint_1", 0)
            if abs(acc) < max_adj:
                accumulator["right_arm_joint_1"] = acc + step

    if events.get("joint_1_negative_held"):  # X键
        if mode == "both":
            acc_left = accumulator.get("left_arm_joint_1", 0)
            acc_right = accumulator.get("right_arm_joint_1", 0)
            if abs(acc_left) < max_adj:
                accumulator["left_arm_joint_1"] = acc_left - step
            if abs(acc_right) < max_adj:
                accumulator["right_arm_joint_1"] = acc_right - step
        elif mode == "left":
            acc = accumulator.get("left_arm_joint_1", 0)
            if abs(acc) < max_adj:
                accumulator["left_arm_joint_1"] = acc - step
        elif mode == "right":
            acc = accumulator.get("right_arm_joint_1", 0)
            if abs(acc) < max_adj:
                accumulator["right_arm_joint_1"] = acc - step

    # === 双臂 joint_3（相对方向）===
    if events.get("joint_3_positive_held"):  # A键
        if mode == "both":
            acc_left = accumulator.get("left_arm_joint_3", 0)
            acc_right = accumulator.get("right_arm_joint_3", 0)
            if abs(acc_left) < max_adj:
                accumulator["left_arm_joint_3"] = acc_left + step
            if abs(acc_right) < max_adj:
                accumulator["right_arm_joint_3"] = acc_right - step  # 相对方向
        elif mode == "left":
            acc = accumulator.get("left_arm_joint_3", 0)
            if abs(acc) < max_adj:
                accumulator["left_arm_joint_3"] = acc + step
        elif mode == "right":
            acc = accumulator.get("right_arm_joint_3", 0)
            if abs(acc) < max_adj:
                accumulator["right_arm_joint_3"] = acc - step

    if events.get("joint_3_negative_held"):  # D键
        if mode == "both":
            acc_left = accumulator.get("left_arm_joint_3", 0)
            acc_right = accumulator.get("right_arm_joint_3", 0)
            if abs(acc_left) < max_adj:
                accumulator["left_arm_joint_3"] = acc_left - step
            if abs(acc_right) < max_adj:
                accumulator["right_arm_joint_3"] = acc_right + step  # 相对方向
        elif mode == "left":
            acc = accumulator.get("left_arm_joint_3", 0)
            if abs(acc) < max_adj:
                accumulator["left_arm_joint_3"] = acc - step
        elif mode == "right":
            acc = accumulator.get("right_arm_joint_3", 0)
            if abs(acc) < max_adj:
                accumulator["right_arm_joint_3"] = acc + step

    # === 腰部 trunk ===
    if events.get("trunk_positive_held"):  # Q键
        acc = accumulator.get("trunk_joint_1", 0)
        if abs(acc) < max_adj:
            accumulator["trunk_joint_1"] = acc + step

    if events.get("trunk_negative_held"):  # E键
        acc = accumulator.get("trunk_joint_1", 0)
        if abs(acc) < max_adj:
            accumulator["trunk_joint_1"] = acc - step

    # === 应用累积调整量到动作 ===
    for joint_key, adjust in accumulator.items():
        action_key = f"{joint_key}.pos"
        if action_key in final_action:
            final_action[action_key] += adjust


@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.hf_dataset.select_columns("action")
    fps = dataset.fps if hasattr(dataset, 'fps') else cfg.dataset.fps

    # === 连接设备 ===
    robot.connect()

    # 连接 Teleoperator（用于 Leader 介入）
    teleop = None
    if cfg.dataset.replay_record.enable and cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logging.info("Teleoperator connected for Leader intervention")

    # === 判断模式 ===
    if cfg.dataset.replay_record.enable:
        # === Replay Record 模式 ===
        replay_cfg = cfg.dataset.replay_record

        # 验证配置
        if replay_cfg.record_repo_id is None:
            raise ValueError("record_repo_id is required when enable_replay_record=True")
        if replay_cfg.record_task is None:
            raise ValueError("record_task is required when enable_replay_record=True")

        # 创建新数据集
        dataset_features = dataset.features

        # 相机配置：根据相机数量动态计算（与 record.py 一致）
        num_cameras = len(robot.cameras) if hasattr(robot, 'cameras') and robot.cameras else 1
        image_writer_threads = 4 * num_cameras  # 每个相机4个线程

        new_dataset = LeRobotDataset.create(
            repo_id=replay_cfg.record_repo_id,
            fps=fps,
            root=replay_cfg.record_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            tolerance_s=replay_cfg.tolerance_s,
            image_writer_processes=0,
            image_writer_threads=image_writer_threads,
        )

        logging.info(f"Dataset created with {num_cameras} cameras, {image_writer_threads} image writer threads")

        # 初始化键盘监听
        listener, events = init_keyboard_listener()
        events["mark_success"] = False
        events["mark_fail"] = False
        events["stop_replay_record"] = False
        events["arm_control_mode"] = replay_cfg.key_adjust.arm_control_mode  # 初始化控制模式

        # 添加 replay_record 专用按键监听（平滑精细控制）
        replay_listener = init_replay_record_keyboard_listener(events, replay_cfg.key_adjust)

        log_say("Replay with recording started. S=save, F=discard, K=mode toggle", cfg.play_sounds, blocking=True)
        logging.info("Key adjust controls: W/X=joint1, A/D=joint3, Q/E=trunk, K=mode")

        episode_count = 0
        while not events["stop_replay_record"]:
            # 执行一个 episode 的 replay_record
            success = replay_record_loop(
                robot=robot,
                dataset=dataset,
                actions=actions,
                new_dataset=new_dataset,
                teleop=teleop,
                events=events,
                fps=fps,
                cfg=replay_cfg,
                single_task=replay_cfg.record_task,
            )

            if success:
                episode_count += 1
                logging.info(f"Episode {episode_count} saved successfully")
            else:
                logging.info("Episode discarded")

            # 检查是否停止
            if events["stop_replay_record"]:
                break

            # 询问是否继续
            log_say("Press ESC to stop, or continue with next episode", cfg.play_sounds, blocking=False)
            time.sleep(1.0)  # 给用户时间反应

        # 清理
        if replay_listener is not None:
            replay_listener.stop()
        if listener is not None:
            listener.stop()

        new_dataset.stop_image_writer()
        logging.info(f"Replay_record completed. Total episodes saved: {episode_count}")

    else:
        # === 普通 Replay 模式 ===
        loop_count = 0
        max_loops = cfg.dataset.num_loops if cfg.dataset.num_loops > 0 else float('inf')

        log_say("Replaying episode", cfg.play_sounds, blocking=True)

        while loop_count < max_loops:
            start_episode_t = time.perf_counter()

            for idx in range(dataset.num_frames):
                action_array = actions[idx]["action"]
                action = {}
                for i, name in enumerate(dataset.features["action"]["names"]):
                    action[name] = action_array[i]

                robot.send_action(action)

                dt_s = time.perf_counter() - start_episode_t
                expected_time = idx / fps
                wait_time = expected_time - dt_s
                if wait_time > 0:
                    busy_wait(wait_time)

            loop_count += 1
            logging.info(f"Replay loop {loop_count} completed")

    # === 断开连接 ===
    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()


if __name__ == "__main__":
    replay()
