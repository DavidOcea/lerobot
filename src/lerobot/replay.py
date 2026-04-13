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

    # Leader intervention parameters (混合模式)
    leader_adjust_enabled: bool = True
    leader_threshold: float = 0.5  # Leader movement threshold (rad) to trigger intervention
    leader_alpha: float = 0.3  # Correction coefficient for Leader delta

    # Keyboard adjustment parameters
    key_adjust_enabled: bool = True
    key_adjust_step: float = 1.0  # Step size for keyboard adjustment (rad)

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

    # Leader intervention tracking
    prev_leader_pos: dict[str, float] | None = None

    # Episode recording state
    episode_start = time.perf_counter()
    frame_index = 0

    logging.info(f"Replay_record started with noise_std={cfg.noise_std}")

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

        # === 4. Leader介入检测（混合模式）===
        if teleop is not None and cfg.leader_adjust_enabled:
            leader_obs = teleop.get_observation()
            leader_pos = {k: v for k, v in leader_obs.items() if k.endswith(".pos")}

            if prev_leader_pos is not None:
                # 计算变化量
                leader_delta = {}
                max_delta = 0.0
                for key, value in leader_pos.items():
                    delta = value - prev_leader_pos.get(key, value)
                    leader_delta[key.replace(".pos", "")] = delta
                    max_delta = max(max_delta, abs(delta))

                # 只有变化量超过阈值才介入
                if max_delta > cfg.leader_threshold:
                    logging.debug(f"Leader介入: max_delta={max_delta:.3f} rad")
                    # 应用修正
                    for joint_key, delta in leader_delta.items():
                        action_key = f"{joint_key}.pos"
                        if action_key in final_action:
                            final_action[action_key] += cfg.leader_alpha * delta

            prev_leader_pos = leader_pos.copy()

        # === 5. 按键介入检测 ===
        if cfg.key_adjust_enabled:
            # 检测按键调整（通过events）
            if events.get("key_adjust_joint_1_up"):
                final_action["left_arm_joint_1.pos"] += cfg.key_adjust_step
                events["key_adjust_joint_1_up"] = False
            if events.get("key_adjust_joint_1_down"):
                final_action["left_arm_joint_1.pos"] -= cfg.key_adjust_step
                events["key_adjust_joint_1_down"] = False
            # 可扩展更多按键...

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


def init_replay_record_keyboard_listener(events: dict) -> Any:
    """Initialize keyboard listener for replay_record mode.

    Keys:
        - S: Mark episode as SUCCESS and save
        - F: Mark episode as FAIL and discard
        - ESC: Stop replay_record
        - Arrow keys: Joint adjustment (optional)
    """
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.KeyCode.from_char('s') or key == keyboard.KeyCode.from_char('S'):
                print("S key pressed. Marking as SUCCESS...")
                events["mark_success"] = True
            elif key == keyboard.KeyCode.from_char('f') or key == keyboard.KeyCode.from_char('F'):
                print("F key pressed. Marking as FAIL...")
                events["mark_fail"] = True
            elif key == keyboard.Key.esc:
                print("ESC key pressed. Stopping replay_record...")
                events["stop_replay_record"] = True
                events["mark_fail"] = True  # Discard current episode
            elif key == keyboard.Key.up:
                events["key_adjust_joint_1_up"] = True
            elif key == keyboard.Key.down:
                events["key_adjust_joint_1_down"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


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
        new_dataset = LeRobotDataset.create(
            repo_id=replay_cfg.record_repo_id,
            fps=fps,
            root=replay_cfg.record_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            tolerance_s=replay_cfg.tolerance_s,
            image_writer_processes=0,
            image_writer_threads=4,
        )

        # 初始化键盘监听
        listener, events = init_keyboard_listener()
        events["mark_success"] = False
        events["mark_fail"] = False
        events["stop_replay_record"] = False

        # 添加 replay_record 专用按键监听
        replay_listener = init_replay_record_keyboard_listener(events)

        log_say("Replay with recording started. Press S=save, F=discard", cfg.play_sounds, blocking=True)

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
