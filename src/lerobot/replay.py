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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    supre_robot_follower
)
# ROS2 follower is optional
try:
    from lerobot.robots import ros2_follower  # noqa: F401
except ImportError:
    pass
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    supre_robot_leader,  # noqa: F401
)
# ROS2 leader is optional
try:
    from lerobot.teleoperators import ros2_leader  # noqa: F401
except ImportError:
    pass
from lerobot.utils.action_corrector import (
    ActionCorrector,
    ActionCorrectorConfig,
    LeaderAdjustConfig,
    KeyAdjustConfig,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
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

    # Leader intervention parameters (状态机设计，滞回机制)
    leader_adjust: LeaderAdjustConfig = LeaderAdjustConfig()

    # Keyboard adjustment parameters（平滑精细控制）
    key_adjust: KeyAdjustConfig = KeyAdjustConfig()

    # Timestamp mode: False = ideal timestamp (frame_index/fps, default), True = actual timestamp (perf_counter)
    use_actual_timestamp: bool = False

    # Timestamp tolerance for recorded dataset (auto-set based on timestamp mode if None)
    tolerance_s: float | None = None  # None = auto: 0.03 if actual, 1e-4 if ideal

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
    use_actual_timestamp: bool = False,  # False=ideal timestamp (default), True=actual timestamp
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
        use_actual_timestamp: Timestamp mode. False=ideal (frame_index/fps), True=actual (perf_counter).

    Returns:
        True if episode was saved successfully, False if discarded.
    """
    rng = np.random.default_rng(cfg.noise_seed)

    # 创建 ActionCorrector 实例（使用独立模块）
    corrector_config = ActionCorrectorConfig(
        enable=True,
        leader=cfg.leader_adjust,
        keyboard=cfg.key_adjust,
    )
    corrector = ActionCorrector(corrector_config, teleop=teleop)

    # Episode recording state
    episode_start = time.perf_counter()
    frame_index = 0

    timestamp_mode = "actual" if use_actual_timestamp else "ideal"
    logging.info(f"Replay_record started with noise_std={cfg.noise_std}, timestamp_mode={timestamp_mode}")
    logging.info(f"Key adjust: joint={cfg.key_adjust.default_joint}, arm_mode={cfg.key_adjust.default_arm_mode}, step={cfg.key_adjust.step_per_frame} deg/frame")
    if cfg.leader_adjust.enable:
        logging.info(f"Leader adjust: trigger={cfg.leader_adjust.trigger_threshold_deg}度, exit={cfg.leader_adjust.exit_threshold_deg}度/{cfg.leader_adjust.exit_frame_count}帧")

    for idx in range(dataset.num_frames):
        loop_start = time.perf_counter()

        # === 1. 根据配置计算时间戳 ===
        if use_actual_timestamp:
            actual_timestamp = time.perf_counter() - episode_start  # 真实时间戳
            # 限制 timestamp 不超出当前帧数对应的时长，防止超出视频范围
            max_timestamp = frame_index / fps
            actual_timestamp = min(actual_timestamp, max_timestamp)
        else:
            actual_timestamp = frame_index / fps  # 理想时间戳（默认）

        # === 2. 获取原始动作 ===
        action_array = actions[idx]["action"]
        base_action = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            base_action[name] = float(action_array[i])

        # === 3. 添加扰动 ===
        noisy_action = add_noise_to_action(base_action, cfg.noise_std, rng)

        # === 4. 使用 ActionCorrector 应用修正 ===
        corrected_action = corrector.correct(
            action=noisy_action,
            events=events,
        )
        final_action = corrected_action

        # === 5. 获取observation ===
        observation = robot.get_observation()

        # === 6. 执行动作 ===
        sent_action = robot.send_action(final_action)

        # === 6.1 力反馈：检查阻力并发送给 Leader ===
        if teleop is not None and isinstance(teleop, Teleoperator):
            if hasattr(robot, 'get_force_feedback') and hasattr(teleop, 'send_feedback'):
                force_feedback = robot.get_force_feedback()
                teleop.send_feedback(force_feedback)

        # === 7. 录制数据 ===
        if new_dataset is not None:
            observation_frame = build_dataset_frame(new_dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(new_dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            new_dataset.add_frame(frame, task=single_task, timestamp=actual_timestamp)

        frame_index += 1

        # === 8. 检测成功/失败按键 ===
        if events.get("mark_fail"):
            logging.info("Episode marked as FAIL, discarding...")
            new_dataset.clear_episode_buffer()
            events["mark_fail"] = False
            corrector.reset()  # 重置修正器
            return False

        if events.get("exit_early"):
            events["exit_early"] = False
            break

        # === 9. 时间同步 ===
        dt_s = time.perf_counter() - loop_start
        expected_interval = 1.0 / fps
        wait_time = expected_interval - dt_s
        if wait_time > 0:
            busy_wait(wait_time)

    # === 10. Episode结束后等待成功按键 ===
    logging.info("Episode replay completed. Press 'S' to save, 'F' to discard...")

    wait_start = time.perf_counter()
    while time.perf_counter() - wait_start < cfg.success_key_timeout:
        if events.get("mark_success"):
            logging.info("Episode marked as SUCCESS, saving...")
            new_dataset.save_episode()
            events["mark_success"] = False
            corrector.reset()  # 重置修正器
            return True
        if events.get("mark_fail"):
            logging.info("Episode marked as FAIL, discarding...")
            new_dataset.clear_episode_buffer()
            events["mark_fail"] = False
            corrector.reset()  # 重置修正器
            return False
        time.sleep(0.1)

    # 超时未按键，默认保存
    logging.info("Timeout, auto-saving episode...")
    new_dataset.save_episode()
    corrector.reset()  # 重置修正器
    return True


def init_replay_record_keyboard_listener(events: dict, key_cfg: KeyAdjustConfig) -> Any:
    """Initialize keyboard listener for replay_record mode.

    关节选择模式 + 动态反转：
    - S: 标记成功，保存
    - F: 标记失败，丢弃
    - ESC: 停止 replay_record
    - 1-6: 选择 joint_1 ~ joint_6
    - 8-9: 选择 trunk_joint_1, trunk_joint_2
    - A: 正向调整（按住持续）
    - D: 负向调整（按住持续）
    - J: 左臂模式
    - K: 双臂模式
    - L: 右臂模式
    - U: 双臂模式下反转左臂当前关节
    - O: 双臂模式下反转右臂当前关节
    - R: 重置选择状态和反转状态
    """
    # 检查是否在 headless 环境
    if is_headless():
        logging.warning(
            "Headless environment detected. Keyboard inputs for replay_record mode will not be available. "
            "You can still use Leader intervention, but keyboard fine-adjustment will be disabled."
        )
        return None

    from pynput import keyboard

    def get_char(key) -> str | None:
        """获取按键字符"""
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            return None
        except:
            return None

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

            # === 关节选择（数字键）===
            elif char in ['1', '2', '3', '4', '5', '6']:
                joint_num = int(char)
                events[f"joint_{joint_num}_selected"] = True
            elif char == '8':
                events["joint_8_selected"] = True
            elif char == '9':
                events["joint_9_selected"] = True

            # === 臂模式选择 ===
            elif char == 'j':
                events["arm_mode_left"] = True
            elif char == 'k':
                events["arm_mode_both"] = True
            elif char == 'l':
                events["arm_mode_right"] = True

            # === 反转控制 ===
            elif char == 'u':
                events["inverse_left"] = True
            elif char == 'o':
                events["inverse_right"] = True

            # === 状态重置 ===
            elif char == 'r':
                events["reset_state"] = True

            # === 调整按键（按住持续）===
            elif char == 'a':
                events["positive_held"] = True
                logging.debug("[Key] A pressed: positive adjustment")
            elif char == 'd':
                events["negative_held"] = True
                logging.debug("[Key] D pressed: negative adjustment")

        except Exception as e:
            print(f"Error handling key press: {e}")

    def on_release(key):
        """按键释放"""
        try:
            char = get_char(key)

            # === 关节选择释放（单次触发，立即清除）===
            if char in ['1', '2', '3', '4', '5', '6']:
                joint_num = int(char)
                events[f"joint_{joint_num}_selected"] = False
            elif char == '8':
                events["joint_8_selected"] = False
            elif char == '9':
                events["joint_9_selected"] = False

            # === 臂模式选择释放 ===
            elif char == 'j':
                events["arm_mode_left"] = False
            elif char == 'k':
                events["arm_mode_both"] = False
            elif char == 'l':
                events["arm_mode_right"] = False

            # === 反转控制释放 ===
            elif char == 'u':
                events["inverse_left"] = False
            elif char == 'o':
                events["inverse_right"] = False

            # === 状态重置释放 ===
            elif char == 'r':
                events["reset_state"] = False

            # === 调整按键释放 ===
            elif char == 'a':
                events["positive_held"] = False
                logging.debug("[Key] A released: positive adjustment stopped")
            elif char == 'd':
                events["negative_held"] = False
                logging.debug("[Key] D released: negative adjustment stopped")

        except Exception as e:
            print(f"Error handling key release: {e}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 打印帮助信息
    print("=" * 60)
    print("键盘控制帮助（关节选择模式）:")
    print("  关节选择: 1-6 (arm joints), 8-9 (trunk)")
    print("  调整: A (正向+), D (负向-)")
    print("  臂模式: J (左臂), K (双臂), L (右臂)")
    print("  反转: U (左臂), O (右臂) - 双臂模式下")
    print("  重置: R (重置选择和反转状态)")
    print("  成功/失败: S/F")
    print("  退出: ESC")
    print("=" * 60)

    return listener


@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # 加载源数据集：使用宽松的 tolerance_s 以兼容 perf_counter 录制的数据
    # 源数据可能使用 tolerance_s=0.03 录制，加载时需要匹配
    source_tolerance_s = cfg.dataset.replay_record.tolerance_s if cfg.dataset.replay_record.enable else 1e-4
    if source_tolerance_s is None:
        source_tolerance_s = 0.03  # 默认使用宽松容差，兼容 perf_counter 录制的数据
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.episode],
        tolerance_s=source_tolerance_s,  # 使用与新录制一致的容差
    )
    actions = dataset.hf_dataset.select_columns("action")
    fps = dataset.fps if hasattr(dataset, 'fps') else cfg.dataset.fps

    logging.info(f"Source dataset loaded with tolerance_s={source_tolerance_s}")

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

        # 创建新数据集：使用当前 robot 的实际 features（而非源数据集的 features）
        # 这样可以兼容：当前环境没有摄像头但源数据集有图像的情况
        action_features = hw_to_dataset_features(robot.action_features, "action", True)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
        dataset_features = {**action_features, **obs_features}

        # 相机配置：根据相机数量动态计算（与 record.py 一致）
        num_cameras = len(robot.cameras) if hasattr(robot, 'cameras') and robot.cameras else 0
        image_writer_threads = 4 * num_cameras  # 每个相机4个线程（无相机则为0）

        # 根据时间戳模式自动设置容差
        if replay_cfg.tolerance_s is None:
            tolerance_s = 0.03 if replay_cfg.use_actual_timestamp else 1e-4
        else:
            tolerance_s = replay_cfg.tolerance_s

        new_dataset = LeRobotDataset.create(
            repo_id=replay_cfg.record_repo_id,
            fps=fps,
            root=replay_cfg.record_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            tolerance_s=tolerance_s,
            image_writer_processes=0,
            image_writer_threads=image_writer_threads,
        )

        logging.info(f"Dataset created with {num_cameras} cameras, {image_writer_threads} image writer threads, tolerance_s={tolerance_s}")

        # 初始化键盘监听
        listener, events = init_keyboard_listener()
        events["mark_success"] = False
        events["mark_fail"] = False
        events["stop_replay_record"] = False

        # 初始化关节选择模式按键事件
        # 关节选择
        for i in [1, 2, 3, 4, 5, 6, 8, 9]:
            events[f"joint_{i}_selected"] = False
        # 臂模式选择
        events["arm_mode_left"] = False
        events["arm_mode_both"] = False
        events["arm_mode_right"] = False
        # 反转控制
        events["inverse_left"] = False
        events["inverse_right"] = False
        # 状态重置
        events["reset_state"] = False
        # 调整按键
        events["positive_held"] = False
        events["negative_held"] = False

        # 添加 replay_record 专用按键监听（关节选择模式）
        replay_listener = init_replay_record_keyboard_listener(events, replay_cfg.key_adjust)

        log_say("Replay with recording started. S=save, F=discard", cfg.play_sounds, blocking=True)
        logging.info("Key controls: 1-6/8-9=joint select, A/D=adjust, J/K/L=arm mode, U/O=inverse, R=reset")

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
                use_actual_timestamp=replay_cfg.use_actual_timestamp,
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
