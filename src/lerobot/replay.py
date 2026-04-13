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

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
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
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


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


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.hf_dataset.select_columns("action")
    fps = dataset.fps if hasattr(dataset, 'fps') else cfg.dataset.fps
    robot.connect()

    loop_count = 0
    max_loops = cfg.dataset.num_loops if cfg.dataset.num_loops > 0 else float('inf')

    log_say("Replaying episode", cfg.play_sounds, blocking=True)

    while loop_count < max_loops:
        start_episode_t = time.perf_counter()  # 只在episode开始时设置一次

        for idx in range(dataset.num_frames):
            action_array = actions[idx]["action"]
            action = {}
            for i, name in enumerate(dataset.features["action"]["names"]):
                action[name] = action_array[i]

            robot.send_action(action)

            dt_s = time.perf_counter() - start_episode_t
            # 计算期望时间：当前帧索引对应的理想时间
            expected_time = idx / fps
            wait_time = expected_time - dt_s
            if wait_time > 0:
                busy_wait(wait_time)

        loop_count += 1
        logging.info(f"Replay loop {loop_count} completed")

    robot.disconnect()


if __name__ == "__main__":
    replay()
