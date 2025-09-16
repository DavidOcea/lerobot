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
Simple script to control a robot from teleoperation using smooth trajectory execution.

This script reads the leader's position and commands the follower to move to that
position over a short duration, resulting in smoother motion compared to direct
position commands.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

# --- 导入 lerobot 的核心组件 ---
# 确保 lerobot 库已安装，并且您的自定义机器人/遥操作器已注册
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    # 确保您的自定义类在此处可被发现
    # 例如，通过在 lerobot/robots/__init__.py 中导入
    supre_robot_follower,
)
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    # 确保您的自定义类在此处可被发现
    supre_robot_leader,
)
from lerobot.utils.utils import init_logging


@dataclass
class TeleoperateTrajectoryConfig:
    """Configuration for teleoperation with trajectory execution."""
    teleop: TeleoperatorConfig
    robot: RobotConfig
    
    # For debugging: stop teleoperation after a certain duration in seconds.
    # If None, runs indefinitely until Ctrl+C.
    trajectory_duration: float =6.0


def teleop_trajectory(
    teleop: Teleoperator, robot: Robot, trajectory_duration: float | None = None
):
    """
    Main loop for teleoperation using execute_trajectory.

    Args:
        teleop: The initialized teleoperator (leader).
        robot: The initialized robot (follower).
        fps: The target frequency for the control loop.
        duration: Optional duration in seconds to run the loop for.
    """
    
    # For pretty printing action values
    display_len = max(len(key) for key in teleop.action_features)
    
    start_time = time.perf_counter()
    logging.info(f"Starting align. Each trajectory segment will be {trajectory_duration:.3f}s.")
    logging.info("Press Ctrl+C to stop.")

    while True:
        loop_start_time = time.perf_counter()

        # 1. Get the target action from the leader device.
        action = teleop.get_action()

        # 2. Command the follower robot to move to the target smoothly.
        # The `execute_trajectory` function is BLOCKING and will take
        # approximately `trajectory_duration` seconds to complete.
        robot.execute_trajectory(goal_action=action, duration=trajectory_duration)

        # 3. Calculate and display loop performance stats.
        loop_s = time.perf_counter() - loop_start_time

        # --- Console Output ---
        print("\n" + "-" * (display_len + 18))
        print(f"{'JOINT NAME':<{display_len}} | {'TARGET VALUE':>12}")
        print("-" * (display_len + 18))
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>12.4f}")
        print("-" * (display_len + 18))
        break

'''
python your_script_name.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower
'''
@draccus.wrap()
def main(cfg: TeleoperateTrajectoryConfig):
    """
    Main function to set up and run the teleoperation.
    """
    init_logging()
    logging.info("--- Teleoperation with Trajectory Execution ---")
    logging.info(pformat(asdict(cfg)))

    # Instantiate leader and follower from configuration
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    try:
        # Establish connection with the hardware
        logging.info("Connecting to teleoperator (leader)...")
        teleop.connect()
        logging.info("Connecting to robot (follower)...")
        robot.connect()
        time.sleep(3.0)
        # Run the main control loop
        logging.info("Starting align...")
        teleop_trajectory(teleop, robot,trajectory_duration=cfg.trajectory_duration)
        logging.info("Align complete.")
    except KeyboardInterrupt:
        logging.info("\nKeyboard interrupt detected. Shutting down.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Ensure hardware is safely disconnected
        logging.info("Disconnecting from all devices...")
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        logging.info("Shutdown complete.")
    time.sleep(5.0)


if __name__ == "__main__":
    main()