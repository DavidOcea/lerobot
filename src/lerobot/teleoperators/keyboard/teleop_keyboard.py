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

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig, KeyboardBimanualEndEffectorTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict


class KeyboardBimanualEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class for bimanual (dual-arm) keyboard end-effector control.

    Key mapping:
    - Right arm: Arrow keys (X/Y) + Shift/Shift_R (Z)
    - Left arm: WASD (X/Y) + Q/E (Z)

    Output action: [delta_x_right, delta_y_right, delta_z_right, delta_x_left, delta_y_left, delta_z_left]
    With gripper: + gripper_right, gripper_left (8 dimensions total)
    """

    config_class = KeyboardBimanualEndEffectorTeleopConfig
    name = "keyboard_ee_bimanual"

    def __init__(self, config: KeyboardBimanualEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

        # Default key mappings
        self.right_arm_keys = config.right_arm_keys or {
            "up": keyboard.Key.up,        # delta_y negative (forward)
            "down": keyboard.Key.down,    # delta_y positive (backward)
            "left": keyboard.Key.left,    # delta_x positive
            "right": keyboard.Key.right,  # delta_x negative
            "z_up": keyboard.Key.shift_r, # delta_z positive (up)
            "z_down": keyboard.Key.shift, # delta_z negative (down)
        }

        self.left_arm_keys = config.left_arm_keys or {
            "up": 'w',        # delta_y negative (forward)
            "down": 's',      # delta_y positive (backward)
            "left": 'd',      # delta_x positive
            "right": 'a',     # delta_x negative
            "z_up": 'e',      # delta_z positive (up)
            "z_down": 'q',    # delta_z negative (down)
        }

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (8,),
                "names": {
                    "delta_x_right": 0, "delta_y_right": 1, "delta_z_right": 2,
                    "delta_x_left": 3, "delta_y_left": 4, "delta_z_left": 5,
                    "gripper_right": 6, "gripper_left": 7,
                },
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {
                    "delta_x_right": 0, "delta_y_right": 1, "delta_z_right": 2,
                    "delta_x_left": 3, "delta_y_left": 4, "delta_z_left": 5,
                },
            }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardBimanualEndEffectorTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Right arm deltas
        delta_x_right = 0.0
        delta_y_right = 0.0
        delta_z_right = 0.0

        # Left arm deltas
        delta_x_left = 0.0
        delta_y_left = 0.0
        delta_z_left = 0.0

        # Gripper actions (1.0 = stay, 0 = close, 2 = open)
        gripper_right = 1.0
        gripper_left = 1.0

        # Process right arm keys (Arrow keys + Shift)
        for key, val in self.current_pressed.items():
            if key == self.right_arm_keys["up"]:
                delta_y_right = -int(val)
            elif key == self.right_arm_keys["down"]:
                delta_y_right = int(val)
            elif key == self.right_arm_keys["left"]:
                delta_x_right = int(val)
            elif key == self.right_arm_keys["right"]:
                delta_x_right = -int(val)
            elif key == self.right_arm_keys["z_up"]:
                delta_z_right = int(val)
            elif key == self.right_arm_keys["z_down"]:
                delta_z_right = -int(val)

        # Process left arm keys (WASD + Q/E)
        for key, val in self.current_pressed.items():
            if key == self.left_arm_keys["up"]:  # 'w'
                delta_y_left = -int(val)
            elif key == self.left_arm_keys["down"]:  # 's'
                delta_y_left = int(val)
            elif key == self.left_arm_keys["left"]:  # 'd'
                delta_x_left = int(val)
            elif key == self.left_arm_keys["right"]:  # 'a'
                delta_x_left = -int(val)
            elif key == self.left_arm_keys["z_up"]:  # 'e'
                delta_z_left = int(val)
            elif key == self.left_arm_keys["z_down"]:  # 'q'
                delta_z_left = -int(val)

        # Process misc keys (success/failure/intervention)
        misc_keys_to_process = []
        for key, val in list(self.current_pressed.items()):
            # Keys not part of arm control
            is_arm_key = False
            for arm_keys in [self.right_arm_keys, self.left_arm_keys]:
                if key in arm_keys.values():
                    is_arm_key = True
                    break
            if not is_arm_key and val:
                misc_keys_to_process.append(key)

        for key in misc_keys_to_process:
            self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x_right": delta_x_right,
            "delta_y_right": delta_y_right,
            "delta_z_right": delta_z_right,
            "delta_x_left": delta_x_left,
            "delta_y_left": delta_y_left,
            "delta_z_left": delta_z_left,
        }

        if self.config.use_gripper:
            action_dict["gripper_right"] = gripper_right
            action_dict["gripper_left"] = gripper_left

        return action_dict
