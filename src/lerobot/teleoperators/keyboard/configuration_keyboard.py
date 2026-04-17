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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    # TODO(Steven): Consider setting in here the keys that we want to capture/listen
    mock: bool = False


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("keyboard_ee_bimanual")
@dataclass
class KeyboardBimanualEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """Configuration for bimanual (dual-arm) keyboard end-effector teleoperation."""
    use_gripper: bool = False  # Default false since gripper is currently removed
    # Key mapping configuration (optional customization)
    # Right arm: Arrow keys + Shift
    # Left arm: WASD + Q/E
    right_arm_keys: dict = None  # Will use defaults if None
    left_arm_keys: dict = None   # Will use defaults if None
