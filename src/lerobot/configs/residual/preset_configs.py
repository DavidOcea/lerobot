# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Preset configurations for residual RL training.

These configurations are tuned for specific tasks and scenarios,
similar to ResFiT's preset configs.
"""

from dataclasses import dataclass, field

from lerobot.policies.act.config_act import ACTConfig
from lerobot.policies.residual.config_residual import ResidualPolicyConfig, ResidualTrainingConfig
from lerobot.policies.td3.config_td3 import TD3ActorConfig, TD3CriticConfig, TD3Config


@dataclass
class ResidualCoffeeConfig(ResidualTrainingConfig):
    """Configuration for TwoArmCoffee task."""

    policy: ResidualPolicyConfig = field(
        default_factory=lambda: ResidualPolicyConfig(
            base_policy=ACTConfig(),  # Would be overridden by loaded checkpoint
            residual_policy=TD3Config(
                actor=TD3ActorConfig(
                    action_scale=0.1,
                    actor_last_layer_init_scale=0.0,
                ),
                stddev_max=0.05,
                stddev_min=0.05,
            ),
            action_scale=0.1,
        )
    )

    total_timesteps: int = 500_000
    warmup_steps: int = 10_000
    learning_starts: int = 10_000
    buffer_size: int = 300_000

    output_dir: str = "outputs/residual/coffee"


@dataclass
class ResidualBoxCleanupConfig(ResidualTrainingConfig):
    """Configuration for TwoArmBoxCleanup task."""

    policy: ResidualPolicyConfig = field(
        default_factory=lambda: ResidualPolicyConfig(
            base_policy=ACTConfig(),
            residual_policy=TD3Config(
                actor=TD3ActorConfig(
                    action_scale=0.15,
                    actor_last_layer_init_scale=0.0,
                ),
                stddev_max=0.05,
                stddev_min=0.05,
            ),
            action_scale=0.15,
        )
    )

    total_timesteps: int = 500_000
    warmup_steps: int = 10_000
    buffer_size: int = 300_000

    output_dir: str = "outputs/residual/box_cleanup"


@dataclass
class ResidualCanSortConfig(ResidualTrainingConfig):
    """Configuration for TwoArmCanSort task."""

    policy: ResidualPolicyConfig = field(
        default_factory=lambda: ResidualPolicyConfig(
            base_policy=ACTConfig(),
            residual_policy=TD3Config(
                actor=TD3ActorConfig(
                    action_scale=0.1,
                    actor_last_layer_init_scale=0.0,
                ),
                stddev_max=0.05,
                stddev_min=0.05,
            ),
            action_scale=0.1,
        )
    )

    total_timesteps: int = 500_000
    buffer_size: int = 300_000

    output_dir: str = "outputs/residual/can_sort"


@dataclass
class ResidualSafeConfig(ResidualTrainingConfig):
    """
    Safe configuration for real robot training.

    Conservative action scale and exploration for safe deployment.
    """

    policy: ResidualPolicyConfig = field(
        default_factory=lambda: ResidualPolicyConfig(
            base_policy=ACTConfig(),
            residual_policy=TD3Config(
                actor=TD3ActorConfig(
                    action_scale=0.05,  # Very small residual
                    actor_last_layer_init_scale=0.0,
                ),
                stddev_max=0.02,  # Low exploration
                stddev_min=0.01,
                progressive_clipping_enabled=True,  # Progressive clipping
            ),
            action_scale=0.05,
            progressive_clipping_enabled=True,
            progressive_clipping_steps=50_000,
        )
    )

    total_timesteps: int = 200_000
    warmup_steps: int = 5_000
    learning_starts: int = 5_000
    buffer_size: int = 50_000

    output_dir: str = "outputs/residual/safe"


@dataclass
class ResidualFineTuneConfig(ResidualTrainingConfig):
    """
    Configuration for fine-tuning a nearly converged policy.

    Very conservative settings for final polish.
    """

    policy: ResidualPolicyConfig = field(
        default_factory=lambda: ResidualPolicyConfig(
            base_policy=ACTConfig(),
            residual_policy=TD3Config(
                actor=TD3ActorConfig(
                    action_scale=0.05,
                    actor_last_layer_init_scale=0.0,
                ),
                actor_lr=1e-7,  # Very low learning rate
                critic_lr=1e-5,
                stddev_max=0.01,  # Minimal exploration
                stddev_min=0.01,
            ),
            action_scale=0.05,
        )
    )

    total_timesteps: int = 50_000
    warmup_steps: int = 1_000
    buffer_size: int = 20_000

    output_dir: str = "outputs/residual/finetune"


def get_preset_config(task_name: str) -> ResidualTrainingConfig:
    """
    Get preset configuration for a task.

    Args:
        task_name: Name of the task (e.g., "coffee", "box_cleanup", "can_sort")

    Returns:
        Preset configuration for the task
    """
    presets = {
        "coffee": ResidualCoffeeConfig,
        "box_cleanup": ResidualBoxCleanupConfig,
        "can_sort": ResidualCanSortConfig,
        "safe": ResidualSafeConfig,
        "finetune": ResidualFineTuneConfig,
    }

    if task_name not in presets:
        logging.warning(f"No preset for task '{task_name}', using default")
        return ResidualTrainingConfig()

    return presets[task_name]()