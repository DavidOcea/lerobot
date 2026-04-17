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

"""Residual policy configuration."""

from dataclasses import dataclass

from lerobot.policies.act.config_act import ACTConfig
from lerobot.policies.td3.config_td3 import TD3Config


@dataclass
class ResidualPolicyConfig:
    """Configuration for residual policy (ACT + TD3)."""

    # Base policy
    base_policy: ACTConfig

    # Residual policy
    residual_policy: TD3Config

    # Combination parameters
    action_scale: float = 0.1  # Residual action magnitude limit

    # Freeze base policy during residual training
    freeze_base_policy: bool = True

    # Progressive clipping for safe exploration
    progressive_clipping_enabled: bool = False
    progressive_clipping_steps: int = 100_000  # Steps to reach full action_scale


@dataclass
class ResidualTrainingConfig:
    """Full training configuration for residual RL."""

    # Policy configs
    policy: ResidualPolicyConfig

    # Training hyperparameters
    total_timesteps: int = 500_000
    warmup_steps: int = 10_000  # Random exploration warmup
    learning_starts: int = 10_000  # When to start learning
    critic_warmup_steps: int = 10_000  # Critic-only updates

    # Batch and buffer
    batch_size: int = 256
    buffer_size: int = 300_000
    offline_fraction: float = 0.5  # Fraction of offline data in batch

    # N-step returns
    n_step: int = 1
    gamma: float = 0.99

    # Update frequency
    update_every_n_steps: int = 1
    num_updates_per_iteration: int = 4
    actor_update_freq: int = 2  # TD3 delayed policy update

    # Evaluation
    eval_interval: int = 10_000
    eval_episodes: int = 10

    # Logging
    log_freq: int = 100

    # Checkpoint
    checkpoint_interval: int = 10_000
    max_checkpoints: int = 5

    # Output
    output_dir: str = "outputs/residual"
    run_name: str = "run_001"

    # Resume
    resume_checkpoint: str | None = None