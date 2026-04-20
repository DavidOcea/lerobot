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

"""TD3 policy configuration for residual RL."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig


@dataclass
class TD3ActorConfig:
    """Configuration for TD3 actor network."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Action parameters
    action_scale: float = 0.1  # Maximum residual action magnitude
    actor_last_layer_init_scale: float = 0.0  # Zero init for residual (critical!)

    # Weight initialization
    actor_intermediate_layer_init_distribution: str = "orthogonal"
    actor_last_layer_init_distribution: str = "uniform"

    # Feature extraction (if using visual inputs)
    spatial_emb: int = 0  # Spatial embedding dimension (0 = no spatial emb)
    feature_dim: int = 256  # Feature dimension for visual encoder


@dataclass
class TD3CriticConfig:
    """Configuration for TD3 critic network."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Twin Q-networks
    num_q_networks: int = 2


@PreTrainedConfig.register_subclass("td3")
@dataclass
class TD3Config(PreTrainedConfig):
    """Full TD3 configuration."""

    # Network configs
    actor: TD3ActorConfig = field(default_factory=TD3ActorConfig)
    critic: TD3CriticConfig = field(default_factory=TD3CriticConfig)

    # Training hyperparameters
    actor_lr: float = 1e-6  # Conservative for residual RL
    critic_lr: float = 1e-4
    critic_target_tau: float = 0.005  # Soft update rate
    gamma: float = 0.99  # Discount factor

    # TD3 specific hyperparameters
    policy_delay: int = 2  # Delayed policy updates (TD3 feature)
    target_policy_noise_std: float = 0.2  # Target policy smoothing noise
    target_policy_noise_clip: float = 0.5  # Noise clipping

    # Exploration schedule
    stddev_max: float = 0.05
    stddev_min: float = 0.05
    stddev_step: int = 300_000

    # Progressive clipping (optional safety feature)
    progressive_clipping_steps: int = 0  # 0 = disabled

    # Warmup
    actor_lr_warmup_steps: int = 0  # Warmup for actor learning rate

    # Residual specific
    residual_actor: bool = True  # Takes base_action as additional input

    # Device
    device: str = "cuda"

    # Implement abstract methods from PreTrainedConfig
    @property
    def observation_delta_indices(self) -> list | None:
        # TD3 is single-step RL, no temporal observation stacking
        return None

    @property
    def action_delta_indices(self) -> list | None:
        # TD3 is single-step RL, no action chunking
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        # TD3 uses immediate rewards
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        from lerobot.optim.optimizers import AdamConfig
        return AdamConfig(lr=self.actor_lr)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # TD3 is flexible, doesn't require specific feature types
        pass


# Preset configurations for different scenarios


@dataclass
class TD3ConfigConservative(TD3Config):
    """Conservative TD3 for safe exploration."""

    actor: TD3ActorConfig = field(
        default_factory=lambda: TD3ActorConfig(
            action_scale=0.05,  # Smaller residual actions
            actor_last_layer_init_scale=0.0,
        )
    )
    stddev_max: float = 0.02
    stddev_min: float = 0.02


@dataclass
class TD3ConfigAggressive(TD3Config):
    """More aggressive TD3 for fast adaptation."""

    actor: TD3ActorConfig = field(
        default_factory=lambda: TD3ActorConfig(
            action_scale=0.2,  # Larger residual actions
            actor_last_layer_init_scale=0.0,
        )
    )
    stddev_max: float = 0.1
    stddev_min: float = 0.05
    actor_lr: float = 3e-6


@dataclass
class TD3ConfigFineTune(TD3Config):
    """TD3 for fine-tuning a nearly converged policy."""

    actor: TD3ActorConfig = field(
        default_factory=lambda: TD3ActorConfig(
            action_scale=0.1,
            actor_last_layer_init_scale=0.0,
        )
    )
    stddev_max: float = 0.01  # Very low exploration
    stddev_min: float = 0.01
    actor_lr: float = 1e-7  # Very low learning rate
    critic_lr: float = 1e-5