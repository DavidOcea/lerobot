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
TD3 policy implementation for residual RL.

TD3 (Twin Delayed Deep Deterministic Policy Gradient) improvements over DDPG:
1. Twin Q-networks to reduce overestimation bias
2. Delayed policy updates
3. Target policy smoothing

Adapted from ResFiT implementation with modifications for LeRobot integration.
"""

import copy
import logging
from typing import Any

import torch
from torch import nn, optim

from lerobot.policies.td3.config_td3 import TD3ActorConfig, TD3CriticConfig, TD3Config
from lerobot.utils.residual.utils import (
    EvalMode,
    TruncatedNormal,
    init_last_layer_zero,
    orthogonal_weight_init,
    soft_update,
)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    use_layer_norm: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build MLP network.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout rate

    Returns:
        Sequential MLP network
    """
    layers = []

    # First layer: input -> hidden
    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.ReLU())

    # Hidden layers
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    # Output layer (no activation for continuous actions)
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class TD3Actor(nn.Module):
    """
    TD3 Actor network for residual RL.

    Key features:
    1. Takes base_action as additional input (for residual RL)
    2. Outputs residual action in [-action_scale, action_scale]
    3. Last layer initialized to zero (no initial residual)
    4. Supports visual features via optional encoder input
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TD3ActorConfig,
        residual_actor: bool = True,
    ):
        """
        Initialize Actor network.

        Args:
            state_dim: State observation dimension
            action_dim: Action dimension
            config: Actor configuration
            residual_actor: Whether this is a residual actor (takes base_action)
        """
        super().__init__()

        self.config = config
        self.residual_actor = residual_actor
        self.action_dim = action_dim

        # Input dimension: state + base_action (if residual)
        input_dim = state_dim
        if residual_actor:
            input_dim += action_dim

        # Build MLP
        self.policy = build_mlp(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=action_dim,
            num_layers=config.num_layers,
            use_layer_norm=config.use_layer_norm,
            dropout=config.dropout,
        )

        # Initialize weights
        self._initialize_weights()

        logging.info(f"TD3Actor initialized: state_dim={state_dim}, action_dim={action_dim}, residual={residual_actor}")

    def _initialize_weights(self) -> None:
        """Apply weight initialization."""
        # Orthogonal init for intermediate layers
        self.policy.apply(orthogonal_weight_init)

        # Zero init for last layer (critical for residual RL)
        if self.config.actor_last_layer_init_scale == 0.0:
            init_last_layer_zero(self.policy)

    def forward(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State observation tensor (batch_size, state_dim)
            base_action: Base policy action (batch_size, action_dim), required if residual_actor=True

        Returns:
            Raw action output (batch_size, action_dim), in [-1, 1] before scaling
        """
        # Build input
        if self.residual_actor:
            if base_action is None:
                raise ValueError("base_action is required for residual_actor=True")
            input_tensor = torch.cat([state, base_action], dim=-1)
        else:
            input_tensor = state

        # Forward through MLP (output in [-1, 1] due to no activation on last layer)
        # Note: The MLP output is raw, we apply Tanh + scaling in act()
        raw_output = self.policy(input_tensor)

        return raw_output

    def act(
        self,
        state: torch.Tensor,
        base_action: torch.Tensor | None = None,
        eval_mode: bool = True,
        stddev: float = 0.05,
    ) -> torch.Tensor:
        """
        Generate action for environment.

        Args:
            state: State observation
            base_action: Base policy action (for residual actor)
            eval_mode: If True, return deterministic action; if False, add noise
            stddev: Exploration noise standard deviation

        Returns:
            Action tensor (scaled by action_scale)
        """
        # Forward pass
        raw_output = self.forward(state, base_action)

        # Apply Tanh to bound to [-1, 1]
        bounded_output = torch.tanh(raw_output)

        # Scale by action_scale
        action = bounded_output * self.config.action_scale

        # Add exploration noise if not eval mode
        if not eval_mode:
            noise = torch.randn_like(action) * stddev
            action = action + noise
            # Clip to action_scale bounds
            action = torch.clamp(action, -self.config.action_scale, self.config.action_scale)

        return action


class TD3Critic(nn.Module):
    """
    TD3 Twin Critic network.

    Uses two Q-networks to reduce overestimation bias.
    The minimum of Q1 and Q2 is used for target computation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TD3CriticConfig,
    ):
        """
        Initialize Critic network.

        Args:
            state_dim: State observation dimension
            action_dim: Action dimension
            config: Critic configuration
        """
        super().__init__()

        self.config = config

        # Build twin Q-networks
        self.q1 = build_mlp(
            input_dim=state_dim + action_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.num_layers,
            use_layer_norm=config.use_layer_norm,
            dropout=config.dropout,
        )

        self.q2 = build_mlp(
            input_dim=state_dim + action_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.num_layers,
            use_layer_norm=config.use_layer_norm,
            dropout=config.dropout,
        )

        # Initialize weights
        self.q1.apply(orthogonal_weight_init)
        self.q2.apply(orthogonal_weight_init)

        logging.info(f"TD3Critic initialized: twin Q-networks, state_dim={state_dim}, action_dim={action_dim}")

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Args:
            state: State observation (batch_size, state_dim)
            action: Action (batch_size, action_dim)

        Returns:
            Tuple of (Q1, Q2) values
        """
        # Concatenate state and action
        input_tensor = torch.cat([state, action], dim=-1)

        # Compute Q values
        q1 = self.q1(input_tensor)
        q2 = self.q2(input_tensor)

        return q1, q2

    def q_min(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute minimum of Q1 and Q2.

        This is used for target computation in TD3.

        Args:
            state: State observation
            action: Action

        Returns:
            Minimum Q value
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class TD3Agent:
    """
    TD3 Agent combining Actor and Critic for training.

    Handles:
    - Training updates (actor and critic)
    - Target network soft updates
    - Action generation for environment interaction
    """

    def __init__(
        self,
        actor: TD3Actor,
        critic: TD3Critic,
        config: TD3Config,
        device: torch.device,
    ):
        """
        Initialize TD3 Agent.

        Args:
            actor: Actor network
            critic: Critic network
            config: TD3 configuration
            device: Device for computation
        """
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.config = config
        self.device = device

        # Target networks
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Training counters
        self.actor_updates = 0
        self.critic_updates = 0

        logging.info(f"TD3Agent initialized on device: {device}")

    def update(
        self,
        batch: dict[str, torch.Tensor],
        stddev: float,
        update_actor: bool = False,
    ) -> dict[str, float]:
        """
        Perform one training update.

        Args:
            batch: Batch of transitions from replay buffer
            stddev: Current exploration stddev (for target policy smoothing)
            update_actor: Whether to update actor this step

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        # Extract batch data
        state = batch["state"]
        action = batch["action"]  # This is residual action in normalized space
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]
        weights = batch.get("weights", None)  # For prioritized replay
        base_action = batch.get("base_action", None)  # For residual RL

        # Critic update
        critic_metrics = self._update_critic(
            state, action, reward, next_state, done, stddev, weights, base_action
        )
        metrics.update(critic_metrics)

        # Actor update (delayed)
        if update_actor:
            actor_metrics = self._update_actor(state, action, stddev, base_action)
            metrics.update(actor_metrics)

            # Soft update target networks
            self._soft_update_targets()

        return metrics

    def _update_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        stddev: float,
        weights: torch.Tensor | None = None,
        base_action: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Update critic networks."""
        self.critic_optimizer.zero_grad()

        # Compute target Q value
        with torch.no_grad():
            # Target policy action (for residual actor, need base_action)
            target_action = self.actor_target.act(
                next_state,
                base_action=base_action,  # Use base_action from batch for target
                eval_mode=True,
            )

            # Target policy smoothing noise
            noise = torch.randn_like(target_action) * self.config.target_policy_noise_std
            noise = torch.clamp(noise, -self.config.target_policy_noise_clip, self.config.target_policy_noise_clip)
            target_action = target_action + noise
            target_action = torch.clamp(
                target_action,
                -self.config.actor.action_scale,
                self.config.actor.action_scale,
            )

            # Target Q value (minimum of twin Q-networks)
            target_q = self.critic_target.q_min(next_state, target_action)
            # Compute target with discount (handle bool done tensor)
            not_done = (~done).float() if done.dtype == torch.bool else (1.0 - done)
            target_q = reward + not_done * self.config.gamma * target_q

        # Compute current Q values
        q1, q2 = self.critic(state, action)

        # Critic loss (MSE)
        loss1 = nn.functional.mse_loss(q1, target_q, reduction='none')
        loss2 = nn.functional.mse_loss(q2, target_q, reduction='none')

        # Apply importance weights if using prioritized replay
        if weights is not None:
            loss1 = (loss1 * weights).mean()
            loss2 = (loss2 * weights).mean()
        else:
            loss1 = loss1.mean()
            loss2 = loss2.mean()

        critic_loss = loss1 + loss2

        # Backward pass
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_updates += 1

        return {
            "train/critic_loss": critic_loss.item(),
            "train/q1_mean": q1.mean().item(),
            "train/q2_mean": q2.mean().item(),
            "train/q1_std": q1.std().item(),
            "train/q2_std": q2.std().item(),
        }

    def _update_actor(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        stddev: float,
        base_action: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Update actor network."""
        self.actor_optimizer.zero_grad()

        # Actor action (for maximizing Q)
        actor_action = self.actor.act(
            state,
            base_action=base_action,  # Use base_action for residual actor
            eval_mode=True,
        )

        # Actor loss: maximize Q (minimize -Q)
        # Use Q1 for actor update (TD3 convention)
        q1 = self.critic.q1(torch.cat([state, actor_action], dim=-1))
        actor_loss = -q1.mean()

        # Backward pass
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_updates += 1

        return {
            "train/actor_loss": actor_loss.item(),
            "train/actor_action_mean": actor_action.mean().item(),
            "train/actor_action_std": actor_action.std().item(),
        }

    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        soft_update(self.actor_target, self.actor, self.config.critic_target_tau)
        soft_update(self.critic_target, self.critic, self.config.critic_target_tau)

    def act(
        self,
        obs: dict[str, torch.Tensor],
        eval_mode: bool = False,
        stddev: float = 0.05,
    ) -> torch.Tensor:
        """
        Generate action for environment.

        Args:
            obs: Observation dictionary (contains state and base_action for residual)
            eval_mode: If True, deterministic; if False, stochastic
            stddev: Exploration noise std

        Returns:
            Residual action tensor
        """
        state = obs["observation.state"]
        base_action = obs.get("observation.base_action", None)

        return self.actor.act(state, base_action, eval_mode, stddev)

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_updates": self.actor_updates,
            "critic_updates": self.critic_updates,
        }, path)
        logging.info(f"TD3Agent saved to: {path}")

    def load(self, path: str) -> None:
        """Load agent state."""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.actor_updates = state["actor_updates"]
        self.critic_updates = state["critic_updates"]
        logging.info(f"TD3Agent loaded from: {path}")

    def eval_mode(self) -> EvalMode:
        """Get evaluation mode context manager."""
        return EvalMode(self)