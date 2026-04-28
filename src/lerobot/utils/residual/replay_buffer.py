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
Replay buffer for residual RL training.

Features:
- Efficient circular buffer storage
- Support for n-step returns
- Mixed online/offline sampling
- Prioritized experience replay (optional)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ReplayBufferConfig:
    """Configuration for ReplayBuffer."""

    buffer_size: int = 300_000  # Maximum buffer size
    n_step: int = 1  # N-step return horizon
    gamma: float = 0.99  # Discount factor

    # Sampling strategy
    sampling_strategy: str = "uniform"  # "uniform" or "prioritized"

    # Prioritized replay parameters (if sampling_strategy == "prioritized")
    priority_alpha: float = 0.6  # Priority exponent
    priority_beta: float = 0.4  # Importance sampling exponent
    priority_beta_increment: float = 0.001  # Beta increment per sample
    priority_epsilon: float = 1e-6  # Minimum priority


class ReplayBuffer:
    """
    Efficient replay buffer for off-policy RL.

    Supports:
    - Circular buffer for efficient memory usage
    - N-step returns for better credit assignment
    - Prioritized experience replay for focusing on high TD error transitions
    - Mixed online/offline sampling

    Example:
        buffer = ReplayBuffer(
            buffer_size=100_000,
            state_dim=7,
            action_dim=7,
            n_step=5,
            gamma=0.995,
        )

        # Add transition
        buffer.add(state, action, reward, next_state, done)

        # Sample batch
        batch = buffer.sample(batch_size=256)

        # Update priorities (for prioritized replay)
        buffer.update_priorities(indices, td_errors)
    """

    def __init__(
        self,
        config: ReplayBufferConfig,
        state_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize ReplayBuffer.

        Args:
            config: ReplayBuffer configuration
            state_dim: Dimension of state observations
            action_dim: Dimension of actions
            device: Device for tensor operations
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.buffer_size = config.buffer_size
        self.n_step = config.n_step
        self.gamma = config.gamma

        # Circular buffer storage
        self.states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.bool_)
        self.priorities = np.ones((self.buffer_size,), dtype=np.float32)

        # For residual RL: store base actions
        self.base_actions = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.has_base_actions = False  # Track if we're storing base actions

        # Buffer state
        self.size = 0
        self.pointer = 0

        # For n-step returns
        self.n_step_buffer: list[tuple] = []

        # For prioritized replay
        self.max_priority = 1.0
        self.beta = config.priority_beta

        logging.info(f"ReplayBuffer initialized: size={self.buffer_size}, n_step={self.n_step}")

    def add(
        self,
        state: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        reward: float,
        next_state: np.ndarray | torch.Tensor,
        done: bool,
        base_action: np.ndarray | torch.Tensor | None = None,  # For residual RL
    ) -> None:
        """
        Add transition to buffer.

        Args:
            state: Current state
            action: Action taken (residual action in residual RL)
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            base_action: Base policy action (for residual RL, optional)
        """
        # Convert to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(base_action, torch.Tensor):
            base_action = base_action.cpu().numpy()

        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Wait until n-step buffer is full
        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_transition()

        # Add to circular buffer
        self._add_to_buffer(n_state, n_action, n_reward, n_next_state, n_done, base_action)

        # Remove oldest from n-step buffer
        self.n_step_buffer.pop(0)

    def _get_n_step_transition(self) -> tuple:
        """Compute n-step return from n-step buffer."""
        state, action, _, _, _ = self.n_step_buffer[0]

        # Compute n-step reward
        n_reward = 0.0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_reward += (self.gamma ** i) * reward
            if done:
                # Truncate if episode ends
                break

        # Get final state and done
        _, _, _, next_state, done = self.n_step_buffer[-1]

        return state, action, n_reward, next_state, done

    def _add_to_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        base_action: np.ndarray | None = None,
    ) -> None:
        """Add computed transition to circular buffer."""
        # Store transition
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        self.priorities[self.pointer] = self.max_priority

        # Store base action if provided (for residual RL)
        if base_action is not None:
            self.base_actions[self.pointer] = base_action
            self.has_base_actions = True

        # Update pointer and size
        self.pointer = (self.pointer + 1) % self.buffer_size
        if self.size < self.buffer_size:
            self.size += 1

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """
        Sample batch from buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary with batch tensors
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size ({self.size}) < batch size ({batch_size})")

        # Sample indices
        if self.config.sampling_strategy == "uniform":
            indices = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:  # prioritized
            indices, weights = self._sample_prioritized(batch_size)

        # Get batch data
        batch = {
            "state": torch.from_numpy(self.states[indices]).to(self.device),
            "action": torch.from_numpy(self.actions[indices]).to(self.device),
            "reward": torch.from_numpy(self.rewards[indices]).to(self.device).unsqueeze(1),
            "next_state": torch.from_numpy(self.next_states[indices]).to(self.device),
            "done": torch.from_numpy(self.dones[indices]).to(self.device).unsqueeze(1),
            "indices": torch.from_numpy(indices).to(self.device),
            "weights": torch.from_numpy(weights).to(self.device),
        }

        # Add base actions if available (for residual RL)
        if self.has_base_actions:
            batch["base_action"] = torch.from_numpy(self.base_actions[indices]).to(self.device)

        return batch

    def _sample_prioritized(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample with prioritized experience replay."""
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = (priorities + self.config.priority_epsilon) ** self.config.priority_alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Update beta
        self.beta = min(1.0, self.beta + self.config.priority_beta_increment)

        return indices, weights

    def update_priorities(
        self,
        indices: np.ndarray | torch.Tensor,
        td_errors: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of transitions to update
            td_errors: TD errors for priority update
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.cpu().numpy()

        # Update priorities
        priorities = np.abs(td_errors) + self.config.priority_epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.buffer_size

    def clear(self) -> None:
        """Clear all data from buffer."""
        self.size = 0
        self.pointer = 0
        self.n_step_buffer.clear()
        self.max_priority = 1.0
        self.beta = self.config.priority_beta

        logging.info("ReplayBuffer cleared")

    def save(self, save_dir: str | Path) -> None:
        """Save buffer data to disk for checkpoint resume.

        Saves the numpy arrays (states, actions, rewards, next_states, dones,
        base_actions, priorities) along with size, pointer, and config metadata.
        Only saves the filled portion of the circular buffer to minimize file size.

        Args:
            save_dir: Directory to save buffer data (created if needed).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Only save filled portion (not the entire pre-allocated buffer)
        if self.size > 0:
            np.save(save_dir / "states.npy", self.states[:self.size])
            np.save(save_dir / "actions.npy", self.actions[:self.size])
            np.save(save_dir / "rewards.npy", self.rewards[:self.size])
            np.save(save_dir / "next_states.npy", self.next_states[:self.size])
            np.save(save_dir / "dones.npy", self.dones[:self.size])
            np.save(save_dir / "priorities.npy", self.priorities[:self.size])
            if self.has_base_actions:
                np.save(save_dir / "base_actions.npy", self.base_actions[:self.size])

        # Save metadata as JSON (safe, no pickle)
        metadata = {
            "size": self.size,
            "pointer": self.pointer,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "buffer_size": self.buffer_size,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "has_base_actions": self.has_base_actions,
            "max_priority": self.max_priority,
            "beta": self.beta,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"ReplayBuffer saved: {self.size} transitions to {save_dir}")

    def load(self, load_dir: str | Path) -> None:
        """Load buffer data from disk to resume training.

        Restores all buffer arrays and state from a previously saved checkpoint.
        After loading, the buffer can be used for sampling immediately.

        Args:
            load_dir: Directory containing saved buffer data.
        """
        load_dir = Path(load_dir)

        metadata_path = load_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No replay buffer metadata at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Validate dimensions match
        if metadata["state_dim"] != self.state_dim:
            raise ValueError(
                f"State dim mismatch: buffer has {self.state_dim}, "
                f"saved has {metadata['state_dim']}"
            )
        if metadata["action_dim"] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: buffer has {self.action_dim}, "
                f"saved has {metadata['action_dim']}"
            )

        saved_size = metadata["size"]
        saved_pointer = metadata["pointer"]

        # Load arrays if they exist
        states_path = load_dir / "states.npy"
        if states_path.exists() and saved_size > 0:
            saved_states = np.load(states_path)
            self.states[:saved_size] = saved_states

            saved_actions = np.load(load_dir / "actions.npy")
            self.actions[:saved_size] = saved_actions

            saved_rewards = np.load(load_dir / "rewards.npy")
            self.rewards[:saved_size] = saved_rewards

            saved_next_states = np.load(load_dir / "next_states.npy")
            self.next_states[:saved_size] = saved_next_states

            saved_dones = np.load(load_dir / "dones.npy")
            self.dones[:saved_size] = saved_dones

            saved_priorities = np.load(load_dir / "priorities.npy")
            self.priorities[:saved_size] = saved_priorities

            # Load base actions if available
            base_actions_path = load_dir / "base_actions.npy"
            if base_actions_path.exists():
                saved_base_actions = np.load(base_actions_path)
                self.base_actions[:saved_size] = saved_base_actions
                self.has_base_actions = metadata.get("has_base_actions", True)

        self.size = saved_size
        self.pointer = saved_pointer
        self.max_priority = metadata.get("max_priority", 1.0)
        self.beta = metadata.get("beta", self.config.priority_beta)
        self.n_step_buffer.clear()  # n-step buffer not persisted (transient state)

        logging.info(f"ReplayBuffer loaded: {self.size} transitions from {load_dir}")

    def get_statistics(self) -> dict[str, float]:
        """Get buffer statistics."""
        return {
            "size": self.size,
            "buffer_size": self.buffer_size,
            "usage": self.size / self.buffer_size,
            "max_priority": self.max_priority,
            "beta": self.beta,
        }


class MixedReplayBuffer:
    """
    Mixed replay buffer for combining online and offline data.

    This is critical for ResFiT which uses:
    - Offline data from BC demonstrations (for stable learning)
    - Online data from robot exploration (for adaptation)

    Example:
        offline_buffer = ReplayBuffer(...)
        online_buffer = ReplayBuffer(...)

        mixed_buffer = MixedReplayBuffer(
            offline_rb=offline_buffer,
            online_rb=online_buffer,
            offline_fraction=0.5,  # 50% offline, 50% online
        )

        # Sample mixed batch
        batch = mixed_buffer.sample(batch_size=256)
    """

    def __init__(
        self,
        offline_rb: ReplayBuffer,
        online_rb: ReplayBuffer,
        offline_fraction: float = 0.5,
    ):
        """
        Initialize MixedReplayBuffer.

        Args:
            offline_rb: Offline replay buffer
            online_rb: Online replay buffer
            offline_fraction: Fraction of batch to sample from offline buffer
        """
        self.offline_rb = offline_rb
        self.online_rb = online_rb
        self.offline_fraction = offline_fraction

        logging.info(f"MixedReplayBuffer initialized: offline_fraction={offline_fraction}")

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """
        Sample mixed batch.

        Args:
            batch_size: Total batch size

        Returns:
            Dictionary with combined batch tensors
        """
        # Compute batch sizes for each buffer
        offline_size = int(batch_size * self.offline_fraction)
        online_size = batch_size - offline_size

        # Sample from each buffer
        batches = []

        if offline_size > 0 and len(self.offline_rb) >= offline_size:
            batches.append(self.offline_rb.sample(offline_size))

        if online_size > 0 and len(self.online_rb) >= online_size:
            batches.append(self.online_rb.sample(online_size))

        if not batches:
            raise ValueError("Insufficient data in buffers")

        # Combine batches
        combined = {}
        for key in batches[0]:
            combined[key] = torch.cat([b[key] for b in batches], dim=0)

        return combined

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for both buffers."""
        return {
            "offline": self.offline_rb.get_statistics(),
            "online": self.online_rb.get_statistics(),
            "offline_fraction": self.offline_fraction,
        }