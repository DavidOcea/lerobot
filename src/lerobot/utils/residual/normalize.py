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
Normalization utilities for residual RL training.

ActionScaler: Normalizes actions to [-1, 1] range for consistent exploration
StateStandardizer: Standardizes states (zero mean, unit variance) for stable learning

These are critical for:
1. Consistent exploration across all action dimensions
2. Stable neural network training
3. Proper residual action composition
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class NormalizationConfig:
    """Configuration for normalization."""

    # Minimum action range to prevent normalization blow-up
    min_action_range: float = 1e-1

    # Minimum state std to prevent division by zero
    min_state_std: float = 1e-1

    # Whether to compute normalization from offline dataset
    compute_from_dataset: bool = True

    # Pre-computed normalization file (if not computing from dataset)
    normalization_file: str | None = None


class ActionScaler:
    """
    Scales actions to [-1, 1] normalized range.

    This is critical for residual RL because:
    1. Exploration noise is applied uniformly in [-1, 1]
    2. Residual actions are additive: base_naction + residual_naction
    3. The combined action is then unscaled back to environment range

    Example:
        scaler = ActionScaler.from_dataset(dataset_actions)

        # During training:
        base_naction = scaler.scale(base_action)  # -> [-1, 1]
        residual_naction = actor(obs) * action_scale  # -> [-0.1, 0.1]
        combined = base_naction + residual_naction
        env_action = scaler.unscale(combined)  # -> original range

        # Save/load for consistency:
        scaler.save("action_scaler.json")
        scaler = ActionScaler.load("action_scaler.json")
    """

    def __init__(
        self,
        action_min: np.ndarray | torch.Tensor,
        action_max: np.ndarray | torch.Tensor,
        min_range: float = 1e-1,
    ):
        """
        Initialize ActionScaler.

        Args:
            action_min: Minimum action values per dimension
            action_max: Maximum action values per dimension
            min_range: Minimum range to prevent blow-up
        """
        # Convert to numpy for storage
        if isinstance(action_min, torch.Tensor):
            action_min = action_min.cpu().numpy()
        if isinstance(action_max, torch.Tensor):
            action_max = action_max.cpu().numpy()

        # Ensure minimum range
        action_range = action_max - action_min
        action_range = np.maximum(action_range, min_range)

        # Store normalized bounds
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_range
        self.action_center = (action_max + action_min) / 2.0

        self.min_range = min_range
        self.action_dim = len(action_min)

        logging.info(f"ActionScaler initialized with {self.action_dim} dimensions")
        logging.info(f"  Action range: {self.action_range}")
        logging.info(f"  Action center: {self.action_center}")

    @classmethod
    def from_dataset(cls, actions: np.ndarray | torch.Tensor, min_range: float = 1e-1) -> "ActionScaler":
        """
        Create ActionScaler from dataset actions.

        Args:
            actions: Array of actions from dataset (N, action_dim)
            min_range: Minimum range to prevent blow-up

        Returns:
            ActionScaler instance
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Use 1st and 99th percentile to handle outliers
        action_min = np.percentile(actions, 1, axis=0)
        action_max = np.percentile(actions, 99, axis=0)

        # Add small margin
        margin = (action_max - action_min) * 0.05
        action_min = action_min - margin
        action_max = action_max + margin

        return cls(action_min, action_max, min_range)

    @classmethod
    def from_env(cls, action_space_low: np.ndarray, action_space_high: np.ndarray, min_range: float = 1e-1) -> "ActionScaler":
        """
        Create ActionScaler from environment action space.

        Args:
            action_space_low: Lower bounds from env.action_space.low
            action_space_high: Upper bounds from env.action_space.high
            min_range: Minimum range to prevent blow-up

        Returns:
            ActionScaler instance
        """
        return cls(action_space_low, action_space_high, min_range)

    def scale(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Scale action from environment range to [-1, 1].

        Args:
            action: Action in environment range

        Returns:
            Normalized action in [-1, 1]
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        # Convert normalization params to tensors on same device as action
        action_center = torch.from_numpy(self.action_center).to(action.device, dtype=action.dtype)
        action_range = torch.from_numpy(self.action_range).to(action.device, dtype=action.dtype)

        # Normalize: (action - center) / (range / 2)
        normalized = (action - action_center) / (action_range / 2.0)

        # Clip to [-1, 1] for safety
        normalized = torch.clamp(normalized, -1.0, 1.0)

        return normalized

    def unscale(self, normalized_action: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Unscale action from [-1, 1] to environment range.

        Args:
            normalized_action: Normalized action in [-1, 1]

        Returns:
            Action in environment range
        """
        if isinstance(normalized_action, np.ndarray):
            normalized_action = torch.from_numpy(normalized_action).float()

        # Convert normalization params to tensors on same device as action
        action_center = torch.from_numpy(self.action_center).to(normalized_action.device, dtype=normalized_action.dtype)
        action_range = torch.from_numpy(self.action_range).to(normalized_action.device, dtype=normalized_action.dtype)

        # Unnormalize: normalized * (range / 2) + center
        action = normalized_action * (action_range / 2.0) + action_center

        return action

    def save(self, path: Path | str) -> None:
        """Save scaler parameters to JSON file."""
        path = Path(path)
        data = {
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "action_range": self.action_range.tolist(),
            "action_center": self.action_center.tolist(),
            "min_range": self.min_range,
            "action_dim": self.action_dim,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"ActionScaler saved to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "ActionScaler":
        """Load scaler parameters from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        scaler = cls(
            action_min=np.array(data["action_min"]),
            action_max=np.array(data["action_max"]),
            min_range=data["min_range"],
        )
        logging.info(f"ActionScaler loaded from: {path}")
        return scaler

    def to(self, device: torch.device) -> "ActionScaler":
        """Move scaler to specified device (for tensor operations)."""
        # Note: This returns a wrapper that converts arrays to tensors on device
        # The underlying arrays are still numpy, but scale/unscale will use tensors on device
        return self


class StateStandardizer:
    """
    Standardizes state observations (zero mean, unit variance).

    This is critical for:
    1. Stable neural network training
    2. Consistent feature scales across dimensions
    3. Better gradient flow during learning

    Example:
        standardizer = StateStandardizer.from_dataset(dataset_states)

        # During training:
        standardized_state = standardizer.standardize(raw_state)

        # Save/load for consistency:
        standardizer.save("state_standardizer.json")
        standardizer = StateStandardizer.load("state_standardizer.json")
    """

    def __init__(
        self,
        state_mean: np.ndarray | torch.Tensor,
        state_std: np.ndarray | torch.Tensor,
        min_std: float = 1e-1,
    ):
        """
        Initialize StateStandardizer.

        Args:
            state_mean: Mean values per dimension
            state_std: Standard deviation per dimension
            min_std: Minimum std to prevent division by zero
        """
        # Convert to numpy for storage
        if isinstance(state_mean, torch.Tensor):
            state_mean = state_mean.cpu().numpy()
        if isinstance(state_std, torch.Tensor):
            state_std = state_std.cpu().numpy()

        # Ensure minimum std
        state_std = np.maximum(state_std, min_std)

        self.state_mean = state_mean
        self.state_std = state_std
        self.min_std = min_std
        self.state_dim = len(state_mean)

        logging.info(f"StateStandardizer initialized with {self.state_dim} dimensions")
        logging.info(f"  Mean range: [{state_mean.min():.4f}, {state_mean.max():.4f}]")
        logging.info(f"  Std range: [{state_std.min():.4f}, {state_std.max():.4f}]")

    @classmethod
    def from_dataset(cls, states: np.ndarray | torch.Tensor, min_std: float = 1e-1) -> "StateStandardizer":
        """
        Create StateStandardizer from dataset states.

        Args:
            states: Array of states from dataset (N, state_dim)
            min_std: Minimum std to prevent division by zero

        Returns:
            StateStandardizer instance
        """
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        # Compute mean and std
        state_mean = np.mean(states, axis=0)
        state_std = np.std(states, axis=0)

        return cls(state_mean, state_std, min_std)

    @classmethod
    def from_config(cls, state_dim: int, min_std: float = 1e-1) -> "StateStandardizer":
        """
        Create identity standardizer (no normalization).

        Useful when normalization should be disabled.

        Args:
            state_dim: Dimension of state
            min_std: Minimum std (not used, but kept for consistency)

        Returns:
            StateStandardizer with zero mean and unit std
        """
        return cls(
            state_mean=np.zeros(state_dim),
            state_std=np.ones(state_dim),
            min_std=min_std,
        )

    def standardize(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Standardize state (zero mean, unit variance).

        Args:
            state: Raw state observation

        Returns:
            Standardized state
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        # Convert normalization params to tensors on same device as state
        state_mean = torch.from_numpy(self.state_mean).to(state.device, dtype=state.dtype)
        state_std = torch.from_numpy(self.state_std).to(state.device, dtype=state.dtype)

        # Standardize: (state - mean) / std
        standardized = (state - state_mean) / state_std

        return standardized

    def unstandardize(self, standardized_state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Unstandardize state back to original scale.

        Args:
            standardized_state: Standardized state

        Returns:
            Raw state
        """
        if isinstance(standardized_state, np.ndarray):
            standardized_state = torch.from_numpy(standardized_state).float()

        # Convert normalization params to tensors on same device as state
        state_mean = torch.from_numpy(self.state_mean).to(standardized_state.device, dtype=standardized_state.dtype)
        state_std = torch.from_numpy(self.state_std).to(standardized_state.device, dtype=standardized_state.dtype)

        # Unstandardize: standardized * std + mean
        state = standardized_state * state_std + state_mean

        return state

    def save(self, path: Path | str) -> None:
        """Save standardizer parameters to JSON file."""
        path = Path(path)
        data = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "min_std": self.min_std,
            "state_dim": self.state_dim,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"StateStandardizer saved to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "StateStandardizer":
        """Load standardizer parameters from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        standardizer = cls(
            state_mean=np.array(data["state_mean"]),
            state_std=np.array(data["state_std"]),
            min_std=data["min_std"],
        )
        logging.info(f"StateStandardizer loaded from: {path}")
        return standardizer

    def to(self, device: torch.device) -> "StateStandardizer":
        """Move standardizer to specified device."""
        return self


def compute_normalization_from_dataset(
    dataset: Any,  # LeRobotDataset
    action_keys: list[str] = ["action"],
    state_keys: list[str] = ["observation.state"],
    config: NormalizationConfig = NormalizationConfig(),
) -> tuple[ActionScaler, StateStandardizer]:
    """
    Compute normalization parameters from LeRobotDataset.

    Args:
        dataset: LeRobotDataset instance
        action_keys: Keys for action data in dataset
        state_keys: Keys for state data in dataset
        config: Normalization configuration

    Returns:
        Tuple of (ActionScaler, StateStandardizer)
    """
    logging.info("Computing normalization from dataset...")

    # Extract actions
    # LeRobotDataset stores actions as episode_data
    actions = []
    for episode in dataset.episode_data:
        if action_keys[0] in episode:
            actions.append(episode[action_keys[0]])

    if actions:
        actions = np.concatenate(actions, axis=0)
        action_scaler = ActionScaler.from_dataset(actions, config.min_action_range)
    else:
        raise ValueError(f"No action data found with keys: {action_keys}")

    # Extract states
    states = []
    for episode in dataset.episode_data:
        for key in state_keys:
            if key in episode:
                states.append(episode[key])

    if states:
        states = np.concatenate(states, axis=0)
        state_standardizer = StateStandardizer.from_dataset(states, config.min_state_std)
    else:
        logging.warning(f"No state data found with keys: {state_keys}, using identity standardizer")
        state_standardizer = StateStandardizer.from_config(action_scaler.action_dim, config.min_state_std)

    logging.info("Normalization computed successfully")

    return action_scaler, state_standardizer


def save_normalization(
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    save_dir: Path | str,
) -> None:
    """
    Save normalization parameters to directory.

    Args:
        action_scaler: ActionScaler instance
        state_standardizer: StateStandardizer instance
        save_dir: Directory to save parameters
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    action_scaler.save(save_dir / "action_scaler.json")
    state_standardizer.save(save_dir / "state_standardizer.json")

    logging.info(f"Normalization parameters saved to: {save_dir}")


def load_normalization(
    load_dir: Path | str,
) -> tuple[ActionScaler, StateStandardizer]:
    """
    Load normalization parameters from directory.

    Args:
        load_dir: Directory containing normalization files

    Returns:
        Tuple of (ActionScaler, StateStandardizer)
    """
    load_dir = Path(load_dir)

    action_scaler = ActionScaler.load(load_dir / "action_scaler.json")
    state_standardizer = StateStandardizer.load(load_dir / "state_standardizer.json")

    logging.info(f"Normalization parameters loaded from: {load_dir}")

    return action_scaler, state_standardizer