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
Environment wrapper for residual RL training.

Adapts the existing LeRobot gym_manipulator environment for residual RL by:
1. Passing observations through base policy to get base actions
2. Building residual policy observations (state + base_action)
3. Combining base + residual actions before environment execution

This wrapper is designed for single-environment (non-vectorized) training,
unlike ResFiT's vectorized environment wrapper.
"""

import logging
import gymnasium as gym
import torch
import numpy as np
from typing import Any

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.residual.normalize import ActionScaler, StateStandardizer


class ResidualEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for residual RL training.

    Wraps a LeRobot robot environment (from gym_manipulator) to enable
    residual RL training with a base BC policy.

    Key responsibilities:
    1. Get base policy action from raw observation
    2. Normalize actions and states
    3. Build residual policy observation with base_action
    4. Combine base + residual actions
    5. Apply action to environment

    Example:
        # Create base environment
        env = make_robot_env(cfg=env_config)

        # Create wrapper
        residual_env = ResidualEnvWrapper(
            env=env,
            base_policy=act_policy,
            action_scaler=scaler,
            state_standardizer=standardizer,
            action_scale=0.1,
        )

        # Use in training loop
        obs, _ = residual_env.reset()
        residual_action = td3_agent.act(obs)
        next_obs, reward, done, truncated, info = residual_env.step(residual_action)
    """

    def __init__(
        self,
        env: gym.Env,
        base_policy: ACTPolicy,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
        action_scale: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize ResidualEnvWrapper.

        Args:
            env: Base robot environment (from make_robot_env)
            base_policy: Pre-trained ACT base policy
            action_scaler: Action normalization scaler
            state_standardizer: State standardizer
            action_scale: Maximum residual action magnitude
            device: Device for computation
        """
        super().__init__(env)

        self.base_policy = base_policy.to(device)
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.action_scale = action_scale
        self.device = device

        # Store last base normalized action for step
        self._last_base_naction: torch.Tensor | None = None

        # Action dimension
        self.action_dim = env.action_space.shape[-1]

        # Image keys for base policy
        self.image_keys = list(base_policy.config.image_features.keys())

        logging.info(f"ResidualEnvWrapper initialized: action_dim={self.action_dim}, action_scale={action_scale}")

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """
        Reset environment and base policy.

        Returns:
            Tuple of (augmented_obs, info)
            - augmented_obs: Observation with base_action and standardized state
            - info: Environment info
        """
        # Reset underlying environment
        raw_obs, info = self.env.reset(**kwargs)

        # Reset base policy
        self.base_policy.reset()

        # Convert raw_obs to tensors if needed
        raw_obs_tensors = self._convert_obs_to_tensors(raw_obs)

        # Get base policy action
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs_tensors)

        # Normalize base action
        base_naction = self.action_scaler.scale(base_action)

        # Build residual observation
        augmented_obs = self._build_residual_obs(raw_obs_tensors, base_naction)

        # Store for next step
        self._last_base_naction = base_naction

        return augmented_obs, info

    def step(
        self,
        residual_naction: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict]:
        """
        Step environment with residual action.

        Args:
            residual_naction: Residual action from TD3 (normalized, in [-action_scale, action_scale])

        Returns:
            Tuple of (augmented_obs, reward, terminated, truncated, info)
        """
        # Combine base and residual actions
        combined_naction = self._last_base_naction + residual_naction

        # Clip to normalized bounds [-1, 1]
        combined_naction = torch.clamp(combined_naction, -1.0, 1.0)

        # Unnormalize to environment action space
        env_action = self.action_scaler.unscale(combined_naction)

        # Convert to numpy for environment
        if isinstance(env_action, torch.Tensor):
            env_action_np = env_action.cpu().numpy()
        else:
            env_action_np = env_action

        # Step environment
        raw_obs, reward, terminated, truncated, info = self.env.step(env_action_np)

        # Add action info to info dict
        info["residual_action"] = residual_naction.cpu().numpy() if isinstance(residual_naction, torch.Tensor) else residual_naction
        info["base_action"] = self._last_base_naction.cpu().numpy() if isinstance(self._last_base_naction, torch.Tensor) else self._last_base_naction
        info["combined_action"] = combined_naction.cpu().numpy() if isinstance(combined_naction, torch.Tensor) else combined_naction

        # Convert raw_obs to tensors
        raw_obs_tensors = self._convert_obs_to_tensors(raw_obs)

        # Get next base action
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs_tensors)

        # Normalize
        base_naction = self.action_scaler.scale(base_action)

        # Build residual observation
        augmented_obs = self._build_residual_obs(raw_obs_tensors, base_naction)

        # Store for next step
        self._last_base_naction = base_naction

        return augmented_obs, reward, terminated, truncated, info

    def _convert_obs_to_tensors(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Convert raw observation to tensors on device.

        Args:
            raw_obs: Raw observation from environment (may be numpy arrays)

        Returns:
            Observation dictionary with tensors
        """
        obs_tensors = {}

        for key, value in raw_obs.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                obs_tensors[key] = value.to(self.device)
            else:
                obs_tensors[key] = value

        return obs_tensors

    def _build_residual_obs(
        self,
        raw_obs: dict[str, torch.Tensor],
        base_naction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Build observation for residual policy.

        Args:
            raw_obs: Raw observation from environment
            base_naction: Normalized base action

        Returns:
            Augmented observation with:
            - observation.state: Standardized state
            - observation.base_action: Normalized base action
            - Other keys (images, etc.) unchanged
        """
        augmented_obs = raw_obs.copy()

        # Add base action
        augmented_obs["observation.base_action"] = base_naction

        # Standardize state if present
        if "observation.state" in augmented_obs:
            augmented_obs["observation.state"] = self.state_standardizer.standardize(
                augmented_obs["observation.state"]
            )

        return augmented_obs

    def get_base_action(self) -> torch.Tensor | None:
        """Get the last base action (for monitoring)."""
        return self._last_base_naction

    def set_action_scale(self, action_scale: float) -> None:
        """Update action scale (for progressive clipping)."""
        self.action_scale = action_scale


class ResidualEnvWrapperWithIntervention(ResidualEnvWrapper):
    """
    Residual environment wrapper with human intervention support.

    Extends ResidualEnvWrapper to support:
    - Human intervention via leader arm or gamepad
    - Intervention detection (from BaseLeaderControlWrapper)
    - Recording intervention actions for HIL-style training

    This can be used for hybrid HIL + Residual RL training.
    """

    def __init__(
        self,
        env: gym.Env,
        base_policy: ACTPolicy,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
        action_scale: float = 0.1,
        device: torch.device = torch.device("cuda"),
        use_intervention: bool = False,
    ):
        """
        Initialize with intervention support.

        Args:
            use_intervention: Whether to enable intervention mode
        """
        super().__init__(
            env=env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
            action_scale=action_scale,
            device=device,
        )

        self.use_intervention = use_intervention
        self.intervention_active = False

        logging.info(f"Intervention mode: {use_intervention}")

    def step(
        self,
        residual_naction: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict]:
        """
        Step with intervention support.

        If intervention is detected (via info["is_intervention"]),
        the residual action is replaced with intervention action.
        """
        # Check for intervention in underlying environment
        # This requires the env to be wrapped with BaseLeaderControlWrapper
        obs, reward, terminated, truncated, info = super().step(residual_naction)

        # Check intervention status
        if "is_intervention" in info and info["is_intervention"]:
            self.intervention_active = True
            # Intervention action is in info["action_intervention"]
            if "action_intervention" in info:
                # Store intervention action for potential HIL training
                info["intervention_residual"] = info["action_intervention"]
        else:
            self.intervention_active = False

        return obs, reward, terminated, truncated, info