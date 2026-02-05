"""
Task execution without Policy Server - Local inference mode.

This module provides direct policy execution without the need for a separate
Policy Server process. The orchestrator loads and executes policies locally,
resulting in lower latency and simpler architecture.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.tasks.config import TaskConfig

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


class LocalPolicyExecutor:
    """Executes policies locally without needing a Policy Server.

    This executor:
    - Loads ACT policies directly from disk
    - Performs inference on the device (CPU/GPU)
    - Returns actions synchronously

    Usage:
        executor = LocalPolicyExecutor(device="cuda")
        executor.load_policy("/path/to/model")
        action = executor.get_action(observation)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the local policy executor.

        Args:
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.device = device
        self.policy: ACTPolicy | None = None
        self.current_policy_path: str | None = None

        # Observation buffer for temporal batching
        self.observation_buffer: list[dict[str, Any]] = []
        self.n_obs_steps = 1

        # Cache for joint names from robot config
        self._joint_names_from_robot: list[str] | None = None

    def load_policy(self, policy_path: str, policy_type: str = "act") -> bool:
        """Load a policy from disk.

        Args:
            policy_path: Path to the pretrained model directory.
            policy_type: Type of policy ("act", "diffusion", etc.).

        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            logger.info(f"Loading policy from: {policy_path}")

            if policy_type == "act":
                self.policy = ACTPolicy.from_pretrained(policy_path)
                self.policy.to(self.device)
            else:
                raise ValueError(f"Unsupported policy type: {policy_type}")

            # Get n_obs_steps from policy config
            self.n_obs_steps = self.policy.config.n_obs_steps

            # Clear observation buffer
            self.observation_buffer.clear()

            self.current_policy_path = policy_path
            logger.info(f"Policy loaded successfully")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  n_obs_steps: {self.n_obs_steps}")
            logger.info(f"  chunk_size: {self.policy.config.chunk_size}")
            logger.info(f"  n_action_steps: {self.policy.config.n_action_steps}")

            return True

        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            return False

    def set_joint_names(self, joint_names: list[str]) -> None:
        """Set the joint names to use for observation/action mapping.

        This should match the order used during policy training.

        Args:
            joint_names: List of joint names in the order expected by the policy.
        """
        self._joint_names_from_robot = joint_names
        logger.info(f"Set joint names from robot config: {joint_names}")

    def get_action(self, observation: dict[str, Any]) -> dict[str, float] | None:
        """Get action from the current policy.

        Args:
            observation: Current observation dict containing:
                - {joint}.pos: joint positions
                - {joint}.force: joint torques
                - {camera}: camera images

        Returns:
            Action dict with joint positions, or None if inference fails.
        """
        if self.policy is None:
            logger.error("No policy loaded")
            return None

        try:
            # Prepare observation batch
            batch = self._prepare_observation_batch(observation)

            # Run inference
            with torch.no_grad():
                self.policy.eval()
                action_tensor = self.policy.select_action(batch)

            # Convert to dict
            action = self._action_tensor_to_dict(action_tensor)

            return action

        except Exception as e:
            logger.error(f"Failed to get action: {e}")
            return None

    def get_action_chunk(
        self, observation: dict[str, Any]
    ) -> list[dict[str, float]] | None:
        """Get a chunk of actions from the current policy.

        Args:
            observation: Current observation dict.

        Returns:
            List of action dicts, or None if inference fails.
        """
        if self.policy is None:
            logger.error("No policy loaded")
            return None

        try:
            # Prepare observation batch
            batch = self._prepare_observation_batch(observation)

            # Run inference
            with torch.no_grad():
                self.policy.eval()
                action_tensor = self.policy.predict_action_chunk(batch)

            # Convert to list of dicts
            action_chunk = []
            chunk_size = action_tensor.shape[0]
            for i in range(chunk_size):
                action = self._action_tensor_to_dict(action_tensor[i])
                action_chunk.append(action)

            return action_chunk

        except Exception as e:
            logger.error(f"Failed to get action chunk: {e}")
            return None

    def _prepare_observation_batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Prepare observation batch for policy inference.

        Args:
            observation: Current observation dict.

        Returns:
            Batch dict ready for policy input.
        """
        # Add to buffer
        self.observation_buffer.append(observation.copy())

        # Keep only the last n_obs_steps
        if len(self.observation_buffer) > self.n_obs_steps:
            self.observation_buffer = self.observation_buffer[-self.n_obs_steps:]

        # Prepare batch
        batch = {}

        # Use joint names from robot config if available, otherwise fallback to sorted keys
        if self._joint_names_from_robot is not None:
            # Use the order from robot config (matches training data)
            pos_keys = [f"{name}.pos" for name in self._joint_names_from_robot]
            force_keys = [f"{name}.force" for name in self._joint_names_from_robot]
        else:
            # Fallback: collect all position and force keys, sort alphabetically
            pos_keys = sorted([k for k in observation.keys() if k.endswith('.pos')])
            force_keys = sorted([k for k in observation.keys() if k.endswith('.force') or '.force' in k])

        # Add state vector to batch
        if pos_keys:
            state_values = []
            for obs in self.observation_buffer:
                # Extract state from each observation in buffer (same order)
                obs_state = [obs.get(k, 0.0) for k in pos_keys]
                state_values.append(obs_state)

            # Pad if needed
            while len(state_values) < self.n_obs_steps:
                state_values.insert(0, state_values[0] if state_values else [0.0] * len(pos_keys))

            # Convert to tensor: (n_obs_steps, n_joints)
            state_tensor = torch.tensor(state_values, dtype=torch.float32)
            if self.n_obs_steps == 1:
                # (1, n_joints) - squeeze extra dimension
                batch["observation.state"] = state_tensor
            else:
                # (n_obs_steps, n_joints) -> add batch dimension
                batch["observation.state"] = state_tensor.unsqueeze(0)

        # Add force vector to batch
        if force_keys:
            force_values = []
            for obs in self.observation_buffer:
                # Extract force from each observation in buffer (same order)
                obs_force = [obs.get(k, 0.0) for k in force_keys]
                force_values.append(obs_force)

            # Pad if needed
            while len(force_values) < self.n_obs_steps:
                force_values.insert(0, force_values[0] if force_values else [0.0] * len(force_keys))

            # Convert to tensor: (n_obs_steps, n_joints)
            force_tensor = torch.tensor(force_values, dtype=torch.float32)
            if self.n_obs_steps == 1:
                # (1, n_joints) - squeeze extra dimension
                batch["observation.force"] = force_tensor
            else:
                # (n_obs_steps, n_joints) -> add batch dimension
                batch["observation.force"] = force_tensor.unsqueeze(0)

        # Handle nested images structure if present
        if 'images' in observation and isinstance(observation['images'], dict):
            for cam_name in observation['images'].keys():
                # Collect values from buffer
                values = []
                for obs in self.observation_buffer:
                    if 'images' in obs and isinstance(obs['images'], dict) and cam_name in obs['images']:
                        values.append(obs['images'][cam_name])
                    else:
                        values.append(None)

                # Filter out None values
                if all(v is None for v in values):
                    continue  # Skip this camera if all values are None

                # Use first non-None value for padding
                first_valid = next((v for v in values if v is not None), None)

                if first_valid is None:
                    continue

                # Pad if needed
                while len(values) < self.n_obs_steps:
                    values.insert(0, first_valid)

                # Convert numpy arrays to torch tensors
                if hasattr(first_valid, 'shape'):
                    import numpy as np

                    if self.n_obs_steps == 1:
                        # No temporal stacking needed, just use the single observation
                        img = values[0]
                        # Convert to tensor: (H, W, C) -> (1, C, H, W)
                        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    else:
                        # Stack multiple observations along temporal dimension
                        stacked = np.stack([v if v is not None else first_valid for v in values])
                        # Convert to tensor: (time, H, W, C) -> (1, C, time, H, W)
                        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0

                    # Use the prefixed key format expected by the policy
                    batch[f"observation.images.{cam_name}"] = tensor

        # Add batch dimension

        # Add batch dimension
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                batch[key] = value.unsqueeze(0)

        # Move to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        return batch

    def _action_tensor_to_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        """Convert action tensor to dictionary format.

        Args:
            action_tensor: Action tensor from policy output.

        Returns:
            Dictionary mapping joint names to positions.
        """
        # Get action features from policy
        action_features = self.policy.config.output_features

        # Determine joint names to use
        if self._joint_names_from_robot is not None:
            # Use joint names from robot config (matches training data order)
            joint_names = self._joint_names_from_robot
        else:
            # Fallback: use hardcoded joint order
            # This matches trunk_config_supre_robot_joint.yaml order
            joint_names = [
                "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
                "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
                "left_arm_joint_7",
                "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
                "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
                "right_arm_joint_7",
                "trunk_joint_1", "trunk_joint_2",
            ]

        # Trim to action_dim
        for key in action_features.keys():
            if key == "action":
                feature = action_features[key]
                if hasattr(feature, "shape") and len(feature.shape) > 0:
                    action_dim = feature.shape[0]
                    joint_names = joint_names[:action_dim]
                break

        # Convert to dict
        action_dict = {}
        action_tensor_flat = action_tensor.flatten() if action_tensor.numel() > 0 else action_tensor
        for i, joint_name in enumerate(joint_names):
            if i < len(action_tensor_flat):
                action_dict[f"{joint_name}.pos"] = float(action_tensor_flat[i].item())

        return action_dict

    def reset(self):
        """Reset the executor state."""
        self.observation_buffer.clear()
        logger.debug("Executor reset")

    def get_info(self) -> dict[str, Any]:
        """Get executor information."""
        return {
            "policy_loaded": self.policy is not None,
            "current_policy_path": self.current_policy_path,
            "device": self.device,
            "n_obs_steps": self.n_obs_steps,
            "buffer_size": len(self.observation_buffer),
        }
