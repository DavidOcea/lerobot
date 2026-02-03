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

        # Stack observations for temporal dimension
        for key in observation.keys():
            # Collect values from buffer
            values = [obs.get(key, 0) for obs in self.observation_buffer]

            # Pad if needed
            while len(values) < self.n_obs_steps:
                values.insert(0, values[0])

            # Convert to tensor
            if isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            elif isinstance(values[0], dict):
                # Handle nested structures (like images)
                batch[key] = values[0]  # TODO: proper stacking for images
            else:
                batch[key] = values[0]

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

        # Extract joint names from action features
        joint_names = []
        for key in action_features.keys():
            if key == "action":
                # For single action output
                import json
                if hasattr(action_features[key], "shape"):
                    # Assume standard joint order
                    joint_names = [
                        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
                        "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
                        "left_arm_joint_7", "right_arm_joint_1", "right_arm_joint_2",
                        "right_arm_joint_3", "right_arm_joint_4", "right_arm_joint_5",
                        "right_arm_joint_6", "right_arm_joint_7", "trunk_joint_1",
                        "trunk_joint_2",
                    ][:action_tensor.shape[0]]
                break

        # Convert to dict
        action_dict = {}
        for i, joint_name in enumerate(joint_names):
            if i < action_tensor.shape[-1]:
                action_dict[f"{joint_name}.pos"] = float(action_tensor[i].item())

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
