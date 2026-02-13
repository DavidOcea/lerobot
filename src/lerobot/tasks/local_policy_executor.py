#!/usr/bin/env python3

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

# ruff: noqa: E402

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACTPolicy


class LocalPolicyExecutor:
    """Execute policy locally on the robot without using Policy Server.

    This class handles:
    - Loading and switching between ACT policies
    - Converting observation dictionary to model input format
    - Running policy model to get actions
    - Converting action tensors back to dictionary format
    - Managing action queue with temporal ensembling
    - Applying device transfers and type conversions
    """

    def __init__(
        self,
        policy_model: "ACTPolicy | None" = None,
        robot_state_feature: str | None = None,
        image_features: list[str] | None = None,
        n_obs_steps: int = 1,
        device: str = "cpu",
    ):
        """Initialize the local policy executor.

        Args:
            policy_model: Trained ACT model for generating actions (can be None initially).
            robot_state_feature: Robot state feature name (optional).
            image_features: List of image feature names (optional).
            n_obs_steps: Number of observation steps to stack.
            device: Device to run on ("cpu" or "cuda").
        """
        self.policy_model = policy_model
        self.robot_state_feature = robot_state_feature
        self.image_features = image_features if image_features else []
        self.n_obs_steps = n_obs_steps
        self.device = device

        # For observation stacking
        self.observation_buffer: list[dict[str, Any]] = []
        self._action_queue = []

        # Joint names from robot (for action dict conversion)
        self._joint_names_from_robot: list[str] | None = None

    def set_joint_names(self, joint_names: list[str]) -> None:
        """Set joint names from robot configuration.

        Args:
            joint_names: List of joint names from robot.observation_joint_names.
        """
        self._joint_names_from_robot = joint_names
        logger.debug(f"Set joint names: {joint_names[:3]}... (total {len(joint_names)})")

    def load_policy(self, policy_path: str, policy_type: str = "act") -> bool:
        """Load a policy model from disk.

        Args:
            policy_path: Path to the pretrained policy model.
            policy_type: Type of policy (e.g., "act", "diffusion").

        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            policy_path = str(Path(policy_path).expanduser().resolve())

            # Load policy based on type
            if policy_type == "act":
                self.policy_model = ACTPolicy.from_pretrained(policy_path)
            elif policy_type == "diffusion":
                self.policy_model = DiffusionPolicy.from_pretrained(policy_path)
            else:
                logger.error(f"Unknown policy type: {policy_type}")
                return False

            logger.info(f"Loaded policy from {policy_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load policy from {policy_path}: {e}")
            return False

    def reset(self) -> None:
        """Reset the executor state for a new task execution.

        Clears action queue and observation buffer, and resets
        the policy's temporal ensembler if available.
        """
        self._action_queue = []
        self.observation_buffer = []

        # Reset policy's temporal ensembler if it exists
        if self.policy_model is not None and hasattr(self.policy_model, "reset"):
            self.policy_model.reset()

        logger.debug("Policy executor reset")

    def get_action(
        self,
        observation: dict[str, Any],
    ) -> dict[str, float]:
        """Execute policy and return action dictionary.

        This method manages the action queue internally:
        - When queue is empty, run policy inference to populate it
        - When queue has actions, return the next one
        - Supports temporal ensembling for smooth trajectories

        Args:
            observation: Observation dictionary containing sensor data and images.

        Returns:
            Dictionary mapping joint names to position values.
        """
        # Add observation to buffer for stacking
        self.observation_buffer.append(observation)
        if len(self.observation_buffer) > self.n_obs_steps:
            self.observation_buffer.pop(0)

        # Check if we need to run inference (queue empty or policy requires fresh obs)
        needs_inference = len(self._action_queue) == 0

        if needs_inference and self.policy_model is None:
            logger.error("Policy model not loaded. Call load_policy() first.")
            return {}

        if needs_inference:
            # Prepare batch for model inference
            batch = self._prepare_batch(observation)

            # Run policy inference
            try:
                with torch.no_grad():
                    if hasattr(self.policy_model, "select_action"):
                        # ACT policy with temporal ensembling
                        action_tensor = self.policy_model.select_action(batch)
                    elif hasattr(self.policy_model, "forward"):
                        # Generic policy
                        action_tensor = self.policy_model(batch)[0]
                    else:
                        logger.error(f"Policy model {type(self.policy_model)} has no select_action or forward method")
                        return {}
            except Exception as e:
                logger.error(f"Policy inference failed: {e}")
                return {}

            # Process output: actions is (chunk_size, action_dim) tensor
            if action_tensor.dim() == 1:
                # Single action: add to queue
                self._action_queue.append(action_tensor)
            elif action_tensor.dim() == 2:
                # Action chunk: add all to queue
                for i in range(action_tensor.shape[0]):
                    self._action_queue.append(action_tensor[i])

        # Return next action from queue
        if not self._action_queue:
            logger.warning("Action queue is empty - inference may have failed")
            return {}

        # Get next action and convert to dict
        next_action_tensor = self._action_queue.pop(0)
        return self._action_tensor_to_dict(next_action_tensor)

    def _prepare_batch(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Prepare observation batch for model inference.

        Args:
            observation: Current observation dictionary.

        Returns:
            Batch dictionary ready for model input.
        """
        batch = {}

        # Handle observation.state
        if "observation.state" in observation:
            batch["observation.state"] = torch.from_numpy(observation["observation.state"]).unsqueeze(0)
        elif "state" in observation:
            batch["observation.state"] = torch.from_numpy(observation["state"]).unsqueeze(0)

        # Handle observation.images - create individual keys for each camera
        # The ACT model expects: batch["observation.images.{cam_name}"]
        if "observation.images" in observation:
            images_data = observation["observation.images"]

            # Check if it's a dict (multiple cameras)
            if isinstance(images_data, dict):
                # Extract all camera images and create individual keys for each
                for cam_name in sorted(images_data.keys()):
                    img = images_data[cam_name]
                    if img is not None:
                        # Create placeholder for None images
                        batch[f"observation.images.{cam_name}"] = torch.zeros((3, 224, 224))
                    else:
                        # Convert to tensor and normalize: H,W,C -> C,H,W
                        if isinstance(img, np.ndarray):
                            img_tensor = torch.from_numpy(img).float()
                        else:
                            img_tensor = torch.as_tensor(img, dtype=torch.float32)

                        # Normalize and permute: (H,W,C) -> (C,H,W)
                        if img_tensor.dim() == 3:
                            img_tensor = img_tensor.permute(2, 0, 1)
                        img_tensor = img_tensor / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor

            # Check if it's a list/tuple (multiple cameras as list)
            elif isinstance(images_data, (list, tuple)):
                for i, img in enumerate(images_data):
                    cam_name = f"cam_{i}"  # Default camera naming
                    if img is not None:
                        batch[f"observation.images.{cam_name}"] = torch.zeros((3, 224, 224))
                    else:
                        if isinstance(img, np.ndarray):
                            img_tensor = torch.from_numpy(img).float()
                        else:
                            img_tensor = torch.as_tensor(img, dtype=torch.float32)
                        if img_tensor.dim() == 3:
                            img_tensor = img_tensor.permute(2, 0, 1)
                        img_tensor = img_tensor / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor

            # Single numpy array (fallback)
            else:
                if isinstance(images_data, np.ndarray):
                    img_tensor = torch.from_numpy(images_data).float()
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.permute(2, 0, 1)
                    img_tensor = img_tensor / 255.0
                    batch["observation.images"] = img_tensor.unsqueeze(0)
                else:
                    batch["observation.images"] = None
        else:
            batch["observation.images"] = None

        # Handle observation.environment_state if present
        if "observation.environment_state" in observation:
            batch["observation.environment_state"] = torch.from_numpy(
                observation["observation.environment_state"]
            ).unsqueeze(0)

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
            action_tensor: Action tensor from policy output. Can be 1D or 2D.

        Returns:
            Dictionary mapping joint names to position values.
        """
        # Handle different action tensor shapes
        if action_tensor.dim() == 1:
            # (action_dim,) -> get all values
            actions = action_tensor
        elif action_tensor.dim() == 2:
            # (batch, action_dim) -> use first row
            actions = action_tensor[0]
        else:
            logger.error(f"Unexpected action tensor shape: {action_tensor.shape}")
            return {}

        # Get action features from policy config
        if self.policy_model is None:
            logger.error("Policy model not loaded, cannot determine output features")
            return {}

        action_features = self.policy_model.config.output_features

        # Determine joint names to use
        if self._joint_names_from_robot is not None:
            # Use joint names from robot config (matches training data order)
            joint_names = self._joint_names_from_robot
        else:
            # Fallback: use hardcoded joint order
            joint_names = [
                "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
                "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
                "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
                "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
            ]

        # Build action dictionary
        action_dict = {}
        for i, joint_name in enumerate(joint_names):
            if i < actions.shape[0]:
                action_dict[joint_name] = actions[i].item()

        return action_dict

    def get_info(self) -> dict[str, Any]:
        """Get information about the executor state.

        Returns:
            Dictionary with executor information.
        """
        return {
            "policy_loaded": self.policy_model is not None,
            "policy_type": type(self.policy_model).__name__ if self.policy_model else None,
            "device": self.device,
            "observation_buffer_size": len(self.observation_buffer),
            "action_queue_size": len(self._action_queue),
            "joint_names_configured": self._joint_names_from_robot is not None,
        }
