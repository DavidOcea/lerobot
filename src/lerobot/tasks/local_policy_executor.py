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
from typing import Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class LocalPolicyExecutor:
    """Execute policy locally on the robot without using Policy Server.

    This class handles:
    - Converting observation dictionary to model input format
    - Running policy model to get actions
    - Converting action tensors back to dictionary format
    - Applying device transfers and type conversions
    """

    def __init__(
        self,
        policy_model,
        robot_state_feature: str | None = None,
        image_features: list[str] | None = None,
        n_obs_steps: int = 1,
        device: str = "cpu",
    ):
        """Initialize the local policy executor.

        Args:
            policy_model: Trained ACT model for generating actions.
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

    def get_action(
        self,
        observation: dict[str, Any],
    ) -> dict[str, float]:
        """Execute policy and return action dictionary.

        Args:
            observation: Observation dictionary containing sensor data and images.

        Returns:
            Dictionary mapping joint names to position values.
        """
        # Handle nested images structure if present
        if 'images' in observation and isinstance(observation['images'], dict):
            # Collect all camera images into a single list, preserving order
            values = []
            for obs in observation_buffer:
                if 'images' in obs and isinstance(obs['images'], dict):
                    # Extract all camera images from obs, not just head_cam
                    for cam_name in obs['images'].keys():
                        values.append(obs['images'][cam_name])
                else:
                    # No images key, add None placeholder
                    values.append(None)
            # Stack all camera images together with shape (num_cameras, H, W)
            batch["observation.images"] = torch.stack(values).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            # Fallback: create single observation from images
            if 'images' in observation and isinstance(observation['images'], (list, tuple)):
                img = observation['images'][0] if isinstance(observation['images'], list) else observation['images']
                batch["observation.images"] = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            else:
                batch["observation.images"] = None

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
            Dictionary mapping joint names to position values.
        """
        # Get action features from policy config
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

        # Extract actions from model output
        actions = self.policy_model(action_tensor)

        # Build action dictionary
        action_dict = {}
        for i, joint_name in enumerate(joint_names):
            if i < actions.shape[1]:
                action_dict[joint_name] = actions[i].item()

        return action_dict
