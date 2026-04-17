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
Residual Policy that combines BC base policy with TD3 residual policy.

Key features:
1. Base policy (ACT) provides stable baseline actions
2. Residual policy (TD3) learns small corrections
3. Actions are combined: base_action + residual_action
4. Safe action composition with clipping
"""

import logging
import torch
from typing import Any

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.residual.config_residual import ResidualPolicyConfig
from lerobot.policies.td3.modeling_td3 import TD3Agent
from lerobot.utils.residual.normalize import ActionScaler, StateStandardizer


class ResidualPolicy:
    """
    Residual policy combining base BC policy and residual TD3 policy.

    This is the core policy class used during training and evaluation.

    Action flow:
    1. Base policy (ACT) outputs base_action from observation
    2. ActionScaler normalizes base_action to [-1, 1] (base_naction)
    3. TD3 actor takes state + base_naction as input
    4. TD3 outputs residual_naction in [-action_scale, action_scale]
    5. Combined: combined_naction = base_naction + residual_naction
    6. ActionScaler unnormalizes to environment action space

    Example:
        # Initialize
        residual_policy = ResidualPolicy(
            base_policy=act_policy,
            td3_agent=td3_agent,
            action_scaler=scaler,
            state_standardizer=standardizer,
            config=config,
        )

        # Select action
        env_action, info = residual_policy.select_action(
            obs={"observation.state": state, "pixels": images},
            eval_mode=True,
        )

        # Send to robot
        robot.send_action(env_action)
    """

    def __init__(
        self,
        base_policy: ACTPolicy,
        td3_agent: TD3Agent,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
        config: ResidualPolicyConfig,
        device: torch.device,
    ):
        """
        Initialize ResidualPolicy.

        Args:
            base_policy: Pre-trained ACT policy
            td3_agent: TD3 agent for residual actions
            action_scaler: Action normalization scaler
            state_standardizer: State standardizer
            config: Residual policy configuration
            device: Device for computation
        """
        self.base_policy = base_policy.to(device)
        self.td3_agent = td3_agent
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.config = config
        self.device = device

        # Freeze base policy if configured
        if config.freeze_base_policy:
            for param in self.base_policy.parameters():
                param.requires_grad = False
            logging.info("Base policy frozen for residual training")

        # Progressive clipping state
        self.current_action_scale = 0.0  # Start from 0
        self.progressive_step = 0

        logging.info(f"ResidualPolicy initialized: action_scale={config.action_scale}")

    def select_action(
        self,
        obs: dict[str, torch.Tensor],
        eval_mode: bool = True,
        stddev: float = 0.05,
        step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Select combined action (base + residual).

        Args:
            obs: Observation dictionary containing state and images
            eval_mode: If True, deterministic action; if False, add exploration noise
            stddev: Exploration noise std (for TD3)
            step: Current training step (for progressive clipping)

        Returns:
            Tuple of (env_action, action_info)
            - env_action: Combined action in environment space
            - action_info: Dictionary with base_naction, residual_naction, combined_naction
        """
        # Compute current action scale (progressive clipping)
        action_scale = self._get_current_action_scale(step)

        with torch.no_grad():
            # 1. Get base policy action
            base_action = self.base_policy.select_action(obs)
            base_naction = self.action_scaler.scale(base_action)

            # 2. Standardize state for TD3
            state = self.state_standardizer.standardize(obs["observation.state"])

            # 3. Build TD3 observation
            td3_obs = {
                "observation.state": state,
                "observation.base_action": base_naction,
            }

            # 4. Get residual action
            residual_naction = self.td3_agent.act(td3_obs, eval_mode, stddev)

            # Apply progressive clipping
            residual_naction = torch.clamp(
                residual_naction,
                -action_scale,
                action_scale,
            )

            # 5. Combine actions
            combined_naction = base_naction + residual_naction

            # Clip combined to [-1, 1] (normalized bounds)
            combined_naction = torch.clamp(combined_naction, -1.0, 1.0)

            # 6. Unnormalize to environment action space
            env_action = self.action_scaler.unscale(combined_naction)

        # Build action info
        action_info = {
            "base_action": base_action,
            "base_naction": base_naction,
            "residual_naction": residual_naction,
            "combined_naction": combined_naction,
            "env_action": env_action,
            "residual_magnitude": torch.abs(residual_naction).mean().item(),
        }

        return env_action, action_info

    def _get_current_action_scale(self, step: int | None) -> float:
        """
        Get current action scale (for progressive clipping).

        If progressive clipping is enabled, action scale increases from 0
        to max over progressive_clipping_steps.

        Args:
            step: Current training step

        Returns:
            Current action scale
        """
        if not self.config.progressive_clipping_enabled or step is None:
            return self.config.action_scale

        # Progressive increase
        progress = min(1.0, step / self.config.progressive_clipping_steps)
        return self.config.action_scale * progress

    def reset(self) -> None:
        """Reset policy state (for base policy with temporal components)."""
        self.base_policy.reset()

    def get_residual_statistics(self) -> dict[str, float]:
        """Get statistics about residual actions (for monitoring)."""
        # This would be computed from recent actions
        # Placeholder for now
        return {
            "residual_scale": self.config.action_scale,
            "progressive_enabled": self.config.progressive_clipping_enabled,
        }


class ResidualPolicyEvaluator:
    """
    Evaluator for residual policy.

    Handles:
    - Evaluation episodes
    - Success rate computation
    - Video recording
    - Logging to LocalLogger
    """

    def __init__(
        self,
        policy: ResidualPolicy,
        env: Any,  # RobotEnv or similar
        action_scaler: ActionScaler,
        logger: Any,  # LocalLogger
    ):
        """
        Initialize evaluator.

        Args:
            policy: ResidualPolicy to evaluate
            env: Environment to evaluate in
            action_scaler: ActionScaler for action processing
            logger: LocalLogger for logging results
        """
        self.policy = policy
        self.env = env
        self.action_scaler = action_scaler
        self.logger = logger

    def evaluate(
        self,
        num_episodes: int = 10,
        save_video: bool = False,
        step: int = 0,
    ) -> dict[str, float]:
        """
        Run evaluation episodes.

        Args:
            num_episodes: Number of episodes to run
            save_video: Whether to save evaluation videos
            step: Current training step (for logging)

        Returns:
            Dictionary with evaluation metrics
        """
        successes = 0
        total_rewards = 0.0
        episode_lengths = []
        residual_magnitudes = []

        for episode_idx in range(num_episodes):
            obs, _ = self.env.reset()
            self.policy.reset()

            episode_reward = 0.0
            episode_length = 0
            episode_residual_magnitudes = []

            while True:
                # Select action
                action, action_info = self.policy.select_action(
                    obs,
                    eval_mode=True,
                    stddev=0.0,
                    step=step,
                )

                # Step environment
                # Note: action might need conversion to numpy
                if isinstance(action, torch.Tensor):
                    action_np = action.cpu().numpy()
                else:
                    action_np = action

                next_obs, reward, terminated, truncated, info = self.env.step(action_np)

                episode_reward += reward
                episode_length += 1
                episode_residual_magnitudes.append(action_info["residual_magnitude"])

                obs = next_obs

                if terminated or truncated:
                    if info.get("success", False) or reward > 0.5:  # Success threshold
                        successes += 1
                    break

            total_rewards += episode_reward
            episode_lengths.append(episode_length)
            residual_magnitudes.extend(episode_residual_magnitudes)

        # Compute metrics
        success_rate = successes / num_episodes
        avg_reward = total_rewards / num_episodes
        avg_length = sum(episode_lengths) / num_episodes
        avg_residual_magnitude = sum(residual_magnitudes) / len(residual_magnitudes)

        metrics = {
            "eval/success_rate": success_rate,
            "eval/avg_reward": avg_reward,
            "eval/avg_episode_length": avg_length,
            "eval/avg_residual_magnitude": avg_residual_magnitude,
            "eval/num_episodes": num_episodes,
        }

        # Log metrics
        self.logger.log_metrics(metrics, step)

        logging.info(f"Evaluation at step {step}: success_rate={success_rate:.4f}, avg_reward={avg_reward:.4f}")

        return metrics