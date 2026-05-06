# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You can obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone inference script for Phase2 residual RL policy.

Loads both ACT base policy and TD3 residual actor, combines their outputs
via ResidualEnvWrapper, and runs deterministic episodes on real robot.

Usage:
    python -m lerobot.scripts.rl.residual.eval_residual \
        --env_config_path configs/env_coffee.yaml \
        --base_policy_checkpoint outputs/train/act_0418_1/checkpoints/040000/pretrained_model \
        --td3_checkpoint outputs/residual/phase2/checkpoints/best \
        --num_episodes 10
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch

from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.td3.config_td3 import TD3ActorConfig, TD3Config
from lerobot.policies.td3.modeling_td3 import TD3Actor, TD3Agent, TD3Critic
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.scripts.rl.residual.env_wrapper import ResidualEnvWrapper
from lerobot.utils.residual.checkpoint import CheckpointManager, CheckpointConfig
from lerobot.utils.residual.normalize import load_normalization
from lerobot.utils.residual.utils import EvalMode
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class EvalResidualConfig:
    """Configuration for residual RL inference."""

    # Policies (required, no defaults - must come first)
    base_policy_checkpoint: str  # ACT model checkpoint path
    td3_checkpoint: str  # TD3 checkpoint dir (e.g., "outputs/residual/phase2/checkpoints/best")

    # Env config
    env_config_path: str | None = None  # Path to env YAML config
    env_config: HILSerlRobotEnvConfig | None = None  # Or direct config
    base_policy_config_path: str | None = None  # Optional ACT config.json path
    normalization_dir: str | None = None  # Auto-derived from td3_checkpoint if None

    # Inference parameters
    action_scale: float = 0.1  # Must match training config
    n_action_steps: int = 20  # ACT: steps per inference cycle
    fps: int = 50  # Control frequency
    num_episodes: int = 10  # Number of test episodes
    device: str = "cuda"


def make_policy_from_checkpoint(
    checkpoint_path: str,
    config_path: str | None,
    device: torch.device,
    env_config: "HILSerlRobotEnvConfig | None" = None,
) -> ACTPolicy:
    """Load ACT policy from checkpoint.

    If checkpoint config.json lacks input_features, derives them from env_config.features.
    """
    import json
    from safetensors.torch import load_file

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_file() and checkpoint_path.suffix == '.safetensors':
        checkpoint_dir = checkpoint_path.parent
        model_file = checkpoint_path
    elif checkpoint_path.is_dir():
        checkpoint_dir = checkpoint_path
        model_file = checkpoint_dir / "model.safetensors"
    else:
        raise ValueError(f"Checkpoint path must be a .safetensors file or directory: {checkpoint_path}")

    config_file = checkpoint_dir / "config.json"

    incompatible_fields = {
        'use_relative_action', 'only_first_step', 'label_smoothing',
        'use_warmup_cosine_scheduler', 'warmup_steps', 'min_lr_ratio',
        'type',
    }

    config = None

    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        config_dict = {k: v for k, v in config_dict.items() if k not in incompatible_fields}

        if 'input_features' in config_dict:
            input_features = {}
            for key, ft_dict in config_dict['input_features'].items():
                ft_type = FeatureType(ft_dict['type'])
                ft_shape = tuple(ft_dict['shape'])
                input_features[key] = PolicyFeature(type=ft_type, shape=ft_shape)
            config_dict['input_features'] = input_features
        elif env_config and env_config.features:
            # Derive input_features from env config (STATE + VISUAL keys only)
            input_features = {}
            for key, feat in env_config.features.items():
                if feat.type in (FeatureType.STATE, FeatureType.VISUAL):
                    input_features[key] = PolicyFeature(type=feat.type, shape=tuple(feat.shape))
            config_dict['input_features'] = input_features
            logging.info(f"Derived input_features from env_config: {list(input_features.keys())}")

        if 'output_features' in config_dict:
            output_features = {}
            for key, ft_dict in config_dict['output_features'].items():
                ft_type = FeatureType(ft_dict['type'])
                ft_shape = tuple(ft_dict['shape'])
                output_features[key] = PolicyFeature(type=ft_type, shape=ft_shape)
            config_dict['output_features'] = output_features
        elif env_config and env_config.features:
            output_features = {}
            for key, feat in env_config.features.items():
                if feat.type == FeatureType.ACTION:
                    output_features[key] = PolicyFeature(type=feat.type, shape=tuple(feat.shape))
            config_dict['output_features'] = output_features

        if 'normalization_mapping' in config_dict:
            norm_mapping = {}
            for key, mode_str in config_dict['normalization_mapping'].items():
                norm_mapping[key] = NormalizationMode(mode_str)
            config_dict['normalization_mapping'] = norm_mapping

        try:
            config = ACTConfig(**config_dict)
        except TypeError as e:
            logging.warning(f"Config loading failed: {e}. Will build from env_config.")
    elif config_path:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        try:
            config = ACTConfig(**config_dict)
        except TypeError as e:
            logging.warning(f"Config path loading failed: {e}.")

    if config is None or (not config.input_features and env_config):
        # Build ACTConfig from env_config.features when checkpoint config is missing/incomplete
        logging.info("Building ACTConfig from env_config.features...")
        input_features = {}
        for key, feat in env_config.features.items():
            if feat.type in (FeatureType.STATE, FeatureType.VISUAL):
                input_features[key] = PolicyFeature(type=feat.type, shape=tuple(feat.shape))
        output_features = {}
        for key, feat in env_config.features.items():
            if feat.type == FeatureType.ACTION:
                output_features[key] = PolicyFeature(type=feat.type, shape=tuple(feat.shape))
        config = ACTConfig(input_features=input_features, output_features=output_features)

    policy = ACTPolicy(config)

    if model_file.exists():
        state_dict = load_file(model_file, device=str(device))
        policy.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded weights from: {model_file}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_file}")

    return policy.to(device)


def resolve_td3_checkpoint(cfg: EvalResidualConfig) -> tuple[str, str]:
    """Resolve td3_checkpoint path into (checkpoint_dir, checkpoint_name)."""
    td3_path = Path(cfg.td3_checkpoint)
    if td3_path.is_dir():
        # Full path like outputs/.../checkpoints/best
        return str(td3_path.parent), td3_path.name
    else:
        # Just a directory containing checkpoints
        return str(td3_path), "best"


@draccus.wrap()
def run_residual_inference(cfg: EvalResidualConfig):
    """Run trained residual RL policy on real robot (deterministic, no exploration noise)."""
    init_logging()
    device = get_safe_torch_device(cfg.device, log=True)

    logging.info("=" * 60)
    logging.info("Residual RL Inference (ACT Base + TD3 Residual)")
    logging.info("=" * 60)

    # ========================================
    # 1. Load env config and create environment
    # ========================================
    if cfg.env_config_path:
        from lerobot.envs.configs import EnvConfig
        env_config = draccus.parse(
            config_class=EnvConfig,
            config_path=cfg.env_config_path,
            args=[],
        )
    elif cfg.env_config:
        env_config = cfg.env_config
    else:
        raise ValueError("Environment config required (--env_config_path or --env_config)")

    env_config.fps = cfg.fps

    action_dim = env_config.features["action"].shape[0]
    state_dim = env_config.features["observation.state"].shape[0] if "observation.state" in env_config.features else action_dim

    logging.info(f"Dimensions: state_dim={state_dim}, action_dim={action_dim}")

    base_env = make_robot_env(cfg=env_config)

    # ========================================
    # 2. Load ACT base policy
    # ========================================
    logging.info(f"Loading ACT base policy from: {cfg.base_policy_checkpoint}")

    base_policy = make_policy_from_checkpoint(
        checkpoint_path=cfg.base_policy_checkpoint,
        config_path=cfg.base_policy_config_path,
        device=device,
        env_config=env_config,
    )
    base_policy.eval()
    for param in base_policy.parameters():
        param.requires_grad = False

    # Override n_action_steps for responsive execution
    original_n = base_policy.config.n_action_steps
    if cfg.n_action_steps != original_n:
        logging.info(f"Overriding ACT n_action_steps: {original_n} -> {cfg.n_action_steps}")
        base_policy.config.n_action_steps = cfg.n_action_steps
        base_policy._action_queue = deque([], maxlen=cfg.n_action_steps)

    # ========================================
    # 3. Load normalization
    # ========================================
    checkpoint_dir, checkpoint_name = resolve_td3_checkpoint(cfg)

    if cfg.normalization_dir:
        normalization_dir = Path(cfg.normalization_dir)
    else:
        # Derive from checkpoint structure: checkpoints/best -> ../normalization
        normalization_dir = Path(checkpoint_dir).parent / "normalization"

    if normalization_dir.exists():
        logging.info(f"Loading normalization from: {normalization_dir}")
        action_scaler, state_standardizer = load_normalization(normalization_dir)
    else:
        raise RuntimeError(
            f"Normalization not found at {normalization_dir}!\n"
            f"Refusing to start — mismatched normalization causes dangerous robot movements."
        )

    # ========================================
    # 4. Create TD3 actor and load weights
    # ========================================
    logging.info("Creating TD3 residual actor...")

    td3_actor_config = TD3ActorConfig(
        action_scale=cfg.action_scale,
        actor_last_layer_init_scale=0.0,
    )

    td3_actor = TD3Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        config=td3_actor_config,
        residual_actor=True,
    ).to(device)

    checkpoint_mgr = CheckpointManager(CheckpointConfig(checkpoint_dir=checkpoint_dir))
    checkpoint_mgr.load_actor_only(td3_actor, checkpoint_name=checkpoint_name)
    td3_actor.eval()

    # TD3Agent wrapper needed because its .act() method extracts obs keys for the actor
    dummy_critic = TD3Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        config=TD3Config().critic,
    ).to(device)

    td3_agent = TD3Agent(
        actor=td3_actor,
        critic=dummy_critic,
        config=TD3Config(actor=td3_actor_config),
        device=device,
    )

    # ========================================
    # 5. Wrap environment for residual RL
    # ========================================
    env = ResidualEnvWrapper(
        env=base_env,
        base_policy=base_policy,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        action_scale=cfg.action_scale,
        device=device,
    )

    # ========================================
    # 6. Inference loop
    # ========================================
    logging.info(f"Starting inference: {cfg.num_episodes} episodes")
    logging.info(f"  ACT base: {cfg.base_policy_checkpoint}")
    logging.info(f"  TD3 residual: {checkpoint_dir}/{checkpoint_name}")
    logging.info(f"  action_scale={cfg.action_scale}, n_action_steps={cfg.n_action_steps}")

    successes = 0
    total_rewards = 0.0
    episode_lengths = []

    for ep_idx in range(cfg.num_episodes):
        logging.info(f"\n--- Episode {ep_idx + 1}/{cfg.num_episodes} ---")

        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        with EvalMode(td3_agent):
            while True:
                residual_action = td3_agent.act(obs, eval_mode=True, stddev=0.0)
                next_obs, reward, terminated, truncated, info = env.step(residual_action)

                episode_reward += reward
                episode_length += 1
                obs = next_obs

                # Periodic monitoring log
                if episode_length % 50 == 0:
                    base_act = info.get("base_action")
                    res_act = info.get("residual_action")
                    logging.info(
                        f"  Step {episode_length}: reward={reward:.3f}, "
                        f"base_norm_magnitude={_norm_magnitude(base_act):.3f}, "
                        f"res_norm_magnitude={_norm_magnitude(res_act):.3f}"
                    )

                if terminated or truncated:
                    is_success = reward > 0.5 or info.get("success", False)
                    if is_success:
                        successes += 1
                    logging.info(
                        f"  Episode {ep_idx + 1}: length={episode_length}, "
                        f"reward={episode_reward:.3f}, success={is_success}"
                    )
                    break

        total_rewards += episode_reward
        episode_lengths.append(episode_length)

    # ========================================
    # 7. Summary
    # ========================================
    success_rate = successes / cfg.num_episodes
    avg_reward = total_rewards / cfg.num_episodes
    avg_length = sum(episode_lengths) / cfg.num_episodes

    logging.info("\n" + "=" * 60)
    logging.info("Inference Results")
    logging.info("=" * 60)
    logging.info(f"  Success rate: {success_rate:.1%} ({successes}/{cfg.num_episodes})")
    logging.info(f"  Avg reward:   {avg_reward:.3f}")
    logging.info(f"  Avg length:   {avg_length:.1f} steps")
    logging.info(f"  Episode lengths: {episode_lengths}")

    env.close()

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_episode_length": avg_length,
    }


def _norm_magnitude(action) -> float:
    """Compute L2 norm of an action array (for monitoring logs)."""
    if action is None:
        return 0.0
    import numpy as np
    if isinstance(action, np.ndarray):
        return float(np.linalg.norm(action))
    return 0.0


if __name__ == "__main__":
    run_residual_inference()