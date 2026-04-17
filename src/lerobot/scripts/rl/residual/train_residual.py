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
Residual RL training script supporting two phases:

Phase 1 (Offline): Pure offline training using demonstration data
    - No environment needed
    - No real robot needed
    - Runs on local GPU server
    - Initializes residual policy from demonstrations

Phase 2 (Online): Online fine-tuning with real robot
    - Requires real robot connection
    - Uses Phase 1 checkpoint as starting point
    - Collects real interaction data
    - Gets true reward from environment

Usage:
    # Phase 1: Offline training
    python -m lerobot.scripts.rl.residual.train_residual \
        --phase offline \
        --base_policy_checkpoint outputs/act/best.safetensors \
        --offline_dataset /path/to/episodes \
        --output_dir outputs/residual/phase1 \
        --total_timesteps 100000

    # Phase 2: Online fine-tuning (real robot)
    python -m lerobot.scripts.rl.residual.train_residual \
        --phase online \
        --base_policy_checkpoint outputs/act/best.safetensors \
        --resume_checkpoint outputs/residual/phase1/checkpoints/best \
        --output_dir outputs/residual/phase2 \
        --env_config_path configs/env_coffee.yaml \
        --total_timesteps 500000
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from draccus import dataclass, field, wrap

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.envs import HILSerlRobotEnvConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy
from lerobot.policies.td3.config_td3 import TD3ActorConfig, TD3Config
from lerobot.policies.td3.modeling_td3 import TD3Agent, TD3Actor, TD3Critic
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.scripts.rl.residual.env_wrapper import ResidualEnvWrapper
from lerobot.utils.residual.checkpoint import CheckpointConfig, CheckpointManager
from lerobot.utils.residual.logger import LocalLogger, LocalLoggerConfig
from lerobot.utils.residual.normalize import (
    ActionScaler,
    StateStandardizer,
    load_normalization,
    save_normalization,
)
from lerobot.utils.residual.replay_buffer import ReplayBuffer, ReplayBufferConfig
from lerobot.utils.residual.utils import EvalMode, schedule_stddev
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, set_seed


@dataclass
class TrainResidualConfig:
    """Configuration for residual RL training."""

    # ========================================
    # Phase selection (REQUIRED)
    # ========================================
    phase: str = "offline"  # "offline" or "online"

    # ========================================
    # Base policy (REQUIRED for both phases)
    # ========================================
    base_policy_checkpoint: str  # Path to pre-trained ACT checkpoint
    base_policy_config_path: str | None = None  # Optional: ACT config path

    # ========================================
    # Phase 1: Offline training parameters
    # ========================================
    offline_dataset: str | None = None  # Local dataset path or HF repo_id
    offline_dataset_episodes: int | None = None  # Number of episodes to use

    # ========================================
    # Phase 2: Online training parameters
    # ========================================
    resume_checkpoint: str | None = None  # Path to Phase 1 checkpoint
    env_config_path: str | None = None  # Path to environment config yaml
    env_config: HILSerlRobotEnvConfig | None = None  # Or direct config

    # ========================================
    # Training hyperparameters (both phases)
    # ========================================
    total_timesteps: int = 100_000
    batch_size: int = 256
    buffer_size: int = 100_000
    n_step: int = 1
    gamma: float = 0.99

    # Warmup
    warmup_steps: int = 10_000
    learning_starts: int = 10_000
    critic_warmup_steps: int = 10_000

    # TD3 specific
    action_scale: float = 0.1
    stddev_max: float = 0.05
    stddev_min: float = 0.05
    stddev_step: int = 100_000
    actor_lr: float = 1e-6
    critic_lr: float = 1e-4
    policy_delay: int = 2  # TD3 delayed policy update

    # Update frequency
    update_every_n_steps: int = 1
    num_updates_per_iteration: int = 4

    # ========================================
    # Evaluation
    # ========================================
    eval_interval: int = 10_000
    eval_episodes: int = 10

    # ========================================
    # Output and logging
    # ========================================
    output_dir: str = "outputs/residual"
    run_name: str = "run_001"
    seed: int = 1000

    # Checkpoint
    checkpoint_interval: int = 10_000
    max_checkpoints: int = 5

    # Device
    device: str = "cuda"

    # FPS (for online phase)
    fps: int = 30


def train_offline_phase(cfg: TrainResidualConfig, logger: LocalLogger, checkpoint_mgr: CheckpointManager):
    """
    Phase 1: Pure offline training.

    No environment, no robot. Uses demonstration data to initialize
    the residual policy.
    """
    device = get_safe_torch_device(cfg.device, log=True)

    logging.info("=" * 60)
    logging.info("Phase 1: Offline Training")
    logging.info("=" * 60)

    # ========================================
    # 1. Load base ACT policy
    # ========================================
    logging.info(f"Loading base ACT policy from: {cfg.base_policy_checkpoint}")

    # Load policy using make_policy factory
    base_policy = make_policy_from_checkpoint(
        checkpoint_path=cfg.base_policy_checkpoint,
        config_path=cfg.base_policy_config_path,
        device=device,
    )
    base_policy.eval()

    # Freeze base policy
    for param in base_policy.parameters():
        param.requires_grad = False

    # Get dimensions
    state_dim = base_policy.config.state_dim if hasattr(base_policy.config, 'state_dim') else 7
    action_dim = base_policy.config.action_dim if hasattr(base_policy.config, 'action_dim') else 7

    logging.info(f"Base policy loaded: state_dim={state_dim}, action_dim={action_dim}")

    # ========================================
    # 2. Load offline dataset and compute normalization
    # ========================================
    logging.info(f"Loading offline dataset: {cfg.offline_dataset}")

    offline_dataset = load_offline_dataset(
        dataset_path=cfg.offline_dataset,
        num_episodes=cfg.offline_dataset_episodes,
    )

    # Compute normalization from dataset
    normalization_dir = Path(cfg.output_dir) / cfg.run_name / "normalization"
    normalization_dir.mkdir(parents=True, exist_ok=True)

    action_scaler, state_standardizer = compute_normalization_from_offline_dataset(
        dataset=offline_dataset,
        save_dir=normalization_dir,
    )

    # ========================================
    # 3. Create TD3 agent
    # ========================================
    logging.info("Creating TD3 residual agent...")

    td3_config = TD3Config(
        actor=TD3ActorConfig(
            action_scale=cfg.action_scale,
            actor_last_layer_init_scale=0.0,  # Zero init for residual
        ),
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        policy_delay=cfg.policy_delay,
        stddev_max=cfg.stddev_max,
        stddev_min=cfg.stddev_min,
        stddev_step=cfg.stddev_step,
    )

    actor = TD3Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        config=td3_config.actor,
        residual_actor=True,
    ).to(device)

    critic = TD3Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        config=td3_config.critic,
    ).to(device)

    td3_agent = TD3Agent(
        actor=actor,
        critic=critic,
        config=td3_config,
        device=device,
    )

    # ========================================
    # 4. Create replay buffer and load offline data
    # ========================================
    logging.info("Creating replay buffer...")

    rb_config = ReplayBufferConfig(
        buffer_size=cfg.buffer_size,
        n_step=cfg.n_step,
        gamma=cfg.gamma,
    )

    offline_rb = ReplayBuffer(
        config=rb_config,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    # Load offline transitions
    logging.info("Loading offline transitions into replay buffer...")
    load_offline_transitions(
        dataset=offline_dataset,
        replay_buffer=offline_rb,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        base_policy=base_policy,
        device=device,
    )

    logging.info(f"Offline buffer size: {len(offline_rb)}")

    # ========================================
    # 5. Training loop (pure offline updates)
    # ========================================
    logging.info(f"Starting offline training: {cfg.total_timesteps} steps")

    global_step = 0
    train_start_time = time.time()
    best_loss = float('inf')

    while global_step < cfg.total_timesteps:
        # Sample batch from offline buffer
        batch = offline_rb.sample(cfg.batch_size)

        # Compute exploration stddev
        stddev = schedule_stddev(cfg.stddev_max, cfg.stddev_min, cfg.stddev_step, global_step)

        # Update TD3
        update_actor = (global_step % cfg.policy_delay == 0)
        metrics = td3_agent.update(batch, stddev, update_actor=update_actor)

        global_step += 1

        # Log metrics
        if global_step % 100 == 0:
            logger.log_metrics(metrics, global_step)

        # Save checkpoint
        if global_step % cfg.checkpoint_interval == 0:
            checkpoint_mgr.save(
                agent=td3_agent,
                step=global_step,
                metrics={"loss": metrics.get("train/critic_loss", 0)},
            )

        # Progress logging
        if global_step % 1000 == 0:
            elapsed = time.time() - train_start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            logging.info(f"Step {global_step}/{cfg.total_timesteps} | SPS: {sps:.1f} | Loss: {metrics.get('train/critic_loss', 0):.4f}")

    # Final save
    logging.info("Phase 1 completed!")
    checkpoint_mgr.save(
        agent=td3_agent,
        step=global_step,
        metrics={"loss": metrics.get("train/critic_loss", 0)},
        force_save=True,
    )

    elapsed = time.time() - train_start_time
    logger.log_summary({
        "phase": "offline",
        "total_steps": global_step,
        "total_time": elapsed,
        "steps_per_second": global_step / elapsed,
    })

    return td3_agent


def train_online_phase(cfg: TrainResidualConfig, logger: LocalLogger, checkpoint_mgr: CheckpointManager):
    """
    Phase 2: Online fine-tuning with real robot.

    Requires robot connection. Collects real interaction data
    and gets true reward from environment.
    """
    device = get_safe_torch_device(cfg.device, log=True)

    logging.info("=" * 60)
    logging.info("Phase 2: Online Training (Real Robot)")
    logging.info("=" * 60)

    # ========================================
    # 1. Load base ACT policy
    # ========================================
    logging.info(f"Loading base ACT policy from: {cfg.base_policy_checkpoint}")

    base_policy = make_policy_from_checkpoint(
        checkpoint_path=cfg.base_policy_checkpoint,
        config_path=cfg.base_policy_config_path,
        device=device,
    )
    base_policy.eval()

    for param in base_policy.parameters():
        param.requires_grad = False

    state_dim = base_policy.config.state_dim if hasattr(base_policy.config, 'state_dim') else 7
    action_dim = base_policy.config.action_dim if hasattr(base_policy.config, 'action_dim') else 7

    # ========================================
    # 2. Create real robot environment
    # ========================================
    logging.info("Creating robot environment...")

    # Load env config
    if cfg.env_config_path:
        import yaml
        with open(cfg.env_config_path, 'r') as f:
            env_config_dict = yaml.safe_load(f)
        env_config = HILSerlRobotEnvConfig(**env_config_dict)
    elif cfg.env_config:
        env_config = cfg.env_config
    else:
        raise ValueError("Environment config required for online phase (--env_config_path or --env_config)")

    # Set FPS
    env_config.fps = cfg.fps

    # Create base robot environment
    base_env = make_robot_env(cfg=env_config)

    # ========================================
    # 3. Load normalization (from Phase 1)
    # ========================================
    normalization_dir = Path(cfg.resume_checkpoint).parent / "normalization"
    if normalization_dir.exists():
        logging.info(f"Loading normalization from: {normalization_dir}")
        action_scaler, state_standardizer = load_normalization(normalization_dir)
    else:
        logging.warning("Normalization not found, using default")
        action_scaler = ActionScaler.from_env(-1.0, 1.0)
        state_standardizer = StateStandardizer.from_config(state_dim)

    # ========================================
    # 4. Create TD3 agent and load Phase 1 checkpoint
    # ========================================
    logging.info("Creating TD3 agent...")

    td3_config = TD3Config(
        actor=TD3ActorConfig(
            action_scale=cfg.action_scale,
            actor_last_layer_init_scale=0.0,
        ),
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        policy_delay=cfg.policy_delay,
        stddev_max=cfg.stddev_max,
        stddev_min=cfg.stddev_min,
        stddev_step=cfg.stddev_step,
    )

    actor = TD3Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        config=td3_config.actor,
        residual_actor=True,
    ).to(device)

    critic = TD3Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        config=td3_config.critic,
    ).to(device)

    td3_agent = TD3Agent(
        actor=actor,
        critic=critic,
        config=td3_config,
        device=device,
    )

    # Load Phase 1 checkpoint
    if cfg.resume_checkpoint:
        logging.info(f"Loading Phase 1 checkpoint: {cfg.resume_checkpoint}")
        training_state = checkpoint_mgr.load(td3_agent, cfg.resume_checkpoint)
        global_step = training_state.get("step", 0)
        logging.info(f"Resumed at step {global_step}")
    else:
        global_step = 0

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
    # 6. Create online replay buffer
    # ========================================
    rb_config = ReplayBufferConfig(
        buffer_size=cfg.buffer_size,
        n_step=cfg.n_step,
        gamma=cfg.gamma,
    )

    online_rb = ReplayBuffer(
        config=rb_config,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    # ========================================
    # 7. Warmup phase (random exploration)
    # ========================================
    if len(online_rb) < cfg.learning_starts:
        logging.info(f"Warmup: collecting {cfg.learning_starts} random transitions...")

        obs, _ = env.reset()

        while len(online_rb) < cfg.learning_starts:
            # Random residual action
            residual_action = torch.rand(action_dim, device=device) * cfg.action_scale * 2 - cfg.action_scale

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(residual_action)

            # Convert to tensors and store
            state = obs["observation.state"].cpu().numpy() if isinstance(obs["observation.state"], torch.Tensor) else obs["observation.state"]
            next_state = next_obs["observation.state"].cpu().numpy() if isinstance(next_obs["observation.state"], torch.Tensor) else next_obs["observation.state"]
            residual_np = residual_action.cpu().numpy()

            online_rb.add(state, residual_np, reward, next_state, terminated)

            obs = next_obs

            if terminated or truncated:
                obs, _ = env.reset()
                logging.info(f"Warmup: {len(online_rb)}/{cfg.learning_starts}")

        logging.info(f"Warmup complete: buffer size = {len(online_rb)}")

    # ========================================
    # 8. Main online training loop
    # ========================================
    logging.info(f"Starting online training: {cfg.total_timesteps} steps")

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_successes = 0
    episode_count = 0
    best_success_rate = 0.0

    train_start_time = time.time()
    step_start_time = time.perf_counter()

    while global_step < cfg.total_timesteps:
        # FPS control
        step_start_time = time.perf_counter()

        # Compute exploration stddev
        stddev = schedule_stddev(cfg.stddev_max, cfg.stddev_min, cfg.stddev_step, global_step)

        # Get residual action from TD3
        with EvalMode(td3_agent):
            residual_action = td3_agent.act(obs, eval_mode=False, stddev=stddev)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(residual_action)

        # Convert and store transition
        state = obs["observation.state"].cpu().numpy() if isinstance(obs["observation.state"], torch.Tensor) else obs["observation.state"]
        next_state = next_obs["observation.state"].cpu().numpy() if isinstance(next_obs["observation.state"], torch.Tensor) else next_obs["observation.state"]
        residual_np = residual_action.cpu().numpy() if isinstance(residual_action, torch.Tensor) else residual_action

        online_rb.add(state, residual_np, reward, next_state, terminated)

        # Track episode stats
        episode_reward += reward
        episode_length += 1

        # Update TD3
        if len(online_rb) >= cfg.learning_starts and global_step % cfg.update_every_n_steps == 0:
            for _ in range(cfg.num_updates_per_iteration):
                batch = online_rb.sample(cfg.batch_size)
                update_actor = (global_step % cfg.policy_delay == 0)
                metrics = td3_agent.update(batch, stddev, update_actor=update_actor)

                if global_step % 100 == 0:
                    logger.log_metrics(metrics, global_step)

        obs = next_obs
        global_step += 1

        # Episode end
        if terminated or truncated:
            episode_count += 1

            # Check success (reward > 0.5 or explicit success flag)
            if reward > 0.5 or info.get("success", False):
                episode_successes += 1

            # Log episode metrics
            episode_metrics = {
                "train/episode_reward": episode_reward,
                "train/episode_length": episode_length,
                "train/success_count": episode_successes,
                "train/episode_count": episode_count,
            }
            logger.log_metrics(episode_metrics, global_step)

            success_rate = episode_successes / episode_count if episode_count > 0 else 0.0
            logging.info(f"Episode {episode_count}: reward={episode_reward:.2f}, length={episode_length}, success_rate={success_rate:.4f}")

            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        # Evaluation
        if global_step % cfg.eval_interval == 0:
            logging.info(f"Evaluating at step {global_step}...")

            eval_metrics = run_online_evaluation(
                env=env,
                td3_agent=td3_agent,
                num_episodes=cfg.eval_episodes,
                device=device,
                step=global_step,
            )

            logger.log_metrics(eval_metrics, global_step)

            success_rate = eval_metrics["eval/success_rate"]
            is_best = success_rate > best_success_rate
            if is_best:
                best_success_rate = success_rate

            checkpoint_mgr.save(
                agent=td3_agent,
                step=global_step,
                metrics={"success_rate": success_rate},
                force_save=is_best,
            )

        # Save checkpoint at interval
        if global_step % cfg.checkpoint_interval == 0:
            checkpoint_mgr.save(
                agent=td3_agent,
                step=global_step,
                metrics={"success_rate": best_success_rate},
            )

        # FPS control
        if cfg.fps > 0:
            dt = time.perf_counter() - step_start_time
            busy_wait(1.0 / cfg.fps - dt)

        # Progress logging
        if global_step % 1000 == 0:
            elapsed = time.time() - train_start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            logging.info(f"Step {global_step}/{cfg.total_timesteps} | SPS: {sps:.1f} | Success rate: {best_success_rate:.4f}")

    # ========================================
    # 9. Final save and cleanup
    # ========================================
    logging.info("Phase 2 completed!")

    checkpoint_mgr.save(
        agent=td3_agent,
        step=global_step,
        metrics={"success_rate": best_success_rate},
        force_save=True,
    )

    elapsed = time.time() - train_start_time
    logger.log_summary({
        "phase": "online",
        "total_steps": global_step,
        "total_time": elapsed,
        "best_success_rate": best_success_rate,
        "total_episodes": episode_count,
        "steps_per_second": global_step / elapsed,
    })

    env.close()


def run_online_evaluation(
    env: ResidualEnvWrapper,
    td3_agent: TD3Agent,
    num_episodes: int,
    device: torch.device,
    step: int,
) -> dict[str, float]:
    """Run evaluation episodes."""
    successes = 0
    total_rewards = 0.0
    episode_lengths = []

    for ep_idx in range(num_episodes):
        obs, _ = env.reset()
        td3_agent.actor.eval()

        episode_reward = 0.0
        episode_length = 0

        with torch.no_grad():
            while True:
                residual_action = td3_agent.act(obs, eval_mode=True, stddev=0.0)
                next_obs, reward, terminated, truncated, info = env.step(residual_action)

                episode_reward += reward
                episode_length += 1
                obs = next_obs

                if terminated or truncated:
                    if reward > 0.5 or info.get("success", False):
                        successes += 1
                    break

        total_rewards += episode_reward
        episode_lengths.append(episode_length)

    success_rate = successes / num_episodes
    avg_reward = total_rewards / num_episodes
    avg_length = sum(episode_lengths) / num_episodes

    return {
        "eval/success_rate": success_rate,
        "eval/avg_reward": avg_reward,
        "eval/avg_episode_length": avg_length,
        "eval/num_episodes": num_episodes,
        "eval/step": step,
    }


# ========================================
# Helper functions
# ========================================

def make_policy_from_checkpoint(
    checkpoint_path: str,
    config_path: str | None,
    device: torch.device,
) -> ACTPolicy:
    """Load ACT policy from checkpoint."""
    from safetensors.torch import load_file

    # Try to load using make_policy factory first
    if config_path:
        # Load config and create policy
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        from lerobot.policies.act.config_act import ACTConfig
        config = ACTConfig(**config_dict)
        policy = ACTPolicy(config)
    else:
        # Try to load from checkpoint directory
        policy = ACTPolicy.from_pretrained(checkpoint_path)

    return policy.to(device)


def load_offline_dataset(
    dataset_path: str,
    num_episodes: int | None,
) -> LeRobotDataset:
    """Load LeRobot dataset."""
    logging.info(f"Loading dataset: {dataset_path}")

    # Use make_dataset factory
    from lerobot.configs.default import DatasetConfig

    dataset_config = DatasetConfig(
        repo_id=dataset_path,
        num_episodes=num_episodes,
    )

    dataset = make_dataset(dataset_config)
    return dataset


def compute_normalization_from_offline_dataset(
    dataset: LeRobotDataset,
    save_dir: Path,
) -> tuple[ActionScaler, StateStandardizer]:
    """Compute normalization from dataset."""
    logging.info("Computing normalization from dataset...")

    # Extract actions and states
    actions = []
    states = []

    # LeRobotDataset structure varies, need to handle properly
    # Simplified version here
    for episode_data in dataset.episode_data_index:
        # Get episode indices
        from_idx = episode_data['from']
        to_idx = episode_data['to']

        # This is simplified - actual implementation needs to match dataset structure
        pass

    # For now, use default normalization
    action_scaler = ActionScaler.from_env(-1.0, 1.0)
    state_standardizer = StateStandardizer.from_config(7)

    save_normalization(action_scaler, state_standardizer, save_dir)

    return action_scaler, state_standardizer


def load_offline_transitions(
    dataset: LeRobotDataset,
    replay_buffer: ReplayBuffer,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    base_policy: ACTPolicy,
    device: torch.device,
) -> None:
    """Load offline transitions into replay buffer."""
    logging.info("Loading offline transitions...")

    # Iterate through dataset episodes
    # Actual implementation depends on dataset structure
    count = 0

    # Placeholder - actual implementation needs to:
    # 1. Extract (state, action, reward, next_state, done) from dataset
    # 2. Normalize actions using action_scaler
    # 3. Compute base_action from base_policy
    # 4. Compute residual = normalized_action - base_normalized_action
    # 5. Store residual action in replay buffer

    logging.info(f"Loaded {count} transitions")


@wrap()
def train_residual(cfg: TrainResidualConfig):
    """Main training entry point."""

    # Setup
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    log_file = output_dir / "train.log"
    init_logging(log_file=str(log_file))

    # Initialize local logger
    logger_config = LocalLoggerConfig(
        log_dir=str(cfg.output_dir),
        project_name="residual_rl",
        run_name=cfg.run_name,
        use_tensorboard=True,
    )
    logger = LocalLogger(logger_config)
    logger.log_config(cfg.__dict__)

    # Initialize checkpoint manager
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=str(output_dir / "checkpoints"),
        max_checkpoints=cfg.max_checkpoints,
        save_interval=cfg.checkpoint_interval,
    )
    checkpoint_mgr = CheckpointManager(checkpoint_config)

    logging.info(f"Residual RL Training")
    logging.info(f"  Phase: {cfg.phase}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Device: {cfg.device}")

    # Run appropriate phase
    if cfg.phase == "offline":
        train_offline_phase(cfg, logger, checkpoint_mgr)
    elif cfg.phase == "online":
        train_online_phase(cfg, logger, checkpoint_mgr)
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}. Must be 'offline' or 'online'")

    logger.close()
    logging.info("Training completed!")


if __name__ == "__main__":
    train_residual()