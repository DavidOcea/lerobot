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
    - Press 'p' to pause/resume (adjust camera/robot position during training)
    - Resume from any Phase2 checkpoint with --resume_checkpoint

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

    # Phase 2: Resume from previous Phase2 checkpoint
    python -m lerobot.scripts.rl.residual.train_residual \
        --phase online \
        --base_policy_checkpoint outputs/act/best.safetensors \
        --resume_checkpoint outputs/residual/phase2/checkpoints/checkpoint_step_50000 \
        --output_dir outputs/residual/phase2 \
        --env_config_path configs/env_coffee.yaml \
        --total_timesteps 500000

    # During Phase 2 training, press 'p' to:
    #   PAUSE  — robot holds position, adjust camera/robot
    #   RESUME — continue training from paused state
"""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from dataclasses import dataclass, field

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy
from lerobot.policies.td3.config_td3 import TD3ActorConfig, TD3Config
from lerobot.policies.td3.modeling_td3 import TD3Agent, TD3Actor, TD3Critic
from lerobot.envs.utils import preprocess_observation
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
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.random_utils import set_seed


@dataclass
class TrainResidualConfig:
    """Configuration for residual RL training."""

    # ========================================
    # Base policy (REQUIRED for both phases)
    # ========================================
    base_policy_checkpoint: str  # Path to pre-trained ACT checkpoint (REQUIRED)

    # ========================================
    # Phase selection (REQUIRED)
    # ========================================
    phase: str = "offline"  # "offline" or "online"

    # ========================================
    # Base policy config (optional)
    # ========================================
    base_policy_config_path: str | None = None  # Optional: ACT config path

    # ========================================
    # Phase 1: Offline training parameters
    # ========================================
    offline_dataset: str | None = None  # Local dataset path or HF repo_id
    offline_dataset_episodes: int | None = None  # Number of episodes to use
    batch_inference_size: int = 64  # Batch size for ACT inference during data loading

    # ========================================
    # Phase 2: Online training parameters
    # ========================================
    resume_checkpoint: str | None = None  # Path to Phase 1 checkpoint
    env_config_path: str | None = None  # Path to environment config yaml
    env_config: HILSerlRobotEnvConfig | None = None  # Or direct config
    n_action_steps: int = 20  # ACT: how many steps to execute before re-querying (lower = smoother, more GPU work)
    fps: int = 50  # Control frequency for online phase (higher = smoother motion)

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

    # Get dimensions from normalization (computed from actual dataset)
    state_dim = state_standardizer.state_dim
    action_dim = action_scaler.action_dim

    logging.info(f"Dimensions: state_dim={state_dim}, action_dim={action_dim}")

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

    # Load offline transitions with batch inference optimization
    logging.info("Loading offline transitions into replay buffer...")
    load_offline_transitions(
        dataset=offline_dataset,
        replay_buffer=offline_rb,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        base_policy=base_policy,
        device=device,
        batch_inference_size=cfg.batch_inference_size if hasattr(cfg, 'batch_inference_size') else 64,
    )

    logging.info(f"Offline buffer size: {len(offline_rb)}")

    # ========================================
    # 5. Training loop (pure offline updates)
    # ========================================
    logging.info(f"Starting offline training: {cfg.total_timesteps} steps")

    global_step = 0
    train_start_time = time.time()

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

        # Save checkpoint at interval
        # CheckpointManager handles best tracking automatically (loss with min mode)
        if global_step % cfg.checkpoint_interval == 0:
            current_loss = metrics.get("train/critic_loss", float('inf'))
            checkpoint_mgr.save(
                agent=td3_agent,
                step=global_step,
                metrics={"loss": current_loss},
            )

        # Progress logging
        if global_step % 1000 == 0:
            elapsed = time.time() - train_start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            current_loss = metrics.get("train/critic_loss", 0)
            logging.info(
                f"Step {global_step}/{cfg.total_timesteps} | "
                f"SPS: {sps:.1f} | "
                f"Loss: {current_loss:.4f} | "
                f"Best: {checkpoint_mgr.best_metric_value:.6f} (step {checkpoint_mgr.best_step})"
            )

    # ========================================
    # 6. Final save
    # ========================================
    logging.info("Phase 1 completed!")
    logging.info(f"Best loss: {checkpoint_mgr.best_metric_value:.6f} at step {checkpoint_mgr.best_step}")

    # Save final checkpoint (CheckpointManager will update best if needed)
    final_loss = metrics.get("train/critic_loss", 0)
    checkpoint_mgr.save(
        agent=td3_agent,
        step=global_step,
        metrics={"loss": final_loss},
        force_save=True,
    )

    elapsed = time.time() - train_start_time
    logger.log_summary({
        "phase": "offline",
        "total_steps": global_step,
        "total_time": elapsed,
        "steps_per_second": global_step / elapsed,
        "best_loss": checkpoint_mgr.best_metric_value,
        "best_step": checkpoint_mgr.best_step,
    })

    return td3_agent


# ========================================
# Safety validation helpers for Phase2
# ========================================

def _validate_action_scaler(action_scaler):
    """
    Validate loaded ActionScaler parameters for safety.

    Minimal validation - since normalization file was successfully loaded,
    the values represent actual training data and don't need arbitrary threshold checks.

    Only checks:
    1. All action_range values are positive (zero/negative would be broken)
    2. Log action range info for user awareness

    Raises RuntimeError only for truly broken cases (range <= 0).
    """
    logging.info("Validating ActionScaler parameters for safety...")

    action_range = action_scaler.action_range
    action_center = action_scaler.action_center

    # Minimal sanity check: ranges must be positive
    for i, range_val in enumerate(action_range):
        if range_val <= 0:
            raise RuntimeError(
                f"SAFETY ERROR: Action dimension {i} has range {range_val:.2f}° (must be positive).\n"
                f"This indicates a bug in normalization computation.\n"
                f"Refusing to start for safety."
            )

    # Log info for awareness
    logging.info(f"ActionScaler validation passed:")
    logging.info(f"  Action range: {action_range}")
    logging.info(f"  Action center: {action_center}")
    logging.info(
        "Normalization loaded successfully from Phase1 training data. "
        "Action ranges reflect actual joint movement during training."
    )


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
    # 1. Load env config and get dimensions FIRST
    # ========================================
    # We need env dimensions to create TD3 agent correctly
    logging.info("Loading env config...")

    if cfg.env_config_path:
        import draccus
        from lerobot.envs.configs import EnvConfig
        env_config = draccus.parse(
            config_class=EnvConfig,
            config_path=cfg.env_config_path,
            args=[],  # No CLI overrides
        )
    elif cfg.env_config:
        env_config = cfg.env_config
    else:
        raise ValueError("Environment config required for online phase (--env_config_path or --env_config)")

    # Set FPS
    env_config.fps = cfg.fps

    # Get dimensions from env config features
    # Action dim from features.action.shape
    action_dim = env_config.features["action"].shape[0]
    # State dim from features.observation.state.shape (if exists)
    if "observation.state" in env_config.features:
        state_dim = env_config.features["observation.state"].shape[0]
    else:
        state_dim = action_dim  # Fallback to action dim

    logging.info(f"Dimensions from env config: state_dim={state_dim}, action_dim={action_dim}")

    # ========================================
    # 2. Create real robot environment
    # ========================================
    logging.info("Creating robot environment...")
    base_env = make_robot_env(cfg=env_config)

    # ========================================
    # 3. Load base ACT policy (for residual action computation)
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

    # Override ACT's n_action_steps for smoother online execution
    # Original model may have n_action_steps=100 (infer every 3.3s at 30Hz)
    # Lower value means more frequent re-inference = more responsive base action
    original_n_action_steps = base_policy.config.n_action_steps
    if cfg.n_action_steps != original_n_action_steps:
        logging.info(
            f"Overriding ACT n_action_steps: {original_n_action_steps} -> {cfg.n_action_steps} "
            f"(re-inference every {cfg.n_action_steps/cfg.fps:.2f}s instead of {original_n_action_steps/30:.2f}s)"
        )
        base_policy.config.n_action_steps = cfg.n_action_steps
        # Reset action queue with new maxlen
        base_policy._action_queue = deque([], maxlen=cfg.n_action_steps)

    # ========================================
    # 4. Load normalization (from Phase 1)
    # ========================================
    # Checkpoint path structure: phase1/checkpoints/checkpoint_step_*
    # Normalization is at: phase1/normalization/
    normalization_dir = Path(cfg.resume_checkpoint).parent.parent / "normalization"

    if normalization_dir.exists():
        logging.info(f"Loading normalization from: {normalization_dir}")
        action_scaler, state_standardizer = load_normalization(normalization_dir)

        # Safety validation: check if loaded normalization is valid
        _validate_action_scaler(action_scaler)
    else:
        error_msg = (
            f"CRITICAL SAFETY ERROR: Normalization not found at {normalization_dir}!\n"
            f"Without proper normalization, robot actions will have large deviations.\n"
            f"Please ensure Phase1 has saved normalization files.\n"
            f"Refusing to start for safety."
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # ========================================
    # 5. Create TD3 agent and load Phase 1 checkpoint
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
        logging.info(f"Loading checkpoint: {cfg.resume_checkpoint}")
        training_state = checkpoint_mgr.load(td3_agent, cfg.resume_checkpoint)
        global_step = training_state.get("step", 0)
        logging.info(f"Resumed at step {global_step}")
    else:
        global_step = 0

    # ========================================
    # 6. Wrap environment for residual RL
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
    # 7. Create online replay buffer
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

    # Load replay buffer from checkpoint if resuming
    if cfg.resume_checkpoint:
        rb_dir = Path(cfg.resume_checkpoint) / "replay_buffer"
        if rb_dir.exists():
            logging.info(f"Loading replay buffer from: {rb_dir}")
            online_rb.load(rb_dir)
            logging.info(f"Replay buffer loaded: {len(online_rb)} transitions")
        else:
            logging.info(f"No replay buffer found at {rb_dir}, starting with empty buffer")

    # ========================================
    # 8. Warmup phase (random exploration)
    # ========================================
    # Start pause listener BEFORE warmup so user can pause during warmup too
    pause_lock = Lock()
    pause_requested = [False]  # mutable list for cross-thread toggle
    pause_count = 0
    total_pause_time = 0.0
    last_pause_start = 0.0
    was_paused = False

    def _on_key_press(key):
        try:
            from pynput import keyboard as kb
            if hasattr(key, 'char') and key.char == 'p':
                with pause_lock:
                    pause_requested[0] = not pause_requested[0]
        except Exception:
            pass

    try:
        from pynput import keyboard as kb
        pause_listener = kb.Listener(on_press=_on_key_press)
        pause_listener.start()
        logging.info("Pause/resume listener started — press 'p' to pause/resume training")
    except ImportError:
        logging.warning("pynput not installed — pause/resume via keyboard not available")
        pause_listener = None

    logging.info("  Press 'p' to PAUSE (adjust camera/robot), press 'p' again to RESUME")

    if len(online_rb) < cfg.learning_starts:
        logging.info(f"Warmup: collecting {cfg.learning_starts} random transitions...")

        obs, _ = env.reset()

        while len(online_rb) < cfg.learning_starts:
            # Pause check during warmup
            with pause_lock:
                is_paused = pause_requested[0]

            if is_paused:
                if not was_paused:
                    last_pause_start = time.time()
                    logging.info("PAUSED during warmup — robot holding position, press 'p' to resume")
                    was_paused = True
                try:
                    raw_env = env.unwrapped
                    actual_positions = raw_env.robot.get_current_position()
                    raw_env.robot.send_action(actual_positions)
                    raw_env._get_observation()
                except Exception as e:
                    logging.warning(f"Warmup pause loop send_action failed: {e}")
                time.sleep(1.0 / cfg.fps)
                continue

            if was_paused:
                # Resume from warmup pause — re-read observation
                pause_duration = time.time() - last_pause_start
                total_pause_time += pause_duration
                pause_count += 1
                obs, _ = env.reset()
                logging.info(f"RESUMED warmup (pause #{pause_count}, duration: {pause_duration:.1f}s)")
                was_paused = False

            # Random residual action
            residual_action = torch.rand(action_dim, device=device) * cfg.action_scale * 2 - cfg.action_scale

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(residual_action)

            # Convert to tensors and store
            state = obs["observation.state"].cpu().numpy() if isinstance(obs["observation.state"], torch.Tensor) else obs["observation.state"]
            next_state = next_obs["observation.state"].cpu().numpy() if isinstance(next_obs["observation.state"], torch.Tensor) else next_obs["observation.state"]
            residual_np = residual_action.cpu().numpy()

            # Get base_action from env wrapper (stored in info dict)
            base_action_np = info.get("base_action", None)

            online_rb.add(state, residual_np, reward, next_state, terminated, base_action=base_action_np)

            obs = next_obs

            if terminated or truncated:
                obs, _ = env.reset()
                logging.info(f"Warmup: {len(online_rb)}/{cfg.learning_starts}")

        logging.info(f"Warmup complete: buffer size = {len(online_rb)}")

    # ========================================
    # 9. Main online training loop (pause listener already started in section 8)
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

        # ========================================
        # Pause/resume check
        # ========================================
        with pause_lock:
            is_paused = pause_requested[0]

        if is_paused:
            # Entering pause (transition from running to paused)
            if not was_paused:
                last_pause_start = time.time()
                logging.info(
                    f"PAUSED at step {global_step} — "
                    f"robot will hold current position, adjust camera/robot, then press 'p' to resume"
                )
                was_paused = True

            # PAUSED: repeatedly send current joint positions to keep robot still
            # This bypasses ACT + residual entirely — robot stays exactly where it was
            # Re-read actual positions each loop to handle hardware drift
            try:
                raw_env = env.unwrapped
                # Re-read current positions from hardware to counteract any drift
                actual_positions = raw_env.robot.get_current_position()
                # Send same positions back — keys use observation_joint_names (no .pos suffix)
                # robot.send_action() handles .pos removal internally
                raw_env.robot.send_action(actual_positions)
                raw_env._get_observation()  # Refresh observation (camera images update)
            except Exception as e:
                logging.warning(f"Pause loop send_action failed: {e}, retrying next cycle")

            # FPS control during pause
            if cfg.fps > 0:
                dt = time.perf_counter() - step_start_time
                busy_wait(1.0 / cfg.fps - dt)

            continue

        # Exiting pause (transition from paused to running)
        if was_paused:
            pause_duration = time.time() - last_pause_start
            total_pause_time += pause_duration
            pause_count += 1
            # Reset ACT base policy action queue so it starts fresh from current position.
            # Without this, ACT would continue the trajectory from before the pause,
            # causing a sudden jump when resumed.
            env.base_policy.reset()
            # Get fresh observation through the FULL processing chain (not raw env).
            # preprocess_observation does: HWC→CHW, batch dim, float32 norm —
            # without it, ACT crashes with "size 480 vs 3" in normalize_inputs.
            raw_env = env.unwrapped
            raw_env._get_observation()
            raw_obs_hwc = raw_env.current_observation
            processed_obs = preprocess_observation(raw_obs_hwc)
            # Move to same device as ACT model (preprocess_observation outputs CPU tensors)
            for key, val in processed_obs.items():
                if isinstance(val, torch.Tensor):
                    processed_obs[key] = val.to(device)
                elif isinstance(val, list):
                    processed_obs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in val]
            # Now processed_obs has CHW images with batch dim — safe for ACT
            with torch.no_grad():
                base_action = env.base_policy.select_action(processed_obs)
            base_naction = env.action_scaler.scale(base_action)
            # Sync env wrapper's internal state with fresh base action
            env._last_base_naction = base_naction
            # Re-build augmented obs with fresh base action and standardized state
            obs = env._build_residual_obs(processed_obs, base_naction)
            logging.info(
                f"RESUMED at step {global_step} "
                f"(pause #{pause_count}, duration: {pause_duration:.1f}s, "
                f"total pause time: {total_pause_time:.1f}s, "
                f"ACT policy reset and re-synced)"
            )
            was_paused = False

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

        # Get base_action from env wrapper (stored in info dict)
        base_action_np = info.get("base_action", None)

        online_rb.add(state, residual_np, reward, next_state, terminated, base_action=base_action_np)

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
            # Save replay buffer alongside best checkpoint
            best_cp_path = checkpoint_mgr.get_best_checkpoint_path()
            if best_cp_path:
                online_rb.save(best_cp_path / "replay_buffer")

        # Save checkpoint at interval
        if global_step % cfg.checkpoint_interval == 0:
            cp_path = checkpoint_mgr.save(
                agent=td3_agent,
                step=global_step,
                metrics={"success_rate": best_success_rate},
            )
            # Save replay buffer inside checkpoint directory
            if cp_path:
                online_rb.save(cp_path / "replay_buffer")

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
    # 10. Final save and cleanup
    # ========================================
    logging.info("Phase 2 completed!")

    # Log pause statistics
    if pause_count > 0:
        logging.info(f"Pause statistics: {pause_count} pauses, total pause time: {total_pause_time:.1f}s")

    # Stop keyboard listener
    if pause_listener is not None:
        pause_listener.stop()

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
        "pause_count": pause_count,
        "total_pause_time": total_pause_time,
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
    import json
    from pathlib import Path
    from safetensors.torch import load_file

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    checkpoint_path = Path(checkpoint_path)

    # Handle both directory and file paths
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.safetensors':
        checkpoint_dir = checkpoint_path.parent
        model_file = checkpoint_path
    elif checkpoint_path.is_dir():
        checkpoint_dir = checkpoint_path
        model_file = checkpoint_dir / "model.safetensors"
    else:
        raise ValueError(f"Checkpoint path must be a .safetensors file or directory: {checkpoint_path}")

    # Check for config file
    config_file = checkpoint_dir / "config.json"

    # Fields that may be incompatible with current ACTConfig
    incompatible_fields = {
        'use_relative_action', 'only_first_step', 'label_smoothing',
        'use_warmup_cosine_scheduler', 'warmup_steps', 'min_lr_ratio',
        'type',  # 'type' is used for subclass registration, not a direct field
    }

    if config_file.exists():
        # Load config and filter incompatible fields
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        # Remove incompatible fields
        config_dict = {k: v for k, v in config_dict.items() if k not in incompatible_fields}

        # Convert input_features and output_features dicts to PolicyFeature objects
        if 'input_features' in config_dict:
            input_features = {}
            for key, ft_dict in config_dict['input_features'].items():
                ft_type = FeatureType(ft_dict['type'])
                ft_shape = tuple(ft_dict['shape'])
                input_features[key] = PolicyFeature(type=ft_type, shape=ft_shape)
            config_dict['input_features'] = input_features

        if 'output_features' in config_dict:
            output_features = {}
            for key, ft_dict in config_dict['output_features'].items():
                ft_type = FeatureType(ft_dict['type'])
                ft_shape = tuple(ft_dict['shape'])
                output_features[key] = PolicyFeature(type=ft_type, shape=ft_shape)
            config_dict['output_features'] = output_features

        # Convert normalization_mapping if needed
        if 'normalization_mapping' in config_dict:
            norm_mapping = {}
            for key, mode_str in config_dict['normalization_mapping'].items():
                norm_mapping[key] = NormalizationMode(mode_str)
            config_dict['normalization_mapping'] = norm_mapping

        try:
            config = ACTConfig(**config_dict)
        except TypeError as e:
            logging.warning(f"Config loading failed: {e}. Using default config.")
            config = ACTConfig()
    elif config_path:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = ACTConfig(**config_dict)
    else:
        # Use default config
        config = ACTConfig()

    # Create policy with config
    policy = ACTPolicy(config)

    # Load weights
    if model_file.exists():
        state_dict = load_file(model_file, device=str(device))
        policy.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded weights from: {model_file}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_file}")

    return policy.to(device)


def load_offline_dataset(
    dataset_path: str,
    num_episodes: int | None,
) -> LeRobotDataset:
    """Load LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    logging.info(f"Loading dataset: {dataset_path}")

    # Get dataset metadata to check available episodes
    meta = LeRobotDatasetMetadata(dataset_path)
    available_episodes = list(meta.episodes_stats.keys())
    total_available = len(available_episodes)

    logging.info(f"Dataset has {total_available} episodes available")

    # If num_episodes is specified, limit to that number (but not more than available)
    if num_episodes is not None:
        if num_episodes > total_available:
            logging.warning(f"Requested {num_episodes} episodes but only {total_available} available. Using all available.")
            episodes = available_episodes
        else:
            episodes = available_episodes[:num_episodes]
    else:
        episodes = None  # Use all episodes

    logging.info(f"Using episodes: {episodes if episodes else 'all'}")

    dataset = LeRobotDataset(
        repo_id=dataset_path,
        episodes=episodes,
    )
    return dataset


def compute_normalization_from_offline_dataset(
    dataset: LeRobotDataset,
    save_dir: Path,
) -> tuple[ActionScaler, StateStandardizer]:
    """Compute normalization from dataset (using precomputed stats for speed)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    logging.info("Computing normalization from dataset...")

    # Use precomputed stats from dataset metadata (MUCH faster than iterating)
    # LeRobotDataset already has stats computed during dataset creation
    meta = dataset.meta if hasattr(dataset, 'meta') else LeRobotDatasetMetadata(dataset.repo_id, root=dataset.root)
    stats = meta.stats

    # Get action stats
    action_stats = stats.get('action', {})
    if action_stats:
        action_min = np.array(action_stats['min'])
        action_max = np.array(action_stats['max'])
        action_mean = np.array(action_stats['mean'])
        action_std = np.array(action_stats['std'])
        logging.info(f"Using precomputed action stats (shape: {action_min.shape})")
    else:
        logging.warning("No action stats found, using default")
        action_min = np.array([-1.0] * 7)
        action_max = np.array([1.0] * 7)

    # Get state stats
    state_stats = stats.get('observation.state', {})
    if state_stats:
        state_mean = np.array(state_stats['mean'])
        state_std = np.array(state_stats['std'])
        logging.info(f"Using precomputed state stats (shape: {state_mean.shape})")
    else:
        logging.warning("No state stats found, using default")
        state_mean = np.zeros(7)
        state_std = np.ones(7)

    # Create ActionScaler (min-max scaling to [-1, 1])
    # Add small margin to handle values slightly outside dataset range
    margin = (action_max - action_min) * 0.05
    action_min = action_min - margin
    action_max = action_max + margin
    action_scaler = ActionScaler(action_min=action_min, action_max=action_max)

    # Create StateStandardizer (mean-std standardization)
    state_std = np.maximum(state_std, 1e-1)  # Prevent division by zero
    state_standardizer = StateStandardizer(state_mean=state_mean, state_std=state_std)

    save_normalization(action_scaler, state_standardizer, save_dir)

    logging.info(f"Action range: {action_scaler.action_range}")
    logging.info(f"State mean: {state_standardizer.state_mean}")
    logging.info(f"State std: {state_standardizer.state_std}")

    return action_scaler, state_standardizer


def load_offline_transitions(
    dataset: LeRobotDataset,
    replay_buffer: ReplayBuffer,
    action_scaler: ActionScaler,
    state_standardizer: StateStandardizer,
    base_policy: ACTPolicy,
    device: torch.device,
    batch_inference_size: int = 64,
) -> int:
    """Load offline transitions into replay buffer with batch inference optimization.

    For imitation learning datasets (no rewards), we:
    1. Extract (state, action) from each frame
    2. Compute base_action from frozen ACT policy (BATCH inference for speed)
    3. Compute residual = normalized_action - base_normalized_action
    4. Set reward=0 (offline RL with BC-like initialization)
    5. Use next_state from next frame, done at episode end

    Args:
        batch_inference_size: Number of frames to process in one ACT inference batch.
                             Larger = faster but more memory. Default 64 works on most GPUs.

    Returns:
        Number of transitions loaded
    """
    logging.info("Loading offline transitions with batch inference...")
    base_policy.eval()

    # Get episode boundaries from dataset
    episode_data_index = dataset.episode_data_index
    num_episodes = len(dataset.episodes) if hasattr(dataset, 'episodes') else len(episode_data_index['from'])
    logging.info(f"Processing {num_episodes} episodes...")

    count = 0
    all_frames_data = []  # Collect all frames for batch processing

    # Step 1: Collect all frame data first
    for ep_idx in range(num_episodes):
        ep_start = episode_data_index['from'][ep_idx].item()
        ep_end = episode_data_index['to'][ep_idx].item()

        for frame_idx in range(ep_start, ep_end - 1):  # Skip last frame (no next_state)
            item = dataset[frame_idx]
            next_item = dataset[frame_idx + 1]

            # Extract data
            state = item['observation.state'].cpu().numpy()
            action = item['action'].cpu().numpy()
            next_state = next_item['observation.state'].cpu().numpy()
            done = (frame_idx == ep_end - 2)

            # Build observation dict for ACT
            obs_dict = {}
            for key in item.keys():
                if key.startswith('observation.') or 'cam' in key.lower():
                    obs_dict[key] = item[key]

            all_frames_data.append({
                'frame_idx': frame_idx,
                'ep_idx': ep_idx,
                'state': state,
                'action': action,
                'next_state': next_state,
                'done': done,
                'obs_dict': obs_dict,
            })

    total_frames = len(all_frames_data)
    logging.info(f"Collected {total_frames} frames, computing base actions in batches...")

    # Step 2: Batch compute base_actions using predict_action_chunk (bypass action queue)
    # IMPORTANT: Use predict_action_chunk directly, not select_action!
    # select_action has an action_queue mechanism that breaks batch inference.
    # predict_action_chunk returns (batch_size, n_action_steps, action_dim)
    # For offline data, we take the first action in each chunk [:, 0, :]
    all_base_actions = []
    for batch_start in range(0, total_frames, batch_inference_size):
        batch_end = min(batch_start + batch_inference_size, total_frames)
        batch_frames = all_frames_data[batch_start:batch_end]

        # Build batched observation dict
        batched_obs = {}
        for key in batch_frames[0]['obs_dict'].keys():
            tensors = [f['obs_dict'][key] for f in batch_frames]
            # Stack and move to device
            batched_obs[key] = torch.stack(tensors).to(device)

        # Batch inference using predict_action_chunk (bypass action queue)
        with torch.no_grad():
            # predict_action_chunk returns (batch_size, n_action_steps, action_dim)
            action_chunks = base_policy.predict_action_chunk(batched_obs)
            # Take the first action from each chunk for offline data
            batch_base_actions = action_chunks[:, 0, :]  # (batch_size, action_dim)
            batch_base_actions = batch_base_actions.cpu()

        all_base_actions.extend(batch_base_actions)

        if (batch_end) % 500 == 0 or batch_end == total_frames:
            logging.info(f"Computed base actions for {batch_end}/{total_frames} frames...")

    # Step 3: Process and add to replay buffer
    logging.info("Computing residuals and adding to replay buffer...")

    for i, frame_data in enumerate(all_frames_data):
        state = frame_data['state']
        action = frame_data['action']
        next_state = frame_data['next_state']
        done = frame_data['done']
        base_action = all_base_actions[i]

        # Normalize
        normalized_action = action_scaler.scale(torch.from_numpy(action).float()).numpy()
        normalized_base_action = action_scaler.scale(base_action).numpy()

        # Compute residual
        residual_action = normalized_action - normalized_base_action

        # Standardize states
        standardized_state = state_standardizer.standardize(torch.from_numpy(state).float()).numpy()
        standardized_next_state = state_standardizer.standardize(torch.from_numpy(next_state).float()).numpy()

        # Add to buffer
        replay_buffer.add(
            state=standardized_state,
            action=residual_action,
            reward=0.0,
            next_state=standardized_next_state,
            done=done,
            base_action=normalized_base_action,
        )
        count += 1

    logging.info(f"Total loaded: {count} transitions")
    return count


@draccus.wrap()
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
    # Phase 1 (offline): Track loss (minimize)
    # Phase 2 (online): Track success_rate (maximize)
    if cfg.phase == "offline":
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(output_dir / "checkpoints"),
            max_checkpoints=cfg.max_checkpoints,
            save_interval=cfg.checkpoint_interval,
            best_metric_key="loss",
            best_metric_mode="min",  # Lower loss is better
        )
    else:  # online phase
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(output_dir / "checkpoints"),
            max_checkpoints=cfg.max_checkpoints,
            save_interval=cfg.checkpoint_interval,
            best_metric_key="success_rate",
            best_metric_mode="max",  # Higher success rate is better
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