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
Checkpoint management for residual RL training.

Features:
- Save/load actor and critic weights using safetensors
- Track best model based on success rate
- Resume training from any checkpoint
- Automatic cleanup of old checkpoints
"""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


@dataclass
class CheckpointConfig:
    """Configuration for CheckpointManager."""

    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 5  # Maximum number of checkpoints to keep
    save_best_only: bool = False  # If True, only save when metric improves
    save_interval: int = 10_000  # Save checkpoint every N steps

    # New fields for flexible best tracking
    best_metric_key: str = "success_rate"  # Metric key to track for best checkpoint
    best_metric_mode: str = "max"  # "max" for maximize (success_rate), "min" for minimize (loss)


class CheckpointManager:
    """
    Manages model checkpoints for residual RL training.

    Handles saving, loading, and tracking of model weights and training state.
    Uses safetensors for efficient and safe tensor storage.

    Example:
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=Path("outputs/checkpoints"),
            max_checkpoints=5,
        )

        # Save checkpoint
        checkpoint_mgr.save(
            agent=td3_agent,
            step=10000,
            metrics={"success_rate": 0.75},
        )

        # Load checkpoint for resume
        state = checkpoint_mgr.load(
            agent=td3_agent,
            checkpoint_name="best",
        )
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track best metric value
        # Initialize based on mode: max -> 0.0 (or -inf for loss), min -> inf
        if config.best_metric_mode == "max":
            self.best_metric_value = 0.0
        else:
            self.best_metric_value = float('inf')
        self.best_step = 0

        # Load best metrics from existing checkpoints if available
        self._load_best_metrics()

        logging.info(f"CheckpointManager initialized at: {self.checkpoint_dir}")
        logging.info(f"Tracking best metric: {config.best_metric_key} (mode: {config.best_metric_mode})")

    def save(
        self,
        agent: Any,  # TD3Agent type
        step: int,
        metrics: dict[str, float],
        force_save: bool = False,
    ) -> Path | None:
        """
        Save a checkpoint.

        Args:
            agent: TD3Agent instance with actor, critic, critic_target
            step: Current training step
            metrics: Dictionary of metrics (e.g., {"success_rate": 0.75, "loss": 0.01})
            force_save: Force save even if save_best_only=True

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        # Get the metric to track for best
        metric_key = self.config.best_metric_key
        metric_mode = self.config.best_metric_mode
        current_metric = metrics.get(metric_key, 0.0 if metric_mode == "max" else float('inf'))

        # Check if should save
        should_save = False

        # Determine if this is a new best
        if metric_mode == "max":
            is_best = current_metric > self.best_metric_value
        else:  # "min"
            is_best = current_metric < self.best_metric_value

        # Always save if it's a new best
        if is_best:
            should_save = True

        # Save at interval if not save_best_only
        if not self.config.save_best_only:
            if step % self.config.save_interval == 0:
                should_save = True

        # Force save
        if force_save:
            should_save = True

        if not should_save:
            return None

        # Create checkpoint directory
        checkpoint_name = f"checkpoint_step_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        save_file(agent.actor.state_dict(), checkpoint_path / "actor.safetensors")
        save_file(agent.critic.state_dict(), checkpoint_path / "critic.safetensors")
        save_file(agent.critic_target.state_dict(), checkpoint_path / "critic_target.safetensors")

        # Save optimizer states
        torch.save(agent.actor_optimizer.state_dict(), checkpoint_path / "actor_optimizer.pt")
        torch.save(agent.critic_optimizer.state_dict(), checkpoint_path / "critic_optimizer.pt")

        # Save training state
        training_state = {
            "step": step,
            metric_key: current_metric,
            f"best_{metric_key}": self.best_metric_value,
            "best_step": self.best_step,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")

        # Update best if this is best checkpoint
        if is_best:
            self.best_metric_value = current_metric
            self.best_step = step
            self._save_best_checkpoint(agent, step, metrics)
            logging.info(f"🎉 New best {metric_key}: {current_metric:.6f} at step {step}")

        logging.info(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        agent: Any,  # TD3Agent type
        checkpoint_name: str = "best",
    ) -> dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            agent: TD3Agent instance to load weights into
            checkpoint_name: Name of checkpoint ("best" or "checkpoint_step_N")

        Returns:
            Dictionary with loaded training state
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model weights
        actor_weights = load_file(checkpoint_path / "actor.safetensors")
        critic_weights = load_file(checkpoint_path / "critic.safetensors")
        critic_target_weights = load_file(checkpoint_path / "critic_target.safetensors")

        agent.actor.load_state_dict(actor_weights)
        agent.critic.load_state_dict(critic_weights)
        agent.critic_target.load_state_dict(critic_target_weights)

        # Load optimizer states if available
        optimizer_path = checkpoint_path / "actor_optimizer.pt"
        if optimizer_path.exists():
            agent.actor_optimizer.load_state_dict(torch.load(optimizer_path))

        optimizer_path = checkpoint_path / "critic_optimizer.pt"
        if optimizer_path.exists():
            agent.critic_optimizer.load_state_dict(torch.load(optimizer_path))

        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt")

        # Update best metrics
        metric_key = self.config.best_metric_key
        self.best_metric_value = training_state.get(f"best_{metric_key}",
            0.0 if self.config.best_metric_mode == "max" else float('inf'))
        self.best_step = training_state.get("best_step", 0)

        logging.info(f"Checkpoint loaded: {checkpoint_path}")
        logging.info(f"  Step: {training_state['step']}")
        if metric_key in training_state:
            logging.info(f"  {metric_key}: {training_state[metric_key]:.6f}")
        if f"best_{metric_key}" in training_state:
            logging.info(f"  Best {metric_key}: {self.best_metric_value:.6f}")

        return training_state

    def load_actor_only(
        self,
        actor: torch.nn.Module,
        checkpoint_name: str = "best",
    ) -> None:
        """
        Load only actor weights (for evaluation).

        Args:
            actor: Actor network to load weights into
            checkpoint_name: Name of checkpoint
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        actor_weights = load_file(checkpoint_path / "actor.safetensors")
        actor.load_state_dict(actor_weights)

        logging.info(f"Actor weights loaded from: {checkpoint_path}")

    def get_best_checkpoint_path(self) -> Path:
        """Get path to best checkpoint."""
        return self.checkpoint_dir / "best"

    def get_latest_checkpoint_path(self) -> Path | None:
        """Get path to latest checkpoint (by step number)."""
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))

    def get_all_checkpoints(self) -> list[Path]:
        """Get list of all checkpoint paths."""
        return self._list_checkpoints()

    def _save_best_checkpoint(
        self,
        agent: Any,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Save/update the best checkpoint."""
        best_path = self.checkpoint_dir / "best"
        best_path.mkdir(parents=True, exist_ok=True)

        # Copy weights
        save_file(agent.actor.state_dict(), best_path / "actor.safetensors")
        save_file(agent.critic.state_dict(), best_path / "critic.safetensors")
        save_file(agent.critic_target.state_dict(), best_path / "critic_target.safetensors")

        # Copy optimizer states
        torch.save(agent.actor_optimizer.state_dict(), best_path / "actor_optimizer.pt")
        torch.save(agent.critic_optimizer.state_dict(), best_path / "critic_optimizer.pt")

        # Save training state
        metric_key = self.config.best_metric_key
        training_state = {
            "step": step,
            metric_key: metrics.get(metric_key, 0.0),
            f"best_{metric_key}": self.best_metric_value,
            "best_step": self.best_step,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        torch.save(training_state, best_path / "training_state.pt")

        logging.info(f"Best checkpoint updated: {best_path}")

    def _load_best_metrics(self) -> None:
        """Load best metrics from existing best checkpoint."""
        best_path = self.checkpoint_dir / "best"
        state_path = best_path / "training_state.pt"

        if state_path.exists():
            training_state = torch.load(state_path)
            metric_key = self.config.best_metric_key
            self.best_metric_value = training_state.get(f"best_{metric_key}",
                0.0 if self.config.best_metric_mode == "max" else float('inf'))
            self.best_step = training_state.get("best_step", 0)
            logging.info(f"Loaded best metrics: {metric_key}={self.best_metric_value:.6f}, step={self.best_step}")

    def _list_checkpoints(self) -> list[Path]:
        """List all checkpoint directories (excluding 'best')."""
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint_step_"):
                checkpoints.append(path)
        return sorted(checkpoints, key=lambda p: int(p.name.split("_")[-1]))

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to keep only max_checkpoints."""
        checkpoints = self._list_checkpoints()

        while len(checkpoints) > self.config.max_checkpoints:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            logging.info(f"Removed old checkpoint: {oldest}")

    def delete_all(self) -> None:
        """Delete all checkpoints (use with caution)."""
        for checkpoint in self._list_checkpoints():
            shutil.rmtree(checkpoint)

        best_path = self.checkpoint_dir / "best"
        if best_path.exists():
            shutil.rmtree(best_path)

        # Reset best metrics based on mode
        if self.config.best_metric_mode == "max":
            self.best_metric_value = 0.0
        else:
            self.best_metric_value = float('inf')
        self.best_step = 0

        logging.info("All checkpoints deleted")