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
Local logging system for residual RL training.
Replaces wandb dependency with fully local storage.

Features:
- JSONL metrics log (easy to parse and analyze)
- Optional TensorBoard support for visualization
- Artifact storage (videos, images, checkpoints)
- No external network dependency
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LocalLoggerConfig:
    """Configuration for LocalLogger."""

    log_dir: str = "outputs/residual_logs"
    project_name: str = "residual_rl"
    run_name: str = "run_001"

    # Optional TensorBoard
    use_tensorboard: bool = False

    # Log frequency control
    log_every_n_steps: int = 100
    print_every_n_steps: int = 1000

    # Artifact settings
    save_videos: bool = True
    save_images: bool = True


class LocalLogger:
    """
    Local logging system that replaces wandb.

    Stores all training logs, configs, and artifacts locally.
    Supports optional TensorBoard for visualization.

    Example:
        logger = LocalLogger(
            log_dir=Path("outputs"),
            project_name="coffee_residual",
            run_name="run_001",
        )
        logger.log_config({"lr": 1e-4, "batch_size": 256})
        logger.log_metrics({"train/loss": 0.5, "eval/success": 0.8}, step=1000)
        logger.log_summary({"best_success_rate": 0.95})
    """

    def __init__(self, config: LocalLoggerConfig):
        self.config = config

        # Create log directory structure
        self.log_dir = Path(config.log_dir) / config.project_name / config.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.metrics_dir = self.log_dir / "metrics"
        self.artifacts_dir = self.log_dir / "artifacts"
        self.metrics_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

        # File paths
        self.metrics_file = self.metrics_dir / "metrics.jsonl"
        self.config_file = self.log_dir / "config.json"
        self.summary_file = self.log_dir / "summary.json"

        # Initialize TensorBoard if enabled
        self.tb_writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(exist_ok=True)
                self.tb_writer = SummaryWriter(str(tb_dir))
                logging.info(f"TensorBoard logging enabled: {tb_dir}")
            except ImportError:
                logging.warning("TensorBoard not available, disabling")
                self.tb_writer = None

        # Track logged steps for deduplication
        self._logged_steps: set[int] = set()
        self._last_print_step = 0

        logging.info(f"LocalLogger initialized at: {self.log_dir}")

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Save training configuration.

        Args:
            config: Configuration dictionary to save
        """
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logging.info(f"Config saved to: {self.config_file}")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        force_log: bool = False,
    ) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
            force_log: Force logging even if step already logged
        """
        # Check if should log based on frequency
        if not force_log and step in self._logged_steps:
            return

        if not force_log and step % self.config.log_every_n_steps != 0:
            return

        self._logged_steps.add(step)

        # Write to JSONL file
        entry = {
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics,
        }
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

        # Write to TensorBoard if enabled
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            self.tb_writer.add_scalar(f"{key}/{sub_key}", sub_value, step)

        # Print progress periodically
        if step - self._last_print_step >= self.config.print_every_n_steps:
            self._print_metrics(metrics, step)
            self._last_print_step = step

    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log final training summary.

        Args:
            summary: Summary dictionary with final results
        """
        # Update summary file
        existing_summary = {}
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                existing_summary = json.load(f)

        existing_summary.update(summary)
        existing_summary["final_timestamp"] = time.time()

        with open(self.summary_file, 'w') as f:
            json.dump(existing_summary, f, indent=2)

        logging.info(f"Summary saved to: {self.summary_file}")

    def save_artifact(
        self,
        file_path: Path | str,
        artifact_name: str,
        artifact_type: str = "misc",
    ) -> Path:
        """
        Save an artifact file (video, image, etc).

        Args:
            file_path: Path to the file to save
            artifact_name: Name for the artifact
            artifact_type: Type subdirectory (e.g., "videos", "images")

        Returns:
            Path where artifact was saved
        """
        source_path = Path(file_path)
        if not source_path.exists():
            logging.warning(f"Artifact source not found: {source_path}")
            return source_path

        # Create type-specific subdirectory
        type_dir = self.artifacts_dir / artifact_type
        type_dir.mkdir(exist_ok=True)

        # Copy file
        dest_path = type_dir / artifact_name
        shutil.copy2(source_path, dest_path)

        logging.info(f"Artifact saved: {dest_path}")
        return dest_path

    def save_video(self, video_path: Path | str, step: int) -> Path | None:
        """
        Save evaluation video.

        Args:
            video_path: Path to video file
            step: Training step for naming

        Returns:
            Path where video was saved, or None if save_videos=False
        """
        if not self.config.save_videos:
            return None

        return self.save_artifact(
            video_path,
            f"eval_step_{step}.mp4",
            artifact_type="videos",
        )

    def save_image(self, image_path: Path | str, name: str) -> Path | None:
        """
        Save an image.

        Args:
            image_path: Path to image file
            name: Image name

        Returns:
            Path where image was saved, or None if save_images=False
        """
        if not self.config.save_images:
            return None

        return self.save_artifact(image_path, name, artifact_type="images")

    def _print_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Print metrics to console."""
        # Format key metrics
        key_metrics = {
            "step": step,
            "train/loss": metrics.get("train/loss", "N/A"),
            "eval/success_rate": metrics.get("eval/success_rate", "N/A"),
        }

        # Add any metrics with "train/" or "eval/" prefix
        for key, value in metrics.items():
            if key.startswith("train/") or key.startswith("eval/"):
                if isinstance(value, float):
                    key_metrics[key] = f"{value:.4f}"

        # Print formatted
        print(f"[Step {step}] ", end="")
        for key, value in key_metrics.items():
            if key != "step":
                print(f"{key}={value} ", end="")
        print()

    def close(self) -> None:
        """Close logger and flush any pending data."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

        logging.info(f"LocalLogger closed. All data at: {self.log_dir}")

    def get_log_dir(self) -> Path:
        """Get the log directory path."""
        return self.log_dir

    def load_metrics(self) -> list[dict[str, Any]]:
        """Load all logged metrics."""
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics

    def load_summary(self) -> dict[str, Any]:
        """Load training summary."""
        if not self.summary_file.exists():
            return {}

        with open(self.summary_file, 'r') as f:
            return json.load(f)