#!/usr/bin/env python

"""
Action Chunking Transformer with Flow Matching DiT action head.

Replaces the CVAE (VAE encoder + autoregressive decoder) in standard ACT
with a Diffusion Transformer (DiT) trained via flow matching. The DiT generates
the full action chunk in parallel (not autoregressive), eliminating the training-
inference gap and improving small-data generalization.

Key differences from standard ACT:
    - No VAE encoder / latent bottleneck
    - No autoregressive decoder → full chunk generated at once
    - Flow matching loss (velocity prediction) instead of CVAE KL + L1
    - Demos same ACTPolicy wrapper → select_action, temporal_ensemble unchanged
"""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@PreTrainedConfig.register_subclass("act_flow")
@dataclass
class ACTFlowConfig(PreTrainedConfig):
    """Configuration for ACT with Flow Matching DiT action head.

    Shares most parameters with standard ACT. The main differences are
    flow-matching-specific parameters for the diffusion scheduler.
    """

    # ── Flow matching parameters ──
    num_train_timesteps: int = 1000         # Number of noise levels for training
    num_inference_steps: int = 4            # ODE solver steps at inference (4 is fast, 1 with consistency)
    flow_schedule: str = "linear"           # "linear" or "cosine"
    flow_prediction_type: str = "velocity"  # "velocity" (default) or "score"

    # ── Shared with ACT ──
    n_obs_steps: int = 1
    chunk_size: int = 20
    n_action_steps: int = 1

    # ── Vision ──
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_head_img: bool = True

    # ── Model dimensions ──
    dim_model: int = 512
    dim_feedforward: int = 3200
    n_heads: int = 8
    n_encoder_layers: int = 8
    n_decoder_layers: int = 1   # Not used in flow matching (kept for config compat)
    dropout: float = 0.1

    # ── Features ──
    use_state: bool = True
    use_vae: bool = False        # Always False for flow matching
    latent_dim: int = 32         # Not used (kept for compat)
    n_vae_encoder_layers: int = 4  # Not used
    use_robot_position: bool = False
    img_cross_atten: bool = False

    # ── Regularization ──
    state_dropout: float = 0.0
    force_dropout: float = 0.0
    head_dropout: float = 0.0
    label_smoothing: float = 0.0

    # ── Action ──
    use_relative_action: bool = False
    only_first_step: bool = False

    # ── Inference ──
    temporal_ensemble_coeff: float | None = 0.01

    # ── Training ──
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    use_amp: bool = False
    use_warmup_cosine_scheduler: bool = False
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1

    # ── Normalization ──
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "FORCE": NormalizationMode.MEAN_STD,
        }
    )

    # ── Legacy / compat ──
    use_robot_position: bool = False
    img_cross_atten: bool = False
    pre_norm: bool = False
    replace_final_stride_with_dilation: bool = False
    kl_weight: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Force these values for flow matching
        self.use_vae = False

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> list | None:
        if self.n_obs_steps <= 1:
            return None
        return list(range(-self.n_obs_steps + 1, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        if self.use_warmup_cosine_scheduler:
            from lerobot.optim.schedulers import CosineDecayWithWarmupConfig
            return CosineDecayWithWarmupConfig(
                num_warmup_steps=self.warmup_steps,
                min_lr_ratio=self.min_lr_ratio,
            )
        return None
