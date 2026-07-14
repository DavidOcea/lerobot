#!/usr/bin/env python

"""
ACT-Flow: Action Chunking Transformer with Flow Matching DiT.

Flow matching replaces the CVAE in standard ACT. Instead of a VAE encoder
sampling a latent and an autoregressive decoder producing actions step by step,
a Diffusion Transformer (DiT) generates the full action chunk conditioned on
observations and a noise schedule.

Training:
    t ~ Uniform(0, 1)
    x_t = (1 - t) * x_0 + t * noise       # linear interpolation
    v_target = noise - x_0                  # ground-truth velocity field
    v_pred = DiT(x_t, t, obs_features)      # predicted velocity
    loss = MSE(v_pred, v_target)

Inference:
    x_1 ~ N(0, I)                           # start from pure noise
    for i in range(num_inference_steps):
        t = 1 - i / num_inference_steps
        v = DiT(x_t, t, obs_features)
        x_{t - dt} = x_t - v * dt           # Euler step (or use DPM-Solver)
    return x_0                              # denoised action chunk

The DiT uses AdaLN-Zero modulation: timestep embedding → scale/shift/gate
for each transformer block, with zero-initialized output projection for
training stability.
"""

import math
from collections import deque, OrderedDict
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.configuration_act_flow import ACTFlowConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


# ═══════════════════════════════════════════════════════════════════════
# Helpers — sinusoidal position / timestep embeddings
# ═══════════════════════════════════════════════════════════════════════

def create_sinusoidal_pos_embedding(num_pos: int, dim: int) -> Tensor:
    """1D sinusoidal positional embedding, shape (num_pos, dim)."""
    assert dim % 2 == 0
    positions = torch.arange(num_pos, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_pos, dim)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) or (B, 1) float in [0, 1]."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


# ═══════════════════════════════════════════════════════════════════════
# DiT Block with AdaLN-Zero
# ═══════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with zero-initialized modulation.

    Given a conditioning vector c (timestep embedding), produces:
        shift_msa, scale_msa, gate_msa  — for self-attention sublayer
        shift_mlp, scale_mlp, gate_mlp  — for feed-forward sublayer
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, dim * 6)
        # Zero-init — only the output projection matters for stability
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: Tensor) -> tuple[Tensor, ...]:
        params = self.proj(c)  # (B, 6*dim)
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = params.chunk(6, dim=-1)
        return scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp


class DiTBlock(nn.Module):
    """Single DiT block: Self-Attn + Cross-Attn + MLP, with AdaLN-Zero."""

    def __init__(self, dim: int, n_heads: int, dim_feedforward: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        self.adaln = AdaLNZero(dim, cond_dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        ctx: Tensor,
        ctx_mask: Tensor | None = None,
    ) -> Tensor:
        """x: (B, S, D), cond: (B, D_cond), ctx: (B, S_ctx, D)."""
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaln(cond)

        # ── Self-attention ──
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # ── Cross-attention ──
        x_norm = self.norm2(x)
        attn_out, _ = self.cross_attn(x_norm, ctx, ctx, key_padding_mask=ctx_mask)
        x = x + attn_out

        # ── MLP ──
        x_norm = self.norm3(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# ═══════════════════════════════════════════════════════════════════════
# ACTSinusoidalPositionEmbedding2d  (same as ACT for image feature positions)
# ═══════════════════════════════════════════════════════════════════════

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) → (B, dim, H, W) positional embedding."""
        B, _, H, W = x.shape
        device = x.device
        half = self.dim // 2

        y_embed = torch.arange(H, device=device).float().view(1, 1, H, 1) / (H - 1) * 2 - 1
        x_embed = torch.arange(W, device=device).float().view(1, 1, 1, W) / (W - 1) * 2 - 1
        y_embed = y_embed.expand(B, half // 2, H, W) * 100
        x_embed = x_embed.expand(B, half // 2, H, W) * 100
        pe = torch.cat([x_embed.sin(), x_embed.cos(), y_embed.sin(), y_embed.cos()], dim=1)
        if half % 2 == 1:
            pe = torch.cat([pe, torch.zeros(B, half % 2, H, W, device=device)], dim=1)
        return pe


# ═══════════════════════════════════════════════════════════════════════
# ACTFlowModel — DiT-based action chunk generator
# ═══════════════════════════════════════════════════════════════════════

class ACTFlowModel(nn.Module):
    """DiT model for flow-matching action generation.

    Replaces ACT's CVAE encoder + causal decoder. Generates the full
    action chunk (B, chunk_size, action_dim) in one parallel forward pass
    per denoising step.

    Conditioning:
        - Image features from ResNet18 backbone → flattened → ctx tokens
        - State token from Linear projection
        - Force token from Linear projection

    The action chunk is treated as a sequence of chunk_size tokens,
    each with a learnable positional embedding. The DiT cross-attends
    to the context tokens (image patches + state + force).
    """

    def __init__(self, config: ACTFlowConfig):
        super().__init__()
        self.config = config
        chunk_size = config.chunk_size
        dim = config.dim_model

        # ── Vision backbone ──
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights,
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        # Freeze BN layers for training stability
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ── Image feature projector ──
        self.encoder_img_feat_input_proj = nn.Conv2d(512, dim, kernel_size=1)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(dim // 2)

        # ── State / Force / Latent projections ──
        self.robot_state_feature = config.use_state and config.robot_state_feature
        if self.robot_state_feature:
            _state_in_dim = config.robot_state_feature.shape[0]
            if config.n_obs_steps > 1:
                _state_in_dim *= config.n_obs_steps
            self.encoder_robot_state_input_proj = nn.Linear(_state_in_dim, dim)

        self.encoder_latent_input_proj = nn.Linear(config.latent_dim if config.latent_dim > 0 else 1, dim)

        self.robot_force_feature = getattr(config, 'robot_force_feature', False)
        if self.robot_force_feature and config.robot_force_feature:
            _force_in_dim = config.robot_force_feature.shape[0]
            if config.n_obs_steps > 1:
                _force_in_dim *= config.n_obs_steps
            self.encoder_robot_force_input_proj = nn.Linear(_force_in_dim, dim)

        # ── Position embeddings ──
        self.encoder_1d_feature_pos_embed = nn.Embedding(4, dim)  # latent, state, force + padding
        self.action_pos_embed = nn.Parameter(create_sinusoidal_pos_embedding(chunk_size, dim))

        # ── Input projection (action_dim → model_dim) ──
        action_dim = config.output_features[ACTION].shape[0]
        self.action_in_proj = nn.Linear(action_dim, dim)

        # ── Timestep embedding ──
        self.time_embed = TimestepEmbedding(dim)

        # ── DiT blocks ──
        self.dit_blocks = nn.ModuleList([
            DiTBlock(dim, config.n_heads, config.dim_feedforward, dim, config.dropout)
            for _ in range(config.n_encoder_layers)
        ])

        # ── Final layer norm + action head ──
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.action_head = nn.Linear(dim, config.output_features[ACTION].shape[0])

        # ── Dropout (same as ACT) ──
        self.state_dropout = None
        if config.state_dropout > 0 and self.robot_state_feature:
            self.state_dropout = nn.Dropout(p=config.state_dropout)
        self.force_dropout = None
        if config.force_dropout > 0 and self.robot_force_feature:
            self.force_dropout = nn.Dropout(p=config.force_dropout)
        self.head_dropout = None
        if config.head_dropout > 0:
            self.head_dropout = config.head_dropout
            print("head_dropout =", self.head_dropout)
        if config.state_dropout > 0:
            print("state_dropout =", config.state_dropout)
        if config.force_dropout > 0:
            print("force_dropout =", config.force_dropout)

        # Image feature cache (for heatmap, same as ACT)
        self.original_img_size: list[tuple[int, int]] = []
        self.original_img: list[Tensor] = []
        self.attention_weights: list[Tensor] = []
        self.feature_map_size: list[tuple[int, int]] = []

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.dit_blocks.parameters(), [self.action_head.weight, self.action_head.bias]):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def _build_context(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """Build conditioning context from observations.

        Returns:
            ctx_tokens: (B, N_ctx, D) or None if no visual context
            state_ctx:  (B, D) conditioning vector for timestep embedding
        """
        device = batch["observation.state"].device
        B = batch["observation.state"].shape[0]

        encoder_in_tokens: list[Tensor] = []
        encoder_in_pos: list[Tensor] = []

        # ── State token ──
        if self.robot_state_feature:
            if self.state_dropout is not None and self.training:
                state_tok = self.encoder_robot_state_input_proj(
                    self.state_dropout(batch["observation.state"])
                )
            else:
                state_tok = self.encoder_robot_state_input_proj(batch["observation.state"])
            encoder_in_tokens.append(state_tok.unsqueeze(1))  # (B, 1, D)
            encoder_in_pos.append(self.encoder_1d_feature_pos_embed.weight[1:2].unsqueeze(0).expand(B, -1, -1))

        # ── Force token ──
        if self.robot_force_feature:
            if self.force_dropout is not None and self.training:
                force_tok = self.encoder_robot_force_input_proj(
                    self.force_dropout(batch["observation.force"])
                )
            else:
                force_tok = self.encoder_robot_force_input_proj(batch["observation.force"])
            encoder_in_tokens.append(force_tok.unsqueeze(1))
            encoder_in_pos.append(self.encoder_1d_feature_pos_embed.weight[2:3].unsqueeze(0).expand(B, -1, -1))

        # ── Image features ──
        if "observation.images" in batch:
            for img in batch["observation.images"]:
                if self.head_dropout is not None and self.training and \
                   torch.equal(img, batch.get('observation.images.head_cam', img)):
                    img = F.dropout(img, p=self.head_dropout)
                cam_features = self.backbone(img)["feature_map"]
                cam_pos = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                # (B, D, H, W) → (B, H*W, D)
                cam_features = cam_features.flatten(2).transpose(1, 2)
                cam_pos = cam_pos.flatten(2).transpose(1, 2)
                encoder_in_tokens.append(cam_features)
                encoder_in_pos.append(cam_pos)
        elif hasattr(self.config, 'image_features') and self.config.image_features:
            for key in self.config.image_features:
                if key in batch:
                    img = batch[key]
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)
                    cam_features = cam_features.flatten(2).transpose(1, 2)
                    cam_pos = cam_pos.flatten(2).transpose(1, 2)
                    encoder_in_tokens.append(cam_features)
                    encoder_in_pos.append(cam_pos)

        if not encoder_in_tokens:
            return None, None

        ctx = torch.cat(encoder_in_tokens, dim=1)  # (B, N_ctx, D)

        # Build a simple pooled context for timestep conditioning
        pooled = ctx.mean(dim=1)  # (B, D)

        return ctx, pooled

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[None, None]]:
        """Forward pass.

        Training mode:
            - Samples random t, adds noise to clean actions
            - Predicts velocity field
            - Returns (predicted_velocity, (None, None))

        Inference mode:
            - Calls generate_actions() internally (handled by ACTPolicy wrapper)
        """
        ctx, _pooled = self._build_context(batch)
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device
        chunk_size = self.config.chunk_size
        action_dim = self.config.output_features[ACTION].shape[0]

        if "action" not in batch:
            # Inference mode — ACTPolicy calls predict_action_chunk
            # which calls model(batch)[0]. We handle this by generating.
            return self.generate_actions(ctx, B, chunk_size, action_dim, device), (None, None)

        clean_actions = batch["action"]  # (B, chunk_size, action_dim)

        # ── Sample random timesteps ──
        t = torch.rand(B, device=device, dtype=clean_actions.dtype)

        # ── Add noise: x_t = (1-t)*x_0 + t*noise ──
        noise = torch.randn_like(clean_actions)
        x_t = (1 - t.view(-1, 1, 1)) * clean_actions + t.view(-1, 1, 1) * noise

        # ── Target velocity: v = noise - x_0 ──
        v_target = noise - clean_actions

        # ── Predict velocity ──
        x_t_proj = self.action_in_proj(x_t)  # (B, S, 15) → (B, S, D)
        v_pred_dim = self._forward_dit(x_t_proj, t, ctx)  # (B, S, D)
        v_pred = self.action_head(v_pred_dim)  # (B, S, D) → (B, S, 15)

        # ── Loss ──
        loss = F.mse_loss(v_pred, v_target)

        return loss, (None, None)

    def _forward_dit(self, x: Tensor, t: Tensor, ctx: Tensor | None) -> Tensor:
        """Run noisy actions through DiT → predicted velocity.

        Args:
            x: (B, S, D) noisy action tokens
            t: (B,) timesteps in [0, 1]
            ctx: (B, N_ctx, D) conditioning context

        Returns:
            (B, S, action_dim) predicted velocity
        """
        B, S, D = x.shape

        # Add action position embeddings
        pos = self.action_pos_embed[:S].unsqueeze(0).to(x.device)  # (1, S, D)
        x = x + pos

        # Timestep conditioning
        t_emb = self.time_embed(t)  # (B, D)

        # DiT blocks
        for block in self.dit_blocks:
            x = block(x, t_emb, ctx if ctx is not None else x)

        x = self.final_norm(x)
        return x  # (B, S, dim_model)

    @torch.no_grad()
    def generate_actions(
        self,
        ctx: Tensor | None,
        B: int,
        chunk_size: int,
        action_dim: int,
        device: torch.device,
    ) -> Tensor:
        """Generate action chunk via flow matching ODE integration.

        Uses Euler method with config.num_inference_steps.
        """
        # Start from pure noise (t=1)
        x = torch.randn(B, chunk_size, self.config.dim_model, device=device, dtype=torch.float32)

        dt = 1.0 / self.config.num_inference_steps

        for step in range(self.config.num_inference_steps):
            t_val = 1.0 - step * dt  # from 1.0 down to ~0.0
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            v = self._forward_dit(x, t, ctx)
            # Euler step: x_{t-dt} = x_t - v * dt
            x = x - v * dt

        # Map from model dim to action dim
        x = self.final_norm(x)
        actions = self.action_head(x)
        return actions


# ═══════════════════════════════════════════════════════════════════════
# ACTTemporalEnsembler (copied from modeling_act.py for self-contained module)
# ═══════════════════════════════════════════════════════════════════════

class ACTTemporalEnsemblerFlow(nn.Module):
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions: Tensor | None = None
        self.ensembled_actions_count: Tensor | None = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


# ═══════════════════════════════════════════════════════════════════════
# ACTFlowPolicy — reuses same interface as ACTPolicy
# ═══════════════════════════════════════════════════════════════════════

class ACTFlowPolicy(PreTrainedPolicy):
    """Policy wrapper compatible with standard ACT infrastructure.

    Shares the same interface as ACTPolicy:
        - select_action(obs) → action tensor
        - forward(batch) → (loss, output_dict) for training
        - reset() → clear temporal ensemble
    """

    config_class = ACTFlowConfig
    name = "act_flow"

    def __init__(
        self,
        config: ACTFlowConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = ACTFlowModel(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsemblerFlow(config.temporal_ensemble_coeff, config.chunk_size)

        # Multi-step history buffer (P2 — same as ACT)
        self._state_buffer = deque(maxlen=config.n_obs_steps) if config.n_obs_steps > 1 else None
        self._force_buffer = deque(maxlen=config.n_obs_steps) if config.n_obs_steps > 1 else None

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        if self._state_buffer is not None:
            self._state_buffer.clear()
        if self._force_buffer is not None:
            self._force_buffer.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        # ── Multi-step history stacking (P2 — same as ACT) ──
        if self._state_buffer is not None:
            if batch["observation.state"].ndim == 3:
                pass  # pre-stacked by caller
            else:
                batch = dict(batch)
                self._state_buffer.append(batch["observation.state"])
                states = list(self._state_buffer)
                if len(states) < self.config.n_obs_steps:
                    states = [states[0]] * (self.config.n_obs_steps - len(states)) + states
                batch["observation.state"] = torch.stack(states).unsqueeze(0)
                if "observation.force" in batch:
                    self._force_buffer.append(batch["observation.force"])
                    forces = list(self._force_buffer)
                    if len(forces) < self.config.n_obs_steps:
                        forces = [forces[0]] * (self.config.n_obs_steps - len(forces)) + forces
                    batch["observation.force"] = torch.stack(forces).unsqueeze(0)

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            if self.config.use_head_img:
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
            else:
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features if 'head_cam' not in key]

        # Flatten multi-step state/force (P2 safety net)
        if self._state_buffer is not None:
            batch = dict(batch)
            batch["observation.state"] = batch["observation.state"].reshape(1, -1)
            if "observation.force" in batch:
                batch["observation.force"] = batch["observation.force"].reshape(1, -1)

        actions = self.model(batch)[0]

        if self.config.use_relative_action and "observation.state" in batch:
            state_norm = batch["observation.state"]
            if self.config.only_first_step:
                actions[:, 0:1, :] = actions[:, 0:1, :] + state_norm.unsqueeze(1)
            else:
                state_expanded = state_norm.unsqueeze(1).expand_as(actions)
                actions = actions + state_expanded

        return self.unnormalize_outputs({ACTION: actions})[ACTION]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            if self.config.use_head_img:
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
            else:
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features if 'head_cam' not in key]

        batch = self.normalize_targets(batch)

        if self.config.use_relative_action and "observation.state" in batch:
            if self.config.only_first_step:
                state_expanded = batch["observation.state"].unsqueeze(1)
                batch["action"][:, 0:1, :] = batch["action"][:, 0:1, :] - state_expanded
            else:
                state_expanded = batch["observation.state"].unsqueeze(1).expand_as(batch["action"])
                batch["action"] = batch["action"] - state_expanded

        # ── Flatten multi-step state/force for training (P2) ──
        if "observation.state" in batch and batch["observation.state"].ndim == 3:
            batch = dict(batch)
            batch["observation.state"] = batch["observation.state"].flatten(1)
        if "observation.force" in batch and batch["observation.force"].ndim == 3:
            batch = dict(batch)
            batch["observation.force"] = batch["observation.force"].flatten(1)

        loss, _ = self.model(batch)
        return loss, {}
