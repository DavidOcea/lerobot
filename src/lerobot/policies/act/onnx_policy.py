#!/usr/bin/env python3
"""
Step 3: ONNX inference wrapper — drop-in replacement for ACTPolicy.

Provides the same select_action/forward/reset interface as ACTPolicy,
but runs backbone+encoder through ONNX Runtime and keeps decoder
in PyTorch for the autoregressive for-loop.

Usage:
    from lerobot.policies.act.onnx_policy import ACTPolicyONNX

    policy = ACTPolicyONNX(
        onnx_path="outputs/export/backbone_encoder.onnx",
        checkpoint_path="outputs/train/act_xxx/checkpoints/last/pretrained_model",
        device="cuda",
    )

    action = policy.select_action(obs)  # same as ACTPolicy
    policy.reset()                       # same as ACTPolicy
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lerobot.constants import ACTION, OBS_IMAGES

logger = logging.getLogger("onnx_policy")


# ═══════════════════════════════════════════════════════════════════════
# Decoder-only wrapper (PyTorch)
# ═══════════════════════════════════════════════════════════════════════

class DecoderOnly(nn.Module):
    """Subset of ACT model: decoder + action head only.

    Takes encoder output, runs autoregressive decoder, returns action chunk.

    Optimization: for n_obs_steps>1 models, the temporal ensemble weights
    exp(-0.01*i) are near-uniform (range 1.0→0.83 over 20 steps). We can
    safely truncate the autoregressive loop from chunk_size steps to
    `decoder_steps` steps, repeating the last output to fill the chunk.
    This cuts decoder latency proportionally without retraining.
    """

    def __init__(self, act_model, config, decoder_steps: int = 5):
        super().__init__()
        self.decoder = act_model.decoder
        self.decoder_pos_embed = act_model.decoder_pos_embed
        self.action_head = act_model.action_head
        self.chunk_size = config.chunk_size
        self.decoder_steps = min(decoder_steps, self.chunk_size)
        self.dim_model = config.dim_model
        self.latent_dim = config.latent_dim
        # Fixed start token (zeros) — same as ACT training
        self.register_buffer(
            "tgt_embed",
            torch.zeros(1, self.dim_model),
        )

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Run truncated autoregressive decoder.

        Args:
            encoder_out: (seq_len, B, dim_model) from backbone+encoder

        Returns:
            (B, chunk_size, action_dim) action chunk (padded to full chunk_size)
        """
        B = encoder_out.shape[1]
        device = encoder_out.device

        x = self.tgt_embed.unsqueeze(0).expand(1, B, self.dim_model)  # (1, B, D)

        for i in range(self.decoder_steps):
            pos = self.decoder_pos_embed.weight[i:i+1].unsqueeze(1)  # (1, 1, D)
            dec_in = x[-1:] + pos
            out = self.decoder(
                dec_in, encoder_out,
                decoder_pos_embed=pos, encoder_pos_embed=None,
            )  # (1, B, D)
            x = torch.cat([x, out], dim=0)

        # Take the last decoder_steps outputs and pad to chunk_size by
        # repeating the final output. Temporal ensemble with coeff=0.01
        # weights these near-uniformly, so truncation is lossless.
        last_out = out  # (1, B, D)
        repeat_needed = self.chunk_size - self.decoder_steps
        pad = last_out.repeat(repeat_needed, 1, 1)  # (repeat, B, D)
        decoder_out = torch.cat([x[1:], pad], dim=0)  # skip init token
        decoder_out = decoder_out[:self.chunk_size]   # ensure exact length

        decoder_out = decoder_out.transpose(0, 1)     # (B, chunk_size, D)
        actions = self.action_head(decoder_out)        # (B, chunk_size, action_dim)
        return actions


# ═══════════════════════════════════════════════════════════════════════
# Temporal Ensemble (same as ACT, kept in PyTorch for simplicity)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Reuse original temporal ensemble from modeling_act (no need to reimplement)
# ═══════════════════════════════════════════════════════════════════════
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler as TemporalEnsembler


# ═══════════════════════════════════════════════════════════════════════
# Main ONNX Policy class
# ═══════════════════════════════════════════════════════════════════════

class ACTPolicyONNX:
    """ONNX-accelerated ACT policy with PyTorch decoder.

    Replaces ACTPolicy for inference only (forward/train not supported).

    Key differences from ACTPolicy:
        - backbone + encoder → ONNX Runtime (GPU accelerated)
        - decoder + action_head → PyTorch (autoregressive loop)
        - temporal ensembler → PyTorch (stateful, not ONNX-compatible)
        - normalize/unnormalize → PyTorch (needs dataset stats)
    """

    def __init__(
        self,
        onnx_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ── Load checkpoint (config + weights) ──
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        full_policy = ACTPolicy.from_pretrained(checkpoint_path)
        full_policy = full_policy.to(self.device)
        config = full_policy.config
        self.chunk_size = config.chunk_size
        self.n_action_steps = config.n_action_steps
        self.n_obs_steps = config.n_obs_steps
        self.use_relative_action = config.use_relative_action
        self.only_first_step = config.only_first_step

        logger.info(
            f"  chunk_size={self.chunk_size}, n_obs_steps={self.n_obs_steps}, "
            f"n_action_steps={self.n_action_steps}, "
            f"temporal_ensemble_coeff={config.temporal_ensemble_coeff}"
        )

        # ── Normalization (PyTorch, uses dataset stats) ──
        self.normalize_inputs = full_policy.normalize_inputs
        self.unnormalize_outputs = full_policy.unnormalize_outputs

        # ── Encoder inference backend ──
        # Auto-detect: .engine (TensorRT, Jetson Orin) > .onnx (ONNX Runtime, H100/CPU)
        self._encoder_backend: str = "ort"
        self._ort_session = None
        self._trt_context = None
        self._trt_engine = None

        if onnx_path.endswith(".engine"):
            self._init_trt_backend(onnx_path)
        else:
            self._init_ort_backend(onnx_path)

        # ── Decoder (PyTorch, extracted from full model) ──
        full_policy.model.eval()
        self._decoder = DecoderOnly(full_policy.model, config, decoder_steps=5).to(self.device)
        self._decoder.eval()

        # ── Temporal ensemble ──
        self._te_coeff = config.temporal_ensemble_coeff
        if self._te_coeff is not None:
            self._temporal_ensembler = TemporalEnsembler(self._te_coeff, self.chunk_size)
            self._action_queue = None
        else:
            self._temporal_ensembler = None
            self._action_queue = deque([], maxlen=self.n_action_steps)

        # ── State buffer for n_obs_steps > 1 ──
        self._state_buffer = deque(maxlen=self.n_obs_steps) if self.n_obs_steps > 1 else None
        self._force_buffer = deque(maxlen=self.n_obs_steps) if self.n_obs_steps > 1 else None

        # ── Image key config ──
        self._img_keys = sorted(
            [k for k, ft in config.input_features.items() if ft.type.name == "VISUAL"]
        )

        self.reset()
        logger.info("✅ ACTPolicyONNX ready")

    def _init_ort_backend(self, path: str):
        """Set up ONNX Runtime backend (standard GPU or CPU)."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime-gpu"
            )
        _avail = ort.get_available_providers()
        _preferred = [
            p for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider",
                         "CPUExecutionProvider")
            if p in _avail
        ]
        self._ort_session = ort.InferenceSession(path, providers=_preferred)
        logger.info(f"  Encoder backend: ONNX Runtime ({self._ort_session.get_providers()[0]})")
        self._encoder_backend = "ort"

    def _init_trt_backend(self, path: str):
        """Set up TensorRT backend (Jetson Orin, NVIDIA GPU)."""
        import tensorrt as trt

        with open(path, "rb") as f:
            engine_data = f.read()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        self._trt_engine = runtime.deserialize_cuda_engine(engine_data)
        self._trt_context = self._trt_engine.create_execution_context()

        # Resolve input/output tensor names once, convert Dims to tuple
        self._trt_in_names = [self._trt_engine.get_tensor_name(i) for i in range(5)]
        self._trt_out_name = self._trt_engine.get_tensor_name(5)
        _dims = self._trt_engine.get_tensor_shape(self._trt_out_name)
        self._trt_out_shape = tuple(_dims)

        logger.info(f"  Encoder backend: TensorRT ({len(engine_data)/1e6:.1f} MB engine)")
        self._encoder_backend = "trt"

    def reset(self):
        """Reset stateful components (call at episode start)."""
        if self._temporal_ensembler is not None:
            self._temporal_ensembler.reset()
        if self._state_buffer is not None:
            self._state_buffer.clear()
        if self._force_buffer is not None:
            self._force_buffer.clear()

    def select_action(self, obs: dict[str, Any]) -> torch.Tensor:
        """Select a single action given environment observations.

        Args:
            obs: Dict with keys like:
                observation.state: (15,) tensor
                observation.force: (15,) tensor
                observation.images.head_cam: (3, 480, 640) tensor
                observation.images.left_wrist_cam: (3, 480, 640) tensor
                observation.images.right_wrist_cam: (3, 480, 640) tensor

        Returns:
            (15,) tensor — single action in unnormalized space
        """
        # ── Multi-step history (same as n_obs_steps training) ──
        batch = self._build_batch(obs)

        # ── Normalize ──
        batch = self.normalize_inputs(batch)

        # ── ONNX backbone+encoder ──
        encoder_out = self._run_encoder(batch)

        # ── PyTorch decoder ──
        with torch.no_grad():
            actions = self._decoder(encoder_out)  # (B, chunk_size, 15)

        # ── Unnormalize ──
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        # ── Temporal ensemble or action queue ──
        if self._temporal_ensembler is not None:
            action = self._temporal_ensembler.update(actions)
            return action
        else:
            if len(self._action_queue) == 0:
                chunk = actions[:, :self.n_action_steps]
                self._action_queue.extend(chunk.transpose(0, 1))
            return self._action_queue.popleft()

    def _build_batch(self, obs: dict) -> dict:
        """Build single-sample batch dict for ONNX inference.

        Handles multi-step state/force stacking (n_obs_steps > 1).
        """
        batch = {}

        # ── State ──
        state = obs["observation.state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.to(self.device)

        if self._state_buffer is not None:
            self._state_buffer.append(state)
            states = list(self._state_buffer)
            if len(states) < self.n_obs_steps:
                states = [states[0]] * (self.n_obs_steps - len(states)) + states
            state = torch.stack(states)  # (T, 15)
        batch["observation.state"] = state

        # ── Force ──
        force = obs.get("observation.force")
        if force is not None:
            if isinstance(force, np.ndarray):
                force = torch.from_numpy(force).float()
            force = force.to(self.device)
            if self._force_buffer is not None:
                self._force_buffer.append(force)
                forces = list(self._force_buffer)
                if len(forces) < self.n_obs_steps:
                    forces = [forces[0]] * (self.n_obs_steps - len(forces)) + forces
                force = torch.stack(forces)
            batch["observation.force"] = force

        # ── Images (keep as-is, normalization handles conversion) ──
        for key in self._img_keys:
            img = obs.get(key)
            if img is not None:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                batch[key] = img.to(self.device)

        return batch

    def _run_encoder(self, batch: dict) -> torch.Tensor:
        """Run backbone+encoder through the active backend (ORT or TensorRT).

        Returns:
            (seq_len, 1, 512) encoder output tensor on self.device
        """
        if self._encoder_backend == "trt":
            return self._run_trt_encoder(batch)
        return self._run_ort_encoder(batch)

    def _prepare_encoder_inputs(self, batch: dict) -> tuple:
        """Common input preparation for both ORT and TRT backends."""
        # Images — collect and match 3-camera export order
        img_keys = sorted([k for k in batch if "image" in k.lower()])
        images = []
        for key in img_keys:
            img = batch[key]
            if img.ndim == 3:
                img = img.unsqueeze(0)
            images.append(img)
        while len(images) < 3:
            images.append(images[0])

        # State/force — reshape + zero-pad to (1, n_obs_steps * 15)
        state = batch["observation.state"].reshape(1, -1)
        _expected = self.n_obs_steps * 15
        if state.shape[1] < _expected:
            _padded = torch.zeros(1, _expected, dtype=state.dtype, device=state.device)
            _padded[0, :state.shape[1]] = state[0]
            state = _padded

        force = batch.get("observation.force", torch.zeros_like(state))
        if force is not None:
            force = force.reshape(1, -1)
            if force.shape[1] < _expected:
                _padded_f = torch.zeros(1, _expected, dtype=force.dtype, device=force.device)
                _padded_f[0, :force.shape[1]] = force[0]
                force = _padded_f

        return tuple(images), state, force

    def _run_ort_encoder(self, batch: dict) -> torch.Tensor:
        images, state, force = self._prepare_encoder_inputs(batch)
        feed = {
            "img0": images[0].cpu().numpy().astype(np.float32),
            "img1": images[1].cpu().numpy().astype(np.float32),
            "img2": images[2].cpu().numpy().astype(np.float32),
            "state": state.cpu().numpy().astype(np.float32),
            "force": force.cpu().numpy().astype(np.float32),
        }
        ort_out = self._ort_session.run(None, feed)[0]
        return torch.from_numpy(ort_out).to(self.device)

    def _run_trt_encoder(self, batch: dict) -> torch.Tensor:
        """TensorRT inference — TRT 8.6 bindings API (compatible with JetPack)."""
        import numpy as np

        images, state, force = self._prepare_encoder_inputs(batch)

        # Input tensors on GPU
        in_tensors = [
            images[0].contiguous(), images[1].contiguous(), images[2].contiguous(),
            state.contiguous(), force.contiguous(),
        ]

        # Allocate output buffer
        nelem = int(np.prod(self._trt_out_shape))
        out_tensor = torch.empty(nelem, dtype=torch.float32, device=self.device).contiguous()

        # TRT 8.6 bindings API: list of int device pointers
        bindings = [t.data_ptr() for t in in_tensors] + [out_tensor.data_ptr()]

        # Execute synchronously (simpler, no stream management)
        self._trt_context.execute_v2(bindings)

        return out_tensor.reshape(self._trt_out_shape)

    def __repr__(self):
        return (
            f"ACTPolicyONNX(\n"
            f"    chunk_size={self.chunk_size},\n"
            f"    n_obs_steps={self.n_obs_steps},\n"
            f"    n_action_steps={self.n_action_steps},\n"
            f"    temporal_ensemble_coeff={self._te_coeff},\n"
            f"    device={self.device},\n"
            f")"
        )
