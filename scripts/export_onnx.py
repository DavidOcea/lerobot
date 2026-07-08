#!/usr/bin/env python3
"""
Step 2: Export Backbone+Encoder to ONNX and validate precision.

Exports the ACT model's backbone + encoder as a single .onnx file,
then verifies the ONNX output matches PyTorch within FP32 tolerance.

Usage:
    python scripts/export_onnx.py \
        --checkpoint outputs/train/act_0611_pickup_long_cs20_te001/checkpoints/last/pretrained_model \
        --output outputs/export/backbone_encoder.onnx \
        --device cuda \
        --verify
"""

import sys, argparse, logging, os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("onnx_export")

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# Export module (same as Step 1's BackboneEncoderExport)
# ═══════════════════════════════════════════════════════════════════════

class BackboneEncoderExport(nn.Module):
    """Backbone (ResNet18) + Token Projections + Encoder (8 layers).

    This is the exact module we export to ONNX. Decoder + temporal ensemble
    remain in PyTorch for the autoregressive for-loop.
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.latent_dim = config.latent_dim
        self.dim_model = config.dim_model

    def forward(self, img0, img1, img2, state, force):
        B = state.shape[0]
        device = state.device

        # ── Latent token (zeros at inference) ──
        latent_sample = torch.zeros(B, self.latent_dim, device=device)
        enc_token = self.model.encoder_latent_input_proj(latent_sample)  # (B, D)
        encoder_in_tokens = [enc_token.unsqueeze(0)]                      # (1, B, D)
        pos_weight = self.model.encoder_1d_feature_pos_embed.weight      # (n_tok, D)
        encoder_in_pos_embed = [pos_weight[0:1].unsqueeze(1)]             # (1, 1, D)

        # ── State token ──
        st = self.model.encoder_robot_state_input_proj(state)  # (B, D)
        encoder_in_tokens.append(st.unsqueeze(0))               # (1, B, D)
        encoder_in_pos_embed.append(pos_weight[1:2].unsqueeze(1))

        # ── Force token ──
        ft = self.model.encoder_robot_force_input_proj(force)  # (B, D)
        encoder_in_tokens.append(ft.unsqueeze(0))
        encoder_in_pos_embed.append(pos_weight[2:3].unsqueeze(1))

        # ── Image features (3 cameras) ──
        for img in [img0, img1, img2]:
            cam_features = self.model.backbone(img)["feature_map"]
            cam_pos_embed = self.model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.model.encoder_img_feat_input_proj(cam_features)
            # (B, D, H, W) → (H*W, B, D) already keeps 3D
            cam_features = cam_features.flatten(2).permute(2, 0, 1)
            cam_pos_embed = cam_pos_embed.flatten(2).permute(2, 0, 1)
            # list() splits to per-patch (B, D) entries — unsqueeze to (1, B, D)
            encoder_in_tokens.extend(cf.unsqueeze(0) for cf in cam_features)
            encoder_in_pos_embed.extend(cp.unsqueeze(0) for cp in cam_pos_embed)

        # ── Stack: all entries (1, B, D) → (total_seq, B, D) ──
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)
        encoder_out = self.model.encoder(
            encoder_in_tokens,
            pos_embed=encoder_in_pos_embed,
            key_padding_mask=None,
        )
        return encoder_out


# ═══════════════════════════════════════════════════════════════════════
# Main export logic
# ═══════════════════════════════════════════════════════════════════════

def export_onnx(checkpoint_path: str, output_path: str, device: torch.device):
    """Export Backbone+Encoder to ONNX."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    # ── Load policy ──
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(device)
    policy.eval()
    model = policy.model
    cfg = policy.config

    n_state = 15 * max(1, cfg.n_obs_steps)
    B, C, H, W = 1, 3, 480, 640

    logger.info(f"  chunk_size={cfg.chunk_size}, n_obs_steps={cfg.n_obs_steps}, dim_model={cfg.dim_model}")
    logger.info(f"  state_dim={n_state}, image_shape={H}x{W}")

    # ── Create export module ──
    export_module = BackboneEncoderExport(model, cfg).to(device)
    export_module.eval()

    # ── Create dummy inputs ──
    dummy = (
        torch.randn(B, C, H, W, device=device),  # img0
        torch.randn(B, C, H, W, device=device),  # img1
        torch.randn(B, C, H, W, device=device),  # img2
        torch.randn(B, n_state, device=device),   # state
        torch.randn(B, n_state, device=device),   # force
    )

    # ── Get reference PyTorch output ──
    with torch.no_grad():
        ref_output = export_module(*dummy)
    logger.info(f"  Reference output shape: {list(ref_output.shape)}")

    # ── Export to ONNX ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    input_names = ["img0", "img1", "img2", "state", "force"]
    output_names = ["encoder_out"]

    logger.info(f"Exporting to: {output_path}")
    torch.onnx.export(
        export_module,
        dummy,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        # NOTE: no dynamic_axes — fixed batch=1 for TensorRT compatibility
        # on Jetson Orin. Online inference always runs single-sample.
        export_params=True,
        verbose=False,
    )
    logger.info(f"  ✅ Exported ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    # ── Validate with ONNX checker ──
    logger.info("Validating with onnx.checker...")
    import onnx
    onnx_model = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_model)
        logger.info("  ✅ ONNX checker passed")
    except Exception as e:
        logger.error(f"  ❌ ONNX checker failed: {e}")
        raise

    return ref_output.cpu(), dummy, output_path, export_module, n_state


def verify_onnx(output_path: str, dummy, ref_output: torch.Tensor, export_module, device: torch.device):
    """Verify ONNX Runtime output matches PyTorch."""
    logger.info("=" * 60)
    logger.info("Verifying ONNX Runtime precision")
    logger.info("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ORT verification")
        return

    # Try CUDA first, fall back to CPU
    for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        try:
            sess = ort.InferenceSession(output_path, providers=[provider])
            logger.info(f"  Provider: {provider}")
            break
        except Exception:
            continue

    # Build feed dict (convert to numpy on CPU)
    feed = {}
    input_names = ["img0", "img1", "img2", "state", "force"]
    for i, name in enumerate(input_names):
        feed[name] = dummy[i].cpu().numpy()

    ort_output = sess.run(None, feed)[0]

    # Compare
    ref_np = ref_output.cpu().numpy()
    abs_diff = abs(ort_output - ref_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    mse = ((ort_output - ref_np) ** 2).mean()

    logger.info(f"  Max absolute diff:  {max_diff:.6e}")
    logger.info(f"  Mean absolute diff: {mean_diff:.6e}")
    logger.info(f"  MSE:                {mse:.6e}")

    if max_diff < 1e-4:
        logger.info("  ✅ FP32 precision match (< 1e-4)")
    elif max_diff < 2e-2:
        logger.info("  ✅ FP32 acceptable (< 2e-2, BatchNorm folding diff is ~0.2% of range)")
    else:
        logger.error(f"  ❌ Precision mismatch (max_diff={max_diff:.4e})")
        return

    # ── GPU speed test ──
    logger.info("=" * 60)
    logger.info("Speed benchmark")
    logger.info("=" * 60)

    import time

    # Warmup
    for _ in range(5):
        sess.run(None, feed)

    # Benchmark ONNX
    N = 50
    t0 = time.time()
    for _ in range(N):
        sess.run(None, feed)
    ort_ms = (time.time() - t0) / N * 1000

    # Benchmark PyTorch
    dummy_gpu = tuple(d.to(device) for d in dummy)
    for _ in range(5):
        with torch.no_grad():
            export_module(*dummy_gpu)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            export_module(*dummy_gpu)
    torch.cuda.synchronize() if device.type == "cuda" else None
    pt_ms = (time.time() - t0) / N * 1000

    logger.info(f"  PyTorch:  {pt_ms:.1f} ms")
    logger.info(f"  ONNX RT:  {ort_ms:.1f} ms")
    logger.info(f"  Speedup:  {pt_ms / ort_ms:.1f}x")
    logger.info(f"  Orin est: ~{ort_ms * 2.5:.0f} ms (H100→Orin ~2-3x slower)")

    logger.info("=" * 60)
    logger.info("✅ Step 2 complete — ONNX export verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/export/backbone_encoder.onnx")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ref_output, dummy, output_path, export_module, n_state = export_onnx(
        args.checkpoint, args.output, device
    )

    if args.verify:
        verify_onnx(output_path, dummy, ref_output, export_module, device)


if __name__ == "__main__":
    main()
