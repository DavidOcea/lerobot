#!/usr/bin/env python3
"""
Step 1 (fixed): ONNX operator compatibility — tests wrapped submodules
that strip None outputs, matching the actual ONNX export structure.

Usage:
    python scripts/check_onnx_ops.py \
        --checkpoint outputs/train/act_0611_pickup_long_cs20_te001/checkpoints/last/pretrained_model \
        --device cuda
"""

import sys, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("onnx_check")

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_config_and_weights(checkpoint_path: str, device: torch.device):
    """Load policy config + state_dict without creating full ACT."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    return policy, policy.to(device).eval()


# ── Wrapper modules that strip problematic outputs ───────────────────────

class BackboneExport(nn.Module):
    """Wraps ResNet18 backbone, returns only the feature_map tensor."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, img):
        return self.backbone(img)["feature_map"]


class EncoderLayerExport(nn.Module):
    """Wraps one EncoderLayer, stripping None attention weights."""
    def __init__(self, layer, dim):
        super().__init__()
        self.layer = layer
        self.dim = dim

    def forward(self, x, pos_embed):
        return self.layer(x, pos_embed=pos_embed, key_padding_mask=None)


class EncoderExport(nn.Module):
    """Wraps full Encoder, stripping None key_padding_mask."""
    def __init__(self, encoder, dim):
        super().__init__()
        self.encoder = encoder
        self.dim = dim

    def forward(self, x, pos_embed):
        return self.encoder(x, pos_embed=pos_embed, key_padding_mask=None)


class DecoderLayerExport(nn.Module):
    """Wraps one DecoderLayer, stripping all None args."""
    def __init__(self, layer, dim):
        super().__init__()
        self.layer = layer
        self.dim = dim

    def forward(self, x, encoder_out):
        return self.layer(x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None)


# ── Full backbone+encoder combined module (what we'll actually export) ───

class BackboneEncoderExport(nn.Module):
    """Complete Backbone + Encoder in one module. This IS what we export."""
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
        encoder_in_tokens = [self.model.encoder_latent_input_proj(latent_sample)]  # (B, D)
        # Position embeddings: each is (1, D), we stack them later
        pos_weight = self.model.encoder_1d_feature_pos_embed.weight  # (n_tok, D)
        encoder_in_pos_embed = [pos_weight[0:1]]                     # (1, D) for latent

        # ── State token ──
        encoder_in_tokens.append(self.model.encoder_robot_state_input_proj(state))  # (B, D)
        encoder_in_pos_embed.append(pos_weight[1:2])                                # (1, D)

        # ── Force token ──
        encoder_in_tokens.append(self.model.encoder_robot_force_input_proj(force))  # (B, D)
        encoder_in_pos_embed.append(pos_weight[2:3])                                # (1, D)

        # ── Image features (3 cameras) ──
        for img in [img0, img1, img2]:
            cam_features = self.model.backbone(img)["feature_map"]
            cam_pos_embed = self.model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.model.encoder_img_feat_input_proj(cam_features)
            cam_features = cam_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
            cam_pos_embed = cam_pos_embed.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

        # ── Stack: all entries are (1_or_HW, B, D) → (total_seq, B, D) ──
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)
        encoder_out = self.model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed, key_padding_mask=None)
        return encoder_out


def run_check(name, module, inputs_dict, tolerance=1e-4):
    """Try torch.jit.trace + output comparison."""
    try:
        traced = torch.jit.trace(module, example_kwarg_inputs=inputs_dict)
        with torch.no_grad():
            orig = module(**inputs_dict)
            traced_out = traced(**inputs_dict)
        if isinstance(orig, torch.Tensor):
            diff = (orig - traced_out).abs().max().item()
        else:
            diff = max((o - t).abs().max().item() for o, t in zip(orig, traced_out))
        ok = diff < tolerance
        logger.info(f"  {'✅' if ok else '❌'} {name}: max_diff={diff:.2e} {'OK' if ok else 'TOO LARGE'}")
        return {"name": name, "passed": ok, "max_diff": diff}
    except Exception as e:
        logger.info(f"  ❌ {name}: {str(e)[:150]}")
        return {"name": name, "passed": False, "max_diff": None, "error": str(e)[:150]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    policy, policy_model = load_config_and_weights(args.checkpoint, device)
    model = policy.model
    cfg = policy.config
    logger.info(f"  chunk_size={cfg.chunk_size}, n_obs_steps={cfg.n_obs_steps}, dim_model={cfg.dim_model}")

    results = []
    B, C, H, W = 1, 3, 480, 640
    n_tokens = 1 + 1 + 1 + 3 * (15 * 20)  # latent + state + force + 3 cameras × (H/32 × W/32)
    D = cfg.dim_model

    # ── 1. Backbone (wrapped, looser tolerance for BatchNorm trace diff) ──
    logger.info("=" * 60)
    logger.info("1/6: Backbone (wrapped → returns tensor)")
    logger.info("=" * 60)
    bb_export = BackboneExport(model.backbone).to(device)
    results.append(run_check("backbone", bb_export,
        {"img": torch.randn(B, C, H, W, device=device)}, tolerance=5e-3))

    # ── 2. Image Conv2d projector ──
    logger.info("2/6: Conv2d projector")
    logger.info("=" * 60)
    with torch.no_grad():
        feat_map = model.backbone(torch.randn(B, C, H, W, device=device))["feature_map"]
        _, C_feat, H_feat, W_feat = feat_map.shape
    proj = model.encoder_img_feat_input_proj
    results.append(run_check("conv2d_proj", proj,
        {"input": torch.randn(B, C_feat, H_feat, W_feat, device=device)}))

    # ── 3. Token projections ──
    logger.info("3/6: Token projections (state, force, latent Linear)")
    logger.info("=" * 60)
    n_state = 15 * max(1, cfg.n_obs_steps)
    results.append(run_check("state_proj", model.encoder_robot_state_input_proj,
        {"input": torch.randn(B, n_state, device=device)}))
    results.append(run_check("force_proj", model.encoder_robot_force_input_proj,
        {"input": torch.randn(B, n_state, device=device)}))
    results.append(run_check("latent_proj", model.encoder_latent_input_proj,
        {"input": torch.randn(B, cfg.latent_dim, device=device)}))

    # ── 4. Single Encoder Layer (wrapped) ──
    logger.info("4/6: EncoderLayer (wrapped → strips None)")
    logger.info("=" * 60)
    enc_layer_wrap = EncoderLayerExport(model.encoder.layers[0], D).to(device)
    x = torch.randn(n_tokens, B, D, device=device)
    pos = torch.randn(n_tokens, B, D, device=device)
    results.append(run_check("encoder_layer_0", enc_layer_wrap,
        {"x": x, "pos_embed": pos}))

    # ── 5. Full Encoder (wrapped) ──
    logger.info("5/6: Full Encoder 8 layers (wrapped → strips None)")
    logger.info("=" * 60)
    enc_wrap = EncoderExport(model.encoder, D).to(device)
    results.append(run_check("encoder_full", enc_wrap,
        {"x": x, "pos_embed": pos}))

    # ── 6. Complete Backbone+Encoder combined (the actual export target) ──
    logger.info("6/6: Backbone+Encoder COMBINED (full export module)")
    logger.info("=" * 60)
    combined = BackboneEncoderExport(model, cfg).to(device)
    s = torch.randn(B, n_state, device=device)
    f = torch.randn(B, n_state, device=device)
    i0 = torch.randn(B, C, H, W, device=device)
    i1 = torch.randn(B, C, H, W, device=device)
    i2 = torch.randn(B, C, H, W, device=device)
    results.append(run_check("backbone+encoder", combined,
        {"img0": i0, "img1": i1, "img2": i2, "state": s, "force": f}, tolerance=5e-3))

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    for r in results:
        s = "✅" if r["passed"] else "❌"
        d = f"diff={r['max_diff']:.2e}" if r["max_diff"] is not None else "ERROR"
        e = r.get("error", "")
        logger.info(f"  {s} {r['name']:<30s} {d:<15s} {e}")
    logger.info(f"\nResult: {passed}/{len(results)} passed")
    if passed == len(results):
        logger.info("✅ ALL CHECKS PASSED — ready for Step 2 (ONNX export)")
    else:
        logger.info(f"❌ {len(results)-passed} failures — fix before proceeding to Step 2")


if __name__ == "__main__":
    main()
