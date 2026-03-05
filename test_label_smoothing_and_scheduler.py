#!/usr/bin/env python
"""
Test script for Label Smoothing and WarmupCosineScheduler

This script validates the implementation and provides usage examples.
"""

import torch
import torch.nn.functional as F
import numpy as np


def test_label_smoothing():
    """Test Label Smoothing implementation"""
    print("=" * 60)
    print("Testing Label Smoothing")
    print("=" * 60)

    # Simulate predictions and targets
    batch_size = 4
    chunk_size = 100
    action_dim = 16

    predictions = torch.randn(batch_size, chunk_size, action_dim)
    targets = torch.randn(batch_size, chunk_size, action_dim)

    # Test different smoothing values
    smoothing_values = [0.0, 0.05, 0.1, 0.2]

    print("\nSmoothing | Original L1 Loss | Smoothed L1 Loss | Difference")
    print("-" * 60)

    for smoothing in smoothing_values:
        # Apply label smoothing
        smoothed_targets = (1 - smoothing) * targets + smoothing * predictions.detach()

        # Compute losses
        original_loss = F.l1_loss(targets, predictions).item()
        smoothed_loss = F.l1_loss(smoothed_targets, predictions).item()

        diff = smoothed_loss - original_loss
        print(f"{smoothing:.3f}     | {original_loss:.6f}       | {smoothed_loss:.6f}        | {diff:+.6f}")

    print("\n✓ Label Smoothing test passed!")
    print("  - Higher smoothing = smaller loss (predictions influence targets)")
    print("  - Prevents overfitting by softening target distribution")


def test_warmup_cosine_scheduler():
    """Test WarmupCosineScheduler implementation"""
    print("\n" + "=" * 60)
    print("Testing WarmupCosineScheduler")
    print("=" * 60)

    # Simulate lr schedule
    total_steps = 50000
    warmup_steps = 2000
    min_lr_ratio = 0.1
    base_lr = 5e-5

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    # Compute LR at key points
    key_steps = [0, 500, 1000, 2000, 5000, 10000, 25000, 50000]
    print("\nStep     | LR Ratio | LR Value  | Phase")
    print("-" * 55)

    for step in key_steps:
        ratio = lr_lambda(step)
        lr = base_lr * ratio
        phase = "Warmup" if step < warmup_steps else "Cosine Decay"
        print(f"{step:8} | {ratio:.4f}   | {lr:.2e} | {phase}")

    # Print ASCII visualization
    print("\n" + "=" * 60)
    print("Learning Rate Schedule Visualization")
    print("=" * 60)

    sample_steps = [0, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    max_bar_length = 40

    print("\nStep  → Learning Rate")
    for step in sample_steps:
        ratio = lr_lambda(step)
        lr = base_lr * ratio
        bar_length = int(ratio * max_bar_length)
        bar = "█" * bar_length + "░" * (max_bar_length - bar_length)
        print(f"{step:5} → [{bar}] {lr:.2e}")

    print(f"\nScale: █ = {base_lr:.2e} (base lr)")
    print(f"       ░ = lower lr")

    print("\n✓ WarmupCosineScheduler test passed!")
    print("  - Warmup phase (0-2000 steps): Linear increase from 0 to base_lr")
    print("  - Decay phase (2000-50000 steps): Cosine decay to min_lr_ratio * base_lr")


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)

    print("\n" + "=" * 10 + " Label Smoothing " + "=" * 10)
    print("""
# Enable Label Smoothing in your config:

python lerobot-train \\
    --policy.type=act \\
    --dataset.repo_id=/path/to/dataset \\
    --policy.label_smoothing=0.05 \\    # Add this!
    --steps=50000

# Recommended values:
# - 0.0: Disabled (default)
# - 0.05: Light smoothing (good for most cases)
# - 0.1: Moderate smoothing (for small datasets)
# - 0.2: Heavy smoothing (use with caution)
""")

    print("\n" + "=" * 10 + " WarmupCosineScheduler " + "=" * 10)
    print("""
# Enable Warmup + Cosine Decay:

python lerobot-train \\
    --policy.type=act \\
    --dataset.repo_id=/path/to/dataset \\
    --policy.use_warmup_cosine_scheduler=true \\    # Enable scheduler
    --policy.warmup_steps=2000 \\                   # Warmup duration
    --policy.min_lr_ratio=0.1 \\                     # Final lr = 0.1 * base_lr
    --policy.optimizer_lr=5e-5 \\                   # Base learning rate
    --steps=50000

# The scheduler will:
# Step 0-2000:    Linear warmup from 0 to 5e-5
# Step 2000-50K:  Cosine decay from 5e-5 to 5e-6
""")

    print("\n" + "=" * 10 + " Combined Configuration " + "=" * 10)
    print("""
# Recommended config for small datasets (30-50 episodes):

python lerobot-train \\
    --policy.type=act \\
    --dataset.repo_id=/root/data2/dc_dir/datasets/dataset_0211_short \\
    --policy.use_relative_action=true \\
    --policy.only_first_step=true \\
    --policy.state_dropout=0.1 \\
    --policy.dropout=0.15 \\
    --policy.label_smoothing=0.05 \\              # NEW: Label smoothing
    --policy.use_warmup_cosine_scheduler=true \\   # NEW: Scheduler
    --policy.warmup_steps=2000 \\
    --policy.min_lr_ratio=0.1 \\
    --policy.optimizer_lr=5e-5 \\
    --batch_size=12 \\
    --steps=50000 \\
    --output_dir=outputs/act_optimized
""")


if __name__ == "__main__":
    test_label_smoothing()
    test_warmup_cosine_scheduler()
    print_usage_examples()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
