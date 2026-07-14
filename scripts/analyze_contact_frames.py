#!/usr/bin/env python3
"""
Extract frames with high force deltas for manual contact verification.

Loads the dataset, identifies frames where frame-to-frame force change
exceeds thresholds, and saves the corresponding camera images for review.
"""

import sys, os, numpy as np, pandas as pd
os.chdir('/root/workspace/dc_dir/lerobot')
sys.path.insert(0, 'src')

from PIL import Image as PILImage
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA_ROOT = "/root/data2/dc_dir/datasets/dataset_0611_pickup_long_all"
REPO = "dataset_0611_pickup_long_all"
OUT_DIR = "/tmp/contact_frames"
os.makedirs(OUT_DIR, exist_ok=True)

joint_short = ["L_j1","L_j2","L_j3","L_j4","L_j5","L_j6","L_j7",
               "R_j1","R_j2","R_j3","R_j4","R_j5","R_j6","Tr1","Tr2"]

# ── Step 1: Compute force deltas from parquet (fast, no video) ──
PARQ = os.path.join(DATA_ROOT, "data", "chunk-000")
all_forces = []
for f in sorted(os.listdir(PARQ))[:5]:  # first 5 episodes
    df = pd.read_parquet(os.path.join(PARQ, f))
    forces = np.stack(df['observation.force'].values)
    all_forces.append(forces)
all_forces = np.concatenate(all_forces, axis=0)

body_forces = np.delete(all_forces, [6, 13], axis=1)
body_deltas = np.abs(np.diff(body_forces, axis=0))
body_delta_max = body_deltas.max(axis=1)

# ── Step 2: Find candidate frames ──
threshold = 0.20  # ~P92 — 8% of frames
candidates = np.where(body_delta_max > threshold)[0]

# Take a stratified sample: pick from low, medium, high delta ranges
rng = np.random.RandomState(42)
samples = []
for (lo, hi), label in [
    ((0.20, 0.30), "low"),
    ((0.30, 0.50), "mid"),
    ((0.50, 2.00), "high"),
]:
    pool = [int(c) for c in candidates if lo < float(body_delta_max[c]) <= hi]
    picked = rng.choice(pool, size=min(5, len(pool)), replace=False).tolist()
    for c in picked:
        samples.append((c, label, float(body_delta_max[c])))

# ── Step 3: Load images via LeRobotDataset ──
ds = LeRobotDataset(REPO, root=DATA_ROOT, customer_transforms=False, time_warp=False)

print(f"Found {len(candidates)} candidate frames (delta > {threshold})")
print(f"Sampled {len(samples)} frames for review\n")

for global_idx, label, delta_val in sorted(samples, key=lambda x: x[0]):
    ep = global_idx // 330
    ep_frame = global_idx % 330

    item = ds[global_idx]

    # Save head_cam image
    head_img = item["observation.images.head_cam"]  # (3, 480, 640) float [0,1]
    head_np = (head_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out_name = f"ep{ep:02d}_f{ep_frame:04d}_delta{delta_val:.2f}_{label}.png"
    PILImage.fromarray(head_np).save(os.path.join(OUT_DIR, out_name))

    # Print force context
    force_before = body_forces[global_idx]
    force_after  = body_forces[global_idx + 1] if global_idx + 1 < len(body_forces) else force_before
    delta_per_joint = np.abs(force_after - force_before)
    max_j = int(np.argmax(delta_per_joint))
    print(f"  {out_name}")
    print(f"    max_delta={delta_val:.3f} on joint_{max_j}")
    print(f"    all forces before: {[f'{x:.2f}' for x in force_before.tolist()]}")
    print(f"    all forces after:  {[f'{x:.2f}' for x in force_after.tolist()]}")

print(f"\n✅ Saved {len(samples)} frames to {OUT_DIR}/")
print("Review tip: look for frames where robot is making/releasing contact with object.")
print("If the hand is mid-air or stationary but force spiked → harmonic drive artifact, not contact.")
