#!/usr/bin/env python
"""Dead-expert 检测：加载 MDN checkpoint，统计 K 个 expert 各自的 winner 占比 + 路由 argmax 分布 + 分解 loss。

判定：若某 expert 几乎从不赢（winner 占比≈0），说明 WTA 模态切分塌缩成 dead expert；
若路由 argmax 与 winner 分布不一致，说明路由头没学会 mode 选择。
"""
import os
from collections import Counter

import draccus
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.constants import ACTION, OBS_IMAGES

CKPT = "outputs/train/dp_0918_mdn12_wta_focal_pickup_long_noaug/checkpoints/050000/pretrained_model"
N_BATCH = 200  # 200 batch × 64 × horizon=8 ≈ 10 万个 (B,T) winner 样本

# 复刻 launch 脚本参数（和 run_dp_0917_mdn_wta_exps.sh 完全一致）
argv = [
    "--policy.type=diffusion",
    "--dataset.root=/root/data2/dc_dir/datasets/dataset_0729_pickup_long_all",
    "--dataset.repo_id=dataset_0729_pickup_long_all",
    "--batch_size=64",
    "--num_workers=8",
    "--policy.horizon=8",
    "--policy.n_action_steps=1",
    "--policy.n_obs_steps=2",
    "--policy.drop_n_last_frames=6",
    "--policy.down_dims=[256,512]",
    "--policy.noise_scheduler_type=DDIM",
    "--policy.num_inference_steps=5",
    "--policy.crop_is_random=false",
    "--policy.crop_shape=[84,84]",
    "--policy.mdn_num_components=12",
    "--policy.mdn_mode=wta",
    "--policy.mdn_focal_gamma=2.0",
    "--dataset.time_warp=true",
    "--dataset.customer_transforms=false",
    "--dataset.only_head_transforms=false",
    "--policy.device=cuda",
]
cfg = draccus.parse(config_class=TrainPipelineConfig, args=argv)

print("Loading dataset ...", flush=True)
dataset = make_dataset(cfg)
print(f"dataset: {dataset.num_frames} frames, {dataset.num_episodes} eps", flush=True)

print("Loading policy checkpoint ...", flush=True)
policy = DiffusionPolicy.from_pretrained(CKPT)
policy = policy.to("cuda")
policy.eval()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)

K = cfg.policy.mdn_num_components
winner_counter = Counter()
routing_counter = Counter()
winner_mode_counter = Counter()  # 样本级多数 expert（= winner 在 T 维的众数）
routing_correct = 0
routing_n = 0
comp_total = 0.0
routing_total = 0.0
n_batch = 0

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
        # 复刻 DiffusionPolicy.forward 的 normalize + image stack
        batch = policy.normalize_inputs(batch)
        if policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
        batch = policy.normalize_targets(batch)

        # 复刻 compute_loss 的 MDN 前向（不梯度）
        global_cond = policy.diffusion._prepare_global_conditioning(batch)
        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, policy.diffusion.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],), device=trajectory.device,
        ).long()
        noisy = policy.diffusion.noise_scheduler.add_noise(trajectory, eps, timesteps)
        components, logits = policy.diffusion.unet(noisy, timesteps, global_cond=global_cond)

        target = eps.unsqueeze(2)  # (B,T,1,D)
        sq_err = ((components - target) ** 2).mean(dim=-1)  # (B,T,K)
        winner = sq_err.argmin(dim=-1)  # (B,T)

        for k in winner.flatten().tolist():
            winner_counter[k] += 1
        rout = logits.mean(dim=1).argmax(dim=-1)  # (B,) 推理时选模态的同一口径
        for k in rout.flatten().tolist():
            routing_counter[k] += 1

        # routing 准确率：推理时 argmax 选的 expert 是否 == 该样本 T 步里赢最多的 expert。
        # 对比基线 = winner_mode 最高频 expert 的频率（=「永远猜最常出现模态」的先验准确率）。
        winner_mode = winner.mode(dim=1).values  # (B,)
        routing_correct += (rout == winner_mode).sum().item()
        routing_n += winner_mode.shape[0]
        for k in winner_mode.tolist():
            winner_mode_counter[k] += 1

        comp_total += sq_err.min(dim=-1).values.mean().item()
        routing_total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, K), winner.reshape(-1)
        ).item()

        n_batch += 1
        if n_batch % 50 == 0:
            print(f"  processed {n_batch}/{N_BATCH} batches", flush=True)
        if n_batch >= N_BATCH:
            break

total_w = sum(winner_counter.values())
total_r = sum(routing_counter.values())

print("\n================ winner 分布（每个 expert 赢了多少 (B,T) 样本）================")
for k in range(K):
    c = winner_counter.get(k, 0)
    print(f"  expert {k:2d}: {c:7d}  ({100.0 * c / total_w:5.2f}%)")

print("\n================ routing argmax 分布（推理时 logits.mean.argmax 选的 expert）================")
for k in range(K):
    c = routing_counter.get(k, 0)
    print(f"  expert {k:2d}: {c:7d}  ({100.0 * c / total_r:5.2f}%)")

print("\n================ 分解 loss（应分别对应 component / routing）================")
print(f"  loss_components (min sq_err): {comp_total / n_batch:.4f}")
print(f"  loss_routing    (cross_entropy): {routing_total / n_batch:.4f}")
print(f"  total (两者和): {(comp_total + routing_total) / n_batch:.4f}")

# dead-expert 判定
dead = [k for k in range(K) if winner_counter.get(k, 0) / max(total_w, 1) < 0.01]
print(f"\n>>> dead experts (winner 占比 < 1%): {dead if dead else '无（12 个都活着）'}")
print(f">>> 有效 expert 数（winner 占比 ≥ 1%）: {K - len(dead)}")

# routing 准确率：观测里有没有真实模态信号（关键指标）
routing_acc = routing_correct / max(routing_n, 1)
prior_acc = max(winner_mode_counter.values()) / max(sum(winner_mode_counter.values()), 1)
print("\n================ routing 准确率（关键指标）================")
print(f"  routing argmax == 样本 winner 众数: {routing_acc:.4f}")
print(f"  先验基线（永远猜最频模态）:         {prior_acc:.4f}")
print(f"  >>> 显著超过先验基线 = 观测里有真实模态信号；≈先验 = 零信号")
