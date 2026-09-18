#!/usr/bin/env bash
# dp_0917 MDN 输出头实验（登记 2026-09-17）
# 背景（见 dp-pickup-localization-experiments + policy-alternatives-for-precise-localization）：
#   0818(h8+84+noaug) = 唯一「跟随」配置，但精度差。根因 = MSE 条件均值在多模态上被平均（12 个离散
#   reach 目标 → 回归落在中间）。4w>8w>12w 越训越僵 = mode averaging 定律级证据。
#   本实验把单点 denoiser 换成「K 分量混合密度(MDN)输出头 + 路由头」：K 个 expert 各预测一个 epsilon，
#   路由头 softmax 选一个模态。WTA(winner-take-all) = 硬分配，只让最接近的 expert 收梯度 → 逼 K 个
#   expert 把动作空间切成 K 个 disjoint 模态（在线 K-means）。推理时在最噪声步 argmax 路由 logits 选
#   定 expert，之后固定跑标准 DDIM（mode-commit）。
#   K=12 = 假设每个 expert 吃一个离散目标位；K=8 = 欠分割对照（<12，看是否合并邻近目标）。
# 基座：0818 复刻配置逐字段同构（h8/na1/nobs2/drop6/down_dims[256,512]/DDIM5/crop84/noaug）。
#   唯一变量 = --policy.mdn_num_components + --policy.mdn_mode。
# 环境：conda lerobot；必须 unified_dev_0624 分支（当前已在）。
# GPU：K=12→1，K=8→2（3/4 被 smolvla 占满 100% 勿碰；0 半占、5 半占）。

cd /root/workspace/dc_dir/lerobot

# ── 实验 1：K=12 WTA（主实验，假设 12 expert ≈ 12 离散目标）→ GPU1 ──────
CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
  --policy.type=diffusion \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_0729_pickup_long_all \
  --dataset.repo_id=dataset_0729_pickup_long_all \
  --batch_size=64 \
  --num_workers=8 \
  --steps=200000 \
  --eval_freq=20000 \
  --save_freq=40000 \
  --seed=1000 \
  --policy.horizon=8 \
  --policy.n_action_steps=1 \
  --policy.n_obs_steps=2 \
  --policy.drop_n_last_frames=6 \
  --policy.down_dims="[256,512]" \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=5 \
  --policy.crop_is_random=false \
  --policy.crop_shape="[84,84]" \
  --policy.mdn_num_components=12 \
  --policy.mdn_mode=wta \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0917_mdn12_wta_pickup_long_noaug \
  --job_name=dp_0917_mdn12_wta_pickup_long_noaug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0917_mdn12_wta_pickup_long_noaug.log 2>&1 &

# ── 实验 2：K=8 WTA（欠分割对照，看 8 expert 是否合并邻近目标）→ GPU2 ────
CUDA_VISIBLE_DEVICES=2 nohup /root/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
  --policy.type=diffusion \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_0729_pickup_long_all \
  --dataset.repo_id=dataset_0729_pickup_long_all \
  --batch_size=64 \
  --num_workers=8 \
  --steps=200000 \
  --eval_freq=20000 \
  --save_freq=40000 \
  --seed=1000 \
  --policy.horizon=8 \
  --policy.n_action_steps=1 \
  --policy.n_obs_steps=2 \
  --policy.drop_n_last_frames=6 \
  --policy.down_dims="[256,512]" \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=5 \
  --policy.crop_is_random=false \
  --policy.crop_shape="[84,84]" \
  --policy.mdn_num_components=8 \
  --policy.mdn_mode=wta \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0917_mdn8_wta_pickup_long_noaug \
  --job_name=dp_0917_mdn8_wta_pickup_long_noaug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0917_mdn8_wta_pickup_long_noaug.log 2>&1 &

echo "Launched: mdn12_wta (GPU1), mdn8_wta (GPU2)"
