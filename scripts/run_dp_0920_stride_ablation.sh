#!/usr/bin/env bash
# dp_0920 降 stride 消融：0818 复刻 + vision_backbone_output_stride ∈ {8, 16}（登记 2026-09-20）
# 背景（见 pickup-verified-dead-ends / dp-pickup-localization-experiments）：
#   0818(h8+84+noaug) = 唯一「跟随」配置，但精度差（对准差）。
#   根因坐实：resnet18 在 84×84 输入下把整图压成 512×3×3 feature map（stride≈28），
#   SpatialSoftmax 只有 9 个空间锚点；料片 ~12px < 一个 cell(28px)，精细位置物理不可表达。
#   本消融：关掉 layer4（stride16）或 layer3+layer4（stride8）的 stride-2 下采样，
#   84×84 下 feature map 变成 6×6=36 格（stride16）或 11×11=121 格（stride8）。
#   唯一变量 = --policy.vision_backbone_output_stride，其余与 0818 逐字段同构 → 干净对照。
#   基线 32 见 0818 历史结论（checkpoint 目录已被清，本次不重跑 32）。
# 环境：conda lerobot；必须 unified_dev_0624 分支（当前已在，未切分支）。
# GPU：stride8→GPU3、stride16→GPU4（2026-09-20 两卡空闲；其余卡被占）。

cd /root/workspace/dc_dir/lerobot

# ── 实验 1：0818 复刻 + 降 stride 8（h8 + 84 + noaug + stride8）→ GPU3 ──────
CUDA_VISIBLE_DEVICES=3 nohup /root/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
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
  --policy.vision_backbone_output_stride=8 \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0920_stride8_0818_repro_pickup_long_noaug \
  --job_name=dp_0920_stride8_0818_repro_pickup_long_noaug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0920_stride8_0818_repro_pickup_long_noaug.log 2>&1 &

# ── 实验 2：0818 复刻 + 降 stride 16（h8 + 84 + noaug + stride16）→ GPU4 ────
CUDA_VISIBLE_DEVICES=4 nohup /root/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
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
  --policy.vision_backbone_output_stride=16 \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0920_stride16_0818_repro_pickup_long_noaug \
  --job_name=dp_0920_stride16_0818_repro_pickup_long_noaug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0920_stride16_0818_repro_pickup_long_noaug.log 2>&1 &

echo "Launched: 0818 repro + stride8 (GPU3) + stride16 (GPU4)"
