#!/usr/bin/env bash
# dp_0921 相对动作对照实验：absolute vs relative(delta-from-current)（登记 2026-09-21）
# 背景：料片连续定位在 DP(绝对动作) 上精度不足。改用 Δq 相对动作（预测 action - 当前状态），
#   让 epsilon-prediction 的高斯先验落在近零残差附近 + FiLM 弱条件化下残差更鲁棒。
#   实现已合入 diffusion 策略（--policy.use_relative_action，默认 false，契约不变：
#   select_action 仍输出绝对动作，机器人侧零改动）。本脚本只做单变量 A/B：
#     A 对照组 = 0818 复刻配置 + use_relative_action=false
#     B 实验组 = 0818 复刻配置 + use_relative_action=true
#   两臂除该 flag 外完全相同（seed=1000、同一份当前代码），保证差异仅来自相对动作。
# 0818 复刻参数来源：scripts/run_dp_0915_repro_aug_h16_224.sh 实验 1
# 环境：conda lerobot；必须 unified_dev_0624 分支（当前已在）。
# GPU：ctl→3，treat→4（0 半占、1/2/5/6/7 被占满，勿碰）。

cd /root/workspace/dc_dir/lerobot

# ── A 对照组：绝对动作（0818 复刻 + use_relative_action=false）→ GPU3 ──────
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
  --policy.use_relative_action=false \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0921_rel_ablation_ctl_abs \
  --job_name=dp_0921_rel_ablation_ctl_abs \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0921_rel_ablation_ctl_abs.log 2>&1 &

# ── B 实验组：相对动作（0818 复刻 + use_relative_action=true）→ GPU4 ──────
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
  --policy.use_relative_action=true \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0921_rel_ablation_treat_rel \
  --job_name=dp_0921_rel_ablation_treat_rel \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0921_rel_ablation_treat_rel.log 2>&1 &

echo "Launched: ctl_abs (GPU3, use_relative_action=false), treat_rel (GPU4, use_relative_action=true)"
