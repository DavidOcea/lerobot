#!/usr/bin/env bash
# dp_0918 B 实验：MDN WTA + routing focal loss（登记 2026-09-18）
# 背景（见 mdn-output-head-routing-collapse）：
#   K=12 WTA 检测（step 120000）发现：12 expert 模态切分成功（winner 2-16% 全活、component 0.073
#   健康），但 routing 头塌缩——argmax 75% 集中在 expert 9/10，routing loss 1.844 ≈ ln(1/0.16)
#   ≈「永远猜先验频率」，即 routing 从观测里榨不出任何模态信号。
#   本实验给 routing 加 focal loss（gamma=2），down-weight 高置信度的「易样本」，压掉「塌缩到
#   频率先验」的捷径，短训 50K 步验证：routing 准确率能否显著超过先验基线（~16%）。
#   结论判据（见 dead_expert_check.py 新增的 routing 准确率）：
#     - 准确率显著 > 先验 → 84×84 观测里其实有模态信号，塌缩只是类不平衡/训练动力学问题，MDN 能救；
#     - 准确率 ≈ 先验   → 坐实信号弱到 routing 无解，MDN 路线到此为止，回 224/RGB-D。
# 与 K=12 主实验逐字段同构，唯一变量 = --policy.mdn_focal_gamma=2.0（+短训 steps=50000）。
# 环境：conda lerobot；unified_dev_0624 分支。
# GPU：4（当前 4/5/6/7 全空；1=K12、2=K8 勿碰）。

cd /root/workspace/dc_dir/lerobot

CUDA_VISIBLE_DEVICES=4 nohup /root/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
  --policy.type=diffusion \
  --dataset.root=/root/data2/dc_dir/datasets/dataset_0729_pickup_long_all \
  --dataset.repo_id=dataset_0729_pickup_long_all \
  --batch_size=64 \
  --num_workers=8 \
  --steps=50000 \
  --eval_freq=10000 \
  --save_freq=10000 \
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
  --policy.mdn_focal_gamma=2.0 \
  --dataset.time_warp=true \
  --dataset.customer_transforms=false \
  --dataset.only_head_transforms=false \
  --output_dir=outputs/train/dp_0918_mdn12_wta_focal_pickup_long_noaug \
  --job_name=dp_0918_mdn12_wta_focal_pickup_long_noaug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false > logs/dp_0918_mdn12_wta_focal_pickup_long_noaug.log 2>&1 &

echo "Launched: mdn12_wta_focal_gamma2 (GPU4, 50K steps)"
