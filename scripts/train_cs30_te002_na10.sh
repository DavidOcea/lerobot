#!/bin/bash
# ACT 提速版: n_action_steps=10 (推理频率降至 3Hz, 10x 加速)
# chunk_size=40 提供大 overlap 隐式平滑 (替代 temporal ensemble)
#
# 对比原版 cs20_te001:
#   n_action_steps: 1 → 10   (推理 10x 加速)
#   chunk_size:      20 → 40 (大 overlap 覆盖 30 帧, 隐式平滑)
#   image_transforms: enable   (泛化换场景)
#   state_dropout:    0.3 → 0.4 (额外泛化)
#   dim_feedforward:  3200 → 2048 (推理单次更快, ~25% 提速)


source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

CUDA_VISIBLE_DEVICES=1 python -m lerobot.scripts.train \
    --policy.device=cuda \
    --policy.type=act \
    --dataset.repo_id=dataset_0611_pickup_long_all \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0611_pickup_long_all \
    \
    --policy.chunk_size=40 \
    --policy.n_action_steps=10 \
    \
    --policy.use_vae=true \
    --policy.latent_dim=32 \
    --policy.n_vae_encoder_layers=4 \
    --policy.n_encoder_layers=8 \
    --policy.n_decoder_layers=1 \
    --policy.dim_model=512 \
    --policy.dim_feedforward=2048 \
    --policy.n_heads=8 \
    --policy.dropout=0.1 \
    --policy.state_dropout=0.4 \
    \
    --policy.use_state=true \
    --policy.use_head_img=true \
    --policy.use_relative_action=false \
    \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.optimizer_weight_decay=0.0001 \
    \
    --dataset.customer_transforms=true \
    --dataset.only_head_transforms=true \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.random_order=true \
    \
    --batch_size=32 \
    --steps=200000 \
    --eval_freq=20000 \
    --save_freq=40000 \
    \
    --output_dir=outputs/train/act_0611_pickup_long_cs30_te002_na10 \
    --job_name=act_0611_pickup_long_cs30_te002_na10 \
    --policy.push_to_hub=false \
    > act_0611_pickup_long_cs30_te002_na10.log 2>&1 &

echo "Training launched. PID: $!"
echo "Monitor: tail -f /root/workspace/dc_dir/lerobot/act_0611_pickup_long_cs30_te002_na10.log"
