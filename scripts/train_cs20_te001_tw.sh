#!/bin/bash
# P0 时间扭曲增强训练
# chunk_size=20, temporal_ensemble_coeff=0.01, time_warp=true
# speed ∈ [0.85, 1.15]: 15%速度变化 → 等价于5倍数据多样性
# 注意: 不启用 --dataset.image_transforms.enable（与 customer_transforms 冲突）

source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

CUDA_VISIBLE_DEVICES=3 python -m lerobot.scripts.train \
    --policy.device=cuda \
    --policy.type=act \
    --dataset.repo_id=dataset_0611_pickup_long_all \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0611_pickup_long_all \
    \
    --policy.chunk_size=20 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    \
    --policy.use_vae=true \
    --policy.latent_dim=32 \
    --policy.n_vae_encoder_layers=4 \
    --policy.n_encoder_layers=8 \
    --policy.n_decoder_layers=1 \
    --policy.dim_model=512 \
    --policy.dim_feedforward=3200 \
    --policy.n_heads=8 \
    --policy.dropout=0.1 \
    --policy.state_dropout=0.3 \
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
    --dataset.time_warp=true \
    --dataset.time_warp_speed_min=0.85 \
    --dataset.time_warp_speed_max=1.15 \
    \
    --batch_size=32 \
    --steps=200000 \
    --eval_freq=20000 \
    --save_freq=40000 \
    \
    --output_dir=outputs/train/act_0611_pickup_long_cs20_te001_tw \
    --job_name=act_0611_pickup_long_cs20_te001_tw \
    --policy.push_to_hub=false \
    > act_0611_pickup_long_cs20_te001_tw.log 2>&1 &

echo "Training launched. PID: $!"
echo "Monitor: tail -f /root/workspace/dc_dir/lerobot/act_0611_pickup_long_cs20_te001_tw.log"
