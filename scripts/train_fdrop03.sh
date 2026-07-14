#!/bin/bash
# Force Dropout 训练: n_obs=8 + time_warp + force_dropout=0.3
# force_dropout 逼模型不依赖谐波减速器的摩擦噪声信号

source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

CUDA_VISIBLE_DEVICES=4 nohup python -m lerobot.scripts.train \
    --policy.device=cuda \
    --policy.type=act \
    --dataset.repo_id=dataset_0611_pickup_long_all \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0611_pickup_long_all \
    \
    --policy.chunk_size=20 \
    --policy.n_action_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.n_obs_steps=8 \
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
    --policy.force_dropout=0.3 \
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
    --output_dir=outputs/train/act_fdrop03_nobs8_tw \
    --job_name=act_fdrop03_nobs8_tw \
    --policy.push_to_hub=false \
    > act_fdrop03_nobs8_tw.log 2>&1 &

echo "Training launched. PID: $!"
echo "Monitor: tail -f /root/workspace/dc_dir/lerobot/act_fdrop03_nobs8_tw.log"