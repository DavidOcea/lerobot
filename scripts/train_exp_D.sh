#!/bin/bash
# 实验 D: 组合最优配置
# 结合余弦 LR + 轻量 Label Smoothing + 相对角度

# GPU 设置
export CUDA_VISIBLE_DEVICES=3

# 训练参数
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=dataset_0211_short \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0211_short \
    \
    --policy.use_warmup_cosine_scheduler=true \
    --policy.warmup_steps=5000 \
    --policy.min_lr_ratio=0.1 \
    --policy.optimizer_lr=1e-5 \
    \
    --policy.label_smoothing=0.01 \
    \
    --policy.use_relative_action=true \
    --policy.only_first_step=true \
    \
    --policy.state_dropout=0.15 \
    --policy.dropout=0.12 \
    \
    --dataset.customer_transforms=true \
    --dataset.only_head_transforms=true \
    \
    --batch_size=32 \
    --steps=200000 \
    --eval_freq=20000 \
    --save_freq=40000 \
    \
    --output_dir=outputs/train/act_exp_D_combined \
    --job_name=act_exp_D_combined \
    2>&1 | tee logs/exp_D_combined.log
