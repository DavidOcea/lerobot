#!/bin/bash
# 实验 B: 轻量 Label Smoothing
# 修复 0225_act_4 的问题：smoothing 太强导致动作不稳定

# GPU 设置
export CUDA_VISIBLE_DEVICES=2 nohup

# 训练参数
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=dataset_0211_short \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0211_short \
    \
    --policy.label_smoothing=0.02 \
    \
    --policy.use_warmup_cosine_scheduler=true \
    --policy.warmup_steps=5000 \
    --policy.min_lr_ratio=0.1 \
    --policy.optimizer_lr=1e-5 \
    \
    --policy.state_dropout=0.3 \
    --policy.dropout=0.1 \
    \
    --dataset.customer_transforms=true \
    --dataset.only_head_transforms=true \
    \
    --batch_size=32 \
    --steps=200000 \
    --eval_freq=20000 \
    --save_freq=40000 \
    \
    --output_dir=outputs/train/act_exp_B_light_smooth \
    --job_name=act_exp_B_light_smooth \
    --policy.push_to_hub=false  >  logs/exp_B_light_smooth.log  2>&1 &
