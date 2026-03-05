#!/bin/bash
# 实验 A: 余弦 LR 优化版
# 基于 0225_act_5 的最佳结果进行微调

# GPU 设置
export CUDA_VISIBLE_DEVICES=1 nohup
# 训练参数
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=dataset_0211_short \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0211_short \
    \
    --policy.use_warmup_cosine_scheduler=true \
    --policy.warmup_steps=8000 \
    --policy.min_lr_ratio=0.05 \
    --policy.optimizer_lr=1e-5 \
    \
    --policy.use_relative_action=true \
    --policy.only_first_step=true \
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
    --output_dir=outputs/train/act_exp_A_cosine_opt \
    --job_name=act_exp_A_cosine_opt \
    --policy.push_to_hub=false  >  logs/exp_A_cosine_opt.log 2>&1 &
