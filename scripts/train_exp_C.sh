#!/bin/bash
# 实验 C: 修正分段训练
# 修复 0225_act_0/1/2/3 的问题：LR太高 + dropout太低

# GPU 设置
export CUDA_VISIBLE_DEVICES=1 nohup

# 训练参数
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=dataset_0211_short \
    --dataset.root=/root/data2/dc_dir/datasets/dataset_0211_short \
    \
    --policy.optimizer_lr=3e-5 \
    --policy.optimizer_lr_backbone=3e-5 \
    \
    --policy.state_dropout=0.2 \
    --policy.dropout=0.15 \
    \
    --policy.use_warmup_cosine_scheduler=true \
    --policy.warmup_steps=5000 \
    --policy.min_lr_ratio=0.1 \
    \
    --dataset.customer_transforms=true \
    --dataset.only_head_transforms=true \
    \
    --batch_size=32 \
    --steps=200000 \
    --eval_freq=20000 \
    --save_freq=40000 \
    \
    --output_dir=outputs/train/act_exp_C_fixed_segment \
    --job_name=act_exp_C_fixed_segment \
    --policy.push_to_hub=false  >  logs/exp_C_fixed_segment.log 2>&1 & 
