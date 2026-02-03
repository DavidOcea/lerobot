#!/bin/bash
# Policy Server 启动脚本

set -e

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

echo "=========================================="
echo "启动 Policy Server"
echo "=========================================="
echo "端口: 50051"
echo "FPS: 30"
echo "设备: cuda"
echo "=========================================="

# 启动 Policy Server
python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --robot.config_path=src/lerobot/robots/supre_robot_follower/trunk_config.yaml \
    --port=50051 \
    --fps=30 \
    --device=cuda
