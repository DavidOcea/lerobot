#!/bin/bash
# 示例：完整的任务执行流程

# This script demonstrates how to use the Task Agent system
# to execute a sequence of robotic manipulation tasks.

set -e

# ============================================
# 配置
# ============================================

# 模型路径（根据实际情况修改）
MODEL_PATH="/home/smai/dc_dir/lerobot_0901_pybullet/outputs/train/act_1121_3/checkpoints/last/pretrained_model"

# 配置文件
CONFIG_FILE="configs/task_agent_tasks_test.yaml"

# ============================================
# 步骤1: 验证环境
# ============================================

echo "=========================================="
echo "步骤1: 验证环境"
echo "=========================================="

# 检查 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 检查 Python
python --version

# 检查 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ============================================
# 步骤2: 验证模型
# ============================================

echo ""
echo "=========================================="
echo "步骤2: 验证模型"
echo "=========================================="

if [ -d "$MODEL_PATH" ]; then
    echo "✓ 模型路径存在: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
else
    echo "✗ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# ============================================
# 步骤3: 测试模型加载
# ============================================

echo ""
echo "=========================================="
echo "步骤3: 测试模型加载"
echo "=========================================="

python scripts/test_model_loading.py

# ============================================
# 步骤4: 验证机器人配置
# ============================================

echo ""
echo "=========================================="
echo "步骤4: 验证机器人配置"
echo "=========================================="

python scripts/test_robot_config.py

# ============================================
# 步骤5: 检查相机设备
# ============================================

echo ""
echo "=========================================="
echo "步骤5: 检查相机设备"
echo "=========================================="

echo "可用的相机设备:"
ls -la /dev/video* 2>/dev/null || echo "未找到相机设备"

# ============================================
# 步骤6: 启动系统
# ============================================

echo ""
echo "=========================================="
echo "步骤6: 启动系统"
echo "=========================================="

echo "使用一键启动脚本..."
echo ""
echo "命令: ./scripts/start_full_system.sh $CONFIG_FILE"
echo ""

# 询问用户是否继续
read -p "是否继续启动系统? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./scripts/start_full_system.sh "$CONFIG_FILE"
else
    echo "已取消启动"
    echo ""
    echo "手动启动步骤:"
    echo "1. 终端1: ./scripts/start_policy_server.sh"
    echo "2. 终端2: ./scripts/run_task_sequence.sh $CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
