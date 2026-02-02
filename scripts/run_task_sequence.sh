#!/bin/bash
# 任务序列执行脚本

set -e

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file.yaml> [options]"
    echo ""
    echo "Options:"
    echo "  --debug        启用调试模式"
    echo "  --dry-run       仅验证配置，不执行"
    echo "  --tasks TASKS   指定要运行的任务（逗号分隔）"
    echo ""
    echo "Example:"
    echo "  $0 configs/task_agent_tasks_test.yaml"
    echo "  $0 configs/task_agent_tasks.yaml --debug"
    echo "  $0 configs/task_agent_tasks.yaml --tasks pick_short,pick_long"
    exit 1
fi

CONFIG_FILE=$1
shift

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

echo "=========================================="
echo "启动任务序列执行"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "项目目录: $PROJECT_ROOT"
echo "附加选项: $@"
echo "=========================================="

# 运行任务序列
python src/lerobot/scripts/run_task_agent.py \
    --config "$CONFIG_FILE" \
    "$@"

echo "=========================================="
echo "任务序列执行完成"
echo "=========================================="
