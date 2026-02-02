#!/bin/bash
# 一键启动完整系统（Policy Server + Task Agent）

set -e

# 检查参数
if [ $# -lt 1 ]; then
    echo "=========================================="
    echo "Task Agent 系统一键启动"
    echo "=========================================="
    echo "Usage: $0 <config_file.yaml> [options]"
    echo ""
    echo "Options:"
    echo "  --debug        启用调试模式"
    echo "  --tasks TASKS   指定要运行的任务"
    echo ""
    echo "Example:"
    echo "  $0 configs/task_agent_tasks_test.yaml"
    echo "  $0 configs/task_agent_tasks.yaml --debug"
    echo ""
    exit 1
fi

CONFIG_FILE=$1
shift

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Task Agent 系统"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "项目目录: $PROJECT_ROOT"
echo "=========================================="

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 清理函数
cleanup() {
    echo ""
    echo "=========================================="
    echo "停止所有服务..."
    echo "=========================================="

    # 停止 Policy Server
    if [ -n "$POLICY_PID" ]; then
        echo "停止 Policy Server (PID: $POLICY_PID)..."
        kill $POLICY_PID 2>/dev/null || true
        wait $POLICY_PID 2>/dev/null || true
    fi

    echo "所有服务已停止"
    echo "=========================================="
}

# 设置退出时清理
trap cleanup EXIT INT TERM

# 启动 Policy Server (后台)
echo "启动 Policy Server..."
python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --robot.config_path=src/lerobot/robots/supre_robot_follower/trunk_config.yaml \
    --port=50051 \
    --fps=30 \
    --device=cuda > /tmp/policy_server.log 2>&1 &
POLICY_PID=$!
echo "Policy Server 已启动 (PID: $POLICY_PID)"

# 等待服务器启动
echo "等待 Policy Server 准备就绪..."
sleep 8

# 检查服务器是否运行
if ! kill -0 $POLICY_PID 2>/dev/null; then
    echo "错误: Policy Server 启动失败"
    echo "查看日志: cat /tmp/policy_server.log"
    exit 1
fi

echo "Policy Server 已就绪"
echo "=========================================="

# 运行任务序列
echo "启动任务序列执行..."
echo "=========================================="

python src/lerobot/scripts/run_task_agent.py \
    --config "$CONFIG_FILE" \
    "$@"

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "任务序列执行成功!"
else
    echo "任务序列执行失败 (退出码: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
