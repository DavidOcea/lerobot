#!/bin/bash
# watchdog_train.sh — 监控训练日志是否卡死，超时自动重启
# 用法: nohup bash watchdog_train.sh <logfile> <restart_command> &
#
# 每 5 分钟检查一次日志文件是否有更新。如果超过 30 分钟没新日志
# （训练可能已挂），自动用 restart_command 重启训练。
# 支持从 checkpoint 恢复（需命令中已有 --resume 逻辑）。

LOGFILE="$1"
RESTART_CMD="$2"
STALE_TIMEOUT_SEC=$((30 * 60))  # 30 分钟无更新 → 判定为死
CHECK_INTERVAL_SEC=$((5 * 60))   # 每 5 分钟检查一次

if [ -z "$LOGFILE" ] || [ -z "$RESTART_CMD" ]; then
    echo "用法: watchdog_train.sh <logfile> <restart_command>"
    echo "示例: watchdog_train.sh act_xxx.log 'bash scripts/train_xxx.sh'"
    exit 1
fi

echo "[watchdog] 监控日志: $LOGFILE"
echo "[watchdog] 重启命令: $RESTART_CMD"
echo "[watchdog] 超时阈值: ${STALE_TIMEOUT_SEC}s (${STALE_TIMEOUT_SEC}s)"
echo "[watchdog] PID: $$"

LAST_SIZE=$(stat -c%s "$LOGFILE" 2>/dev/null || echo 0)

while true; do
    sleep "$CHECK_INTERVAL_SEC"

    CURRENT_SIZE=$(stat -c%s "$LOGFILE" 2>/dev/null || echo 0)
    CURRENT_TIME=$(date +%s)
    FILE_MTIME=$(stat -c%Y "$LOGFILE" 2>/dev/null || echo 0)
    AGE=$((CURRENT_TIME - FILE_MTIME))

    if [ "$CURRENT_SIZE" -gt "$LAST_SIZE" ]; then
        # 日志在增长 — 训练正常
        LAST_SIZE="$CURRENT_SIZE"
    elif [ "$AGE" -gt "$STALE_TIMEOUT_SEC" ]; then
        echo "[watchdog] $(date): 日志 $LOGFILE 已 ${AGE}s 未更新，尝试重启..."
        eval "$RESTART_CMD"
    else
        # 日志没增长但还没超时 — 可能在加载数据
        echo "[watchdog] $(date): 日志未更新 (${AGE}s)，未超时，继续等待"
    fi
done
