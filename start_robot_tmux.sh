#!/bin/bash

# =================================================================
#  机器人任务启动器 (TMUX 专业版)
#  - 使用TMUX创建隔离的会话和窗口来管理进程
# =================================================================

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux 未安装。请运行 'sudo apt install tmux' 进行安装。"
    exit 1
fi

SESSION_NAME="robot_session"

# --- 主菜单 ---
echo "请选择要执行的任务:"
echo "  1) 启动遥操作 (Teleoperation)"
echo "  2) 启动双臂自主动作 (Autonomous Action)"
read -p "请输入选项 [1-2]: " choice

# 确保没有同名会话在运行，避免冲突
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
echo "正在创建新的TMUX会话: $SESSION_NAME"

# 创建一个新的、分离的tmux会话，并命名第一个窗口
tmux new-session -d -s $SESSION_NAME -n "ROS_Controller"

# 根据用户选择发送命令
case $choice in
    1)
        # --- 任务1: 启动遥操作 ---
        echo "配置遥操作任务..."
        # 在第一个窗口(0)中启动Controller
        tmux send-keys -t "${SESSION_NAME}:0" "cd ~/workspace/supre_robot_control && ./start_common_gripper_leader_follower.sh" C-m
        
        # 创建第二个窗口并命名
        tmux new-window -t $SESSION_NAME -n "LeRobot_Teleop"
        # 在第二个窗口(1)中运行lerobot脚本
        tmux send-keys -t "${SESSION_NAME}:1" "cd ~/workspace/gitprj/lerobot-env/lerobot && ./run_teleop.sh" C-m
        ;;
    2)
        # --- 任务2: 启动双臂自主动作 ---
        echo "配置自主动作任务..."
        # 在第一个窗口(0)中启动Controller
        tmux send-keys -t "${SESSION_NAME}:0" "cd ~/workspace/supre_robot_control && ./start_common_follower_trajectory.sh" C-m
        
        # 创建第二个窗口并命名
        tmux new-window -t $SESSION_NAME -n "Python_Autonomous"
        # 在第二个窗口(1)中运行Python脚本
        tmux send-keys -t "${SESSION_NAME}:1" "cd ~/workspace/supre_robot_control && conda run -n ros2_env python ./test_dual_arm.py" C-m
        ;;
    *)
        echo "无效选项，正在关闭TMUX会话。"
        tmux kill-session -t $SESSION_NAME
        exit 1
        ;;
esac

echo ""
echo "✅ 任务已在TMUX会话 '$SESSION_NAME' 中启动！"
echo ""
echo "--- 如何操作 ---"
echo "  - 查看所有进程: tmux attach -t $SESSION_NAME"
echo "  - 在TMUX中切换窗口: Ctrl+B, 然后按窗口号 (0 或 1)"
echo "  - 从TMUX中分离 (让程序继续后台运行): Ctrl+B, 然后按 D"
echo "  - **彻底停止所有任务**: tmux kill-session -t $SESSION_NAME"