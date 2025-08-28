#!/bin/bash

# =================================================================
#  机器人任务启动器 (专业版)
#  - 检查多个ROS2 Topic来确认Controller启动成功
#  - 尝试在新终端标签页中打开日志，并提供手动备用方案
#  - 退出时自动清理所有后台进程和日志文件
# =================================================================

# --- 可配置变量 ---
# 日志文件的存放位置
LOG_FILE="/tmp/ros2_controller.log"

# --- 全局变量 ---
CONTROLLER_PID=""

# --- 清理函数 (现在只清理controller进程和日志文件) ---
cleanup() {
    echo ""
    echo "--- 收到退出信号，开始清理 ---"
    
    # 停止 ROS2 Controller 进程
    if [ -n "$CONTROLLER_PID" ]; then
        echo "正在停止ROS2 Controller (PID: $CONTROLLER_PID)..."
        kill -TERM $CONTROLLER_PID
        wait $CONTROLLER_PID 2>/dev/null
        echo "Controller已停止。"
    fi
    
    # 删除日志文件
    if [ -f "$LOG_FILE" ]; then
        rm "$LOG_FILE"
        echo "临时日志文件已删除。"
    fi
    
    echo "清理完毕，退出。"
    exit 0
}

# 设置 trap，捕获退出信号并调用 cleanup 函数
trap cleanup EXIT INT TERM

# --- 主菜单 ---
echo "请选择要执行的任务:"
echo "  1) 启动遥操作 (Teleoperation)"
echo "  2) 启动双臂自主动作 (Autonomous Action)"
read -p "请输入选项 [1-2]: " choice

# 清理旧的日志文件（如果存在）
rm -f "$LOG_FILE"
touch "$LOG_FILE"

# 定义需要检查的Topic列表 (Bash数组)
declare -a EXPECTED_TOPICS

case $choice in
    1)
        # --- 任务1: 启动遥操作 ---
        echo "--- 正在启动遥操作 ---"
        cd ~/workspace/supre_robot_control
        
        echo "[步骤 1/3] 在后台启动ROS2 Controller，日志输出到: $LOG_FILE"
        ./start_common_gripper_leader_follower.sh > "$LOG_FILE" 2>&1 &
        CONTROLLER_PID=$!
        
        # ⚠️ 【重要】请根据你的Controller实际情况修改这里的Topic列表！
        EXPECTED_TOPICS=("/supre_robot_follower/left_arm_controller/commands" "/supre_robot_follower/right_arm_controller/commands" "supre_robot_leader/joint_states")
        ;;
    2)
        # --- 任务2: 启动双臂自主动作 ---
        echo "--- 正在启动双臂自主动作 ---"
        cd ~/workspace/supre_robot_control
        
        echo "[步骤 1/3] 在后台启动ROS2 Controller，日志输出到: $LOG_FILE"
        ./start_common_follower_trajectory.sh > "$LOG_FILE" 2>&1 &
        CONTROLLER_PID=$!
        
        # ⚠️ 【重要】请根据你的Controller实际情况修改这里的Topic列表！
        EXPECTED_TOPICS=("/supre_robot_follower/left_arm_controller/commands" "/supre_robot_follower/right_arm_controller/commands")
        ;;
    *)
        echo "无效选项，退出。"
        exit 1
        ;;
esac

# --- [新功能] 检查多个Topic是否都已启动 ---
echo "[步骤 2/3] 等待ROS2 Controller启动... (将检查 ${#EXPECTED_TOPICS[@]} 个Topics)"
WAIT_TIMEOUT=30 # 最长等待30秒
elapsed_time=0
all_topics_found=false

# 必须先source ROS2环境才能使用 ros2 topic list 命令
# 假设你的环境在 ros2_env conda环境中
echo "正在激活Conda环境以使用ROS2命令..."
source ~/miniconda3/etc/profile.d/conda.sh # 根据你的路径修改
source /opt/ros/humble/setup.bash

conda activate ros2_env

while [ $elapsed_time -lt $WAIT_TIMEOUT ]; do
    found_count=0
    # 优化：只调用一次 ros2 topic list，提高效率
    current_topics=$(ros2 topic list)
    echo "当前ROS2 Topics:"
    echo "$current_topics"
    for topic in "${EXPECTED_TOPICS[@]}"; do
        # 使用 -w 选项进行全词匹配，避免 /joint_states 匹配到 /joint_states_extra 等
        if echo "$current_topics" | grep -q -w "$topic"; then
            found_count=$((found_count + 1))
        fi
    done

    echo "进度: 已找到 $found_count / ${#EXPECTED_TOPICS[@]} 个预期的Topics..."

    if [ "$found_count" -eq "${#EXPECTED_TOPICS[@]}" ]; then
        echo "✅ Controller启动成功！所有Topics均已找到。"
        all_topics_found=true
        break
    fi
    
    sleep 1
    elapsed_time=$((elapsed_time + 1))
done

if [ "$all_topics_found" = false ]; then
    echo "❌ 错误: 在 $WAIT_TIMEOUT 秒内未能找到所有预期的Topics。"
    echo "请检查日志 $LOG_FILE 以获取详细错误信息。脚本将退出。"
    # 退出会触发 cleanup 函数
    exit 1
fi

# --- [新功能] 尝试在新标签页中打开日志 ---
echo ""
echo "Controller正在后台运行。日志文件位于: $LOG_FILE"

# 检查 gnome-terminal 是否可用
if command -v gnome-terminal &> /dev/null; then
    echo "正在尝试在新的终端标签页中打开日志..."
    # --tab 会打开一个新标签页
    # --title 会设置标签页的标题
    # -- bash -c "COMMANDS" 是在新标签页中执行命令的标准方式
    # "exec bash" 能让标签页在 tail 命令结束后（例如按Ctrl+C）依然保持打开
    gnome-terminal --tab --title="Controller Log" -- bash -c "echo '>>> 正在显示日志: $LOG_FILE'; echo '>>> 按 Ctrl+C 停止追踪，此标签页不会关闭。'; tail -f '$LOG_FILE'; exec bash"
else
    # 如果 gnome-terminal 不存在，则提供手动操作指南
    echo "------------------------------------------------------------"
    echo "未能自动打开日志窗口。"
    echo "要实时查看Controller的日志, 请手动打开一个新的终端/标签页"
    echo "然后运行以下命令:"
    echo ""
    echo "    tail -f $LOG_FILE"
    echo "------------------------------------------------------------"
fi


echo ""
echo "------------------------------------------------------------"
echo "现在启动主程序..."
echo "按 Ctrl+C 退出主程序时，所有后台进程将被自动清理。"
echo "------------------------------------------------------------"

# --- [步骤 3/3] 运行主程序 ---
if [ "$choice" -eq 1 ]; then
    cd ~/workspace/gitprj/lerobot-env/lerobot
    ./run_teleop.sh
else
    cd ~/workspace/supre_robot_control
    # 无需再次 conda activate，因为当前shell环境已经激活
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6 python    
    python ./test_dual_arm.py
fi

# 脚本正常结束后也会触发 trap cleanup