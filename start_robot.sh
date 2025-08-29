#!/bin/bash

# =================================================================
#  机器人任务启动器 (交互式专业版)
#  - 在【新的终端标签页】中【前台】运行ROS2 Controller
#  - 主脚本会等待Controller进程结束
#  - 退出时自动清理
# =================================================================

# --- 可配置变量 ---
# 日志文件的存放位置 (虽然不在主脚本中使用，但Controller脚本内部可能需要)
LOG_FILE="/tmp/ros2_controller.log"

# --- 全局变量 ---
# 这个脚本不再需要管理PID，因为进程是前台的
# 但我们仍然设置trap以防万一
cleanup() {
    echo ""
    echo "--- 主脚本退出 ---"
    # 如果有其他需要清理的资源，可以在这里添加
    exit 0
}

# 设置 trap
trap cleanup EXIT INT TERM

# --- 主菜单 ---
echo "请选择要执行的任务:"
echo "  1) 启动遥操作 (Teleoperation)"
echo "  2) 启动双臂自主动作 (Autonomous Action)"
read -p "请输入选项 [1-2]: " choice

# 定义需要检查的Topic列表 (Bash数组)
declare -a EXPECTED_TOPICS
declare CONTROLLER_SCRIPT=""
declare CONTROLLER_DIR=""

case $choice in
    1)
        # --- 任务1: 启动遥操作 ---
        CONTROLLER_SCRIPT="./start_common_gripper_leader_follower.sh"
        CONTROLLER_DIR="~/workspace/supre_robot_control"
        EXPECTED_TOPICS=(
            "/supre_robot_leader/joint_states"
            "/supre_robot_follower/joint_states"
        )
        ;;
    2)
        # --- 任务2: 启动双臂自主动作 ---
        CONTROLLER_SCRIPT="./start_common_follower_trajectory.sh"
        CONTROLLER_DIR="~/workspace/supre_robot_control"
        EXPECTED_TOPICS=(
            "/supre_robot_follower/joint_states"
            "/supre_robot_follower/left_arm_trajectory_controller/joint_trajectory"
            "/supre_robot_follower/right_arm_trajectory_controller/joint_trajectory"
        )
        ;;
    *)
        echo "无效选项，退出。"
        exit 1
        ;;
esac

# --- [新逻辑] 启动Controller在新标签页 ---
echo "[步骤 1/2] 准备在新终端标签页中启动ROS2 Controller..."

# 检查 gnome-terminal 是否可用
if ! command -v gnome-terminal &> /dev/null; then
    echo "❌ 错误: gnome-terminal 命令未找到。"
    echo "此脚本需要 GNOME Terminal。如果你使用其他终端，请手动运行以下命令："
    echo "    cd $CONTROLLER_DIR"
    echo "    $CONTROLLER_SCRIPT"
    exit 1
fi

# --tab 会打开一个新标签页
# --title 会设置标签页的标题
# -- bash -c "COMMANDS; exec bash" 是在新标签页中执行命令的标准方式
#    - 'cd ... && ...' 确保在正确的目录下执行
#    - 'exec bash' 让标签页在Controller脚本结束后（例如按Ctrl+C）依然保持打开，方便查看最终日志
COMMAND_TO_RUN="cd ${CONTROLLER_DIR} && ${CONTROLLER_SCRIPT}; exec bash"
gnome-terminal --tab --title="ROS2 Controller" -- bash -c "$COMMAND_TO_RUN"

# --- 检查Controller是否启动成功 ---
echo "[步骤 2/2] 等待ROS2 Controller启动... (将检查 ${#EXPECTED_TOPICS[@]} 个Topics)"
WAIT_TIMEOUT=30
elapsed_time=0
all_topics_found=false

echo "正在激活Conda环境以使用ROS2命令..."
source ~/miniconda3/etc/profile.d/conda.sh # 根据你的路径修改
conda activate ros2_env

while [ $elapsed_time -lt $WAIT_TIMEOUT ]; do
    found_count=0
    current_topics=$(ros2 topic list)

    for topic in "${EXPECTED_TOPICS[@]}"; do
        if echo "$current_topics" | grep -q -w "$topic"; then
            found_count=$((found_count + 1))
        fi
    done

    # 提供清晰的进度反馈
    echo -ne "进度: 已找到 $found_count / ${#EXPECTED_TOPICS[@]} 个Topics... ($elapsed_time/$WAIT_TIMEOUT s)\r"

    if [ "$found_count" -eq "${#EXPECTED_TOPICS[@]}" ]; then
        echo -e "\n✅ Controller启动成功！所有Topics均已找到。"
        all_topics_found=true
        break
    fi
    
    sleep 1
    elapsed_time=$((elapsed_time + 1))
done

# 清理进度行
echo ""

if [ "$all_topics_found" = false ]; then
    echo "❌ 错误: 在 $WAIT_TIMEOUT 秒内未能找到所有预期的Topics。"
    echo "请检查新打开的'ROS2 Controller'标签页以获取详细错误信息。脚本将退出。"
    exit 1
fi


echo "------------------------------------------------------------"
echo "Controller正在新的标签页中运行。"
echo "请在新标签页中按 Ctrl+C 来停止Controller和主程序。"
echo "现在，将在此窗口启动主程序..."
echo "------------------------------------------------------------"

# --- 运行主程序 ---
if [ "$choice" -eq 1 ]; then
    cd ~/workspace/gitprj/lerobot-env/lerobot
    # 注意：这里我们不再需要后台运行主程序，因为Controller在另一个Tab
    # 当主程序结束时，整个脚本也就结束了
    ./run_teleop.sh
else
    cd ~/workspace/supre_robot_control
    python ./test_dual_arm.py
fi

# 当主程序（如run_teleop.sh）结束后，脚本会到达这里
echo ""
echo "--- 主程序已结束 ---"
echo "请记得在新打开的 'ROS2 Controller' 标签页中按 Ctrl+C 来彻底关闭所有进程。"

# 脚本正常结束后会触发 trap cleanup