source ~/miniconda3/etc/profile.d/conda.sh
source ./setup_binding_env.sh
python ./src/lerobot/teleop_trajectory_align.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower

# 构建配置文件的绝对路径
CONFIG_FILE_PATH="$SCRIPT_DIR/src/lerobot/teleoperators/supre_robot_leader/trunk_teleoperate.yaml"

# 运行 Python 程序
echo "Using config file: $CONFIG_FILE_PATH"
python -m lerobot.teleoperate \
    --robot.type=supre_robot_follower \
    --robot.id=supre_follower \
    --teleop.type=supre_robot_leader \
    --teleop.id=supre_leader \
    --config_path="$CONFIG_FILE_PATH"