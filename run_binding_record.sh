## 1. 启动ros2 controller
source ~/miniconda3/etc/profile.d/conda.sh
source ./setup_binding_env.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 构建配置文件的绝对路径
CONFIG_FILE_PATH="$SCRIPT_DIR/src/lerobot/teleoperators/supre_robot_leader/trunk_teleoperate.yaml"

python ./src/lerobot/teleop_trajectory_align.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower
    --config_path="$CONFIG_FILE_PATH"

python -m lerobot.record \
    --robot.type=supre_robot_follower \
    --robot.id=supre_robot_follower \
    --teleop.type=supre_robot_leader \
    --teleop.id=supre_robot_leader \
    --robot.cameras="{head_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_cam: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.single_task="Grasp the workpiece and put it in the appropriate position." \
    --dataset.repo_id=supdata/dataset_0917_1 \
    --dataset.episode_time_s=30 \
    --dataset.num_episodes=2 \
    --dataset.reset_time_s=10 \
    --config_path="$CONFIG_FILE_PATH"    