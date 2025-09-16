cd ~/workspace/gitprj/lerobot-env/lerobot
source ~/miniconda3/etc/profile.d/conda.sh
echo "Sourcing ROS 2 Humble environment..."
source /opt/ros/humble/setup.bash
source ./setup_binding_env.sh
python ./src/lerobot/teleop_trajectory_align.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower