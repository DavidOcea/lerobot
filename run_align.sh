source ~/miniconda3/etc/profile.d/conda.sh
source ./setup_binding_env.sh
python ./src/lerobot/teleop_trajectory_align.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower