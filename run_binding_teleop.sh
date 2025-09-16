source ~/miniconda3/etc/profile.d/conda.sh
source ./setup_binding_env.sh
python ./src/lerobot/teleop_trajectory_align.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower

python -m lerobot.teleoperate \
--robot.type=supre_robot_follower \
--robot.id=supre_follower \
--teleop.type=supre_robot_leader \
--teleop.id=supre_leader
