source ~/miniconda3/etc/profile.d/conda.sh
source ./setup_binding_env.sh
python your_script_name.py \
    --teleop.type=supre_robot_leader \
    --robot.type=supre_robot_follower

python -m lerobot.teleoperate \
--robot.type=ros2_dual_follower \
--robot.id=eyou_follower \
--teleop.type=ros2_dual_leader \
--teleop.id=eyou_leader
