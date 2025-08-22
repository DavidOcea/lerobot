source ./setup_env.sh
python -m lerobot.teleoperate \
--robot.type=ros2_dual_follower \
--robot.id=eyou_follower \
--teleop.type=ros2_dual_leader \
--teleop.id=eyou_leader
