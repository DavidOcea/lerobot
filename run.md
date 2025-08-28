# 启动遥操作
## 1. 启动ros2 controller
cd ~/workspace/supre_robot_control
./start_common_gripper_leader_follower.sh
## 2. 运行lerobot脚本
cd ~/workspace/gitprj/lerobot-env/lerobot
运行run_teleop.sh,lerobot命令可以自行更改

# 启动双臂自主动作
## 1. 启动ros2 controller
cd ~/workspace/supre_robot_control
./start_common_follower_trajectory.sh
## 2. 运行python脚本
cd ~/workspace/supre_robot_control
conda activate ros2_env
python ./test_dual_arm.py