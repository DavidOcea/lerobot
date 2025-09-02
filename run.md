# 基本信息
## ros2 controller项目
~/workspace/supre_robot_control
#### 主要脚本:
##### 启动遥操作 controller
```
./start_common_gripper_leader_follower.sh 
```
##### 启动双臂自主动作 controller
```
./start_common_follower_trajectory.sh
```
## lerobot项目
~/workspace/gitprj/lerobot-env/lerobot
#### 主要脚本:
##### 一键启动脚本
./start_robot.sh
##### 遥操作脚本
./run_teleop.sh
##### 环境设置脚本
./set_env.sh

### 启动遥操作
##### 1. 启动ros2 controller
```
cd ~/workspace/supre_robot_control
./start_common_gripper_leader_follower.sh
```
##### 2. 运行lerobot脚本
```
cd ~/workspace/gitprj/lerobot-env/lerobot
运行run_teleop.sh,lerobot命令可以自行更改
```

### 启动双臂自主动作
##### 1. 启动ros2 controller
```
cd ~/workspace/supre_robot_control
./start_common_follower_trajectory.sh
```
##### 2. 运行python脚本
```
cd ~/workspace/supre_robot_control
conda activate ros2_env
python ./test_dual_arm.py
```
##### 3. 使用一键脚本启动遥操作和双臂自主动作
```
cd ~/workspace/gitprj/lerobot-env/lerobot
./start_robot.sh
```

##### 4.遥操作前主从机械臂缓慢对齐
```
cd ~/workspace/gitprj/lerobot-env/lerobot
source ./set_env.sh
python ./src/utils/teleop_controller_switcher.py
```
## 意优电机驱动项目
~/workspace/eu_motor
#### 参考test_suite提示操作
```
cd ~/workspace/eu_motor
cd build/bin
sudo ./test_suite --dev 1 --motors 11 --tests status
```

## 夹爪项目
~/workspace/misumi_gripper
#### 参考run_jodell_gripper提示操作
```
cd ~/workspace/misumi_gripper
cd src/build
sudo ./run_jodell_gripper /dev/ttyTHS2 27
```