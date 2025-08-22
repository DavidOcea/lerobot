## 1. 启动ros2 controller
cd ~/workspace/supre_robot_control
./start_common_leader_follower.sh
## 2. 运行lerobot脚本
cd ~/workspace/gitprj/lerobot_env/lerobot
运行下面脚本,lerobot命令可以自行更改

```bash
#!/bin/bash
conda activate ros2_env
sudo chmod 666 /dev/ttyTHS1
sudo chmod 666 /dev/ttyTHS2
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6 python
python -m lerobot.teleoperate \
--robot.type=ros2_dual_follower \
--robot.id=eyou_follower \
--teleop.type=ros2_dual_leader \
--teleop.id=eyou_leader
```