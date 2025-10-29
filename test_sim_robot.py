# 测试仿真机器人集成
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig

# 初始化配置
config = SimRobotConfig(headless=False)

# 创建机器人实例
robot = SimRobot(config)
robot.connect()

# 获取观测
obs = robot.get_observation()
print("观测包含:", obs.keys())
print("关节状态:", {k: v for k, v in obs.items() if k.endswith(".pos")})
print("相机图像形状:", {k: v.shape for k, v in obs.items() if k in ["head_cam", "right_wrist_cam", "left_wrist_cam"]})

# 发送测试动作
action = {f"{name}.pos": 0.0 for name in robot.joint_names}
robot.send_action(action)

# 断开连接
robot.disconnect()