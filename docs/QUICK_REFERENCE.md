# Task Agent 快速参考

## 一键启动（推荐）

```bash
# 运行完整系统
./scripts/start_full_system.sh configs/task_agent_tasks_test.yaml

# 调试模式
./scripts/start_full_system.sh configs/task_agent_tasks_test.yaml --debug

# 运行特定任务
./scripts/start_full_system.sh configs/task_agent_tasks.yaml --tasks pick_short,pick_long
```

## 分步启动

### 步骤1: 启动 Policy Server（终端1）
```bash
./scripts/start_policy_server.sh
```

### 步骤2: 运行任务序列（终端2）
```bash
./scripts/run_task_sequence.sh configs/task_agent_tasks_test.yaml
```

## 测试验证

```bash
# 测试模型加载
./scripts/test_model_loading.py

# 测试机器人配置
./scripts/test_robot_config.py

# 完整工作流示例
./scripts/example_workflow.sh
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs/task_agent_tasks.yaml` | 完整任务配置 |
| `configs/task_agent_tasks_test.yaml` | 测试配置 |
| `src/lerobot/robots/supre_robot_follower/trunk_config.yaml` | 机器人配置 |
| `src/lerobot/robots/supre_robot_follower/trunk_config_supre_robot_joint.yaml` | 关节配置 |

## 相机索引

| 相机 | 索引 | 用途 |
|------|------|------|
| head_cam | 0 | 主视角 |
| left_wrist_cam | 2 | 左手腕 |
| left_wrist_cam2 | 4 | 左手腕2 |
| right_wrist_cam | 6 | 右手腕 |
| right_wrist_cam2 | 8 | 右手腕2 |

## 任务类型

| 任务 | 使用相机 | 手臂 |
|------|----------|------|
| pick_short_workpiece | head, right_wrist, right_wrist_cam2 | 右手 |
| place_short_workpiece | head, right_wrist, right_wrist_cam2 | 右手 |
| pick_long_workpiece | head, left_wrist, left_wrist_cam2 | 左手 |
| place_long_workpiece | head, left_wrist, left_wrist_cam2 | 左手 |
| press_button | head, right_wrist, right_wrist_cam2 | 右手 |

## 常用命令

```bash
# 查看进程
ps aux | grep policy_server
ps aux | grep run_task_agent

# 查看端口
lsof -i :50051

# 查看相机
ls -la /dev/video*

# 查看日志
tail -f /tmp/policy_server.log

# 停止所有服务
pkill -f policy_server
pkill -f run_task_agent
```

## 故障排查

### 问题：无法连接到 Policy Server
```bash
# 检查端口
netstat -tuln | grep 50051

# 检查进程
ps aux | grep policy_server

# 重启服务器
./scripts/start_policy_server.sh
```

### 问题：相机无法打开
```bash
# 检查相机权限
ls -la /dev/video*

# 测试相机
ffplay /dev/video0

# 检查配置
cat src/lerobot/robots/supre_robot_follower/trunk_config.yaml | grep cameras
```

### 问题：CAN 总线连接失败
```bash
# 检查 CAN 接口
ip link show | grep can

# 查看 CAN 状态
candump can0

# 检查配置
cat src/lerobot/robots/supre_robot_follower/trunk_config_supre_robot_joint.yaml | grep can_device
```

## 修改配置

### 更改模型路径
编辑 `configs/task_agent_tasks_test.yaml`:
```yaml
tasks:
  - name: "test_movement"
    policy_path: "/your/new/model/path"
```

### 更改相机索引
编辑 `src/lerobot/robots/supre_robot_follower/trunk_config.yaml`:
```yaml
cameras:
  head_cam:
    index: 0  # 修改为你的相机索引
```

### 调整碰撞阈值
编辑 YAML 配置文件:
```yaml
collision_config:
  collision_threshold: 2.0  # Nm，根据需要调整
```

## 目录结构

```
lerobot_origin/
├── configs/
│   ├── task_agent_tasks.yaml
│   └── task_agent_tasks_test.yaml
├── docs/
│   └── USAGE_GUIDE.md
├── scripts/
│   ├── start_policy_server.sh
│   ├── run_task_sequence.sh
│   ├── start_full_system.sh
│   ├── example_workflow.sh
│   ├── test_model_loading.py
│   └── test_robot_config.py
└── src/
    └── lerobot/
        ├── agent/
        ├── monitoring/
        ├── safety/
        ├── tasks/
        └── robots/
            └── supre_robot_follower/
                ├── trunk_config.yaml
                └── trunk_config_supre_robot_joint.yaml
```
