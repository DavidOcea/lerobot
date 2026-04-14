# Task Agent System 使用指南

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Task Agent Orchestrator                  │
│  - 任务调度执行                                              │
│  - 碰撞检测与恢复                                            │
│  - 任务完成检测                                              │
│  - 相机动态切换                                              │
└─────────────────────────────────────────────────────────────┘
                            │ gRPC
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Policy Server                             │
│  - ACT策略推理                                              │
│  - 动作生成                                                  │
└─────────────────────────────────────────────────────────────┘
                            │ gRPC
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Robot Client                              │
│  - 控制循环                                                  │
│  - 观测发送                                                  │
│  - 动作执行                                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SupreRobot Follower (16关节双臂机器人)          │
│  - 5个相机 (head, left_wrist, left_wrist2, right_wrist,     │
│          right_wrist2)                                      │
│  - 力矩传感                                                  │
└─────────────────────────────────────────────────────────────┘
```

## 使用流程

### 方式一：使用 Task Agent Orchestrator（推荐）

这种方式自动管理整个任务序列，包括：
- 任务调度
- 策略切换
- 相机切换
- 碰撞检测与恢复
- 任务完成检测
- 失败重试

#### 步骤1: 准备配置文件

编辑 `configs/task_agent_tasks.yaml`：

```yaml
# 任务配置
tasks:
  - name: "pick_short_workpiece"
    policy_path: "/path/to/your/act_model_short"
    policy_type: "act"
    max_duration: 30.0
    max_retries: 3
    enabled: true
    cameras:
      - name: "head_cam"
        type: "opencv"
        index: 0
        width: 640
        height: 480
        fps: 30
      # ... 其他相机

  # 更多任务...

# 机器人配置
robot_config:
  type: "supre_robot_follower"
  config_path: "src/lerobot/robots/supre_robot_follower/trunk_config.yaml"
  camera_enabled: true
  force_sensing_enabled: true
  control_frequency: 30.0

# 碰撞检测配置
collision_config:
  collision_threshold: 2.0  # Nm
  detection_window: 5
  recovery_strategy: "stop_and_retreat"
```

#### 步骤2: 启动 Policy Server

```bash
# 终端1: 启动策略服务器
conda activate lerobot

python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --robot.config_path=src/lerobot/robots/supre_robot_follower/trunk_config.yaml \
    --port=50051 \
    --fps=30
```

#### 步骤3: 运行任务序列

```bash
# 终端2: 启动任务执行
conda activate lerobot

python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml
```

### 方式二：手动控制（单任务执行）

如果需要手动控制单个任务：

#### 步骤1: 启动 Policy Server

```bash
conda activate lerobot

python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --robot.config_path=src/lerobot/robots/supre_robot_follower/trunk_config.yaml \
    --port=50051 \
    --fps=30
```

#### 步骤2: 启动 Robot Client

```bash
conda activate lerobot

python src/lerobot/scripts/server/robot_client.py \
    --robot.type=supre_robot_follower \
    --robot.joint_config_file=trunk_config_supre_robot_joint.yaml \
    --robot.cameras="{
        head: {type: opencv, index: 0, width: 640, height: 480, fps: 30},
        right_wrist_cam: {type: opencv, index: 6, width: 640, height: 480, fps: 30}
    }" \
    --robot.id=follower \
    --task="pick_short_workpiece" \
    --server_address=127.0.0.1:50051 \
    --policy_type=act \
    --pretrained_name_or_path=/path/to/your/act_model \
    --policy_device=cuda \
    --actions_per_chunk=100
```

## 完整启动脚本

为了方便使用，创建启动脚本：

### 1. 策略服务器启动脚本

创建 `scripts/start_policy_server.sh`:

```bash
#!/bin/bash

conda activate lerobot

python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --robot.config_path=src/lerobot/robots/supre_robot_follower/trunk_config.yaml \
    --port=50051 \
    --fps=30 \
    --device=cuda
```

### 2. 任务执行启动脚本

创建 `scripts/run_task_sequence.sh`:

```bash
#!/bin/bash

CONFIG_FILE=${1:-"configs/task_agent_tasks.yaml"}

conda activate lerobot

python src/lerobot/scripts/run_task_agent.py \
    --config "$CONFIG_FILE" \
    --debug
```

### 3. 一键启动脚本（推荐）

创建 `scripts/start_full_system.sh`:

```bash
#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file.yaml>"
    echo "Example: $0 configs/task_agent_tasks_test.yaml"
    exit 1
fi

CONFIG_FILE=$1

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "启动 Task Agent 系统"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "项目目录: $PROJECT_ROOT"
echo "=========================================="

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# 启动 Policy Server (后台)
echo "启动 Policy Server..."
python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --robot.config_path=src/lerobot/robots/supre_robot_follower/trunk_config.yaml \
    --port=50051 \
    --fps=30 \
    --device=cuda &
POLICY_PID=$!
echo "Policy Server PID: $POLICY_PID"

# 等待服务器启动
echo "等待 Policy Server 启动..."
sleep 5

# 运行任务序列
echo "启动任务序列..."
python src/lerobot/scripts/run_task_agent.py \
    --config "$CONFIG_FILE"

# 清理
echo "停止 Policy Server..."
kill $POLICY_PID 2>/dev/null

echo "系统执行完成"
```

使用方法：
```bash
chmod +x scripts/start_full_system.sh
./scripts/start_full_system.sh configs/task_agent_tasks_test.yaml
```

## 调试模式

### 查看详细日志

```bash
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml \
    --debug
```

### Dry Run（仅验证配置）

```bash
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml \
    --dry-run
```

### 运行特定任务

```bash
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml \
    --tasks pick_short_workpiece,pick_long_workpiece
```

### 覆盖重试次数

```bash
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml \
    --max-retries 5
```

## 常见问题排查

### 1. 策略服务器连接失败

```bash
# 检查端口是否被占用
lsof -i :50051

# 检查服务器是否运行
ps aux | grep policy_server
```

### 2. 相机无法打开

```bash
# 检查相机设备
ls -la /dev/video*

# 测试相机
ffplay /dev/video0
```

### 3. CAN 总线连接问题

```bash
# 检查 CAN 设备
ip link show

# 查看 CAN 总线状态
candump can0
```

### 4. 查看日志

```bash
# Policy Server 日志
tail -f /tmp/policy_server.log

# Robot Client 日志
tail -f /tmp/robot_client.log
```

## 性能监控

### Prometheus 监控

```bash
# 启动 Prometheus
prometheus --config.file=prometheus.yml

# 访问监控面板
# http://localhost:9090
```

### 查看指标

```bash
# 查看关节位置
curl http://localhost:8000/metrics | grep joint_position

# 查看关节力矩
curl http://localhost:8000/metrics | grep joint_force
```

## 系统要求

### 硬件
- 双臂机器人（16关节）
- CAN 总线接口
- 5个 USB 摄像头
- GPU (CUDA 支持)

### 软件
- Python 3.10+
- PyTorch 2.0+
- conda 环境: lerobot

### 依赖安装

```bash
conda activate lerobot
pip install lerobot
pip install draccus opencv-python pyyaml
```

## 下一步

1. 根据实际硬件修改 `trunk_config_supre_robot_joint.yaml` 中的设备路径
2. 训练或获取 ACT 策略模型
3. 配置任务序列 YAML 文件
4. 运行测试验证系统
5. 部署到生产环境
