# Task Agent 架构说明

## 系统架构对比

### 原架构（Remote 模式 - 需要 Policy Server）

```
┌─────────────────┐         gRPC         ┌──────────────────┐
│ Robot Client    │◄─────────────────────►│  Policy Server   │
│ (控制循环)       │                         │ (异步推理)       │
└────────┬────────┘                         └──────────┬───────┘
         │                                            │
         ▼                                            ▼
┌─────────────────┐                         ┌──────────────────┐
│ SupreRobot      │                         │ ACT Policy Model  │
│ Follower        │                         │ (GPU 推理)        │
└─────────────────┘                         └──────────────────┘
```

**缺点**：
- 需要启动两个进程
- gRPC 通信延迟
- 部署复杂

---

### 新架构（Local 模式 - 推荐）

```
┌────────────────────────────────────────────────────────────────┐
│                     Task Agent Orchestrator                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │TaskScheduler │  │CollisionDet. │  │CompletionDet.│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                │                │                    │
│         ▼                ▼                ▼                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           LocalPolicyExecutor (本地推理)              │  │
│  │   - 直接加载 ACT 模型                                  │  │
│  │   - 本地 GPU 推理                                       │  │
│  │   - 无网络延迟                                         │  │
│  └───────────────────────────┬─────────────────────────────┘  │
│                                │                              │
└────────────────────────────────┼──────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SupreRobot Follower (双臂14关节机器人)         │
│           传感器: Position, Force (已支持)                      │
└─────────────────────────────────────────────────────────────────┘
```

**优点**：
- 单进程部署，简单
- 无网络延迟
- 更低的系统复杂度

---

## 使用方式对比

### Local 模式（推荐）

```bash
# 直接运行，无需启动 Policy Server
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml
```

**架构**：
- Orchestrator 直接加载 ACT 模型
- 本地 GPU 推理
- 直接控制机器人

**适用场景**：
- 机器人和策略在同一台机器上
- 需要低延迟
- 简化部署

### Remote 模式（可选）

```bash
# 终端1: 启动 Policy Server
python src/lerobot/scripts/server/policy_server.py \
    --robot.type=supre_robot_follower \
    --port=50051

# 终端2: 启动任务执行
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks.yaml \
    --remote
```

**架构**：
- Policy Server 独立运行
- gRPC 通信
- 异步推理

**适用场景**：
- 计算资源分离（机器人边缘设备 + 云端服务器）
- 多个机器人共享一个策略服务器
- 需要远程监控和管理

---

## 新增模块

### LocalPolicyExecutor
**位置**: `src/lerobot/tasks/local_policy_executor.py`

本地策略执行器，直接加载和执行 ACT 模型，无需 Policy Server。

```python
executor = LocalPolicyExecutor(device="cuda")
executor.load_policy("/path/to/model")
action = executor.get_action(observation)
```

### LocalTaskScheduler
**位置**: `src/lerobot/tasks/task_scheduler.py`

本地任务调度器，使用 LocalPolicyExecutor 而非 gRPC 客户端。

```python
scheduler = LocalTaskScheduler(
    tasks=tasks,
    robot=robot,
    policy_executor=local_executor,
)
```

---

## 快速开始

### Local 模式（默认）

```bash
cd /home/smai/dc_dir/using/lerobot_origin

# 1. 配置模型路径
vim configs/task_agent_tasks_test.yaml

# 2. 运行
python src/lerobot/scripts/run_task_agent.py \
    --config configs/task_agent_tasks_test.yaml \
    --debug
```

### Remote 模式

```bash
# 1. 启动 Policy Server
./scripts/start_policy_server.sh

# 2. 运行任务（另一个终端）
./scripts/run_task_sequence.sh \
    configs/task_agent_tasks.yaml
```

---

## 配置文件

### YAML 配置
在 `configs/task_agent_tasks.yaml` 中配置任务和策略路径。

### 机器人配置
在 `src/lerobot/robots/supre_robot_follower/trunk_config.yaml` 中配置：
- 关节顺序
- CAN 总线参数
- 相机配置
- 关节限位

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 (必需) | - |
| `--remote` | 使用远程模式（需要 Policy Server） | False (local) |
| `--policy-device` | 策略执行设备 | cuda |
| `--tasks` | 指定运行的任务 | 所有任务 |
| `--debug` | 调试模式 | False |
| `--dry-run` | 仅验证配置 | False |
| `--max-retries` | 覆盖重试次数 | 使用配置值 |
| `--max-duration` | 覆盖最大时长 | 使用配置值 |
| `--collision-threshold` | 覆盖碰撞阈值 | 使用配置值 |

---

## 系统要求

### Local 模式
- Python 3.10+
- PyTorch 2.0+
- CUDA (推荐)
- 足够的 GPU 内存（取决于模型大小）

### Remote 模式
- Python 3.10+
- PyTorch 2.0+
- 网络连接（gRPC）

---

## 迁移指南

### 从 Remote 模式迁移到 Local 模式

如果你之前使用 Remote 模式，迁移很简单：

1. **无需修改配置文件**
   - 所有 YAML 配置保持不变

2. **只需改变启动方式**
   ```bash
   # 之前（需要 Policy Server）
   ./scripts/start_full_system.sh config.yaml

   # 现在（直接运行）
   python src/lerobot/scripts/run_task_agent.py --config config.yaml
   ```

3. **如需使用 Remote 模式**
   ```bash
   python src/lerobot/scripts/run_task_agent.py \
       --config config.yaml \
       --remote
   ```
