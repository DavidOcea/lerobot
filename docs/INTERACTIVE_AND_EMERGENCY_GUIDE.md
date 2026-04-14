# 交互式功能与急停回滚操作指南

## 目录
1. [交互式任务选择](#交互式任务选择)
2. [机器人复位功能](#机器人复位功能)
3. [急停与平滑回滚](#急停与平滑回滚)
4. [配置说明](#配置说明)
5. [故障排查](#故障排查)

---

## 交互式任务选择

### 启动交互式模式

```bash
# 使用本地推理模式（推荐）
python -m lerobot.scripts.run_task_agent \
    --config configs/task_agent_tasks.yaml \
    --interactive

# 使用远程推理模式（需要 Policy Server）
python -m lerobot.scripts.run_task_agent \
    --config configs/task_agent_tasks.yaml \
    --interactive \
    --remote

# 调试模式
python -m lerobot.scripts.run_task_agent \
    --config configs/task_agent_tasks_debug.yaml \
    --interactive \
    --debug
```

### 交互式菜单

进入交互式模式后，系统会显示：

```
============================================================
INTERACTIVE TASK SELECTION
============================================================
Next task: <任务名称>
Policy: <策略路径>

Options:
  1 - Execute next task in sequence
  2 - Create custom task or select existing task
  3 - Toggle automatic/interactive mode
  R - Reset robot joints to zero position (smooth reset)
  0 - Exit

Available tasks:
  1. <任务1>
  2. <任务2>
  ...

Current mode: <当前模式>
============================================================
>>>
```

### 菜单选项说明

| 选项 | 功能 | 说明 |
|------|------|------|
| `1` | 执行下一个任务 | 按顺序执行队列中的下一个任务 |
| `2` | 选择或创建任务 | 选择已有任务或创建自定义任务 |
| `3` | 切换模式 | 在自动模式和交互模式之间切换 |
| `R` | 复位机器人 | 将所有关节平滑复位到零位 |
| `0` | 退出 | 停止任务执行并退出 |

### 输入说明

- **直接按回车**：默认执行选项 1（执行下一个任务）
- **输入任务名称**：直接执行指定任务
- **输入数字**：选择对应的菜单选项

---

## 机器人复位功能

### 功能说明

复位功能将机器人所有关节平滑移动到零位（或自定义位置），用于：
- 任务执行前将机器人归零
- 任务失败后快速恢复到初始状态
- 测试时的快速重置

### 使用方法

在交互式菜单中输入 `R` 或 `reset`：

```
>>> R
```

系统将执行：
1. 获取当前关节位置
2. 计算到目标位置（默认0位）的平滑轨迹
3. 在配置的时间内完成复位（默认3秒）
4. 返回交互式菜单

### 复位配置

在配置文件中添加以下配置：

```yaml
# Robot reset settings
reset_duration: 3.0  # 复位时间（秒）
# 手动零位设置：为指定关节设置复位位置，未指定的关节复位到 0.0
# 示例：将右臂夹爪复位到 0.5（半开状态），其他关节复位到 0.0
reset_positions:
  right_arm_joint_7: 0.5  # 右臂夹爪半开
  left_arm_joint_7: 0.5   # 左臂夹爪半开
```

### 复位效果

- **平滑运动**：使用线性插值，避免突然跳跃
- **安全可控**：在指定时间内完成，避免过快运动
- **一次到位**：跳过 `max_relative_joint_move` 限制，一次性完成复位

---

## 急停与平滑回滚

### 急停触发条件

急停系统在检测到以下情况时自动触发：

1. **力矩异常检测**
   - 单关节力矩超过阈值（`collision_threshold`）
   - 多关节同时超过阈值（`multi_joint_threshold`）

2. **力矩变化率检测**
   - 力矩快速上升超过阈值（`force_rate_threshold`）
   - 瞬间力矩超过绝对限制（`immediate_threshold`）

3. **表面接触检测**
   - 检测到与工作表面接触（`surface_force_threshold`）

### 急停后恢复选项

当急停触发后，系统会提示用户选择恢复选项：

```
Emergency stop detected!
DANGEROUS_ACTION detected: collision detected on joint(s): <关节名称>

Recovery options:
1. Resume (continue from current position)
2. Rollback and continue (reset to previous safe state)
3. Rollback and retry with new model (reset and try again)
4. Abort (stop task execution)

Select option (1/2/3/4):
```

### 恢复选项说明

| 选项 | 功能 | 适用场景 |
|------|------|----------|
| `1` - Resume | 从当前位置继续 | 小碰撞，可以直接继续 |
| `2` - Rollback and continue | 回滚到安全状态后继续 | 中等碰撞，需要恢复位置 |
| `3` - Rollback and retry | 回滚后切换模型重新尝试 | 需要调整策略 |
| `4` - Abort | 停止任务 | 严重碰撞或手动停止 |

### 平滑回滚机制

回滚功能的工作原理：

1. **历史快照**：系统维护过去 `emergency_max_rollback_steps` 步的动作快照
2. **逐步回退**：从最近到最旧，逐步发送历史动作
3. **平滑过渡**：每个动作之间有 `emergency_rollback_step_delay` 延迟（默认20ms）
4. **策略重置**：回滚后重置策略模型状态，避免大动作

### 急停配置

```yaml
# 碰撞检测配置
collision_config:
  collision_threshold: 2.0              # 力矩阈值（Nm）
  detection_window: 5                    # 连续超过阈值的步数
  detection_mode: "immediate"           # 检测模式
  adaptive_mode: true                   # 自适应模式
  velocity_compensation: true           # 速度补偿

  # 急停回滚配置
  emergency_max_rollback_steps: 100     # 最大回滚步数
  emergency_rollback_step_delay: 0.02   # 回滚步间延迟（秒）

  # 多关节检测
  multi_joint_threshold: 3              # 触发多关节检测的关节数
  multi_joint_threshold_per_joint: 0.4  # 多关节检测的单关节阈值

  # 各关节阈值
  joint_specific_thresholds:
    left_arm_joint_7: 0.3   # 左夹爪
    right_arm_joint_7: 0.3  # 右夹爪
    # ... 其他关节阈值
```

---

## 配置说明

### 完整配置示例

```yaml
# configs/task_agent_tasks_debug.yaml

# 复位配置
reset_duration: 3.0  # 复位时间（秒）
reset_positions:
  right_arm_joint_7: 0.5  # 可选：自定义复位位置

# 碰撞检测配置
collision_config:
  collision_threshold: 2.0
  detection_window: 5
  adaptive_mode: true
  velocity_compensation: true

  # 急停回滚
  emergency_max_rollback_steps: 100
  emergency_rollback_step_delay: 0.02

# 执行设置
environment_dt: 0.02  # 控制频率 50Hz
```

### 夹爪动作平滑配置

夹爪（`*_joint_7`）已禁用动作平滑，确保快速响应抓取操作：

- **低通滤波**：禁用（`alpha=1.0`）
- **速度限制**：禁用（`infinite`）
- **目的**：提高抓取成功率

---

## 故障排查

### 问题1：复位功能无效

**症状**：输入 R 后没有动作，或只能部分复位

**原因**：代码版本较旧，未包含修复

**解决**：
1. 拉取最新代码：
   ```bash
   git pull david temp-agent
   ```
2. 重新运行系统

### 问题2：急停后无法回滚

**症状**：急停后提示回滚失败

**原因**：
- 历史快照不足（`emergency_max_rollback_steps` 太小）
- 网络连接问题（远程模式）

**解决**：
1. 增加 `emergency_max_rollback_steps` 值
2. 检查网络连接
3. 使用本地模式（`--remote` 参数不加）

### 问题3：交互式输入无效

**症状**：输入字符后没有响应

**原因**：输入被硬件设备干扰

**解决**：最新代码已修复，确保使用 `/dev/tty` 作为输入源

### 问题4：夹爪反复开合无法抓取

**症状**：夹爪反复张开闭合，无法抓住工件

**原因**：动作平滑导致夹爪响应延迟

**解决**：最新代码已为夹爪禁用动作平滑，无需额外配置

---

## 最近更新记录

### 2026-03-05

- ✅ 修复复位功能：跳过 `max_relative_joint_move` 限制
- ✅ 支持自定义复位位置（`reset_positions`）
- ✅ 禁用夹爪动作平滑，提高抓取成功率
- ✅ 修复回滚后策略状态未重置问题
- ✅ 实现平滑逐步回滚（`emergency_rollback_step_delay`）
- ✅ 修复交互式输入被硬件设备干扰问题