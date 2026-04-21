# AGV + Robot Task Integration

本模块实现了AGV小车与机械臂的协同控制，支持"抓取→移动→放置"的完整工作流程，**无需ROS2依赖**。

## 文件结构

```
src/lerobot/
├── robots/agv/
│   ├── __init__.py                  # AGV模块导出
│   └── seer_agv_controller.py       # 仙工AGV TCP控制器
│
├── tasks/
│   ├── config.py                    # AGVTaskConfig配置类
│   ├── agv_executor.py              # AGV任务执行器
│   └── ...                          # 其他任务模块
│
├── agent/
│   ├── config.py                    # AGVGlobalConfig全局配置
│   └── orchestrator.py              # orchestrator AGV执行逻辑
│
configs/
└── agv_pick_place.yaml              # 示例配置文件

tests/
└── test_agv_controller.py           # AGV控制器测试脚本
```

## 快速使用

### 1. 测试AGV连接

```bash
# 基本连接测试（安全，无移动）
python tests/test_agv_controller.py --host 192.168.1.100

# 包含导航测试（会移动AGV）
python tests/test_agv_controller.py --host 192.168.1.100 --test-navigation --target-station station_B
```

### 2. 运行完整任务序列

```bash
# 修改配置文件中的参数
# - AGV IP地址
# - Policy路径
# - 站点名称

# 运行
python -m lerobot.scripts.run_task_agent --config configs/agv_pick_place.yaml
```

### 3. 交互模式

```bash
# 每个任务前会提示用户确认
python -m lerobot.scripts.run_task_agent --config configs/agv_pick_place.yaml --interactive
```

## 配置说明

### AGVGlobalConfig (全局配置)

```yaml
agv_config:
  enabled: true
  host: "192.168.1.100"  # AGV IP地址
  port: 19204
  connection_timeout: 5.0
  read_timeout: 2.0
  auto_reconnect: true
  check_arm_before_move: true  # 安全检查
  
  # 站点地图
  station_map:
    station_A: [0.0, 0.0, 0.0]
    station_B: [5.0, 0.0, 0.0]
```

### AGVTaskConfig (单个任务)

```yaml
tasks:
  - name: "agv_move_to_B"
    task_type: "agv"
    agv_config:
      target_station: "station_B"  # 或使用 target_position: [x, y, theta]
      wait_for_arrival: true
      arrival_timeout: 60.0
      check_arm_safe_position: true
```

## TCP协议

基于仙工AGV TCP协议实现：

- **端口19204**: 状态查询
- **端口19205**: 控制（急停/暂停）
- **端口19206**: 导航

协议格式：16字节header + JSON payload

```
┌────────────────────────────────────────────────┐
│ sync (0x5A) │ version (0x01) │ seq (2B LE)     │
│ data_len (4B BE) │ api_type (2B BE) │ reserved │
└────────────────────────────────────────────────┘
│ JSON payload                                   │
└────────────────────────────────────────────────┘
```

## API类型码

| API | 代码 | 说明 |
|-----|------|------|
| 状态查询 | 0x03E8 (1000) | 综合状态 |
| 电量查询 | 0x03EA (1002) | Battery % |
| 位置查询 | 0x03F2 (1010) | x, y, theta |
| 站点查询 | 0x044E (1102) | Current station |
| 急停 | 0x07D2 (2002) | Emergency stop |
| 暂停 | 0x07D3 (2003) | Pause |
| 继续 | 0x07D4 (2004) | Resume |
| 导航站点 | 0x07E8 (2024) | Navigate to station |
| 导航坐标 | 0x07E9 (2025) | Navigate to position |

## 安全机制

1. **机械臂位置检查**: AGV移动前验证机械臂是否在安全位置
2. **电量检查**: 低电量警告（<20%）
3. **异常急停**: 错误时自动执行emergency stop
4. **超时重试**: 导航超时可配置重试
5. **到达确认**: 等待站点ID匹配或坐标距离判定

## 工作流程示例

```
1. pick_at_station_A    → 机械臂抓取零件
2. arm_home_position    → 机械臂归位（安全位置）
3. agv_move_to_station_B → AGV移动到B点
4. place_at_station_B   → 机械臂放置零件
5. return_home          → 返回起点（可选）
```

## 调试

```bash
# Debug模式
python -m lerobot.scripts.run_task_agent --config configs/agv_pick_place.yaml --debug

# 查看AGV状态
python tests/test_agv_controller.py --host 192.168.1.100 --debug
```

## 注意事项

1. **AGV路径规划**: 确保AGV移动路径无障碍
2. **机械臂收起**: AGV移动前必须确认机械臂已收起
3. **站点地图**: 建议配置station_map以便坐标导航
4. **电量管理**: 定期检查AGV电量，低电量时及时充电