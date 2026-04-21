# AGV地图构建与站点坐标配置指南

## 概述

仙工AGV需要通过SLAM扫描构建地图，然后在地图上标记站点（位置点）。本指南说明如何：

1. 构建新地图（SLAM扫图）
2. 在地图上标记站点（A点、B点等）
3. 获取站点坐标并配置到lerobot系统

---

## 一、AGV地图构建流程

### 1.1 准备工作

**所需设备和软件：**
- 仙工AGV小车（SRC-880系列）
- SEER Studio上位机软件（或SEARobot手机App）
- 确保AGV电量充足（>30%）
- 确保工作区域平整、无动态障碍物

**网络配置：**
```bash
# AGV默认IP: 192.168.192.5 (或根据实际配置)
# 确保电脑能ping通AGV
ping 192.168.192.5

# 查看AGV当前状态
cd /root/workspace/dc_dir/lerobot
python scripts/record_agv_stations.py --host 192.168.192.5 --list
```

### 1.2 SLAM扫图步骤

**使用SEER Studio（推荐）：**

1. **连接AGV**
   - 打开SEER Studio软件
   - 输入AGV IP地址连接
   - 确认连接成功，能看到AGV当前位置

2. **开始SLAM扫图**
   - 点击"开始扫图"或"Build Map"
   - 驾驶AGV缓慢遍历整个工作区域
   - 建议速度：0.3-0.5 m/s
   - 遍历所有AGV需要到达的位置

3. **扫图技巧**
   - 确保地图边缘完整
   - 多次扫描关键区域（抓取点、放置点）
   - 避免动态障碍物（人员移动）
   - 检查地图质量（无黑洞、重影）

4. **保存地图**
   - 扫图完成后保存地图
   - 建议命名：`workshop_YYYYMMDD.map`
   - 导出地图文件

### 1.3 使用命令行查询地图状态

```bash
# 通过TCP查询AGV地图信息
python -c "
from lerobot.robots.agv.seer_agv_controller import SeerAGVController

controller = SeerAGVController(host='192.168.192.5')
if controller.connect():
    status = controller.get_status()
    print(f'当前地图: {status.current_station}')
    controller.disconnect()
"
```

---

## 二、标记站点位置

### 2.1 站点命名建议

根据工作流程命名站点：

| 站点名称 | 用途 | 说明 |
|---------|------|------|
| `station_A` | 零件抓取点 | A工作台位置 |
| `station_B` | 零件放置点 | B工作台位置 |
| `station_home` | 休息/待命位置 | 安全区域 |
| `charging_station` | 充电位置 | 充电桩附近 |
| `pick_point` | 抓取位置 | 通用命名 |
| `place_point` | 放置位置 | 通用命名 |

### 2.2 使用SEER Studio标记站点

1. **打开地图**
   - 在SEER Studio中打开已保存的地图
   - 确认AGV位置在地图上正确显示

2. **创建站点**
   - 点击"添加站点"或"Add Point"
   - 在地图上点击目标位置
   - 输入站点ID（如 `station_A`）
   - 设置站点属性：
     - 类型：普通站点 / 充电站点 / 货架点
     - 允许停靠方向
     - 是否允许等待

3. **站点坐标获取**
   - 选中站点查看详情
   - 记录坐标值：x, y, theta
   - 或直接从AGV当前位置获取

### 2.3 使用命令行记录站点（推荐）

**使用record_agv_stations.py工具：**

```bash
cd /root/workspace/dc_dir/lerobot

# 方式1: 交互式记录（推荐）
python scripts/record_agv_stations.py --host 192.168.192.5 --interactive

# 方式2: 快速记录当前位置
python scripts/record_agv_stations.py --host 192.168.192.5 --station station_A --desc "抓取位置A"

# 方式3: 查看已记录站点
python scripts/record_agv_stations.py --host 192.168.192.5 --list

# 方式4: 生成YAML配置
python scripts/record_agv_stations.py --generate-yaml
```

---

## 三、配置站点坐标到lerobot系统

### 3.1 站点坐标文件结构

记录的站点保存在：`configs/agv_stations.json`

```json
{
  "station_A": {
    "id": "station_A",
    "x": 1.523,
    "y": 0.832,
    "theta": 0.0,
    "description": "零件抓取点",
    "current_station_name": "A_workbench",
    "battery_at_record": 85,
    "recorded_at": "2026-04-21 10:30:00"
  },
  "station_B": {
    "id": "station_B",
    "x": 5.234,
    "y": 0.815,
    "theta": 0.0,
    "description": "零件放置点",
    ...
  }
}
```

### 3.2 配置到YAML任务文件

**修改 configs/agv_pick_place.yaml：**

```yaml
agv_config:
  enabled: true
  host: "192.168.192.5"  # 你的AGV IP
  
  # 站点地图 - 从记录工具生成
  station_map:
    station_A: [1.523, 0.832, 0.0]   # 抓取点坐标
    station_B: [5.234, 0.815, 0.0]   # 放置点坐标
    station_home: [2.5, 2.5, 0.0]    # 待命位置
    charging_station: [0.0, 5.0, 0.0] # 充电位置

tasks:
  # 抓取任务
  - name: "pick_at_station_A"
    task_type: "policy"
    policy_path: "/path/to/pick_policy"
    # ...

  # AGV移动到B点
  - name: "agv_move_to_station_B"
    task_type: "agv"
    agv_config:
      target_station: "station_B"  # 使用站点ID
      wait_for_arrival: true
      arrival_timeout: 60.0
```

### 3.3 验证配置

```bash
# 测试AGV连接和导航
python tests/test_agv_controller.py --host 192.168.192.5

# 测试导航到指定站点（谨慎操作，AGV会移动）
python tests/test_agv_controller.py --host 192.168.192.5 --test-navigation --target-station station_B
```

---

## 四、完整操作流程示例

### Step 1: 构建地图

```bash
# 1. 确保AGV连接正常
python scripts/record_agv_stations.py --host 192.168.192.5 --list

# 2. 在SEER Studio中扫图（手动驾驶AGV遍历区域）
# 3. 保存地图文件
```

### Step 2: 记录站点坐标

```bash
# 驾驶AGV到抓取位置A
# 记录当前位置为station_A
python scripts/record_agv_stations.py --host 192.168.192.5 --station station_A --desc "零件抓取点"

# 驾驶AGV到放置位置B
# 记录当前位置为station_B
python scripts/record_agv_stations.py --host 192.168.192.5 --station station_B --desc "零件放置点"

# 记录更多站点...
python scripts/record_agv_stations.py --host 192.168.192.5 --station charging_station --desc "充电位置"
```

### Step 3: 生成并配置YAML

```bash
# 生成YAML配置片段
python scripts/record_agv_stations.py --generate-yaml

# 配置会保存在 configs/agv_station_map.yaml
# 将内容复制到 configs/agv_pick_place.yaml 的 station_map 部分
```

### Step 4: 测试导航

```bash
# 基本连接测试
python tests/test_agv_controller.py --host 192.168.192.5

# 导航测试（注意安全！）
python tests/test_agv_controller.py --host 192.168.192.5 --test-navigation --target-station station_B
```

### Step 5: 运行完整任务

```bash
# 修改policy路径后运行
python -m lerobot.scripts.run_task_agent --config configs/agv_pick_place.yaml --interactive
```

---

## 五、常见问题

### Q1: 地图扫图后AGV定位不准？

**解决方案：**
- 确保地面无明显变化
- 避免扫图时有动态障碍物
- 使用SEER Studio重新定位（Re-localization）

### Q2: 导航到站点但位置偏差大？

**解决方案：**
- 检查站点坐标是否正确记录
- 使用坐标导航而非站点ID导航：
```yaml
agv_config:
  target_position: [5.234, 0.815, 0.0]  # 直接用坐标
  arrival_tolerance: 0.5  # 增加容差
```

### Q3: 如何更新站点坐标？

```bash
# 删除旧站点重新记录
python scripts/record_agv_stations.py --delete station_A
python scripts/record_agv_stations.py --host 192.168.192.5 --station station_A --desc "更新后的抓取点"
```

### Q4: AGV电量不足怎么办？

```bash
# 查看电量
python scripts/record_agv_stations.py --host 192.168.192.5 --list

# 电量<20%时：
# 1. 添加充电任务到任务列表
# 2. 或手动驾驶到充电位置
```

---

## 六、API参考

### 站点相关API

| API | 端口 | 说明 |
|-----|------|------|
| 1102 (0x044E) | 19204 | 查询当前站点ID |
| 1010 (0x03F2) | 19204 | 查询位置坐标 (x, y, theta) |
| 2024 (0x07E8) | 19206 | 导航到站点 |
| 2025 (0x07E9) | 19206 | 导航到坐标 |

### SeerAGVController方法

```python
from lerobot.robots.agv import SeerAGVController

controller = SeerAGVController(host="192.168.192.5")
controller.connect()

# 获取当前位置
position = controller.get_position()  # AGVPosition(x, y, theta)

# 获取当前站点
station = controller.get_current_station()  # 返回站点ID字符串

# 导航到站点
controller.move_to_station("station_B")

# 导航到坐标
controller.move_to_position(5.0, 0.0, 0.0)

# 等待到达
controller.wait_for_arrival("station_B", timeout=60.0)
```

---

## 附录：站点命名模板

根据实际工作流程，建议使用以下命名模板：

**生产线场景：**
- `line1_pick` - 生产线1抓取点
- `line1_place` - 生产线1放置点
- `line2_pick` - 生产线2抓取点
- ...

**仓库场景：**
- `shelf_A1_pick` - A货架第1层抓取
- `shelf_A1_place` - A货架第1层放置
- `conveyor_in` - 进料传送带
- `conveyor_out` - 出料传送带

**灵活配置：**
- `station_01`, `station_02`, ... - 按序号命名
- `wp_01`, `wp_02`, ... - waypoint命名