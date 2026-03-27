# Precision Place - 毫米级精准放置

## 快速开始

```bash
cd /home/smai/dc_dir/lerobot
python precision_place/run.py
```

## 功能概览

| 功能模块 | 描述 | 精度 |
|---------|------|------|
| **手眼标定** | ChArUco板 + Tsai-Lenz算法 | RMSE < 1.5px |
| **XY对齐** | 外参矩阵精确坐标变换 | ±1mm |
| **Z轴控制** | 双目立体视觉 + 单目尺寸估计 | ±0.5mm |
| **旋转对齐** | 三标记姿态估计 | ±2° |
| **退化模式** | 标记不足时仍可工作 | ±3mm |
| **IBVS对齐** | 抗遮挡视觉伺服，盲插支持 | ±1mm |

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    精准放置系统 V4                            │
│                 (手眼标定 + 外参矩阵方法)                      │
├─────────────────────────────────────────────────────────────┤
│  标定层 (一次性)                                             │
│  ├── 手眼标定: ChArUco板 + 正运动学 → 外参矩阵               │
│  ├── 重投影验证: RMSE < 1.5px = 合格                         │
│  └── 输出: hand_eye_extrinsic.yaml                          │
├─────────────────────────────────────────────────────────────┤
│  输入层                                                      │
│  ├── 主相机 (right_wrist, 索引6)                            │
│  ├── 副相机 (right_wrist2, 索引8) - Z轴深度估计              │
│  ├── 机器人状态 (关节角度)                                   │
│  └── URDF文件 (仅标定时需要)                                 │
├─────────────────────────────────────────────────────────────┤
│  检测层                                                      │
│  ├── 工件标记: 3个绿色                                       │
│  ├── 卡槽标记: 3个红色                                       │
│  └── 双相机融合检测                                          │
├─────────────────────────────────────────────────────────────┤
│  坐标变换层 (核心)                                           │
│  ├── 像素偏移 → 相机坐标系射线                               │
│  ├── 相机坐标 → 法兰坐标 (外参矩阵)                          │
│  ├── 法兰坐标 → 世界坐标 (正运动学)                          │
│  └── 输出: 精确的世界坐标偏移量                              │
├─────────────────────────────────────────────────────────────┤
│  控制层                                                      │
│  ├── XY控制: 直接TCP位置调整                                 │
│  ├── Z轴控制: 双目/单目深度估计                              │
│  └── 旋转控制: joint_6 (手腕旋转)                            │
└─────────────────────────────────────────────────────────────┘
```

## 配置

| 项目 | 值 |
|------|-----|
| 主相机 | right_wrist (索引6) |
| 副相机 | right_wrist2 (索引8) |
| 工件标记 | 绿色 x3 |
| 卡槽标记 | 红色 x3 |
| 标记直径 | 15mm (默认) |
| XY精度 | ±1mm (手眼标定方法) |
| Z轴精度 | ±0.5mm |

---

## 完整使用流程

### 方法一：手眼标定方法 (推荐)

#### 步骤1：准备ChArUco标定板

```bash
python -m precision_place.hand_eye_calibration
```

这会在当前目录生成 `charuco_board.png`，**打印并固定在工作台上**（标定过程中绝对不能移动）。

#### 步骤2：运行程序并连接设备

```bash
python precision_place/run.py
```

```
菜单 → 1. 连接设备
```

#### 步骤3：手眼标定

```
菜单 → 4. 标定 → H. 手眼标定
```

按提示操作：
1. **输入URDF文件路径**（用于正运动学计算法兰位姿）
2. 移动机械臂，让相机看到标定板
3. 按 **C** 捕获当前姿态
4. **换一个不同的姿态**（倾斜角度变化越大越好）
5. 重复步骤3-4，采集**至少10张**（推荐30张）
6. 按 **S** 开始标定计算
7. 标定成功后结果保存到 `precision_place/hand_eye_extrinsic.yaml`

**标定技巧：**
- 标定板必须固定不动
- 姿态差异越大，标定越准确
- 大角度倾斜比小范围移动更有效
- 避免只在一个平面内移动

#### 步骤4：验证标定（可选但推荐）

```
菜单 → 4. 标定 → R. 重投影验证
```

- **方法1（准确）**：用TCP探针测量验证点世界坐标
- **方法2（简化）**：检查不同姿态下标定板检测一致性

验收标准：**RMSE < 1.5像素**

#### 步骤5：运行对齐

```
菜单 → 8.5 手眼标定对齐
```

**按键操作：**
- 按 **A** 开始自动对齐
- 按 **M** 单步对齐（手动确认每一步）
- 按 **T** 设置目标偏移量（当前偏移作为目标）
- 按 **C** 清除目标偏移量
- 按 **Q** 退出

系统会自动：
1. 检测工件和卡槽位置
2. 计算像素偏移 → 世界坐标偏移
3. 移动TCP进行对齐
4. 重复直到误差 < 5像素

---

### 目标偏移量设置（标记有固定偏移时）

**使用场景：**
- 为了避免遮挡，标记粘贴位置与实际对齐位置有偏移
- 工件标记和卡槽标记中心不对齐时

**操作步骤：**
1. 手动将工件放入卡槽正确位置
2. 进入对齐界面 `菜单 → 8.5`
3. 按 **T** 设置当前偏移为目标偏移
4. 之后对齐时，系统会向目标偏移量靠近

**原理：**
```
修正偏移 = 当前偏移 - 目标偏移

当修正偏移 = 0 时，工件处于正确位置
```

---

### IBVS抗遮挡对齐

**适用场景**：放置时相机被遮挡，无法看到标记。

#### 原理

```
记忆阶段 (Z=15cm)          对齐阶段 (可遮挡)
     │                           │
     ▼                           ▼
捕获特征点3D坐标 ──────→ 正运动学计算虚拟像素
     │                           │
     ▼                           ▼
保存到 ibvs_memory.json    计算误差 → 速度指令 → 移动TCP
```

#### 操作步骤

**步骤1：IBVS记忆**
```
主菜单 → M. IBVS记忆阶段
```
1. 工件夹在夹爪中，移到卡槽上方约15cm
2. 按 **Z** 自动抬高15cm
3. 按 **M** 记忆特征点

**步骤2：IBVS对齐**
```
主菜单 → I. IBVS对齐阶段
```
1. 移动到卡槽上方（可遮挡标记）
2. 按 **A** 开始自动对齐

#### 按键说明

| 阶段 | 按键 | 功能 |
|------|------|------|
| 记忆 | **Z** | 自动抬高15cm |
| 记忆 | **M** | 记忆特征点 |
| 对齐 | **A** | 自动对齐 |
| 对齐 | **M** | 单步对齐 |

---

### 方法二：传统灵敏度方法

仅当无法获取URDF文件时使用此方法。

```
菜单 → 4. 标定
  ├── 1. 像素-毫米标定
  ├── 2. XY关节灵敏度标定 (手动)
  └── 3. XY关节灵敏度标定 (自动)

菜单 → 8. 运行对齐 (传统灵敏度方法)
```

---

## 文件结构

```
precision_place/
├── run.py                      # 主启动脚本
├── __init__.py                 # 主入口，导出公共API
│
├── models/                     # 数据模型
│   ├── marker.py              # Marker, DualMarkerState
│   ├── calibration_data.py    # JointSensitivity, ArmConfig, ARM_CONFIGS
│   └── state.py               # DetectionResult, AlignmentResult
│
├── config/                     # 配置管理
│   └── settings.py            # PrecisionPlaceConfig
│
├── core/                       # 核心逻辑
│   ├── detector.py            # DualPointDetector
│   └── aligner.py             # HandEyeAligner
│
├── calibration/                # 标定模块
│   ├── hand_eye.py            # HandEyeCalibrator
│   ├── forward_kinematics.py  # ForwardKinematics
│   ├── coordinate_transform.py # CoordinateTransformer
│   ├── sync_capture.py        # SynchronizedCapture (同步捕获)
│   ├── ibvs_controller.py     # VirtualIBVSController (IBVS控制)
│   └── tcp_calibrator.py      # TCPCalibrator (TCP标定)
│
├── robot/                      # 机器人接口
│   ├── interface.py           # RobotInterface, MockRobot
│   └── status.py              # RobotStatusReader
│
├── docs/                       # 文档
│   └── TODO_smooth_motion_control.md  # 平滑移动待开发
│
├── dual_point_alignment.py     # 传统控制器 (向后兼容)
├── z_axis_controller.py        # Z轴控制模块
│
├── configs/
│   └── precision_config.yaml   # 配置文件
│
├── hand_eye_extrinsic.yaml     # 手眼标定结果 (生成)
├── ibvs_memory.json            # IBVS记忆数据 (生成)
└── calibration_points.json     # 传统标定数据
```

---

## 两种方法对比

| 项目 | 传统灵敏度方法 (选项8) | 手眼标定方法 (选项8.5) | IBVS方法 (菜单M/I) |
|------|----------------------|----------------------|-------------------|
| 标定次数 | 多次（每个关节、每个高度） | **一次** | **一次**（复用手眼标定） |
| 精度 | 依赖经验标定 | **数学精确计算** | **数学精确计算** |
| 透视补偿 | 手动设置偏移方向 | **自动处理相机倾斜** | **自动处理** |
| 验证 | 无自动验证 | **RMSE自动验证** | **RMSE自动验证** |
| 抗遮挡 | ❌ 需要看到标记 | ❌ 需要看到标记 | ✅ **盲插支持** |
| 需要URDF | 否 | 是（仅标定时） | 是 |
| 推荐程度 | 备选 | 推荐 | **遮挡场景推荐** |

---

## 原理说明

### 手眼标定原理

```
Eye-in-Hand 模式：

  世界坐标系 (Base)
       │
       │ T_base2flange (正运动学计算)
       ▼
  法兰坐标系 (Flange)
       │
       │ T_flange2cam (手眼标定求解)
       ▼
  相机坐标系 (Camera)
       │
       │ 透视投影
       ▼
  像素坐标 (u, v)
```

**外参矩阵 T_flange2cam** 包含：
- 旋转部分：相机相对于法兰的姿态
- 平移部分：相机光心相对于法兰的位置

### 像素偏移 → 世界偏移

```python
# 简化公式
world_offset = R_cam2world @ (pixel_offset * depth / focal_length)

# 其中:
# - R_cam2world: 从外参矩阵和TCP姿态推导
# - depth: 目标深度 (米)
# - focal_length: 相机焦距 (像素)
```

---

## 注意事项

### 手眼标定
- 标定板固定后绝对不能移动
- 采集姿态要有足够的角度变化
- 推荐采集30+张图像以提高精度

### 对齐操作
- 确保光照稳定
- 保持合适高度 (5-20 cm)
- 标记颜色与背景有明显对比
- 首次使用建议先验证标定结果

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 标定失败 | 姿态变化不够 | 增加大角度倾斜姿态 |
| RMSE过高 | 标定板移动过 | 重新固定标定板并重做 |
| 对齐方向错误 | 外参矩阵方向问题 | 检查URDF坐标系定义 |
| 深度估计不准 | 双目未标定 | 运行双目基线标定 |
| IBVS记忆失败 | 标记数量不足 | 确保能看到4个以上标记 |
| IBVS抬高失败 | 关节限位 | 调整机器人姿态后重试 |
| IBVS深度为负 | 外参矩阵方向错误 | 检查相机坐标系方向 |

---

## API 参考

### 手眼标定

```python
from precision_place.calibration.hand_eye import HandEyeCalibrator

# 创建标定器
calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)

# 采集姿态
calibrator.capture_pose(image, flange_position, flange_rotation)

# 执行标定
success, result = calibrator.calibrate()

# 保存结果
calibrator.save("hand_eye_extrinsic.yaml")
```

### 坐标变换

```python
from precision_place.calibration.coordinate_transform import CoordinateTransformer

# 加载外参矩阵
transformer = CoordinateTransformer.from_calibration_file("hand_eye_extrinsic.yaml")

# 设置TCP位姿
transformer.set_tcp_pose(position, quaternion)

# 像素偏移转世界偏移
world_offset = transformer.pixel_offset_to_world_offset((du, dv), depth)
```

### 对齐控制

```python
from precision_place.core.aligner import HandEyeAligner
from precision_place.core.detector import DualPointDetector

# 创建检测器和对齐器
detector = DualPointDetector()
aligner = HandEyeAligner(detector, transformer)

# 执行对齐
result = aligner.align(
    get_image=lambda: camera.read(),
    get_tcp_pose=lambda: (position, rotation),
    get_depth=lambda: depth_m,
    move_tcp=lambda delta: robot.move_tcp(delta)
)
```

### 数据模型

```python
from precision_place.models.marker import Marker, DualMarkerState
from precision_place.models.calibration_data import ARM_CONFIGS

# 使用配置
arm_config = ARM_CONFIGS['right']
print(f"主相机: {arm_config.camera_name} (索引{arm_config.camera_index})")
```

### 向后兼容

```python
# 旧代码仍然可用
from precision_place import PrecisionPlaceController, ZAxisController

controller = PrecisionPlaceController(robot, camera, arm='right')
```

### IBVS控制器

```python
from precision_place.calibration.ibvs_controller import VirtualIBVSController

# 创建IBVS控制器
ibvs = VirtualIBVSController(
    camera_matrix=camera_matrix,
    extrinsic_matrix=T_flange2cam,
    lambda_gain=0.5,  # 控制增益
    pixel_tolerance=3.0  # 对齐容差（像素）
)

# 记忆特征点
ibvs.memorize_from_markers(
    workpiece_markers=wp_markers,
    slot_markers=slot_markers,
    flange_position=flange_pos,
    flange_rotation=flange_rot,
    depth=0.15  # 15cm
)

# 计算速度指令
V_flange, info = ibvs.calculate_velocity(flange_pos, flange_rot)

# 检查是否对齐
if info['aligned']:
    print("对齐成功!")

# 保存/加载记忆
ibvs.save_memory("ibvs_memory.json")
ibvs.load_memory("ibvs_memory.json")
```

### 同步捕获

```python
from precision_place.calibration.sync_capture import SynchronizedCapture

# 创建同步捕获器
sync = SynchronizedCapture(camera, controller, forward_kinematics)

# 同步捕获
result = sync.capture()
if result.success:
    image = result.image
    joints = result.joints
    flange_pos = result.flange_position
    print(f"同步延迟: {result.sync_delay_ms:.1f}ms")
```