# 平滑移动控制 - 待开发文档

> 状态: 待开发
> 优先级: 中
> 创建日期: 2026-03-27
> 依赖: 需要硬件SDK支持

## 1. 背景与目标

### 当前问题

IBVS对齐过程中使用位置增量模式，导致阶梯式移动：

```
位置1 ──等待── 位置2 ──等待── 位置3 ──等待── ...
```

### 目标效果

实现平滑连续的移动轨迹：

```
位置 ──平滑连续──→ 目标
```

### 平滑度对比

| 模式 | 平滑度 | 硬件要求 | 适用场景 |
|------|--------|----------|----------|
| 速度控制 | ★★★ 最高 | 驱动器支持速度模式 | IBVS实时控制 |
| 伺服模式 | ★★★ 高 | 控制器支持servo指令 | 流式轨迹 |
| 位置增量 | ★★ 中 | 标准位置控制 | 当前实现 |

## 2. 当前系统架构

```
precision_place/run.py
  └── ibvs_alignment_phase()
        └── controller.set_cartesian_velocity()  ← 待实现

precision_place/dual_point_alignment.py
  └── PrecisionPlaceController
        └── robot.send_action()  ← 只有位置指令

lerobot/robots/supre_robot_follower/supre_robot_follower.py
  └── SupreRobotFollower
        └── send_action(action)        # 位置控制
        └── send_target_position()     # 位置控制
        └── _hardware_manager.write()  # 只有位置写入

lerobot/robots/supre_robot/supre_robot_hardware_manager.py
  └── SupreRobotHardwareManager
        └── write(command_positions)   # 只有位置写入

底层驱动
  └── EyouMotorHardware.write()        # 需确认是否支持速度模式
```

## 3. 需要的硬件信息

开发前需要确认以下信息：

### 3.1 电机驱动器规格

- [ ] 驱动器型号（EyouMotor具体型号）
- [ ] 支持的控制模式（位置/速度/力矩）
- [ ] 通信协议（CAN/EtherCAT/其他）
- [ ] 控制频率上限

### 3.2 SDK/API文档

- [ ] 速度控制接口文档
- [ ] 控制模式切换方法
- [ ] 速度指令格式（单位、范围）
- [ ] 示例代码

### 3.3 确认问题清单

```
1. EyouMotorHardware 是否支持速度模式？
   └── 如果支持，切换模式的API是什么？

2. 速度指令的单位是什么？
   └── rad/s? deg/s? rpm?

3. 速度控制的响应延迟是多少？
   └── 这决定了IBVS控制频率的上限

4. 能否同时控制多个关节的速度？
   └── 批量发送 vs 单独发送
```

## 4. 实现路径

### 4.1 接口定义（已完成）

`precision_place/robot/interface.py` 已添加：

```python
@abstractmethod
def set_cartesian_velocity(self, vx: float, vy: float, vz: float,
                            wx: float = 0, wy: float = 0, wz: float = 0) -> bool:
    """设置笛卡尔速度（平滑IBVS所需）"""
    pass

@abstractmethod
def servo_to_position(self, x: float, y: float, z: float,
                      qx: float, qy: float, qz: float, qw: float,
                      time_ms: int = 100) -> bool:
    """伺服模式移动（平滑轨迹）"""
    pass

def has_velocity_control(self) -> bool:
    """是否支持速度控制"""
    return False

def has_servo_mode(self) -> bool:
    """是否支持伺服模式"""
    return False
```

### 4.2 IBVS调用逻辑（已完成）

`precision_place/run.py` 已添加三级优先级：

```python
# 优先级1: 速度控制（最平滑）
if self.controller.has_velocity_control():
    self.controller.set_cartesian_velocity(vx, vy, vz)

# 优先级2: 伺服模式（平滑轨迹）
elif self.controller.has_servo_mode():
    self.controller.servo_to_position(x, y, z, qx, qy, qz, qw, time_ms=50)

# 优先级3: 位置增量（当前使用）
else:
    self.controller.move_to_position(x, y, z, qx, qy, qz, qw)
```

### 4.3 底层实现（待开发）

#### Step 1: EyouMotorHardware 添加速度模式

```python
# 文件: lerobot/robots/supre_robot/eyou_motor.py (假设)

class EyouMotorHardware:
    def __init__(self, ...):
        self._control_mode = "position"  # position / velocity / torque

    def set_velocity_mode(self) -> bool:
        """切换到速度控制模式"""
        # TODO: 调用SDK切换模式
        # 示例: self.driver.set_mode("velocity")
        self._control_mode = "velocity"
        return True

    def write_velocity(self, velocities: List[float]) -> bool:
        """写入速度指令"""
        if self._control_mode != "velocity":
            self.set_velocity_mode()

        # TODO: 调用SDK发送速度
        # 示例: self.driver.set_velocity(velocities)
        return True

    def write(self, commands: List[float]):
        """写入指令（根据当前模式）"""
        if self._control_mode == "velocity":
            self.write_velocity(commands)
        else:
            # 原有位置控制逻辑
            ...
```

#### Step 2: SupreRobotHardwareManager 添加速度分发

```python
# 文件: lerobot/robots/supre_robot/supre_robot_hardware_manager.py

class SupreRobotHardwareManager:
    def write_velocity(self, velocities: List[float]):
        """分发速度指令到各硬件"""
        # 类似 write() 的分发逻辑
        hw_velocities = {}
        for instance in self._hardware_instances:
            num_hw_joints = instance.get_joint_count()
            hw_velocities[instance] = [None] * num_hw_joints

        for global_index, velocity in enumerate(velocities):
            mapping = self._joint_map[global_index]
            instance = mapping['instance']
            hw_index = mapping['hw_index']
            hw_velocities[instance][hw_index] = velocity

        for instance, velocities in hw_velocities.items():
            if hasattr(instance, 'write_velocity'):
                instance.write_velocity(velocities)

    def set_control_mode(self, mode: str):
        """设置所有硬件的控制模式"""
        for instance in self._hardware_instances:
            if hasattr(instance, f'set_{mode}_mode'):
                getattr(instance, f'set_{mode}_mode')()
```

#### Step 3: SupreRobotFollower 添加速度控制

```python
# 文件: lerobot/robots/supre_robot_follower/supre_robot_follower.py

class SupreRobotFollower:
    def send_velocity(self, velocities: dict[str, float]) -> dict[str, Any]:
        """发送速度指令"""
        if not self.is_connected:
            raise RuntimeError("Robot is not connected.")

        # 转换为全局速度向量
        global_velocities = [0.0] * len(self.observation_joint_names)
        for name, vel in velocities.items():
            idx = self.observation_joint_names.index(name)
            global_velocities[idx] = vel

        # 发送速度
        self._hardware_manager.write_velocity(global_velocities)

        return velocities

    def set_control_mode(self, mode: str):
        """切换控制模式 (position/velocity)"""
        self._hardware_manager.set_control_mode(mode)
```

#### Step 4: PrecisionPlaceController 实现接口

```python
# 文件: precision_place/dual_point_alignment.py

class PrecisionPlaceController:
    def set_cartesian_velocity(self, vx: float, vy: float, vz: float,
                                wx: float = 0, wy: float = 0, wz: float = 0) -> bool:
        """设置笛卡尔速度"""
        # 1. 雅可比矩阵：笛卡尔速度 → 关节速度
        joint_velocities = self._jacobian @ np.array([vx, vy, vz, wx, wy, wz])

        # 2. 构建速度指令
        velocity_dict = {
            name: float(joint_velocities[i])
            for i, name in enumerate(self.robot.observation_joint_names)
            if i < len(joint_velocities)
        }

        # 3. 发送
        self.robot.send_velocity(velocity_dict)
        return True

    def has_velocity_control(self) -> bool:
        return hasattr(self.robot, 'send_velocity')
```

## 5. 临时优化方案

在硬件支持速度控制之前，可优化位置增量模式：

### 5.1 减小步长提高频率

```python
# 当前
dt = 0.05  # 20Hz
velocity_scale = 0.8

# 优化
dt = 0.02  # 50Hz，更平滑
velocity_scale = 0.5  # 降低增益避免过冲
```

### 5.2 轨迹插值

```python
def smooth_interpolation(start_pos, end_pos, duration, dt):
    """生成平滑轨迹点（S曲线）"""
    num_points = int(duration / dt)
    t = np.linspace(0, 1, num_points)

    # S曲线速度 profile
    s = 3 * t**2 - 2 * t**3  # smoothstep

    trajectory = []
    for i in range(num_points):
        pos = start_pos + s[i] * (end_pos - start_pos)
        trajectory.append(pos)

    return trajectory
```

### 5.3 增益调度

```python
# 根据误差动态调整速度
def adaptive_velocity_scale(error, max_error=50.0):
    """误差大时快，误差小时慢"""
    if error > max_error:
        return 0.8
    else:
        return 0.3 + 0.5 * (error / max_error)
```

## 6. 测试计划

### 6.1 单元测试

```python
# tests/test_velocity_control.py

def test_velocity_interface():
    """测试速度控制接口"""
    robot = MockRobot()
    assert robot.has_velocity_control() == True

    success = robot.set_cartesian_velocity(0.1, 0, 0)
    assert success == True

def test_ibvs_with_velocity():
    """测试IBVS使用速度控制"""
    # ...
```

### 6.2 集成测试

1. 速度模式切换测试
2. 速度指令响应测试
3. IBVS对齐精度测试
4. 平滑度评估（加速度计测量）

## 7. 风险与注意事项

### 7.1 安全风险

- 速度模式需要限幅保护
- 紧急停止功能必须有效
- 首次测试应在低速下进行

### 7.2 兼容性

- 确保不影响现有的位置控制模式
- 保存原有关节限位保护逻辑
- 考虑模式切换的平滑过渡

### 7.3 性能

- 速度控制的延迟可能影响IBVS稳定性
- 需要调整增益参数
- 可能需要前馈补偿

## 8. 参考资料

- IBVS论文: "Visual Servoing: A Tutorial" (Hutchinson et al., 1996)
- 机器人速度控制: "Robot Modeling and Control" (Spong et al.)
- 相关代码:
  - `precision_place/calibration/ibvs_controller.py`
  - `precision_place/robot/interface.py`
  - `lerobot/robots/supre_robot_follower/supre_robot_follower.py`

---

## 开发检查清单

- [ ] 获取电机驱动器SDK文档
- [ ] 确认是否支持速度模式
- [ ] 实现 EyouMotorHardware 速度控制
- [ ] 实现 SupreRobotHardwareManager 速度分发
- [ ] 实现 SupreRobotFollower.send_velocity()
- [ ] 实现 PrecisionPlaceController.set_cartesian_velocity()
- [ ] 测试速度控制功能
- [ ] 调整IBVS增益参数
- [ ] 安全测试与验证

**文档维护者**: 开发完成后请更新此文档状态