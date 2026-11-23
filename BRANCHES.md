# 分支说明（Branch Guide）

> 本项目（双机械臂协作搬箱）的多个分支分别面向不同功能层。本文说明各分支的定位、主要功能和使用场景，
> 便于按需切换。切换分支前请先读对应分支的进展记录（见记忆 `feedback_branch_switch`）。

## 总览

| 分支 | 定位 | 面向 | 活跃度 |
|---|---|---|---|
| `temp-agent` | 系统控制整合（执行层） | 整合机器人所有技能完成指定任务（搬箱） | 🟢 当前主力 |
| `unified_dev_0624` | 数据采集 + 模型训练（训练层） | 试教采集、DP/ACT/smolvla 策略训练 | 🟢 主力 |
| `temp-rough-dh` | 运动学标定（标定层） | VIVE 测 DH，为执行层提供运动学 | 🟡 进行中 |
| `tep_force` | 残差强化学习（RL 层） | RL 微调策略（Residual RL Phase2） | 🟡 进行中 |
| `pybullet` | 仿真训练/评估 | 仿真环境跑 pi0/ACT | 🟡 独立 |
| `feat/skill-ification` | 技能化 + Agent 编排 | 技能封装、LLM/coding-agent 规划 | 🟡 实验 |
| `reverse_dev` | ACT 训练实验（早期） | Label Smoothing、relative action 等 | ⚪ 早期/冻结 |
| `force` | ACT 力/视频实验（早期） | relative action、视频解码、force embedding | ⚪ 早期/冻结 |

---

## 详细说明

### temp-agent —— 系统控制整合（执行层）【当前分支】

把机器人的所有硬件技能整合起来，通过任务编排器完成"搬箱"等指定任务。

- **任务编排**：`orchestrator`（position_sequence / classify / agv / policy 等任务类型，条件分支）
- **机器人硬件**：EYou/supre 双机械臂、Jodell 夹爪、AGV
- **安全系统**：碰撞检测、接触检测、紧急停止
- **接触/柔性控制家族**：`stop_on_contact`（接触停止）、`drop_rate`（卸载检测）、
  `compliant_stop`（CST 软弹簧）、`delta`（相对移动/微退）、`record_force`（受力记录）
- **柔性示教**：零重力漂浮（G 命令）、CST→CSP 回锁

#### 使用教程

**启动任务**：

```bash
scripts/run_task_sequence.sh configs/agv_pick_place.yaml
# 等价于
python src/lerobot/scripts/run_task_agent.py --config configs/agv_pick_place.yaml
```

**命令行参数**：

| 参数 | 作用 |
|---|---|
| `--config <yaml>` | 任务配置文件 |
| `--tasks a,b,c` | 只跑指定任务（逗号分隔） |
| `--debug` | 调试日志 |
| `--dry-run` | 仅校验配置不执行 |
| `--interactive` | 每个任务执行前等人确认 |
| `--max-retries N` | 覆盖重试次数 |
| `--max-duration S` | 覆盖最大时长 |
| `--collision-threshold X` | 覆盖碰撞阈值(Nm) |
| `--emergency-force-threshold X` / `--emergency-max-velocity V` | 急停参数 |
| `--no-emergency-stop` | 禁用急停 |

**config yaml 结构**：

```yaml
robot_config: { type, control_frequency, max_relative_joint_move, cameras }
collision_config: { collision_threshold, detection_window, adaptive_mode, velocity_compensation, auto_calibrate }
agv_config: { host, port, station_map, emergency_stop_enabled, check_arm_before_move }
named_positions: { 位姿名: { joint_name: 角度 } }
tasks:
  - name: xxx
    task_type: position_sequence     # 或 classify / agv / policy / visual_align
    max_duration: 30.0
    max_retries: 1
    enabled: true
    steps: [ ... ]
```

**步骤级参数**（`position_sequence` 的每个 step，详见速查记忆 `eyou-contact-compliant-features`）：

- **运动**：`position`（绝对目标）/ `delta`（相对增量，二者互斥）、`max_duration`（上限）、
  `position_tolerance`、`overlap_next`（连续链消除停顿）、`speed_deg_per_s`（默认 30，调小减速）
- **接触**：`stop_on_contact`、`contact_joints`、`contact_mode`（`abs` 接触/`drop_rate` 卸载）、
  `contact_torque_threshold`(Nm)、`contact_rate_threshold`(Nm/s)
- **柔性**：`compliant_stop`、`compliant_joints`、`compliant_kp/kd`、`compliant_stop_drift_deg` 等
- **记录**：`record_force`、`force_log_path`

### unified_dev_0624 —— 数据采集 + 模型训练（训练层）

试教数据采集和 VLA 策略训练。

- **数据采集**：record / teleop 录制
- **策略训练**：DP、ACT、smolvla
- **训练增强**：`mirror_prior`（镜像先验）、`relative_action`（相对动作）、
  `vision_backbone_output_stride`（空间定位）、数据增强

#### 使用教程

**数据采集（record）**：

```bash
python -m lerobot.record \
  --robot.type=supre_robot_follower --robot.id=supre_robot_follower \
  --teleop.type=supre_robot_leader --teleop.id=supre_robot_leader \
  --robot.cameras="{head_cam:{type:opencv,index:0,width:640,height:480,fps:30}}" \
  --dataset.single_task="Grasp the workpiece and put it in place." \
  --dataset.repo_id=supdata/dataset_xxx \
  --dataset.episode_time_s=30 --dataset.num_episodes=2 --dataset.reset_time_s=10
```

**训练（`python -m lerobot.scripts.train`）**：

```bash
python -m lerobot.scripts.train --policy.type=diffusion | act ...
```

**DP（`--policy.type=diffusion`）关键参数**：

| 参数 | 影响 |
|---|---|
| `horizon` | 预测时域（越长越连续、越难训） |
| `n_action_steps` | 每次执行步数 |
| `n_obs_steps` | 观察帧数 |
| `drop_n_last_frames` | 丢弃末尾帧 |
| `down_dims` | 特征维度 |
| `crop_shape` | 输入分辨率（`[84,84]`/`[224,224]`） |
| `noise_scheduler_type` / `num_inference_steps` | 采样方式/步数 |

**ACT（`--policy.type=act`）关键参数**：

| 参数 | 影响 |
|---|---|
| `chunk_size` | 动作分块 |
| `n_obs_steps` | 观察帧数 |
| `temporal_ensemble_coeff` | 时序集成系数 |
| `use_vae` / `latent_dim` | VAE 编码 |
| `force_dropout` / `state_dropout` | 力/状态 dropout（去摩擦噪声） |
| `use_relative_action` | 相对动作（delta） |

**训练通用参数**：`batch_size`、`steps`、`eval_freq`、`save_freq`、`seed`、
`output_dir`、`job_name`、`--policy.device`、`--wandb.enable`

> 完整实验登记/参数见各 `scripts/run_dp_*.sh`（DP 实验）与 `scripts/train_fdrop03.sh`（ACT）。

### temp-rough-dh —— 运动学标定（标定层）

基于 `temp-agent`，加运动学模板 + VIVE 标定，为执行层提供精确 DH。

- **运动学模板**：screw / Skeleton（PoE）
- **VIVE 标定**：CPA 反测 DH/PoE、基座对齐、UDP 链路、采集（阶段 1-4）

### tep_force —— 残差强化学习（RL 层）

Residual RL（残差强化学习）微调策略。

- Phase2 训练修复、`eval_residual.py` 推理脚本、pause/resume、leader 可选

### pybullet —— 仿真训练/评估

仿真环境训练/评估策略。

- `sim_robots`、pi0/ACT 评估、heatmap、state_dropout

### feat/skill-ification —— 技能化 + Agent 编排

把机器人技能封装成可编排的 skill，用 LLM/coding-agent 规划执行。

- visual success detector（Phase 4）、ADC2 硬件验收（5-gate runbook）、
  coding-agent planner（Phase 6）

### reverse_dev / force —— ACT 训练早期实验（早期/冻结）

较早的 ACT/pi0 训练实验，`force` 从 `reverse_dev` 延续。

- Label Smoothing、WarmupCosineScheduler、relative action、multi img transforms、
  pi0 heatmap、视频解码（ideal vs actual timestamp）、force embedding
- ⚠️ 提交停留较早（2026 上半年），功能可能未整合进 unified_dev_0624，状态待确认

---

## 分支关系（层次）

```
                    ┌── reverse_dev → force（ACT 早期实验，冻结）
  训练层 ───────────┼── unified_dev_0624（数据采集 + DP/ACT/smolvla 训练）
                    ├── tep_force（残差 RL）
                    └── pybullet（仿真）

  标定层 ─────────── temp-rough-dh（VIVE 测 DH，基于 temp-agent）

  执行层 ─────────── temp-agent（系统整合，当前主力）← temp-rough-dh 的基座

  Agent 层 ───────── feat/skill-ification（技能化 + LLM 编排）
```

- **temp-agent** 是执行主线，**temp-rough-dh** 基于它加运动学标定；
- **unified_dev_0624 / tep_force / pybullet** 是训练侧，产出策略供 temp-agent 调用；
- **reverse_dev / force** 是训练侧的早期实验，可能已冻结。

---

## 使用建议

| 想做什么 | 用哪个分支 |
|---|---|
| 跑真机任务、调接触/柔性/放置动作 | `temp-agent` |
| 采集试教数据、训练 DP/ACT/smolvla | `unified_dev_0624` |
| VIVE 标定、测 DH | `temp-rough-dh` |
| RL 微调策略 | `tep_force` |
| 仿真跑策略 | `pybullet` |
| 技能封装、LLM 编排实验 | `feat/skill-ification` |
