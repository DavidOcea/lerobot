# ADC2 真机验收清单（Tier 4 / Tier 5）

Phase 3（LLM planner loop）+ Phase 4（VisualSuccessDetector）离线测试已全绿
（`verify_routing_graph.py` / `verify_llm_config.py` / `verify_llm_planner.py` /
`verify_visual_detector.py` / `verify_tool_spec.py` / `verify_skill_equivalence.py`）。
但离线只能证明**接线正确**，不能证明**硬件上行为正确**。本清单覆盖离线测不到的
三件事，必须在真机（Supre + AGV + 相机）上跑：

1. **LLM 真联网**——离线只验证了 `build_planner("api", …)` 能构造 `OpenAIChatPlanner`
   （无网络），真机才验证 LLM 真的能选 skill + 恢复。
2. **视觉判据真读相机**——离线只验证了合成 numpy 帧上的 `BaselineDiffDetector`，
   真机才验证 `head_cam` 真帧 + `settle_delay` 时序。
3. **全链路端到端**——离线只验证了路由图，真机才验证 AGV / 关节 / 分类器协同。

> **环境**：真机上的 conda 环境 `lerobot`（`~/miniconda3/etc/profile.d/conda.sh` + `conda activate lerobot`，
> 对应 `/root/miniconda3/envs/lerobot`）。入口统一用 `scripts/run_task_sequence.sh`，
> 等价于 `python src/lerobot/scripts/run_task_agent.py --config <yaml> [...]`。

---

## 前置条件（每次真机运行前）

- [ ] **热机 30s**：冷启动力矩反馈有偏移（见 `[[reboot-first-inference-shaking-root-cause]]`），
      task-agent 的 force 观测同样受影响，务必等 30s 再跑。
- [ ] **急停在手**：急停按钮 / `--no-emergency-stop` 默认**不**加；确认 e-stop 阈值未改。
- [ ] **相机对焦**：`head_cam`（视觉判据默认相机）与 `left_wrist_cam`/`right_wrist_cam` 出图正常、无遮挡。
- [ ] **真实 LLM endpoint 就绪**（仅 Tier 4.3+ 需要）：
      准备一个真实推理端点（本地 vLLM / ollama / 或可达的托管端点），记下 `base_url`、`model`、`api_key`（或 `api_key_env`）。

---

## Tier 4 —— LLM + 视觉判据硬件在环软启动

**目标**：在真机上、最小动作范围内，分别点亮 LLM 回路与视觉判据，且默认 flash 路径
**bit-identical 无感**不被破坏。全程可用 `--tasks` 缩小动作范围。

### 4.0 dry-run 复验（真机环境）

在真机上复跑离线已绿的两条，排除「机器环境 ≠ 开发环境」差异：

```bash
scripts/run_task_sequence.sh /root/workspace/dc_dir/agv_class_ADC2.yaml --dry-run
scripts/run_task_sequence.sh configs/agv_class_longloushang_yanshi_V0.6.yaml --dry-run
```

- [ ] flash 版打印 32 个 task（`planner=flash`、`visual_detector enabled=False`）。
- [ ] llm 版不报错，能解析顶层 `planner: llm` / `llm:` / `visual_detector:` 三个键。
- [ ] **判据**：两版都干净退出 `exit 0`；任何 `agent:` 嵌套 / 字段拼写错误在此步就会爆。

### 4.1 flash 无感回归（默认路径 bit-identical）

默认 `planner="flash"` + `visual_detector.enabled=False` 必须和 Phase 4 之前**逐字节一致**。
用一个已知 good 的 flash run 作基线（Phase 1/2 交付时的日志），跑一遍对比：

```bash
scripts/run_task_sequence.sh /root/workspace/dc_dir/agv_class_ADC2.yaml --debug 2>&1 | tee /tmp/flash_regression.log
```

- [ ] 执行序列、每 task 的 `result.success` / `next_task` 与基线一致（无 `visual_evidence` 字段泄漏进 `final_observation`）。
- [ ] 日志无 `visual_detector` / `llm_loop` / `OpenAIChatPlanner` 相关行（默认不可达、懒加载未触发）。
- [ ] **判据**：`diff` 基线日志 vs 新日志，仅时间戳 / PID 差异。任何语义差异 = 无感被破坏，停。

### 4.2 LLM 回路软启动（真实 endpoint + 单 skill 子集）

> ⚠️ **不要直接用 `agv_class_longloushang_yanshi_V0.6.yaml`**——它指向 `127.0.0.1:9999` / `fake-model`，
> 是离线 fixture。真机先复制一份并改 `llm:` 块为真实端点：

```bash
cp configs/agv_class_longloushang_yanshi_V0.6.yaml configs/agv_class_longloushang_yanshi_V0.6_prod.yaml
# 编辑 llm: 块 → base_url / model / api_key（或 api_key_env=OPENAI_API_KEY）
```

先用 `--tasks` 跑一个最小 skill 子集（如单站 pickup 的 classify + pickup），验证 LLM 真的在选 skill：

```bash
scripts/run_task_sequence.sh configs/agv_class_longloushang_yanshi_V0.6_prod.yaml \
  --tasks classify_workpiece,pickup_at_station_00 --debug 2>&1 | tee /tmp/llm_loop.log
```

- [ ] 日志出现 LLM 的 tool-call 决策（`pickup` / `navigate_to` 等），而非 flash 的固定顺序。
- [ ] LLM `max_tool_calls` 上限生效（fixture 里 = 6；到上限后按恢复路径收尾，不无限循环）。
- [ ] `planner=llm` 分派生效（`run()` 里 `getattr(config,"planner","flash")=="llm"` 走 `_run_llm_loop`）。
- [ ] **判据**：机器人按 LLM 决策执行、正常收尾；无 `ValueError: Unsupported planner type`。

### 4.3 视觉判据软启动（evidence-only 验证）

开启 `visual_detector.enabled: true`（prod yaml 里已设）后，验证它**只写证据、不改结论**：

- [ ] 日志出现 `visual_evidence`（`success`/`changed_fraction`/`confidence`），落在 **pickup（grasp）** 节点的
      `final_observation`，而非 agv retreat 节点。
- [ ] **evidence-only 铁律**：即使判据说 `success=False`（如误报「料片没消失」），
      `result.success` / `next_task` / 执行流**完全不变**——机器人照常走 agent 自报的结论。
- [ ] 抓取前后 `head_cam` 恰好读 2 次（pre + post），中间 `settle_delay=0.5s` 生效。
- [ ] **判据**：手动制造一个「判据与 agent 结论冲突」的场景（如料片实际抓走但判据误报），
      确认机器人按 agent 结论继续，判据只作为 `visual_evidence` 供 LLM 参考。

---

## Tier 5 —— ADC2 全流程端到端

**目标**：跑通真实 ADC2 完整 happy path + 抽样失败分支，确认路由图在硬件上成立。

### 5.1 happy path 全链

flash 版（先验证无 LLM 的确定性基线）：

```bash
scripts/run_task_sequence.sh /root/workspace/dc_dir/agv_class_ADC2.yaml --debug 2>&1 | tee /tmp/adc2_happy.log
```

- [ ] 实际执行序列 = `agv_to_station_A → align_to_tag → classify_workpiece →
      pickup_at_station_00 → agv_backward_50cm_A1_1 → check_material_before_pickup_A1 →
      agv_to_station_B1 → check_material_before_place_B1 → agv_forward_5cm_B1_1 →
      place_long_1 → home_and_return_A_after_long_1 → agv_return_from_B1_to_A`
      （与 `verify_routing_graph.py` 的 `HAPPY_PATH` 一致）。
- [ ] classify 真实识别出 `long_00`（YOLO/AprilTag 真图，非 mock 随机噪声）。
- [ ] 全程夹爪 / AGV / 关节无碰撞触发 e-stop。

### 5.2 失败分支抽样（路由图的真机版）

离线路由图已验证 7 节，真机抽样复验其中**有硬件语义**的 3 条（其余纯路由不必真机跑）：

| 分支 | 触发方式 | 期望路径 | 判据 |
|------|---------|---------|------|
| pickup fail 重试 | 手动让 pickup 失败 1 次 | `voice_pickup_failed(success)` → 重新 `classify_workpiece` → 再 pickup | 有限次重试，不无限循环 |
| pickup fail 退避 | pickup 彻底失败 | `voice_pickup_failed(failed)` → `agv_retreat_50cm` | 退避 + cycle_end 收尾 |
| 物料掉落 | pickup 后检测 `none`/`unknown` | `voice_material_dropped_A` → `agv_return_to_A_for_repick` | 触发重抓 |

- [ ] 每条路径 AGV / 关节动作与实际物流动一致（路由图的 `next_task` 跳变在硬件上成立）。
- [ ] classify 重试耗尽（`exhausted` + recovery）走 `timeout_return_home → agv_retreat_50cm` 收尾。

### 5.3 12 站全分支（可选，分站跑）

- [ ] 若需覆盖 12 个 `long_XX` 标签，用 `--tasks` 分站跑（如 `--tasks classify_workpiece,pickup_at_station_XX`），
      不必一次全跑。目的仅验证 classify 真图能稳定产出 12 个标签。

---

## 回退 / 急停协议

- [ ] 任何一步出现 e-stop / 碰撞：立即手动急停，`Ctrl-C` 触发 `KeyboardInterrupt`（`exit 130`）。
- [ ] LLM 乱选 skill / 超出动作范围：`--no-emergency-stop` **不加**，靠 e-stop 兜底；随后回退
      `planner: flash` 的 `agv_class_ADC2.yaml` 恢复确定性。
- [ ] 视觉判据真机误报过多：把 prod yaml 的 `visual_detector.enabled` 改回 `false` 即可恢复无感，
      不影响主链（evidence-only，无状态残留）。
- [ ] 本清单**不在本开发环境执行**（无 AGV / 关节 / 真相机），由真机操作者手动勾选完成。
