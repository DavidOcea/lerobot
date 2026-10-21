# ADC2 真机验收手册（Tier 4/5 展开版）

> 本文是 5-tier 测试流程的最后两层（真机），**无法在本开发环境执行**，由真机操作者逐条勾选。
> 离线 Tier 0–3 已全绿，见 `scripts/verify_routing_graph.py` / `verify_llm_config.py` /
> `verify_llm_planner.py` / `verify_visual_detector.py` / `verify_tool_spec.py`。

环境：真机 conda 环境 `lerobot`（`~/miniconda3/etc/profile.d/conda.sh` + `conda activate lerobot`）。
统一入口 `scripts/run_task_sequence.sh <yaml> [...]`，等价于
`python src/lerobot/scripts/run_task_agent.py --config <yaml> [...]`。

---

## 0. 总览：闸门顺序 + 风险分级

真机测试按风险从低到高**逐 gate 推进**，每个 gate 通过才进下一个；任一 gate 失败立即停，
按「失败动作」回退，不带到下一层。

| Gate | 内容 | 风险 | 前置 | 通过判据（摘要） | 失败动作 |
|------|------|------|------|------------------|----------|
| G0 | 配置解析 dry-run | 无 | — | 两个 yaml 干净 `exit 0`，32 task | 停，修 YAML 语法/字段 |
| G1 | flash 无感回归 | 中（机器人动，确定性） | G0 | 执行序列/`next_task` 与基线逐字节一致 | 停，查无感破坏 |
| G2 | LLM 回路软启动 | 高（LLM 决策） | G1 | `initialized` 出现 + 4 条降级 warning 均无 + 按 skill 短链执行 | 停，回退 flash |
| G3 | 视觉判据 evidence-only | 低（只读不写结论） | G2 | `visual_evidence[...]` 日志出现 + 不改 `success`/`next_task` | 停，`enabled=false` 关掉 |
| G4 | 全链端到端 | 高 | G2+G3 | happy path 全链 + 失败分支抽样正确 | 停，按分支表定位 |

---

## 1. 一次性准备

### 1.1 真机环境

- [ ] conda 环境正确：`conda activate lerobot` 后 `which python` 指向 `/root/miniconda3/envs/lerobot/bin/python`。
- [ ] **热机 30s**：冷启动力矩反馈有偏移（见 [[reboot-first-inference-shaking-root-cause]]），
      会在 force 观测里制造尖峰，等 30s 让偏移趋零再跑任何会动臂的步骤。
- [ ] 急停在手：e-stop 按钮可用；确认**不加** `--no-emergency-stop`，阈值未改。
- [ ] 三相机出图：`head_cam`（视觉判据默认相机）、`left_wrist_cam`（精细对齐）、`right_wrist_cam` 均正常、无遮挡。
- [ ] AGV 可连、在位：AGV TCP 端点可达（flash 里 AGV 连不上会 `SKIPPED`，先确认）。
- [ ] 料片就位：至少一块 long 料片在 00 站，供 G1 抓取。

### 1.2 真实 LLM endpoint（三选一）

LLM 用 OpenAI 兼容 tool-call 协议，**模型必须支持 function-calling**（否则 `decision.tool_call`
恒为 `None`，`finish_reason="stop"` 直接收尾或降级 flash）。三选一：

**A. 本地 vLLM（推荐，若真机/内网有 GPU）**
```yaml
llm:
  model: <served-model-name>          # vLLM --served-model-name 的名字
  base_url: "http://<host>:8000/v1"
  api_key: "EMPTY"                    # vLLM 不校验 key，填任意串
  max_tool_calls: 6
  temperature: 0.0
  timeout: 60.0
```

**B. 本地 ollama（OpenAI 兼容端点）**
```yaml
llm:
  model: "qwen2.5:7b"                 # 需支持 tool-calling 的模型
  base_url: "http://127.0.0.1:11434/v1"
  api_key: "ollama"
  max_tool_calls: 6
  temperature: 0.0
  timeout: 60.0
```

**C. 托管 OpenAI 兼容 API**
```yaml
llm:
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"       # 或 api_key: "sk-..."（不推荐落盘明文）
  max_tool_calls: 6
  temperature: 0.0
  timeout: 60.0
```

- [ ] 先用一个最小 curl 验证端点可达 + 模型能返回 tool-call（见附录 B.2）。

### 1.3 生成 prod yaml（两个开关）

离线 fixture `configs/agv_class_longloushang_yanshi_V0.6.yaml` 指向 `127.0.0.1:9999`/`fake-model`，
**不可直接真机跑**。复制并改 `llm:` 块；`visual_detector.enabled` 先关（G2 隔离 LLM，G3 再开）：

```bash
cd /root/workspace/dc_dir/lerobot
cp configs/agv_class_longloushang_yanshi_V0.6.yaml configs/agv_class_longloushang_yanshi_V0.6_prod.yaml
# 编辑 prod yaml：
#   1) llm: 块 → 真实 base_url / model / api_key（按 1.2 三选一）
#   2) visual_detector.enabled: false     # G2 先关，G3 再开 true
```

- [ ] prod yaml 里 `planner: llm` 保持（G2/G3 都在 LLM 模式下跑）。

### 1.4 ⚠️ 文件同名陷阱（务必核对）

真机上有两个 `agv_class_ADC2.yaml`，内容不同：

| 路径 | 内容 | 用途 |
|------|------|------|
| `/root/workspace/dc_dir/agv_class_ADC2.yaml` | **32 节点**真机 source of truth | flash 回归/全链 **用这个** |
| `configs/agv_class_ADC2.yaml` | **8 节点**旧拓扑（repo 内遗留） | **不要用**，只作对照 |

- [ ] G1/G4 的 flash 步骤全部用 **32 节点**那个绝对路径；不确定就用
      `grep -c "^  - " <yaml>` 或 dry-run 看 task 数（32 才对）。

---

## 2. Gate 0 — 配置解析（dry-run）

**目的**：排除「机器环境 ≠ 开发环境」的解析差异。零风险，纯静态。

```bash
cd /root/workspace/dc_dir/lerobot

# flash（32 节点真机版）
scripts/run_task_sequence.sh /root/workspace/dc_dir/agv_class_ADC2.yaml --dry-run 2>&1 | tee /tmp/g0_flash.log

# llm prod 版
scripts/run_task_sequence.sh configs/agv_class_longloushang_yanshi_V0.6_prod.yaml --dry-run 2>&1 | tee /tmp/g0_llm.log
```

- [ ] 两条都 `exit 0`，无 traceback。
- [ ] flash 版日志含 `Tasks: 32`；llm 版同样能解析（task 数一致）。
- [ ] llm 版无 `agent:` 嵌套报错、无字段拼写报错（顶层 `planner`/`llm`/`visual_detector` 三个键生效）。

**判读**：`grep -E "Tasks:|Error|Traceback" /tmp/g0_*.log` —— 有 `Tasks: 32` 且无 `Error`/`Traceback` 即过。
**失败**：YAML 语法/字段问题，先修到 G0 过再谈动臂。

---

## 3. Gate 1 — flash 无感回归（默认路径 bit-identical）

**目的**：确认 Phase 3/4 的改动在默认 `planner=flash` + `visual_detector.enabled=False` 下
**逐字节无感**，机器人行为与改动前完全一致。用 Phase 1/2 交付时留存的 good 基线日志作对照。

```bash
scripts/run_task_sequence.sh /root/workspace/dc_dir/agv_class_ADC2.yaml --debug 2>&1 | tee /tmp/g1_flash.log
```

- [ ] 实际执行序列 = 基线序列（32 节点图，含 `voice_*`/`check_material_*`/`home_and_return_*`）。
- [ ] 每 task 的 `result.success` / `next_task` 与基线一致。
- [ ] **无** `visual_evidence` / `llm_loop` / `OpenAIChatPlanner` / `visual_detector` 字样泄漏进日志
      （默认不可达、懒加载未触发）。
- [ ] 无碰撞触发 e-stop。

**判读**：
```bash
grep -c "visual_evidence\|LLM planner\|OpenAIChatPlanner\|visual_detector" /tmp/g1_flash.log   # 期望 0
diff <(grep -vE "asctime|PID|timestamp" /tmp/baseline_good.log) \
     <(grep -vE "asctime|PID|timestamp" /tmp/g1_flash.log)                                     # 仅时间戳差异
```

**失败**：任何语义 diff = 无感被破坏，停，逐条比对定位是哪个 task 的 `next_task` 或结果变了。

---

## 4. Gate 2 — LLM 回路软启动

**目的**：验证 LLM 真的在选 skill、机器人按 LLM 决策执行，且降级路径（坏端点/坏 tool）安全。
前置：prod yaml 的 `visual_detector.enabled: false`（先隔离 LLM 变量）。

> ⚠️ **LLM 模式不认 `--tasks`**——那个过滤器只作用于 flash 的 32 节点图（`config.tasks`）；
> LLM 循环靠 `to_tool_spec()` 决定 skill、`_resolve_station` 隐藏预跑 classify 也不看 `enabled`。
> 缩小范围靠两招：① `max_tool_calls: 6` 天然兜底；② 临时在 prod yaml 的 `llm:` 块加
> `system_prompt: "只调用一次 pickup 抓取 long 料片，然后结束"` 把 LLM 约束到单 skill（G4 前删掉）。

```bash
scripts/run_task_sequence.sh configs/agv_class_longloushang_yanshi_V0.6_prod.yaml --debug 2>&1 | tee /tmp/g2_llm.log
```

- [ ] 启动日志出现 `LLM planner initialized (model=...)` —— 证明 `build_planner("api", …)` 成功、未降级。
- [ ] 全程**无** 4 条降级 warning（见附录 A），尤其无 `falling back to flash`。
- [ ] 执行是 **skill 短链**（如 `pickup_at_station_00` + `agv_backward_50cm_…`），
      而非 32 节点全图 —— 这证明走的是 `run_llm_loop` 而非 flash 重放。
- [ ] 机器人按 LLM 决策实际抓取/退避，动作正确。

**判读**：
```bash
grep "LLM planner initialized" /tmp/g2_llm.log                                        # 期望 ≥1
grep -cE "falling back to flash|unreachable|construction failed|Invalid tool call|budget exhausted" /tmp/g2_llm.log   # 期望 0
```

**降级路径专测**（把 prod yaml 的 `base_url` 改成不可达的 `http://127.0.0.1:9999/v1`，`max_tool_calls` 不动）：
- [ ] 出现 `LLM unreachable (…) falling back to flash`，机器人回退到确定性 flash 重放并正常收尾
      （不会卡死、不会无限循环）。
- [ ] 测完改回真实端点。

**失败**：`initialized` 缺失或出现降级 warning → 先查端点可达（curl）/ 模型是否支持 tool-call（见附录 B.2）；
不改动、不回退 flash。

---

## 5. Gate 3 — 视觉判据 evidence-only

**目的**：验证 `BaselineDiffDetector` 真读 `head_cam` 真帧、verdict 落在 pickup（grasp）节点、
且**只写证据不改结论**。前置：prod yaml `visual_detector.enabled: true`。
同样用临时 `system_prompt` 约束到单次 pickup（`--tasks` 在 LLM 模式无效）：

```bash
scripts/run_task_sequence.sh configs/agv_class_longloushang_yanshi_V0.6_prod.yaml --debug 2>&1 | tee /tmp/g3_visual.log
```

- [ ] 出现 `visual_evidence[pickup_at_station_00]: success=… changed_fraction=…` 日志行
      （Phase 4 本次补的观测行，证明 detector 真的触发）。
- [ ] 抓取前后 `head_cam` 恰好读 2 次，中间 `settle_delay=0.5s`（日志里 sleep 前后帧时间差可核）。
- [ ] **evidence-only 铁律**：故意制造「判据与 agent 结论冲突」——如抓取实际成功但用
      `diff_threshold` 调高到几乎必不触发，或反之——确认 `result.success`/`next_task`/执行流
      **完全不变**，机器人照 agent 自报结论走，判据只进 `final_observation["visual_evidence"]`。

**判读**：
```bash
grep "visual_evidence\[" /tmp/g3_visual.log          # 期望：每个 pickup grasp 节点一行
```

**失败**：无 `visual_evidence` 行 → 查 `visual_detector.enabled: true` 是否生效、`skill_name=="pickup"`
是否命中、相机 key 是否 `head_cam`。证据污染结论 → 立即把 `enabled` 改回 `false` 并排查（不应发生，
离线 5 节已锁，但真机帧格式差异需坐实）。

---

## 6. Gate 4 — 全链端到端

**目的**：跑通真实 ADC2 完整 happy path + 抽样失败分支，确认路由图在硬件上成立。

### 6.1 happy path 全链（flash 确定性基线）

```bash
scripts/run_task_sequence.sh /root/workspace/dc_dir/agv_class_ADC2.yaml --debug 2>&1 | tee /tmp/g4_happy.log
```

- [ ] 实际执行序列 = `agv_to_station_A → align_to_tag → classify_workpiece →
      pickup_at_station_00 → agv_backward_50cm_A1_1 → check_material_before_pickup_A1 →
      agv_to_station_B1 → check_material_before_place_B1 → agv_forward_5cm_B1_1 →
      place_long_1 → home_and_return_A_after_long_1 → agv_return_from_B1_to_A`
      （与 `verify_routing_graph.py` 的 `HAPPY_PATH` 一致）。
- [ ] classify 真实识别出 `long_00`（YOLO/AprilTag 真图，非 mock 噪声）。
- [ ] 全程夹爪/AGV/关节无碰撞触发 e-stop。

### 6.2 失败分支抽样（路由图真机版）

离线路由图 7 节已过，真机只复验**有硬件语义**的 3 条（见第 7 节表）。其余纯路由跳变不必真机跑。

---

## 7. 失败分支抽样表

| 分支 | 触发方式 | 期望路径 | 判据 |
|------|---------|---------|------|
| pickup fail 重试 | 手动让 pickup 失败 1 次 | `voice_pickup_failed(success)` → 重新 `classify_workpiece` → 再 pickup | 有限次重试，不无限循环 |
| pickup fail 退避 | pickup 彻底失败 | `voice_pickup_failed(failed)` → `agv_retreat_50cm` | 退避 + cycle_end 收尾 |
| 物料掉落 | pickup 后检测 `none`/`unknown` | `voice_material_dropped_A` → `agv_return_to_A_for_repick` | 触发重抓 |

- [ ] 每条路径 AGV/关节动作与实际物流动一致（`next_task` 跳变在硬件上成立）。
- [ ] classify 重试耗尽（`exhausted` + recovery）走 `timeout_return_home → agv_retreat_50cm` 收尾。

### 6.3 12 站全分支（可选，分站跑）

- [ ] 若需覆盖 12 个 `long_XX` 标签，用 `--tasks` 分站跑（如
      `--tasks classify_workpiece,pickup_at_station_XX`），不必一次全跑。目的仅验证 classify 真图
      能稳定产出 12 个标签。

---

## 8. 回退 / 急停协议

- [ ] 任何一步 e-stop / 碰撞：立即手动急停，`Ctrl-C` 触发 `KeyboardInterrupt`（`exit 130`）。
- [ ] LLM 乱选 skill / 超出动作范围：`--no-emergency-stop` **不加**，靠 e-stop 兜底；随后
      `planner: flash` 的 `/root/workspace/dc_dir/agv_class_ADC2.yaml` 恢复确定性。
- [ ] 视觉判据真机误报过多：prod yaml `visual_detector.enabled: false` 即可恢复，无状态残留
      （evidence-only）。
- [ ] 降级回退 flash 是**安全特性**：LLM 不可达/坏 tool/budget 耗尽都自动回退确定性重放，
      不会停在半途。

---

## 附录 A — 日志行速查表

| 日志行（子串） | 含义 | gate |
|----------------|------|------|
| `LLM planner initialized (model=…)` | `build_planner` 成功，未降级 | G2 |
| `LLM planner construction failed (…); falling back to flash` | 坏 base_url/model/key | G2 降级 |
| `LLM unreachable (…); falling back to flash` | 传输层连不上 | G2 降级 |
| `Invalid tool call (…); falling back to flash` | LLM 发出未知 tool / 坏参数 | G2 降级 |
| `LLM budget exhausted (…); falling back to flash` | 达到 `max_tool_calls` 上限 | G2 降级 |
| `No classify task configured; defaulting station to '00'` | pickup 前无 classify 预跑 | G2 注意 |
| `Classify did not yield a 'long_XX' label (…); defaulting to '00'` | 分类器没出 long 标签 | G2/G4 注意 |
| `visual_evidence[<task>]: success=… changed_fraction=…` | 视觉判据触发（Phase 4 观测行） | G3 |

## 附录 B — 可观测性缺口 + 预检

### B.1 已补 vs 仍缺

- ✅ **已补**（本次）：视觉判据观测行 `visual_evidence[...]`（`llm_loop.py`，无感）。
- ⚠️ **仍缺**：`OpenAIChatPlanner` 不发任何日志，**LLM 选了哪个 skill 的原始 tool-call 不落日志**。
  现靠「`initialized` + 无降级 warning + 执行是 skill 短链」三信号间接判读；若需看原始决策，
  临时在 `planner.solve()` 加一行日志（不进入最终提交）。

### B.2 LLM 端点最小预检

```bash
curl -s http://<base_url>/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"用 pickup 工具抓取 long 料片"}],
       "tools":[{"type":"function","function":{"name":"pickup","description":"抓取料片",
                 "parameters":{"type":"object","properties":{"spec":{"type":"string"}}}}}],
       "tool_choice":"auto"}'
```

- [ ] 返回里 `choices[0].message.tool_calls` 非空、`name` 为 `pickup` —— 证明模型支持 tool-call 且协议对上。
- [ ] 若 `tool_calls` 恒空 → 模型不支持 function-calling，换 1.2 里另一个模型。
