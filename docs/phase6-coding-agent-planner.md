# Phase 6 — Coding-Agent Planner（Claude Code / Codex 当 planner）

> 状态：**开发计划（未实施）**。留待"任务变成开放式/需要长程推理与自我纠错"时再开发。
> 前置：Phase 3（`llm_loop` + `OpenAIChatPlanner`，function-calling loop）已完成；Phase 4（视觉判据）已完成。
> 本文不替代运行时的验收手册，是**开发蓝图**：给未来开发者"照着就能写"的精确落点。

---

## 0. 一句话结论

RPent 的 "Claude Code / Codex 当 planner" **不是**让编码 agent 写代码控机器人，而是**把"推理→调工具→看结果→再推理"这个 ReAct 循环，从我们代码里移交到 agent SDK 手里**；机器人动作永远走我们的确定性 skill 引擎（`_execute_skill_chain`），agent 只通过工具调它，最后用一个 **`finish` 工具显式宣告成功/失败/卡住**。

我们要做的 = 把现有 5 skill 包成一个 toolkit + 加 `finish` 工具 + 写两个 SDK 后端适配器 + 在 `run_llm_loop` 入口加一条分支。确定性引擎**零改动**。

---

## 0.1 认知前提：LLM 涌现 = 决策空间 × 思考机制

本文档"什么时候该做 Phase 6"的判断，建立在一个心智模型上——先说清它：

**LLM 相对 flash 的创造性/涌现，是两个条件相乘的产物，缺一是零：**

| 条件 | 含义 | 由谁负责 |
|---|---|---|
| **决策空间（原材料）** | LLM 有多少"可自由组合 / 权衡 / 创造"的素材 | skill 层：spec 注册表 / named_positions / 新 verb / 多模态观测 |
| **思考机制（催化剂）** | prompt 引导 / thinking / 多轮纠错 / 结果反馈 | planner 层：本 Phase 要上的 Claude Code / Codex |

**关键结论**：
- Claude Code / Codex 只提供**催化剂**（思考机制），不提供**原材料**（决策空间）。它只能放大已有决策空间的探索，不能凭空造空间。
- 在决策空间还是 5 个 skill + `dest∈{A,B}` + 固定 spec 时上 Claude Code，等于给只能走直线的小巷装上超跑引擎——还是只能走直线。**ADC2 上 LLM≈flash 是任务特性决定的**，不是 prompt 没写好、也不是没上 Claude Code。
- 正确顺序：**先扩决策空间（造舞台）→ 再上催化剂（开引擎）**。催化剂里的轻量版（把 Claude Code 的"先分析再行动 / 失败反思 / 防循环"思考技巧抄进 `DEFAULT_SYSTEM_PROMPT`）零重依赖、可先行；重装版（agent SDK 的 thinking 机制）等决策空间真正开放后再上。

---

## 1. 核心架构结论（读 RPent 源码坐实）

### 1.1 根本差异：loop 所有权

| | 我们（lerobot） | RPent |
|---|---|---|
| ReAct 循环在哪 | 我们的 `llm_loop.py:91` 的 `for _step in range(max_tool_calls)` | agent SDK 内部（`claude_agent_sdk` / `openai_codex` / `pydantic_ai.Agent`） |
| `solve` 语义 | 单发：`solve(messages, tools) -> PlannerDecision`（`planner.py:51`） | 会话级：`solve(system_prompt, user_message, toolkit, max_turns) -> PlannerResult`（RPent `base.py:136`） |
| 终止 | `finish_reason == "stop"`（隐式） | 显式 **`finish(status, summary)` 工具** |

### 1.2 三个 backend（RPent `base.py:241-304`）

| backend | 引擎 | 工具暴露方式 |
|---|---|---|
| `api` | `pydantic_ai.Agent` | `FunctionToolset`（= 我们现在的 function-calling，等价物） |
| `claude_code` | **`claude_agent_sdk`**（Claude Agent SDK，Claude Code CLI 底下的引擎） | 进程内 MCP：`create_sdk_mcp_server` |
| `codex` | **`openai_codex`**（OpenAI Codex Python SDK） | 进程内 **HTTP** MCP：`HttpMcpServer` |

两者都是 **SDK 进程内调用**，不是 subprocess 调 CLI。

### 1.3 `finish` 工具 = 终止契约（我们缺失的关键块）

RPent `tools/common.py:87`：

```python
def finish(self, status: str, summary: str) -> ToolResult:
    # status: 'success' / 'failure' / 'stuck'
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})
```

agent 觉得干完了就调 `finish(...)`，planner 捕获 `_finish=True` → 结束会话 → 返回 `finish_result`。

### 1.4 "coding" 能力在，但被约束

- `claude_code.py:82` 默认 `allowed_tools="Bash Read Write Glob Grep"`；`codex.py:121` `sandbox=full_access`。
- 但 `claude_code.py:377` `setting_sources=[]`（忽略用户/项目 `.claude` 配置）；`codex.py:878-886` `project_doc_max_bytes=0` + 禁用 `.agents/skills`（剥掉软件工程上下文）。

→ 编码 agent 只被允许做三件事：**通过 MCP 调机器人 skill、调 `finish` 收尾、沙箱读写文件做记忆**。机器人动作始终走确定性引擎。

### 1.5 一个意外收益：修复可观测性缺口

验收手册附录 B.1 记录 `OpenAIChatPlanner` 不发日志、"LLM 选了哪个 skill 的原始 tool-call 不落日志"。agent SDK 后端天然产出**完整 transcript**（thinking + tool call + tool result），顺手补齐这个缺口。

---

## 2. 与现有代码的精确映射

| 我们 | RPent | 结论 |
|---|---|---|
| `to_tool_spec()`（`tool_spec.py:71`，5 skill） | `Toolkit.list_tools()`（robot skills + `finish` + 文件工具） | 同构，**加 `finish` 即可** |
| `_dispatch(tc, orchestrator)`（`llm_loop.py:146`） | `toolkit.execute_tool(name, args)` 内部 | 同构，**直接复用** |
| `_execute_skill_chain(...)`（`llm_loop.py:228`） | 同上（确定性引擎） | 同构，**零改动** |
| `skill_observation(tc, results)`（`llm_protocol.py:160`） | `ToolResult.to_text()` | 同构 |
| `max_tool_calls`（`config.py` LLMConfig） | `max_turns` + `max_budget_usd` + `timeout_s` | 边界强化（+美元预算） |
| flash fallback（`llm_loop.py:332`） | `FlashPlanner` | 同构 |
| ❌ 无 `finish` | `finish(status, summary)` | **新增** |
| 我们跑 while 循环 | agent SDK 跑循环 | **移交** |
| stdlib `urllib` ~30 行无依赖（`llm_client.py`） | `claude_agent_sdk` / `openai_codex` / `pydantic_ai` | **重依赖（可选 extra）** |

**核心洞察**：`_dispatch` + `_execute_skill_chain` + `skill_observation` 这三个已经存在的函数，**就是** RPent 的 toolkit 执行体。toolkit 只是一个 ~30 行的薄壳把它们包起来。

---

## 3. 设计决策（需拍板，默认值已给出）

### D1. 契约：不碰现有单发 `Planner`，新增"会话级 planner"抽象

- 现有 `Planner` ABC（`planner.py:51` `solve(messages, tools)`）保持不动，`OpenAIChatPlanner`/`FlashPlanner` 及所有既有 verify 全绿。
- 新增独立 `AgentPlanner` ABC：`solve(system_prompt, user_message, toolkit, max_turns) -> AgentPlannerResult`。
- 独立工厂 `build_agent_planner(...)`，不污染 `build_planner`。

**为什么**：这是 RPent 的设计（`api`/`claude_code`/`codex`/`flash` 都是同一 `Planner` ABC 的会话级实现）；但我们的 `Planner` 是单发级，混进同一 ABC 会破坏无感。平级新增最安全。

### D2. `finish` 状态枚举：采用 RPent 的 `success/failure/stuck`

我们的 `_RETURNS.status`（`tool_spec.py:42`）是 `["success", "failed"]`；RPent finish 用 `["success", "failure", "stuck"]`。**`finish` 是独立工具**，直接采用 RPent 三值（`stuck` 是"卡住放弃"信号，我们的 `failed` 没有这个语义）。5 个 skill 工具的 `_RETURNS` 不动。

### D3. 依赖：可选 extra，不破 stdlib 无依赖原则

- `claude_agent_sdk` / `openai_codex` 都是重依赖，只加进 pyproject 的 optional extra（如 `[agent-planners]`），**不**进主依赖。
- 后端模块懒 import：SDK 未安装时 `build_agent_planner` 抛 `ImportError` → 走 flash fallback（与现在 `build_planner` 抛异常→降级同构）。
- 现有 `api`（function-calling）路径继续用 stdlib `urllib`，零影响。

### D4. 安全：默认**不**授予 Bash/shell

RPent `claude_code` 默认 `allowed_tools="Bash Read Write Glob Grep"`。但我们真机是物理安全关键系统，默认应更窄：

- 默认 `allowed_tools` **不含 `Bash`**（或整个 `allowed_tools` 置空），只给：5 个 skill + `finish` + 沙箱目录的 `Read/Write`（做记忆，Phase 5 对齐）。
- `Bash`/`Glob`/`Grep` 作为 config 显式 opt-in（`allowed_tools` 字段），且运行时落到受限 `cwd`（沙箱目录，非 repo root / 非真机配置目录）。

### D5. 终止映射：`finish_result` → `ExecutionSummary`

- `status == "success"` → 直接 `_summary(orchestrator, results)`。
- `status == "failure"` → 返回失败 summary（不降级，agent 主动宣告失败）。
- `status == "stuck"` → 走 `_flash_fallback`（与现在"budget exhausted → flash"同语义：agent 放弃，确定性重放兜底）。
- `error is not None`（timeout/budget/SDK 异常）→ `_flash_fallback`。

### D6. 先做 `claude_code`，`codex` 对称跟随

两者共享同一个 `SkillToolkit`。`claude_code` 用进程内 MCP，`codex` 用 HTTP MCP（仅暴露方式不同）。先落地 `claude_code`（用户当前关注点），`codex` 复用 toolkit + finish 层，只差一个后端适配器。

---

## 4. 改动文件清单 + 精确落点

### 4.1 新增 `src/lerobot/agent/agent_toolkit.py`（共享层，核心复用点）

```python
@dataclass(frozen=True)
class ToolDeclaration:        # ≈ RPent tools/base.py 的 Tool 声明
    name: str
    description: str
    input_schema: dict[str, Any]

@dataclass
class ToolResult:             # ≈ RPent ToolResult（data + images；我们暂不需要 images）
    data: dict[str, Any]
    def to_text(self) -> str: ...   # json.dumps，60KB 截断对齐 RPent

class SkillToolkit:
    def __init__(self, orchestrator): self.orchestrator = orchestrator

    def list_tools(self) -> list[ToolDeclaration]:
        # 5 skill 来自 to_tool_spec()（tool_spec.py:71），parameters 即 input_schema
        # + 一个 finish 声明：{name:"finish", description, input_schema:{status:enum, summary:str}}
        ...

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        if name == "finish":
            return ToolResult({"_finish": True, "status": arguments["status"],
                               "summary": arguments["summary"]})
        tc = ToolCall(id="", name=name, arguments=arguments)   # llm_protocol.py:41
        task_dicts = _dispatch(tc, self.orchestrator)          # llm_loop.py:146 复用！
        results = _execute_skill_chain(                        # llm_loop.py:228 复用！
            self.orchestrator, task_dicts, skill_name=name)
        # skill_observation（llm_protocol.py:160）产出 JSON 文本 → ToolResult.data
        return ToolResult(json.loads(skill_observation(tc, results)))
```

**要点**：`execute_tool` 对 `finish` 之外的所有工具，逻辑 = 现有 `run_llm_loop` 里 `_dispatch` → `_execute_skill_chain` → `skill_observation` 三连（`llm_loop.py:108/113/115`）。这就是"toolkit 只是薄壳"的由来。

### 4.2 新增 `src/lerobot/agent/agent_planner.py`（会话级 planner + 后端）

```python
class AgentPlannerResult:    # ≈ RPent PlannerResult
    finish_result: dict | None   # {"status":"success/failure/stuck","summary":"..."}
    messages: list[dict]         # transcript（顺手修可观测性缺口）
    stats: dict
    error: str | None

class AgentPlanner(ABC):
    def solve(self, *, system_prompt, user_message, toolkit, max_turns) -> AgentPlannerResult: ...

class ClaudeCodeAgentPlanner(AgentPlanner):
    # import claude_agent_sdk（懒 import）
    # ClaudeAgentOptions(cwd=<sandbox>, model, max_turns, max_budget_usd,
    #                     allowed_tools=<D4 默认窄>, mcp_servers={"rpent": create_sdk_mcp_server(toolkit)},
    #                     setting_sources=[])            ← 忽略用户/项目 .claude
    # sdk.query(prompt, options) 流式消费；finish 工具结果 → finish_result
    # 对齐 RPent claude_code.py:70-283

class CodexAgentPlanner(AgentPlanner):
    # import openai_codex（懒 import）
    # 进程内 HttpMcpServer(toolkit)（对齐 RPent codex.py:166）
    # Codex(config).thread_start(approval_mode=deny_all, sandbox=...) .turn(prompt, effort)
    # 对齐 RPent codex.py:90-258

def build_agent_planner(planner_type, *, output_dir, model, max_turns,
                        timeout_s, max_budget_usd, allowed_tools, sandbox_dir,
                        reasoning_effort, dashboard_events=None) -> AgentPlanner: ...
```

### 4.3 改 `src/lerobot/agent/config.py`

- 新增 `AgentPlannerConfig`（顶层，与 `LLMConfig` 平级）：
  ```python
  @dataclass
  class AgentPlannerConfig:
      model: str = "sonnet"          # claude_code 默认
      max_turns: int = 20
      timeout_s: float = 600.0
      max_budget_usd: float = 10.0   # 仅 claude_code
      allowed_tools: str = ""        # 默认空=仅 skill+finish+沙箱读写；Bash 需显式
      sandbox_dir: str = ""          # 沙箱 cwd；空→临时目录
      reasoning_effort: str = "none"
  ```
- 加字段 `agent_planner_config: AgentPlannerConfig = field(default_factory=AgentPlannerConfig)`（与 `llm_config` 并排，参照现有 `llm_config` 字段的插入方式）。

### 4.4 改 `src/lerobot/agent/llm_loop.py`（唯一执行路径触碰）

- 新增 `run_agent_loop(orchestrator)`（或 `run_llm_loop` 顶部加分支），gate 在 `cfg.planner in ("claude_code", "codex")`：
  ```python
  toolkit = SkillToolkit(orchestrator)
  try:
      ap = build_agent_planner(cfg.planner, ...)
      pr = ap.solve(system_prompt=system_prompt, user_message=<任务描述>,
                    toolkit=toolkit, max_turns=cfg.agent_planner_config.max_turns)
  except (ImportError, Exception) as e:      # SDK 缺/构造失败 → 降级
      logger.warning("Agent planner construction failed (%s); falling back to flash", e)
      return orchestrator._execute_with_safety()
  # D5 终止映射：success→_summary；failure→失败 summary；stuck/error→_flash_fallback
  ```
- `run_llm_loop` 顶部（`llm_loop.py:71` 附近）加一句分流，或让 orchestrator 的 `planner` 字段直接指向 agent planner。**优先：不改 orchestrator.py，只改 `llm_loop.py`**（保持 orchestrator 零 diff，对齐 Phase 4 的做法）。
- `_dispatch` / `_execute_skill_chain` / `_resolve_station` **不动**。

### 4.5 改 `pyproject.toml`

- 加 optional extra：
  ```toml
  [project.optional-dependencies]
  agent-planners = ["claude-agent-sdk>=X", "openai-codex>=Y"]
  ```

### 4.6 新增 `scripts/verify_agent_planner.py`（离线可测的部分）

镜像 `verify_llm_planner.py`（FakeOrchestrator + SimpleNamespace config，无网络无硬件）。**只验我们拥有的层**（toolkit + finish + 终止映射），SDK 后端不可离线验（需真 creds + 机器人）：

1. `SkillToolkit.list_tools()` 返回 6 个声明（5 skill + finish），`input_schema` 与 `to_tool_spec()` 一致。
2. `execute_tool("finish", {"status":"success","summary":"ok"})` → `data["_finish"] is True`。
3. `execute_tool("pickup", {"spec":"long"})` → 走 `_dispatch`+`_execute_skill_chain`，FakeOrchestrator 记录一次 pickup 执行。
4. `execute_tool("unknown_tool", {})` → 抛 `InvalidToolCall`（`llm_loop.py:54`）。
5. 终止映射：`finish_result={"status":"stuck"}` → 触发 flash fallback；`"success"` → 正常 summary。
6. **无感**：默认 `planner="flash"` 下 `import lerobot.agent.orchestrator` 后 `sys.modules` 无 `agent_toolkit` / `agent_planner` / SDK 模块。

---

## 5. 实施顺序（每 commit 绿 + 默认不可达，复刻 Phase 3/4 节奏）

| step | 内容 | 可验证性 |
|---|---|---|
| 1 | **config-only**：`AgentPlannerConfig` + `agent_planner_config` 字段 | 无人 import，`planner=="llm"/"flash"` bit-identical |
| 2 | **toolkit + finish**：`agent_toolkit.py`（`SkillToolkit` + `ToolResult` + `finish`） | 未引用→默认不可达；`verify_agent_planner` 1-4 节可离线测 |
| 3 | **会话 planner ABC + 终止映射**：`agent_planner.py`（`AgentPlanner`/`AgentPlannerResult`/`build_agent_planner`，SDK 缺→抛错） | 无 SDK 也可 import（后端懒 import） |
| 4 | **loop 入口分支**：`llm_loop.py` 加 `run_agent_loop`，gate `planner in ("claude_code","codex")` | 默认不可达；`verify_agent_planner` 5-6 节 |
| 5 | **claude_code 后端**：`ClaudeCodeAgentPlanner`（`claude_agent_sdk`） | 仅真机 + 真 creds 验（G2 同级命令） |
| 6 | **codex 后端**：`CodexAgentPlanner`（`openai_codex`） | 同上 |
| 7 | **verify 脚本 + 本文档**（`git add -f`，见 [[markdown-gitignored-needs-force-add]]） | 离线全绿 |

step 1-4 是**纯新增 + 离线可测**的核心价值层；5-6 是薄 SDK 适配器，只能真机验。

---

## 6. 验证

1. **无感**：`git diff feat/skill-ification -- src/lerobot/agent/orchestrator.py` 为空（本阶段不碰 orchestrator）；`llm_loop.py` 改动只在 `planner in ("claude_code","codex")` 分支内；Phase 1-4 的 `verify_skill_equivalence` / `verify_tool_spec` / `verify_llm_planner` / `verify_visual_detector` 仍全绿。
2. **import 卫生**：默认配置下 `agent_toolkit` / `agent_planner` / `claude_agent_sdk` / `openai_codex` 均不进入 `sys.modules`（懒 import）。
3. **离线**：`scripts/verify_agent_planner.py` 6 节全绿（FakeOrchestrator，无网络无 SDK）。
4. **真机**（开发后，需要时）：`planner: claude_code` + 真实 `ANTHROPIC_API_KEY`，`--debug` 跑单 skill 短链；判据 = `finish_result status=="success"` + 执行是 skill 短链 + transcript 里可见 tool call（顺便验 D 的 B.1 可观测性缺口已补）。

---

## 7. 依赖清单

| 依赖 | 用途 | 是否主依赖 |
|---|---|---|
| `claude_agent_sdk` | claude_code 后端 | 否（`[agent-planners]` extra） |
| `openai_codex` | codex 后端 | 否（extra） |
| （参考，不引入）`pydantic_ai` + `pydantic_ai_harness` | RPent 的 `api` 后端等价物 | 否——我们已有 stdlib `OpenAIClient` |

---

## 8. 风险 / 待办

- **SDK 版本/API 漂移**：`claude_agent_sdk` / `openai_codex` 的 API（`ClaudeAgentOptions`、`create_sdk_mcp_server`、`thread_start`、`turn`）随版本变。后端适配器需 pin 版本，落地时以当时 SDK 实际签名为准（本计划里的签名来自 RPent 当前源码，是参照不是契约）。
- **网络**：真机只到 github origin（见 [[robot-only-reaches-github-origin]]），Claude/Codex 官方 API 是否可达需真机实测；不可达则需网关（回到 function-calling 或找 OpenAI 兼容网关）。
- **非确定性**：agent SDK 后端天然违背"无感 bit-identical"铁律，只能靠 `setting_sources=[]` + `approval_mode=deny_all` + `max_turns`/`max_budget_usd`/`timeout_s` 三重硬边界约束，不能指望逐字节稳定。
- **`finish` 与 `_RETURNS.status` 枚举不一致**（D2）：`finish` 用 `success/failure/stuck`，5 skill 用 `success/failed`——这是有意为之（语义不同），但需在 `finish` 工具描述里写清，避免 agent 混淆。
- **沙箱 cwd 安全**（D4）：`Bash` 若 opt-in，必须落到沙箱目录 + 只读挂载真机配置，禁止写 `configs/` 或机器人部署目录。
- **`dashboard_events` 依赖**：RPent 的 planner 强依赖 dashboard 事件系统；我们当前无 dashboard，`AgentPlannerResult.messages` 先落日志（`logger.info`），dashboard 留待后续（不要为这个阶段引入 dashboard）。

---

## 9. 参考源码位置

- RPent（开发时重新 clone 核对）：`github.com/RLinf/RPent`
  - `rpent/planner/base.py`（Planner ABC / PlannerResult / build_planner / execute_tool）
  - `rpent/planner/claude_code.py`（`ClaudeCodePlanner`）
  - `rpent/planner/codex.py`（`CodexPlanner` + `build_codex_config`）
  - `rpent/planner/api_loop.py`（`ApiAgentLoop`，function-calling 等价物）
  - `rpent/tools/base.py`（`ToolResult` / `Tool`）
  - `rpent/tools/common.py`（`finish` / `read_text_file` / `write_text_file`）
- 我们（`feat/skill-ification` @ `fffe3ce`）：
  - `src/lerobot/agent/planner.py`（`Planner`/`PlannerDecision`/`build_planner`）
  - `src/lerobot/agent/llm_client.py`（`OpenAIClient`，stdlib 无依赖）
  - `src/lerobot/agent/llm_protocol.py`（`ToolCall`/`serialize_tools`/`skill_observation`/`DEFAULT_SYSTEM_PROMPT`）
  - `src/lerobot/agent/llm_loop.py`（`run_llm_loop`/`_dispatch`/`_execute_skill_chain`/`_flash_fallback`）
  - `src/lerobot/tasks/tool_spec.py`（`ToolSpec`/`to_tool_spec`/`_RETURNS`）

---

## 10. 结论

**可行性：确定。确定性引擎零改动，`_dispatch`+`_execute_skill_chain`+`skill_observation` 三函数就是 toolkit 执行体。**

**体量：独立一个 Phase**（loop 移交 + finish 契约 + 两个 SDK 后端 + 可选依赖），核心价值层（step 1-4）纯新增、离线可测，后端（step 5-6）是薄适配器。

**触发条件：先扩决策空间，再上催化剂**（见 §0.1）。agent SDK 只提供思考机制（催化剂），不提供决策空间（原材料）——在决策空间还是 5-skill 封闭流程时上 Claude Code，只会得到 flash 的昂贵等价物。真正的杠杆是先把决策空间扩大（多对象 spec 注册表 / named_positions / 堆叠力控 verb / RGB-D 多模态），让 LLM 有素材可创造；那之后 agent SDK 的长程推理 / thinking 才有不可替代的价值。当前 ADC2 的 5-skill 强确定流程，function-calling loop（Phase 3）已覆盖，agent SDK 只会带来重依赖 + 非确定性 + 更难无感。
