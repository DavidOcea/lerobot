"""Provider-neutral LLM message/tool protocol (Phase 3, pure functions).

This module owns the *shape* of what crosses the LLM boundary — tool
serialization, message construction, and the structured observation string the
LLM reads back after each skill executes. It is provider-neutral: nothing here
knows about OpenAI vs vLLM vs pydantic_ai; ``llm_loop`` composes these pure
functions, and a future pydantic_ai upgrade only swaps the transport
(``llm_client``) + planner (``planner``), not this layer.

Three narrow seams from the Phase 3 plan live here:

* ``ToolSpec`` (from ``tool_spec``) is the **sole** tool-declaration source;
  ``serialize_tools`` is the only place a ``ToolSpec`` becomes an OpenAI
  ``tools`` array.
* ``build_messages`` / ``build_assistant_tool_call_message`` /
  ``build_tool_message`` isolate message construction.
* ``skill_observation`` serializes a ``TaskResult`` into the structured text
  the LLM reads. Phase 4 (``VisualSuccessDetector``) swaps only the **input
  source** of this observation (detector conclusions feed ``evidence``), not
  this shape.

Pure data / pure functions only: no orchestrator import, no execution logic.
Default-unreachable — the flash path never imports this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from lerobot.tasks.failure_mode import FAILURE_RECOVERY_TABLE, FailureMode
from lerobot.tasks.tool_spec import ToolSpec


# ---------------------------------------------------------------------------
# Tool-call shape (owned here so planner + loop share one definition)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A single LLM tool-call.

    Provider-neutral: ``planner`` maps an LLM API response (OpenAI
    ``tool_calls`` entry) into this, and ``llm_loop`` dispatches it.
    """

    id: str
    name: str
    arguments: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool serialization (B1 contract → OpenAI function-calling format)
# ---------------------------------------------------------------------------


def _augmented_description(spec: ToolSpec) -> str:
    """Fold ``returns`` and ``failure_modes`` into the tool description text.

    OpenAI function-calling has no ``returns``/``failure_modes`` fields, so the
    contract that the plan calls "the return schema" and "possible failure
    modes" is inlined into the description the model actually sees.
    """
    parts = [spec.description]
    if spec.returns:
        parts.append(f"Returns (JSON): {json.dumps(spec.returns, ensure_ascii=False)}")
    if spec.failure_modes:
        parts.append(
            "Possible failure_modes: " + ", ".join(spec.failure_modes)
            + ". On failure, read failure_mode + llm_hint from the tool result "
            + "to choose the recovery action."
        )
    return "\n".join(parts)


def serialize_tools(tools: Mapping[str, ToolSpec]) -> list[dict[str, Any]]:
    """Emit the OpenAI ``tools`` array from a mapping of ``ToolSpec``.

    ``spec.parameters`` is already JSON-Schema (built in ``tool_spec.py``), so
    it is passed through verbatim. Only the description is augmented here.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": _augmented_description(spec),
                "parameters": spec.parameters,
            },
        }
        for spec in tools.values()
    ]


# ---------------------------------------------------------------------------
# Message construction (isolated pure functions)
# ---------------------------------------------------------------------------


def build_messages(
    system_prompt: str, history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prepend the system prompt to the rolling conversation history."""
    return [{"role": "system", "content": system_prompt}] + history


def build_assistant_tool_call_message(tc: ToolCall) -> dict[str, Any]:
    """Assistant turn that issued one tool call (OpenAI message shape)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
        ],
    }


def build_tool_message(tc: ToolCall, content: str) -> dict[str, Any]:
    """Tool result turn that answers a tool call (OpenAI message shape)."""
    return {"role": "tool", "tool_call_id": tc.id, "content": content}


# ---------------------------------------------------------------------------
# Observation bridge (A1/B2): TaskResult → structured text for the LLM
# ---------------------------------------------------------------------------


def result_failure_mode(last: Any) -> FailureMode | None:
    """Recover the structured failure mode from a ``TaskResult``.

    Two signal sources:

    * ``error_message`` — the free-form failure string, mapped via
      ``map_failure_to_mode``.
    * ``next_task`` — when ``_execute_single_task``'s ``next_tasks["failed"]``
      recovery path fires, it rewrites a FAILED result to COMPLETED + a
      recovery-node name (e.g. ``voice_pickup_failed``), *dropping* the error
      message. The recovery-node name is then the only failure evidence.
    """
    fm = map_failure_to_mode(getattr(last, "error_message", None))
    if fm is not None:
        return fm
    next_task = getattr(last, "next_task", "") or ""
    if next_task == "voice_pickup_failed":
        return FailureMode.EMPTY_GRASP
    if next_task == "timeout_return_home":
        return FailureMode.TIMEOUT
    return None


def skill_observation(tc: ToolCall, chain_results: list[Any]) -> str:
    """Serialize the last ``TaskResult`` of a skill chain into LLM-readable JSON.

    The LLM receives the *structured conclusion*, never raw images. The
    ``task_trace`` (detector output + spec + discovered station + confidence +
    timestamp) is passed through; Phase 4 swaps its **source**, not this shape.
    """
    if not chain_results:
        hint = FAILURE_RECOVERY_TABLE.get(FailureMode.TIMEOUT)
        return json.dumps(
            {
                "tool": tc.name,
                "status": "failed",
                "success": False,
                "error_message": "no results produced",
                "failure_mode": FailureMode.TIMEOUT.value,
                "llm_hint": hint.llm_hint if hint else "",
                "flash_action": hint.flash_action if hint else "",
                "task_trace": {},
            },
            ensure_ascii=False,
        )

    last = chain_results[-1]
    fm = result_failure_mode(last)
    hint = FAILURE_RECOVERY_TABLE.get(fm) if fm is not None else None
    final_obs = getattr(last, "final_observation", None) or {}
    return json.dumps(
        {
            "tool": tc.name,
            "status": getattr(getattr(last, "status", None), "value", None)
            or str(getattr(last, "status", "")),
            "success": bool(getattr(last, "success", False)),
            "error_message": getattr(last, "error_message", "") or "",
            "next_task": getattr(last, "next_task", "") or "",
            "failure_mode": fm.value if fm is not None else None,
            "llm_hint": hint.llm_hint if hint is not None else "",
            "flash_action": hint.flash_action if hint is not None else "",
            "task_trace": final_obs.get("task_trace", {}),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Failure-mode mapping (coarse free-form string → structured FailureMode)
# ---------------------------------------------------------------------------


def map_failure_to_mode(error_message: str | None) -> FailureMode | None:
    """Map a free-form executor error string to a structured ``FailureMode``.

    Aligns with the coarse keys in ``orchestrator._FAILURE_RECOVERY_RULES``
    (``"Tag lost"`` / ``"Classification failed after retries"`` / ...) and the
    executor error signatures. Ordered most-specific first: ``AGV navigation
    timeout`` must resolve to ``NAVIGATION_STUCK`` before the generic
    ``timeout`` heuristic fires.
    """
    if not error_message:
        return None
    msg = error_message
    heuristics: list[tuple[str, FailureMode]] = [
        ("AGV navigation timeout", FailureMode.NAVIGATION_STUCK),
        ("AGV navigation", FailureMode.NAVIGATION_STUCK),
        ("navigation_stuck", FailureMode.NAVIGATION_STUCK),
        ("Tag lost", FailureMode.TAG_LOST),
        ("tag_lost", FailureMode.TAG_LOST),
        ("Classification failed after retries", FailureMode.CLASSIFY_EXHAUSTED),
        ("Classification failed", FailureMode.CLASSIFY_EXHAUSTED),
        ("classify_exhausted", FailureMode.CLASSIFY_EXHAUSTED),
        ("No workpiece detected after", FailureMode.NO_DETECT),
        ("No workpiece detected", FailureMode.NO_DETECT),
        ("no_detect", FailureMode.NO_DETECT),
        ("not arrived", FailureMode.NOT_ARRIVED),
        ("Not arrived", FailureMode.NOT_ARRIVED),
        ("not_arrived", FailureMode.NOT_ARRIVED),
        ("empty grasp", FailureMode.EMPTY_GRASP),
        ("empty_grasp", FailureMode.EMPTY_GRASP),
        ("grasp slip", FailureMode.GRASP_SLIP),
        ("grasp_slip", FailureMode.GRASP_SLIP),
        ("dropped", FailureMode.DROPPED),
        ("misplace", FailureMode.MISPLACE),
        ("topple", FailureMode.TOPPLE),
        ("not released", FailureMode.NOT_RELEASED),
        ("not_released", FailureMode.NOT_RELEASED),
        ("timed out", FailureMode.TIMEOUT),
        ("Timed out", FailureMode.TIMEOUT),
        ("timeout", FailureMode.TIMEOUT),
        ("Timeout", FailureMode.TIMEOUT),
    ]
    for key, mode in heuristics:
        if key in msg:
            return mode
    return None


# ---------------------------------------------------------------------------
# Default system prompt (中文；enumerate the 5 tools + recovery contract)
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = """\
你是 Supre 机器人的任务规划器。根据当前任务状态，决定下一步调用哪个工具（skill）。

可用工具：
- navigate_to(dest): 长距离 AGV 导航到区域 A 或 B。dest ∈ {A, B}。
- pickup(spec): 抓取指定 spec 的料片（内部自动 classify 定位站点）。spec 目前只有 "long"。
- place(spec): 把夹持的料片放到目标区。
- speak(msg): TTS 语音播报（提示用户 / 请求协助）。
- home(): 机械臂回 home + AGV 退站 + 返回 A（收尾 / 复位）。

规则：
1. 抓取任务的标准流程是 navigate_to(A) → pickup → navigate_to(B) → place → home。
2. 每次工具调用后你会收到一个 JSON 结果。如果 status 是 failed 或 failure_mode 非空，
   读取 failure_mode 和 llm_hint，据此决定恢复动作（重试、回 A 重抓、换 spec、或 home 复位）。
3. 不要在 pickup/place 里指定站号参数——站号由内部 classify 自动发现。
4. 完成后直接停止（不再调用工具），不要空转。
"""
