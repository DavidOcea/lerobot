"""LLM tool boundary (B1): declare the 5 skills as tool specs (pure data).

``to_tool_spec()`` turns the Phase 1 skill surface (``skills.py``) into the
LLM-facing contract: name / description / visible parameters / return schema /
failure modes. It declares *what the LLM may call* — it does NOT execute
anything. Phase 3 (``llm_loop``) consumes these specs and runs ``skills.py``
generators through ``parse_task_dict`` / ``_execute_single_task``.

Provider-neutral: no OpenAI/Anthropic-specific payload here; Phase 3 adapts a
``ToolSpec`` to the target LLM's function-calling format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from lerobot.tasks.failure_mode import FailureMode
from lerobot.tasks.skills import SPEC_REGISTRY, ObjectSpec


@dataclass(frozen=True)
class ToolSpec:
    """One LLM-facing tool.

    ``parameters`` / ``returns`` are JSON-schema-ish dicts (plain data, not tied
    to a specific LLM API). ``failure_modes`` names the possible failure modes
    (A2 enum values) this tool may report.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]
    failure_modes: tuple[str, ...]


# Three-piece return schema shared by every skill tool (B1).
_RETURNS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "failed"]},
        "failure_mode": {
            "type": ["string", "null"],
            "enum": [m.value for m in FailureMode],
        },
        "evidence": {
            "type": "object",
            "description": "detector 输出 + spec + 发现站号 + confidence + 时间戳",
        },
    },
}


def _spec_param(spec_keys: list[str]) -> dict[str, Any]:
    return {
        "type": "string",
        "description": "对象 spec key（= classify label 前缀）",
        "enum": spec_keys,
    }


def _retreat_param(default_cm: float) -> dict[str, Any]:
    return {
        "type": "number",
        "description": "抓/放后退站清场距离（cm）",
        "default": default_cm,
    }


def to_tool_spec(registry: Mapping[str, ObjectSpec] = SPEC_REGISTRY) -> dict[str, ToolSpec]:
    """Emit the 5 B1 tool specs, keyed by tool name."""
    spec_keys = sorted(registry)
    # Pickup retreat is sourced from the spec (no drift). Place retreat is a
    # literal because ObjectSpec has no place-retreat field yet (A4 retreat.place
    # = 30); the spec migration in Phase 3-4 will carry it.
    pickup_retreat = registry.get("long", SPEC_REGISTRY["long"]).retreat_cm
    place_retreat = 30.0

    return {
        "navigate_to": ToolSpec(
            name="navigate_to",
            description=(
                "长距离 AGV 导航到宏观区域 A 或 B。何时用：需要移动到另一个工作站时。"
                "可能返回 not_arrived / navigation_stuck / timeout（到 B 还可能有 dropped）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dest": {
                        "type": "string",
                        "enum": ["A", "B"],
                        "description": "目标区域",
                    },
                },
                "required": ["dest"],
            },
            returns=_RETURNS,
            failure_modes=(
                FailureMode.NOT_ARRIVED.value,
                FailureMode.NAVIGATION_STUCK.value,
                FailureMode.TIMEOUT.value,
                FailureMode.DROPPED.value,
            ),
        ),
        "pickup": ToolSpec(
            name="pickup",
            description=(
                "抓取指定 spec 的料片（内部先 classify 定位站点再抓取）。"
                "可能返回 empty_grasp / grasp_slip / dropped。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "spec": _spec_param(spec_keys),
                    "retreat_cm": _retreat_param(pickup_retreat),
                },
                "required": ["spec"],
            },
            returns=_RETURNS,
            failure_modes=(
                FailureMode.EMPTY_GRASP.value,
                FailureMode.GRASP_SLIP.value,
                FailureMode.DROPPED.value,
            ),
        ),
        "place": ToolSpec(
            name="place",
            description=(
                "把夹持的料片放到目标区。可能返回 misplace / topple / not_released。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "spec": _spec_param(spec_keys),
                    "retreat_cm": _retreat_param(place_retreat),
                },
                "required": ["spec"],
            },
            returns=_RETURNS,
            failure_modes=(
                FailureMode.MISPLACE.value,
                FailureMode.TOPPLE.value,
                FailureMode.NOT_RELEASED.value,
            ),
        ),
        "speak": ToolSpec(
            name="speak",
            description="TTS 语音播报（提示用户 / 请求协助）。",
            parameters={
                "type": "object",
                "properties": {
                    "msg": {"type": "string", "description": "要播报的文本"},
                },
                "required": ["msg"],
            },
            returns=_RETURNS,
            failure_modes=(),
        ),
        "home": ToolSpec(
            name="home",
            description="机械臂回 home + AGV 退站 + 返回 A（收尾 / 复位）。",
            parameters={"type": "object", "properties": {}},
            returns=_RETURNS,
            failure_modes=(),
        ),
    }
