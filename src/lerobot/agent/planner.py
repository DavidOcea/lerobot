"""Planner abstraction + factory (Phase 3 narrow seam).

``planner`` is a static either-or decision point in ``run()``: the deterministic
flash path (``planner == "flash"``, default) replays the YAML verbatim; the LLM
path (``planner == "llm"``) delegates "which skill next / how to recover" to a
``Planner``. Execution still flows through the existing deterministic engine in
both cases.

This module is the **narrow seam** that makes a future ``pydantic_ai`` upgrade
drop-in: add a ``PlannerType`` member + one ``Planner`` subclass implementing
``solve(messages, tools) -> PlannerDecision``, and register it in
``build_planner``. Nothing else in the loop changes.

Imports only ``agent.config`` / ``agent.llm_client`` / ``agent.llm_protocol`` —
never the orchestrator (no import cycle).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lerobot.agent.llm_client import LLMResponseParseError, OpenAIClient
from lerobot.agent.llm_protocol import ToolCall


class PlannerType(str, Enum):
    """Registered planner backends. ``API`` = OpenAI-compatible chat endpoint."""

    API = "api"
    FLASH = "flash"


@dataclass(frozen=True)
class PlannerDecision:
    """One planner output: a tool call to execute, or a stop/surrender signal.

    ``finish_reason`` mirrors the LLM API (``"tool_calls"`` / ``"stop"`` /
    ``"length"``) and additionally ``"flash"`` when the flash planner is asked
    to decide (i.e. the loop should degrade to deterministic replay).
    """

    tool_call: ToolCall | None
    finish_reason: str
    raw: dict[str, Any] | None = None


class Planner(ABC):
    """Planner contract: ``solve(messages, tools) -> PlannerDecision``."""

    @property
    @abstractmethod
    def planner_type(self) -> PlannerType:
        ...

    @abstractmethod
    def solve(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> PlannerDecision:
        ...


class FlashPlanner(Planner):
    """Deterministic planner: always surrenders back to flash replay."""

    @property
    def planner_type(self) -> PlannerType:
        return PlannerType.FLASH

    def solve(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> PlannerDecision:
        return PlannerDecision(tool_call=None, finish_reason="flash")


class OpenAIChatPlanner(Planner):
    """OpenAI-compatible chat planner over ``OpenAIClient``.

    Maps ``choices[0].message.tool_calls`` / ``choices[0].finish_reason`` to a
    ``PlannerDecision``. A malformed response raises ``LLMResponseParseError``
    (an ``LLMClientError``) so the loop degrades to flash.
    """

    def __init__(self, llm_config) -> None:
        self._client = OpenAIClient(llm_config)

    @property
    def planner_type(self) -> PlannerType:
        return PlannerType.API

    def solve(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> PlannerDecision:
        data = self._client.chat_completion(messages, tools)
        return self._to_decision(data)

    @staticmethod
    def _to_decision(data: dict[str, Any]) -> PlannerDecision:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseParseError(f"no choices in response: {str(data)[:200]}") from e

        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or ""
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return PlannerDecision(tool_call=None, finish_reason=finish_reason, raw=data)

        tc0 = tool_calls[0]
        fn = tc0.get("function") or {}
        arguments = fn.get("arguments") or "{}"
        try:
            parsed_args = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}

        return PlannerDecision(
            tool_call=ToolCall(
                id=tc0.get("id") or "",
                name=fn.get("name") or "",
                arguments=parsed_args,
            ),
            finish_reason=finish_reason,
            raw=data,
        )


def build_planner(planner_type: str, llm_config=None) -> Planner:
    """Factory: string → ``Planner``. Raises ``ValueError`` on unknown type.

    Called lazily from ``_run_llm_loop``; never imported by the flash path.
    """
    pt = PlannerType(planner_type)
    if pt is PlannerType.FLASH:
        return FlashPlanner()
    if pt is PlannerType.API:
        if llm_config is None:
            raise ValueError("LLM planner requires llm_config")
        return OpenAIChatPlanner(llm_config)
    raise ValueError(f"Unsupported planner type: {planner_type}")
