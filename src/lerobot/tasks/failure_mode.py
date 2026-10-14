"""Structured failure-mode enum + recovery table (A2 contract).

Flash mode classifies failures by ``error_message.split(":")[0]`` — a coarse
free-form string. The LLM planner (Phase 3+) needs a *structured* enum so every
skill tool can declare the failure modes it may return, and the planner can map
each mode to a recovery direction.

Pure data: no execution logic, no orchestrator imports. Phase 3 wires this into
``_classify_failure_mode``; until then nothing reads it (default-unreachable).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    """Structured failure modes (A2 enum). Values are the wire/JSON strings."""

    TAG_LOST = "tag_lost"
    CLASSIFY_EXHAUSTED = "classify_exhausted"
    NO_DETECT = "no_detect"
    NOT_ARRIVED = "not_arrived"  # B1 navigate_to: destination not reached
    NAVIGATION_STUCK = "navigation_stuck"  # A2: navigation did not converge
    EMPTY_GRASP = "empty_grasp"
    GRASP_SLIP = "grasp_slip"
    DROPPED = "dropped"
    MISPLACE = "misplace"
    TOPPLE = "topple"
    NOT_RELEASED = "not_released"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RecoveryHint:
    """Per-mode recovery contract: deterministic flash action + LLM direction."""

    flash_action: str  # deterministic action (existing node/rule name)
    llm_hint: str  # directional hint; the LLM picks the concrete action


# A2 table. ``flash_action`` aligns with the existing ``_FAILURE_RECOVERY_RULES``
# (orchestrator) and the YAML ``next_tasks.failed`` branches; ``llm_hint`` gives a
# direction without prescribing a concrete action.
FAILURE_RECOVERY_TABLE: dict[FailureMode, RecoveryHint] = {
    FailureMode.TAG_LOST: RecoveryHint("skip_retries", "语音重试，重新 align"),
    FailureMode.CLASSIFY_EXHAUSTED: RecoveryHint("skip_retries", "语音调整料片，重新 classify"),
    FailureMode.NO_DETECT: RecoveryHint("skip_retries", "语音要求放料片，重新检测"),
    FailureMode.NAVIGATION_STUCK: RecoveryHint("skip_retries", "回退已知位，重新导航"),
    FailureMode.NOT_ARRIVED: RecoveryHint("skip_retries", "回退已知位，重新导航"),
    FailureMode.EMPTY_GRASP: RecoveryHint("voice_pickup_failed", "回 A 重抓或换 spec"),
    FailureMode.GRASP_SLIP: RecoveryHint("voice_pickup_failed", "回 A 重抓或换 spec"),
    FailureMode.DROPPED: RecoveryHint("agv_return_to_A_for_repick", "回 A 重抓"),
    # verify_place modes: only produced once Phase 4 (VisualSuccessDetector) lands.
    FailureMode.MISPLACE: RecoveryHint("", "局部重放 place 或回 A 重抓"),
    FailureMode.TOPPLE: RecoveryHint("", "回 A 重抓（倒了不能继续）"),
    FailureMode.NOT_RELEASED: RecoveryHint("", "局部重放 release（夹爪再松一次）"),
    FailureMode.TIMEOUT: RecoveryHint("timeout_return_home", "回 home 重启循环"),
}
