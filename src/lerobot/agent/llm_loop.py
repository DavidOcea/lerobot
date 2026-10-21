"""LLM planner loop + skill dispatch (Phase 3).

``run_llm_loop(orchestrator)`` is the entry: when ``planner == "llm"``,
``orchestrator.run()`` calls ``_run_llm_loop()``, which imports this module
lazily. The loop lets the LLM pick the next skill and read back a structured
observation after each; every failure path degrades to deterministic flash
replay (see ``_flash_fallback``).

Execution still flows through the existing deterministic engine: skill
generators (``skills.py``) → ``parse_task_dict`` → ``_execute_single_task``.
The LLM only decides *which* skill and *how to recover* — it never drives
joints or the AGV directly.

Imported lazily from the orchestrator (never by the default flash path), so it
may import orchestrator-side modules freely without an import cycle.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from typing import Any

from lerobot.agent.llm_client import LLMClientError
from lerobot.agent.llm_protocol import (
    DEFAULT_SYSTEM_PROMPT,
    ToolCall,
    build_assistant_tool_call_message,
    build_messages,
    build_tool_message,
    result_failure_mode,
    serialize_tools,
    skill_observation,
)
from lerobot.agent.planner import build_planner
from lerobot.tasks.config import parse_task_dict
from lerobot.tasks.failure_mode import FAILURE_RECOVERY_TABLE, RecoveryHint
from lerobot.tasks.skills import (
    SPEC_REGISTRY,
    home_task_dict,
    navigate_to_task_dict,
    pickup_task_dict,
    place_task_dict,
    speak_task_dict,
)
from lerobot.tasks.task_scheduler import ExecutionSummary, TaskResult, TaskStatus
from lerobot.tasks.tool_spec import to_tool_spec

logger = logging.getLogger(__name__)


class InvalidToolCall(Exception):
    """LLM emitted an unknown tool or malformed arguments (→ flash fallback)."""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_llm_loop(orchestrator, planner=None) -> ExecutionSummary:
    """Run the LLM decide → execute → observe loop until stop or budget.

    ``planner`` may be a pre-built ``Planner`` (e.g. the orchestrator's
    ``self.planner``, constructed at init); when ``None`` it is built here.
    Any construction/transport/dispatch failure degrades to flash replay via
    ``_flash_fallback``, so the robot always ends in a deterministic state.
    """
    cfg = orchestrator.config
    llm_cfg = getattr(cfg, "llm_config", None)
    if llm_cfg is None:
        return orchestrator._execute_with_safety()

    system_prompt = llm_cfg.system_prompt or DEFAULT_SYSTEM_PROMPT
    max_tool_calls = llm_cfg.max_tool_calls
    tools = serialize_tools(to_tool_spec())

    if planner is None:
        try:
            planner = build_planner("api", llm_cfg)
        except Exception as e:  # bad base_url/model/key → degrade
            logger.warning("LLM planner construction failed (%s); falling back to flash", e)
            return orchestrator._execute_with_safety()

    history: list[dict[str, Any]] = []
    results: list[TaskResult] = []
    last_hint: RecoveryHint | None = None

    for _step in range(max_tool_calls):
        messages = build_messages(system_prompt, history)
        try:
            decision = planner.solve(messages, tools)
        except LLMClientError as e:
            logger.warning("LLM unreachable (%s); falling back to flash", e)
            return _flash_fallback(orchestrator, results, last_hint)

        if decision.tool_call is None:
            if decision.finish_reason == "stop":
                return _summary(orchestrator, results)
            # finish_reason "flash"/"length"/other → surrender to deterministic replay
            return _flash_fallback(orchestrator, results, last_hint)

        tc = decision.tool_call
        history.append(build_assistant_tool_call_message(tc))
        try:
            task_dicts = _dispatch(tc, orchestrator)
        except InvalidToolCall as e:
            logger.warning("Invalid tool call (%s); falling back to flash", e)
            return _flash_fallback(orchestrator, results, last_hint)

        chain_results = _execute_skill_chain(orchestrator, task_dicts, skill_name=tc.name)
        results.extend(chain_results)
        history.append(build_tool_message(tc, skill_observation(tc, chain_results)))
        last_hint = _last_recovery_hint(chain_results)

    logger.warning("LLM budget exhausted (%d tool calls); falling back to flash", max_tool_calls)
    return _flash_fallback(orchestrator, results, last_hint)


# ---------------------------------------------------------------------------
# Dispatch (tool call → skill generator task_dicts) — the payload-bearing point
# ---------------------------------------------------------------------------


def _require_spec(spec: Any) -> str:
    """Validate a spec key against the registry; raise ``InvalidToolCall``."""
    if not isinstance(spec, str) or spec not in SPEC_REGISTRY:
        raise InvalidToolCall(f"unknown spec {spec!r}; available: {sorted(SPEC_REGISTRY)}")
    return spec


def _sanitize_tts_msg(msg: str) -> str:
    """Strip characters that would break the ``system_command`` executor.

    The executor runs ``task.command.split()`` + ``shell=False`` (no shell), so
    there is **no shell-injection surface** for LLM text. The single-quote
    wrapping in ``speak_task_dict`` survives ``.split()`` literally (``speak.py``
    strips it), so the only real hazard is a single-quote in ``msg`` breaking
    that strip, or control whitespace fragmenting the arg list.
    """
    return msg.replace("'", "").replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _dispatch(tc: ToolCall, orchestrator) -> list[dict[str, Any]]:
    """Map a tool call to skill-generator task_dicts.

    ``navigate_to``/``home``/``place``/``speak`` are direct; ``pickup`` resolves
    its station via a hidden pre-run classify (``_resolve_station``). Returns a
    list of task_dicts (single dicts are normalized to a one-element list).
    """
    name = tc.name
    a = tc.arguments or {}

    if name == "navigate_to":
        dest = a.get("dest")
        if dest not in ("A", "B"):
            raise InvalidToolCall(f"navigate_to requires dest in ['A','B'], got {dest!r}")
        return [navigate_to_task_dict(dest)]

    if name == "pickup":
        spec = _require_spec(a.get("spec", "long"))
        station = _resolve_station(orchestrator, spec)
        return pickup_task_dict(
            station=station, spec_key=spec, retreat_cm=a.get("retreat_cm")
        )

    if name == "place":
        spec = _require_spec(a.get("spec", "long"))
        # place retreat_cm is ignored for now (ObjectSpec has no place_retreat field).
        return place_task_dict(spec_key=spec)

    if name == "home":
        return home_task_dict()

    if name == "speak":
        msg = a.get("msg", "")
        if not isinstance(msg, str):
            msg = str(msg)
        return [speak_task_dict(name=f"voice_llm_{tc.id or _seq()}", msg=_sanitize_tts_msg(msg))]

    raise InvalidToolCall(f"unknown tool {name!r}")


_seq_counter = 0


def _seq() -> int:
    """Monotonic fallback suffix for speak task names when a tool_call has no id."""
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


def _resolve_station(orchestrator, spec: str) -> str:
    """Hidden pre-run classify: extract ``long_XX`` → ``"XX"``; fallback ``"00"``.

    The classify task is a **hidden primitive** (not an LLM tool): its label
    (``long_00``…``long_32``) encodes spec + station, aligned with
    ``SPEC_REGISTRY["long"].grasp_waypoints = "pickup_station_{station}_{1..5}"``.
    Phase 4 (``VisualSuccessDetector``) replaces this resolver's input source.
    """
    classify_task = None
    for task in getattr(orchestrator.config, "tasks", []) or []:
        if getattr(task, "task_type", None) == "classify":
            classify_task = task
            break
    if classify_task is None:
        logger.info("No classify task configured; defaulting station to '00'")
        return "00"

    result = orchestrator._execute_single_task(classify_task)
    trace = (getattr(result, "final_observation", None) or {}).get("task_trace", {})
    label = trace.get("label")
    m = re.search(r"long_(\d+)", str(label or ""))
    if m:
        return m.group(1)
    logger.warning("Classify did not yield a 'long_XX' label (got %r); defaulting to '00'", label)
    return "00"


# ---------------------------------------------------------------------------
# Skill-chain execution (reuse _execute_single_task; scope = one skill only)
# ---------------------------------------------------------------------------


def _execute_skill_chain(
    orchestrator, task_dicts: list[dict[str, Any]], skill_name: str = ""
) -> list[TaskResult]:
    """Execute one skill's task_dict list through the deterministic engine.

    Mirrors the non-interactive routing loop, but scoped to a single skill's
    dict list: parse all dicts, execute in order, follow intra-skill
    ``next_task`` jumps (e.g. grasp success → retreat agv), stop at ``cycle_end``
    or a failure. The **outer** skill choice belongs to the LLM, so
    ``next_task``/``cycle_end`` never select the next skill here.
    """
    named_positions = getattr(orchestrator.config, "named_positions", None) or {}
    agv = getattr(orchestrator.config, "agv_config", None)
    default_arm_safe = getattr(agv, "default_arm_safe_positions", {}) if agv else {}
    default_arm_home = getattr(agv, "default_arm_home_positions", {}) if agv else {}

    parsed = []
    by_name: dict[str, Any] = {}
    for td in task_dicts:
        tc = parse_task_dict(td, named_positions, default_arm_safe, default_arm_home)
        parsed.append(tc)
        by_name[tc.name] = tc

    # Phase 4 visual detector — built once per skill; inert unless enabled AND
    # the skill is pickup. Lazily imported so the flash path never loads it.
    vcfg = getattr(orchestrator.config, "visual_detector_config", None)
    detector = None
    if vcfg is not None and getattr(vcfg, "enabled", False) and skill_name == "pickup":
        from lerobot.agent.visual_detector import build_visual_detector, capture_frame
        detector = build_visual_detector(vcfg)

    results: list[TaskResult] = []
    idx = 0
    while idx < len(parsed):
        task = parsed[idx]
        if not task.enabled:
            idx += 1
            continue

        # Bracket only the grasp node (position_sequence) — the AGV retreat's
        # motion would swamp the material-disappearance diff.
        do_visual = detector is not None and task.task_type == "position_sequence"
        before_frame = None
        if do_visual:
            robot = getattr(orchestrator, "robot", None)
            before_frame = capture_frame(robot, vcfg.camera_key) if robot is not None else None

        orchestrator._memory_warmstart(task)
        result = orchestrator._execute_single_task(task)
        results.append(result)
        orchestrator._memory_record(task, result)
        if result.collision_detected:
            orchestrator.total_collision_count += 1

        if do_visual and before_frame is not None:
            if vcfg.settle_delay > 0:
                time.sleep(vcfg.settle_delay)
            robot = getattr(orchestrator, "robot", None)
            after_frame = capture_frame(robot, vcfg.camera_key) if robot is not None else None
            if after_frame is not None:
                verdict = detector.detect(before_frame, after_frame, camera_key=vcfg.camera_key)
                if not isinstance(result.final_observation, dict):
                    result.final_observation = {}
                result.final_observation["visual_evidence"] = asdict(verdict)
                logger.info(
                    "visual_evidence[%s]: success=%s changed_fraction=%.3f",
                    task.name, verdict.success, verdict.changed_fraction,
                )

        if result.status == TaskStatus.FATAL_FAILURE:
            break
        if result.status == TaskStatus.FAILED and task.stop_on_failure:
            break

        if result.next_task:
            # intra-skill jump only; an unknown/recovery node ends the chain so
            # the LLM sees the failure (its result carries next_task evidence).
            if result.next_task in by_name:
                idx = parsed.index(by_name[result.next_task])
                continue
            break

        if task.cycle_end:
            break
        idx += 1

    return results


# ---------------------------------------------------------------------------
# Flash fallback + summary
# ---------------------------------------------------------------------------


def _last_recovery_hint(chain_results: list[TaskResult]) -> RecoveryHint | None:
    """Recovery hint for the last result's failure mode (or None if clean)."""
    if not chain_results:
        return None
    fm = result_failure_mode(chain_results[-1])
    if fm is None:
        return None
    return FAILURE_RECOVERY_TABLE.get(fm)


def _flash_fallback(
    orchestrator, results: list[TaskResult], last_hint: RecoveryHint | None
) -> ExecutionSummary:
    """Degrade to deterministic replay.

    (a) A node-mappable ``flash_action`` (speak / home) is replayed directly;
    (b) otherwise — signal-only actions ("skip_retries"), empty hints, or an
    unreachable LLM before any result — the full ``_execute_with_safety()``
    deterministic chain runs and its results are merged in.
    """
    if last_hint is not None:
        action = last_hint.flash_action or ""
        if action == "voice_pickup_failed":
            extra = _execute_skill_chain(
                orchestrator, [speak_task_dict(name="voice_pickup_failed", msg="抓取失败，请协助")]
            )
            results.extend(extra)
            return _summary(orchestrator, results)
        if action in ("timeout_return_home", "agv_return_to_A_for_repick"):
            extra = _execute_skill_chain(orchestrator, home_task_dict())
            results.extend(extra)
            return _summary(orchestrator, results)

    flash_summary = orchestrator._execute_with_safety()
    results.extend(flash_summary.task_results)
    return _summary(orchestrator, results)


def _summary(orchestrator, results: list[TaskResult]) -> ExecutionSummary:
    """Build an ``ExecutionSummary`` from a flat result list."""
    completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
    failed = sum(1 for r in results if r.status in (TaskStatus.FAILED, TaskStatus.FATAL_FAILURE))
    skipped = sum(1 for r in results if r.status == TaskStatus.SKIPPED)
    fatal = sum(1 for r in results if r.status == TaskStatus.FATAL_FAILURE)
    return ExecutionSummary(
        total_tasks=len(results),
        completed_tasks=completed,
        failed_tasks=failed,
        skipped_tasks=skipped,
        total_duration=sum(r.duration for r in results),
        task_results=results,
        overall_success=(fatal == 0 and completed > 0),
        collision_count=getattr(orchestrator, "total_collision_count", 0),
        total_retries=sum(max(0, r.attempts - 1) for r in results),
    )
