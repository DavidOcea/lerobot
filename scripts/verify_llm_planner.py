#!/usr/bin/env python
"""Verify the Phase 3 LLM planner loop without a real LLM or robot.

Six sections, all network/hardware-free:

  1. protocol   — ``serialize_tools`` emits the 5 tools in OpenAI format,
                  returns/failure_modes are folded into the description, and
                  hidden fields (station/nudge/waypoints/…) never leak.
  2. messages   — ``build_messages`` / assistant-tool-call / tool-result shapes,
                  and ``skill_observation`` serializes a structured conclusion.
  3. dispatch   — the 5 tool names map to generator task_dicts that
                  ``parse_task_dict`` accepts (correct ``task_type`` + validate).
  4. chain      — ``_execute_skill_chain`` follows intra-skill ``next_task``
                  jumps and stops at ``cycle_end`` (fake ``_execute_single_task``).
  5. failure    — ``map_failure_to_mode`` / ``result_failure_mode`` recover a
                  structured ``FailureMode`` (error_message + next_task signals).
  6. end-to-end — a scripted ``FakePlanner`` drives navigate_to→pickup→…→home→stop
                  and aggregates a summary with no flash fallback; a planner that
                  raises ``LLMClientError`` degrades to ``_execute_with_safety``.

Exit 0 = all green, 1 = any failure. Run with the lerobot env python.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from lerobot.agent.llm_client import LLMClientError
from lerobot.agent.llm_loop import _dispatch, _execute_skill_chain, run_llm_loop
from lerobot.agent.llm_protocol import (
    ToolCall,
    build_assistant_tool_call_message,
    build_messages,
    build_tool_message,
    map_failure_to_mode,
    result_failure_mode,
    serialize_tools,
    skill_observation,
)
from lerobot.agent.planner import PlannerDecision
from lerobot.tasks.config import ClassifyConfig, TaskConfig, parse_task_dict
from lerobot.tasks.failure_mode import FAILURE_RECOVERY_TABLE, FailureMode
from lerobot.tasks.skills import SPEC_REGISTRY, home_task_dict, pickup_task_dict
from lerobot.tasks.task_scheduler import ExecutionSummary, TaskResult, TaskStatus
from lerobot.tasks.tool_spec import to_tool_spec


EXPECTED_TOOLS = ["navigate_to", "pickup", "place", "speak", "home"]
HIDDEN_FIELDS = {"station", "nudge", "waypoints", "gripper", "align", "primitive", "agv", "classify"}


# ---------------------------------------------------------------------------
# Fake orchestrator — the only object the loop/chain/dispatch need from the real
# orchestrator is config + the three execute/memory hooks.
# ---------------------------------------------------------------------------


class FakeOrchestrator:
    def __init__(self, single_task_fn=None, classify_label="long_07", safety_summary=None):
        self.single_task_fn = single_task_fn or self._default_single
        self.classify_label = classify_label
        self._safety_summary = safety_summary or self._default_safety()
        self.total_collision_count = 0
        self.executed: list[str] = []
        self.safety_called = False
        self.config = self._build_config()

    def _default_single(self, task: TaskConfig) -> TaskResult:
        return TaskResult(task_name=task.name, status=TaskStatus.COMPLETED, success=True)

    def _default_safety(self) -> ExecutionSummary:
        return ExecutionSummary(
            total_tasks=1, completed_tasks=1, failed_tasks=0, skipped_tasks=0,
            total_duration=0.0, task_results=[], overall_success=True,
        )

    def _build_config(self):
        classify = TaskConfig(
            name="classify_workpiece", task_type="classify", classify_config=ClassifyConfig()
        )
        named_positions = {"home": {}}
        named_positions.update({f"pickup_station_07_{i}": {} for i in range(1, 6)})
        named_positions.update({wp: {} for wp in SPEC_REGISTRY["long"].place_waypoints})
        llm_cfg = SimpleNamespace(
            model="test-model", base_url="http://localhost:8000/v1", api_key="",
            api_key_env="TEST_KEY", system_prompt="", max_tool_calls=12,
            temperature=0.0, timeout=1.0,
        )
        return SimpleNamespace(
            tasks=[classify], named_positions=named_positions, agv_config=None, llm_config=llm_cfg,
        )

    # -- hooks the loop/chain touch -----------------------------------------
    def _execute_single_task(self, task: TaskConfig) -> TaskResult:
        self.executed.append(task.name)
        if getattr(task, "task_type", None) == "classify":
            return TaskResult(
                task_name=task.name, status=TaskStatus.COMPLETED, success=True,
                final_observation={"task_trace": {"label": self.classify_label}},
            )
        return self.single_task_fn(task)

    def _memory_warmstart(self, task: TaskConfig) -> None:
        return None

    def _memory_record(self, task: TaskConfig, result: TaskResult) -> None:
        return None

    def _execute_with_safety(self) -> ExecutionSummary:
        self.safety_called = True
        return self._safety_summary


class FakePlanner:
    """Pops one decision per ``solve``; empty script → stop."""

    def __init__(self, script):
        self.script = list(script)

    def solve(self, messages, tools):
        if not self.script:
            return PlannerDecision(tool_call=None, finish_reason="stop")
        return self.script.pop(0)


class FailingPlanner:
    def solve(self, messages, tools):
        raise LLMClientError("simulated transport failure")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_protocol(fail: list[str]) -> None:
    specs = to_tool_spec()
    tools = serialize_tools(specs)
    if len(tools) != 5 or {t["function"]["name"] for t in tools} != set(EXPECTED_TOOLS):
        fail.append(f"protocol: 5 tools expected, got {[t['function']['name'] for t in tools]}")
        return

    for t in tools:
        fn = t["function"]
        if t.get("type") != "function":
            fail.append(f"{fn['name']}: type != function")
        props = set(fn["parameters"].get("properties", {}))
        leak = props & HIDDEN_FIELDS
        if leak:
            fail.append(f"{fn['name']}: hidden fields leaked {sorted(leak)}")
        if "Returns (JSON):" not in fn["description"]:
            fail.append(f"{fn['name']}: returns not folded into description")

    # failure_modes folded in only for tools that declare them
    if "Possible failure_modes:" not in next(t for t in tools if t["function"]["name"] == "navigate_to")["function"]["description"]:
        fail.append("navigate_to: failure_modes missing from description")
    if "Possible failure_modes:" in next(t for t in tools if t["function"]["name"] == "speak")["function"]["description"]:
        fail.append("speak: failure_modes should be absent from description")


def section_messages(fail: list[str]) -> None:
    tc = ToolCall(id="call_1", name="navigate_to", arguments={"dest": "A"})
    msgs = build_messages("sys", [])
    if msgs != [{"role": "system", "content": "sys"}]:
        fail.append(f"build_messages: {msgs}")

    asst = build_assistant_tool_call_message(tc)
    if asst["role"] != "assistant" or asst["tool_calls"][0]["id"] != "call_1":
        fail.append(f"assistant tool-call shape: {asst}")

    tool = build_tool_message(tc, "done")
    if tool["role"] != "tool" or tool["tool_call_id"] != "call_1":
        fail.append(f"tool-result shape: {tool}")

    # structured observation: a failed result exposes failure_mode + llm_hint
    bad = TaskResult(task_name="pickup_at_station_07", status=TaskStatus.FAILED,
                     success=False, error_message="No workpiece detected")
    obs = skill_observation(tc, [bad])
    if '"failure_mode"' not in obs or '"llm_hint"' not in obs or '"success": false' not in obs:
        fail.append(f"skill_observation: {obs}")


def section_dispatch(fail: list[str]) -> None:
    orch = FakeOrchestrator()

    expected = [
        ("navigate_to", {"dest": "A"}, [("agv", "agv_to_station_A")]),
        ("navigate_to", {"dest": "B"}, [("agv", "agv_to_station_B1")]),
        ("pickup", {"spec": "long"}, [("position_sequence", "pickup_at_station_07"),
                                      ("agv", "agv_backward_50cm_A1_1")]),
        ("place", {"spec": "long"}, [("agv", "agv_forward_5cm_B1_1"),
                                     ("position_sequence", "place_long_1")]),
        ("home", {}, [("parallel", "home_and_return_A_after_long_1"),
                      ("agv", "agv_return_from_B1_to_A")]),
        ("speak", {"msg": "hello 'world"}, [("system_command", "voice_llm_call_9")]),
    ]
    for name, args, want in expected:
        tc = ToolCall(id="call_9", name=name, arguments=args)
        task_dicts = _dispatch(tc, orch)
        if len(task_dicts) != len(want):
            fail.append(f"{name}: {len(task_dicts)} dicts != {len(want)}")
            continue
        for td, (wt_type, wt_name) in zip(task_dicts, want):
            parsed = parse_task_dict(td, orch.config.named_positions, {}, {})
            if parsed.task_type != wt_type or parsed.name != wt_name:
                fail.append(f"{name}: got ({parsed.task_type}, {parsed.name}) != ({wt_type}, {wt_name})")
            try:
                parsed.validate()
            except Exception as e:  # noqa: BLE001
                fail.append(f"{name}/{parsed.name}: validate failed: {e}")

    # station resolved from the hidden classify pre-run (long_07 → "07")
    pickup_dicts = _dispatch(ToolCall(id="c", name="pickup", arguments={"spec": "long"}), orch)
    if pickup_dicts[0]["name"] != "pickup_at_station_07":
        fail.append(f"dispatch: station not resolved from classify ({pickup_dicts[0]['name']})")

    # classify missing → fallback "00"
    orch_no_classify = FakeOrchestrator()
    orch_no_classify.config.tasks = []
    fallback = _dispatch(ToolCall(id="c", name="pickup", arguments={"spec": "long"}), orch_no_classify)
    if fallback[0]["name"] != "pickup_at_station_00":
        fail.append(f"dispatch: classify-missing fallback wrong ({fallback[0]['name']})")


def section_chain(fail: list[str]) -> None:
    # (a) intra-skill next_task jump: pickup position_sequence → agv retreat
    def jump_fn(task):
        if task.name == "pickup_at_station_07":
            return TaskResult(task_name=task.name, status=TaskStatus.COMPLETED, success=True,
                              next_task="agv_backward_50cm_A1_1")
        return TaskResult(task_name=task.name, status=TaskStatus.COMPLETED, success=True)

    orch = FakeOrchestrator(single_task_fn=jump_fn)
    results = _execute_skill_chain(orch, pickup_task_dict(station="07", spec_key="long"))
    if orch.executed != ["pickup_at_station_07", "agv_backward_50cm_A1_1"] or len(results) != 2:
        fail.append(f"chain jump: executed={orch.executed} results={len(results)}")

    # (b) cycle_end stop: home parallel → agv return-to-A (cycle_end)
    orch2 = FakeOrchestrator()
    results2 = _execute_skill_chain(orch2, home_task_dict())
    if len(results2) != 2 or orch2.executed[-1] != "agv_return_from_B1_to_A":
        fail.append(f"chain cycle_end: executed={orch2.executed}")


def section_failure(fail: list[str]) -> None:
    cases = {
        "AGV navigation timeout at station LM2": FailureMode.NAVIGATION_STUCK,
        "task timed out": FailureMode.TIMEOUT,
        "Tag lost during align": FailureMode.TAG_LOST,
        "Classification failed after retries": FailureMode.CLASSIFY_EXHAUSTED,
        "No workpiece detected after 3s": FailureMode.NO_DETECT,
    }
    for msg, want in cases.items():
        got = map_failure_to_mode(msg)
        if got != want:
            fail.append(f"map_failure_to_mode({msg!r}) = {got} != {want}")
    if map_failure_to_mode("") is not None or map_failure_to_mode(None) is not None:
        fail.append("map_failure_to_mode empty/None should be None")

    # result_failure_mode: error_message first, then next_task evidence
    r1 = TaskResult(task_name="x", status=TaskStatus.COMPLETED, error_message="No workpiece detected")
    if result_failure_mode(r1) != FailureMode.NO_DETECT:
        fail.append("result_failure_mode: error_message path")
    r2 = TaskResult(task_name="pickup", status=TaskStatus.COMPLETED, next_task="voice_pickup_failed")
    if result_failure_mode(r2) != FailureMode.EMPTY_GRASP:
        fail.append("result_failure_mode: next_task→EMPTY_GRASP")
    r3 = TaskResult(task_name="x", status=TaskStatus.COMPLETED, next_task="timeout_return_home")
    if result_failure_mode(r3) != FailureMode.TIMEOUT:
        fail.append("result_failure_mode: next_task→TIMEOUT")

    for mode in FailureMode:
        if mode not in FAILURE_RECOVERY_TABLE:
            fail.append(f"FAILURE_RECOVERY_TABLE missing {mode}")


def _happy_script():
    def d(cid, name, args):
        return PlannerDecision(tool_call=ToolCall(id=cid, name=name, arguments=args),
                               finish_reason="tool_calls")
    return [
        d("c1", "navigate_to", {"dest": "A"}),
        d("c2", "pickup", {"spec": "long"}),
        d("c3", "navigate_to", {"dest": "B"}),
        d("c4", "place", {"spec": "long"}),
        d("c5", "home", {}),
        PlannerDecision(tool_call=None, finish_reason="stop"),
    ]


def section_end_to_end(fail: list[str]) -> None:
    # happy path: scripted LLM, no fallback
    orch = FakeOrchestrator()
    summary = run_llm_loop(orch, planner=FakePlanner(_happy_script()))
    if summary.total_tasks != 8 or summary.completed_tasks != 8 or not summary.overall_success:
        fail.append(f"e2e happy: total={summary.total_tasks} completed={summary.completed_tasks} ok={summary.overall_success}")
    if orch.safety_called:
        fail.append("e2e happy: flash fallback should NOT have run")

    # transport failure → degrade to _execute_with_safety (its results merged back)
    replay_result = TaskResult(task_name="flash_replay", status=TaskStatus.COMPLETED, success=True)
    orch2 = FakeOrchestrator(safety_summary=ExecutionSummary(
        total_tasks=1, completed_tasks=1, failed_tasks=0, skipped_tasks=0,
        total_duration=0.0, task_results=[replay_result], overall_success=True,
    ))
    summary2 = run_llm_loop(orch2, planner=FailingPlanner())
    if not orch2.safety_called:
        fail.append("e2e failing: _execute_with_safety should have been called")
    if not any(r.task_name == "flash_replay" for r in summary2.task_results):
        fail.append("e2e failing: flash replay results not merged into summary")


def main() -> int:
    fail: list[str] = []
    section_protocol(fail)
    section_messages(fail)
    section_dispatch(fail)
    section_chain(fail)
    section_failure(fail)
    section_end_to_end(fail)

    if fail:
        print("FAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("OK: Phase 3 LLM planner loop verified (6 sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
