#!/usr/bin/env python
"""Verify the ADC2 task-graph routing topology offline (no hardware/network).

The ADC2 fixture's executors are hardware-coupled (AGV TCP, 15 named supre
joints, YOLO on real camera frames), so a mock-robot "full run" can only prove
that the routing engine *called* executors — not that the graph is right. This
script tests the only layer that is genuinely offline-testable: the routing
graph itself. It loads the real YAML through the same ``load_config_from_yaml``
the orchestrator uses, then mirrors the routing contract exactly:

* classify success label ``L``            → ``next_task = next_tasks.get(L, default_next_task)``
* classify retries exhausted + recovery   → ``COMPLETED + next_task = recovery_task``
* classify exhausted, no recovery         → ``FAILED`` (``stop_on_failure`` breaks)
* non-classify success                    → ``next_tasks["success"]`` (else sequential)
* non-classify failure                    → ``next_tasks["failed"]`` (recovery) else
                                            ``stop_on_failure`` break / sequential
* loop: ``next_task`` name-jump (first match wins) / ``cycle_end`` break / seq +1

Sections:
  1. topology   — no dangling ``next_tasks``/``default_next_task``/``recovery_task``
                  references; duplicate task names reported as warnings.
  2. happy      — full success + ``long_00`` + check-material ``long`` reaches the
                  12-node terminal chain ``agv_return_from_B1_to_A`` (cycle_end).
  3. branches   — each of the 12 ``long_XX`` labels routes to ``pickup_at_station_XX``
                  and the full path terminates at cycle_end.
  4. pickup fail— ``voice_pickup_failed`` re-enters ``classify_workpiece`` on success
                  (bounded retry) and goes ``agv_retreat_50cm`` on failure.
  5. align fail — ``voice_align_failed`` → ``agv_retreat_align`` (cycle_end).
  6. drop       — check-material ``none``/``unknown`` → voice + ``agv_return_to_A_for_repick``.
  7. exhausted  — ``classify_workpiece`` exhaustion → ``timeout_return_home`` →
                  ``agv_retreat_50cm`` (cycle_end).

Exit 0 = all green (warnings printed but non-fatal), 1 = any failure. Run with
the lerobot env python.
"""

from __future__ import annotations

import sys
from typing import Callable

from lerobot.tasks.config import load_config_from_yaml

DEFAULT_CONFIG = "/root/workspace/dc_dir/agv_class_ADC2.yaml"

# A policy answers "what did the executor produce" for each task: a label for
# classify nodes, "exhausted" for classify giving up, else "success"/"failed".
OutcomeFn = Callable[["TaskConfig", dict[str, int]], str]


def _static_outcome(mapping: dict[str, str], default_label: str = "long") -> OutcomeFn:
    """Wrap a {task_name: outcome} dict into a callable policy."""
    def fn(task, seen):
        if task.name in mapping:
            return mapping[task.name]
        return default_label if task.task_type == "classify" else "success"
    return fn


def derive(task, outcome: str) -> tuple[str, str | None]:
    """Mirror the executor + generic routing → (next_task, stop_reason)."""
    if task.task_type == "classify":
        cc = task.classify_config
        if outcome == "exhausted":
            nt = (cc.recovery_task or "") if cc else ""
            if nt:
                return nt, None  # COMPLETED + recovery jump
            return "", "CLASSIFY_FAILED"  # FAILED, stop_on_failure breaks
        nt = task.next_tasks.get(outcome, (cc.default_next_task if cc else "") or "")
        return nt, None

    nt = task.next_tasks.get(outcome, "")
    if outcome == "failed" and not nt and task.stop_on_failure:
        return "", "STOP_ON_FAILURE"
    return nt, None


def walk(tasks, outcome_fn: OutcomeFn, max_steps: int = 500):
    """Mirror the non-interactive routing loop; returns (trace, stop_reason)."""
    by_name: dict[str, int] = {}
    for i, t in enumerate(tasks):
        if t.name not in by_name:
            by_name[t.name] = i  # first occurrence wins, like the linear scan

    idx = 0
    seen: dict[str, int] = {}
    trace: list[tuple[str, str, str, str | None]] = []
    for _ in range(max_steps):
        while idx < len(tasks) and not tasks[idx].enabled:
            idx += 1
        if idx >= len(tasks):
            return trace, "EXHAUSTED"
        task = tasks[idx]
        seen[task.name] = seen.get(task.name, 0) + 1
        outcome = outcome_fn(task, seen)
        nt, stop = derive(task, outcome)
        trace.append((task.name, outcome, nt, stop))
        if stop:
            return trace, stop
        if nt:
            idx = by_name[nt] if nt in by_name else idx + 1
            continue
        if task.cycle_end:
            return trace, "CYCLE_END"
        idx += 1
    return trace, "LOOP_GUARD"


def _names(trace) -> list[str]:
    return [n for n, _, _, _ in trace]


HAPPY_PATH = [
    "agv_to_station_A",
    "align_to_tag",
    "classify_workpiece",
    "pickup_at_station_00",
    "agv_backward_50cm_A1_1",
    "check_material_before_pickup_A1",
    "agv_to_station_B1",
    "check_material_before_place_B1",
    "agv_forward_5cm_B1_1",
    "place_long_1",
    "home_and_return_A_after_long_1",
    "agv_return_from_B1_to_A",
]

# 12 classify labels → their station suffix.
CLASSIFY_LABELS = [f"{r}{c}" for r in "0123" for c in "012"]


def section_topology(fail: list[str], warn: list[str], tasks) -> None:
    by_name: dict[str, int] = {}
    for t in tasks:
        by_name[t.name] = by_name.get(t.name, 0) + 1
    for name, k in by_name.items():
        if k > 1:
            warn.append(f"duplicate task name {name!r} appears {k}x (first match wins)")

    for t in tasks:
        for src, target in t.next_tasks.items():
            if target not in by_name:
                fail.append(f"{t.name}: next_tasks[{src}]={target!r} dangling")
        if t.task_type == "classify" and t.classify_config is not None:
            cc = t.classify_config
            if cc.default_next_task and cc.default_next_task not in by_name:
                fail.append(f"{t.name}: default_next_task={cc.default_next_task!r} dangling")
            if cc.recovery_task and cc.recovery_task not in by_name:
                fail.append(f"{t.name}: recovery_task={cc.recovery_task!r} dangling")


def section_happy(fail: list[str], tasks) -> None:
    policy = _static_outcome(
        {"classify_workpiece": "long_00", "check_material_before_pickup_A1": "long",
         "check_material_before_place_B1": "long"}
    )
    trace, stop = walk(tasks, policy)
    names = _names(trace)
    if stop != "CYCLE_END":
        fail.append(f"happy: expected CYCLE_END, got {stop!r} ({names})")
        return
    if names != HAPPY_PATH:
        fail.append(f"happy: path mismatch\n  got={names}\n  exp={HAPPY_PATH}")


def section_branches(fail: list[str], tasks) -> None:
    seen_stations: list[str] = []
    for label in [f"long_{s}" for s in CLASSIFY_LABELS]:
        station = f"pickup_at_station_{label.split('_')[1]}"
        policy = _static_outcome(
            {"classify_workpiece": label, "check_material_before_pickup_A1": "long",
             "check_material_before_place_B1": "long"}
        )
        trace, stop = walk(tasks, policy)
        names = _names(trace)
        if len(names) < 4 or names[3] != station:
            fail.append(f"branch {label}: expected {station} right after classify, got {names[:4]}")
        if stop != "CYCLE_END" or names[-1] != "agv_return_from_B1_to_A":
            fail.append(f"branch {label}: did not terminate at agv_return_from_B1_to_A ({stop}, {names[-3:]})")
        seen_stations.append(station)
    if len(set(seen_stations)) != 12:
        fail.append(f"branches: 12 labels did not cover 12 distinct stations ({seen_stations})")


def section_pickup_fail(fail: list[str], tasks) -> None:
    # voice success → re-enter classify_workpiece (bounded retry; pickup succeeds 2nd time)
    def retry_policy(task, seen):
        if task.name == "pickup_at_station_00":
            return "failed" if seen.get(task.name, 0) == 1 else "success"
        if task.name == "classify_workpiece":
            return "long_00"
        if task.task_type == "classify":
            return "long"
        return "success"

    trace, stop = walk(tasks, retry_policy)
    names = _names(trace)
    if stop != "CYCLE_END":
        fail.append(f"pickup-fail retry: expected CYCLE_END, got {stop!r} ({names})")
    elif names.count("classify_workpiece") != 2 or names.count("pickup_at_station_00") != 2:
        fail.append(f"pickup-fail retry: expected classify+pickup 2x each, got {names}")

    # voice failure → agv_retreat_50cm (cycle_end)
    trace2, stop2 = walk(tasks, _static_outcome(
        {"pickup_at_station_00": "failed", "voice_pickup_failed": "failed",
         "classify_workpiece": "long_00"}
    ))
    names2 = _names(trace2)
    if stop2 != "CYCLE_END" or names2[-1] != "agv_retreat_50cm":
        fail.append(f"pickup-fail voice-fail: expected agv_retreat_50cm CYCLE_END, got {stop2!r} ({names2})")


def section_align_fail(fail: list[str], tasks) -> None:
    trace, stop = walk(tasks, _static_outcome({"align_to_tag": "failed"}))
    names = _names(trace)
    if stop != "CYCLE_END" or names[-1] != "agv_retreat_align":
        fail.append(f"align-fail: expected agv_retreat_align CYCLE_END, got {stop!r} ({names})")


def section_drop(fail: list[str], tasks) -> None:
    # pickup-side check-material none/unknown → voice_material_dropped_A → repick (cycle_end)
    for label in ("none", "unknown"):
        policy = _static_outcome(
            {"classify_workpiece": "long_00", "check_material_before_pickup_A1": label}
        )
        trace, stop = walk(tasks, policy)
        names = _names(trace)
        if stop != "CYCLE_END" or names[-1] != "agv_return_to_A_for_repick" or "voice_material_dropped_A" not in names:
            fail.append(f"drop-pickup {label}: expected voice_material_dropped_A→repick, got {stop!r} ({names})")

    # place-side check-material none/unknown → voice_material_dropped_B → repick (cycle_end)
    for label in ("none", "unknown"):
        policy = _static_outcome(
            {"classify_workpiece": "long_00", "check_material_before_pickup_A1": "long",
             "check_material_before_place_B1": label}
        )
        trace, stop = walk(tasks, policy)
        names = _names(trace)
        if stop != "CYCLE_END" or names[-1] != "agv_return_to_A_for_repick" or "voice_material_dropped_B" not in names:
            fail.append(f"drop-place {label}: expected voice_material_dropped_B→repick, got {stop!r} ({names})")


def section_exhausted(fail: list[str], tasks) -> None:
    trace, stop = walk(tasks, _static_outcome({"classify_workpiece": "exhausted"}))
    names = _names(trace)
    if stop != "CYCLE_END" or names[-1] != "agv_retreat_50cm" or "timeout_return_home" not in names:
        fail.append(f"exhausted: expected timeout_return_home→agv_retreat_50cm, got {stop!r} ({names})")


def main(argv: list[str]) -> int:
    config_path = argv[1] if len(argv) > 1 else DEFAULT_CONFIG
    config = load_config_from_yaml(config_path)
    tasks = config.tasks

    fail: list[str] = []
    warn: list[str] = []
    section_topology(fail, warn, tasks)
    section_happy(fail, tasks)
    section_branches(fail, tasks)
    section_pickup_fail(fail, tasks)
    section_align_fail(fail, tasks)
    section_drop(fail, tasks)
    section_exhausted(fail, tasks)

    for w in warn:
        print(f"WARN: {w}")
    if fail:
        print("FAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print(f"OK: routing graph verified ({len(tasks)} nodes, 7 sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
