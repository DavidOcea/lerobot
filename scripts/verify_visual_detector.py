#!/usr/bin/env python
"""Verify the Phase 4 visual success detector without a real camera or robot.

Five sections, all hardware/network-free:

  1. detector      — ``BaselineDiffDetector`` on synthetic ``np.uint8`` frames:
                     change fires success in the right direction; identical
                     frames (empty station) do not; ROI-outside change ignored;
                     below-threshold diff does not; confidence is monotonic in
                     changed_fraction.
  2. evidence-only — a verdict stored in ``final_observation`` leaves
                     ``success`` / ``status`` / ``next_task`` byte-for-byte
                     unchanged (the detector adds a slot, never overwrites).
  3. protocol      — ``skill_observation`` emits ``visual_evidence`` from the
                     *non-last* (grasp) result and ``null`` when none carries it.
  4. loop hook     — ``_execute_skill_chain`` with ``enabled=True`` + a fake
                     robot captures exactly two frames and attaches the verdict
                     to the ``position_sequence`` result, leaving the AGV
                     retreat result untouched.
  5. disabled      — ``enabled=False``, a non-pickup ``skill_name``, and a
                     ``robot=None`` each make the hook a no-op (no camera read).

Exit 0 = all green, 1 = any failure. Run with the lerobot env python.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from lerobot.agent.config import VisualDetectorConfig
from lerobot.agent.llm_loop import _execute_skill_chain
from lerobot.agent.llm_protocol import ToolCall, skill_observation
from lerobot.agent.visual_detector import build_visual_detector
from lerobot.tasks.skills import SPEC_REGISTRY, pickup_task_dict
from lerobot.tasks.task_scheduler import TaskResult, TaskStatus


_UNSET = object()


class _FakeRobot:
    """Returns a dark before-frame then a bright after-frame, counting reads."""

    def __init__(self):
        self.calls = 0

    def get_observation(self):
        self.calls += 1
        frame = (
            np.full((20, 20, 3), 20, np.uint8)
            if self.calls == 1
            else np.full((20, 20, 3), 220, np.uint8)
        )
        return {"images": {"head_cam": frame}}


def _make_orch(enabled: bool, skill_name: str, single_fn=None, robot=_UNSET):
    orch = SimpleNamespace()
    orch.robot = _FakeRobot() if robot is _UNSET else robot
    orch.total_collision_count = 0
    orch.executed: list[str] = []
    named = {"home": {}}
    named.update({f"pickup_station_07_{i}": {} for i in range(1, 6)})
    named.update({wp: {} for wp in SPEC_REGISTRY["long"].place_waypoints})
    orch.config = SimpleNamespace(
        tasks=[],
        named_positions=named,
        agv_config=None,
        visual_detector_config=VisualDetectorConfig(enabled=enabled, settle_delay=0.0),
    )

    def _single(task):
        orch.executed.append(task.name)
        if single_fn is not None:
            return single_fn(task)
        return TaskResult(task_name=task.name, status=TaskStatus.COMPLETED, success=True)

    orch._execute_single_task = _single
    orch._memory_warmstart = lambda task: None
    orch._memory_record = lambda task, result: None
    return orch


def section_detector(fail: list[str]) -> None:
    d = build_visual_detector(VisualDetectorConfig())

    before = np.full((100, 100, 3), 20, np.uint8)
    after = np.full((100, 100, 3), 220, np.uint8)
    v = d.detect(before, after, camera_key="head_cam")
    if not v.success or abs(v.changed_fraction - 1.0) > 1e-6 or v.direction != "decrease":
        fail.append(f"detector dark->bright: {v}")

    v_same = d.detect(before, before)
    if v_same.success or v_same.changed_fraction != 0.0:
        fail.append(f"detector identical (empty station) should not succeed: {v_same}")

    # change entirely outside the ROI is ignored
    d_roi = build_visual_detector(VisualDetectorConfig(roi=(0, 0, 50, 50)))
    b = np.zeros((100, 100), np.uint8)
    a = np.zeros((100, 100), np.uint8)
    a[60:100, 60:100] = 255
    v_roi = d_roi.detect(b, a)
    if v_roi.success or v_roi.changed_fraction != 0.0:
        fail.append(f"detector ROI-outside ignored: {v_roi}")

    # sub-threshold diff (|Δ|=5 < pixel_threshold 30) -> no change detected
    v_low = d.detect(np.full((50, 50, 3), 100, np.uint8), np.full((50, 50, 3), 105, np.uint8))
    if v_low.success or v_low.changed_fraction != 0.0:
        fail.append(f"detector sub-threshold: {v_low}")

    # confidence monotonic in changed_fraction (50% vs 90% of pixels changed)
    d_c = build_visual_detector(VisualDetectorConfig(diff_threshold=0.01))
    base = np.zeros((100, 100), np.uint8)
    half = base.copy(); half[:50, :] = 255
    most = base.copy(); most[:90, :] = 255
    c_half = d_c.detect(base, half).confidence
    c_most = d_c.detect(base, most).confidence
    if not (c_most >= c_half):
        fail.append(f"detector confidence not monotonic: {c_half} -> {c_most}")

    try:
        build_visual_detector(VisualDetectorConfig(method="nope"))
        fail.append("detector: unknown method should raise")
    except ValueError:
        pass


def section_evidence_only(fail: list[str]) -> None:
    result = TaskResult(
        task_name="pickup_at_station_07",
        status=TaskStatus.FAILED,
        success=False,
        error_message="No workpiece detected",
        next_task="voice_pickup_failed",
        final_observation={},
    )
    result.final_observation["visual_evidence"] = {"success": True, "changed_fraction": 0.7}
    if result.success is not False:
        fail.append("evidence-only: result.success mutated")
    if result.status != TaskStatus.FAILED:
        fail.append("evidence-only: result.status mutated")
    if result.next_task != "voice_pickup_failed":
        fail.append("evidence-only: result.next_task mutated")
    if result.final_observation["visual_evidence"]["success"] is not True:
        fail.append("evidence-only: visual_evidence not stored in final_observation")


def section_protocol(fail: list[str]) -> None:
    tc = ToolCall(id="c1", name="pickup", arguments={"spec": "long"})
    grasp = TaskResult(
        task_name="pickup_at_station_07",
        status=TaskStatus.COMPLETED,
        success=True,
        final_observation={"visual_evidence": {"success": True, "changed_fraction": 0.62}},
    )
    retreat = TaskResult(task_name="agv_backward_50cm_A1_1", status=TaskStatus.COMPLETED, success=True)

    s = json.loads(skill_observation(tc, [grasp, retreat]))
    if s.get("visual_evidence") != {"success": True, "changed_fraction": 0.62}:
        fail.append(f"protocol: visual_evidence from non-last result: {s.get('visual_evidence')}")

    s2 = json.loads(skill_observation(tc, [retreat]))
    if "visual_evidence" not in s2 or s2["visual_evidence"] is not None:
        fail.append(f"protocol: absent should be null: {s2.get('visual_evidence')!r}")


def section_hook(fail: list[str]) -> None:
    orch = _make_orch(enabled=True, skill_name="pickup")
    results = _execute_skill_chain(
        orch, pickup_task_dict(station="07", spec_key="long"), skill_name="pickup"
    )
    if orch.robot.calls != 2:
        fail.append(f"hook: expected 2 camera reads, got {orch.robot.calls}")
    if orch.executed != ["pickup_at_station_07", "agv_backward_50cm_A1_1"]:
        fail.append(f"hook: unexpected execution order {orch.executed}")

    grasp = next(r for r in results if r.task_name == "pickup_at_station_07")
    retreat = next(r for r in results if r.task_name.startswith("agv"))
    if "visual_evidence" not in (grasp.final_observation or {}):
        fail.append("hook: verdict not attached to grasp result")
    if "visual_evidence" in (retreat.final_observation or {}):
        fail.append("hook: retreat result should be untouched")
    ve = (grasp.final_observation or {}).get("visual_evidence", {})
    if not ve.get("success") or abs(ve.get("changed_fraction", 0.0) - 1.0) > 1e-6:
        fail.append(f"hook: wrong verdict {ve}")


def section_hook_evidence_only(fail: list[str]) -> None:
    # the detector adds a slot to final_observation but never rewrites semantics
    def fail_grasp(task):
        if task.name == "pickup_at_station_07":
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.FAILED,
                success=False,
                error_message="No workpiece detected",
                next_task="voice_pickup_failed",
            )
        return TaskResult(task_name=task.name, status=TaskStatus.COMPLETED, success=True)

    orch = _make_orch(enabled=True, skill_name="pickup", single_fn=fail_grasp)
    results = _execute_skill_chain(
        orch, pickup_task_dict(station="07", spec_key="long"), skill_name="pickup"
    )
    grasp = next(r for r in results if r.task_name == "pickup_at_station_07")
    if grasp.success is not False or grasp.status != TaskStatus.FAILED or grasp.next_task != "voice_pickup_failed":
        fail.append(f"hook evidence-only: semantic fields mutated: {grasp}")
    if "visual_evidence" not in (grasp.final_observation or {}):
        fail.append("hook evidence-only: verdict missing")


def section_disabled(fail: list[str]) -> None:
    # enabled=False -> no camera read
    orch = _make_orch(enabled=False, skill_name="pickup")
    results = _execute_skill_chain(
        orch, pickup_task_dict(station="07", spec_key="long"), skill_name="pickup"
    )
    if orch.robot.calls != 0:
        fail.append(f"disabled: camera should not be read, got {orch.robot.calls} calls")
    if any("visual_evidence" in (r.final_observation or {}) for r in results):
        fail.append("disabled: a verdict leaked in")

    # enabled but non-pickup skill_name -> no camera read
    orch2 = _make_orch(enabled=True, skill_name="place")
    _execute_skill_chain(orch2, pickup_task_dict(station="07", spec_key="long"), skill_name="place")
    if orch2.robot.calls != 0:
        fail.append(f"non-pickup: camera should not be read, got {orch2.robot.calls} calls")

    # enabled + pickup but robot=None -> graceful no-op
    orch3 = _make_orch(enabled=True, skill_name="pickup", robot=None)
    results3 = _execute_skill_chain(
        orch3, pickup_task_dict(station="07", spec_key="long"), skill_name="pickup"
    )
    if any("visual_evidence" in (r.final_observation or {}) for r in results3):
        fail.append("robot=None: a verdict leaked in")


def main() -> int:
    fail: list[str] = []
    section_detector(fail)
    section_evidence_only(fail)
    section_protocol(fail)
    section_hook(fail)
    section_hook_evidence_only(fail)
    section_disabled(fail)

    if fail:
        print("FAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("OK: Phase 4 visual success detector verified (5 sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
