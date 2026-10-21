#!/usr/bin/env python
"""Verify ``skills.py`` generators reproduce YAML task nodes bit-identically.

Usage:
    python scripts/verify_skill_equivalence.py /path/to/agv_class_online_test.yaml

For each YAML node that Phase 1 folds into a skill, this compares
``parse_task_dict(yaml_node)`` against ``parse_task_dict(generated_dict)``
using ``dataclasses.asdict`` equality (all TaskConfig fields, resolved named
positions, and injected arm-safe/home defaults). Exit 0 = full equivalence,
1 = any diff.

Phase 2-3 nodes are intentionally NOT compared (align_to_tag, classify_workpiece,
check_material_*, voice_pickup_failed with next_tasks, agv_return_to_A_for_repick,
agv_retreat_align, agv_retreat_50cm).
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from lerobot.tasks.config import parse_task_dict
from lerobot.tasks.skills import (
    home_task_dict,
    navigate_to_task_dict,
    pickup_task_dict,
    place_task_dict,
    speak_task_dict,
)

# 12 pickup stations (4 rows x 中间/偏左/偏右).
PICKUP_STATIONS = ["00", "01", "02", "10", "11", "12", "20", "21", "22", "30", "31", "32"]

# Simple voice nodes (no next_tasks) folded by speak_task_dict.
SPEAK_NODES = {
    "voice_align_failed": "无法对齐，请调整工作台位置",
    "timeout_return_home": "检测超时，返回起点",
    "voice_material_dropped_A": "料片掉落，准备返回重新抓取",
    "voice_material_dropped_B": "料片掉落，准备返回重新抓取",
}


def _diff(label: str, left: dict, right: dict) -> list[str]:
    """Return a readable list of per-key diffs between two asdict() outputs."""
    lines = [f"--- {label} ---"]
    for key in sorted(set(left) | set(right)):
        lv, rv = left.get(key, "<MISSING>"), right.get(key, "<MISSING>")
        if lv != rv:
            lines.append(f"  {key}:\n    yaml={lv!r}\n    gen ={rv!r}")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    yaml_path = Path(argv[1])
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    named_positions = cfg.get("named_positions", {})
    global_agv = cfg.get("agv_config", {})
    default_safe = global_agv.get("default_arm_safe_positions", {})
    default_home = global_agv.get("default_arm_home_positions", {})

    # name -> raw YAML task_dict for the nodes we fold.
    yaml_nodes = {t["name"]: t for t in cfg.get("tasks", [])}

    def parse(task_dict: dict) -> dict:
        return asdict(parse_task_dict(task_dict, named_positions, default_safe, default_home))

    failures: list[tuple[str, dict, dict]] = []
    checks = 0

    def check(yaml_name: str, generated: dict) -> None:
        nonlocal checks
        checks += 1
        if yaml_name not in yaml_nodes:
            failures.append((yaml_name, {}, {"<error>": f"YAML node not found"}))
            return
        left = parse(yaml_nodes[yaml_name])
        right = parse(generated)
        if left != right:
            failures.append((yaml_name, left, right))

    # navigate_to: A / B
    check("agv_to_station_A", navigate_to_task_dict("A"))
    check("agv_to_station_B1", navigate_to_task_dict("B"))

    # pickup: 12 stations + one shared retreat node
    retreat_checked = False
    for station in PICKUP_STATIONS:
        seq, retreat = pickup_task_dict(station)
        check(f"pickup_at_station_{station}", seq)
        if not retreat_checked:
            check("agv_backward_50cm_A1_1", retreat)
            retreat_checked = True

    # place: nudge + sequence
    nudge, place_seq = place_task_dict()
    check("agv_forward_5cm_B1_1", nudge)
    check("place_long_1", place_seq)

    # home: parallel + return-to-A
    parallel, return_to_a = home_task_dict()
    check("home_and_return_A_after_long_1", parallel)
    check("agv_return_from_B1_to_A", return_to_a)

    # speak: simple voice nodes
    for name, msg in SPEAK_NODES.items():
        check(name, speak_task_dict(name, msg))

    if failures:
        print(f"FAIL: {len(failures)}/{checks} checks differ\n")
        for yaml_name, left, right in failures:
            for line in _diff(yaml_name, left, right):
                print(line)
            print()
        return 1

    print(f"OK: {checks} checks bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
