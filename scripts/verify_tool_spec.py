#!/usr/bin/env python
"""Verify ``to_tool_spec()`` matches the B1 tool-boundary contract.

Checks the 5 skill tools expose exactly the visible parameters (no hidden
fields: station/nudge/waypoints/gripper/align/primitive), the three-piece return
schema, and the per-tool failure_mode sets. Exit 0/1.
"""

from __future__ import annotations

import sys

from lerobot.tasks.failure_mode import FAILURE_RECOVERY_TABLE, FailureMode
from lerobot.tasks.skills import SPEC_REGISTRY
from lerobot.tasks.tool_spec import to_tool_spec

EXPECTED_TOOLS = ["navigate_to", "pickup", "place", "speak", "home"]

# Visible params per tool (exactly these — nothing hidden).
EXPECTED_PARAMS = {
    "navigate_to": {"dest"},
    "pickup": {"spec", "retreat_cm"},
    "place": {"spec", "retreat_cm"},
    "speak": {"msg"},
    "home": set(),
}

# Hidden-field blacklist: must not appear in any tool's parameters.
HIDDEN_FIELDS = {"station", "nudge", "waypoints", "gripper", "align", "primitive", "agv", "classify"}

EXPECTED_FAILURE_MODES = {
    "navigate_to": {"not_arrived", "navigation_stuck", "timeout", "dropped"},
    "pickup": {"empty_grasp", "grasp_slip", "dropped"},
    "place": {"misplace", "topple", "not_released"},
    "speak": set(),
    "home": set(),
}


def main() -> int:
    specs = to_tool_spec()
    failures: list[str] = []

    if set(specs) != set(EXPECTED_TOOLS):
        failures.append(f"tools {sorted(specs)} != {EXPECTED_TOOLS}")

    for name in EXPECTED_TOOLS:
        if name not in specs:
            failures.append(f"missing tool {name}")
            continue
        spec = specs[name]
        props = set(spec.parameters.get("properties", {}))

        if props != EXPECTED_PARAMS[name]:
            failures.append(f"{name}: visible params {sorted(props)} != {sorted(EXPECTED_PARAMS[name])}")
        hidden_hits = props & HIDDEN_FIELDS
        if hidden_hits:
            failures.append(f"{name}: hidden fields leaked {sorted(hidden_hits)}")

        ret_props = set(spec.returns.get("properties", {}))
        if ret_props != {"status", "failure_mode", "evidence"}:
            failures.append(f"{name}: returns {sorted(ret_props)} != three-piece")

        if set(spec.failure_modes) != EXPECTED_FAILURE_MODES[name]:
            failures.append(f"{name}: failure_modes {set(spec.failure_modes)} != {EXPECTED_FAILURE_MODES[name]}")

    # Retreat defaults: pickup sourced from spec, place literal 30.0 (A4).
    pickup_retreat = specs["pickup"].parameters["properties"]["retreat_cm"]["default"]
    if pickup_retreat != SPEC_REGISTRY["long"].retreat_cm:
        failures.append(f"pickup retreat default {pickup_retreat} != spec {SPEC_REGISTRY['long'].retreat_cm}")
    place_retreat = specs["place"].parameters["properties"]["retreat_cm"]["default"]
    if place_retreat != 30.0:
        failures.append(f"place retreat default {place_retreat} != 30.0")

    # Every declared failure_mode resolves to a known enum value.
    valid = {m.value for m in FailureMode}
    for spec in specs.values():
        for fm in spec.failure_modes:
            if fm not in valid:
                failures.append(f"{spec.name}: unknown failure_mode {fm!r}")

    # A2 table is total over the enum.
    missing_modes = [m.value for m in FailureMode if m not in FAILURE_RECOVERY_TABLE]
    if missing_modes:
        failures.append(f"FAILURE_RECOVERY_TABLE missing {missing_modes}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: {len(EXPECTED_TOOLS)} tool specs match B1 contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
