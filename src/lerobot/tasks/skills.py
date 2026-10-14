"""Parameterized skill generators for the Harness-VLA task layer (Phase 1).

Phase 1 scope — **无感 / zero-impact**: reproduce the flat YAML task nodes of
``agv_class_online_test.yaml`` as parameterized generators. This module is
**pure addition** and **default-unreachable**: the flash orchestrator keeps
reading the YAML verbatim, and nothing here is imported by the execution path.
Later phases wire these generators into the LLM tool layer
(``to_tool_spec`` / ``llm_loop`` / ``VisualSuccessDetector`` / ``MemoryManager``).

Insertion point: ``parse_task_dict`` (``tasks/config.py``) resolves
``position: <str>`` against ``named_positions``. The generators only do
f-string / waypoint expansion and emit YAML-node-shaped dicts; the
parser/executor are untouched.

Skill surface (5 generators, from ``pickup-llm-skill-decomposition``):
    navigate_to(dest)      -> long-range AGV nav to station A/B
    pickup(station)        -> 5-step grasp sequence + retreat agv
    place()                -> 6-step place sequence + forward nudge agv
    home()                 -> parallel arm-home+agv-back + return-to-A agv
    speak(msg)             -> TTS system_command
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Spec registry (data layer)
# ---------------------------------------------------------------------------
# Phase 1 keeps a single object; multi-object migrates to YAML later. The
# registry key == classify label space ("long"), so extending objects is adding
# a key + an ObjectSpec with zero new skills (see pickup-llm-skill-decomposition).


@dataclass(frozen=True)
class ObjectSpec:
    """Per-object parameters that parameterize a verb skill.

    ``frozen=True`` freezes the *reference* only — ``place_waypoints`` is still
    a mutable list; treat the registry as read-only by convention.
    """

    key: str  # registry key == classify label, e.g. "long"
    # grasp waypoints: "pickup_station_{station}_{1..5}" template OR explicit list
    grasp_waypoints: str | list[str]
    # place waypoints: explicit list (place step naming is irregular)
    place_waypoints: list[str]
    # Phase 2-3 hook: LLM-adjustable retreat distance (cm). Phase 1 uses the
    # YAML-faithful value baked into the retreat agv node.
    retreat_cm: float = 50.0


SPEC_REGISTRY: dict[str, ObjectSpec] = {
    "long": ObjectSpec(
        key="long",
        grasp_waypoints="pickup_station_{station}_{1..5}",
        place_waypoints=[
            "B_put_long_1",
            "B_put_long_2",
            "B_put_long_3",
            "B_put_long2_3_2",
            "B_put_long2_4",
            "B_put_long2_5",
        ],
        retreat_cm=50.0,
    ),
}


# ---------------------------------------------------------------------------
# Waypoint expansion ("list 为底 + prefix 语法糖")
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"\{(\d+)\.\.(\d+)\}")


def expand_waypoints(template: str | list[str], station: str | None = None) -> list[str]:
    """Expand a grasp-waypoint template into a concrete list of named-position keys.

    * A list is returned unchanged.
    * A string supports two substitutions:
        ``{station}`` -> the station id (e.g. "00"), and
        ``{n..m}``     -> inclusive integer range (``{1..5}`` -> 1..5), repeated
                          against the surrounding prefix/suffix so each index
                          becomes its own waypoint (``pre_{1..5}`` ->
                          ``pre_1..pre_5``).

    ``parse_task_dict`` later resolves each key against ``named_positions``.
    """
    if isinstance(template, list):
        return list(template)

    s = template
    if station is not None:
        s = s.replace("{station}", station)

    match = _RANGE_RE.search(s)
    if match is None:
        return [s]

    lo, hi = int(match.group(1)), int(match.group(2))
    prefix, suffix = s[: match.start()], s[match.end():]
    return [f"{prefix}{i}{suffix}" for i in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# Step definitions (positions injected from waypoints)
# ---------------------------------------------------------------------------

# (step_name, max_duration, overlap_next)
_PICKUP_STEPS: list[tuple[str, float, bool]] = [
    ("手臂摊开", 0.5, True),
    ("手臂下降", 2.0, True),
    ("手臂闭合", 2.0, True),
    ("夹爪闭合", 2.0, True),
    ("手臂抬起", 1.0, True),
]

_PLACE_STEPS: list[tuple[str, float, bool]] = [
    ("step_2", 2.0, True),
    ("step_3", 1.2, True),
    ("step_4", 1.1, True),
    ("step_4_5", 0.7, True),
    ("step_5", 0.2, True),
    ("step_6", 1.1, False),  # last step: no overlap_next
]

_DEST_STATION: dict[str, dict[str, str]] = {
    "A": {"name": "agv_to_station_A", "station": "LM16"},
    "B": {"name": "agv_to_station_B1", "station": "LM2"},
}


def _position_sequence(
    name: str,
    waypoints: list[str],
    step_defs: list[tuple[str, float, bool]],
    max_duration: float,
    next_tasks: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a ``position_sequence`` task_dict (YAML-node shaped)."""
    steps = [
        {
            "name": step_name,
            "position": waypoint,
            "max_duration": duration,
            "position_tolerance": 3.0,
            "overlap_next": overlap,
        }
        for (step_name, duration, overlap), waypoint in zip(step_defs, waypoints)
    ]
    node: dict[str, Any] = {
        "name": name,
        "task_type": "position_sequence",
        "max_duration": max_duration,
        "max_retries": 1,
        "enabled": True,
        "steps": steps,
    }
    if next_tasks is not None:
        node["next_tasks"] = next_tasks
    return node


# ---------------------------------------------------------------------------
# Skill generators (return standard task_dict(s), run through parse_task_dict)
# ---------------------------------------------------------------------------


def navigate_to_task_dict(dest: str) -> dict[str, Any]:
    """Long-range AGV navigation to station A or B.

    Folds ``agv_to_station_A`` / ``agv_to_station_B1``.
    """
    if dest not in _DEST_STATION:
        raise ValueError(f"Unknown dest {dest!r}; expected one of {sorted(_DEST_STATION)}")
    info = _DEST_STATION[dest]
    return {
        "name": info["name"],
        "task_type": "agv",
        "max_duration": 60.0,
        "max_retries": 2,
        "enabled": True,
        "agv_config": {
            "target_station": info["station"],
            "wait_for_arrival": True,
            "arrival_timeout": 60.0,
            "arrival_tolerance": 0.3,
            "check_arm_safe_position": False,
            "retry_on_timeout": True,
            "retry_count": 2,
            "emergency_stop_on_error": True,
        },
    }


def pickup_task_dict(
    station: str, spec_key: str = "long", retreat_cm: float | None = None
) -> list[dict[str, Any]]:
    """Grasp sequence + retreat agv.

    Folds ``pickup_at_station_XX`` (12x) + ``agv_backward_50cm_A1_1``.
    ``retreat_cm`` is a Phase 2-3 hook; Phase 1 defaults to the spec's value
    (50.0 -> translate_dist 0.50), matching the YAML retreat node.
    """
    spec = SPEC_REGISTRY[spec_key]
    if retreat_cm is None:
        retreat_cm = spec.retreat_cm

    waypoints = expand_waypoints(spec.grasp_waypoints, station=station)
    seq = _position_sequence(
        name=f"pickup_at_station_{station}",
        waypoints=waypoints,
        step_defs=_PICKUP_STEPS,
        max_duration=20.0,
        next_tasks={"success": "agv_backward_50cm_A1_1", "failed": "voice_pickup_failed"},
    )
    retreat = {
        "name": "agv_backward_50cm_A1_1",
        "task_type": "agv",
        "max_duration": 30.0,
        "max_retries": 1,
        "enabled": True,
        "agv_config": {
            "translate_dist": retreat_cm / 100.0,
            "translate_vx": -0.1,
            "translate_mode": 0,
            "wait_for_arrival": True,
            "arrival_timeout": 30.0,
            "check_arm_safe_position": False,
            "retry_on_timeout": True,
            "retry_count": 1,
            "emergency_stop_on_error": True,
        },
    }
    return [seq, retreat]


def place_task_dict(spec_key: str = "long") -> list[dict[str, Any]]:
    """Forward nudge agv + 6-step place sequence.

    Folds ``agv_forward_5cm_B1_1`` (进站微调 / nudge) + ``place_long_1``.
    The place *retreat* + return-to-A live in :func:`home_task_dict`, not here.
    """
    spec = SPEC_REGISTRY[spec_key]
    nudge = {
        "name": "agv_forward_5cm_B1_1",
        "task_type": "agv",
        "max_duration": 30.0,
        "max_retries": 1,
        "enabled": True,
        "agv_config": {
            "translate_dist": 0.05,
            "translate_vx": 0.1,
            "translate_mode": 0,
            "wait_for_arrival": True,
            "arrival_timeout": 30.0,
            "check_arm_safe_position": False,
            "retry_on_timeout": True,
            "retry_count": 1,
            "emergency_stop_on_error": True,
            "angle_correction": True,
            "angle_correction_tolerance": 0.03,
        },
    }
    seq = _position_sequence(
        name="place_long_1",
        waypoints=spec.place_waypoints,
        step_defs=_PLACE_STEPS,
        max_duration=16.0,
    )
    return [nudge, seq]


def home_task_dict() -> list[dict[str, Any]]:
    """Arm home (parallel with agv back 20cm) + return-to-A agv.

    Folds ``home_and_return_A_after_long_1`` + ``agv_return_from_B1_to_A``.
    """
    parallel = {
        "name": "home_and_return_A_after_long_1",
        "task_type": "parallel",
        "tasks": [
            {
                "name": "arm_home_after_place_long_1",
                "task_type": "position",
                "max_duration": 10.0,
                "max_retries": 1,
                "enabled": True,
                "completion_criteria": {
                    "type": "position",
                    "position": "home",
                    "position_tolerance": 3.0,
                },
            },
            {
                "name": "agv_backward_20cm_B1_1",
                "task_type": "agv",
                "max_duration": 30.0,
                "max_retries": 1,
                "enabled": True,
                "agv_config": {
                    "translate_dist": 0.20,
                    "translate_vx": -0.1,
                    "translate_mode": 0,
                    "wait_for_arrival": True,
                    "arrival_timeout": 30.0,
                    "check_arm_safe_position": False,
                    "retry_on_timeout": True,
                    "retry_count": 1,
                    "emergency_stop_on_error": True,
                },
            },
        ],
    }
    return_to_a = {
        "name": "agv_return_from_B1_to_A",
        "cycle_end": True,
        "task_type": "agv",
        "max_duration": 60.0,
        "max_retries": 2,
        "enabled": True,
        "agv_config": {
            "target_station": "LM16",
            "wait_for_arrival": True,
            "arrival_timeout": 60.0,
            "arrival_tolerance": 0.3,
            "check_arm_safe_position": False,
            "retry_on_timeout": True,
            "retry_count": 2,
            "emergency_stop_on_error": True,
        },
    }
    return [parallel, return_to_a]


def speak_task_dict(name: str, msg: str) -> dict[str, Any]:
    """TTS system_command node (folds the simple ``voice_*`` / ``timeout_*`` nodes).

    The single-quoted shell-string format is the **existing** ``system_command``
    executor contract (the YAML already hardcodes ``speak.py '<msg>'``); Phase 1
    reproduces it verbatim for bit-identical equivalence, and ``msg`` is a
    hardcoded literal here (no untrusted input).

    ⚠️ Phase 2-3: once ``msg`` becomes LLM-generated, shell-escape it before this
    call site (``shlex.quote(msg)``) or switch the executor to an argv list —
    do NOT pass untrusted text through this string unescaped.
    """
    return {
        "name": name,
        "task_type": "system_command",
        "command": f"python /home/t/workspace/dc_dir/tts_cache/speak.py '{msg}'",
    }
