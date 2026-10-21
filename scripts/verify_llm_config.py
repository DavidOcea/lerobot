#!/usr/bin/env python
"""Verify Phase 3/4 config→runtime wiring offline (no hardware/network).

``verify_llm_planner.py`` injects a ``FakePlanner`` directly and
``verify_visual_detector.py`` builds a ``SimpleNamespace`` config by hand;
neither exercises the real seam where a YAML file becomes the orchestrator
config. This script closes that gap: it loads the real flash fixture and its
LLM variant through the same ``load_config_from_yaml`` the orchestrator uses,
and asserts the three opt-in keys land in the right fields.

The three keys must be YAML **top-level** (the hand-rolled parser in
``tasks/config.py`` reads ``config_dict.get("planner"/"llm"/"visual_detector")``,
never nested under an ``agent:`` block) — this is the exact gotcha that has
already bitten once.

Sections:
  1. flash-default — base fixture parses with planner="flash",
     visual_detector_config.enabled=False, LLMConfig at defaults (无感 invariant
     at the config layer), and run()'s dispatch expression is False.
  2. llm-optin     — llm fixture parses with planner="llm", the LLMConfig
     fields populated from the top-level ``llm:`` block, visual_detector
     enabled with the right camera_key, and dispatch expression True.
  3. planner       — build_planner("api", llm_config) constructs an
     OpenAIChatPlanner without touching the network; missing llm_config and
     unknown planner_type each raise ValueError.

Exit 0 = all green, 1 = any failure. Run with the lerobot env python.
"""

from __future__ import annotations

import sys

from lerobot.agent.planner import OpenAIChatPlanner, build_planner
from lerobot.tasks.config import load_config_from_yaml

FLASH = "/root/workspace/dc_dir/agv_class_ADC2.yaml"
LLM = "/root/workspace/dc_dir/lerobot/configs/agv_class_longloushang_yanshi_V0.6.yaml"


def _dispatch_expr(config) -> bool:
    """The exact expression ``run()`` uses to pick the LLM path (orchestrator.py)."""
    return getattr(config, "planner", "flash") == "llm"


def section_flash(fail: list[str]) -> None:
    c = load_config_from_yaml(FLASH)
    if c.planner != "flash":
        fail.append(f"flash: planner={c.planner!r}, expected 'flash'")
    if c.visual_detector_config.enabled is not False:
        fail.append("flash: visual_detector_config.enabled should default to False")
    if c.llm_config.model != "gpt-4o-mini":
        fail.append(f"flash: llm_config.model={c.llm_config.model!r}, expected default")
    if _dispatch_expr(c) is not False:
        fail.append("flash: run() dispatch expression should be False")


def section_llm(fail: list[str]) -> None:
    c = load_config_from_yaml(LLM)
    if c.planner != "llm":
        fail.append(f"llm: planner={c.planner!r}, expected 'llm'")
    lc = c.llm_config
    if lc.model != "fake-model" or lc.base_url != "http://127.0.0.1:9999/v1" or lc.max_tool_calls != 6:
        fail.append(f"llm: llm_config not parsed from top-level llm: block -> {lc}")
    vd = c.visual_detector_config
    if vd.enabled is not True or vd.camera_key != "head_cam":
        fail.append(f"llm: visual_detector_config not parsed -> {vd}")
    if _dispatch_expr(c) is not True:
        fail.append("llm: run() dispatch expression should be True")


def section_planner(fail: list[str]) -> None:
    c = load_config_from_yaml(LLM)
    p = build_planner("api", c.llm_config)
    if not isinstance(p, OpenAIChatPlanner):
        fail.append(f"planner: build_planner('api') -> {type(p).__name__}, expected OpenAIChatPlanner")

    try:
        build_planner("api", None)
        fail.append("planner: build_planner('api', None) should raise ValueError")
    except ValueError:
        pass

    try:
        build_planner("nope", c.llm_config)
        fail.append("planner: unknown planner_type should raise ValueError")
    except ValueError:
        pass


def main() -> int:
    fail: list[str] = []
    section_flash(fail)
    section_llm(fail)
    section_planner(fail)
    if fail:
        print("FAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("OK: Phase 3/4 config wiring verified (3 sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
