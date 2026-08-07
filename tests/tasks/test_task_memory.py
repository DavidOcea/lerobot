"""Tests for the TaskMemoryStore — JSON-backed cross-task parameter memory.

All tests use tmp_path for file isolation — no real filesystem side effects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.tasks.task_memory import TaskMemoryConfig, TaskMemoryStore


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def enabled_cfg(tmp_path: Path) -> TaskMemoryConfig:
    """Config with task memory enabled and a temp file path."""
    return TaskMemoryConfig(
        enabled=True,
        store_path=str(tmp_path / ".task_memory.json"),
    )


@pytest.fixture
def disabled_cfg() -> TaskMemoryConfig:
    """Config with task memory disabled (default)."""
    return TaskMemoryConfig(enabled=False)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_store_init_empty(enabled_cfg: TaskMemoryConfig) -> None:
    """New store with no existing file starts with empty state."""
    store = TaskMemoryStore(enabled_cfg)
    assert store.lookup("any_task") is None
    assert store.lookup("any_task", default={}) == {}


def test_store_record_and_lookup(enabled_cfg: TaskMemoryConfig) -> None:
    """record() stores data; lookup() retrieves it."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {"gain_sequence": [0.8, 0.6], "iterations": 5}

    store.record("align_station_A", trace)
    result = store.lookup("align_station_A")

    assert result == trace
    assert result["gain_sequence"] == [0.8, 0.6]
    assert result["iterations"] == 5


def test_store_file_persistence(enabled_cfg: TaskMemoryConfig) -> None:
    """Data written by one store instance is readable by another at the same path."""
    trace = {"phase1_iterations": 12, "mode": "reference"}

    # Write via first instance
    s1 = TaskMemoryStore(enabled_cfg)
    s1.record("align_station_B", trace)
    s1.close()

    # Read via second instance (same path)
    s2 = TaskMemoryStore(enabled_cfg)
    result = s2.lookup("align_station_B")
    assert result == trace

    # Verify the file actually exists and contains valid JSON
    raw = json.loads(Path(enabled_cfg.store_path).read_text())
    assert raw["align_station_B"] == trace


def test_store_corrupt_json(tmp_path: Path) -> None:
    """Corrupt JSON file → store starts fresh (no crash, no exception)."""
    bad_path = tmp_path / "corrupt_task_memory.json"
    bad_path.write_text("this is not valid json {{{")

    cfg = TaskMemoryConfig(enabled=True, store_path=str(bad_path))
    store = TaskMemoryStore(cfg)  # Must not raise

    # Should start fresh — lookup returns None
    assert store.lookup("any_task") is None

    # Should be able to write and read back normally
    store.record("recovery_task", {"ok": True})
    assert store.lookup("recovery_task") == {"ok": True}

    # Corrupt file should have been backed up
    bak_files = list(tmp_path.glob("*.bak"))
    assert len(bak_files) >= 1, "Corrupt file should be renamed to .bak"


def test_store_key_isolation(enabled_cfg: TaskMemoryConfig) -> None:
    """Different task names get independent entries."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("task_A", {"a": 1})
    store.record("task_B", {"b": 2})

    assert store.lookup("task_A") == {"a": 1}
    assert store.lookup("task_B") == {"b": 2}
    assert store.lookup("task_C") is None


def test_store_overwrite_same_key(enabled_cfg: TaskMemoryConfig) -> None:
    """Recording the same task name again overwrites the previous entry."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("task_X", {"version": 1})
    store.record("task_X", {"version": 2})

    assert store.lookup("task_X") == {"version": 2}


def test_store_flush_and_close(enabled_cfg: TaskMemoryConfig) -> None:
    """flush() and close() persist data to disk and clear memory."""
    store = TaskMemoryStore(enabled_cfg)
    store.record("flush_test", {"data": "present"})

    # flush() writes to disk but keeps in-memory
    store.flush()
    assert store.lookup("flush_test") == {"data": "present"}

    # Verify on-disk
    raw = json.loads(Path(enabled_cfg.store_path).read_text())
    assert raw["flush_test"] == {"data": "present"}

    # close() clears in-memory state
    store.close()
    assert store.lookup("flush_test") is None


def test_store_default_value(enabled_cfg: TaskMemoryConfig) -> None:
    """lookup with explicit default returns default when key is missing."""
    store = TaskMemoryStore(enabled_cfg)

    assert store.lookup("missing") is None
    assert store.lookup("missing", default=[]) == []
    assert store.lookup("missing", default={"fallback": True}) == {"fallback": True}


def test_store_non_dict_value(enabled_cfg: TaskMemoryConfig) -> None:
    """Store supports any JSON-serializable value, not just dicts."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("list_task", [1, 2, 3])
    store.record("string_task", "just a string")
    store.record("int_task", 42)

    assert store.lookup("list_task") == [1, 2, 3]
    assert store.lookup("string_task") == "just a string"
    assert store.lookup("int_task") == 42


def test_store_empty_trace(enabled_cfg: TaskMemoryConfig) -> None:
    """Empty dict trace is stored and retrieved correctly."""
    store = TaskMemoryStore(enabled_cfg)
    store.record("empty_task", {})
    assert store.lookup("empty_task") == {}


# ── Warm-start logic tests ───────────────────────────────────────────────────
# These tests validate the decision-making patterns used by _memory_warmstart()
# in the orchestrator.  They operate on trace dicts stored via TaskMemoryStore
# and exercise the same conditional logic that drives warm-start decisions.


def _should_reuse_gains(trace: dict) -> bool:
    """Replicate _memory_warmstart visual-align fast-converge logic."""
    p1 = trace.get("phase1", {})
    return (
        p1.get("converged") is True
        and p1.get("iterations", 99) <= 2
        and p1.get("oscillation_detected") is False
        and bool(p1.get("gain_sequence"))
    )


def _should_use_conservative(trace: dict) -> bool:
    """Replicate _memory_warmstart visual-align oscillation logic."""
    p1 = trace.get("phase1", {})
    return p1.get("oscillation_detected") is True


def _should_cache_classify(trace: dict, retry_labels: set | None = None) -> bool:
    """Replicate _memory_warmstart classify cache logic."""
    if trace.get("task_type") != "classify":
        return False
    label = trace.get("label")
    confidence = trace.get("confidence", 0.0)
    if label is None or confidence < 0.85:
        return False
    retry_set = retry_labels or set()
    if label in retry_set or label == "unknown":
        return False
    return True


def _should_damp_auto_align(trace: dict) -> bool:
    """Replicate _memory_warmstart auto-align oscillation logic."""
    aa = trace.get("auto_align")
    return aa is not None and aa.get("oscillation_detected") is True


def test_warmstart_va_fast_converge(enabled_cfg: TaskMemoryConfig) -> None:
    """Trace with 2 iterations, no oscillation → gain sequence reused."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "visual_align",
        "phase1": {
            "converged": True,
            "iterations": 2,
            "oscillation_detected": False,
            "gain_sequence": [0.8, 0.6],
            "mode": "reference",
        },
        "search": {"attempts": 1, "total_turned_deg": 0.0},
    }
    assert _should_reuse_gains(trace) is True
    assert _should_use_conservative(trace) is False


def test_warmstart_va_oscillation(enabled_cfg: TaskMemoryConfig) -> None:
    """Trace with oscillation → conservative gains applied."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "visual_align",
        "phase1": {
            "converged": True,
            "iterations": 4,
            "oscillation_detected": True,
            "gain_sequence": [0.8, 0.6, 0.5, 0.4],
            "mode": "approach",
        },
        "search": {"attempts": 2, "total_turned_deg": 20.0},
    }
    assert _should_reuse_gains(trace) is False   # oscillated → no reuse
    assert _should_use_conservative(trace) is True


def test_warmstart_va_slow_converge_no_reuse(enabled_cfg: TaskMemoryConfig) -> None:
    """Slow convergence (3+ iterations) → NOT eligible for gain reuse."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "visual_align",
        "phase1": {
            "converged": True,
            "iterations": 3,
            "oscillation_detected": False,
            "gain_sequence": [0.8, 0.6, 0.5],
            "mode": "reference",
        },
        "search": {"attempts": 1, "total_turned_deg": 0.0},
    }
    assert _should_reuse_gains(trace) is False
    assert _should_use_conservative(trace) is False


def test_warmstart_va_not_converged(enabled_cfg: TaskMemoryConfig) -> None:
    """Non-converged trace → NOT eligible for any warm-start."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "visual_align",
        "phase1": {
            "converged": False,
            "iterations": 3,
            "oscillation_detected": False,
            "gain_sequence": [0.8, 0.6, 0.5],
            "mode": "approach",
        },
        "search": {"attempts": 9, "total_turned_deg": 90.0},
    }
    assert _should_reuse_gains(trace) is False
    assert _should_use_conservative(trace) is False


def test_warmstart_classify_cache(enabled_cfg: TaskMemoryConfig) -> None:
    """Classify trace with confidence 0.93 → eligible for cache."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "classify",
        "method": "yolo_roi",
        "label": "long_01",
        "confidence": 0.93,
        "duration_s": 1.5,
        "next_task": "pickup_station_A",
    }
    assert _should_cache_classify(trace) is True


def test_warmstart_classify_low_confidence(enabled_cfg: TaskMemoryConfig) -> None:
    """Classify trace with confidence 0.70 → NOT eligible for cache."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "classify",
        "method": "yolo_roi",
        "label": "long_01",
        "confidence": 0.70,
        "duration_s": 2.0,
    }
    assert _should_cache_classify(trace) is False


def test_warmstart_classify_retry_label(enabled_cfg: TaskMemoryConfig) -> None:
    """Classify trace with retry label ('no_detection') → NOT cached."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "classify",
        "method": "apriltag",
        "label": "no_detection",
        "confidence": 1.0,
        "duration_s": 0.5,
    }
    assert _should_cache_classify(trace, retry_labels={"no_detection"}) is False


def test_warmstart_auto_align_oscillation(enabled_cfg: TaskMemoryConfig) -> None:
    """Auto-align trace with oscillation → step damping triggered."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "classify",
        "method": "yolo_roi",
        "label": "long_02",
        "confidence": 0.90,
        "auto_align": {
            "ran": True,
            "attempts": 3,
            "converged": True,
            "oscillation_detected": True,
            "final_label": "long_02",
        },
    }
    assert _should_damp_auto_align(trace) is True


def test_warmstart_auto_align_clean(enabled_cfg: TaskMemoryConfig) -> None:
    """Auto-align trace without oscillation → NO damping."""
    store = TaskMemoryStore(enabled_cfg)
    trace = {
        "task_type": "classify",
        "method": "yolo_roi",
        "label": "long_01",
        "confidence": 0.95,
        "auto_align": {
            "ran": True,
            "attempts": 2,
            "converged": True,
            "oscillation_detected": False,
            "final_label": "long_01",
        },
    }
    assert _should_damp_auto_align(trace) is False


def test_warmstart_mixed_traces_isolation(enabled_cfg: TaskMemoryConfig) -> None:
    """Different task types have independent warm-start decisions."""
    store = TaskMemoryStore(enabled_cfg)
    va_trace = {
        "task_type": "visual_align",
        "phase1": {"converged": True, "iterations": 2,
                   "oscillation_detected": False, "gain_sequence": [0.8, 0.6]},
    }
    cls_trace = {
        "task_type": "classify",
        "label": "short_01", "confidence": 0.92,
    }

    # Visual-align gets gain reuse
    assert _should_reuse_gains(va_trace) is True
    # Classify gets cached
    assert _should_cache_classify(cls_trace) is True
    # Classify trace should NOT trigger VA reuse (by type check)
    assert cls_trace.get("task_type") != "visual_align"
    # VA trace should NOT trigger classify cache
    assert va_trace.get("task_type") != "classify"
