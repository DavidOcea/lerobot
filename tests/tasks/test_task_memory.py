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

    # Verify the file actually exists and contains valid JSON with stats
    raw = json.loads(Path(enabled_cfg.store_path).read_text())
    entry = raw["align_station_B"]
    for k, v in trace.items():
        assert entry.get(k) == v
    assert "_stats" in entry
    assert entry["_stats"]["total_count"] >= 1


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

    # Verify on-disk (contains stats + trace fields)
    raw = json.loads(Path(enabled_cfg.store_path).read_text())
    assert raw["flush_test"]["data"] == "present"
    assert "_stats" in raw["flush_test"]

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


# ── Cumulative merge tests ───────────────────────────────────────────────
# These validate the new _merge() accumulation logic while ensuring
# lookup() stays backward-compatible with the warm-start tests above.


def test_cumulative_success_count(enabled_cfg: TaskMemoryConfig) -> None:
    """Record same task 3 times → success_count=3, total_count=3."""
    store = TaskMemoryStore(enabled_cfg)
    for _ in range(3):
        store.record("va_station", {
            "task_type": "visual_align",
            "phase1": {"iterations": 2, "converged": True},
            "search": {"attempts": 1},
        })

    st = store.stats("va_station")
    assert st["total_count"] == 3
    assert st["success_count"] == 3
    assert st["failure_count"] == 0

    # lookup() should still return latest trace without _stats
    latest = store.lookup("va_station")
    assert "task_type" in latest
    assert "_stats" not in latest


def test_cumulative_failure_modes(enabled_cfg: TaskMemoryConfig) -> None:
    """Failures with different error messages cluster into failure_modes."""
    store = TaskMemoryStore(enabled_cfg)
    store.record("va_station", {
        "task_type": "visual_align",
        "success": False,
        "error": "Classification failed after retries",
    })
    store.record("va_station", {
        "task_type": "visual_align",
        "success": False,
        "error": "Classification failed after retries",
    })
    store.record("va_station", {
        "task_type": "visual_align",
        "success": False,
        "error": "Tag lost: no detection for 5 frames",
    })
    store.record("va_station", {
        "task_type": "visual_align",
        "success": True,
        "phase1": {"iterations": 2, "converged": True},
    })

    st = store.stats("va_station")
    assert st["total_count"] == 4
    assert st["success_count"] == 1
    assert st["failure_count"] == 3
    assert st["failure_modes"] == {
        "Classification failed after retries": 2,
        "Tag lost": 1,
    }


def test_cumulative_best_gain_schedule(enabled_cfg: TaskMemoryConfig) -> None:
    """Track gain schedule from the run with fewest phase1 iterations."""
    store = TaskMemoryStore(enabled_cfg)

    # First run: 3 iterations
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 3, "converged": True,
            "gain_sequence": [0.8, 0.6, 0.5],
        },
        "search": {"attempts": 1},
    })
    assert store.stats("va_station")["best_gain_schedule"] == [0.8, 0.6, 0.5]

    # Second run: 2 iterations (better — should replace)
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 2, "converged": True,
            "gain_sequence": [0.7, 0.5],
        },
        "search": {"attempts": 1},
    })
    assert store.stats("va_station")["best_gain_schedule"] == [0.7, 0.5]

    # Third run: 4 iterations (worse — keep [0.7, 0.5])
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 4, "converged": True,
            "gain_sequence": [0.9, 0.8, 0.7, 0.6],
        },
        "search": {"attempts": 2},
    })
    assert store.stats("va_station")["best_gain_schedule"] == [0.7, 0.5]


def test_cumulative_running_averages(enabled_cfg: TaskMemoryConfig) -> None:
    """phase1_iters_avg uses 0.7/0.3 exponential decay."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 2, "converged": True,
            "gain_sequence": [0.8, 0.6],
        },
        "search": {"attempts": 1},
    })
    assert store.stats("va_station")["phase1_iters_avg"] == 2.0

    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 4, "converged": True,
            "gain_sequence": [0.8, 0.6, 0.5, 0.4],
        },
        "search": {"attempts": 2},
    })
    # EMA: 2.0*0.7 + 4*0.3 = 1.4 + 1.2 = 2.6
    assert store.stats("va_station")["phase1_iters_avg"] == 2.6

    # search_attempts_avg: 1.0*0.7 + 2*0.3 = 1.3
    assert store.stats("va_station")["search_attempts_avg"] == 1.3


def test_cumulative_oscillation_rate(enabled_cfg: TaskMemoryConfig) -> None:
    """Oscillation rate tracks proportion of runs with oscillation."""
    store = TaskMemoryStore(enabled_cfg)

    # Run 1: no oscillation
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 2, "converged": True,
            "oscillation_detected": False,
            "gain_sequence": [0.8, 0.6],
        },
        "search": {"attempts": 1},
    })
    assert store.stats("va_station")["oscillation_rate"] == 0.0

    # Run 2: oscillation
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 4, "converged": True,
            "oscillation_detected": True,
            "gain_sequence": [0.8, 0.6, 0.5, 0.4],
        },
        "search": {"attempts": 2},
    })
    assert store.stats("va_station")["oscillation_rate"] == 0.5

    # Run 3: no oscillation
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 2, "converged": True,
            "oscillation_detected": False,
            "gain_sequence": [0.8, 0.6],
        },
        "search": {"attempts": 1},
    })
    assert store.stats("va_station")["oscillation_rate"] == round(1 / 3, 3)


def test_cumulative_label_counts(enabled_cfg: TaskMemoryConfig) -> None:
    """Classify trace accumulates per-label counts."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("classify_workpiece", {
        "task_type": "classify",
        "label": "long_00", "confidence": 0.93, "duration_s": 1.5,
    })
    store.record("classify_workpiece", {
        "task_type": "classify",
        "label": "long_00", "confidence": 0.95, "duration_s": 1.2,
    })
    store.record("classify_workpiece", {
        "task_type": "classify",
        "label": "long_01", "confidence": 0.88, "duration_s": 1.8,
    })

    lc = store.stats("classify_workpiece")["label_counts"]
    assert lc == {"long_00": 2, "long_01": 1}


def test_cumulative_confidence_avg(enabled_cfg: TaskMemoryConfig) -> None:
    """Classify confidence_avg is running arithmetic mean."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("classify_workpiece", {
        "task_type": "classify",
        "label": "long_00", "confidence": 0.90, "duration_s": 1.0,
    })
    assert store.stats("classify_workpiece")["confidence_avg"] == 0.90

    store.record("classify_workpiece", {
        "task_type": "classify",
        "label": "long_00", "confidence": 0.80, "duration_s": 1.0,
    })
    assert store.stats("classify_workpiece")["confidence_avg"] == 0.85

    store.record("classify_workpiece", {
        "task_type": "classify",
        "label": "long_01", "confidence": 1.0, "duration_s": 1.0,
    })
    assert store.stats("classify_workpiece")["confidence_avg"] == 0.90


def test_cumulative_auto_align_oscillation(enabled_cfg: TaskMemoryConfig) -> None:
    """Classify with auto_align tracks oscillation count."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("classify_wp", {
        "task_type": "classify",
        "label": "long_00", "confidence": 0.90,
        "auto_align": {"ran": True, "oscillation_detected": True},
    })
    store.record("classify_wp", {
        "task_type": "classify",
        "label": "long_01", "confidence": 0.92,
        "auto_align": {"ran": True, "oscillation_detected": False},
    })
    store.record("classify_wp", {
        "task_type": "classify",
        "label": "long_00", "confidence": 0.95,
        "auto_align": {"ran": True, "oscillation_detected": True},
    })

    assert store.stats("classify_wp")["auto_align_oscillation_count"] == 2


def test_cumulative_backward_compat_lookup(enabled_cfg: TaskMemoryConfig) -> None:
    """lookup() returns trace without _stats — existing warm-start code unchanged."""
    store = TaskMemoryStore(enabled_cfg)

    trace = {
        "task_type": "visual_align",
        "phase1": {
            "converged": True, "iterations": 2,
            "oscillation_detected": False, "gain_sequence": [0.8, 0.6],
        },
        "search": {"attempts": 1, "total_turned_deg": 0.0},
    }
    store.record("va_station", trace)
    assert store.lookup("va_station") == trace
    assert "_stats" not in store.lookup("va_station")

    st = store.stats("va_station")
    assert st["total_count"] == 1
    assert st["success_count"] == 1


def test_cumulative_non_dict_trace(enabled_cfg: TaskMemoryConfig) -> None:
    """Non-dict traces stored as-is, no stats, no crash."""
    store = TaskMemoryStore(enabled_cfg)

    store.record("list_task", [1, 2, 3])
    store.record("str_task", "hello")
    store.record("int_task", 42)

    assert store.lookup("list_task") == [1, 2, 3]
    assert store.lookup("str_task") == "hello"
    assert store.lookup("int_task") == 42
    assert store.stats("list_task") == {}


def test_cumulative_existing_stats_migration(enabled_cfg: TaskMemoryConfig) -> None:
    """File with _stats from a previous session is migrated, not discarded."""
    store = TaskMemoryStore(enabled_cfg)

    # Simulate an old-format file with _stats already present
    store._data["va_station"] = {
        "task_type": "visual_align",
        "phase1": {"iterations": 2, "converged": True},
        "_stats": {
            "total_count": 5,
            "success_count": 4,
            "failure_count": 1,
            "phase1_iters_avg": 2.3,
        },
    }
    store._save()

    # Record a new trace — should migrate prev_stats
    store.record("va_station", {
        "task_type": "visual_align",
        "phase1": {
            "iterations": 1, "converged": True,
            "gain_sequence": [0.7],
        },
        "search": {"attempts": 1},
    })

    st = store.stats("va_station")
    assert st["total_count"] == 6  # 5 + 1
    assert st["success_count"] == 5  # 4 + 1
    assert st["failure_count"] == 1
    assert st["best_gain_schedule"] == [0.7]


def test_cumulative_empty_trace(enabled_cfg: TaskMemoryConfig) -> None:
    """Empty dict trace creates minimal stats without crashing."""
    store = TaskMemoryStore(enabled_cfg)
    store.record("empty_task", {})

    st = store.stats("empty_task")
    assert st["total_count"] == 1
    assert st["success_count"] == 1  # success=True is the default
    assert st["failure_count"] == 0
    assert st["failure_modes"] == {}


# ── Idea 2: Global Memory failure recovery tests ─────────────────────────
# These validate the failure-mode classification and recovery-action
# dispatch logic for the orchestrator's _maybe_recovery_before_retry hook.


def _make_fake_orch(store: TaskMemoryStore):
    """Stub orchestrator-like object with just enough API for recovery tests."""
    class Fake:
        pass
    o = Fake()
    o.task_memory = store
    return o


def _classify_failure_mode(error_message: str | None) -> str:
    """Replicate orchestrator._classify_failure_mode for unit testing.

    Extracts the failure signature by splitting on the first colon.
    """
    if not error_message:
        return ""
    return error_message.split(":")[0].strip()[:80]


def test_recovery_classify_failure_mode() -> None:
    """Error messages are parsed to coarse signatures matching _stats keys."""
    assert _classify_failure_mode(None) == ""
    assert _classify_failure_mode("") == ""
    assert _classify_failure_mode("Tag lost: no detection for 5 frames") == "Tag lost"
    assert _classify_failure_mode("Classification failed after retries") == "Classification failed after retries"
    assert _classify_failure_mode("No workpiece detected after max attempts") == "No workpiece detected after max attempts"


def _recovery_for_failure(
    store: TaskMemoryStore, task_name: str, error_message: str | None,
    rules: dict | None = None,
) -> dict | None:
    """Replicate orchestrator._recovery_for_failure logic."""
    if rules is None:
        rules = {
            "Tag lost": (3, {"action": "skip_retries", "reason": "tag_lost_persistent"}),
            "Classification failed after retries": (3, {"action": "skip_retries", "reason": "classify_exhausted"}),
        }
    if store is None or not error_message:
        return None
    mode_key = _classify_failure_mode(error_message)
    if not mode_key:
        return None

    rule = rules.get(mode_key)
    if rule is None:
        for prefix, r in rules.items():
            if mode_key.startswith(prefix):
                rule = r
                break
    if rule is None:
        return None

    min_count, action = rule
    stats = store.stats(task_name)
    failure_modes = stats.get("failure_modes") or {}
    current_count = failure_modes.get(mode_key, 0)
    if current_count >= min_count:
        return action
    return None


def test_recovery_not_triggered_below_threshold(enabled_cfg: TaskMemoryConfig) -> None:
    """Recovery does NOT fire when failure count < threshold (3)."""
    store = TaskMemoryStore(enabled_cfg)
    store.record("va_station", {
        "task_type": "visual_align",
        "success": False,
        "error": "Tag lost: no detection for 5 frames",
    })
    store.record("va_station", {
        "task_type": "visual_align",
        "success": False,
        "error": "Tag lost: no detection for 5 frames",
    })
    # Only 2 failures → below threshold of 3
    result = _recovery_for_failure(store, "va_station", "Tag lost: no detection")
    assert result is None


def test_recovery_triggered_at_threshold(enabled_cfg: TaskMemoryConfig) -> None:
    """Recovery fires when failure count reaches threshold (3)."""
    store = TaskMemoryStore(enabled_cfg)
    for _ in range(3):
        store.record("va_station", {
            "task_type": "visual_align",
            "success": False,
            "error": "Tag lost: no detection for 5 frames",
        })

    result = _recovery_for_failure(store, "va_station", "Tag lost: no detection")
    assert result == {"action": "skip_retries", "reason": "tag_lost_persistent"}


def test_recovery_different_error_keys_isolated(enabled_cfg: TaskMemoryConfig) -> None:
    """Different failure modes have independent counts."""
    store = TaskMemoryStore(enabled_cfg)
    # 3 "Tag lost" failures
    for _ in range(3):
        store.record("va_station", {
            "task_type": "visual_align",
            "success": False,
            "error": "Tag lost: no detection for 5 frames",
        })
    # Only 1 "Classification failed"
    store.record("va_station", {
        "task_type": "classify",
        "success": False,
        "error": "Classification failed after retries",
    })

    # Tag lost → triggered
    assert _recovery_for_failure(store, "va_station", "Tag lost: ...") is not None
    # Classification → below threshold
    assert _recovery_for_failure(store, "va_station", "Classification failed after retries") is None


def test_recovery_skip_retries_action(enabled_cfg: TaskMemoryConfig) -> None:
    """Recovery action skip_retries is recognised."""
    import dataclasses

    @dataclasses.dataclass
    class FakeTask:
        name: str = "test_task"

    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = None

    fake_task = FakeTask()

    # skip_retries action → True (should stop retries)
    result = orch._execute_recovery_action(
        {"action": "skip_retries", "reason": "tag_lost_persistent"},
        fake_task,
    )
    assert result is True

    # Unknown action → False
    result = orch._execute_recovery_action(
        {"action": "unknown_future_action"},
        fake_task,
    )
    assert result is False


# ── Explore / Deploy dual-mode tests (Idea 5) ──────────────────────────────


def _build_fake_task(
    name: str = "test_task",
    task_type: str = "visual_align",
    max_retries: int = 1,
    pos_tol: float = 0.02,
    angle_tol: float = 2.0,
    max_iter: int = 3,
):
    """Build a lightweight fake TaskConfig for explore/deploy tests."""
    import dataclasses

    @dataclasses.dataclass
    class FakeVisualAlignConfig:
        position_tolerance: float = 0.02
        angle_tolerance: float = 2.0
        max_iterations: int = 3
        warm_gain_sequence: list | None = None

    @dataclasses.dataclass
    class FakeTaskConfig:
        name: str = "test_task"
        task_type: str = "visual_align"
        max_retries: int = 1
        visual_align_config: object | None = None

    va = (
        FakeVisualAlignConfig(
            position_tolerance=pos_tol,
            angle_tolerance=angle_tol,
            max_iterations=max_iter,
        )
        if task_type == "visual_align"
        else None
    )
    return FakeTaskConfig(
        name=name,
        task_type=task_type,
        max_retries=max_retries,
        visual_align_config=va,
    )


def test_explore_deploy_disabled_gate() -> None:
    """When task_memory is None, _apply_explore_deploy_mode returns None."""
    import dataclasses

    @dataclasses.dataclass
    class FakeTask:
        name: str = "any_task"

    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = None

    task = FakeTask()
    result = orch._apply_explore_deploy_mode(task)
    assert result is None


def test_explore_deploy_initial_state_no_stats(enabled_cfg: TaskMemoryConfig) -> None:
    """When no prior record exists → success_count=0 → explore mode active."""
    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    store = TaskMemoryStore(enabled_cfg)
    # No records at all — store is empty

    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = store

    task = _build_fake_task(name="new_station_A", task_type="visual_align")
    original_max_retries = task.max_retries

    restore = orch._apply_explore_deploy_mode(task)

    # Explore: max_retries boosted
    assert task.max_retries >= 5
    assert task.visual_align_config.position_tolerance == 0.04  # doubled
    assert task.visual_align_config.angle_tolerance == 4.0  # doubled
    assert task.visual_align_config.max_iterations >= 5

    # Restore dict holds originals
    assert restore is not None
    assert restore["va_position_tolerance"] == 0.02
    assert restore["va_angle_tolerance"] == 2.0
    assert restore["va_max_iterations"] == 3

    # ── Simulate restore ───────────────────────────────────────
    va = task.visual_align_config
    va.position_tolerance = restore["va_position_tolerance"]
    va.angle_tolerance = restore["va_angle_tolerance"]
    va.max_iterations = restore["va_max_iterations"]
    task.max_retries = original_max_retries

    assert va.position_tolerance == 0.02
    assert va.angle_tolerance == 2.0
    assert va.max_iterations == 3


def test_explore_deploy_after_three_successes(enabled_cfg: TaskMemoryConfig) -> None:
    """After 3 successes → deploy mode → no parameter changes."""
    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    store = TaskMemoryStore(enabled_cfg)

    # Accumulate 3 successful runs
    for _ in range(3):
        store.record("station_B", {
            "task_type": "visual_align",
            "success": True,
            "phase1": {"converged": True, "iterations": 2, "gain_sequence": [0.7, 0.5]},
            "duration_s": 18.0,
        })

    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = store

    task = _build_fake_task(name="station_B", task_type="visual_align")
    original_pos_tol = task.visual_align_config.position_tolerance
    original_angle_tol = task.visual_align_config.angle_tolerance
    original_max_iter = task.visual_align_config.max_iterations
    original_max_retries = task.max_retries

    restore = orch._apply_explore_deploy_mode(task)

    # Deploy mode: no overrides
    assert task.visual_align_config.position_tolerance == original_pos_tol
    assert task.visual_align_config.angle_tolerance == original_angle_tol
    assert task.visual_align_config.max_iterations == original_max_iter
    assert task.max_retries == original_max_retries

    # Restore dict exists but has all None values (nothing overridden)
    assert restore is not None
    assert restore["va_position_tolerance"] is None
    assert restore["va_angle_tolerance"] is None
    assert restore["va_max_iterations"] is None


def test_explore_deploy_exactly_at_threshold(enabled_cfg: TaskMemoryConfig) -> None:
    """success_count == 3 is deploy mode (>= is the gate, not >)."""
    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    store = TaskMemoryStore(enabled_cfg)

    # Precisely 3 successful runs
    for i in range(3):
        store.record("station_C", {
            "task_type": "visual_align",
            "success": True,
            "phase1": {"converged": True, "iterations": 1, "gain_sequence": [0.6]},
            "duration_s": 15.0,
        })

    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = store

    task = _build_fake_task(name="station_C", task_type="visual_align")
    original_tol = task.visual_align_config.position_tolerance

    orch._apply_explore_deploy_mode(task)

    # Exactly 3 → deploy, no tolerance change
    assert task.visual_align_config.position_tolerance == original_tol


def test_explore_deploy_below_threshold_still_explore(enabled_cfg: TaskMemoryConfig) -> None:
    """2 successes → still explore mode (threshold is 3)."""
    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    store = TaskMemoryStore(enabled_cfg)

    for _ in range(2):
        store.record("station_D", {
            "task_type": "visual_align",
            "success": True,
            "phase1": {"converged": True, "iterations": 2, "gain_sequence": [0.7, 0.5]},
            "duration_s": 18.0,
        })

    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = store

    task = _build_fake_task(name="station_D", task_type="visual_align")

    restore = orch._apply_explore_deploy_mode(task)

    # 2 successes → still explore → tolerances doubled
    assert task.visual_align_config.position_tolerance == 0.04
    assert restore["va_position_tolerance"] == 0.02


def test_explore_deploy_max_retries_already_high(enabled_cfg: TaskMemoryConfig) -> None:
    """When max_retries is already ≥ 5, explore mode doesn't lower it."""
    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    store = TaskMemoryStore(enabled_cfg)
    # No prior records → explore

    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = store

    task = _build_fake_task(
        name="station_E", task_type="visual_align", max_retries=7,
    )

    orch._apply_explore_deploy_mode(task)

    # max_retries should stay at 7 (not lowered to 5)
    assert task.max_retries == 7


def test_explore_deploy_non_visual_align_task(enabled_cfg: TaskMemoryConfig) -> None:
    """Explore mode still boosts max_retries even without visual_align_config."""
    from lerobot.agent.orchestrator import TaskAgentOrchestrator
    store = TaskMemoryStore(enabled_cfg)

    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.task_memory = store

    # agv task — no visual_align_config
    task = _build_fake_task(name="navigate_home", task_type="agv", max_retries=2)

    restore = orch._apply_explore_deploy_mode(task)

    # Still boosts retries
    assert task.max_retries >= 5
    # Restore exists but VA fields are None (no VA config to override)
    assert restore is not None
    assert restore["va_position_tolerance"] is None
    assert restore["va_angle_tolerance"] is None
