"""
Task execution memory for cross-cycle parameter memorization.

Stores per-task execution results in a JSON file so later runs can
warm-start with parameters known to work well (e.g., gain schedules,
convergence timing) instead of starting from defaults every time.

Each ``record()`` call **accumulates** statistics across cycles while
keeping the latest trace at the top level.  ``lookup()`` returns the
latest trace dict (backward-compatible with existing warm-start logic);
``stats()`` exposes aggregated data for introspection.

Design principles:
- One JSON file per orchestrator instance (path from config).
- One record per task_name (string key).  Top-level keys are the latest
  trace fields; aggregated statistics live under a reserved ``_stats`` key.
- Corrupt or missing file → silently treated as empty (graceful degradation).
- All I/O wrapped in try/except — this module MUST NOT crash the orchestrator.
- Fully gated behind ``TaskMemoryConfig.enabled``; when False, the store is
  never constructed.

Usage::

    from lerobot.tasks.task_memory import TaskMemoryConfig, TaskMemoryStore

    cfg = TaskMemoryConfig(enabled=True, store_path="/tmp/.task_memory.json")
    store = TaskMemoryStore(cfg)
    store.record("align_station_A", {"phase1_iterations": 2, "gain_sequence": [0.8, 0.6]})
    data = store.lookup("align_station_A")
    st  = store.stats("align_station_A")
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Reserved key for aggregated statistics — must not clash with trace field names.
_RESERVED_STATS_KEY: str = "_stats"


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class TaskMemoryConfig:
    """Configuration for the cross-task parameter memory system.

    When ``enabled=True``, stores per-task execution results (e.g.,
    visual_align convergence data) in a JSON file so downstream tasks
    or later cycles can adapt.  When ``enabled=False``, zero code paths
    run beyond the single boolean check.
    """

    enabled: bool = False
    store_path: str = ""  # "" = auto: ~/.lerobot/task_memory.json


# ── Store ─────────────────────────────────────────────────────────────────


class TaskMemoryStore:
    """Thread-safe JSON-backed store for cross-task parameter memory.

    All I/O is **best-effort**: a corrupt file, missing directory, or
    write error is caught, logged, and silently recovered — the store
    is never the source of a crash.

    Atomic writes via tempfile + rename so that a crash during write
    never produces a partially-written JSON file.

    Parameters
    ----------
    config : TaskMemoryConfig
        Configuration with store_path.
    """

    def __init__(self, config: TaskMemoryConfig) -> None:
        self._config = config
        self._filepath = self._resolve_path(config.store_path)
        self._data: dict[str, Any] = {}
        self._load()

    # ── public API ────────────────────────────────────────────────────

    def lookup(self, task_name: str, default: Any = None) -> Any:
        """Return the **latest trace** for *task_name*, or *default* if not found.

        The reserved ``_stats`` key is stripped from the returned value so
        callers see the same shape they saw before cumulative merge was added.

        Returns
        -------
        Any
            The stored value (typically a dict), or *default*.
        """
        entry = self._data.get(task_name, default)
        if isinstance(entry, dict) and _RESERVED_STATS_KEY in entry:
            # Return a shallow copy without _stats so warm-start code
            # is unaffected by the new aggregation layer.
            return {k: v for k, v in entry.items() if k != _RESERVED_STATS_KEY}
        return entry

    def stats(self, task_name: str) -> dict[str, Any]:
        """Return aggregated statistics for *task_name*, or empty dict.

        Returns
        -------
        dict
            Keys may include ``success_count``, ``failure_count``,
            ``failure_modes``, ``phase1_iters_avg``, ``label_counts``,
            ``best_gain_schedule``, ``last_success``, ``last_run``, etc.
        """
        entry = self._data.get(task_name)
        if isinstance(entry, dict):
            return dict(entry.get(_RESERVED_STATS_KEY) or {})
        return {}

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Return ``{task_name: _stats dict}`` for every recorded task.

        The orchestrator's read-only diagnostics loop uses this to inspect
        every task's running averages without reaching into ``_data``.
        Non-dict entries (stored verbatim) are skipped.
        """
        out: dict[str, dict[str, Any]] = {}
        for name, entry in self._data.items():
            if isinstance(entry, dict):
                out[name] = dict(entry.get(_RESERVED_STATS_KEY) or {})
        return out

    def record(self, task_name: str, trace: dict[str, Any]) -> None:
        """Store *trace* under *task_name* and persist to disk.

        This is a **cumulative merge**: the top-level fields of *trace*
        replace the previous snapshot, but aggregated statistics (cycle
        count, running averages, failure modes, label frequency, etc.)
        accumulate across calls.

        Parameters
        ----------
        task_name : str
            Unique task key (e.g. ``"align_station_B1"``).
        trace : dict
            Arbitrary JSON-serializable data captured during execution.
        """
        self._merge(task_name, trace)
        try:
            self._save()
        except Exception:
            logger.debug(
                "TaskMemory: failed to persist record for '%s' (non-fatal)",
                task_name,
                exc_info=True,
            )

    def flush(self) -> None:
        """Explicitly persist in-memory state to JSON (normally auto on record)."""
        self._save()

    def close(self) -> None:
        """Flush and clear in-memory state."""
        self._save()
        self._data.clear()

    # ── internal ───────────────────────────────────────────────────────

    def _resolve_path(self, store_path: str) -> Path:
        if store_path:
            return Path(store_path)
        default = Path.home() / ".lerobot" / "task_memory.json"
        return default

    def _load(self) -> None:
        """Load from JSON file. Silently uses empty dict on any error."""
        try:
            if self._filepath.exists():
                with open(self._filepath, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._data = data
                    logger.debug(
                        "TaskMemory: loaded %d entries from %s",
                        len(self._data),
                        self._filepath,
                    )
                else:
                    logger.warning(
                        "TaskMemory: %s is not a dict (found %s), starting fresh",
                        self._filepath,
                        type(data).__name__,
                    )
                    self._data = {}
            else:
                logger.debug("TaskMemory: no existing file at %s, starting fresh", self._filepath)
        except json.JSONDecodeError:
            logger.warning(
                "TaskMemory: corrupt JSON in %s, starting fresh (old file preserved as .bak)",
            )
            try:
                bak = self._filepath.with_suffix(".json.bak")
                self._filepath.rename(bak)
            except Exception:
                pass
            self._data = {}
        except Exception:
            logger.debug("TaskMemory: could not read %s, starting fresh", self._filepath, exc_info=True)
            self._data = {}

    def _save(self) -> None:
        """Persist in-memory data to JSON file.

        Uses tempfile + rename for atomicity: a crash during write will
        never leave a partially-written file.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._filepath.parent,
            prefix=".task_memory_",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        )
        try:
            json.dump(self._data, tmp, indent=2, ensure_ascii=False, default=str)
            tmp.flush()
            tmp.close()
            Path(tmp.name).replace(self._filepath)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise

    # ── cumulative merge ───────────────────────────────────────────────

    def _merge(self, task_name: str, trace: Any) -> None:
        """Accumulate *trace* into the per-task entry.

        Top-level trace fields replace the previous snapshot (backward-compat
        for warm-start consumers).  A reserved ``_stats`` sub-dict accumulates
        cycle count, per-task-type running averages, failure-mode counts,
        label frequencies, and a best-gain-schedule heuristic.

        Non-dict traces (lists, strings, ints) are stored as-is — no
        statistics are accumulated because there is no structured schema
        to extract fields from.
        """
        # ── non-dict traces: store verbatim (backward compat) ─────
        if not isinstance(trace, dict):
            self._data[task_name] = trace
            return

        old_entry = self._data.get(task_name)
        old = old_entry if isinstance(old_entry, dict) else {}
        old_stats = old.get(_RESERVED_STATS_KEY)
        prev_stats: dict[str, Any] = (
            dict(old_stats) if isinstance(old_stats, dict) else {}
        )

        # ── bump counters ────────────────────────────────────────────
        total = prev_stats.get("total_count", 0) + 1
        prev_ok = prev_stats.get("success_count", 0)
        prev_fail = prev_stats.get("failure_count", 0)

        is_success = trace.get("success", True) is not False
        if is_success:
            prev_ok += 1
        else:
            prev_fail += 1

        new_stats: dict[str, Any] = {
            "total_count": total,
            "success_count": prev_ok,
            "failure_count": prev_fail,
            "last_run": time.time(),
        }

        # ── failure-mode tracking ────────────────────────────────────
        failure_modes: dict[str, int] = dict(
            prev_stats.get("failure_modes") or {}
        )
        error = trace.get("error", "")
        if not is_success and error:
            # Coarse mode key: up to first colon or newline
            mode_key = error.split(":")[0].strip()[:80]
            failure_modes[mode_key] = failure_modes.get(mode_key, 0) + 1
        new_stats["failure_modes"] = failure_modes

        # ── per-task-type accumulators ───────────────────────────────
        task_type = trace.get("task_type", "")

        if task_type == "visual_align":
            self._acc_va_success_stats(new_stats, prev_stats, trace)
        elif task_type == "classify":
            self._acc_classify_stats(new_stats, prev_stats, trace)

        # ── assemble entry ───────────────────────────────────────────
        new_entry = {k: v for k, v in trace.items() if k != _RESERVED_STATS_KEY}
        new_entry[_RESERVED_STATS_KEY] = new_stats
        self._data[task_name] = new_entry

    # ── per-type stat helpers ─────────────────────────────────────────

    def _acc_va_success_stats(
        self,
        new_stats: dict[str, Any],
        prev_stats: dict[str, Any],
        trace: dict[str, Any],
    ) -> None:
        """Accumulate visual_align-specific running averages and best gains."""
        p1 = trace.get("phase1") or {}
        p1_iters: int = p1.get("iterations", 0)
        search = trace.get("search") or {}
        search_attempts: int = search.get("attempts", 0)

        if trace.get("success", True) is not False and p1_iters > 0:
            total_ok = new_stats["success_count"]

            # ── exponential-decay running averages (0.7 old / 0.3 new) ─
            old_p1 = prev_stats.get("phase1_iters_avg", float(p1_iters))
            new_stats["phase1_iters_avg"] = round(
                old_p1 * 0.7 + p1_iters * 0.3, 2,
            )
            old_search = prev_stats.get("search_attempts_avg", float(search_attempts))
            new_stats["search_attempts_avg"] = round(
                old_search * 0.7 + search_attempts * 0.3, 2,
            )

            # ── duration average ──────────────────────────────────────
            prev_dur = prev_stats.get("total_exec_time_s_avg", 0.0)
            this_dur = trace.get("duration_s", 0.0)
            new_stats["total_exec_time_s_avg"] = round(
                (prev_dur * (total_ok - 1) + this_dur) / total_ok, 2,
            )

            # ── best gain schedule (fewest phase1 iterations) ─────────
            prev_best_iters = prev_stats.get("best_phase1_iters", float("inf"))
            prev_best_gains = prev_stats.get("best_gain_schedule")
            if p1_iters < prev_best_iters and p1.get("gain_sequence"):
                new_stats["best_phase1_iters"] = p1_iters
                new_stats["best_gain_schedule"] = list(p1["gain_sequence"])
            else:
                new_stats["best_phase1_iters"] = prev_best_iters
                if prev_best_gains is not None:
                    new_stats["best_gain_schedule"] = prev_best_gains

            # ── oscillation rate ──────────────────────────────────────
            osc_count: int = prev_stats.get("oscillation_count", 0)
            if p1.get("oscillation_detected") is True:
                osc_count += 1
            new_stats["oscillation_count"] = osc_count
            new_stats["oscillation_rate"] = round(
                osc_count / total_ok, 3,
            ) if total_ok > 0 else 0.0

            new_stats["last_success"] = time.time()
        else:
            # Carry forward existing averages on failure / zero-iter
            for k in (
                "phase1_iters_avg", "search_attempts_avg",
                "total_exec_time_s_avg", "best_phase1_iters",
                "best_gain_schedule", "oscillation_count",
                "oscillation_rate", "last_success",
            ):
                if k in prev_stats:
                    new_stats[k] = prev_stats[k]

    def _acc_classify_stats(
        self,
        new_stats: dict[str, Any],
        prev_stats: dict[str, Any],
        trace: dict[str, Any],
    ) -> None:
        """Accumulate classify-specific running averages and label frequency."""
        label = trace.get("label")
        if label and trace.get("success", True) is not False:
            total_ok = new_stats["success_count"]

            # ── label frequency ──────────────────────────────────────
            label_counts: dict[str, int] = dict(
                prev_stats.get("label_counts") or {}
            )
            label_counts[label] = label_counts.get(label, 0) + 1
            new_stats["label_counts"] = label_counts

            # ── running average confidence ───────────────────────────
            conf = trace.get("confidence", 0.0)
            prev_c = prev_stats.get("confidence_avg", 0.0)
            new_stats["confidence_avg"] = round(
                (prev_c * (total_ok - 1) + conf) / total_ok, 3,
            )

            # ── duration average ──────────────────────────────────────
            prev_d = prev_stats.get("duration_s_avg", 0.0)
            new_stats["duration_s_avg"] = round(
                (prev_d * (total_ok - 1) + trace.get("duration_s", 0.0))
                / total_ok, 2,
            )

            # ── auto-align oscillation rate ───────────────────────────
            aa = trace.get("auto_align")
            if aa is not None:
                aa_osc: int = prev_stats.get("auto_align_oscillation_count", 0)
                if aa.get("oscillation_detected") is True:
                    aa_osc += 1
                new_stats["auto_align_oscillation_count"] = aa_osc

            new_stats["last_success"] = time.time()
        else:
            # Carry forward on failure
            for k in (
                "label_counts", "confidence_avg", "duration_s_avg",
                "auto_align_oscillation_count", "last_success",
            ):
                if k in prev_stats:
                    new_stats[k] = prev_stats[k]
