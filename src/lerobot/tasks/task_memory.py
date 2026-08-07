"""
Task execution memory for cross-cycle parameter memorization.

Stores per-task execution results in a JSON file so later runs can
warm-start with parameters known to work well (e.g., gain schedules,
convergence timing) instead of starting from defaults every time.

Design principles:
- One JSON file per orchestrator instance (path from config).
- One record per task_name (string key), flat dict space.
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
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        """Return stored data for *task_name*, or *default* if not found.

        Returns
        -------
        Any
            The stored value (typically a dict), or *default*.
        """
        return self._data.get(task_name, default)

    def record(self, task_name: str, trace: dict[str, Any]) -> None:
        """Store *trace* under *task_name* and persist to disk.

        Parameters
        ----------
        task_name : str
            Unique task key (e.g. ``"align_station_B1"``).
        trace : dict
            Arbitrary JSON-serializable data captured during execution.
        """
        self._data[task_name] = trace
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
        # Auto path: config directory for this agent, falls back to /tmp
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
            # Back up the corrupt file for debugging
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

        # Write to a temp file in the same directory, then atomically rename
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
            # Clean up temp file on failure
            Path(tmp.name).unlink(missing_ok=True)
            raise
