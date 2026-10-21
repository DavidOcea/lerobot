"""Phase 4: evidence-only visual success detection (RGB baseline differencing).

Pickup-only for now: success = "workpiece disappeared" = a significant pixel
change in the station ROI between the frame captured immediately BEFORE the
grasp node and the frame captured AFTER. Verdicts are advisory — they enrich
``skill_observation`` for the LLM and never alter ``result.success`` /
execution flow.

This module is imported lazily from ``llm_loop`` (only when
``visual_detector_config.enabled`` is True), so the default flash path never
loads it. It imports ``cv2``/``numpy`` freely here rather than at module top of
the orchestrator, keeping the 无感 (zero-impact) boundary clean.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VisualVerdict:
    """Evidence-only conclusion from one before/after frame pair.

    ``direction`` is advisory metadata for the LLM (``"decrease"`` = pickup,
    ``"increase"`` = place): the baseline-diff ``success`` gate is direction-
    agnostic (``changed_fraction >= diff_threshold``), because the empty-vs-
    material gray shift sign is not known a priori.
    """

    success: bool = False  # changed_fraction >= diff_threshold
    confidence: float = 0.0  # 0..1, scaled from changed_fraction vs threshold
    changed_fraction: float = 0.0  # fraction of ROI pixels whose |Δgray| > pixel_threshold
    direction: str = "decrease"  # "decrease" (pickup) | "increase" (place, future)
    camera_key: str = "head_cam"
    roi: tuple[int, int, int, int] | None = None  # (x, y, w, h) px; None = full frame
    elapsed: float = 0.0  # seconds spent in detect()
    details: dict[str, Any] = field(default_factory=dict)


class VisualSuccessDetector(abc.ABC):
    """Interface mirroring ``TaskCompletionDetector.detect(observation, history)``.

    Concrete detectors implement ``detect(before, after, camera_key)``, taking
    two ``np.ndarray`` frames (RGB ``uint8`` HxWx3) and returning a
    ``VisualVerdict``. Implementations must never raise — a detection failure
    is represented by a verdict with ``success=False`` and empty ``details``.
    """

    @abc.abstractmethod
    def detect(
        self, before: np.ndarray, after: np.ndarray, camera_key: str = "head_cam"
    ) -> VisualVerdict:
        ...


class BaselineDiffDetector(VisualSuccessDetector):
    """``cv2.absdiff`` grayscale baseline differencing (no stored reference, no depth)."""

    def __init__(self, config: "VisualDetectorConfig"):
        self.config = config

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray | None:
        """Coerce an RGB/RGBA/gray frame to single-channel uint8 gray, else None."""
        if img is None:
            return None
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.ndim == 3 and img.shape[2] == 1:
            return img[:, :, 0]
        return None

    def detect(
        self, before: np.ndarray, after: np.ndarray, camera_key: str = "head_cam"
    ) -> VisualVerdict:
        t0 = time.time()
        verdict = VisualVerdict(
            camera_key=camera_key,
            direction=self.config.direction,
            roi=tuple(self.config.roi) if self.config.roi is not None else None,
        )

        if before is None or after is None:
            return verdict

        before_gray = self._to_gray(before)
        after_gray = self._to_gray(after)
        if before_gray is None or after_gray is None:
            return verdict

        # Apply ROI (config.roi may arrive as a list from YAML; unpack handles both).
        roi = self.config.roi
        roi_applied = None
        if roi is not None:
            x, y, w, h = roi
            H, W = before_gray.shape
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                return verdict  # empty/degenerate ROI → no evidence
            before_gray = before_gray[y0:y1, x0:x1]
            after_gray = after_gray[y0:y1, x0:x1]
            roi_applied = [x0, y0, x1 - x0, y1 - y0]

        if before_gray.shape != after_gray.shape:
            return verdict  # mismatched frames → no evidence

        delta = cv2.absdiff(before_gray, after_gray)
        changed = delta > self.config.pixel_threshold
        changed_fraction = float(np.mean(changed))

        verdict.changed_fraction = changed_fraction
        verdict.success = changed_fraction >= self.config.diff_threshold
        verdict.confidence = (
            min(1.0, changed_fraction / (self.config.diff_threshold * 2.0))
            if self.config.diff_threshold > 0
            else 0.0
        )
        verdict.details = {
            "mean_before": float(before_gray.mean()),
            "mean_after": float(after_gray.mean()),
            "mean_delta": float(delta.mean()),
            "roi_applied": roi_applied,
            "shape": list(before_gray.shape),
        }
        verdict.elapsed = time.time() - t0
        return verdict


def build_visual_detector(config: "VisualDetectorConfig") -> VisualSuccessDetector:
    """Factory: config.method → detector. Raises ``ValueError`` on unknown method."""
    if config.method == "baseline_diff":
        return BaselineDiffDetector(config)
    raise ValueError(f"unknown visual detector method: {config.method!r}")


def capture_frame(robot: Any, camera_key: str) -> np.ndarray | None:
    """Return the latest RGB frame from ``robot.get_observation()["images"][camera_key]``.

    Returns ``None`` (never raises) on any failure / missing image / missing
    robot, so the detector degrades to "no evidence" instead of failing the run.
    """
    if robot is None:
        return None
    try:
        obs = robot.get_observation()
        images = obs.get("images") if isinstance(obs, dict) else {}
        frame = images.get(camera_key)
        return frame if frame is not None else None
    except Exception:  # noqa: BLE001 — evidence-only; a missing frame is "no verdict"
        logger.debug("capture_frame failed for %s", camera_key, exc_info=True)
        return None
