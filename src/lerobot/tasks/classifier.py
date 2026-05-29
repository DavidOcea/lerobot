"""Pluggable workpiece classifier for conditional task branching.

The orchestrator only reads ClassifyResult.next_task for routing —
it doesn't care how the classification was made.  New classifier
implementations just subclass BaseClassifier and register in the
factory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from lerobot.agent.visual_align import detect_marker, _get_detector
from lerobot.tasks.config import VisualAlignConfig


@dataclass
class ClassifyResult:
    """Output of a classifier — only next_task matters for routing."""

    label: str = ""
    confidence: float = 0.0
    next_task: str = ""  # orchestrator reads this for branch routing


class BaseClassifier(ABC):
    """Abstract interface for workpiece classifiers."""

    @abstractmethod
    def classify(self, image: np.ndarray) -> ClassifyResult:
        """Return classification result from a single image."""
        ...

    def reset(self):
        """Reset internal state (called at start of each classify task)."""
        pass


class AprilTagClassifier(BaseClassifier):
    """Classify workpiece by reading an AprilTag ID in the image.

    Maps detected tag IDs to labels via a configurable tag_map.
    """

    def __init__(self, marker_id_map: dict[int, str], marker_family: str = "tag36h11",
                 marker_size: float = 0.05, default_label: str = "",
                 default_next_task: str = ""):
        self.marker_id_map = marker_id_map
        self.marker_family = marker_family
        self.marker_size = marker_size
        self.default_label = default_label
        self.default_next_task = default_next_task
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = _get_detector(self.marker_family)
        return self._detector

    def classify(self, image: np.ndarray) -> ClassifyResult:
        # Build a minimal config for detect_marker (only marker_id=None
        # to scan all visible IDs; marker_size / family matter for pose
        # but not for ID-only detection).
        config = VisualAlignConfig(
            marker_id=None,
            marker_size=self.marker_size,
            marker_family=self.marker_family,
        )
        marker = detect_marker(image, config, self.detector)
        if marker is None:
            return ClassifyResult(label="unknown", confidence=0.0)

        tag_id = marker["id"]
        label = self.marker_id_map.get(tag_id, self.default_label)
        confidence = 0.95  # tag detected → high confidence

        return ClassifyResult(label=label, confidence=confidence)

    def reset(self):
        self._detector = None


def make_classifier(method: str, **kwargs) -> BaseClassifier:
    """Factory to create a classifier from a method name.

    Supported methods:
      - "apriltag": AprilTagClassifier
    """
    method = method.lower()
    if method == "apriltag":
        return AprilTagClassifier(
            marker_id_map=kwargs.get("marker_id_map", {}),
            marker_family=kwargs.get("marker_family", "tag36h11"),
            marker_size=kwargs.get("marker_size", 0.05),
            default_label=kwargs.get("default_label", "unknown"),
            default_next_task=kwargs.get("default_next_task", ""),
        )
    raise ValueError(f"Unknown classify method: {method}")
