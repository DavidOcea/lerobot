"""Pluggable workpiece classifier for conditional task branching.

The orchestrator only reads ClassifyResult.next_task for routing —
it doesn't care how the classification was made.  New classifier
implementations just subclass BaseClassifier and register in the
factory.

Supported methods:
  - "apriltag":     AprilTagClassifier
  - "yolo_detect":  YOLOClassifier (ONNX), no marker needed
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


class YOLOClassifier(BaseClassifier):
    """Classify workpiece using a trained YOLOv8 ONNX model.

    No markers needed — the model directly predicts class from the image.
    Falls back gracefully to default_label if no detection or ONNX missing.
    """

    def __init__(self, model_path: str, classes: list[str] | None = None,
                 default_label: str = "unknown", default_next_task: str = "",
                 conf_threshold: float = 0.15):
        self.model_path = model_path
        self.classes = classes or ["short", "long"]
        self.default_label = default_label
        self.default_next_task = default_next_task
        self.conf_threshold = conf_threshold
        self._session = None
        self._available = None  # None=tried yet, True/False after first attempt

    def _init_session(self):
        if self._session is not None:
            return True
        if self._available is False:
            return False
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path)
            self._available = True
            return True
        except ImportError:
            self._available = False
            print("[YOLOClassifier] onnxruntime not installed — pip install onnxruntime")
            return False
        except Exception as e:
            self._available = False
            print(f"[YOLOClassifier] Failed to load {self.model_path}: {e}")
            return False

    def classify(self, image: np.ndarray) -> ClassifyResult:
        if not self._init_session():
            return ClassifyResult(label=self.default_label, confidence=0.0)

        h0, w0 = image.shape[:2]
        # Preprocess: BGR→RGB (copy needed — cv2.resize rejects negative strides)
        img = image[:, :, ::-1].copy()
        img = cv2.resize(img, (640, 640)).transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        outputs = self._session.run(None, {"images": img})
        preds = outputs[0][0]  # [4+nc, 8400]

        boxes_xywh, scores, cls_ids = [], [], []
        nc = len(self.classes)
        for i in range(preds.shape[1]):
            cx, cy, bw, bh = preds[0:4, i]
            cls_conf = preds[4:4 + nc, i]
            max_conf = float(cls_conf.max())
            if max_conf < self.conf_threshold:
                continue
            cls_id = int(cls_conf.argmax())
            x1 = (cx - bw / 2) / 640 * w0
            y1 = (cy - bh / 2) / 640 * h0
            x2 = x1 + bw / 640 * w0
            y2 = y1 + bh / 640 * h0
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(max_conf)
            cls_ids.append(cls_id)

        if not boxes_xywh:
            return ClassifyResult(label="no_detection", confidence=0.0)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf_threshold, 0.45)
        if len(indices) == 0:
            return ClassifyResult(label="no_detection", confidence=0.0)

        best_idx = max(indices.flatten(), key=lambda i: scores[i])
        best_cls = cls_ids[best_idx]
        best_conf = scores[best_idx]
        label = self.classes[best_cls] if best_cls < len(self.classes) else f"cls{best_cls}"

        if __debug__ or True:  # always log first few detections for debugging
            print(f"[YOLOClassifier] {len(boxes_xywh)} raw → {len(indices)} after NMS → "
                  f"{label} (conf={best_conf:.3f})")

        return ClassifyResult(label=label, confidence=float(best_conf))

    def reset(self):
        pass  # stateless


import cv2  # needed for resize, cvtColor in YOLOClassifier.classify


def make_classifier(method: str, **kwargs) -> BaseClassifier:
    """Factory to create a classifier from a method name.

    Supported methods:
      - "apriltag":     AprilTagClassifier (marker-based)
      - "yolo_detect":  YOLOClassifier (ONNX, no marker needed)
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
    if method == "yolo_detect":
        return YOLOClassifier(
            model_path=kwargs.get("model_path", ""),
            classes=kwargs.get("classes", ["short", "long"]),
            default_label=kwargs.get("default_label", "unknown"),
            default_next_task=kwargs.get("default_next_task", ""),
            conf_threshold=kwargs.get("conf_threshold", 0.15),
        )
    raise ValueError(f"Unknown classify method: {method}")
