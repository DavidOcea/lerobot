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
from typing import Optional

import numpy as np

from lerobot.agent.visual_align import detect_marker, _get_detector
from lerobot.tasks.config import VisualAlignConfig

# ── global label counter (resets per cycle, used for alternating placements) ──

_LABEL_COUNTERS: dict[str, int] = {}


def reset_classify_counters(keywords: Optional[list[str]] = None):
    """Reset label counters at the start of each cycle.

    Called by the orchestrator before each cycle begins.
    If keywords is None, resets ALL counters.
    If keywords is a list, only resets those specific counters.
    """
    global _LABEL_COUNTERS
    if keywords is None:
        _LABEL_COUNTERS.clear()
    else:
        for kw in keywords:
            _LABEL_COUNTERS.pop(kw, None)


def _counted_label(label: str, counter_keywords: list[str], modulo: int = 0) -> str:
    """Append sequence number to label if it matches a counted keyword.

    e.g. "long" → "long_1", then "long_2", etc.
    If modulo > 0, wraps after reaching modulo: 1→2→...→modulo→1→...
    Counter resets when orchestrator calls reset_classify_counters().
    """
    for kw in counter_keywords:
        if kw == label:
            global _LABEL_COUNTERS
            _LABEL_COUNTERS[label] = _LABEL_COUNTERS.get(label, 0) + 1
            seq = _LABEL_COUNTERS[label]
            if modulo > 0:
                seq = ((seq - 1) % modulo) + 1
            return f"{label}_{seq}"
    return label


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
        self.classes = classes or ["short", "long", "box"]
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


class RoiIouClassifier(YOLOClassifier):
    """YOLO + ROI-IoU classifier for workpiece position disambiguation.

    Extends YOLOClassifier: runs detection first, then computes IoU of
    the detected bounding-box against pre-defined ROI regions loaded
    from a JSON reference file (created by capture_roi_regions.py).

    The region with the highest IoU wins. If no region exceeds
    iou_threshold, returns default_label (for voice prompt / fallback).
    """

    def __init__(self, model_path: str, classes=None,
                 default_label="unknown", default_next_task="",
                 conf_threshold=0.15,
                 roi_reference_path="", iou_threshold=0.3):
        super().__init__(model_path=model_path, classes=classes,
                         default_label=default_label,
                         default_next_task=default_next_task,
                         conf_threshold=conf_threshold)
        self.roi_reference_path = roi_reference_path
        self.iou_threshold = iou_threshold
        self._roi_regions = {}
        self._roi_boxes: list[tuple[str, tuple, float]] = []
        self._roi_loaded = False
        self._boundary_box = None          # {"x","y","w","h"} — workspace outer boundary
        self._out_of_bounds_label = "unknown"  # label returned when bbox exceeds boundary

    def _load_rois(self):
        if self._roi_loaded:
            return
        if not self.roi_reference_path:
            self._roi_loaded = True
            return
        import json
        try:
            with open(self.roi_reference_path) as f:
                data = json.load(f)
            self._roi_regions = data.get("regions", {})
            for label, r in self._roi_regions.items():
                if label == "_boundary":
                    self._boundary_box = (r["x"], r["y"], r["w"], r["h"])
                    continue
                threshold = r.get("iou_threshold", self.iou_threshold)
                self._roi_boxes.append((label, (r["x"], r["y"], r["w"], r["h"]), threshold))
            if self._boundary_box:
                print(f"[RoiIouClassifier] Boundary box: {self._boundary_box}")
            print(f"[RoiIouClassifier] {len(self._roi_boxes)} ROIs from {self.roi_reference_path}")
        except Exception as e:
            print(f"[RoiIouClassifier] Failed to load ROIs: {e}")
        self._roi_loaded = True

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, aw, ah = a
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_contained(inner, outer):
        """True if inner bbox is fully inside outer bbox."""
        ix1, iy1, iw, ih = inner
        ix2, iy2 = ix1 + iw, iy1 + ih
        ox1, oy1, ow, oh = outer
        ox2, oy2 = ox1 + ow, oy1 + oh
        return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

    @staticmethod
    def _det_to_px_bbox(x1, y1, x2, y2):
        return (max(0, int(x1)), max(0, int(y1)), int(x2 - x1), int(y2 - y1))

    def classify(self, image):
        yolo_result = super().classify(image)
        if yolo_result.label in ("no_detection", "unknown", self.default_label):
            return yolo_result

        self._load_rois()
        if not self._roi_boxes:
            return yolo_result

        if not self._init_session():
            return ClassifyResult(label=self.default_label, confidence=0.0)

        h0, w0 = image.shape[:2]
        img = image[:, :, ::-1].copy()
        img = cv2.resize(img, (640, 640)).transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        outputs = self._session.run(None, {"images": img})
        preds = outputs[0][0]
        nc = len(self.classes)

        boxes_xywh, scores = [], []
        for i in range(preds.shape[1]):
            cx, cy, bw, bh = preds[0:4, i]
            cc = preds[4:4 + nc, i]
            mc = float(cc.max())
            if mc < self.conf_threshold:
                continue
            x1 = (cx - bw / 2) / 640 * w0
            y1 = (cy - bh / 2) / 640 * h0
            x2 = x1 + bw / 640 * w0
            y2 = y1 + bh / 640 * h0
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(mc)

        if not boxes_xywh:
            return ClassifyResult(label=self.default_label, confidence=0.0)

        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf_threshold, 0.45)
        if len(idxs) == 0:
            return ClassifyResult(label=self.default_label, confidence=0.0)

        best = max(idxs.flatten(), key=lambda i: scores[i])
        det_px = self._det_to_px_bbox(
            boxes_xywh[best][0], boxes_xywh[best][1],
            boxes_xywh[best][0] + boxes_xywh[best][2],
            boxes_xywh[best][1] + boxes_xywh[best][3],
        )

        # ── Boundary check: reject if any part of bbox is outside workspace ─
        if self._boundary_box is not None:
            bx, by, bw, bh = det_px
            if not self._bbox_contained(det_px, self._boundary_box):
                print(f"[RoiIou] bbox partially outside boundary → {self.default_label}")
                return ClassifyResult(label=self.default_label, confidence=0.0)

        best_label = self.default_label
        best_iou = 0.0
        second_iou = 0.0
        best_threshold = self.iou_threshold
        for label, roi_box, threshold in self._roi_boxes:
            iou = self._bbox_iou(det_px, roi_box)
            if iou > best_iou:
                second_iou = best_iou
                best_iou = iou
                best_label = label
                best_threshold = threshold
            elif iou > second_iou:
                second_iou = iou

        # Ambiguity check: top two IOU values too close → reject
        if best_iou - second_iou < 0.10 and second_iou > 0:
            print(f"[RoiIou] Ambiguous: best={best_label}({best_iou:.3f}) vs 2nd({second_iou:.3f}) → {self.default_label}")
            return ClassifyResult(label=self.default_label, confidence=float(best_iou))

        if best_iou < best_threshold:
            print(f"[RoiIou] IoU={best_iou:.3f}<{best_threshold} -> {self.default_label}")
            return ClassifyResult(label=self.default_label, confidence=float(best_iou))

        print(f"[RoiIou] {yolo_result.label} -> {best_label} (IoU={best_iou:.3f}, thr={best_threshold})")
        return ClassifyResult(label=best_label, confidence=float(best_iou))

    def reset(self):
        super().reset()


import cv2  # needed for resize, cvtColor in YOLOClassifier.classify


def make_classifier(method: str, **kwargs) -> BaseClassifier:
    """Factory to create a classifier from a method name.

    Supported methods:
      - "apriltag":     AprilTagClassifier (marker-based)
      - "yolo_detect":  YOLOClassifier (ONNX, no marker needed)
      - "yolo_roi":     RoiIouClassifier (YOLO + ROI-IoU position matching)
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
    if method == "yolo_roi":
        return RoiIouClassifier(
            model_path=kwargs.get("model_path", ""),
            classes=kwargs.get("classes", ["short", "long"]),
            default_label=kwargs.get("default_label", "unknown"),
            default_next_task=kwargs.get("default_next_task", ""),
            conf_threshold=kwargs.get("conf_threshold", 0.15),
            roi_reference_path=kwargs.get("roi_reference_path", ""),
            iou_threshold=kwargs.get("iou_threshold", 0.3),
        )
    raise ValueError(f"Unknown classify method: {method}")
