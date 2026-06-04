#!/usr/bin/env python3
"""Workpiece classification method benchmark — standalone test script.

Connects to a camera, captures template images of each workpiece type,
then benchmarks multiple classification methods with both automatic
(Otsu threshold) and manual (mouse drag) ROI selection.

Usage:
    python test_classification_methods.py --camera 0
    python test_classification_methods.py --camera 0 --load-templates  # reuse saved templates

Keys during benchmark:
    S  = test capture (auto ROI)
    M  = test capture (manual ROI — click-drag in popup window)
    T  = toggle auto/manual comparison mode (both ROIs on same frame)
    Q  = finish and show summary
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── helpers ──────────────────────────────────────────────────


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={index}")
    return cap


# ── mouse-drag ROI selector ───────────────────────────────────


class ROISelector:
    """OpenCV mouse callback that lets user drag a rectangle to define ROI."""

    def __init__(self, window_name: str = "Select ROI"):
        self.window_name = window_name
        self.start = None
        self.rect = None       # (x, y, w, h)
        self.confirmed = False
        self._draw = None

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.rect = None
            self.confirmed = False
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            x0, y0 = self.start
            self.rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            x0, y0 = self.start
            self.rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            self.start = None

    def select(self, frame: np.ndarray) -> tuple | None:
        """Show frame, let user drag ROI.  Returns (x, y, w, h) or None if cancelled."""
        self.start = None
        self.rect = None
        self.confirmed = False
        self._draw = frame.copy()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._callback)

        print("  → Drag rectangle on the image.  ENTER=confirm  ESC=cancel")

        while True:
            display = self._draw.copy()
            if self.rect and self.rect[2] > 0 and self.rect[3] > 0:
                x, y, w, h = self.rect
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, f"ROI: {w}x{h}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(display, "ENTER=confirm  ESC=cancel", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.rect is not None and self.rect[2] > 0:  # Enter
                cv2.destroyWindow(self.window_name)
                return self.rect
            elif key == 27:  # Esc
                cv2.destroyWindow(self.window_name)
                return None


def _roi_from_rect(frame, rect):
    """Crop frame to rect, return (roi, contour)."""
    x, y, w, h = rect
    roi = frame[y:y + h, x:x + w]
    # Build a synthetic contour covering the full manual ROI
    contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
                       dtype=np.int32)
    return roi, contour


def extract_roi(frame, method="otsu", bg_frame=None, canny_low=50, canny_high=150):
    """Extract workpiece ROI using various segmentation methods.

    Methods:
      - otsu:     Otsu threshold (needs foreground/background contrast)
      - canny:    Canny edge detection → contours (works when edges visible)
      - bg_diff:  Background subtraction (needs empty-bench reference frame)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "otsu":
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "canny":
        edges = cv2.Canny(blur, canny_low, canny_high)
        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(edges, kernel, iterations=2)
    elif method == "bg_diff":
        if bg_frame is None:
            return None, None, None, None
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
        diff = cv2.absdiff(blur, bg_blur)
        _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    else:
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi = frame[y:y + h, x:x + w]
    return roi, (x, y, w, h), largest, binary


def _get_roi(frame, roi_method="otsu", bg_frame=None, fixed_rect=None):
    """Return (roi, bbox, contour, binary) — using fixed rect if available.

    For fixed_rect mode: crops to the fixed rectangle, then runs Otsu
    thresholding WITHIN the crop to find the workpiece contour.  This
    is far more robust than bg_diff because the sub-window histogram
    is not dominated by the entire bench background.
    """
    if fixed_rect is not None:
        x, y, w, h = fixed_rect
        crop = frame[y:y + h, x:x + w]

        # Segment within the fixed rect using Otsu
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_crop = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        _, binary_crop = cv2.threshold(blur_crop, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Also try a simple fixed threshold as fallback (dark workpiece on lighter bench)
        _, binary_fixed = cv2.threshold(blur_crop, 80, 255, cv2.THRESH_BINARY_INV)

        # Pick whichever gives a reasonable contour (not the whole crop)
        best_contour = None
        for bmap in [binary_crop, binary_fixed]:
            contours, _ = cv2.findContours(bmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                ratio = area / (w * h)
                if 0.02 < ratio < 0.95:  # not noise, not whole crop
                    best_contour = largest
                    break

        if best_contour is not None:
            cx, cy, cw, ch = cv2.boundingRect(best_contour)
            fx, fy = x + cx, y + cy
            tight_cont = np.array([[[fx, fy]], [[fx + cw, fy]],
                                   [[fx + cw, fy + ch]], [[fx, fy + ch]]], dtype=np.int32)
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = np.zeros_like(gray_full)
            binary[fy:fy + ch, fx:fx + cw] = 255
            roi = frame[fy:fy + ch, fx:fx + cw]
            return roi, (fx, fy, cw, ch), tight_cont, binary

        # Fallback: use whole fixed rect
        roi, contour = _roi_from_rect(frame, fixed_rect)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = np.zeros_like(gray)
        cv2.rectangle(binary, (x, y), (x + w, y + h), 255, -1)
        return roi, (x, y, w, h), contour, binary
    return extract_roi(frame, method=roi_method, bg_frame=bg_frame)


# ── classification methods ───────────────────────────────────


def method_contour_area(roi, contour, _templates, _template_contours):
    if contour is None:
        return "none"
    area = cv2.contourArea(contour)
    if area < 100:
        return "none"
    return f"area={int(area)}"


def method_contour_area_vs_templates(roi, contour, templates, template_contours):
    if contour is None:
        return "no_object"
    area = cv2.contourArea(contour)
    if area < 100:
        return "no_object"
    scores = {}
    for label, tc in template_contours.items():
        if tc is None or cv2.contourArea(tc) < 100:
            scores[label] = 0
        else:
            t_area = cv2.contourArea(tc)
            ratio = max(area, t_area) / max(min(area, t_area), 1)
            scores[label] = 1.0 / ratio
    best = max(scores, key=scores.get)
    return f"{best} (conf={scores[best]:.2f})"


def method_hu_moments(roi, contour, templates, template_contours):
    if contour is None or cv2.contourArea(contour) < 100:
        return "no_object"
    cur_hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    cur_hu = np.sign(cur_hu) * np.log10(np.abs(cur_hu) + 1e-10)
    scores = {}
    for label, tc in template_contours.items():
        if tc is None or cv2.contourArea(tc) < 100:
            scores[label] = 0
        else:
            t_hu = cv2.HuMoments(cv2.moments(tc)).flatten()
            t_hu = np.sign(t_hu) * np.log10(np.abs(t_hu) + 1e-10)
            dist = np.linalg.norm(cur_hu - t_hu)
            scores[label] = 1.0 / (1.0 + dist)
    best = max(scores, key=scores.get)
    return f"{best} (conf={scores[best]:.2f})"


def method_circularity(roi, contour, _templates, _tc):
    if contour is None:
        return "no_object"
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area < 100 or perimeter < 10:
        return "no_object"
    c = (perimeter * perimeter) / (4 * math.pi * area)
    return f"circularity={c:.2f}"


def method_aspect_ratio(roi, contour, _templates, _tc):
    if contour is None or cv2.contourArea(contour) < 100:
        return "no_object"
    x, y, w, h = cv2.boundingRect(contour)
    ar = max(w, h) / max(min(w, h), 1)
    return f"AR={ar:.2f}"


def method_hsv_histogram(roi, contour, templates, template_hists):
    if roi is None or roi.size == 0:
        return "no_object"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cur_hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(cur_hist, cur_hist, 0, 1, cv2.NORM_MINMAX)
    scores = {}
    for label in templates:
        key = label + "_hist"
        if key in template_hists:
            score = cv2.compareHist(cur_hist, template_hists[key], cv2.HISTCMP_BHATTACHARYYA)
            scores[label] = 1.0 - score
        else:
            scores[label] = 0
    best = max(scores, key=scores.get)
    return f"{best} (conf={scores[best]:.2f})"


def method_template_match(roi, _contour, templates, _tc):
    if roi is None or roi.size == 0:
        return "no_object"
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scores = {}
    for label, tmpl in templates.items():
        if tmpl is None or tmpl.size == 0:
            scores[label] = 0
            continue
        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        if gray_tmpl.shape != gray_roi.shape:
            gray_tmpl = cv2.resize(gray_tmpl, (gray_roi.shape[1], gray_roi.shape[0]))
        result = cv2.matchTemplate(gray_roi, gray_tmpl, cv2.TM_CCOEFF_NORMED)
        scores[label] = (float(result) + 1.0) / 2.0
    best = max(scores, key=scores.get)
    return f"{best} (conf={scores[best]:.2f})"


def method_orb_features(roi, _contour, templates, _tc):
    if roi is None or roi.size == 0:
        return "no_object"
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    scores = {}
    for label, tmpl in templates.items():
        if tmpl is None or tmpl.size == 0:
            scores[label] = 0
            continue
        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray_roi, None)
        kp2, des2 = orb.detectAndCompute(gray_tmpl, None)
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            scores[label] = 0
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        scores[label] = len(good) / max(len(kp1), 1)
    best = max(scores, key=scores.get)
    return f"{best} (conf={scores[best]:.2f})"


# ── YOLO / ONNX detection ─────────────────────────────────────

_YOLO_MODEL = None          # ultralytics YOLO instance (server) or None
_ONNX_SESSION = None        # onnxruntime session (robot) or None
_MODEL_PATH_PT  = "/root/workspace/dc_dir/detection_model/runs/workpiece_yolo/weights/best.pt"
_MODEL_PATH_ONNX = "/root/workspace/dc_dir/detection_model/runs/workpiece_yolo/weights/best.onnx"
_YOLO_CLASSES = ["short", "long"]  # cls_id → label
_YOLO_PREVIEW_TICK = 0

# ONNX image size — must match export imgsz
_ONNX_IMGSZ = 640
_ONNX_INPUT_NAME = "images"
_ORT_AVAILABLE = False


def _init_onnx():
    """Initialise onnxruntime session. Called once on first use."""
    global _ONNX_SESSION, _ORT_AVAILABLE
    if _ONNX_SESSION is not None:
        return True
    try:
        import onnxruntime as ort
        _ONNX_SESSION = ort.InferenceSession(_MODEL_PATH_ONNX)
        _ORT_AVAILABLE = True
        print(f"[ONNX] Session loaded: {_MODEL_PATH_ONNX}")
        return True
    except ImportError:
        _ORT_AVAILABLE = False
        print("[ONNX] onnxruntime not installed — pip install onnxruntime")
        return False
    except Exception as e:
        _ORT_AVAILABLE = False
        print(f"[ONNX] Failed to load model: {e}")
        return False


def _ort_detect(frame_bgr):
    """Run ONNX inference, return list of (x1,y1,x2,y2,conf,cls_id)."""
    _init_onnx()
    if _ONNX_SESSION is None:
        return []

    h0, w0 = frame_bgr.shape[:2]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (_ONNX_IMGSZ, _ONNX_IMGSZ))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    outputs = _ONNX_SESSION.run(None, {_ONNX_INPUT_NAME: img})
    preds = outputs[0][0]  # [4+nc, 8400]

    boxes_xywh, scores, cls_ids = [], [], []
    for i in range(preds.shape[1]):
        cx, cy, bw, bh = preds[0:4, i]
        cls_conf = preds[4:, i]
        max_conf = float(cls_conf.max())
        if max_conf < 0.15:
            continue
        cls_id = int(cls_conf.argmax())
        # Convert cxcywh → xyxy, scale back to original image size
        x1 = (cx - bw / 2) / _ONNX_IMGSZ * w0
        y1 = (cy - bh / 2) / _ONNX_IMGSZ * h0
        x2 = (cx + bw / 2) / _ONNX_IMGSZ * w0
        y2 = (cy + bh / 2) / _ONNX_IMGSZ * h0
        # NMS expects [x, y, w, h] format
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(max_conf)
        cls_ids.append(cls_id)

    # NMS — remove duplicate boxes
    if boxes_xywh:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.15, 0.45)
        if len(indices) > 0:
            result = []
            for i in indices.flatten():
                x, y, w, h = boxes_xywh[i]
                result.append((x, y, x + w, y + h, scores[i], cls_ids[i]))
            return result
    return []


def _get_yolo():
    """Return ultralytics YOLO instance, or None (falls back to ONNX)."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(_MODEL_PATH_PT)
        except ImportError:
            return None
    return _YOLO_MODEL


def _run_detection(frame):
    """Run YOLO (ultralytics) or ONNX, whichever is available.

    Returns list of (x1, y1, x2, y2, conf, cls_id).
    """
    # Try ultralytics first (server), fall back to ONNX (robot)
    model = _get_yolo()
    if model is not None:
        results = model(frame, conf=0.15, verbose=False)
        boxes = results[0].boxes
        return [(float(b.xyxy[0][0]), float(b.xyxy[0][1]),
                 float(b.xyxy[0][2]), float(b.xyxy[0][3]),
                 float(b.conf[0]), int(b.cls[0])) for b in boxes]
    # ONNX fallback
    return _ort_detect(frame)


def _draw_detection_boxes(frame, detections):
    """Draw detection boxes onto frame (in-place)."""
    for x1, y1, x2, y2, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = _YOLO_CLASSES[cls_id] if cls_id < len(_YOLO_CLASSES) else f"cls{cls_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)


def method_yolo_classify(frame, _roi, _contour, templates, template_contours):
    """YOLO/ONNX detection → direct class prediction."""
    detections = _run_detection(frame)
    if not detections:
        return "no_detection"

    # Draw boxes for live preview
    _draw_detection_boxes(frame, detections)

    # Take the highest-confidence detection
    best = max(detections, key=lambda d: d[4])
    _, _, _, _, best_conf, best_cls = best
    label = _YOLO_CLASSES[best_cls] if best_cls < len(_YOLO_CLASSES) else f"cls{best_cls}"
    return f"{label} (conf={best_conf:.2f})"


# ── main ─────────────────────────────────────────────────────


METHODS = {
    "contour_area":          (method_contour_area,                "Contour pixel area (raw)"),
    "contour_vs_template":   (method_contour_area_vs_templates,   "Contour area vs templates"),
    "hu_moments":            (method_hu_moments,                  "Hu moments shape match"),
    "circularity":           (method_circularity,                 "Perimeter²/(4π·area)"),
    "aspect_ratio":          (method_aspect_ratio,                "Bounding rect W/H ratio"),
    "hsv_histogram":         (method_hsv_histogram,               "HSV colour histogram"),
    "template_match":        (method_template_match,              "Template match (ROI)"),
    "orb_features":          (method_orb_features,                "ORB keypoint matching"),
    "yolo_detect":           (None,                                "YOLO-nano detect + area match"),
}


def benchmark_one(methods, frame, roi, contour, templates, template_contours, template_hists):
    """Run all methods once, return {name: (output_str, time_ms)}."""
    result = {}
    for name, (func, _) in methods.items():
        t0 = time.perf_counter()
        try:
            if name == "hsv_histogram":
                out = func(roi, contour, templates, template_hists)
            elif name == "yolo_detect":
                out = method_yolo_classify(frame, roi, contour, templates, template_contours)
            else:
                out = func(roi, contour, templates, template_contours)
        except Exception as e:
            out = f"ERR: {e}"
        t_ms = (time.perf_counter() - t0) * 1000
        result[name] = (out, t_ms)
    return result


def print_one_result(label, roi_info, bench, indent="  "):
    """Pretty-print a single benchmark column."""
    print(f"\n{indent}┌─ {label} ({roi_info})")
    print(f"{indent}│ {'Method':<25} {'Time':>8}  {'Result'}")
    print(f"{indent}│ {'─'*25} {'─'*8}  {'─'*40}")
    for name, (_, _) in METHODS.items():
        out, t_ms = bench[name]
        print(f"{indent}│ {name:<25} {t_ms:>6.1f}ms  {out}")
    print(f"{indent}└{'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark workpiece classification methods")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of test captures per method")
    parser.add_argument("--labels", type=str, default="type_A,type_B",
                        help="Comma-separated workpiece type labels (e.g. 'large,small')")
    parser.add_argument("--save-templates", type=str, default="workpiece_templates.json",
                        help="Path to save/load templates")
    parser.add_argument("--load-templates", action="store_true",
                        help="Skip template capture, reuse saved templates")
    parser.add_argument("--roi-method", type=str, default="otsu",
                        choices=["otsu", "canny", "bg_diff", "fixed"],
                        help="Auto ROI method (default: otsu). 'fixed'=use same manual rect for all")
    args = parser.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(labels) < 2:
        print("ERROR: need at least 2 labels (--labels type_A,type_B)")
        sys.exit(1)

    cap = open_camera(args.camera)
    print(f"Camera opened: {args.camera}")

    roi_method = args.roi_method

    # ── Shared fixed ROI (drawn BEFORE bg capture, so user can see workpiece) ──
    fixed_roi_rect = None
    if roi_method == "fixed":
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │  Place a workpiece so you can see where  │")
        print("  │  the work area is. Draw ONE rectangle    │")
        print("  │  covering the entire work zone.          │")
        print("  │  This rect will be used for ALL captures. │")
        print("  └─────────────────────────────────────────┘")
        selector = ROISelector("Define Fixed ROI")
        ok, frame_tmp = cap.read()
        for _ in range(5):
            cap.read()
        ok, frame_tmp = cap.read()
        fixed_roi_rect = selector.select(frame_tmp)
        if fixed_roi_rect is not None:
            x, y, w, h = fixed_roi_rect
            print(f"  Fixed ROI: ({x}, {y}, {w}, {h})")
        else:
            print("  Fixed ROI cancelled, falling back to otsu")
            roi_method = "otsu"

    # ── Capture empty-bench background (for bg_diff or fixed+bg_diff) ──
    bg_frame = None
    need_bg = roi_method in ("bg_diff",)
    if need_bg:
        existing_bg = cv2.imread("background_bench.png")
        if existing_bg is not None:
            bg_frame = existing_bg
            print("  Loaded existing background_bench.png")
        else:
            print("\n  ┌─────────────────────────────────────────┐")
            print("  │  Clear the workbench (no workpiece)      │")
            print("  │  Press S to capture background           │")
            print("  └─────────────────────────────────────────┘")
            bg_confirmed = False
            while not bg_confirmed:
                ok, frame_tmp = cap.read()
                if ok:
                    display = frame_tmp.copy()
                    cv2.putText(display, "S=capture bg  Q=skip", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("Capture Background", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        bg_frame = frame_tmp.copy()
                        cv2.imwrite("background_bench.png", bg_frame)
                        print("  Background captured → background_bench.png")
                        bg_confirmed = True
                    elif key == ord('q'):
                        print("  Background skipped — using raw fixed ROI")
                        bg_confirmed = True
                else:
                    time.sleep(0.1)
            cv2.destroyWindow("Capture Background")

    # ── Phase 1: capture / load templates ──
    templates = {}            # label → ROI image
    template_contours = {}    # label → contour
    template_hists = {}       # "label_hist" → histogram
    save_path = Path(args.save_templates)

    if args.load_templates and save_path.exists():
        data = json.loads(save_path.read_text())
        for label, info in data.items():
            img = cv2.imread(info["image_path"])
            if img is not None:
                _, _, cont, _ = extract_roi(img)
                if cont is not None:
                    templates[label] = img  # store full template image
                    template_contours[label] = cont
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    template_hists[label + "_hist"] = hist
                print(f"  Loaded template '{label}' from {info['image_path']}")
    else:
        print("\n— Phase 1: Capture templates —")
        for label in labels:
            print(f"\n>>> Place workpiece '{label}' in front of the camera <<<")
            ok, frame = cap.read()
            for _ in range(10):  # warm up
                cap.read()
            ok, frame = cap.read()

            # Let user choose auto or manual ROI for template
            print("  A=auto threshold   M=mouse drag ROI")
            while True:
                display = frame.copy()
                cv2.putText(display, f"TEMPLATE: {label}  A=auto  M=manual  Q=skip",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Capture Template", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('a'):
                    # Preview auto ROI before saving
                    roi, bbox, cont, binary = _get_roi(frame, roi_method, bg_frame, fixed_roi_rect)
                    if roi is not None:
                        preview = frame.copy()
                        x, y, w, h = bbox
                        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(preview, f"Auto ROI: {w}x{h}  area={int(cv2.contourArea(cont))}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(preview, "ENTER=save  R=retry  M=manual",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        # Also show binary mask in corner
                        binary_disp = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                        binary_disp = cv2.resize(binary_disp, (160, 120))
                        preview[0:120, 480:640] = binary_disp
                        cv2.putText(preview, "mask", (485, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        cv2.imshow("Capture Template - Preview", preview)
                        while True:
                            k2 = cv2.waitKey(1) & 0xFF
                            if k2 == 13:  # Enter → confirm
                                cv2.destroyWindow("Capture Template - Preview")
                                cv2.destroyWindow("Capture Template")
                                templates[label] = roi
                                template_contours[label] = cont
                                print(f"  Auto ROI: {bbox}  area={int(cv2.contourArea(cont))}")
                                break
                            elif k2 == ord('r'):  # R → retry auto
                                cv2.destroyWindow("Capture Template - Preview")
                                print("  Retrying auto ROI...")
                                break  # back to outer A/M/Q loop
                            elif k2 == ord('m'):  # M → switch to manual
                                cv2.destroyWindow("Capture Template - Preview")
                                cv2.destroyWindow("Capture Template")
                                selector = ROISelector("Select Template ROI")
                                rect = selector.select(frame)
                                if rect is not None:
                                    roi2, cont2 = _roi_from_rect(frame, rect)
                                    templates[label] = roi2
                                    template_contours[label] = cont2
                                    print(f"  Manual ROI: {rect}  area={rect[2]*rect[3]}")
                                else:
                                    print("  Manual ROI cancelled")
                                    cont = None
                                break  # break outer too
                        if k2 == 13 or k2 == ord('m'):
                            break  # jump to outer break
                        # if k2 == 'r', continue outer while
                    else:
                        print("  Auto ROI failed — no contour found, try manual or reposition")
                        cont = None
                        break
                elif key == ord('m'):
                    cv2.destroyWindow("Capture Template")
                    selector = ROISelector("Select Template ROI")
                    rect = selector.select(frame)
                    if rect is not None:
                        roi, cont = _roi_from_rect(frame, rect)
                        templates[label] = roi
                        template_contours[label] = cont
                        print(f"  Manual ROI: {rect}  area={rect[2]*rect[3]}")
                    else:
                        print("  Manual ROI cancelled")
                        cont = None
                    break
                elif key == ord('q'):
                    cv2.destroyWindow("Capture Template")
                    cont = "skip"
                    break

            if cont is not None and isinstance(cont, np.ndarray):
                hsv = cv2.cvtColor(templates[label], cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                template_hists[label + "_hist"] = hist
                out_path = f"template_{label}.png"
                cv2.imwrite(out_path, templates[label])
                area = cv2.contourArea(template_contours[label])
                print(f"  Template '{label}' saved → {out_path} (area={int(area)})")

        data = {}
        for label in templates:
            data[label] = {
                "image_path": f"template_{label}.png",
                "contour_area": int(cv2.contourArea(template_contours.get(label, np.zeros((1,1,3), dtype=np.uint8)))),
            }
        save_path.write_text(json.dumps(data, indent=2))
        print(f"\nTemplates saved to {save_path}")

    if len(templates) < 2:
        print("ERROR: Need at least 2 templates. Re-run capture.")
        cap.release()
        sys.exit(1)

    # Report template info
    print("\n  Template summary:")
    for label in templates:
        tc = template_contours.get(label)
        if tc is not None:
            area = int(cv2.contourArea(tc))
            x, y, w, h = cv2.boundingRect(tc)
            print(f"    {label}: area={area}  bbox=({x},{y},{w},{h})")
            if len(templates) > 1:
                other = [l for l in templates if l != label][0]
                other_area = int(cv2.contourArea(template_contours[other]))
                ratio = max(area, other_area) / max(min(area, other_area), 1)
                print(f"           area_ratio({label}/{other}) = {ratio:.2f}")

    # ── Phase 2: benchmark ──
    print(f"\n{'='*60}")
    print(f"  Phase 2: Benchmark  (ROI method: {roi_method})")
    print(f"  S = test auto ROI | M = test manual ROI")
    print(f"  1=Otsu  2=Canny  3=BG-diff  |  switch ROI method")
    print(f"  C = compare both ROIs on same frame")
    print(f"  Q = finish and show summary")
    print(f"{'='*60}\n")

    auto_results = {name: [] for name in METHODS}
    manual_results = {name: [] for name in METHODS}
    selector = ROISelector()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Auto extraction
        auto_roi, auto_bbox, auto_contour, auto_binary = _get_roi(
            frame, roi_method, bg_frame, fixed_roi_rect)
        display = frame.copy()

        # Show auto ROI
        if auto_bbox:
            x, y, w, h = auto_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "auto", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Stats
        n_auto = sum(1 for r in auto_results["contour_area"] if r.get("roi_type") == "auto")
        n_manual = sum(1 for r in auto_results["contour_area"] if r.get("roi_type") == "manual")
        info = (f"ROI={roi_method} | S=auto({n_auto}) M=man({n_manual}) "
                f"1=Otsu 2=Canny 3=BG | C=compare Q=finish")
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Live YOLO/ONNX preview (every 5th frame to avoid lag)
        global _YOLO_PREVIEW_TICK
        _YOLO_PREVIEW_TICK += 1
        if _YOLO_PREVIEW_TICK % 5 == 0:
            try:
                dets = _run_detection(frame)
                _draw_detection_boxes(display, dets)
            except Exception:
                pass

        cv2.imshow("Benchmark", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            roi_method = "otsu"
            print(f"  → ROI method switched to: {roi_method}")
            continue
        elif key == ord('2'):
            roi_method = "canny"
            print(f"  → ROI method switched to: {roi_method}")
            continue
        elif key == ord('3'):
            roi_method = "bg_diff"
            if bg_frame is None:
                # Capture bg on demand with confirmation
                print("  Capturing empty-bench background (press S to confirm)...")
                time.sleep(0.3)
                for _ in range(5):
                    cap.read()
                bg_confirmed = False
                while not bg_confirmed:
                    ok, frame_tmp = cap.read()
                    if ok:
                        display = frame_tmp.copy()
                        cv2.putText(display, "S=capture bg  Q=cancel", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.imshow("Capture Background", display)
                        key_bg = cv2.waitKey(1) & 0xFF
                        if key_bg == ord('s'):
                            bg_frame = frame_tmp.copy()
                            cv2.imwrite("background_bench.png", bg_frame)
                            print("  Background captured → background_bench.png")
                            bg_confirmed = True
                        elif key_bg == ord('q'):
                            print("  Cancelled, falling back to otsu")
                            roi_method = "otsu"
                            bg_confirmed = True
                cv2.destroyWindow("Capture Background")
            print(f"  → ROI method switched to: {roi_method}")
            continue

        elif key == ord('q'):
            break

        elif key == ord('s'):
            # Auto ROI only
            if auto_roi is None:
                print("  Auto ROI not found! Try M for manual.")
                continue
            bench = benchmark_one(METHODS, frame, auto_roi, auto_contour, templates,
                                  template_contours, template_hists)
            print_one_result("AUTO", f"bbox={auto_bbox}", bench)
            prompt = f"  True label ({'/'.join(labels)}) or Enter to skip: "
            true_label = input(prompt).strip()
            if true_label:
                for name in METHODS:
                    out, t = bench[name]
                    correct = true_label in out
                    auto_results[name].append({
                        "true": true_label, "output": out, "correct": correct,
                        "time_ms": t, "roi_type": "auto",
                    })

        elif key == ord('m'):
            # Manual ROI
            rect = selector.select(frame)
            if rect is None:
                print("  Manual ROI cancelled")
                continue
            manual_roi, manual_contour = _roi_from_rect(frame, rect)
            if manual_roi is None or manual_roi.size == 0:
                continue
            bench = benchmark_one(METHODS, frame, manual_roi, manual_contour, templates,
                                  template_contours, template_hists)
            print_one_result("MANUAL", f"rect={rect}", bench)
            prompt = f"  True label ({'/'.join(labels)}) or Enter to skip: "
            true_label = input(prompt).strip()
            if true_label:
                for name in METHODS:
                    out, t = bench[name]
                    correct = true_label in out
                    manual_results[name].append({
                        "true": true_label, "output": out, "correct": correct,
                        "time_ms": t, "roi_type": "manual",
                    })

        elif key == ord('c'):
            # Side-by-side: auto + manual on same frame
            print("\n  — Auto ROI —")
            if auto_roi is not None:
                bench_auto = benchmark_one(METHODS, frame, auto_roi, auto_contour, templates,
                                           template_contours, template_hists)
                print_one_result("AUTO", f"bbox={auto_bbox}", bench_auto)
            else:
                bench_auto = None
                print("  [auto ROI not found]")

            print("\n  — Manual ROI (drag rectangle) —")
            rect = selector.select(frame)
            if rect is not None:
                manual_roi, manual_contour = _roi_from_rect(frame, rect)
                bench_manual = benchmark_one(METHODS, frame, manual_roi, manual_contour, templates,
                                             template_contours, template_hists)
                print_one_result("MANUAL", f"rect={rect}", bench_manual)

                # Comparison table
                if bench_auto is not None:
                    print(f"\n  ╔{' AUTO vs MANUAL comparison ':=^62}╗")
                    print(f"  ║ {'Method':<25} {'Auto':>15}  {'Manual':>15} ║")
                    print(f"  ║ {'─'*25} {'─'*15}  {'─'*15} ║")
                    for name, (_, _) in METHODS.items():
                        a_out, a_t = bench_auto[name]
                        m_out, m_t = bench_manual[name]
                        match = " ✓" if a_out.split()[0] == m_out.split()[0] else " ✗"
                        print(f"  ║ {name:<25} {a_out:<15}  {m_out:<15} ║{match}")
                    print(f"  ╚{'='*62}╝")

                prompt = f"  True label ({'/'.join(labels)}) or Enter to skip: "
                true_label = input(prompt).strip()
                if true_label:
                    # Record both
                    if bench_auto:
                        for name in METHODS:
                            out, t = bench_auto[name]
                            correct = true_label in out
                            auto_results[name].append({
                                "true": true_label, "output": out, "correct": correct,
                                "time_ms": t, "roi_type": "auto",
                            })
                    if bench_manual:
                        for name in METHODS:
                            out, t = bench_manual[name]
                            correct = true_label in out
                            manual_results[name].append({
                                "true": true_label, "output": out, "correct": correct,
                                "time_ms": t, "roi_type": "manual",
                            })
            else:
                print("  Manual ROI cancelled")

    cv2.destroyAllWindows()
    cap.release()

    # ── Phase 3: summary ──
    def _print_summary(title, results):
        print(f"\n{'─'*65}")
        print(f"  {title}")
        print(f"  {'Method':<25} {'Time':>7}  {'Acc':>7}  {'N'}")
        print(f"  {'─'*25} {'─'*7}  {'─'*7}  {'─'*3}")
        for name, (_, _) in METHODS.items():
            entries = results[name]
            if not entries:
                continue
            correct = sum(1 for e in entries if e["correct"])
            avg_time = sum(e["time_ms"] for e in entries) / len(entries)
            acc = correct / len(entries) * 100
            print(f"  {name:<25} {avg_time:>6.1f}ms {acc:>6.1f}%  {len(entries)}")

    _print_summary("AUTO ROI (Otsu threshold)", auto_results)
    _print_summary("MANUAL ROI (mouse drag)", manual_results)

    # Combined comparison
    if any(auto_results[n] for n in METHODS) and any(manual_results[n] for n in METHODS):
        print(f"\n  ╔{' AUTO vs MANUAL ACCURACY ':─^62}╗")
        print(f"  ║ {'Method':<25} {'Auto':>8}  {'Manual':>8}  {'Diff':>8} ║")
        print(f"  ║ {'─'*25} {'─'*8}  {'─'*8}  {'─'*8} ║")
        for name, (_, _) in METHODS.items():
            ae = auto_results[name]
            me = manual_results[name]
            if not ae or not me:
                continue
            a_acc = sum(1 for e in ae if e["correct"]) / len(ae) * 100
            m_acc = sum(1 for e in me if e["correct"]) / len(me) * 100
            diff = m_acc - a_acc
            diff_s = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
            print(f"  ║ {name:<25} {a_acc:>7.1f}% {m_acc:>7.1f}% {diff_s:>8} ║")
        print(f"  ╚{'='*62}╝")


if __name__ == "__main__":
    main()
