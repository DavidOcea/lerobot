#!/usr/bin/env python3
"""
边缘模板匹配验证脚本 — 针对无纹理金属件

原理:
  1. 离线: 提取模板图像的边缘梯度方向 (对光照变化鲁棒)
  2. 在线: 在搜索图像中匹配模板, 输出 XY + 旋转 + 尺度

与 SimpleIBVS 的接口兼容:
  detect() → [{'center': (cx,cy), 'rotation_deg': rot, 'size_px': s, ...}]

用法:
  # 从相机采集模板
  python edge_template_matcher.py --camera 4 --save-template metal_part.png

  # 用模板搜索测试图像
  python edge_template_matcher.py --template metal_part.png --search test.png

  # 实时跟踪模式
  python edge_template_matcher.py --template metal_part.png --camera 4 --live

依赖: opencv-python, numpy
"""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path


class EdgeTemplateMatcher:
    """基于边缘梯度方向的模板匹配器 — 适合金属件等无纹理物体"""

    def __init__(self, template_path: str = None, physical_size_mm: float = 20.0,
                 canny_low: int = 50, canny_high: int = 150,
                 angle_range: float = 180, angle_step: float = 1.0,
                 scale_range: float = 0.2, scale_step: float = 0.05,
                 match_threshold: float = 0.5):
        """
        Args:
            template_path: 模板图像路径
            physical_size_mm: 物体物理尺寸 (mm), 用于深度估算
            canny_low/high: Canny 边缘检测阈值
            angle_range: 旋转搜索范围 (±度)
            angle_step: 旋转搜索步长 (度)
            scale_range: 尺度搜索范围 (±比例)
            scale_step: 尺度搜索步长
            match_threshold: 匹配阈值 (0~1), 低于此值认为未检测到
        """
        self.physical_size_mm = physical_size_mm
        self.tag_size_mm = physical_size_mm  # 兼容 SimpleIBVS 接口
        self.detector_type = 'edgetemplate'  # 标识检测器类型
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.match_threshold = match_threshold

        self._template_gray = None
        self._template_edges = None
        self._template_mask = None

        if template_path:
            self.load_template(template_path)

    # ── 模板管理 ──────────────────────────────────────────────

    def load_template(self, path: str):
        """加载模板图像并提取边缘特征"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"无法加载模板: {path}")
        self._template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._template_edges = self._extract_edges(self._template_gray)
        print(f"✓ 模板加载: {path} ({img.shape[1]}x{img.shape[0]})")
        print(f"  边缘像素数: {np.sum(self._template_edges > 0)}")

    def save_template(self, image: np.ndarray, path: str):
        """保存当前帧作为模板"""
        cv2.imwrite(path, image)
        print(f"✓ 模板已保存: {path}")
        self.load_template(path)

    def set_template_from_frame(self, image: np.ndarray):
        """从内存中的图像设置模板 (不写文件)"""
        self._template_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._template_edges = self._extract_edges(self._template_gray)
        print(f"✓ 模板已设置: {image.shape[1]}x{image.shape[0]}")
        print(f"  边缘像素数: {np.sum(self._template_edges > 0)}")

    def _extract_edges(self, gray: np.ndarray) -> np.ndarray:
        """提取边缘 + 生成边缘 ROI mask (边缘区域扩大)"""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        # 膨胀边缘区域以增加容错
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(edges, kernel, iterations=1)

    # ── 梯度方向特征 (光照鲁棒方案) ──────────────────────────

    def _compute_gradient_orientation(self, gray: np.ndarray) -> tuple:
        """计算梯度方向图 (0~180度) 和幅值"""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        ori = np.arctan2(gy, gx)  # -pi ~ pi
        ori = np.rad2deg(ori) % 180  # 映射到 0~180
        return ori, mag

    def _match_gradient(self, search_gray: np.ndarray) -> list:
        """基于梯度方向匹配 (对光照变化最鲁棒) — 带旋转/尺度搜索

        返回: [(score, cx, cy, angle_deg, scale, template_w, template_h), ...]
        """
        if self._template_gray is None:
            return []

        t_h, t_w = self._template_gray.shape
        s_h, s_w = search_gray.shape
        if t_h > s_h or t_w > s_w:
            return []

        # 模板梯度
        t_ori, t_mag = self._compute_gradient_orientation(self._template_gray)
        t_mask = t_mag > np.percentile(t_mag, 30)  # 只用强梯度点

        results = []
        angles = np.arange(-self.angle_range, self.angle_range + 0.1, self.angle_step)
        scales = np.arange(1.0 - self.scale_range, 1.0 + self.scale_range + 0.01, self.scale_step)

        for scale in scales:
            for angle in angles:
                # 旋转 + 缩放模板
                M = cv2.getRotationMatrix2D((t_w/2, t_h/2), angle, scale)
                t_rot = cv2.warpAffine(self._template_gray, M, (t_w, t_h))
                t_edge_rot = self._extract_edges(t_rot)

                # 在搜索图上做边缘匹配
                s_edges = self._extract_edges(search_gray)
                if np.sum(t_edge_rot > 0) < 10:
                    continue

                result = cv2.matchTemplate(s_edges, t_edge_rot, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > self.match_threshold:
                    cx = max_loc[0] + t_w / 2
                    cy = max_loc[1] + t_h / 2
                    results.append((max_val, cx, cy, angle, scale, t_w, t_h))

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ── 快速边缘匹配 (默认方案, 速度优先) ─────────────────────

    def _match_edges_fast(self, search_gray: np.ndarray) -> list:
        """边缘模板匹配 — 旋转/尺度搜索, 速度优化版

        返回: [(score, cx, cy, angle_deg, scale, template_w, template_h), ...]
        """
        if self._template_gray is None:
            return []

        t_h, t_w = self._template_gray.shape
        s_h, s_w = search_gray.shape
        if t_h > s_h or t_w > s_w:
            return []

        s_edges = self._extract_edges(search_gray)
        results = []
        angles = np.arange(-self.angle_range, self.angle_range + 0.1, self.angle_step)
        scales = np.arange(1.0 - self.scale_range, 1.0 + self.scale_range + 0.01, self.scale_step)

        for scale in scales:
            new_w = max(20, int(t_w * scale))
            new_h = max(20, int(t_h * scale))

            for angle in angles:
                M = cv2.getRotationMatrix2D((t_w/2, t_h/2), angle, scale)
                t_rot = cv2.warpAffine(self._template_edges, M, (new_w, new_h),
                                       flags=cv2.INTER_NEAREST)

                if new_h > s_h or new_w > s_w:
                    continue

                result = cv2.matchTemplate(s_edges, t_rot, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > self.match_threshold:
                    cx = max_loc[0] + new_w / 2
                    cy = max_loc[1] + new_h / 2
                    results.append((max_val, cx, cy, angle, scale, new_w, new_h))

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ── 深度估算 (与 SimpleIBVS 兼容) ─────────────────────────

    def estimate_depth_mm(self, detection: dict, camera_fx: float) -> float:
        """从检测到的物体尺寸估算深度 (与 SimpleIBVS 接口兼容)"""
        if detection.get('size_px', 0) <= 0:
            return 0.0
        return self.physical_size_mm * camera_fx / detection['size_px']

    # ── 主检测接口 (与 SimpleIBVS 兼容) ───────────────────────

    def detect(self, image: np.ndarray, method: str = "edge",
               return_debug: bool = False) -> list:
        """
        检测图像中的模板物体

        Args:
            image: BGR 图像
            method: "edge" (边缘匹配) 或 "gradient" (梯度方向, 更鲁棒)
            return_debug: 返回调试信息 (匹配热力图等)

        Returns:
            [{'id': 0, 'center': (cx,cy), 'corners': [...],
              'rotation_deg': rot, 'size_px': s, 'size_mm': 20.0, 'score': score}, ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "gradient":
            raw = self._match_gradient(gray)
        else:
            raw = self._match_edges_fast(gray)

        if not raw:
            return []

        best = raw[0]
        score, cx, cy, angle, scale, t_w, t_h = best

        detections = [{
            'id': 0,
            'center': (float(cx), float(cy)),
            'corners': self._compute_corners(cx, cy, t_w, t_h, angle),
            'rotation_deg': float(angle),
            'size_px': float(max(t_w, t_h) * scale),
            'size_mm': self.physical_size_mm,
            'score': float(score),
        }]

        return detections

    def _compute_corners(self, cx, cy, w, h, angle_deg):
        """计算旋转后 4 角点坐标"""
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        corners = []
        for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
            x = cx + dx * cos_a - dy * sin_a
            y = cy + dx * sin_a + dy * cos_a
            corners.append((float(x), float(y)))
        return corners

    def draw_tags(self, image: np.ndarray, detections: list,
                  show_ids: bool = True) -> np.ndarray:
        """绘制检测结果 (兼容 AprilTagDetector 接口)"""
        return draw_detection(image, detections, show_score=True)


# ═══════════════════════════════════════════════════════════════
# 调试/可视化工具
# ═══════════════════════════════════════════════════════════════

def draw_detection(image: np.ndarray, detections: list,
                   show_score: bool = True) -> np.ndarray:
    """在图像上绘制检测结果"""
    vis = image.copy()
    for d in detections:
        cx, cy = d['center']
        corners = d['corners']

        # 中心十字
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)

        # 边界框
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        # 旋转方向线 (从角点0→角点1)
        cv2.line(vis,
                 (int(corners[0][0]), int(corners[0][1])),
                 (int(corners[1][0]), int(corners[1][1])),
                 (0, 0, 255), 2)

        label_parts = [f"rot={d['rotation_deg']:.1f}°", f"size={d['size_px']:.0f}px"]
        if show_score:
            label_parts.insert(0, f"{d['score']:.3f}")
        label = " ".join(label_parts)
        cv2.putText(vis, label, (int(cx) + 15, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis


def _select_roi(image: np.ndarray) -> tuple:
    """弹出窗口让用户用鼠标框选 ROI 区域

    Returns:
        (x, y, w, h) 或 None (用户取消)
    """
    roi_result = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_result['start'] = (x, y)
            roi_result['drawing'] = True
        elif event == cv2.EVENT_MOUSEMOVE and roi_result.get('drawing'):
            pass  # 预览在循环中绘制
        elif event == cv2.EVENT_LBUTTONUP:
            roi_result['end'] = (x, y)
            roi_result['drawing'] = False
            roi_result['done'] = True

    window_name = "Select ROI - drag mouse, ENTER=confirm, ESC=cancel"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    clone = image.copy()
    roi_result['drawing'] = False
    roi_result['done'] = False

    print("  请用鼠标框选工件区域 → ENTER 确认, ESC 取消")

    while True:
        display = clone.copy()
        if roi_result.get('drawing') and 'start' in roi_result:
            # 获取当前鼠标位置需要从窗口读取，这里用临时存储
            pass

        if 'start' in roi_result and 'end' in roi_result:
            x1, y1 = roi_result['start']
            x2, y2 = roi_result['end']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(50) & 0xFF

        if key == 13:  # ENTER
            if 'start' in roi_result and 'end' in roi_result:
                x1, y1 = roi_result['start']
                x2, y2 = roi_result['end']
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 10 and h > 10:
                    cv2.destroyWindow(window_name)
                    return (max(0, x), max(0, y), w, h)
            print("  请先框选区域再按 ENTER")
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return None


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

def open_camera(index: int):
    """尝试打开相机"""
    for api in [cv2.CAP_V4L2, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            return cap
    return None


def main():
    parser = argparse.ArgumentParser(description="边缘模板匹配验证")
    parser.add_argument("--template", "-t", type=str,
                        help="模板图像路径")
    parser.add_argument("--search", "-s", type=str,
                        help="搜索图像路径")
    parser.add_argument("--camera", "-c", type=int, default=None,
                        help="相机索引")
    parser.add_argument("--save-template", type=str,
                        help="从相机采集一帧保存为模板")
    parser.add_argument("--live", "-l", action="store_true",
                        help="实时跟踪模式")
    parser.add_argument("--method", "-m", choices=["edge", "gradient"],
                        default="edge",
                        help="匹配方法: edge(快速) / gradient(光照鲁棒)")
    parser.add_argument("--angle-range", type=float, default=90,
                        help="旋转搜索范围 ±度 (默认90)")
    parser.add_argument("--angle-step", type=float, default=1.0,
                        help="旋转搜索步长 (默认1度)")
    parser.add_argument("--scale-range", type=float, default=0.15,
                        help="尺度搜索范围 ±比例 (默认0.15)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="匹配阈值 (默认0.4)")
    parser.add_argument("--size-mm", type=float, default=20.0,
                        help="物体物理尺寸mm (默认20)")
    parser.add_argument("--output", "-o", type=str,
                        help="输出结果图像路径")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式: 相机预览, T拍模板, 空格拍测试并匹配")
    args = parser.parse_args()

    matcher = EdgeTemplateMatcher(
        physical_size_mm=args.size_mm,
        angle_range=args.angle_range,
        angle_step=args.angle_step,
        scale_range=args.scale_range,
        match_threshold=args.threshold,
    )

    # ── 模式: 采集模板 ──
    if args.save_template:
        cap = open_camera(args.camera or 0)
        if cap is None:
            print("✗ 无法打开相机")
            return 1
        for _ in range(10):
            cap.read()  # 预热
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("✗ 无法读取帧")
            return 1
        cv2.imwrite(args.save_template, frame)
        print(f"✓ 模板已保存: {args.save_template}")
        return 0

    # ── 交互模式 (无需预置模板) ──
    if args.interactive:
        pass  # 模板在交互循环中通过按键采集
    else:
        # ── 加载模板 ──
        if not args.template:
            print("✗ 需要 --template 或 --interactive 参数")
            return 1
        matcher.load_template(args.template)

    # ── 模式: 单张图片搜索 ──
    if args.search:
        img = cv2.imread(args.search)
        if img is None:
            print(f"✗ 无法加载: {args.search}")
            return 1

        t0 = time.perf_counter()
        detections = matcher.detect(img, method=args.method)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n搜索完成 ({elapsed:.1f}ms):")
        if detections:
            d = detections[0]
            print(f"  XY=({d['center'][0]:.1f}, {d['center'][1]:.1f})")
            print(f"  旋转={d['rotation_deg']:.1f}°")
            print(f"  尺寸={d['size_px']:.1f}px (物理={d['size_mm']:.1f}mm)")
            print(f"  得分={d['score']:.4f}")
        else:
            print("  未检测到物体")

        vis = draw_detection(img, detections)
        if args.output:
            cv2.imwrite(args.output, vis)
            print(f"  结果已保存: {args.output}")
        else:
            cv2.imshow("Edge Template Match", vis)
            print("  按任意键关闭...")
            cv2.waitKey(0)
        return 0

    # ── 模式: 交互式拍两张照片对比 ──
    if args.interactive:
        cap = open_camera(args.camera or 0)
        if cap is None:
            print("✗ 无法打开相机")
            return 1
        for _ in range(10):
            cap.read()

        template_frame = None
        test_frame = None
        result_text = ""
        roi = None  # (x, y, w, h) ROI 框选

        # 临时缩小搜索范围以提速 (交互模式默认值)
        matcher.angle_range = min(args.angle_range, 45)
        matcher.angle_step = max(args.angle_step, 2.0)

        print("\n交互模式:")
        print("  T = 拍模板 → 鼠标框选工件区域 → ENTER确认")
        print("  空格 = 拍测试照片 (自动匹配)")
        print("  R = 重新框选 ROI")
        print("  Q = 退出")
        print(f"\n请放置工件 → 按 T 拍模板")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()

            # 状态提示
            if template_frame is None:
                cv2.putText(display, "Press T to capture TEMPLATE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                status = "Press SPACE to match | R=re-ROI | T=re-template"
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制 ROI 框
            if roi is not None:
                rx, ry, rw, rh = roi
                cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
                cv2.putText(display, f"ROI: {rw}x{rh}px", (rx, ry-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if result_text:
                for i, line in enumerate(result_text.split('\n')):
                    cv2.putText(display, line, (10, 430 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Edge Template Match - Interactive", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('t'):
                template_frame = frame.copy()
                test_frame = None
                result_text = ""
                # 弹出窗口让用户框选 ROI
                roi = _select_roi(template_frame)
                if roi is not None:
                    rx, ry, rw, rh = roi
                    cropped = template_frame[ry:ry+rh, rx:rx+rw]
                    matcher.set_template_from_frame(cropped)
                    print(f"\n✓ 模板已设置 (ROI: {rw}x{rh}px)")
                    print("  移动工件 → 按空格拍测试照片")
                else:
                    # 用户取消 ROI → 使用全图
                    matcher.set_template_from_frame(template_frame)
                    roi = (0, 0, template_frame.shape[1], template_frame.shape[0])
                    print(f"\n✓ 模板已设置 (全图: {template_frame.shape[1]}x{template_frame.shape[0]})")
                    print("  移动工件 → 按空格拍测试照片")
            elif key == ord('r') and template_frame is not None:
                roi = _select_roi(template_frame)
                if roi is not None:
                    rx, ry, rw, rh = roi
                    cropped = template_frame[ry:ry+rh, rx:rx+rw]
                    matcher.set_template_from_frame(cropped)
                    print(f"✓ ROI 已更新: {rw}x{rh}px")
                result_text = ""
            elif key == ord(' ') and template_frame is not None:
                test_frame = frame.copy()

                t0 = time.perf_counter()
                detections = matcher.detect(test_frame, method=args.method)
                elapsed = (time.perf_counter() - t0) * 1000

                if detections:
                    d = detections[0]
                    h, w = test_frame.shape[:2]
                    dx = d['center'][0] - w / 2
                    dy = d['center'][1] - h / 2
                    result_text = (
                        f"Match OK ({elapsed:.0f}ms) score={d['score']:.3f}\n"
                        f"  Center: ({d['center'][0]:.1f}, {d['center'][1]:.1f})\n"
                        f"  Offset from img center: dx={dx:.1f}px dy={dy:.1f}px\n"
                        f"  Rotation: {d['rotation_deg']:.1f} deg\n"
                        f"  Size: {d['size_px']:.0f}px (physical={d['size_mm']:.1f}mm)"
                    )
                    print(f"\n{'='*50}")
                    print(f"匹配完成 ({elapsed:.0f}ms):")
                    print(f"  模板中心: ({w/2:.0f}, {h/2:.0f}) (图像中心)")
                    print(f"  测试中心: ({d['center'][0]:.1f}, {d['center'][1]:.1f})")
                    print(f"  XY偏移:   dx={dx:.1f}px  dy={dy:.1f}px")
                    print(f"  旋转角度: {d['rotation_deg']:.1f}°")
                    print(f"  像素尺寸: {d['size_px']:.0f}px")
                    print(f"  匹配得分: {d['score']:.4f}")
                    print(f"{'='*50}")
                    result_display = draw_detection(test_frame, detections)
                    cv2.imshow("Edge Template Match - Result", result_display)
                    print("\n  关闭结果窗口继续, 或按 T 重新拍模板")
                else:
                    result_text = "NO MATCH - try adjusting threshold or lighting"
                    print(f"\n✗ 未检测到物体 (elapsed={elapsed:.0f}ms)")

        cap.release()
        cv2.destroyAllWindows()
        return 0

    # ── 模式: 实时跟踪 ──
    if args.live:
        cap = open_camera(args.camera or 0)
        if cap is None:
            print("✗ 无法打开相机")
            return 1
        for _ in range(5):
            cap.read()

        print(f"\n实时跟踪 (按 Q 退出, S 更新模板):")
        print(f"  搜索范围: ±{args.angle_range}° 步长{args.angle_step}°")
        fps_counter = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = matcher.detect(frame, method=args.method)
            elapsed = (time.perf_counter() - t0) * 1000
            fps_counter.append(elapsed)

            vis = draw_detection(frame, detections)
            if fps_counter:
                avg_ms = sum(fps_counter[-30:]) / min(len(fps_counter), 30)
                cv2.putText(vis, f"{avg_ms:.0f}ms", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(vis, f"{1000/avg_ms:.0f} FPS", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("Edge Template Tracker", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                matcher.set_template_from_frame(frame)

        cap.release()
        cv2.destroyAllWindows()
        avg_ms = sum(fps_counter) / len(fps_counter) if fps_counter else 0
        print(f"\n平均匹配时间: {avg_ms:.0f}ms ({1000/avg_ms:.0f} FPS)")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    exit(main())