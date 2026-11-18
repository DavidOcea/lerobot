"""vive_calib/circle_fit.py — 3D 圆拟合（VIVE 标定的核心几何算子）。

输入：单关节慢扫得到的 tracker 位置序列（基座系，按 θ 递增排序）。
输出：圆心（轴上一点）、轴方向（单位法向，符号由扫描方向定）、半径、RMS 残差。

算法：SVD 平面拟合 → 投影到平面 → 2D 代数圆拟合（Kasa 法）→ 反投影。
"""
import numpy as np


def _orthonormal_basis(n: np.ndarray) -> np.ndarray:
    """给定平面法向 n，返回平面内的两个正交单位基向量，shape (2, 3)。"""
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u0 = np.cross(n, ref)
    u0 /= np.linalg.norm(u0)
    u1 = np.cross(n, u0)
    u1 /= np.linalg.norm(u1)
    return np.stack([u0, u1])


def _fit_circle_2d(xy: np.ndarray):
    """2D 圆拟合：Kasa 代数法给初值，再 Gauss-Newton 细化到几何距离最小。"""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2.0, -E / 2.0
    r = float(np.sqrt(max(cx * cx + cy * cy - F, 0.0)))

    # Gauss-Newton：最小化几何距离 Σ(‖(x,y)-(cx,cy)‖ - r)²
    for _ in range(30):
        dx = x - cx
        dy = y - cy
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-12)
        res = d - r
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        cx += step[0]
        cy += step[1]
        r += step[2]
        if np.linalg.norm(step) < 1e-14:
            break
    return cx, cy, r


def fit_circle(points, resolve_sign: bool = True):
    """拟合 3D 圆。

    参数
    ----
    points : (N, 3) 按扫描方向（θ 递增）有序的点。
    resolve_sign : True 时用扫描方向（右手定则）定轴的正负号。

    返回
    ----
    (center, normal, radius, rms)
      center : (3,) 圆心 = 轴上一点
      normal : (3,) 单位轴方向
      radius : float 半径
      rms    : float 点到圆的 RMS 距离
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        raise ValueError("至少需要 3 个点才能拟合圆")

    c0 = P.mean(axis=0)
    Pc = P - c0

    # 平面拟合：最小奇异值对应的右奇异向量 = 法向
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)

    # 平面基 + 投影到 2D
    u = _orthonormal_basis(normal)
    xy = Pc @ u.T
    cx, cy, r = _fit_circle_2d(xy)

    # 3D 圆心
    center = c0 + cx * u[0] + cy * u[1]

    # RMS 残差
    dist = np.linalg.norm(P - center, axis=1)
    rms = float(np.sqrt(np.mean((dist - r) ** 2)))

    # 用扫描方向定轴正负号：速度 v ≈ ω × (p0 - center) 应沿切线方向。
    # 用相隔 ~1/8 扫程的两点估切线（密采样下相邻点差 ~1mm 会被噪声淹没）。
    if resolve_sign and P.shape[0] >= 2:
        step = max(1, P.shape[0] // 8)
        tangent = P[step] - P[0]
        r_vec = P[0] - center
        if np.dot(np.cross(normal, r_vec), tangent) < 0:
            normal = -normal

    return center, normal, float(r), rms
