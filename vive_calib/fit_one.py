#!/usr/bin/env python3
"""vive_calib/fit_one.py — 单次采集 CSV 的圆拟合质检（第一道闸）。

用法：
  python fit_one.py sweep_xxx.csv

输出：点数、θ 范围、圆半径、圆心(轴上一点)、轴方向、残差 RMS。
判据：残差 RMS 应 ≈ 1mm（tracker 噪声地板），明显更大 = 该次扫被污染，需重扫。
"""
import csv
import sys

import numpy as np

from circle_fit import fit_circle


def main():
    if len(sys.argv) < 2:
        print("用法: python fit_one.py sweep_xxx.csv")
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        rows = list(csv.reader(f))

    header = rows[0]
    # 期望列：t, theta, tx, ty, tz, qx, qy, qz, qw
    data = np.array([[float(x) for x in r] for r in rows[1:]], dtype=float)

    n = len(data)
    theta = data[:, 1]
    pts = data[:, 2:5]  # tx, ty, tz

    center, normal, radius, rms = fit_circle(pts, resolve_sign=True)

    print(f"文件: {path}")
    print(f"点数: {n}   θ范围: [{theta.min():.1f}, {theta.max():.1f}]°")
    print(f"半径: {radius:.4f} m")
    print(f"圆心(轴上一点): {np.round(center, 4)}")
    print(f"轴方向:        {np.round(normal, 4)}")
    print(f"残差 RMS: {rms * 1e3:.3f} mm")
    print()
    print("判据: 残差 RMS 应 ≈ 1mm（tracker 噪声地板）。")
    print("      明显更大 → 该次扫被污染（其它关节动/遮挡丢锁/振动）→ 查因重扫。")


if __name__ == "__main__":
    main()
