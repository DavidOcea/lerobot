"""vive_calib/synthetic_closed_loop.py — 阶段 1 端到端合成数据闭环。

流程：
  造一条已知 screw 轴的假臂 → 全局 PoE FK 生成带噪 tracker 轨迹
  （单关节扫 + 多关节随机位姿）→ 圆拟合恢复轴 → 装配 Skeleton → FK 预测 vs 真值。

验收（对齐 VIVE ~1mm 噪声地板）：
  - 恢复轴与真轴夹角 ≈ 噪声级（应 ≪ 1°）
  - 圆心到真轴线的距离 ≈ 噪声级
  - 多关节随机位姿下 FK 预测位置误差 RMS < 1mm

用法（用 lerobot 的 conda 环境）：
  /root/miniconda3/envs/lerobot/bin/python synthetic_closed_loop.py
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from lerobot.model.skeleton import (  # noqa: E402
    Skeleton, Element, Transform, forward_kinematics, screw_transform,
)
from circle_fit import fit_circle  # noqa: E402


def make_fake_arm(n_joints: int, seed: int = 0):
    """随机造一条 n 关节假臂。返回 (joints, M)。joints = [(axis, point)]，基座系。"""
    rng = np.random.default_rng(seed)
    joints = []
    for _ in range(n_joints):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        point = rng.normal(size=3) * 0.3  # 轴上一点（米）
        joints.append((axis, point))
    qx, qy, qz = 0.1, 0.2, 0.3
    qw = float(np.sqrt(1 - qx * qx - qy * qy - qz * qz))
    M = Transform(x=0.5, y=0.1, z=0.4, qx=qx, qy=qy, qz=qz, qw=qw)
    return joints, M


def fk_true(joints, M, angles_deg):
    """真值 FK：T = exp(S0θ0)…exp(Snθn)·M。angles_deg 按关节顺序。"""
    T = M.matrix()
    for (axis, point), a in zip(reversed(joints), reversed(angles_deg)):
        T = screw_transform(axis, point, np.deg2rad(a)) @ T
    return T


def main():
    n_joints = 7
    noise_sigma = 1e-3       # 1mm 噪声（VIVE 噪声地板）
    sweep_deg = 80.0         # 单关节扫 ±80°（贴近真实关节全范围）
    n_sweep = 800            # 每度 5 点 × 160°（真实采集是每度 10~20 点，这里仍偏保守）
    n_verify = 100

    joints, M = make_fake_arm(n_joints, seed=0)
    rng = np.random.default_rng(1)

    # --- 单关节扫 + 圆拟合，恢复每关节 (point, axis) ---
    recovered = []
    print("=== 单关节圆拟合恢复质量 ===")
    for i in range(n_joints):
        thetas = np.linspace(-sweep_deg, sweep_deg, n_sweep)  # 递增 → 方向可定号
        pts = []
        for th in thetas:
            others = [0.0] * n_joints
            others[i] = th
            p = fk_true(joints, M, others)[:3, 3]
            pts.append(p + rng.normal(scale=noise_sigma, size=3))
        pts = np.array(pts)

        center, normal, radius, rms = fit_circle(pts, resolve_sign=True)
        recovered.append((center, normal))

        true_axis, true_point = joints[i]
        ang = np.degrees(np.arccos(np.clip(np.dot(normal, true_axis), -1, 1)))  # 0°=对齐,180°=反向
        axis_dist = np.linalg.norm(np.cross(center - true_point, true_axis))    # 圆心到真轴线距离
        print(f"joint {i}: 轴夹角={ang:6.3f}°  圆心到真轴距={axis_dist*1e3:6.4f}mm  "
              f"半径={radius:5.3f}m  rms={rms*1e3:6.4f}mm")

    # --- 装配 fitted Skeleton ---
    sk = Skeleton(name="fitted")
    for i, (center, normal) in enumerate(recovered):
        e = Element(id=i, name=f"j{i}", kind="joint", parent=(i - 1) if i else None)
        e.axis = normal
        e.point = center
        sk.elements.append(e)
    # M 测量：零位姿静止，平均多次压低噪声（~90Hz 采几秒，现实做法；单次采会带 ~1.7mm 固定偏移）
    n_m = 500
    m_pos = np.array([M.x, M.y, M.z]) + rng.normal(scale=noise_sigma, size=(n_m, 3)).mean(axis=0)
    sk.M = Transform(x=m_pos[0], y=m_pos[1], z=m_pos[2], qx=M.qx, qy=M.qy, qz=M.qz, qw=M.qw)

    # --- FK 验证：多关节随机位姿 ---
    errs = []
    for _ in range(n_verify):
        angles = rng.uniform(-30, 30, size=n_joints)
        T_fitted = forward_kinematics(sk, {f"j{i}": angles[i] for i in range(n_joints)})
        T_true = fk_true(joints, M, angles)
        errs.append(np.linalg.norm(T_fitted[:3, 3] - T_true[:3, 3]))
    errs = np.array(errs)
    print(f"\n=== FK 验证（{n_verify} 个随机位姿）===")
    print(f"位置误差 RMS = {errs.mean()*1e3:.4f}mm   max = {errs.max()*1e3:.4f}mm")
    print(f"阈值：RMS < 1mm  ->  {'✅ 通过' if errs.mean() < 1e-3 else '❌ 超阈值'}")


if __name__ == "__main__":
    main()
