#!/usr/bin/env python3
"""vive_calib/acquire.py — 半自动采集主脚本（阶段 4，跑在 Jetson）。

职责：
  1. 用户先把机器人摆到「零位姿」（θ=0 参考位姿），脚本读当前角作为各关节零点。
  2. 录零位姿 M = 静止平均 T2_in_T0（探针 tracker 相对基座 tracker）。
  3. 目标关节从 lo 慢扫到 hi（其它关节锁零点），同步录 (t, θ, T2_in_T0) 到 CSV。

用法（Jetson 上，lerobot conda 环境；需先跑 x86 的 sender 推流）：
  python acquire.py --config configs/agv_class_ADC2.yaml \
      --joint trunk_joint_1 --lo -20 --hi 20 --speed 5 \
      --base-dev WM0 --probe-dev WM1 --out sweep_trunk_j1.csv

说明：
  - 角度单位：度（与 robot.get_current_position() 一致）。
  - T2_in_T0 = inv(T0) @ T2（4x4），存成 pos(x,y,z) + quat(qx,qy,qz,qw)。
  - base/probe 的 device 名是 receiver 里的 WM0/WM1/WM2，按实际贴片映射。
  - 注意：robot.get_current_position() 每次会 print（既有行为，无害，CSV 不受影响）。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from lerobot.model.skeleton import quat_to_rot, rot_to_quat  # noqa: E402
from receiver import PoseReceiver  # noqa: E402  (同目录)


# --------------------------------------------------------------------------- #
# 位姿工具
# --------------------------------------------------------------------------- #

def pose_to_matrix(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*quat)
    T[:3, 3] = pos
    return T


def matrix_to_pose(T) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, 3], rot_to_quat(T[:3, :3])


def rel_pose(latest: dict, base_dev: str, probe_dev: str) -> np.ndarray:
    """T2_in_T0 = inv(T0) @ T2。latest[dev] = (ts, seq, pos, quat)。"""
    _, _, p0, q0 = latest[base_dev]
    _, _, p2, q2 = latest[probe_dev]
    return np.linalg.inv(pose_to_matrix(p0, q0)) @ pose_to_matrix(p2, q2)


# --------------------------------------------------------------------------- #
# 机器人控制
# --------------------------------------------------------------------------- #

def move_to(robot, obs_names, targets: dict, tol: float = 1.0, timeout: float = 60.0) -> bool:
    """命令机器人到目标位姿（各关节绝对角，度）。send_action 每步有安全钳位，需循环逼近。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        cur = robot.get_current_position()
        if max(abs(targets[n] - cur[n]) for n in obs_names) <= tol:
            return True
        robot.send_action({f"{n}.pos": targets[n] for n in obs_names})
        time.sleep(0.03)
    return False


def record_zero_pose(rx: PoseReceiver, base_dev: str, probe_dev: str,
                     duration: float = 3.0):
    """静止平均零位姿 M = T2_in_T0。返回 (pos(3,), quat(4,)) 或 None。"""
    Ts = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        rx.poll(timeout=0.005)
        latest = rx.get_latest()
        if base_dev in latest and probe_dev in latest:
            Ts.append(rel_pose(latest, base_dev, probe_dev))
    if not Ts:
        return None
    Ts = np.array(Ts)
    pos = Ts[:, :3, 3].mean(axis=0)
    R = Ts[:, :3, :3].mean(axis=0)          # 姿态：旋转矩阵平均后正交化
    U, _, Vt = np.linalg.svd(R)
    quat = rot_to_quat(U @ Vt)
    return pos, quat


def sweep(robot, rx, joint: str, obs_names, neutral: dict,
          lo: float, hi: float, speed: float,
          base_dev: str, probe_dev: str, csv_writer, settle: float = 0.5):
    """目标关节「相对零位」从 lo 慢扫到 hi，其它锁 neutral，录 (t, θ相对, T2_in_T0)。"""
    dt = 0.05  # ~20 Hz 控制
    zero = neutral[joint]  # 该关节零位编码器角

    targets = dict(neutral)
    targets[joint] = zero + lo
    print(f"  到位 lo={lo}°（编码器 {zero + lo:.1f}°）...")
    move_to(robot, obs_names, targets)
    time.sleep(settle)

    print(f"  扫 {lo}° -> {hi}°（相对零位）@ {speed}°/s ...")
    t0 = time.monotonic()
    target = lo
    while target <= hi:
        robot.send_action({f"{n}.pos": (zero + target if n == joint else neutral[n])
                           for n in obs_names})
        theta = robot.get_current_position()[joint] - zero  # 相对角（FK 用这个）
        rx.poll(timeout=0.001)
        latest = rx.get_latest()
        if base_dev in latest and probe_dev in latest:
            T = rel_pose(latest, base_dev, probe_dev)
            pos, quat = matrix_to_pose(T)
            csv_writer.writerow([time.monotonic() - t0, theta, *pos, *quat])
        target += speed * dt
        time.sleep(dt)

    # 回零位：保证下一个关节从同一零位起扫（否则零位漂移、几何不一致）。
    # 用紧容差——近端关节残留 0.1° 就会把远端轴顶偏 ~0.5mm。
    print("  回零位 ...")
    move_to(robot, obs_names, neutral, tol=0.1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="单关节慢扫 + 录 VIVE 位姿 CSV + 零位姿 M")
    ap.add_argument("--config", required=True, help="机器人 YAML 配置")
    ap.add_argument("--joint", required=True, help="要扫的关节名，如 trunk_joint_1")
    ap.add_argument("--lo", type=float, default=-30.0, help="扫程下限（相对零位，度）")
    ap.add_argument("--hi", type=float, default=30.0, help="扫程上限（相对零位，度）")
    ap.add_argument("--speed", type=float, default=5.0, help="扫速（度/秒）")
    ap.add_argument("--base-dev", default="WM0", help="基座 tracker 的 device 名")
    ap.add_argument("--probe-dev", default="WM1", help="探针 tracker 的 device 名")
    ap.add_argument("--port", type=int, default=12345, help="UDP 端口")
    ap.add_argument("--out", required=True, help="CSV 输出路径")
    ap.add_argument("--m-out", default=None, help="M(零位姿) JSON 输出路径")
    args = ap.parse_args()

    # 机器人
    from lerobot.tasks.config import load_config_from_yaml
    from lerobot.robots import make_robot_from_config
    cfg = load_config_from_yaml(args.config)
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()

    obs_names = robot.observation_joint_names
    if args.joint not in obs_names:
        sys.exit(f"关节 {args.joint} 不在观测关节 {obs_names}")

    lo, hi = args.lo, args.hi

    # 接收器
    rx = PoseReceiver("0.0.0.0", args.port)
    print(f"监听 UDP :{args.port}")

    # 零位姿：读当前角作为 neutral（用户已摆到 θ=0）
    neutral = robot.get_current_position()
    print(f"中性位姿(θ=0)读自当前角: { {n: round(v, 1) for n, v in neutral.items()} }")
    zero = neutral[args.joint]
    print(f"扫程: {args.joint} 相对零位 [{lo}, {hi}]° "
          f"= 编码器 [{zero + lo:.1f}, {zero + hi:.1f}]° @ {args.speed}°/s")

    # 录 M
    print("录零位姿 M（静止 3s 平均）...")
    m = record_zero_pose(rx, args.base_dev, args.probe_dev, duration=3.0)
    if m is None:
        print("⚠ 未收到 tracker 位姿——检查 x86 sender 是否在推流、device 名对不对")
    else:
        pos, quat = m
        print(f"M = pos {np.round(pos, 3)}  quat {np.round(quat, 4)}")
        if args.m_out:
            with open(args.m_out, "w") as f:
                json.dump({"pos": pos.tolist(), "quat": quat.tolist(),
                           "zero_joint_offsets": {k: float(v) for k, v in neutral.items()}},
                          f, indent=2)
            print(f"M 已写 {args.m_out}")

    # 扫 + 录 CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "theta", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        sweep(robot, rx, args.joint, obs_names, neutral, lo, hi, args.speed,
              args.base_dev, args.probe_dev, w)
    print(f"CSV 已写 {args.out}")

    try:
        robot.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
