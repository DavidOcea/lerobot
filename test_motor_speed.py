#!/usr/bin/env python3
"""Motor speed benchmark — small-angle (±15°) slew rate test.

Takes each joint one at a time, sends a 15° step from the current
position, and measures how long the motor takes to settle.
Safe: small angle, single joint, sub-second.

Usage:
    python test_motor_speed.py --config configs/agv_class_ADC2.yaml
"""

import argparse
import sys
import time
import math
from pathlib import Path

# ── Connect to robot ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Motor speed benchmark")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--angle", type=float, default=15.0, help="Step angle in degrees (default: 15)")
    parser.add_argument("--settle", type=float, default=0.05, help="Settle threshold in degrees (default: 0.05)")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-joint timeout in seconds")
    args = parser.parse_args()

    from lerobot.tasks.config import load_config_from_yaml
    from lerobot.robots import make_robot_from_config

    print(f"Loading config: {args.config}")
    cfg = load_config_from_yaml(args.config)
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print("Robot connected.\n")

    joint_names = robot.observation_joint_names
    # Exclude gripper and trunk joints — not relevant for speed profiling
    skip = {"joint_7", "trunk"}
    test_joints = [j for j in joint_names if not any(s in j.lower() for s in skip)]
    print(f"Testing {len(test_joints)} joints with ±{args.angle}° step\n")

    results = {}

    for joint_name in test_joints:
        initial = robot.get_current_position()
        if joint_name not in initial:
            print(f"  SKIP {joint_name}: not in current_position")
            continue

        start_pos = initial[joint_name]

        # Build a single-joint action: copy current, offset ONE joint
        action = {}
        for j in joint_names:
            if j in initial:
                action[f"{j}.pos"] = initial[j]
        target = start_pos + args.angle
        action[f"{joint_name}.pos"] = target

        # Send and time
        t0 = time.perf_counter()
        robot.send_action(action)

        # Poll position until settled
        peak_speed = 0.0
        last_pos = start_pos
        last_time = t0
        converged = False
        while time.perf_counter() - t0 < args.timeout:
            cur = robot.get_current_position()
            actual = cur.get(joint_name, target)
            error = abs(actual - target)
            dt = time.perf_counter() - last_time
            if dt > 0.001:
                speed = abs(actual - last_pos) / dt  # deg/s
                if speed > peak_speed:
                    peak_speed = speed
                last_pos = actual
                last_time = time.perf_counter()
            if error < args.settle:
                converged = True
                break
            time.sleep(0.005)

        elapsed = time.perf_counter() - t0
        final = robot.get_current_position().get(joint_name, target)
        actual_move = final - start_pos
        avg_speed = abs(actual_move) / elapsed if elapsed > 0 else 0

        status = "✓" if converged else "✗TIMEOUT"
        print(f"  {joint_name:<24} {status}  "
              f"avg={avg_speed:>6.0f}°/s  peak={peak_speed:>6.0f}°/s  "
              f"moved={actual_move:>+5.1f}°  in {elapsed*1000:.0f}ms")

        results[joint_name] = {
            "converged": converged,
            "avg_speed": avg_speed,
            "peak_speed": peak_speed,
            "elapsed_ms": elapsed * 1000,
        }

        # Return to start position
        action[f"{joint_name}.pos"] = start_pos
        robot.send_action(action)
        time.sleep(0.3)

    robot.disconnect()
    print("\nDone.\n")

    # ── Summary ──
    if not results:
        print("No results collected.")
        return

    converged = [v["avg_speed"] for v in results.values() if v["converged"]]
    if converged:
        print(f"  Min speed:  {min(converged):.0f}°/s")
        print(f"  Max speed:  {max(converged):.0f}°/s")
        print(f"  Mean speed: {sum(converged)/len(converged):.0f}°/s")
        print()

    # ── YAML max_duration reference table ──
    print(f"  {'Joint':<24} {'Speed':>8} {'5°→':>8} {'10°→':>8} {'20°→':>8} {'40°→':>8}")
    print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    slowest_speed = float('inf')
    for jname in sorted(results):
        r = results[jname]
        spd = r["avg_speed"]
        if spd < slowest_speed:
            slowest_speed = spd
        def dur(deg):
            return f"{deg/spd:.1f}s" if spd > 0 else "?"
        print(f"  {jname:<24} {spd:>6.0f}°/s {dur(5):>8} {dur(10):>8} {dur(20):>8} {dur(40):>8}")
    print()

    # ── Copy-paste YAML snippet ──
    print(f"  Suggested position task max_duration (based on slowest joint {slowest_speed:.0f}°/s + 50% margin):")
    print(f"    position_tolerance: 2.0")
    for deg in (5, 10, 20, 40):
        t = deg / slowest_speed * 1.5 if slowest_speed > 0 else 1.0
        print(f"    # {deg}° move → max_duration: {max(t, 0.5):.1f}")
    print(f"")
    print(f"  → speed_multiplier 安全上限 ≈ {slowest_speed/15:.1f}x")
    print(f"  Tip: 最慢关节的速度决定整组动作的下限, max_duration 按它计算。")


if __name__ == "__main__":
    main()
