#!/usr/bin/env python3
"""Motor oscillation stress test — sustained back-and-forth at max speed.

Moves each joint ±angle repeatedly while live-monitoring force
feedback.  Stops if the joint stalls (|target - actual| grows) or
force exceeds a safety threshold.

Usage:
    python test_motor_oscillation.py --config configs/agv_class_ADC2.yaml
    python test_motor_oscillation.py --config configs/agv_class_ADC2.yaml --joint left_arm_joint_4 --cycles 30
"""

import argparse, sys, time, math, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Motor oscillation stress test")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--angle", type=float, default=10.0,
                        help="Oscillation amplitude in degrees (default: 10)")
    parser.add_argument("--cycles", type=int, default=20,
                        help="Number of full back-and-forth cycles per joint")
    parser.add_argument("--joint", type=str, default=None,
                        help="Single joint to test (default: all arm joints)")
    parser.add_argument("--force-limit", type=float, default=3.0,
                        help="Force limit Nm — stop if exceeded")
    parser.add_argument("--position-lag", type=float, default=3.0,
                        help="Position error ° — stop if |target-actual| exceeds this")
    parser.add_argument("--cooldown", type=float, default=1.5,
                        help="Cooldown seconds between joints")
    parser.add_argument("--report", type=str, default="oscillation_report.json",
                        help="Output JSON report path")
    parser.add_argument("--settle", type=float, default=0.3,
                        help="Settle threshold in degrees (default: 0.3)")
    args = parser.parse_args()

    from lerobot.tasks.config import load_config_from_yaml
    from lerobot.robots import make_robot_from_config

    print(f"Loading config: {args.config}")
    cfg = load_config_from_yaml(args.config)
    # Strip cameras — test scripts only use get_current_position() (motors),
    # don't need cameras at all.  This lets the script run on robots with
    # any number of cameras without failing on missing device indices.
    if hasattr(cfg.robot_config, 'cameras'):
        cfg.robot_config.cameras = {}
    robot = make_robot_from_config(cfg.robot_config)
    robot.connect()
    print("Robot connected.\n")

    joint_names = robot.observation_joint_names

    # Select test joints
    skip = {"joint_7", "trunk"}
    if args.joint:
        test_joints = [args.joint]
    else:
        test_joints = [j for j in joint_names if not any(s in j.lower() for s in skip)]

    print(f"Testing {len(test_joints)} joints: ±{args.angle}° × {args.cycles} cycles each")
    print(f"Force limit: {args.force_limit}Nm  |  Position lag limit: {args.position_lag}°")
    print(f"{'='*70}\n")

    report = {"config": args.config, "angle": args.angle, "cycles": args.cycles,
              "results": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for joint_name in test_joints:
        initial_state = robot.get_current_position()
        if joint_name not in initial_state:
            print(f"  SKIP {joint_name}: not in current_position")
            continue

        center = initial_state[joint_name]
        joint_ids = robot.observation_joint_names
        joint_idx = joint_ids.index(joint_name)
        print(f"\n  ┌─ {joint_name} (center={center:.2f}°)")

        peak_forces = []       # all force readings
        peak_position_errors = []
        stall_events = 0
        stopped_early = False
        stop_reason = ""

        start_time = time.perf_counter()

        for cycle in range(args.cycles):
            # ── Move to +angle ──
            high_target = center + args.angle
            targets = [initial_state.get(j, center) for j in joint_ids]
            targets[joint_idx] = high_target
            robot.send_target_position(targets)
            time.sleep(0.05)  # give CSP a head start

            # Monitor until settled or stalled
            high_ok = _monitor_move(robot, joint_name, high_target,
                                    joint_ids, args, joint_idx,
                                    peak_forces, peak_position_errors)
            if not high_ok:
                stall_events += 1
                stop_reason = f"spike at +{args.angle}° (cycle {cycle+1})"
                stopped_early = True
                break

            # ── Move to -angle ──
            low_target = center - args.angle
            targets = [initial_state.get(j, center) for j in joint_ids]
            targets[joint_idx] = low_target
            robot.send_target_position(targets)
            time.sleep(0.05)

            low_ok = _monitor_move(robot, joint_name, low_target,
                                   joint_ids, args, joint_idx,
                                   peak_forces, peak_position_errors)
            if not low_ok:
                stall_events += 1
                stop_reason = f"spike at -{args.angle}° (cycle {cycle+1})"
                stopped_early = True
                break

            # Progress
            if (cycle + 1) % 5 == 0:
                elapsed = time.perf_counter() - start_time
                pf = max(peak_forces[-10:]) if peak_forces else 0
                print(f"    cycle {cycle+1}/{args.cycles}  "
                      f"peak_force={pf:.2f}Nm  pos_err={max(peak_position_errors[-10:]):.2f}°  "
                      f"elapsed={elapsed:.1f}s")

        elapsed = time.perf_counter() - start_time
        completed = args.cycles if not stopped_early else cycle

        # ── Return to center ──
        targets = [initial_state.get(j, center) for j in joint_ids]
        targets[joint_idx] = center
        robot.send_target_position(targets)
        time.sleep(0.5)

        # ── Report ──
        max_force = max(peak_forces) if peak_forces else 0
        avg_force = sum(peak_forces) / len(peak_forces) if peak_forces else 0
        max_lag = max(peak_position_errors) if peak_position_errors else 0

        status = "✓ STABLE" if not stopped_early else f"✗ STOPPED ({stop_reason})"
        print(f"  └─ {status}")
        print(f"     completed: {completed}/{args.cycles} cycles")
        print(f"     max_force: {max_force:.2f}Nm  avg_force: {avg_force:.2f}Nm")
        print(f"     max_position_lag: {max_lag:.2f}°")
        print(f"     elapsed: {elapsed:.1f}s")

        report["results"][joint_name] = {
            "center": round(center, 2),
            "completed_cycles": completed,
            "target_cycles": args.cycles,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "stall_events": stall_events,
            "max_force_Nm": round(max_force, 3),
            "avg_force_Nm": round(avg_force, 3),
            "max_position_lag_deg": round(max_lag, 2),
            "elapsed_s": round(elapsed, 1),
            "avg_cycle_hz": round(completed * 2 / elapsed, 1) if elapsed > 0 else 0,
        }

        # Cooldown between joints
        if joint_name != test_joints[-1]:
            print(f"    (cooldown {args.cooldown}s...)")
            time.sleep(args.cooldown)

    robot.disconnect()

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  STRESS TEST SUMMARY")
    print(f"  {'Joint':<24} {'Cycles':>7} {'Status':>12} {'Max F(Nm)':>10} {'Max Lag(°)':>11}")
    print(f"  {'─'*24} {'─'*7} {'─'*12} {'─'*10} {'─'*11}")
    all_stable = True
    for j, r in report["results"].items():
        s = "✓ STABLE" if not r["stopped_early"] else "✗ " + r["stop_reason"][:20]
        if r["stopped_early"]:
            all_stable = False
        print(f"  {j:<24} {r['completed_cycles']:>6}/{r['target_cycles']} "
              f"{s:>12} {r['max_force_Nm']:>9.2f} {r['max_position_lag_deg']:>9.2f}°")
    print(f"  {'─'*24} {'─'*7} {'─'*12} {'─'*10} {'─'*11}")
    print(f"  Overall: {'ALL STABLE ✓' if all_stable else 'SOME JOINTS TRIGGERED PROTECTION ✗'}")
    print(f"\n  Report saved → {args.report}")

    # Quick YAML speed reference
    stable = {j: r for j, r in report["results"].items() if not r["stopped_early"]}
    if stable:
        print(f"\n  Speed ref (stable joints only):")
        for j, r in sorted(stable.items()):
            hz = r.get("avg_cycle_hz", 0)
            print(f"    {j}: {hz:.1f} half-cycles/s  (≈{hz*args.angle:.0f}°/s avg sweep)")


def _monitor_move(robot, joint_name, target, joint_ids, args, joint_idx,
                  peak_forces_out, peak_errs_out):
    """Monitor a single move until settled or unsafe. Returns True if OK."""
    start = time.perf_counter()
    poll_count = 0
    while time.perf_counter() - start < args.cooldown:
        # Fast path — position only (no cameras, ~0.3ms)
        cur_pos = robot.get_current_position()
        actual = cur_pos.get(joint_name, target)
        error = abs(actual - target)
        peak_errs_out.append(error)

        # Slow path — force check every 4th poll (~40ms)
        force = 0.0
        poll_count += 1
        if poll_count % 4 == 0:
            try:
                full_obs = robot.get_observation()
                force = full_obs.get(f"{joint_name}.force", 0.0)
            except Exception:
                force = 0.0
        peak_forces_out.append(abs(force))

        # ── Safety checks ──
        if abs(force) > args.force_limit:
            print(f"    ⚠ FORCE LIMIT: {joint_name} {abs(force):.2f}Nm > {args.force_limit}Nm")
            _robot_stop(robot)
            return False
        if error > args.position_lag:
            print(f"    ⚠ POSITION LAG: {joint_name} |target-actual|={error:.2f}° > {args.position_lag}°")
            _robot_stop(robot)
            return False

        # Settled
        if error < args.settle:
            return True

        time.sleep(0.01)

    return True  # fine but didn't settle within timeout — not a safety failure


def _robot_stop(robot):
    """Try to stop the robot gracefully."""
    try:
        # Send current position as target to halt movement
        pos = robot.get_current_position()
        jids = robot.observation_joint_names
        targets = [pos.get(j, 0) for j in jids]
        robot.send_target_position(targets)
    except Exception:
        pass


if __name__ == "__main__":
    main()
