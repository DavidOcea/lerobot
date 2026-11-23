#!/usr/bin/env python3
"""Run the *production* ``EyouMotorHardware.compliant_hold`` on the robot.

Run ON THE ROBOT MACHINE.

Why this exists
---------------
``eyou_probe_cst.py --soft-hold`` re-implements the software-spring loop
*inline* to validate the CST math and hardware. The math is now production code
in ``cst_soft_spring.py`` + ``EyouMotorHardware.compliant_hold``. This script
drives that production path directly (no re-implementation), so it verifies the
two things the old probe can never touch:

    1. ``stop_drift_deg`` early stop -- the joint returns normally once the box
       has conformed it to the target drift.
    2. ``hold_on_stop=True`` -- instead of ``disable()`` (limp), the joint
       re-latches to CSP position-hold at its final (conformed) position. This
       CST->CSP re-latch sequence is the one hardware interaction not covered
       by any unit test and MUST be confirmed on the robot.

It constructs a single-joint ``EyouMotorHardware`` via ``init()``, then calls
``hw.compliant_hold(0, ...)``. The hold itself is silent (the production
primitive does not print); this script reports state before and after and
classifies how the hold ended.

Safety
------
- Run with the robot IDLE, arm in a safe pose, person at e-stop.
- LIGHT-LOAD joint only. Preferred: arm joint 5 (node 15 right / 25 left).
  joint 6/7 (node 16/17/26/27) are the SECOND batch of motors (TPDO feedback
  dead -> slow raw-SDO 0x6064 read, revision placeholder); CST is NOT verified
  on them yet -- verify CST there first before using them for compliant hold.
- The waist (node 1/2) carries the whole robot; a limp fall would topple it.
  This script refuses node 1/2 unless ``--allow-waist`` is passed.

Usage
-----
    # touch feel only (disable on end), 10 s
    python3 scripts/eyou_probe_compliant_hold.py --node-id 15 --seconds 10

    # verify early-stop + CST->CSP re-latch: push the joint ~3 deg during the
    # hold; it should stop early and come back to CSP (0x6061 -> 8), NOT disable
    python3 scripts/eyou_probe_compliant_hold.py --node-id 15 --kp 20 --kd 5 \
        --stop-drift-deg 3.0 --seconds 10 --hold-on-stop
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import eu_motor_py  # noqa: F401  (guards the rest of the script)
except ImportError as exc:  # pragma: no cover - dev environment
    sys.exit(
        "eu_motor_py is not importable. This script must run on the robot "
        f"machine.\nImportError: {exc}"
    )

from lerobot.motors.eyou.cst_soft_spring import CompliantDriftTrip
from lerobot.motors.eyou.eyou_motor_hardware import EyouMotorHardware

# robot1 node id -> global joint name (see profiles/robot1_trunk.yaml).
_NODE_NAMES = {
    1: "trunk_joint_1",
    2: "trunk_joint_2",
    11: "right_arm_joint_1",
    12: "right_arm_joint_2",
    13: "right_arm_joint_3",
    14: "right_arm_joint_4",
    15: "right_arm_joint_5",
    16: "right_arm_joint_6",
    17: "right_arm_joint_7",
    18: "right_arm_joint_8",
    21: "left_arm_joint_1",
    22: "left_arm_joint_2",
    23: "left_arm_joint_3",
    24: "left_arm_joint_4",
    25: "left_arm_joint_5",
    26: "left_arm_joint_6",
    27: "left_arm_joint_7",
    28: "left_arm_joint_8",
}


def read_u8_safe(motor, index, sub=0):
    try:
        return motor.read_u8(index, sub)
    except Exception:
        return None


def read_u16_safe(motor, index, sub=0):
    try:
        return motor.read_u16(index, sub)
    except Exception:
        return None


def _statusword_summary(sw):
    """Summarize the CiA 402 statusword 0x6041.

    The decisive bit for "is the drive still energised" is bit 2 (operation
    enabled): after disable() it should read 0. A joint that reads 0 here but
    still "can't be pushed" is held by harmonic-drive stiction, not the drive.
    """
    if sw is None:
        return "unreadable"
    return f"0x{sw:04X} (operation_enabled={int(bool(sw & 0x04))})"


def _mode_name(mode):
    return {8: "CSP", 10: "CST", 11: "MIT", 13: "CSF"}.get(mode, str(mode))


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--node-id", type=int, default=15,
                   help="CANopen node ID (default 15 = right_arm_joint_5, light load).")
    p.add_argument("--device", type=int, default=1, help="CAN device index (default 1).")
    p.add_argument("--can-baud-rate", default="1M",
                   choices=["1M", "500K", "250K"], help="CAN baud rate (default 1M).")

    p.add_argument("--kp", type=float, default=20.0,
                   help="Spring gain, per-thousand torque per degree (default 20).")
    p.add_argument("--kd", type=float, default=5.0,
                   help="Damping, per-thousand torque per deg/s (default 5).")
    p.add_argument("--vel-filter", type=float, default=0.3,
                   help="Velocity low-pass alpha, 0-1 (default 0.3).")
    p.add_argument("--spring-limit", type=float, default=300.0,
                   help="Spring torque clamp, per-thousand (default 300).")
    p.add_argument("--damp-limit", type=float, default=300.0,
                   help="Damping torque clamp, per-thousand (default 300).")
    p.add_argument("--total-limit", type=float, default=500.0,
                   help="Final total torque cap, per-thousand (default 500).")
    p.add_argument("--max-drift-deg", type=float, default=15.0,
                   help="Safety trip if drift exceeds this (deg).")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="Max hold duration (default 10).")
    p.add_argument("--loop-ms", type=int, default=20,
                   help="Loop period in ms (default 20).")
    p.add_argument("--stop-drift-deg", type=float, default=None,
                   help="Normal early-stop drift threshold (deg). Push the joint "
                        "past this to verify early stop. None = timeout only.")
    p.add_argument("--hold-on-stop", action="store_true",
                   help="Re-latch to CSP position-hold on end instead of disable "
                        "(the key CST->CSP re-latch to verify).")
    p.add_argument("--allow-waist", action="store_true",
                   help="Override the waist (node 1/2) refusal. NOT recommended.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    node_id = args.node_id
    joint_name = _NODE_NAMES.get(node_id, f"node_{node_id}")

    if node_id in (1, 2) and not args.allow_waist:
        sys.exit(
            f"[refusing] node {node_id} is the waist ({joint_name}); it carries "
            "the whole robot and a limp fall would topple it. Use a light-load "
            "arm joint (15/25 = joint_5) instead, or pass --allow-waist to "
            "override at your own risk."
        )

    print("=" * 72)
    print("  WARNING: this ENABLES the joint in CST torque mode,")
    print("  then applies the production compliant_hold primitive.")
    print("  Arm in a safe pose, person at e-stop, light-load joint only.")
    print("=" * 72)
    time.sleep(1.0)

    # Build a single-joint EyouMotorHardware through its real init() path.
    config = {
        "can_device_index": args.device,
        "can_baud_rate": args.can_baud_rate,
        "joints": [{"name": joint_name, "parameters": {"node_id": node_id}}],
    }
    hw = EyouMotorHardware()
    if not hw.init(config):
        sys.exit("[error] EyouMotorHardware.init failed")
    motor = hw.motor_nodes_[0]

    theta0 = motor.get_position()
    mode_before = read_u8_safe(motor, 0x6061)
    sw_before = read_u16_safe(motor, 0x6041)
    print(f"\nJoint: {joint_name} (node {node_id})")
    print(f"Reference theta0 = {theta0:.3f} deg (still in CSP)")
    print(f"Mode before    = {_mode_name(mode_before)} (0x6061={mode_before})")
    print(f"Statusword before = {_statusword_summary(sw_before)}")
    print(f"Spring: kp={args.kp}, kd={args.kd}, vel_alpha={args.vel_filter}, "
          f"limits {args.spring_limit}/{args.damp_limit}/{args.total_limit}")
    print(f"max_drift={args.max_drift_deg} deg, stop_drift={args.stop_drift_deg} deg, "
          f"seconds={args.seconds}, hold_on_stop={args.hold_on_stop}\n")
    if args.stop_drift_deg is not None:
        print(f">>> During the hold, gently push the joint past "
              f"{args.stop_drift_deg} deg to trigger the early stop.\n")
    else:
        print(">>> No stop_drift_deg set: hold runs to timeout (touch feel only).\n")

    try:
        final_drift = hw.compliant_hold(
            0,
            kp=args.kp,
            kd=args.kd,
            vel_alpha=args.vel_filter,
            spring_limit=args.spring_limit,
            damp_limit=args.damp_limit,
            total_limit=args.total_limit,
            max_drift_deg=args.max_drift_deg,
            seconds=args.seconds,
            loop_ms=args.loop_ms,
            stop_drift_deg=args.stop_drift_deg,
            hold_on_stop=args.hold_on_stop,
        )
        if args.stop_drift_deg is not None and abs(final_drift) >= args.stop_drift_deg:
            how = "early stop (stop_drift_deg reached)"
        else:
            how = "timeout"
        print(f"\n[OK] compliant_hold returned final drift = {final_drift:+.3f} deg "
              f"— {how}")
    except CompliantDriftTrip as exc:
        # NOTE: the finally in compliant_hold still runs hold_on_stop's re-latch
        # here, so with --hold-on-stop a trip leaves the joint CSP-latched at the
        # drifted position rather than disabled. Observe the post mode below and
        # decide whether trip+hold_on_stop should instead disable (open question).
        print(f"\n[TRIP] {exc}")
    except RuntimeError as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)

    mode_after = read_u8_safe(motor, 0x6061)
    sw_after = read_u16_safe(motor, 0x6041)
    pos_after = motor.get_position()
    print(f"Mode after     = {_mode_name(mode_after)} (0x6061={mode_after})")
    print(f"Statusword after = {_statusword_summary(sw_after)}")
    print(f"Position after = {pos_after:.3f} deg (drift {pos_after - theta0:+.3f} deg)")
    if args.hold_on_stop:
        if mode_after == 8:
            print("[PASS] hold_on_stop re-latched to CSP (0x6061=8).")
        else:
            print("[CHECK] expected CSP (8) after hold_on_stop, got "
                  f"{_mode_name(mode_after)} — re-latch sequence needs attention.")
    else:
        # disable() was called, but its semantics are unconfirmed: report the
        # real statusword instead of asserting "limp". bit2=0 -> de-energized
        # (and "can't push" is just harmonic stiction); bit2=1 -> still energized.
        if sw_after is not None and (sw_after & 0x04):
            print("[CHECK] hold_on_stop=False but operation_enabled=1 after "
                  "disable() — disable may not have de-energized the joint.")
        else:
            print("[INFO] hold_on_stop=False: operation_enabled=0 after disable() "
                  "(de-energized; 'can't push' is likely harmonic stiction).")
    print("Done.")


if __name__ == "__main__":
    main()
