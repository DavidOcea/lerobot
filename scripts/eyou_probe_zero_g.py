#!/usr/bin/env python3
"""Run the *production* ``EyouMotorHardware.zero_g_float`` on the robot.

Run ON THE ROBOT MACHINE.

Why this exists
---------------
``zero_g_float`` is the multi-joint zero-gravity float primitive used by
``record_joint_positions.py``'s ``G`` command for one-person hand-guiding.
It depends on a trick the other probes cannot exercise: the joint must be in
CSP position-hold when ``get_torque()`` is read, so the read returns the real
gravity+friction holding torque (``tau_hold``), which is then fed forward as
``tau_ff`` in CST. This script:

    1. enables the joints via the production ``activate()`` path, sends a hold
       at the current position, then prints each joint's holding torque from
       BOTH the TPDO feedback (torque_milli) and the raw SDO get_torque() --
       if both are ~0 on a gravity-loaded joint, the torque-memory premise is
       broken and there is no point floating yet.
    2. calls ``hw.zero_g_float(indices, ...)`` and reports per-joint mode /
       statusword / position / drift afterwards, so the CST->CSP re-latch can
       be confirmed (0x6061 back to 8 = CSP).

Safety
------
- Run with the robot IDLE, arm in a safe pose, person at e-stop.
- LIGHT-LOAD joint only to start: arm joint 5 (node 15 right / 25 left).
  joint 6/7 (node 16/17/26/27) are the SECOND batch of motors; CST is NOT
  verified on them yet (see eyou_probe_compliant_hold.py) -- verify CST there
  first.  The waist (node 1/2) is refused unless ``--allow-waist``.
- ``tau_ff`` = the measured holding torque, so a wrong ``get_torque()`` read
  (wrong sign / scale) can drive the joint INTO the floor or ceiling. Start
  with ``--max-torque`` low (e.g. 200) and one hand on the joint.

Usage
-----
    # Step 0 + Step 1 combined: enable joint 5, print tau_hold, float 10 s.
    python3 scripts/eyou_probe_zero_g.py --node-ids 15 --seconds 10 --max-torque 200

    # multi-joint arm segment (joint 5 + 6): indices passed to zero_g_float are
    # the list positions 0..N-1 in the order of --node-ids.
    python3 scripts/eyou_probe_zero_g.py --node-ids 15,16 --seconds 15 --max-torque 200
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


def _mode_name(mode):
    return {8: "CSP", 10: "CST", 11: "MIT", 13: "CSF"}.get(mode, str(mode))


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--node-ids", type=str, default="15",
                   help="Comma-separated CANopen node IDs (default '15' = "
                        "right_arm_joint_5, light load). Multi-joint e.g. '15,16'.")
    p.add_argument("--device", type=int, default=1, help="CAN device index (default 1).")
    p.add_argument("--can-baud-rate", default="1M",
                   choices=["1M", "500K", "250K"], help="CAN baud rate (default 1M).")

    p.add_argument("--kd", type=float, default=5.0,
                   help="Damping, per-thousand torque per deg/s (default 5).")
    p.add_argument("--vel-filter", type=float, default=0.3,
                   help="Velocity low-pass alpha, 0-1 (default 0.3).")
    p.add_argument("--backup-kp", type=float, default=5.0,
                   help="Safety backup spring, per-thousand per deg (default 5).")
    p.add_argument("--max-torque", type=float, default=200.0,
                   help="Total torque clamp, per-thousand (default 200; "
                        "start LOW, wrong tau_hold sign can drive the joint).")
    p.add_argument("--max-drift-deg", type=float, default=15.0,
                   help="Safety trip if drift exceeds this (deg).")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="Max float duration (default 10).")
    p.add_argument("--loop-ms", type=int, default=20,
                   help="Loop period in ms (default 20).")
    p.add_argument("--allow-waist", action="store_true",
                   help="Override the waist (node 1/2) refusal. NOT recommended.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    node_ids = [int(x) for x in args.node_ids.split(",") if x.strip()]

    for nid in node_ids:
        if nid in (1, 2) and not args.allow_waist:
            sys.exit(
                f"[refusing] node {nid} is the waist; it carries the whole robot "
                "and a limp fall would topple it. Use a light-load arm joint "
                "(15/25 = joint_5) instead, or pass --allow-waist at your own risk."
            )

    joint_names = [_NODE_NAMES.get(nid, f"node_{nid}") for nid in node_ids]

    print("=" * 72)
    print("  WARNING: enables joints in CSP, reads their holding torque,")
    print("  then applies the production zero_g_float primitive in CST.")
    print("  Arm in a safe pose, person at e-stop, one hand on the joint.")
    print("=" * 72)
    time.sleep(1.0)

    config = {
        "can_device_index": args.device,
        "can_baud_rate": args.can_baud_rate,
        "joints": [
            {"name": name, "parameters": {"node_id": nid}}
            for name, nid in zip(joint_names, node_ids)
        ],
    }
    hw = EyouMotorHardware()
    if not hw.init(config):
        sys.exit("[error] EyouMotorHardware.init failed")
    if not hw.activate():
        sys.exit("[error] EyouMotorHardware.activate failed (joints not CSP-held)")

    motors = hw.motor_nodes_
    names = hw.joint_names_

    # activate() leaves the joint in CSP mode but sends no target position, so it
    # may not be actively holding. Command a hold at the current position first,
    # then give the drive a beat to build up holding torque before reading it.
    theta0s = []
    for m in motors:
        theta0 = m.get_position()
        theta0s.append(theta0)
        m.send_csp_target_position(theta0, 0, False)
    time.sleep(0.5)

    # Step 0: read the holding torque while STILL in CSP. This is the whole
    # premise of the torque-memory approach -- it must be non-zero on a joint
    # that is fighting gravity. Report BOTH sources: the TPDO feedback
    # (torque_milli, what zero_g_float now uses) and the raw SDO get_torque()
    # (0x6077, which measured 0 on this drive).
    print("\n=== Step 0: CSP holding torque (per-thousand) ===")
    tau_holds = []
    for m, name, theta0 in zip(motors, names, theta0s):
        fb = m.get_latest_feedback()
        tpd = float(fb.torque_milli) if fb is not None else float("nan")
        sdo = float(m.get_torque())
        tau = tpd if fb is not None else sdo
        tau_holds.append(tau)
        verdict = "OK (non-zero)" if abs(tau) > 1.0 else "SUSPECT (~0, torque-memory broken)"
        print(f"  {name}: theta0={theta0:+.3f} deg  "
              f"TPDO torque_milli={tpd:+.1f}  SDO get_torque()={sdo:+.1f}  "
              f"-> use {tau:+.1f} {verdict}")
    if any(abs(t) <= 1.0 for t in tau_holds):
        print("\n  [STOP] at least one joint reads ~0 holding torque. The joint is")
        print("         not actually CSP-holding (or torque feedback is broken).")
        print("         Floating now would NOT compensate gravity. Investigate first.")

    mode_before = [read_u8_safe(m, 0x6061) for m in motors]
    print("\nMode before = " + ", ".join(_mode_name(m) for m in mode_before))

    print(f"\n>>> Now floating for {args.seconds} s. One hand on the joint;")
    print(">>> expect '松手停在原地 / 轻推随动'. Ctrl-C ends and re-latches.\n")

    indices = list(range(len(motors)))  # zero_g_float takes motor_nodes_ indices
    try:
        drifts = hw.zero_g_float(
            indices,
            kd=args.kd,
            vel_alpha=args.vel_filter,
            backup_kp=args.backup_kp,
            max_torque=args.max_torque,
            max_drift_deg=args.max_drift_deg,
            seconds=args.seconds,
            loop_ms=args.loop_ms,
        )
        print(f"\n[OK] zero_g_float returned drifts = "
              f"{['%+.2f' % d for d in drifts]}")
    except CompliantDriftTrip as exc:
        print(f"\n[TRIP] {exc}  (finally already re-latched to CSP)")
    except RuntimeError as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Ctrl-C] ending early (finally re-latches to CSP)")

    print("\n=== Post-float state ===")
    for m, name, theta0 in zip(motors, names, theta0s):
        mode_after = read_u8_safe(m, 0x6061)
        sw_after = read_u16_safe(m, 0x6041)
        pos_after = m.get_position()
        op_en = bool(sw_after & 0x04) if sw_after is not None else None
        print(f"  {name}: mode={_mode_name(mode_after)} (0x6061={mode_after}), "
              f"op_enabled={op_en}, pos={pos_after:+.3f} deg "
              f"(drift {pos_after - theta0:+.3f} deg)")
        if mode_after == 8:
            print("    [OK] 已回 CSP 锁定")
        elif op_en is False:
            # CST 但已 disable 掉电抱闸 = 安全（刹车锁住，不跑飞）
            print("    [SAFE] 未回 CSP，但已掉电抱闸(op_enabled=False)，关节被刹车锁住")
        else:
            print(f"    [DANGER] CST 且仍在使能(op_enabled=True)——残留扭矩可能空转，"
                  f"立即人工扶住并检查")
    print("Done.")


if __name__ == "__main__":
    main()
