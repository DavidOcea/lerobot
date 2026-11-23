#!/usr/bin/env python3
"""CST (cyclic-synchronous torque) software-spring test for EYou joints.

Run ON THE ROBOT MACHINE.

Why this exists
---------------
The MIT impedance mode (0x6060=11) turned out to be unimplemented on the
robot's firmware (revision 16 accepts the mode *number* in 0x6061 but has no
0x2150-0x2169 protocol objects -- see ``eyou_probe_mit.py --probe-mit-objects``).
CST torque mode (0x6060=10) *is* fully implemented, so we reproduce MIT's
control law on the host and push the result through CST::

    tau = Kp * (theta0 - theta) - Kd * omega

The Kp*(theta0 - theta) term doubles as gravity compensation: when the joint
sags under its own weight, the position error grows and produces a restoring
torque. No explicit gravity model is needed (a static deflection error is the
price of low Kp), which is exactly why we wanted MIT in the first place.

Units (comm manual V1.07, table 4-11):
    0x6071  Target torque   INT    per-thousand of rated   (RxPDO)
    0x6072  Max torque      UINT   per-thousand of rated   (limits 0x6071)
    0x6076  Rated torque    UINT   mNm                    (for display)
    0x6077  Actual torque   INT    per-thousand of rated   (TxPDO)

So ``tau`` is in per-thousand (1000 == rated). ``theta`` is in degrees.

Safety
------
- Run with the robot IDLE, arm in a safe pose, person at e-stop.
- The default (no flags) mode is READ-ONLY (identity + CST objects).
- ``--soft-hold`` enables the joint in CST and applies a software spring. In
  CST a zero torque command = limp, so a LIGHT-LOAD joint is mandatory: use
  arm joint 5 (node 15 right / 25 left) or 6 (16 / 26). Do NOT use the waist
  (node 1/2) -- it carries the whole robot and a limp fall would topple it.
- The torque command is clamped to ``--max-torque`` (per-thousand) and the
  script trips back to disable if position drifts past ``--max-drift-deg`` or
  on Ctrl-C.

Usage
-----
    python3 scripts/eyou_probe_cst.py                         # read-only CST report
    python3 scripts/eyou_probe_cst.py --node-id 15 --soft-hold --kp 30 --kd 40 --seconds 30
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import eu_motor_py
except ImportError as exc:  # pragma: no cover - dev environment
    sys.exit(
        "eu_motor_py is not importable. This script must run on the robot "
        f"machine.\nImportError: {exc}"
    )


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


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


def read_s16_safe(motor, index, sub=0):
    try:
        return motor.read_s16(index, sub)
    except Exception:
        return None


def do_report(args, dev_index, baud):
    can = eu_motor_py.CanNetworkManager()
    can.init_device(eu_motor_py.DeviceType.Canable, dev_index, baud)
    motor = eu_motor_py.EuMotorNode(dev_index, args.node_id)

    print(f"=== Node {args.node_id} — CST object report (read-only) ===\n")
    for name, index, sub, meth in [
        ("vendor_id   0x1018:01", 0x1018, 1, read_u32_safe),
        ("product_code 0x1018:02", 0x1018, 2, read_u32_safe),
        ("revision    0x1018:03", 0x1018, 3, read_u32_safe),
    ]:
        try:
            print(f"  {name} = {meth(motor, index, sub)}")
        except Exception as exc:
            print(f"  {name} = [error] {exc}")

    mode = read_u8_safe(motor, 0x6061)
    target = read_s16_safe(motor, 0x6071)
    maxt = read_u16_safe(motor, 0x6072)
    rated_mnm = read_u16_safe(motor, 0x6076)
    actual = read_s16_safe(motor, 0x6077)
    print(f"  0x6061 mode display        = {mode}")
    print(f"  0x6071 target torque       = {target}  (per-thousand)")
    print(f"  0x6072 max torque          = {maxt}  (per-thousand)")
    print(f"  0x6076 rated torque        = {rated_mnm}  (mNm)")
    print(f"  0x6077 actual torque       = {actual}  (per-thousand)")
    if rated_mnm is not None:
        rated_nm = rated_mnm / 1000.0
        print(f"  rated torque               = {rated_nm:.3f} Nm")
        if actual is not None:
            print(f"  actual torque              = {actual / 1000.0 * rated_nm:.3f} Nm")
    print()


def do_soft_hold(args, dev_index, baud):
    can = eu_motor_py.CanNetworkManager()
    can.init_device(eu_motor_py.DeviceType.Canable, dev_index, baud)
    motor = eu_motor_py.EuMotorNode(dev_index, args.node_id)

    # Sanity-check the total torque cap against the motor's own 0x6072 limit.
    maxt = read_u16_safe(motor, 0x6072)
    if maxt is not None and args.total_limit > maxt:
        print(f"[warn] --total-limit {args.total_limit} exceeds 0x6072 max "
              f"torque {maxt}; clamping to {maxt}.")
        args.total_limit = maxt

    # Capture the hold reference BEFORE entering CST: configure_cst_mode()
    # enables the joint with no torque command yet (limp), so any sleep or
    # get_position() after it reads a fallen position, not the original.
    theta0 = motor.get_position()  # degrees
    print(f"Hold reference theta0 = {theta0:.3f} deg (read while still in CSP)")

    print("Enabling joint in CST torque mode (async RPDO)...")
    if not motor.clear_fault():
        sys.exit("[error] clear_fault failed")
    if not motor.configure_cst_mode(20, 0, False):
        sys.exit("[error] configure_cst_mode failed")
    # No sleep here: the loop's first iteration sends the spring command
    # immediately, minimizing the limp window.
    kp = args.kp
    kd = args.kd
    dt = args.loop_ms / 1000.0
    spring_limit = args.max_torque
    damp_limit = args.damp_limit
    total_limit = args.total_limit
    vel_alpha = args.vel_filter
    print(f"Spring: tau = clamp({kp}*(theta0-theta), +/-{spring_limit})"
          f" + clamp(-{kd}*omega_f, +/-{damp_limit}) [vel LPF a={vel_alpha}], "
          f"total clamp +/-{total_limit} (per-thousand), max drift "
          f"{args.max_drift_deg} deg. Ctrl-C to abort.\n")

    theta_prev = theta0
    omega_f = 0.0
    try:
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            theta = motor.get_position()
            omega_raw = (theta - theta_prev) / dt
            # Low-pass filter the velocity. The raw one-sample difference is
            # dominated by encoder/quantization noise, which a high Kd amplifies
            # into a ~12-25 Hz limit cycle (seen on the robot). Filter it before
            # feeding the damper, then clamp spring/damping separately so a
            # saturated spring never cuts the damping.
            omega_f = vel_alpha * omega_raw + (1.0 - vel_alpha) * omega_f
            tau_spring = _clamp(kp * (theta0 - theta), -spring_limit, spring_limit)
            tau_damp = _clamp(-kd * omega_f, -damp_limit, damp_limit)
            tau = _clamp(tau_spring + tau_damp, -total_limit, total_limit)
            motor.send_cst_target_torque(int(tau), 0, False)

            actual = motor.get_torque()  # per-thousand
            drift = theta - theta0
            print(f"  theta={theta:+8.3f} (drift {drift:+6.3f})  "
                  f"spr={tau_spring:+7.0f}  dmp={tau_damp:+7.0f}  "
                  f"cmd={tau:+7.0f}  act={actual:+6.0f}")

            if abs(drift) > args.max_drift_deg:
                print(f"\n[TRIP] drift {drift:+.3f} deg exceeds "
                      f"{args.max_drift_deg} deg; disabling.")
                sys.exit(1)

            theta_prev = theta
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nCtrl-C: disabling.")
    finally:
        try:
            motor.disable()
            print("Disabled.")
        except Exception as exc:
            print(f"[warn] disable failed: {exc}")
    print("Done.")


def read_u32_safe(motor, index, sub=0):
    try:
        return motor.read_u32(index, sub)
    except Exception:
        return None


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node-id", type=int, default=15,
                   help="CANopen node ID (default 15 = right_arm_joint_5, "
                        "light load; safer than waist).")
    p.add_argument("--device", type=int, default=1, help="CAN device index (default 1).")
    p.add_argument("--can-baud-rate", default="1M",
                   choices=["1M", "500K", "250K"], help="CAN baud rate (default 1M).")

    p.add_argument("--soft-hold", action="store_true",
                   help="Enable CST + software spring hold (manual push).")

    p.add_argument("--kp", type=float, default=15.0,
                   help="Spring gain, per-thousand torque per degree (default 15). "
                        "Higher = stiffer return; static sag ≈ gravity/kp.")
    p.add_argument("--kd", type=float, default=15.0,
                   help="Damping, per-thousand torque per deg/s (default 15). "
                        "Higher = less bounce, but too high self-oscillates at 20ms loop.")
    p.add_argument("--vel-filter", type=float, default=0.3,
                   help="Velocity low-pass alpha, 0-1 (default 0.3). Lower = smoother "
                        "damping; essential to stop velocity noise from self-oscillating.")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="Duration of the hold loop (default 30).")
    p.add_argument("--loop-ms", type=int, default=20,
                   help="Loop period in ms (default 20).")
    p.add_argument("--max-torque", type=float, default=300.0,
                   help="Spring torque clamp, per-thousand (default 300 = 30%% rated).")
    p.add_argument("--damp-limit", type=float, default=300.0,
                   help="Damping torque clamp, per-thousand (default 300). "
                        "Kept separate so damping is never cut when the spring saturates.")
    p.add_argument("--total-limit", type=float, default=500.0,
                   help="Final total torque cap, per-thousand (default 500 = 50%% rated).")
    p.add_argument("--max-drift-deg", type=float, default=15.0,
                   help="Trip + disable if position drifts past this (deg).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    baud_map = {
        "1M": eu_motor_py.Baudrate.BPS_1M,
        "500K": eu_motor_py.Baudrate.BPS_500K,
        "250K": eu_motor_py.Baudrate.BPS_250K,
    }
    baud = baud_map[args.can_baud_rate]

    if args.soft_hold:
        print("=" * 72)
        print("  WARNING: this ENABLES the joint in CST torque mode.")
        print("  Arm in a safe pose, person at e-stop, light-load joint only.")
        print("=" * 72)
        time.sleep(1.0)
        do_soft_hold(args, args.device, baud)
    else:
        do_report(args, args.device, baud)


if __name__ == "__main__":
    main()
