#!/usr/bin/env python3
"""Verify EYou PHU/RHU joint firmware version and MIT-mode support.

Run ON THE ROBOT MACHINE. ``eu_motor_py`` and the CAN hardware are not
available in the dev environment.

Background
----------
MIT (Mixed Integrated Torque) mode is the EYou actuator's impedance mode
(0x6060 = 11). Its control law is::

    tau = Kp * (Pos_cmd - Pos) + Kd * (Vel_cmd - Vel) + T_ff

Kp / Kd / T_ff are re-parameterised every control cycle, so a *low-Kp soft
spring* gives force-following compliance ("柔性停止") without a full gravity
model and without switching operating mode mid-motion. That is the primitive
we want for the arm-closing contact / compliant-stop feature.

SDK mapping: the pybind11 ``OperateMode`` enum names mode 11 as
``TorquePositionFixed``. NOTE: ``EuMotorNode.switch_mode()`` is a *no-op* in
this SDK build (the real ``harmonic_setOperateMode`` call is commented out in
``eu_motor.cpp``), so the reliable mode switch is a raw ``write_u8(0x6060, …)``.

MIT was added to the EYou CANopen comm manual in **V1.07 (2026-07-17)**. The
firmware version object 0x100A (software version) is a STRING and therefore
not readable through the binding's numeric-only generic read, so we read the
numeric identity instead::

    0x1018:02  Product code   (UDINT)   -> read_u32(0x1018, 2)
    0x1018:03  Revision        (UDINT)   -> read_u32(0x1018, 3)   <-- firmware rev
    0x1018:04  Serial          (UDINT)   -> read_u32(0x1018, 4)

The *authoritative* "does my firmware have MIT" test is not the version
number, but writing 0x6060 = 11 and reading 0x6061 (modes of operation
display) back. An old firmware rejects the write or leaves 0x6061 unchanged.

Modes (table 4-1): PP=1, PV=3, PT=4, HM=6, CSP=8, CSV=9, CST=10, MIT=11, CSF=13.

Safety
------
- Run with the robot IDLE (not in a task), arm in a safe, supported pose.
- The default mode (no flags) is READ-ONLY: identity + current mode + status.
- ``--probe-modes`` writes 0x6060 and reads 0x6061 WITHOUT enabling the drive;
  it never applies torque and never activates MIT, so it is safe.
- ``--probe-mit-objects`` is READ-ONLY and tells you whether the firmware
  actually implements the MIT protocol objects (0x2150-0x2169), as opposed to
  merely accepting mode number 11 in 0x6061.
- ``--mit-hold`` and ``--kp-scan`` DO enable the joint and enter MIT mode. They
  pre-load a hold command (current position + Kp) BEFORE switching to MIT so
  there is no limp/droop window, then trip back to CSP + disable if position
  drifts more than ``--max-drift-deg`` or on Ctrl-C. Use with a person at the
  e-stop and the arm over a soft surface.
- For the torque-applying tests, use a LIGHT-LOAD joint: arm joint 5 (node 15
  right / 25 left) or joint 6 (node 16 / 26, but note joint 6 is a second-batch
  motor with placeholder identity). Do NOT use the waist (node 1/2) — it holds
  the whole robot's weight and a limp fall would topple the robot.

Usage
-----
    python3 scripts/eyou_probe_mit.py                       # read-only identity
    python3 scripts/eyou_probe_mit.py --node-id 2          # another joint
    python3 scripts/eyou_probe_mit.py --scan-range 1 30    # firmware of many nodes
    python3 scripts/eyou_probe_mit.py --probe-modes        # MIT/CST/CSF support test
    python3 scripts/eyou_probe_mit.py --probe-mit-objects  # does firmware have MIT objects?
    python3 scripts/eyou_probe_mit.py --node-id 15 --mit-hold --kp 30 --kd 2 --seconds 10
    python3 scripts/eyou_probe_mit.py --node-id 15 --kp-scan --kp 0 10 30 60 --kd 2
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
        "machine (CAN hardware + the pybind11 binding live there).\n"
        f"ImportError: {exc}"
    )

DEG2RAD = 3.141592653589793 / 180.0

# SDO data types to try when probing object presence, in (label, method) pairs.
_SDO_TYPES = [("u8", "read_u8"), ("s8", "read_s8"), ("u16", "read_u16"),
              ("s16", "read_s16"), ("u32", "read_u32"), ("s32", "read_s32")]

# MIT protocol objects (comm manual V1.07, section 4.8).
_MIT_OBJECTS = [
    (0x2150, "mit_ctrl_cobcmd1", "UDINT"),
    (0x2151, "mit_ctrl_cobcmd2", "UDINT"),
    (0x2152, "mit_reply_cobdat1", "UDINT"),
    (0x2153, "mit_reply_cobdat2", "UDINT"),
    (0x2160, "MIT_P_MIN", "DINT"), (0x2161, "MIT_P_MAX", "DINT"),
    (0x2162, "MIT_V_MIN", "DINT"), (0x2163, "MIT_V_MAX", "DINT"),
    (0x2164, "MIT_KP_MIN", "DINT"), (0x2165, "MIT_KP_MAX", "DINT"),
    (0x2166, "MIT_KD_MIN", "DINT"), (0x2167, "MIT_KD_MAX", "DINT"),
    (0x2168, "MIT_T_MIN", "DINT"), (0x2169, "MIT_T_MAX", "DINT"),
]


# --------------------------------------------------------------------------- #
# MIT protocol packing (identical to the MIT Cheetah actuator wire format).
# --------------------------------------------------------------------------- #

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Map x in [x_min, x_max] onto an unsigned `bits`-bit integer."""
    span = (x_max - x_min) or 1.0
    n = (1 << bits) - 1
    return _clamp(int(round((x - x_min) / span * n)), 0, n)


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    span = (x_max - x_min)
    n = (1 << bits) - 1
    return x_min + x_int * span / n


def pack_mit_cmd(pos, vel, kp, kd, t_ff, scales):
    """Pack a MIT command into the two uint32 objects (0x2150, 0x2151).

    scales = dict(p_min, p_max, v_min, v_max, kp_min, kp_max,
                  kd_min, kd_max, t_min, t_max)  (all in SI units)
    """
    pos_i = float_to_uint(pos, scales["p_min"], scales["p_max"], 16)
    vel_i = float_to_uint(vel, scales["v_min"], scales["v_max"], 12)
    kp_i = float_to_uint(kp, scales["kp_min"], scales["kp_max"], 12)
    kd_i = float_to_uint(kd, scales["kd_min"], scales["kd_max"], 12)
    t_i = float_to_uint(t_ff, scales["t_min"], scales["t_max"], 12)

    # 0x2150 bytes: Pos[15:8] | Pos[7:0] | Vel[11:4] | Vel[3:0]|Kp[11:8]
    cmd1 = ((pos_i >> 8) & 0xFF) \
        | (((pos_i & 0xFF)) << 8) \
        | (((vel_i >> 4) & 0xFF) << 16) \
        | ((((vel_i & 0xF) << 4) | ((kp_i >> 8) & 0xF)) << 24)

    # 0x2151 bytes: Kp[7:0] | Kd[11:4] | Kd[3:0]|T_ff[11:8] | T_ff[7:0]
    cmd2 = (kp_i & 0xFF) \
        | (((kd_i >> 4) & 0xFF) << 8) \
        | ((((kd_i & 0xF) << 4) | ((t_i >> 8) & 0xF)) << 16) \
        | ((t_i & 0xFF) << 24)

    return cmd1, cmd2


def unpack_mit_fb(fb1, fb2, scales):
    """Decode MIT feedback (0x2152, 0x2153) -> (err, pos, vel, torque, temp)."""
    err = fb1 & 0xFF
    pos_hi = (fb1 >> 8) & 0xFF
    pos_lo = (fb1 >> 16) & 0xFF
    vel_hi = (fb1 >> 24) & 0xFF

    b0 = fb2 & 0xFF
    t_lo = (fb2 >> 8) & 0xFF
    temp = (fb2 >> 16) & 0xFF
    if temp & 0x80:  # int8 signed
        temp -= 0x100

    vel_lo = (b0 >> 4) & 0xF
    t_hi = b0 & 0xF

    pos = uint_to_float((pos_hi << 8) | pos_lo,
                        scales["p_min"], scales["p_max"], 16)
    vel = uint_to_float((vel_hi << 4) | vel_lo,
                        scales["v_min"], scales["v_max"], 12)
    torque = uint_to_float((t_hi << 8) | t_lo,
                           scales["t_min"], scales["t_max"], 12)
    return err, pos, vel, torque, temp


# --------------------------------------------------------------------------- #
# Object-dictionary helpers
# --------------------------------------------------------------------------- #

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


def read_u32_safe(motor, index, sub=0):
    try:
        return motor.read_u32(index, sub)
    except Exception:
        return None


def read_identity(motor):
    """Read numeric identity objects. Returns a dict (None on SDO failure)."""
    def _r(name, index, sub):
        v = read_u32_safe(motor, index, sub)
        if v is None:
            print(f"  [warn] read {name} (0x{index:04X}:{sub}) failed")
        return v

    return {
        "vendor_id": _r("vendor_id", 0x1018, 1),
        "product_code": _r("product_code", 0x1018, 2),
        "revision": _r("revision", 0x1018, 3),   # firmware revision (numeric)
        "serial": _r("serial", 0x1018, 4),
    }


def read_mit_scales(motor):
    """Read MIT scaling objects 0x2160..0x2169 (DINT, unit 0.001)."""
    keys = ["p_min", "p_max", "v_min", "v_max",
            "kp_min", "kp_max", "kd_min", "kd_max", "t_min", "t_max"]
    scales = {}
    for i, key in enumerate(keys):
        try:
            raw = motor.read_s32(0x2160 + i, 0)
            scales[key] = raw / 1000.0  # 0.001 rad / rad/s / Nm / coeff -> SI
        except Exception as exc:
            print(f"  [warn] read MIT scale 0x{0x2160 + i:04X} ({key}) failed: {exc}")
            scales[key] = None
    return scales


def _resolve_nodes(args):
    if args.scan_range:
        start, end = args.scan_range
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    return [args.node_id]


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

def do_identity(args, dev_index, baud):
    node_ids = _resolve_nodes(args)
    can = eu_motor_py.CanNetworkManager()
    can.init_device(eu_motor_py.DeviceType.Canable, dev_index, baud)
    print(f"CAN device {dev_index} ({args.can_baud_rate}) initialised.\n")

    for node_id in node_ids:
        print(f"=== Node {node_id} ===")
        motor = eu_motor_py.EuMotorNode(dev_index, node_id)
        ident = read_identity(motor)

        if ident["revision"] is None and ident["product_code"] is None:
            print("  [error] identity objects unreadable; node may be offline or\n"
                  "          not in a reachable NMT state. Bring it to\n"
                  "          Pre-Operational/Operational and retry.")
            continue

        print(f"  vendor_id    = {ident['vendor_id']}")
        print(f"  product_code = {ident['product_code']}")
        print(f"  revision     = {ident['revision']}   <-- firmware revision (numeric)")
        print(f"  serial       = {ident['serial']}")

        mode_disp = read_u8_safe(motor, 0x6061)
        mode_req = read_u8_safe(motor, 0x6060)
        status = read_u16_safe(motor, 0x6041)
        print(f"  0x6060 modes-of-operation (req)     = {mode_req}")
        print(f"  0x6061 modes-of-operation (display) = {mode_disp}")
        if status is not None:
            print(f"  0x6041 status word                  = {status} (0x{status:04X})")

        scales = read_mit_scales(motor)
        if any(v is not None for v in scales.values()):
            print("  MIT scales (0x2160..0x2169, SI):")
            for k, v in scales.items():
                print(f"    {k:7s} = {v}")
        print()

    print("Interpretation: MIT mode (0x6060=11) ships with the EYou CANopen\n"
          "comm manual V1.07 (2026-07-17). The authoritative check is\n"
          "`--probe-modes`, which writes 0x6060=11 and reads 0x6061 back.\n")


def do_probe_modes(args, dev_index, baud):
    node_id = args.node_id
    can = eu_motor_py.CanNetworkManager()
    can.init_device(eu_motor_py.DeviceType.Canable, dev_index, baud)
    motor = eu_motor_py.EuMotorNode(dev_index, node_id)

    original = read_u8_safe(motor, 0x6060)
    print(f"Node {node_id}: 0x6060 (requested mode) before probe = {original}\n")

    results = {}
    for mode in (11, 10, 13):
        label = {11: "MIT", 10: "CST", 13: "CSF"}[mode]
        try:
            ok = motor.write_u8(0x6060, 0, mode)
            time.sleep(0.05)
            disp = read_u8_safe(motor, 0x6061)
            accepted = (disp == mode)
            results[label] = accepted
            print(f"  {label:4s} (0x6060={mode:2d}): write={ok}, 0x6061={disp} "
                  f"-> {'ACCEPTED' if accepted else 'NOT ACCEPTED'}")
        except Exception as exc:
            results[label] = False
            print(f"  {label:4s} (0x6060={mode:2d}): write raised -> {exc}")

    # cross-check with the SDK's display-mode read (0x6061)
    try:
        print(f"\n  SDK get_operation_mode() reads back: {motor.get_operation_mode()}")
    except Exception as exc:
        print(f"\n  [warn] get_operation_mode() failed: {exc}")

    # restore
    if original is not None:
        try:
            motor.write_u8(0x6060, 0, original)
            print(f"Restored 0x6060 = {original}")
        except Exception as exc:
            print(f"[error] failed to restore 0x6060={original}: {exc}")

    print("\nSummary:")
    for label, accepted in results.items():
        verdict = "SUPPORTED" if accepted else "not supported / not writable in this state"
        print(f"  {label:4s}: {verdict}")
    print("\nNote: some firmware only accepts a 0x6060 write in the\n"
          "'enabled & stationary' state. If a mode reads back as NOT ACCEPTED\n"
          "here, re-test with the joint enabled in CSP via --mit-hold. A\n"
          "firmware *without* MIT will reject 0x6060=11 outright.\n")


def do_probe_mit_objects(args, dev_index, baud):
    """Read-only: check whether the MIT protocol objects actually exist.

    The 0x6060=11 mode number being accepted by 0x6061 only proves the *mode
    table* knows about 11. The MIT feature itself lives in 0x2150-0x2169; if
    those are unreadable, MIT is not implemented by the firmware regardless of
    what 0x6061 echoes.
    """
    node_id = args.node_id
    can = eu_motor_py.CanNetworkManager()
    can.init_device(eu_motor_py.DeviceType.Canable, dev_index, baud)
    motor = eu_motor_py.EuMotorNode(dev_index, node_id)
    print(f"Node {node_id}: probing MIT object presence (read with each data type)\n")

    any_present = False
    for index, name, expected in _MIT_OBJECTS:
        hits = {}
        for tname, meth in _SDO_TYPES:
            try:
                hits[tname] = getattr(motor, meth)(index, 0)
            except Exception:
                hits[tname] = None
        present = [t for t, v in hits.items() if v is not None]
        if present:
            any_present = True
            vals = {t: hits[t] for t in present}
            print(f"  0x{index:04X} {name:20s} expected={expected:6s} -> "
                  f"READABLE as {present}: {vals}")
        else:
            print(f"  0x{index:04X} {name:20s} expected={expected:6s} -> ABSENT")

    print()
    if any_present:
        print("Verdict: some MIT objects are present -> MIT is likely usable;\n"
              "proceed to --mit-hold on a light-load joint.")
    else:
        print("Verdict: ALL MIT objects ABSENT -> this firmware recognizes mode\n"
              "number 11 in 0x6061 but does NOT implement the MIT protocol\n"
              "objects, so MIT is not usable. Either update the firmware to the\n"
              "V1.07-comm-manual firmware (ask EYou), or fall back to CST torque\n"
              "mode (0x6060=10) + a software spring on the host.")


def do_mit(args, dev_index, baud, kp_values, cycles_per_kp):
    """Enable one joint, enter MIT, hold position with a soft spring, log.

    kp_values is the list of Kp to try in sequence; cycles_per_kp bounds how
    many control cycles each Kp is held for.
    """
    node_id = args.node_id
    can = eu_motor_py.CanNetworkManager()
    can.init_device(eu_motor_py.DeviceType.Canable, dev_index, baud)
    motor = eu_motor_py.EuMotorNode(dev_index, node_id)

    scales = read_mit_scales(motor)
    if any(v is None for v in scales.values()):
        sys.exit("[error] could not read MIT scales (0x2160..0x2169); "
                 "aborting before enabling.")

    # 1. enable stationary in CSP (production path; brings node to Op + enables
    #    the CiA 402 state machine and holds position).
    print("Enabling joint in CSP mode (stationary)...")
    if not motor.clear_fault():
        sys.exit("[error] clear_fault failed")
    if not motor.configure_csp_mode(0, True):
        sys.exit("[error] configure_csp_mode failed")
    time.sleep(0.3)

    start_deg = motor.get_position()
    start_rad = start_deg * DEG2RAD
    print(f"Start position = {start_deg:.3f} deg ({start_rad:.4f} rad)")

    kd = args.kd
    kp_values = list(kp_values)

    # 2. pre-load the first MIT hold command BEFORE switching mode, so the drive
    #    never sees a zero command (which would mean Kp=0 -> limp -> droop).
    cmd1, cmd2 = pack_mit_cmd(start_rad, 0.0, kp_values[0], kd, 0.0, scales)
    motor.write_u32(0x2150, 0, cmd1)
    motor.write_u32(0x2151, 0, cmd2)

    # 3. switch to MIT (raw 0x6060 write; switch_mode() is a no-op in this SDK).
    try:
        motor.write_u8(0x6060, 0, 11)
        time.sleep(0.05)
        disp = read_u8_safe(motor, 0x6061)
        if disp != 11:
            motor.write_u8(0x6060, 0, 8)  # back to CSP
            sys.exit(f"[error] 0x6061={disp} after requesting MIT; firmware may "
                     "not support mode 11. Restored CSP.")
    except Exception as exc:
        sys.exit(f"[error] failed to enter MIT mode: {exc}")

    print(f"Entering MIT hold loop: kp={kp_values}, kd={kd}, "
          f"max drift {args.max_drift_deg} deg. Ctrl-C to abort.\n")

    try:
        for kp in kp_values:
            for _cycle in range(cycles_per_kp):
                cmd1, cmd2 = pack_mit_cmd(start_rad, 0.0, kp, kd, 0.0, scales)
                motor.write_u32(0x2150, 0, cmd1)
                motor.write_u32(0x2151, 0, cmd2)

                fb1 = read_u32_safe(motor, 0x2152)
                fb2 = read_u32_safe(motor, 0x2153)
                if fb1 is not None and fb2 is not None:
                    err, pos, vel, torque, temp = unpack_mit_fb(fb1, fb2, scales)
                else:
                    pos = motor.get_position() * DEG2RAD
                    torque = motor.get_torque() / 1000.0
                    vel = temp = float("nan")
                    err = -1

                drift_deg = (pos - start_rad) / DEG2RAD
                print(f"  kp={kp:6.1f}  pos={pos:+.4f} rad ({drift_deg:+.3f} deg)  "
                      f"torque={torque:+.3f} Nm  temp={temp}  err={err}")

                if abs(drift_deg) > args.max_drift_deg:
                    print(f"\n[TRIP] drift {drift_deg:+.3f} deg exceeds "
                          f"{args.max_drift_deg} deg; restoring CSP + disable.")
                    sys.exit(1)

                time.sleep(args.loop_ms / 1000.0)
    except KeyboardInterrupt:
        print("\nCtrl-C: restoring.")
    finally:
        _restore_csp_and_disable(motor, start_deg)
    print("Done.")


def _restore_csp_and_disable(motor, start_deg):
    print("Restoring CSP mode and re-asserting position hold...")
    try:
        motor.write_u8(0x6060, 0, 8)  # CSP
    except Exception as exc:
        print(f"  [warn] 0x6060=8 write failed: {exc}")
    try:
        motor.send_csp_target_position(start_deg, 0, False)
    except Exception as exc:
        print(f"  [warn] csp target write failed: {exc}")
    try:
        motor.disable()
        print("Disabled.")
    except Exception as exc:
        print(f"  [warn] disable failed: {exc}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node-id", type=int, default=15,
                   help="CANopen node ID of the joint to probe "
                        "(default 15 = right_arm_joint_5, a light-load "
                        "first-batch motor; safer for --mit-hold than waist).")
    p.add_argument("--scan-range", nargs=2, type=int, metavar=("START", "END"),
                   help="Scan a range of node IDs for firmware versions.")
    p.add_argument("--device", type=int, default=1, help="CAN device index (default 1).")
    p.add_argument("--can-baud-rate", default="1M",
                   choices=["1M", "500K", "250K"], help="CAN baud rate (default 1M).")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--probe-modes", action="store_true",
                   help="Write 0x6060={11,10,13} and read 0x6061 back (no enable).")
    g.add_argument("--probe-mit-objects", action="store_true",
                   help="Read-only: probe 0x2150-0x2169 presence (each SDO data type).")
    g.add_argument("--mit-hold", action="store_true",
                   help="Enable joint, enter MIT, hold position with soft spring (manual push).")
    g.add_argument("--kp-scan", action="store_true",
                   help="Enable joint, enter MIT, ramp through --kp values automatically.")

    p.add_argument("--kp", type=float, nargs="+", default=[30.0],
                   help="Kp value(s) for --mit-hold (single) or --kp-scan (list).")
    p.add_argument("--kd", type=float, default=2.0,
                   help="Kd (damping) for MIT hold. Higher = more damping.")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="Seconds budget for --mit-hold (single Kp).")
    p.add_argument("--per-kp-cycles", type=int, default=50,
                   help="Cycles per Kp value in --kp-scan.")
    p.add_argument("--loop-ms", type=int, default=20,
                   help="Loop period in ms for the MIT hold loop (default 20).")
    p.add_argument("--max-drift-deg", type=float, default=5.0,
                   help="Trip + restore if position drifts more than this (deg).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    baud_map = {
        "1M": eu_motor_py.Baudrate.BPS_1M,
        "500K": eu_motor_py.Baudrate.BPS_500K,
        "250K": eu_motor_py.Baudrate.BPS_250K,
    }
    baud = baud_map[args.can_baud_rate]

    if args.mit_hold or args.kp_scan:
        print("=" * 72)
        print("  WARNING: this ENABLES the joint and enters MIT mode.")
        print("  Arm must be in a safe pose, person at e-stop.")
        print("=" * 72)
        time.sleep(1.0)

    if args.mit_hold:
        cycles = int(args.seconds / (args.loop_ms / 1000.0)) + 1
        do_mit(args, args.device, baud, [args.kp[0]], cycles)
    elif args.kp_scan:
        do_mit(args, args.device, baud, args.kp, args.per_kp_cycles)
    elif args.probe_modes:
        do_probe_modes(args, args.device, baud)
    elif args.probe_mit_objects:
        do_probe_mit_objects(args, args.device, baud)
    else:
        do_identity(args, args.device, baud)


if __name__ == "__main__":
    main()
