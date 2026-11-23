#!/usr/bin/env python3
"""Diagnose the CST -> CSP re-latch on a single EYou joint.

Run ON THE ROBOT MACHINE.

Why this exists
---------------
The zero-gravity float (G command) and the compliant-stop both enter CST
(torque) mode and must come back to CSP (position) mode at the end. On this
drive CSP -> CST works, and CST -> CSP fails UNLESS the joint is stationary
(zero torque) first. A re-latch attempted while a non-zero torque command is
still applied leaves the drive in CST with that non-zero torque -> open-loop
runaway (0420.log, snapped a cable). ``EyouMotorHardware._relatch_csp`` now
zeroes torque, waits for the joint to settle, switches via ``configure_csp_mode``
and holds position, falling back to ``disable()`` on any failure.

The communication manual (``EYou-PHU&RHU...通信手册 v1.07``, section 4.3/4.6.1)
states the requirement the production path is missing:

    "关于运行模式切换，目前支持在使能静止状态下切换运行模式。从其他运行模式
     切换至位置模式（PP、CSP）之前，需将实际位置 0x6064 的值赋予目标位置
     0x607A。"

i.e. the canonical CSP sequence is: (1) 0x6060=8, (2) **read 0x6064 and write
0x607A = 0x6064**, (3) 0x6040 = 0x06->0x07->0x0F, (4) stream 0x607A. The SDK
``configureCspMode`` sets the mode and remaps the RPDO but never preloads 0x607A
from 0x6064 before switching. This probe confirms which step is missing by
running several candidate transition sequences and reporting 0x6061/0x6041 after
each.

Safety
------
- Run with the robot IDLE, arm in a safe pose, person at e-stop.
- LIGHT-LOAD joint only: node 15/25 (arm joint_5). The waist (node 1/2) carries
  the whole robot and is refused unless ``--allow-waist``.
- Before every transition the drive is put back into CST with **zero torque**
  (``send_cst_target_torque(0)``), so a failed switch leaves the joint stationary,
  never spinning.
- By default the probe does NOT stream position (no ``send_csp_target_position``),
  so it cannot trigger a garbage-torque runaway. Pass ``--hold-test`` to also
  verify position streaming after a successful mode switch (with a drift guard
  that disables immediately on movement).

Usage
-----
    # find which sequence switches 0x6061 back to 8
    python3 scripts/eyou_probe_relatch.py --node-id 15

    # single sequence only
    python3 scripts/eyou_probe_relatch.py --node-id 15 --sequence preload_switch

    # also verify position streaming (drift guard + auto-disable)
    python3 scripts/eyou_probe_relatch.py --node-id 15 --hold-test

    # verify the position stream actually MOVES the joint (step response)
    python3 scripts/eyou_probe_relatch.py --node-id 15 --move-test

    # walk raw controlword 0x06->0x07->0x0F, print statusword per step
    # (diagnose why second-batch node 16/17 never reach Operation Enabled)
    python3 scripts/eyou_probe_relatch.py --node-id 17 --probe-enable

Sequences
---------
    sdk_async             configure_csp_mode(0, False)   (old production re-latch)
    sdk_sync              configure_csp_mode(0, True)    (startup path)
    raw_mode              write_u8(0x6060,0,8)           (pre-fix raw switch, no RPDO remap)
    preload_switch        read 0x6064 -> write 0x607A, then 0x6060=8   [HYPOTHESIS]
    preload_then_sdk_sync read 0x6064 -> write 0x607A, then configure_csp_mode(0, True)
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time

try:
    import eu_motor_py  # noqa: F401  (guards the rest of the script)
except ImportError as exc:  # pragma: no cover - dev environment
    sys.exit(
        "eu_motor_py is not importable. This script must run on the robot "
        f"machine.\nImportError: {exc}"
    )

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

_MODE_NAME = {8: "CSP", 10: "CST", 11: "MIT", 13: "CSF"}


def _read_safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _mode_name(mode):
    return _MODE_NAME.get(mode, str(mode))


def _statusword_summary(sw):
    if sw is None:
        return "unreadable"
    return f"0x{sw:04X} (operation_enabled={int(bool(sw & 0x04))})"


def _read_mode(motor):
    return _read_safe(motor.read_u8, 0x6061, 0)


def _read_status(motor):
    return _read_safe(motor.read_u16, 0x6041, 0)


def _read_position_fresh(motor):
    """新鲜读位置（度）。

    第二批电机(node 16/17) TPDO 被动反馈失效，get_position() 返回冻结旧值；这里
    优先走 get_latest_feedback() 的新鲜反馈，超时则回退裸 SDO 读 0x6064 按
    52953088 脉冲/圈换算角度（与 EyouMotorHardware.read() 的回退分支一致）。
    """
    try:
        fb = motor.get_latest_feedback()
        if fb.last_update_time > datetime.timedelta(0):
            return fb.position_deg
    except Exception:
        pass
    raw = motor.read_s32(0x6064, 0)
    return raw / 52953088.0 * 360.0


def _enter_cst(motor):
    """Force the joint back into CST with zero torque (known-good direction)."""
    motor.clear_fault()
    if not motor.configure_cst_mode(20, 0, False):
        raise RuntimeError("configure_cst_mode failed")
    motor.send_cst_target_torque(0, 0, False)
    time.sleep(0.1)


def _preload_target_from_actual(motor):
    """Read actual position 0x6064 and write it to target position 0x607A.

    This is the manual-required step before switching to CSP. Returns the raw
    value written (or None on failure).
    """
    actual = _read_safe(motor.read_s32, 0x6064, 0)
    if actual is None:
        return None
    ok = motor.write_s32(0x607A, 0, actual)
    return actual if ok else None


# --------------------------------------------------------------------------- #
# Candidate transition sequences. Each returns a dict of observations.
# --------------------------------------------------------------------------- #

def _seq_sdk_async(motor):
    desc = "configure_csp_mode(0, False)  [old production re-latch]"
    ok = motor.configure_csp_mode(0, False)
    return {"desc": desc, "ok": ok, "rpdo_remapped": True, "sync": False}


def _seq_sdk_sync(motor):
    desc = "configure_csp_mode(0, True)  [startup path]"
    ok = motor.configure_csp_mode(0, True)
    return {"desc": desc, "ok": ok, "rpdo_remapped": True, "sync": True}


def _seq_raw_mode(motor):
    desc = "write_u8(0x6060,0,8)  [pre-fix raw switch, no RPDO remap]"
    ok = motor.write_u8(0x6060, 0, 8)
    return {"desc": desc, "ok": ok, "rpdo_remapped": False, "sync": False}


def _seq_preload_switch(motor):
    desc = "0x607A=0x6064 preload, then 0x6060=8  [HYPOTHESIS]"
    pre = _preload_target_from_actual(motor)
    ok = motor.write_u8(0x6060, 0, 8)
    return {"desc": desc, "ok": ok, "preloaded": pre,
            "rpdo_remapped": False, "sync": False}


def _seq_preload_then_sdk_sync(motor):
    desc = "0x607A=0x6064 preload, then configure_csp_mode(0, True)"
    pre = _preload_target_from_actual(motor)
    ok = motor.configure_csp_mode(0, True)
    return {"desc": desc, "ok": ok, "preloaded": pre,
            "rpdo_remapped": True, "sync": True}


_SEQUENCES = {
    "sdk_async": _seq_sdk_async,
    "sdk_sync": _seq_sdk_sync,
    "raw_mode": _seq_raw_mode,
    "preload_switch": _seq_preload_switch,
    "preload_then_sdk_sync": _seq_preload_then_sdk_sync,
}


def _hold_test(motor, is_sync, max_drift_deg=2.0, seconds=1.5):
    """After a claimed CSP switch, verify position streaming holds still.

    Only safe when the RPDO is already remapped to 0x607A (the sequence reported
    ``rpdo_remapped``). Sends the *current* position as a hold target and watches
    for drift; any movement beyond ``max_drift_deg`` aborts and disables.
    """
    theta0 = _read_position_fresh(motor)
    try:
        motor.send_csp_target_position(theta0, 0, is_sync)
    except Exception:
        return "send-failed"
    t_end = time.perf_counter() + seconds
    max_drift = 0.0
    while time.perf_counter() < t_end:
        time.sleep(0.05)
        theta = _read_position_fresh(motor)
        drift = abs(theta - theta0)
        max_drift = max(max_drift, drift)
        if drift > max_drift_deg:
            try:
                motor.send_cst_target_torque(0, 0, False)
            except Exception:
                pass
            motor.disable()
            return f"DRIFT {drift:+.2f} deg -> aborted+disabled"
    return f"hold-ok (max drift {max_drift:+.2f} deg)"


def _read_rpdo_mapping(motor):
    """Read RPDO1 mapping (0x1600) to see which object the position stream lands
    in. Returns ``(n_mapped, first_mapped_raw)``. The first mapped object's high
    16 bits are the object index: 0x607A=position (safe), 0x6071=torque (the
    exact "32-bit position -> 16-bit torque" garbage that caused the 0420
    runaway). 0x1600 is the CiA 402 standard address — confirm the printed raw
    value on the robot the first time; adjust if this drive maps elsewhere."""
    n = _read_safe(motor.read_u8, 0x1600, 0)
    val = _read_safe(motor.read_u32, 0x1600, 1)
    return n, val


def _move_test(motor, is_sync, step_deg=3.0, guard_margin_deg=3.0,
               poll_s=0.02, settle_s=1.5):
    """After a claimed CSP switch, verify the position stream actually MOVES the
    joint: command theta0 + step_deg and confirm it travels ~step_deg and holds.

    This closes the hold-test blind spot: on a gravity-balanced wrist, zero torque
    cannot move the joint, so a "hold at current position" test passes trivially
    (max drift 0.00 deg) even if the position write lands on the torque channel as
    garbage. A commanded step that the joint actually follows proves the position
    stream really reaches 0x607A.

    Safety is layered, not a single guard:
    1. PREVENTIVE — read RPDO1 mapping and REFUSE to step if it maps 0x6071
       (torque). Unreadable/unknown mapping warns and falls through to (2).
    2. REACTIVE — a tight runaway guard (|theta-theta0| > step+margin, polled
       every poll_s) zeroes torque and disables. This is a *catch*, not a
       guarantee: a very fast runaway could outrun one poll interval. The tiny
       step + light joint + person-at-e-stop are what bound that residual risk.
    """
    n, mapped = _read_rpdo_mapping(motor)
    if mapped is not None:
        idx = (mapped >> 16) & 0xFFFF
        if idx == 0x6071:
            return (f"REFUSED (RPDO maps 0x{idx:04X}=torque; a position write "
                    f"would be garbage torque — NOT stepping)")
        if idx != 0x607A:
            print(f"  [warn] RPDO1 maps 0x{idx:04X} (not 0x607A); "
                  f"proceeding under guard only")
    else:
        print("  [warn] RPDO1 mapping unreadable; proceeding under guard only")

    theta0 = _read_position_fresh(motor)
    target = theta0 + step_deg
    guard_deg = step_deg + guard_margin_deg
    try:
        motor.send_csp_target_position(target, 0, is_sync)
    except Exception:
        return "send-failed"
    t_end = time.perf_counter() + settle_s
    theta = theta0
    max_abs = 0.0
    while time.perf_counter() < t_end:
        time.sleep(poll_s)
        theta = _read_position_fresh(motor)
        max_abs = max(max_abs, abs(theta - theta0))
        if abs(theta - theta0) > guard_deg:
            try:
                motor.send_cst_target_torque(0, 0, False)
            except Exception:
                pass
            motor.disable()
            return (f"RUNAWAY {theta - theta0:+.2f} deg (guard {guard_deg:.1f}) "
                    f"-> aborted+disabled")
    actual = theta - theta0
    if abs(actual - step_deg) < 0.5:
        return (f"move-ok (moved {actual:+.2f} deg, target {step_deg:.2f} deg, "
                f"max |drift| {max_abs:.2f} deg)")
    return (f"move-FAIL (moved {actual:+.2f} deg, expected {step_deg:.2f} deg, "
            f"max |drift| {max_abs:.2f} deg)")


def _run_relatch(hw, motor, args):
    """End-to-end production path: enter CST, then ``hw._relatch_csp`` (zero
    torque -> wait stationary -> configure_csp_mode -> verify 0x6061 -> hold),
    then optionally step/verify the position stream. This exercises the real
    wait-stationary logic on the robot, not just the raw switch sequence."""
    _enter_cst(motor)
    mode_before = _read_mode(motor)
    print(f"\n[relatch] production _relatch_csp "
          f"(zero torque -> wait stationary -> configure -> hold)")
    print(f"  mode before   = {_mode_name(mode_before)} (0x6061={mode_before})")

    ok = hw._relatch_csp(motor)
    mode_after = _read_mode(motor)
    status_after = _read_status(motor)
    print(f"  _relatch_csp  = {ok} (True=CSP-held, False=disabled)")
    print(f"  mode after    = {_mode_name(mode_after)} (0x6061={mode_after})")
    print(f"  status after  = {_statusword_summary(status_after)}")

    if ok and args.move_test:
        move = _move_test(motor, False, step_deg=args.move_step,
                          guard_margin_deg=args.move_guard)
        print(f"  move test     = {move}")
    if ok and args.hold_test:
        hold = _hold_test(motor, False)
        print(f"  hold test     = {hold}")


def _probe_enable(motor, poll_s=0.3):
    """Walk the raw CiA 402 controlword 0x6040 = 0x06 -> 0x07 -> 0x0F, printing the
    statusword (0x6041) at each step.

    This isolates WHERE a second-batch motor stops reaching Operation Enabled
    (statusword bit2). On node 16/17 (0420_3.log) the SDK ``configure_csp_mode``
    flips 0x6061 to 8 but the state machine sticks at Ready-to-Switch-On
    (bit2=0), so position commands do nothing. Walking the raw controlword with
    per-step statusword shows whether it is the 0x07 (Switch On) or 0x0F (Enable
    Operation) transition that fails, or whether a fault (bit3) appears.

    Safe: the sequence only re-arms the drive in its current mode. The caller's
    finally block zeroes torque and disables afterwards.
    """
    motor.clear_fault()
    print(f"\n[probe-enable] raw controlword walk (0x6040): 0x06 -> 0x07 -> 0x0F")
    sw = _read_status(motor)
    print(f"  initial        statusword={_statusword_summary(sw)}")

    for cmd, name in [(0x06, "Shutdown"), (0x07, "SwitchOn"), (0x0F, "EnableOperation")]:
        ok = _read_safe(motor.write_u16, 0x6040, 0, cmd)
        time.sleep(poll_s)
        sw = _read_status(motor)
        if sw is None:
            print(f"  write 0x6040=0x{cmd:04X} ({name:16s}) ok={ok} "
                  f"-> statusword UNREADABLE")
            continue
        switched_on = bool(sw & 0x02)
        op_enabled = bool(sw & 0x04)
        fault = bool(sw & 0x08)
        print(f"  write 0x6040=0x{cmd:04X} ({name:16s}) ok={ok} -> "
              f"0x{sw:04X} switched_on={switched_on} operation_enabled={op_enabled} "
              f"fault={fault}")

    sw = _read_status(motor)
    op_enabled = bool(sw & 0x04) if sw is not None else False
    print(f"  verdict: operation_enabled={op_enabled} "
          f"({'ENABLED' if op_enabled else 'NOT enabled'})")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--node-id", type=int, default=15,
                   help="CANopen node ID (default 15 = right_arm_joint_5, light load).")
    p.add_argument("--device", type=int, default=1, help="CAN device index (default 1).")
    p.add_argument("--can-baud-rate", default="1M",
                   choices=["1M", "500K", "250K"], help="CAN baud rate (default 1M).")
    p.add_argument("--sequence", default="all",
                   help="Which sequence to run: %s (default 'all')."
                        % ", ".join(sorted(_SEQUENCES)))
    p.add_argument("--hold-test", action="store_true",
                   help="After a successful mode switch, also verify position "
                        "streaming (only for sequences that remapped the RPDO).")
    p.add_argument("--move-test", action="store_true",
                   help="After a successful mode switch, command a +step step and "
                        "verify the joint actually moves (closes the hold-test "
                        "blind spot on a gravity-balanced wrist).")
    p.add_argument("--move-step", type=float, default=3.0,
                   help="Step size in degrees for --move-test (default 3.0, "
                        "small and safe for a light-load wrist).")
    p.add_argument("--move-guard", type=float, default=3.0,
                   help="Runaway guard margin in degrees beyond --move-step "
                        "(guard = step + margin; default 3.0). Smaller = safer "
                        "but more false trips on servo overshoot.")
    p.add_argument("--relatch", action="store_true",
                   help="Run the production EyouMotorHardware._relatch_csp "
                        "end-to-end (enter CST, then zero torque -> wait "
                        "stationary -> switch -> hold) instead of the raw switch "
                        "sequences. Combine with --move-test/--hold-test.")
    p.add_argument("--probe-enable", action="store_true",
                   help="Walk the raw CiA 402 controlword 0x6040 "
                        "(0x06->0x07->0x0F) and print statusword at each step, "
                        "to find where a second-batch motor stops reaching "
                        "Operation Enabled (bit2).")
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
            "the whole robot. Use a light-load arm joint (15/25 = joint_5) "
            "instead, or pass --allow-waist to override at your own risk."
        )

    if args.sequence != "all" and args.sequence not in _SEQUENCES:
        sys.exit(f"[error] unknown --sequence '{args.sequence}'. Choose from: "
                 + ", ".join(sorted(_SEQUENCES)) + ", all.")

    print("=" * 72)
    print("  WARNING: this re-enters CST (torque) with ZERO torque, then")
    print("  attempts a CST->CSP re-latch. Arm in a safe pose, person at")
    print("  e-stop, light-load joint only.")
    print("=" * 72)
    time.sleep(1.0)

    config = {
        "can_device_index": args.device,
        "can_baud_rate": args.can_baud_rate,
        "joints": [{"name": joint_name, "parameters": {"node_id": node_id}}],
    }
    hw = EyouMotorHardware()
    if not hw.init(config):
        sys.exit("[error] EyouMotorHardware.init failed")
    motor = hw.motor_nodes_[0]

    # Identity (vendor/product/revision) — correlates first-batch (vendor 1640)
    # vs second-batch (vendor 4247, placeholder revision) motors.
    vendor = _read_safe(motor.read_u32, 0x1018, 1)
    product = _read_safe(motor.read_u32, 0x1018, 2)
    revision = _read_safe(motor.read_u32, 0x1018, 3)
    print(f"\nJoint: {joint_name} (node {node_id})")
    print(f"Identity: vendor=0x{vendor:08X} product=0x{product:08X} "
          f"revision=0x{revision:08X}" if vendor is not None else
          f"Identity: unreadable")
    print(f"Mode before = {_mode_name(_read_mode(motor))} "
          f"(0x6061={_read_mode(motor)})")

    names = sorted(_SEQUENCES) if args.sequence == "all" else [args.sequence]

    try:
        if args.probe_enable:
            _probe_enable(motor)
            return
        if args.relatch:
            _run_relatch(hw, motor, args)
            return
        for name in names:
            seq = _SEQUENCES[name]
            _enter_cst(motor)
            mode_before = _read_mode(motor)
            if mode_before != 10:
                print(f"\n[{name}] WARN: expected CST(10) before transition, "
                      f"got {_mode_name(mode_before)} — skipping")
                continue

            result = seq(motor)
            time.sleep(0.1)
            mode_after = _read_mode(motor)
            status_after = _read_status(motor)

            switched = (mode_after == 8)
            verdict = "PASS" if switched else "FAIL"
            print(f"\n[{name}] {result['desc']}")
            print(f"  call ok        = {result.get('ok')}")
            if "preloaded" in result:
                print(f"  preload 0x607A = {result.get('preloaded')} "
                      "(raw, None=read/write failed)")
            print(f"  mode after     = {_mode_name(mode_after)} (0x6061={mode_after})  -> {verdict}")
            print(f"  status after   = {_statusword_summary(status_after)}")

            if switched and args.hold_test and result.get("rpdo_remapped"):
                hold = _hold_test(motor, result.get("sync", False))
                print(f"  hold test      = {hold}")
            elif switched and args.hold_test and not result.get("rpdo_remapped"):
                print("  hold test      = skipped (RPDO not remapped; do NOT stream position)")

            if switched and args.move_test and result.get("rpdo_remapped"):
                move = _move_test(motor, result.get("sync", False),
                                  step_deg=args.move_step,
                                  guard_margin_deg=args.move_guard)
                print(f"  move test      = {move}")
            elif switched and args.move_test and not result.get("rpdo_remapped"):
                print("  move test      = skipped (RPDO not remapped; do NOT stream position)")
    finally:
        # Never leave the joint in CST with a non-zero torque command.
        try:
            motor.send_cst_target_torque(0, 0, False)
        except Exception:
            pass
        try:
            motor.disable()
        except Exception:
            pass
        print("\nDone. Joint zeroed and disabled (safe).")


if __name__ == "__main__":
    main()
