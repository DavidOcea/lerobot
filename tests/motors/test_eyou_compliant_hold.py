"""Tests for ``EyouMotorHardware.compliant_hold`` (CST soft-spring I/O loop).

``compliant_hold`` wraps the pure ``CstSoftSpringController`` in the CAN I/O
loop: read position -> compute torque -> send CST target torque, with a drift
safety trip. These tests pin the safety-critical ordering and trip behaviour
using a fake motor node (no SDK / robot required).
"""

import time

from lerobot.motors.eyou.cst_soft_spring import CompliantDriftTrip
from lerobot.motors.eyou.eyou_motor_hardware import EyouMotorHardware


class _FakeMotor:
    def __init__(self, positions, pos_fn=None):
        self._positions = list(positions)
        self._pos_fn = pos_fn
        self._i = 0
        self.calls = []
        self.sent_torques = []
        self.mode_after_switch = 8  # value read_u8(0x6061) reports
        self.statusword = 0x0237  # value read_u16(0x6041) reports; bit2=op_enabled
        self.enable_works = True  # False => raw 0x6040=0x0F won't set op_enabled
        self._pending_switch_on = False  # 0x07 takes effect on next statusword poll

    def get_position(self):
        self.calls.append("get_position")
        if self._pos_fn is not None:
            return self._pos_fn()
        p = self._positions[min(self._i, len(self._positions) - 1)]
        self._i += 1
        return p

    def clear_fault(self):
        self.calls.append("clear_fault")
        return True

    def configure_cst_mode(self, *args, **kwargs):
        self.calls.append("configure_cst_mode")
        return True

    def configure_csp_mode(self, *args, **kwargs):
        self.calls.append("configure_csp_mode")
        return True

    def send_csp_target_position(self, pos, *args, **kwargs):
        self.calls.append(("send_csp_pos", pos))
        return 0

    def send_cst_target_torque(self, tau, *args, **kwargs):
        self.calls.append(("send_torque", tau))
        self.sent_torques.append(tau)

    def disable(self):
        self.calls.append("disable")

    def read_u8(self, obj, sub):
        self.calls.append(("read_u8", obj, sub))
        return self.mode_after_switch

    def read_u16(self, obj, sub):
        self.calls.append(("read_u16", obj, sub))
        if obj == 0x6041 and self._pending_switch_on:
            # Switch On (0x07) only lands once the host has polled it, modelling the
            # real drive's need to confirm each state transition before the next.
            self.statusword |= 0x02  # Switched On (bit1)
            self._pending_switch_on = False
        return self.statusword

    def write_u16(self, obj, sub, val):
        self.calls.append(("write_u16", obj, sub, val))
        if obj == 0x6040:  # controlword: model the CiA 402 enable state machine
            if val == 0x06:  # Shutdown -> Ready to Switch On (bit0=1, bit1=bit2=0)
                self.statusword = (self.statusword & ~0x06) | 0x01
                self._pending_switch_on = False
            elif val == 0x07:  # Switch On -> lands on next statusword poll
                self._pending_switch_on = True
            elif val == 0x0F and self.enable_works:  # Enable Operation
                if self.statusword & 0x02:  # only if already Switched On
                    self.statusword |= 0x04
        return True

    def write_u8(self, obj, sub, val):
        self.calls.append(("write_u8", obj, sub, val))
        return 0


class _FakeClock:
    """可控假时钟：让 ``_wait_stationary`` 的超时/采样窗在测试里确定，且不真睡 2s。"""

    def __init__(self):
        self.t = 0.0

    def perf_counter(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _make_hw(motor):
    hw = EyouMotorHardware.__new__(EyouMotorHardware)
    hw.motor_nodes_ = [motor]
    hw.joint_names_ = ["right_arm_joint_5"]
    return hw


def test_compliant_hold_reads_reference_before_enabling_cst(monkeypatch):
    """theta0 must be captured while still position-held, BEFORE the CST switch
    (in CST a zero-torque command is limp, so a later read would be a fallen
    position, not the original)."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([5.0, 5.0])
    hw = _make_hw(motor)
    hw.compliant_hold(0, seconds=0.01, loop_ms=5)
    assert motor.calls[0] == "get_position"
    assert motor.calls[1] == "clear_fault"
    assert motor.calls[2] == "configure_cst_mode"
    assert motor.calls[-1] == "disable"


def test_compliant_hold_commands_zero_torque_at_rest(monkeypatch):
    """At the reference with no motion, the spring+damper sum to zero, so every
    torque command is 0 and is always an int."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([5.0, 5.0])
    hw = _make_hw(motor)
    hw.compliant_hold(0, seconds=0.01, loop_ms=5, kp=20.0, kd=5.0)
    assert motor.sent_torques, "expected at least one torque command"
    assert all(t == 0 for t in motor.sent_torques)
    assert all(isinstance(t, int) for t in motor.sent_torques)


def test_compliant_hold_disables_and_raises_on_drift_trip(monkeypatch):
    """A drift past max_drift_deg must raise CompliantDriftTrip and disable the
    joint (the safety net)."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # First get_position() is the reference (0.0); the first loop read returns
    # 20.0 -> drift 20 deg -> trip with max_drift_deg=5.
    motor = _FakeMotor([0.0, 20.0])
    hw = _make_hw(motor)
    try:
        hw.compliant_hold(0, seconds=0.01, loop_ms=5, max_drift_deg=5.0)
    except CompliantDriftTrip:
        pass
    else:
        raise AssertionError("expected CompliantDriftTrip")
    assert "disable" in motor.calls


def test_compliant_hold_stops_when_drift_reaches_stop_threshold(monkeypatch):
    """Reaching stop_drift_deg (the conform point) ends the hold normally,
    before the larger max_drift_deg safety trip."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # theta0 = 0.0; the first loop read is 4.0 -> drift 4.0 >= stop_drift_deg 3.0.
    motor = _FakeMotor([0.0, 4.0])
    hw = _make_hw(motor)
    drift = hw.compliant_hold(
        0, seconds=1.0, loop_ms=5, stop_drift_deg=3.0, max_drift_deg=10.0
    )
    assert drift >= 3.0
    assert "disable" in motor.calls  # hold_on_stop defaults False


def test_compliant_hold_holds_on_stop_relatches_to_csp(monkeypatch):
    """hold_on_stop=True re-latches the joint to CSP position-hold at its final
    position instead of disabling it (limp)."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([5.0, 5.0])
    hw = _make_hw(motor)
    hw.compliant_hold(0, seconds=0.01, loop_ms=5, hold_on_stop=True)
    assert "configure_csp_mode" in motor.calls
    assert any(isinstance(c, tuple) and c[0] == "send_csp_pos" for c in motor.calls)
    assert "disable" not in motor.calls


def test_compliant_hold_trip_wins_over_stop_when_drift_overshoots_max(monkeypatch):
    """When a single tick's drift overshoots max_drift_deg, the safety trip must
    fire even though stop_drift_deg is also set. If the stop check runs first,
    a runaway drift (|drift| > max) is silently returned as 'conformed' success
    and the safety net never trips."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # theta0 = 0.0; the first loop read is 20.0 -> drift 20, above BOTH
    # max_drift_deg=15.0 and stop_drift_deg=3.0. Trip must win.
    motor = _FakeMotor([0.0, 20.0])
    hw = _make_hw(motor)
    try:
        hw.compliant_hold(
            0, seconds=1.0, loop_ms=5,
            stop_drift_deg=3.0, max_drift_deg=15.0,
        )
    except CompliantDriftTrip:
        pass
    else:
        raise AssertionError(
            "expected CompliantDriftTrip to fire before the normal stop break"
        )


def test_relatch_csp_zeroes_torque_then_remaps_and_streams_position(monkeypatch):
    """A successful CST->CSP re-latch must: zero torque FIRST (so the drive stops
    driving), then WAIT FOR STATIONARY (read position until it stops moving) before
    switching, then go through ``configure_csp_mode`` (which remaps the RPDO to
    0x607A), then stream the settled position. The raw ``write_u8(0x6060, 8)`` path
    is forbidden because it leaves the RPDO on the torque channel, so the position
    stream writes garbage."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([12.5])
    hw = _make_hw(motor)
    assert hw._relatch_csp(motor) is True
    # safety ordering: zero torque is the very first command
    assert motor.calls[0] == ("send_torque", 0)
    # stationary must be checked (position read) BEFORE the CSP switch
    assert motor.calls.index("get_position") < motor.calls.index("configure_csp_mode")
    # RPDO remap via configure_csp_mode, never the raw 0x6060 write
    assert "configure_csp_mode" in motor.calls
    assert not any(isinstance(c, tuple) and c[0] == "write_u8" for c in motor.calls)
    # position streamed to the remapped 0x607A channel (the settled position)
    assert any(
        isinstance(c, tuple) and c[0] == "send_csp_pos" and c[1] == 12.5
        for c in motor.calls
    )
    assert "disable" not in motor.calls


def test_relatch_csp_disables_when_switch_fails(monkeypatch):
    """If 0x6061 never reports CSP(8), the joint must be disabled (brake) and
    must NOT receive a position stream — a position write into a drive still in
    CST goes to the torque channel and would open-loop the joint."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([12.5])
    motor.mode_after_switch = 10  # switch rejected
    hw = _make_hw(motor)
    assert hw._relatch_csp(motor) is False
    assert motor.calls[0] == ("send_torque", 0)
    assert "disable" in motor.calls
    assert not any(
        isinstance(c, tuple) and c[0] == "send_csp_pos" for c in motor.calls
    )


def test_wait_stationary_returns_settled_position_when_stable(monkeypatch):
    """Two consecutive samples within threshold_deg => stationary; return the
    settled position, which becomes the hold target for the re-latch."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    motor = _FakeMotor([10.0])
    hw = _make_hw(motor)
    assert hw._wait_stationary(motor) == 10.0


def test_wait_stationary_times_out_when_continuously_moving(monkeypatch):
    """A joint that keeps moving (|Δθ| >= threshold every window) must time out
    to None, never falsely report 'stationary'."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    state = {"pos": 10.0}

    def pos_fn():
        state["pos"] += 1.0
        return state["pos"]

    motor = _FakeMotor([], pos_fn=pos_fn)
    hw = _make_hw(motor)
    assert hw._wait_stationary(motor) is None


def test_wait_stationary_settles_after_transient(monkeypatch):
    """A brief transient (one moving sample) then stability must still return the
    settled position — the streak resets on motion, then rebuilds."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    motor = _FakeMotor([10.0, 10.6, 10.7])  # 10.0 -> 10.6 (moving) -> 10.7 stable
    hw = _make_hw(motor)
    assert hw._wait_stationary(motor) == 10.7


def test_relatch_csp_disables_when_joint_never_settles(monkeypatch):
    """If the joint never settles (still moving after zero torque), _relatch_csp
    must brake WITHOUT attempting the CSP switch or streaming a position —
    switching into a still-moving joint is the jerk/runaway hazard."""
    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    state = {"pos": 0.0}

    def pos_fn():
        state["pos"] += 2.0
        return state["pos"]

    motor = _FakeMotor([], pos_fn=pos_fn)
    hw = _make_hw(motor)
    assert hw._relatch_csp(motor) is False
    assert motor.calls[0] == ("send_torque", 0)
    assert "disable" in motor.calls
    assert "configure_csp_mode" not in motor.calls
    assert not any(
        isinstance(c, tuple) and c[0] == "send_csp_pos" for c in motor.calls
    )


def test_relatch_csp_disables_when_mode_switches_but_not_operation_enabled(monkeypatch):
    """0x6061==8 alone is a FALSE success on second-batch motors, and when the raw
    controlword fallback (0x06->0x07->0x0F) ALSO fails to reach Operation Enabled,
    _relatch_csp must disable and return False. (The hold target may still have been
    preloaded — that is harmless, since the drive is never enabled.)"""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([12.5])
    motor.mode_after_switch = 8  # mode register flips to CSP...
    motor.statusword = 0x0331  # ...but operation_enabled bit2 stays 0
    motor.enable_works = False  # raw controlword fallback also can't enable
    hw = _make_hw(motor)
    assert hw._relatch_csp(motor) is False
    assert motor.calls[0] == ("send_torque", 0)
    assert "disable" in motor.calls


def test_relatch_csp_raw_enables_and_preloads_target_before_enable(monkeypatch):
    """Second-batch motors (node 16/17): configure_csp_mode flips 0x6061 to 8 and
    remaps the RPDO but leaves statusword bit2 operation_enabled=0 (0x0331). The raw
    controlword 0x06->0x07->0x0F DOES reach Operation Enabled (0420.log). _relatch_csp
    must fall back to that raw enable, and it must PRELOAD the hold target (0x607A =
    settled) BEFORE enabling, so the drive never chases a stale target on enable."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    motor = _FakeMotor([12.5])
    motor.mode_after_switch = 8
    motor.statusword = 0x0331  # not enabled; write_u16(0x6040,0x0F) will set bit2
    hw = _make_hw(motor)
    assert hw._relatch_csp(motor) is True
    assert motor.calls[0] == ("send_torque", 0)
    # raw controlword walk issued
    assert any(
        isinstance(c, tuple) and c[0] == "write_u16" and c[1] == 0x6040 and c[3] == 0x0F
        for c in motor.calls
    )
    # hold target streamed
    assert any(
        isinstance(c, tuple) and c[0] == "send_csp_pos" and c[1] == 12.5
        for c in motor.calls
    )
    # PRELOAD-before-enable: the position write precedes the raw EnableOperation
    assert motor.calls.index(("send_csp_pos", 12.5)) < motor.calls.index(
        ("write_u16", 0x6040, 0, 0x0F)
    )
    assert "disable" not in motor.calls
