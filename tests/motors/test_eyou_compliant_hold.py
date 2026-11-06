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
    def __init__(self, positions):
        self._positions = list(positions)
        self._i = 0
        self.calls = []
        self.sent_torques = []

    def get_position(self):
        self.calls.append("get_position")
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
