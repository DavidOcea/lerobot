"""Unit tests for the CST software-spring controller (pure control law).

This is the host-side reproduction of the MIT impedance law
``tau = Kp*(theta0 - theta) - Kd*omega`` that ``eyou_probe_cst.py`` verified on
the robot. It is pure math (no CAN/SDK I/O) so it can be tested on a dev
machine. The I/O loop that calls it is ``EyouMotorHardware.compliant_hold``
(covered separately with a fake motor node).
"""

from lerobot.motors.eyou.cst_soft_spring import CstSoftSpringController


def _ctl(**kw):
    params = dict(kp=20.0, kd=5.0, theta0=0.0, dt=0.02)
    params.update(kw)
    return CstSoftSpringController(**params)


def test_zero_drift_at_rest_returns_zero_torque():
    out = _ctl().step(0.0)
    assert out.tau == 0
    assert out.spring == 0.0
    assert out.damp == 0.0
    assert out.drift == 0.0


def test_spring_restores_when_pushed_below_reference():
    # theta < theta0 -> theta0 - theta > 0 -> restoring torque positive.
    out = _ctl(kp=10.0, kd=0.0).step(-1.0)
    assert out.spring == 10.0
    assert out.tau == 10


def test_spring_pushes_back_when_above_reference():
    out = _ctl(kp=10.0, kd=0.0).step(2.0)
    assert out.spring == -20.0


def test_spring_clamps_at_spring_limit():
    out = _ctl(kp=100.0, kd=0.0, spring_limit=50.0).step(-10.0)
    assert out.spring == 50.0


def test_damping_opposes_motion():
    # alpha=1 disables the LPF so omega_f == omega_raw == 1.0 deg/s for a single
    # +0.02 deg step over dt=0.02 s. The damper must oppose that velocity.
    out = _ctl(kp=0.0, kd=10.0, vel_alpha=1.0).step(0.02)
    assert out.omega_f == 1.0
    assert out.damp == -10.0
    assert out.tau == -10


def test_velocity_lpf_smooths_raw_velocity():
    out = _ctl(kp=0.0, kd=0.0, vel_alpha=0.3).step(0.02)
    # omega_raw = (0.02 - 0.0) / 0.02 = 1.0; omega_f = 0.3*1.0 + 0.7*0.0 = 0.3
    assert abs(out.omega_f - 0.3) < 1e-9


def test_total_torque_clamped():
    out = _ctl(
        kp=100.0, kd=100.0, vel_alpha=1.0,
        spring_limit=1000.0, damp_limit=1000.0, total_limit=80.0,
    ).step(-10.0)
    assert out.tau == 80


def test_returns_integer_torque():
    out = _ctl(kp=1.0, kd=0.0).step(-1.0)
    assert out.tau == 1
    assert isinstance(out.tau, int)
