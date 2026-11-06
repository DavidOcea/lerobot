"""CST software-spring controller — pure control law (no CAN/SDK I/O).

The robot's firmware (revision 16) implements CST torque mode (0x6060=10) but
not MIT impedance mode (0x6060=11, no 0x2150-0x2169 objects). To get compliant
(soft-spring) behaviour we reproduce MIT's control law on the host and push the
result through CST at the CAN loop rate::

    tau = Kp * (theta0 - theta) - Kd * omega

- ``Kp*(theta0 - theta)`` is the virtual spring: it doubles as gravity
  compensation (a static sag becomes a steady position error, which produces a
  restoring torque), so no explicit gravity model is needed.
- ``-Kd*omega`` is the virtual damper. ``omega`` is a single-step difference,
  so it is dominated by encoder/quantization noise; it MUST be low-pass filtered
  before feeding the damper, or a high Kd self-oscillates (limit cycle) at the
  host loop rate — this was observed and fixed on the real robot.

Everything here is pure and deterministic so it unit-tests on a dev machine;
the I/O that calls it is ``EyouMotorHardware.compliant_hold``.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class CompliantDriftTrip(RuntimeError):
    """Raised when a compliant-hold joint drifts past ``max_drift_deg``.

    This is a safety net: the joint should float around ``theta0``, so a large
    drift means the spring torque could not hold it (too heavy a load, or a
    wrong joint chosen — e.g. the waist). The caller must stop all joints.
    """


@dataclass
class SpringStep:
    """One controller output tick (all torque values in per-thousand)."""

    tau: int          # final clamped torque command sent to 0x6071
    spring: float     # Kp*(theta0 - theta), clamped to +/-spring_limit
    damp: float       # -Kd*omega_f, clamped to +/-damp_limit
    omega_f: float    # low-pass-filtered velocity (deg/s)
    drift: float      # theta - theta0 (deg)


@dataclass
class CstSoftSpringController:
    """Stateful host-side CST spring-damper.

    Args:
        kp: Spring gain, per-thousand torque per degree. Higher = stiffer return;
            static sag ≈ gravity/kp.
        kd: Damping gain, per-thousand torque per deg/s. Higher = less bounce,
            but too high self-oscillates at the host loop rate.
        theta0: Hold reference (degrees). MUST be captured while the joint is
            still position-held (CSP), *before* entering CST — in CST a zero
            torque command is limp, so a reference read after the switch would
            be a fallen position, not the original.
        dt: Loop period in seconds (e.g. 0.02).
        vel_alpha: Velocity low-pass alpha in (0, 1]. Lower = smoother damping;
            essential to stop velocity noise from self-oscillating.
        spring_limit / damp_limit / total_limit: Per-thousand clamps. Spring and
            damping are clamped separately so a saturated spring never cuts the
            damping; the sum is then clamped to total_limit.
    """

    kp: float
    kd: float
    theta0: float
    dt: float
    vel_alpha: float = 0.3
    spring_limit: float = 300.0
    damp_limit: float = 300.0
    total_limit: float = 500.0

    def __post_init__(self) -> None:
        self._theta_prev: float = self.theta0
        self._omega_f: float = 0.0

    def step(self, theta: float) -> SpringStep:
        """Advance one tick and return the torque command + diagnostics."""
        omega_raw = (theta - self._theta_prev) / self.dt
        self._omega_f = (
            self.vel_alpha * omega_raw + (1.0 - self.vel_alpha) * self._omega_f
        )
        spring = _clamp(
            self.kp * (self.theta0 - theta), -self.spring_limit, self.spring_limit
        )
        damp = _clamp(-self.kd * self._omega_f, -self.damp_limit, self.damp_limit)
        tau = _clamp(spring + damp, -self.total_limit, self.total_limit)
        self._theta_prev = theta
        return SpringStep(
            tau=int(tau),
            spring=spring,
            damp=damp,
            omega_f=self._omega_f,
            drift=theta - self.theta0,
        )
