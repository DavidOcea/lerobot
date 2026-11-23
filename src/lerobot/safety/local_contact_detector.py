"""Local-differential contact detection.

Detects the onset of physical contact as a torque rise relative to a short,
pose-independent baseline captured *just before* contact. Unlike an absolute
base-torque calibration, the baseline here is re-anchored continuously while
torque stays within a normal band, so it tracks gravity and slow load drift
without a dynamics model. The baseline is frozen the moment a rise is
suspected, so the rise itself is never absorbed into the baseline and remains
measurable across the confirmation window.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LocalContactConfig:
    """Configuration for a local-differential contact detector.

    Args:
        joints: Bare joint names to monitor (no ``.force`` suffix).
        threshold: Torque rise (in the observation's torque unit, typically
            Nm) above the local baseline that marks a suspected contact.
        baseline_window: Number of trailing "normal" samples used to compute
            the baseline mean. Sized to capture the just-before-contact torque
            while filtering sensor noise.
        confirm_window: Number of consecutive rising samples required to
            confirm contact. Rejects single-tick transients (e.g. backlash).
    """

    joints: list[str]
    threshold: float
    baseline_window: int = 5
    confirm_window: int = 3
    # Detection mode:
    #   "abs"        — existing |current − baseline| rise/drop detection (default).
    #   "drop_rate"  — sustained fast torque fall: windowed slope ≤ −rate_threshold
    #                  Nm/s for confirm_window ticks. Used for unload detection
    #                  (box weight released onto a surface) where the fall *rate*
    #                  separates the event from slow pose-change drift, so it is
    #                  independent of both box weight and surface height.
    mode: str = "abs"
    rate_threshold: float = 0.3  # Nm/s, drop_rate mode only
    rate_window: int = 5  # samples for the slope window, drop_rate mode only


class LocalContactDetector:
    """Detect contact as a sustained torque rise over a local baseline.

    Feed one torque sample per control tick via :meth:`update`. The detector
    returns ``True`` exactly once, when any monitored joint's torque has
    stayed at least ``threshold`` above its local baseline for
    ``confirm_window`` consecutive ticks.
    """

    def __init__(self, config: LocalContactConfig) -> None:
        self.config = config
        # drop_rate mode uses the buffer as a slope history; abs mode uses it as
        # the baseline. Only one mode is active per detector, so one buffer works.
        window = config.rate_window if config.mode == "drop_rate" else config.baseline_window
        self._buffers: dict[str, deque[float]] = {
            joint: deque(maxlen=window) for joint in config.joints
        }
        self._consecutive: dict[str, int] = {joint: 0 for joint in config.joints}
        # joint -> delta at the most recent trigger (empty when the last update
        # did not trigger). Exposed so callers can log *which* joint rose and by
        # how much, rather than just "contact detected".
        self.last_trigger: dict[str, float] = {}

    def update(self, torques: dict[str, float], dt: float = 1.0) -> bool:
        """Feed one sample and report whether the configured condition fired.

        Args:
            torques: Current torque per bare joint name (missing joints skipped).
            dt: Control-tick interval in seconds. Used only by ``drop_rate`` mode
                to convert a windowed slope to Nm/s; ignored by ``abs`` mode.

        Returns:
            ``True`` the first time the condition is confirmed; ``False``
            otherwise. After a ``True`` the counters reset.
        """
        if self.config.mode == "drop_rate":
            return self._update_drop_rate(torques, dt)
        return self._update_abs(torques)

    def _update_abs(self, torques: dict[str, float]) -> bool:
        """Existing |current − baseline| rise/drop detection (unchanged)."""
        triggered = False
        triggered_joints: dict[str, float] = {}

        for joint in self.config.joints:
            current = torques.get(joint)
            if current is None:
                continue

            buffer = self._buffers[joint]

            # Not enough history yet: just grow the baseline.
            if len(buffer) < self.config.baseline_window:
                buffer.append(current)
                continue

            baseline = sum(buffer) / len(buffer)
            delta = abs(current - baseline)

            if delta >= self.config.threshold:
                self._consecutive[joint] += 1
                # Freeze the baseline during the rise so it is not absorbed.
            else:
                self._consecutive[joint] = 0
                buffer.append(current)  # normal motion re-anchors the baseline

            if self._consecutive[joint] >= self.config.confirm_window:
                triggered = True
                triggered_joints[joint] = delta

        self._finish(triggered, triggered_joints)
        return triggered

    def _update_drop_rate(self, torques: dict[str, float], dt: float) -> bool:
        """Sustained fast torque fall: windowed slope ≤ −rate_threshold Nm/s."""
        triggered = False
        triggered_joints: dict[str, float] = {}

        for joint in self.config.joints:
            current = torques.get(joint)
            if current is None:
                continue

            buffer = self._buffers[joint]
            buffer.append(current)
            if len(buffer) < self.config.rate_window:
                continue

            window_time = (len(buffer) - 1) * dt
            if window_time <= 0:
                continue
            # Magnitude change: |torque| shrinking = unload (box weight released).
            # Using abs() makes it sign-agnostic so the left/right mirror arms —
            # whose torques have opposite signs — behave identically. Positive =
            # |torque| rising (loading), negative = |torque| falling (unload).
            slope = (abs(buffer[-1]) - abs(buffer[0])) / window_time

            if slope <= -self.config.rate_threshold:
                self._consecutive[joint] += 1
            else:
                self._consecutive[joint] = 0

            if self._consecutive[joint] >= self.config.confirm_window:
                triggered = True
                triggered_joints[joint] = -slope  # report fall rate as positive

        self._finish(triggered, triggered_joints)
        return triggered

    def _finish(self, triggered: bool, triggered_joints: dict[str, float]) -> None:
        if triggered:
            for joint in self.config.joints:
                self._consecutive[joint] = 0
            self.last_trigger = triggered_joints
        else:
            self.last_trigger = {}

    def reset(self) -> None:
        """Clear baseline buffers and counters, returning to a fresh state."""
        for buffer in self._buffers.values():
            buffer.clear()
        for joint in self._consecutive:
            self._consecutive[joint] = 0
