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


class LocalContactDetector:
    """Detect contact as a sustained torque rise over a local baseline.

    Feed one torque sample per control tick via :meth:`update`. The detector
    returns ``True`` exactly once, when any monitored joint's torque has
    stayed at least ``threshold`` above its local baseline for
    ``confirm_window`` consecutive ticks.
    """

    def __init__(self, config: LocalContactConfig) -> None:
        self.config = config
        self._buffers: dict[str, deque[float]] = {
            joint: deque(maxlen=config.baseline_window) for joint in config.joints
        }
        self._consecutive: dict[str, int] = {joint: 0 for joint in config.joints}

    def update(self, torques: dict[str, float]) -> bool:
        """Feed one sample and report whether contact is now confirmed.

        Args:
            torques: Current torque per bare joint name (missing joints skipped).

        Returns:
            ``True`` the first time a joint's rise is confirmed; ``False``
            otherwise. After a ``True`` the counters reset, so a subsequent
            update does not re-fire until another ``confirm_window`` rises.
        """
        triggered = False

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

        if triggered:
            for joint in self.config.joints:
                self._consecutive[joint] = 0

        return triggered

    def reset(self) -> None:
        """Clear baseline buffers and counters, returning to a fresh state."""
        for buffer in self._buffers.values():
            buffer.clear()
        for joint in self._consecutive:
            self._consecutive[joint] = 0
