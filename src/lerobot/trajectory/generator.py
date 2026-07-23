"""Trajectory generator for position_sequence-based data collection.

Converts a list of PositionSequenceStep definitions into a frame-by-frame
action trajectory.  Non-overlap steps use cosine ease-in-out interpolation.
Consecutive overlap_next steps are merged into a continuous chain with
constant-speed linear interpolation + EMA corner smoothing + final-segment
ease-out, matching the orchestrator's _execute_position_sequence_task.

All position values are in degrees.  The output action dicts use the
".pos" suffix format required by robot.send_action().
"""

import math
from dataclasses import dataclass, field

from lerobot.tasks.config import PositionSequenceStep


class TrajectoryGenerator:
    """Pre-compute a deterministic frame-by-frame action trajectory from position_sequence steps.

    The trajectory is fully determined by the steps, fps, and start_position.
    The same inputs always produce the same trajectory — diversity for training
    comes from noise injection at execution time, not from trajectory generation.

    Two generation modes (matching the orchestrator):
    - Standalone step: cosine ease-in-out per max_duration
    - Overlap chain:    constant-speed linear + EMA + final ease-out

    Properties total_frames / total_duration are accurate after generate()
    has been called, regardless of mode.
    """

    # ── Configuration constants (mirrors orchestrator) ────────────────
    CHAIN_SPEED_DEG_PER_S: float = 30.0    # constant speed for chain
    EMA_ALPHA: float = 0.22                # EMA smoothing factor

    def __init__(
        self,
        steps: list[PositionSequenceStep],
        fps: int,
        all_joint_names: list[str],
    ):
        self.steps = steps
        self.fps = fps
        self.all_joint_names = all_joint_names
        # Populated by generate()
        self._actual_frames: int = 0
        self._actual_duration: float = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def generate(self, start_position: dict[str, float]) -> list[dict[str, float]]:
        """Generate complete trajectory as list of action dicts.

        Args:
            start_position: Current robot position {"joint_name": float} in degrees,
                            no ".pos" suffix.

        Returns:
            List of action dicts per frame in {"joint_name.pos": float} format.
        """
        trajectory: list[dict[str, float]] = []
        current_pos = start_position.copy()

        # Helper: build a single action dict from positions (no .pos → .pos)
        def _make_action(pos: dict[str, float]) -> dict[str, float]:
            return {f"{jn}.pos": pos.get(jn, current_pos.get(jn, 0.0))
                    for jn in self.all_joint_names}

        i = 0
        while i < len(self.steps):
            step = self.steps[i]

            # ── Overlap chain ────────────────────────────────────────
            if step.overlap_next and i + 1 < len(self.steps):
                chain_start = i
                while i < len(self.steps) and self.steps[i].overlap_next:
                    i += 1
                # Include the following non-overlap step as final waypoint
                if i < len(self.steps):
                    i += 1
                chain = self.steps[chain_start:i]
                chain_frames = self._generate_chain(chain, current_pos)
                trajectory.extend(chain_frames)
                # Update current_pos to last step's target
                last_target = chain[-1].position
                for jn in last_target:
                    current_pos[jn] = last_target[jn]
                continue

            # ── Standalone step (cosine) ──────────────────────────────
            target = step.position.copy()
            step_start = current_pos.copy()

            # 按实际运动距离计算帧数（和 agent 对齐），max_duration 仅做上限约束
            max_delta = 0.0
            for jn in self.all_joint_names:
                if jn in target:
                    max_delta = max(max_delta, abs(target[jn] - step_start[jn]))
            distance_num_frames = int(max(1, max_delta / self.CHAIN_SPEED_DEG_PER_S * self.fps))
            max_duration_frames = int(step.max_duration * self.fps)
            num_frames = max(1, min(distance_num_frames, max_duration_frames)) if max_delta > 0.01 else max_duration_frames

            for frame_idx in range(num_frames):
                progress = (frame_idx + 1) / num_frames
                smooth = 0.5 * (1 - math.cos(math.pi * progress))

                pos = {}
                for jn in self.all_joint_names:
                    if jn in target:
                        pos[jn] = step_start[jn] + (target[jn] - step_start[jn]) * smooth
                    else:
                        pos[jn] = step_start[jn]
                trajectory.append(_make_action(pos))

            for jn in target:
                current_pos[jn] = target[jn]
            i += 1

        # Cache actual totals for properties
        self._actual_frames = len(trajectory)
        self._actual_duration = self._actual_frames / self.fps

        return trajectory

    # ──────────────────────────────────────────────────────────────────
    # Continuous chain generation (mirrors orchestrator chain logic)
    # ──────────────────────────────────────────────────────────────────

    def _generate_chain(
        self,
        chain: list[PositionSequenceStep],
        start_position: dict[str, float],
    ) -> list[dict[str, float]]:
        """Generate frames for a continuous overlap chain.

        Algorithm:
        1. Build waypoints = [start_pos, step0.target, ..., stepN.target]
        2. Compute per-segment distances (max-joint deltas)
        3. Constant speed for all-but-last segment, eased stop on final
        4. EMA low-pass at every frame to round corners
        """
        jn = self.all_joint_names
        fps = self.fps
        speed = self.CHAIN_SPEED_DEG_PER_S

        # ── 1. Build waypoints ───────────────────────────────────────
        waypoints: list[dict[str, float]] = [start_position.copy()]
        for step in chain:
            wp = {k: step.position.get(k, waypoints[-1].get(k, 0.0)) for k in jn}
            waypoints.append(wp)

        # ── 2. Segment distances (max-joint delta) ───────────────────
        seg_lens: list[float] = []
        for k in range(len(waypoints) - 1):
            max_d = 0.0
            for j_name in jn:
                max_d = max(max_d, abs(waypoints[k + 1][j_name] - waypoints[k][j_name]))
            seg_lens.append(max_d)

        total_len = sum(seg_lens)
        if total_len < 0.01:
            return []

        cum = [0.0]
        for sl in seg_lens:
            cum.append(cum[-1] + sl)

        # ── 3. Timing: constant then eased final segment ─────────────
        pre_last = cum[-2] if len(cum) >= 2 else 0.0
        last_len = total_len - pre_last
        const_dur = pre_last / speed if pre_last > 0 else 0.0
        ease_dur = max(last_len / speed * 1.5, 0.3)
        total_dur = const_dur + ease_dur
        # Settling hold frames so EMA converges to the final waypoint
        # (mirrors orchestrator's hold_frames=5)
        hold_frames = 5
        total_frames = max(1, int(total_dur * fps)) + hold_frames

        # ── 4. Generate frames ────────────────────────────────────────
        frames: list[dict[str, float]] = []
        filtered: dict[str, float] = {}
        ema = self.EMA_ALPHA

        for fi in range(total_frames):
            elapsed = fi / fps  # seconds from chain start

            # Distance along path
            if elapsed < const_dur:
                distance = speed * elapsed
            elif elapsed < total_dur:
                local = (elapsed - const_dur) / ease_dur  # ≤ 1.0
                # s(t)=t+t²-t³: s'(0)=1, s'(1)=0 (C¹ smooth stop)
                eased = local + local * local - local * local * local
                distance = pre_last + last_len * eased
            else:
                # Hold at final waypoint
                distance = total_len

            # Segment lookup
            seg_idx = 0
            for s in range(len(cum) - 1):
                if cum[s] <= distance <= cum[s + 1]:
                    seg_idx = s
                    break
            slen = seg_lens[seg_idx]
            t = (distance - cum[seg_idx]) / slen if slen > 0 else 1.0

            # Raw linear position + EMA filter
            pos = {}
            for j_name in jn:
                w0 = waypoints[seg_idx][j_name]
                w1 = waypoints[seg_idx + 1][j_name]
                raw = w0 + (w1 - w0) * t

                prev = filtered.get(j_name, raw)
                smooth = prev + ema * (raw - prev)
                filtered[j_name] = smooth
                pos[j_name] = smooth

            frames.append({f"{k}.pos": v for k, v in pos.items()})

        return frames

    # ──────────────────────────────────────────────────────────────────
    # Properties — accurate after generate() has been called
    # ──────────────────────────────────────────────────────────────────

    @property
    def total_frames(self) -> int:
        """Total number of frames in the last generated trajectory.

        Accurate for both standalone and chain modes because it is
        computed from the actual trajectory length after generate().
        Returns 0 if generate() has not been called yet.
        """
        if self._actual_frames > 0:
            return self._actual_frames
        # Fallback estimate before generate() is called
        return sum(max(1, int(step.max_duration * self.fps)) for step in self.steps)

    @property
    def total_duration(self) -> float:
        """Total duration of the last generated trajectory in seconds.

        Returns total_frames / fps.  Returns 0 if generate() has not
        been called yet.
        """
        if self._actual_frames > 0:
            return self._actual_frames / self.fps
        # Fallback estimate before generate() is called
        return sum(step.max_duration for step in self.steps)
