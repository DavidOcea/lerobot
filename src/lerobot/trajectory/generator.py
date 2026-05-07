"""Trajectory generator for position_sequence-based data collection.

Converts a list of PositionSequenceStep definitions into a frame-by-frame
action trajectory using ease-in-out cosine interpolation, matching the
algorithm used by _execute_position_task() in the orchestrator.

All position values are in degrees. The output action dicts use the
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

    Interpolation uses ease-in-out cosine:
        smooth_progress = 0.5 * (1 - cos(pi * progress))

    This matches the algorithm in orchestrator._execute_position_task() (line 1268).
    """

    def __init__(
        self,
        steps: list[PositionSequenceStep],
        fps: int,
        all_joint_names: list[str],
    ):
        self.steps = steps
        self.fps = fps
        self.all_joint_names = all_joint_names

    def generate(self, start_position: dict[str, float]) -> list[dict[str, float]]:
        """Generate complete trajectory as list of action dicts.

        Args:
            start_position: Current robot position {"joint_name": float} in degrees,
                            no ".pos" suffix. Typically from robot.get_current_position().

        Returns:
            List of action dicts, one per frame. Each dict has format
            {"joint_name.pos": float} in degrees, matching send_action() interface.

        Key behaviors:
        - Pre-computed: deterministic, reproducible with same inputs
        - Joints not in a step's position dict hold their value from the
          previous step's end (or start_position for the first step)
        - Steps are chained: each step starts from where the previous ended
        - Output uses ".pos" suffix for direct send_action() compatibility
        """
        trajectory = []
        current_pos = start_position.copy()

        for step in self.steps:
            target = step.position.copy()
            num_frames = max(1, int(step.max_duration * self.fps))
            step_start = current_pos.copy()

            for frame_idx in range(num_frames):
                # +1 so progress reaches 1.0 on the last frame
                progress = (frame_idx + 1) / num_frames
                smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))

                action = {}
                for joint_name in self.all_joint_names:
                    if joint_name in target:
                        value = step_start[joint_name] + (
                            target[joint_name] - step_start[joint_name]
                        ) * smooth_progress
                    else:
                        # Joint not specified in this step — hold position
                        value = step_start[joint_name]
                    action[f"{joint_name}.pos"] = value

                trajectory.append(action)

            # Update current_pos to this step's target for the next step
            for joint_name in target:
                current_pos[joint_name] = target[joint_name]

        return trajectory

    @property
    def total_frames(self) -> int:
        """Total number of frames across all steps."""
        return sum(max(1, int(step.max_duration * self.fps)) for step in self.steps)

    @property
    def total_duration(self) -> float:
        """Total duration in seconds across all steps."""
        return sum(step.max_duration for step in self.steps)