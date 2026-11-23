"""Tests for b-min: a standalone position task with a *partial* target
(only some joints listed) must hold the omitted joints at their current
position instead of sending an incomplete action dict that the robot rejects
with "Could not find required joint '...'".
"""

from types import SimpleNamespace

import pytest

from lerobot.agent.orchestrator import TaskAgentOrchestrator
from lerobot.tasks.config import CompletionCriteria, TaskConfig
from lerobot.tasks.task_scheduler import TaskStatus


class _TrackingRobot:
    """Records every action and follows the commanded positions, so the
    final `get_current_position()` reflects what was actually sent."""

    def __init__(self, joint_names, current):
        self.observation_joint_names = list(joint_names)
        self.gripper_joint_names = []
        self._current = dict(current)
        self.actions = []

    def get_current_position(self):
        return dict(self._current)

    def send_action(self, action):
        self.actions.append(dict(action))
        for key, value in action.items():
            joint = key[:-4] if key.endswith(".pos") else key
            self._current[joint] = value

    def get_observation(self):
        return {}


def _make_orch(robot, control_frequency=500):
    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.robot = robot
    orch.config = SimpleNamespace(
        robot_config=SimpleNamespace(control_frequency=control_frequency)
    )
    orch.monitor_collector = None
    orch.collision_detector = None
    orch.total_collision_count = 0
    orch.current_cycle = 0
    orch.total_cycles = 0
    orch._add_monitor_event = lambda *a, **k: None
    return orch


def _partial_task():
    return TaskConfig(
        name="腰部旋转",
        task_type="position",
        max_duration=0.1,
        completion_criteria=CompletionCriteria(
            type="position",
            target_joint_positions={"trunk_joint_1": 45.0},  # only waist moves
            position_tolerance=1.0,
        ),
    )


def test_omitted_joints_are_held_at_current_position():
    robot = _TrackingRobot(
        ["trunk_joint_1", "trunk_joint_2", "left_arm_joint_1", "right_arm_joint_1"],
        {
            "trunk_joint_1": 0.0,
            "trunk_joint_2": 5.0,
            "left_arm_joint_1": -96.0,
            "right_arm_joint_1": 95.0,
        },
    )
    orch = _make_orch(robot)

    result = orch._execute_position_task(_partial_task())

    assert result.status == TaskStatus.COMPLETED, result.error_message
    assert robot.actions, "expected send_action to be called"

    expected_keys = {
        "trunk_joint_1.pos",
        "trunk_joint_2.pos",
        "left_arm_joint_1.pos",
        "right_arm_joint_1.pos",
    }
    for action in robot.actions:
        assert set(action) == expected_keys, f"incomplete action dict: {sorted(action)}"
        # omitted joints are held exactly at their start position
        assert action["trunk_joint_2.pos"] == 5.0
        assert action["left_arm_joint_1.pos"] == -96.0
        assert action["right_arm_joint_1.pos"] == 95.0

    # the configured joint actually advanced toward its target
    assert robot.actions[-1]["trunk_joint_1.pos"] > 0.0


def test_unknown_target_joint_raises():
    """A typo'd joint name must still fail loudly, not silently hold everything."""
    robot = _TrackingRobot(["trunk_joint_1"], {"trunk_joint_1": 0.0})
    orch = _make_orch(robot)
    task = TaskConfig(
        name="typo",
        task_type="position",
        max_duration=0.1,
        completion_criteria=CompletionCriteria(
            type="position",
            target_joint_positions={"trunk_joint_10": 45.0},  # typo: not on robot
            position_tolerance=1.0,
        ),
    )
    with pytest.raises(ValueError):
        orch._execute_position_task(task)
