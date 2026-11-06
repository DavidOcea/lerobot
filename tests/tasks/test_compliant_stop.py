"""Tests for two-stage compliant-stop config parsing and orchestrator wiring.

Compliant-stop extends contact-stop: instead of a rigid ``break`` at the touch
threshold, the contact joints switch to a CST soft-spring float (compliant_hold)
and stop all joints once the float conforms (drift reaches stop_drift_deg) or
times out.
"""

from types import SimpleNamespace

from lerobot.agent.orchestrator import TaskAgentOrchestrator
from lerobot.tasks.config import CompletionCriteria, TaskConfig, parse_task_dict
from lerobot.tasks.task_scheduler import TaskResult, TaskStatus


# ── config parsing ──────────────────────────────────────────────────────────

def _compliant_step(**overrides):
    step = {
        "name": "手臂闭合",
        "position": {"left_arm_joint_3": 0.0},
        "max_duration": 2.0,
        "position_tolerance": 3.0,
        "overlap_next": False,
        "stop_on_contact": True,
    }
    step.update(overrides)
    return step


def _parse(step):
    return parse_task_dict(
        {"name": "pickup", "task_type": "position_sequence", "steps": [step]}
    ).steps[0]


def test_step_parses_compliant_fields():
    parsed = _parse(_compliant_step(
        compliant_stop=True,
        compliant_kp=25.0,
        compliant_kd=7.0,
        compliant_stop_drift_deg=4.0,
        compliant_hold_seconds=3.0,
        compliant_max_drift_deg=20.0,
    ))
    assert (
        parsed.compliant_stop,
        parsed.compliant_kp,
        parsed.compliant_kd,
        parsed.compliant_stop_drift_deg,
        parsed.compliant_hold_seconds,
        parsed.compliant_max_drift_deg,
    ) == (True, 25.0, 7.0, 4.0, 3.0, 20.0)


def test_compliant_fields_default_off():
    parsed = _parse(_compliant_step())
    assert parsed.compliant_stop is False
    assert parsed.compliant_kp == 20.0
    assert parsed.compliant_kd == 5.0


def test_step_parses_compliant_joints():
    parsed = _parse(_compliant_step(
        compliant_joints=["left_arm_joint_3", "left_arm_joint_5"],
    ))
    assert parsed.compliant_joints == ["left_arm_joint_3", "left_arm_joint_5"]


def test_compliant_joints_default_empty():
    parsed = _parse(_compliant_step())
    assert parsed.compliant_joints == []


# ── orchestrator wiring ─────────────────────────────────────────────────────

class _CompliantRobot:
    def __init__(self, target, force_sequence):
        self._target = target
        self._force_sequence = force_sequence
        self._i = 0
        self.observation_joint_names = list(target.keys())
        self.gripper_joint_names = []
        self.compliant_hold_calls = []

    def get_current_position(self):
        return dict(self._target)

    def send_action(self, action):
        pass

    def get_observation(self):
        torque = self._force_sequence[min(self._i, len(self._force_sequence) - 1)]
        self._i += 1
        joint = self.observation_joint_names[0]
        return {f"{joint}.force": torque}

    def compliant_hold(self, joint_name, **params):
        self.compliant_hold_calls.append((joint_name, params))
        return 0.0


def _make_orch(robot, control_frequency=500):
    orch = TaskAgentOrchestrator.__new__(TaskAgentOrchestrator)
    orch.robot = robot
    orch.config = SimpleNamespace(robot_config=SimpleNamespace(control_frequency=control_frequency))
    orch.monitor_collector = None
    orch.collision_detector = None
    orch.total_collision_count = 0
    orch.current_cycle = 0
    orch.total_cycles = 0
    orch._add_monitor_event = lambda *a, **k: None
    return orch


def _compliant_criteria():
    return CompletionCriteria(
        type="position",
        target_joint_positions={"left_arm_joint_3": 10.0},
        position_tolerance=3.0,
        stop_on_contact=True,
        contact_joints=["left_arm_joint_3"],
        contact_torque_threshold=1.0,
        contact_baseline_window=5,
        contact_confirm_window=3,
        compliant_stop=True,
        compliant_kp=25.0,
        compliant_kd=7.0,
        compliant_stop_drift_deg=4.0,
        compliant_hold_seconds=3.0,
        compliant_max_drift_deg=20.0,
    )


def test_compliant_stop_calls_compliant_hold_then_stops():
    robot = _CompliantRobot({"left_arm_joint_3": 10.0}, [2.0] * 5 + [3.5] * 20)
    orch = _make_orch(robot)
    task = TaskConfig(
        name="close", task_type="position", max_duration=0.2,
        completion_criteria=_compliant_criteria(),
    )
    result = orch._execute_position_task(task)
    assert result.status == TaskStatus.COMPLETED
    # The compliant branch must have called robot.compliant_hold once per contact joint.
    assert robot.compliant_hold_calls
    joint, params = robot.compliant_hold_calls[0]
    assert joint == "left_arm_joint_3"
    assert params["kp"] == 25.0
    assert params["kd"] == 7.0
    assert params["stop_drift_deg"] == 4.0
    assert params["seconds"] == 3.0
    assert params["max_drift_deg"] == 20.0
    assert params["hold_on_stop"] is True


def test_robot_without_compliant_hold_falls_back_to_rigid_stop():
    # A robot without compliant_hold (non-supre) must still complete via the
    # rigid break path rather than crashing.
    from tests.tasks.test_contact_stop import _FakeRobot

    robot = _FakeRobot({"left_arm_joint_3": 10.0}, [2.0] * 5 + [3.5] * 20)
    orch = _make_orch(robot)
    task = TaskConfig(
        name="close", task_type="position", max_duration=0.2,
        completion_criteria=_compliant_criteria(),
    )
    result = orch._execute_position_task(task)
    assert result.status == TaskStatus.COMPLETED


def test_compliant_stop_uses_compliant_joints_over_contact_joints():
    # When compliant_joints is set, the compliant float must target those
    # joints (not the contact-detection joints), decoupling "which joints
    # detect the touch" from "which joints go soft".
    robot = _CompliantRobot({"left_arm_joint_3": 10.0}, [2.0] * 5 + [3.5] * 20)
    orch = _make_orch(robot)
    criteria = _compliant_criteria()
    criteria.compliant_joints = ["left_arm_joint_5", "left_arm_joint_6"]
    task = TaskConfig(
        name="close", task_type="position", max_duration=0.2,
        completion_criteria=criteria,
    )
    result = orch._execute_position_task(task)
    assert result.status == TaskStatus.COMPLETED
    joints = [call[0] for call in robot.compliant_hold_calls]
    assert joints == ["left_arm_joint_5", "left_arm_joint_6"]
