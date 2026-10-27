from types import SimpleNamespace

from lerobot.agent.orchestrator import TaskAgentOrchestrator
from lerobot.tasks.config import CompletionCriteria, TaskConfig, parse_task_dict
from lerobot.tasks.task_scheduler import TaskResult, TaskStatus


# ── Task 2: config parsing ──────────────────────────────────────────────────

def _contact_step(**overrides):
    step = {
        "name": "手臂闭合",
        "position": {"left_arm_joint_3": 0.0},
        "max_duration": 2.0,
        "position_tolerance": 3.0,
        "overlap_next": False,
    }
    step.update(overrides)
    return step


def _parse(step):
    return parse_task_dict(
        {"name": "pickup", "task_type": "position_sequence", "steps": [step]}
    )


def test_step_parses_contact_fields():
    step = _contact_step(
        stop_on_contact=True,
        contact_joints=["left_arm_joint_3"],
        contact_torque_threshold=7.5,
        contact_baseline_window=4,
        contact_confirm_window=2,
    )
    parsed = _parse(step).steps[0]
    assert (
        parsed.stop_on_contact,
        parsed.contact_joints,
        parsed.contact_torque_threshold,
        parsed.contact_baseline_window,
        parsed.contact_confirm_window,
    ) == (True, ["left_arm_joint_3"], 7.5, 4, 2)


def test_stop_on_contact_with_overlap_next_raises():
    step = _contact_step(stop_on_contact=True, overlap_next=True)
    try:
        _parse(step)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for stop_on_contact + overlap_next")


# ── Task 3: orchestrator wiring ─────────────────────────────────────────────

class _FakeRobot:
    def __init__(self, target, force_sequence):
        self._target = target
        self._force_sequence = force_sequence
        self._i = 0
        self.observation_joint_names = list(target.keys())
        self.gripper_joint_names = []

    def get_current_position(self):
        return dict(self._target)

    def send_action(self, action):
        pass

    def get_observation(self):
        torque = self._force_sequence[min(self._i, len(self._force_sequence) - 1)]
        self._i += 1
        joint = self.observation_joint_names[0]
        return {f"{joint}.force": torque}


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


def _contact_criteria():
    return CompletionCriteria(
        type="position",
        target_joint_positions={"left_arm_joint_3": 10.0},
        position_tolerance=3.0,
        stop_on_contact=True,
        contact_joints=["left_arm_joint_3"],
        contact_torque_threshold=1.0,
        contact_baseline_window=5,
        contact_confirm_window=3,
    )


def test_execute_position_task_stops_on_contact():
    robot = _FakeRobot({"left_arm_joint_3": 10.0}, [2.0] * 5 + [3.5] * 20)
    orch = _make_orch(robot)
    task = TaskConfig(
        name="close", task_type="position", max_duration=0.2,
        completion_criteria=_contact_criteria(),
    )
    result = orch._execute_position_task(task)
    assert result.status == TaskStatus.COMPLETED and result.error_message is None
    assert result.duration < 0.2  # broke early on contact


def test_execute_position_task_reaches_target_without_contact():
    robot = _FakeRobot({"left_arm_joint_3": 10.0}, [2.0] * 100)
    orch = _make_orch(robot)
    task = TaskConfig(
        name="close", task_type="position", max_duration=0.2,
        completion_criteria=_contact_criteria(),
    )
    result = orch._execute_position_task(task)
    assert result.status == TaskStatus.COMPLETED and result.error_message is None


# ── Final-review fixes ──────────────────────────────────────────────────────

def test_stop_on_contact_step_runs_standalone_not_in_chain(monkeypatch):
    """A stop_on_contact step after an overlap_next step must run standalone,
    not be absorbed as a chain waypoint."""
    calls = []

    def spy(self, step_task):
        calls.append((step_task.name, step_task.completion_criteria.stop_on_contact))
        return TaskResult(task_name=step_task.name, status=TaskStatus.COMPLETED)

    monkeypatch.setattr(TaskAgentOrchestrator, "_execute_position_task", spy)

    task = parse_task_dict({
        "name": "pickup",
        "task_type": "position_sequence",
        "steps": [
            {"name": "手臂下降", "position": {"left_arm_joint_3": 1.0}, "overlap_next": True},
            {"name": "手臂闭合", "position": {"left_arm_joint_3": 2.0},
             "overlap_next": False, "stop_on_contact": True},
        ],
    })
    robot = _FakeRobot({"left_arm_joint_3": 0.0}, [2.0] * 100)
    orch = _make_orch(robot)
    orch._execute_position_sequence_task(task)
    assert ("pickup/手臂闭合", True) in calls


def test_collision_detector_suppressed_for_contact_stop():
    """For a stop_on_contact step, the collision backstop must not run and
    preempt the graceful contact-stop."""

    class _StubCollisionDetector:
        def __init__(self):
            self.checked = 0

        def check_collision(self, observation, action):
            self.checked += 1
            return SimpleNamespace(is_detected=True)

    robot = _FakeRobot({"left_arm_joint_3": 10.0}, [2.0] * 5 + [3.5] * 20)
    orch = _make_orch(robot)
    stub = _StubCollisionDetector()
    orch.collision_detector = stub
    task = TaskConfig(
        name="close", task_type="position", max_duration=0.2,
        completion_criteria=_contact_criteria(),
    )
    result = orch._execute_position_task(task)
    assert result.status == TaskStatus.COMPLETED and result.error_message is None
    assert stub.checked == 0  # collision backstop suppressed
