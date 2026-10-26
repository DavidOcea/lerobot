from lerobot.tasks.config import parse_task_dict


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
