"""Tests for ``SupreRobotHardwareManager.compliant_hold`` — maps a global joint
name to the owning hardware instance + index and forwards the soft-spring
parameters (the plumbing that lets the orchestrator reach ``compliant_hold``).
"""

import pytest

from lerobot.motors.eyou import EyouMotorHardware
from lerobot.robots.supre_robot.supre_robot_hardware_manager import (
    SupreRobotHardwareManager,
)


def _make_manager(joint_order, joint_map):
    mgr = SupreRobotHardwareManager.__new__(SupreRobotHardwareManager)
    mgr.joint_order = joint_order
    mgr._joint_map = joint_map
    return mgr


def test_compliant_hold_maps_joint_name_to_instance_and_index():
    calls = []
    hw = EyouMotorHardware.__new__(EyouMotorHardware)

    def fake_hold(hw_index, **params):
        calls.append((hw_index, params))
        return 1.5

    hw.compliant_hold = fake_hold
    mgr = _make_manager(["left_arm_joint_5"], {0: {"instance": hw, "hw_index": 2}})

    result = mgr.compliant_hold("left_arm_joint_5", kp=25.0, hold_on_stop=True)

    assert result == 1.5
    assert calls == [(2, {"kp": 25.0, "hold_on_stop": True})]


def test_compliant_hold_unknown_joint_raises():
    mgr = _make_manager(["left_arm_joint_5"], {})
    with pytest.raises(ValueError):
        mgr.compliant_hold("right_arm_joint_5")


def test_compliant_hold_non_eyou_joint_raises():
    class _OtherHardware:
        pass

    mgr = _make_manager(["gripper"], {0: {"instance": _OtherHardware(), "hw_index": 0}})
    with pytest.raises(ValueError):
        mgr.compliant_hold("gripper")
