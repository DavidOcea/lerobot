from pathlib import Path

import pytest

from lerobot.robots.supre_robot_profile import load_profile

# robot1（新）：18 关节 = 7 臂 + joint_8 电机夹爪（左右都有），无串口夹爪
PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "robot1_trunk.yaml"
# robot2（旧）：15 关节 = 6 臂 + left_arm_joint_7 串口夹爪（仅左），+ trunk1/2
PROFILE_2 = Path(__file__).resolve().parents[2] / "profiles" / "robot2_trunk.yaml"

JOINT_ORDER_18 = [
    "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
    "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
    "left_arm_joint_7", "left_arm_joint_8",
    "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
    "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
    "right_arm_joint_7", "right_arm_joint_8",
    "trunk_joint_1", "trunk_joint_2",
]


def test_load_profile_robot1_joint_order():
    p = load_profile(str(PROFILE))
    assert p.joint_order == JOINT_ORDER_18
    assert p.num_joints == 18
    assert p.joint_direction == [1] * 18


def test_load_profile_robot1_calibration():
    p = load_profile(str(PROFILE))
    cal = {c.joint_name: c for c in p.calibration}
    # joint_8 是夹爪（CAN 电机驱动），范围 -45~0
    assert cal["left_arm_joint_8"].min_position == -45.0
    assert cal["left_arm_joint_8"].max_position == 0.0
    assert cal["right_arm_joint_8"].max_position == 0.0
    # joint_7 是臂，范围 -180~180
    assert cal["left_arm_joint_7"].min_position == -180.0
    assert cal["left_arm_joint_7"].max_position == 180.0
    # trunk_joint_2 max=60
    assert cal["trunk_joint_2"].max_position == 60.0


def test_load_profile_robot1_hardware_interfaces():
    p = load_profile(str(PROFILE))
    # 新机器人夹爪是电机（node_id），无 JodellGripperHardware
    assert len(p.hardware_interfaces) == 1
    hw = p.hardware_interfaces[0]
    assert hw["type"] == "EyouMotorHardware"
    assert len(hw["config"]["joints"]) == 18
    node_ids = {j["name"]: j["parameters"]["node_id"] for j in hw["config"]["joints"]}
    assert node_ids["left_arm_joint_8"] == 28   # 左夹爪（电机）
    assert node_ids["right_arm_joint_8"] == 18  # 右夹爪（电机）
    assert node_ids["trunk_joint_2"] == 2


def test_load_profile_robot2_joint_order():
    p = load_profile(str(PROFILE_2))
    assert p.num_joints == 15
    assert "left_arm_joint_7" in p.joint_order
    assert "left_arm_joint_8" not in p.joint_order
    assert p.joint_direction == [1] * 15


def test_load_profile_robot2_hardware_interfaces():
    p = load_profile(str(PROFILE_2))
    assert len(p.hardware_interfaces) == 2
    motor_hw = next(hw for hw in p.hardware_interfaces if hw["type"] == "EyouMotorHardware")
    gripper_hw = next(hw for hw in p.hardware_interfaces if hw["type"] == "JodellGripperHardware")
    assert len(motor_hw["config"]["joints"]) == 14
    assert gripper_hw["name"] == "left_gripper"
    assert gripper_hw["config"]["device"] == "/dev/ttyTHS2"
    assert gripper_hw["config"]["joints"][0]["parameters"]["slave_id"] == 27


def test_load_profile_rejects_bad_direction(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("joints:\n  - {name: j1, direction: 2, min: 0, max: 1, node_id: 1}\n")
    with pytest.raises(ValueError, match="direction must be 1 or -1"):
        load_profile(str(bad))


def test_load_profile_rejects_duplicate_names(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "joints:\n"
        "  - {name: j1, direction: 1, min: 0, max: 1, node_id: 1}\n"
        "  - {name: j1, direction: 1, min: 0, max: 1, node_id: 2}\n"
    )
    with pytest.raises(ValueError, match="Duplicate joint name"):
        load_profile(str(bad))


def test_load_profile_rejects_motor_gripper_conflict(tmp_path):
    bad = tmp_path / "conflict.yaml"
    bad.write_text(
        "joints:\n"
        "  - {name: j1, direction: 1, min: 0, max: 1, node_id: 1, device: /dev/ttyTHS2, slave_id: 27}\n"
    )
    with pytest.raises(ValueError, match="not both"):
        load_profile(str(bad))


def test_load_profile_rejects_gripper_missing_device(tmp_path):
    bad = tmp_path / "grip_missing.yaml"
    bad.write_text(
        "joints:\n"
        "  - {name: left_arm_joint_7, direction: 1, min: 0, max: 1, slave_id: 27}\n"
    )
    with pytest.raises(ValueError, match="requires both"):
        load_profile(str(bad))


def test_follower_init_with_profile():
    pytest.importorskip("eu_motor_py", reason="requires native motor driver (robot-only)")
    from lerobot.robots.supre_robot_follower.supre_robot_follower import SupreRobotFollower
    from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig

    cfg = SupreRobotFollowerConfig(robot_profile=str(PROFILE), prometheus_port=None)
    robot = SupreRobotFollower(cfg)
    assert robot.num_joints == 18
    assert robot._joint_order[0] == "left_arm_joint_1"
    assert robot._joint_order[-1] == "trunk_joint_2"
    assert robot._joint_direction_map["trunk_joint_2"] == 1
    assert "trunk_joint_1" in robot.calibration_limits
    assert robot._profile is not None


def test_hardware_manager_accepts_config_dict():
    pytest.importorskip("eu_motor_py", reason="requires native motor driver (robot-only)")
    from lerobot.robots.supre_robot.supre_robot_hardware_manager import SupreRobotHardwareManager

    p = load_profile(str(PROFILE))
    cfg = {"joint_order": p.joint_order, "hardware_interfaces": p.hardware_interfaces}
    mgr = SupreRobotHardwareManager(config=cfg, control_frequency=30, use_interpolation=False)
    assert mgr.num_joints == 18
    assert mgr.joint_order == p.joint_order
    assert mgr._config is cfg


def test_follower_init_without_profile_uses_legacy():
    pytest.importorskip("eu_motor_py", reason="requires native motor driver (robot-only)")
    from lerobot.robots.supre_robot_follower.supre_robot_follower import SupreRobotFollower
    from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig

    cfg = SupreRobotFollowerConfig(
        joint_config_file="trunk_config_supre_robot_joint.yaml",
        prometheus_port=None,
    )
    robot = SupreRobotFollower(cfg)
    assert robot._profile is None
    assert robot._joint_order[-1] == "trunk_joint_2"
    assert robot._joint_direction_map["left_arm_joint_1"] == 1
