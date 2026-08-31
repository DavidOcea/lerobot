"""Load a supre_robot profile (single-file robot description) into derived structures.

Self-contained on purpose: no imports from the hardware-dependent packages
(`supre_robot` / `supre_robot_follower`), so this module stays importable in
environments without the native motor driver (`eu_motor_py`).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

# CAN 总线参数是 supre_robot 控制器的固定属性，所有机器人一致（后续若有差异再进 profile）。
CAN_DEVICE_INDEX = 1
CAN_BAUD_RATE = "1M"


@dataclass
class JointCalibration:
    joint_name: str
    min_position: float
    max_position: float


@dataclass
class RobotProfile:
    joint_order: List[str]
    joint_direction: List[int]  # 与 joint_order 平行，每项 +1 或 -1
    calibration: List[JointCalibration]
    hardware_interfaces: List[Dict[str, Any]]
    num_joints: int


def load_profile(profile_path: str) -> RobotProfile:
    """读 profile YAML，派生 joint_order / direction / calibration / hardware_interfaces。"""
    path = Path(profile_path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    joints = raw["joints"]
    joint_order: List[str] = []
    joint_direction: List[int] = []
    calibration: List[JointCalibration] = []
    motor_joints: List[Dict[str, Any]] = []

    for j in joints:
        name = j["name"]
        direction = j["direction"]
        if direction not in (1, -1):
            raise ValueError(f"Joint '{name}': direction must be 1 or -1, got {direction}")
        if j["min"] >= j["max"]:
            raise ValueError(f"Joint '{name}': min ({j['min']}) must be < max ({j['max']})")

        has_node_id = "node_id" in j
        has_device = "device" in j or "slave_id" in j
        if has_node_id and has_device:
            raise ValueError(
                f"Joint '{name}': must be motor (node_id) OR gripper (device+slave_id), not both"
            )
        if has_device:
            # 夹爪：参考机器人无夹爪，本次不支持
            raise NotImplementedError(f"Joint '{name}': gripper joints are not supported in this first cut")

        joint_order.append(name)
        joint_direction.append(direction)
        calibration.append(
            JointCalibration(joint_name=name, min_position=j["min"], max_position=j["max"])
        )
        motor_joints.append({"name": name, "parameters": {"node_id": j["node_id"]}})

    if len(set(joint_order)) != len(joint_order):
        raise ValueError(f"Duplicate joint name in profile: {joint_order}")

    hardware_interfaces: List[Dict[str, Any]] = []
    if motor_joints:
        hardware_interfaces.append({
            "name": "arm_motors",
            "type": "EyouMotorHardware",
            "interpolation": {"interpolation_n": 3},
            "config": {
                "can_device_index": CAN_DEVICE_INDEX,
                "can_baud_rate": CAN_BAUD_RATE,
                "joints": motor_joints,
            },
        })

    return RobotProfile(
        joint_order=joint_order,
        joint_direction=joint_direction,
        calibration=calibration,
        hardware_interfaces=hardware_interfaces,
        num_joints=len(joint_order),
    )
