"""
Configuration classes for the safety module.

Defines collision detection configuration and related settings.
"""

from dataclasses import dataclass, field
from typing import Any

import draccus


@dataclass
class CollisionConfig:
    """Configuration for collision detection system.

    This configuration controls how collisions are detected based on
    torque/force sensing from the robot joints.
    """

    # Detection thresholds
    collision_threshold: float = 2.0  # Nm - torque anomaly threshold
    detection_window: int = 5  # Consecutive steps above threshold

    # Advanced detection settings
    adaptive_mode: bool = True  # Adjust threshold based on motion state
    velocity_compensation: bool = True  # Compensate for inertial torques
    inertia_compensation_factor: float = 0.1  # Factor for inertia estimation

    # Per-joint thresholds (overrides global threshold)
    joint_specific_thresholds: dict[str, float] = field(default_factory=dict)

    # Safety limits
    max_torque_limit: float = 5.0  # Absolute maximum torque (Nm) before emergency stop

    # Recovery settings
    recovery_strategy: str = "stop_and_retreat"  # "stop", "stop_and_retreat", "retry"
    recovery_retreat_distance: float = 0.05  # Radians to retreat after collision
    recovery_timeout: float = 5.0  # Seconds to wait for recovery

    # Calibration settings
    auto_calibrate: bool = True  # Automatically calibrate base torques on startup
    calibration_samples: int = 100  # Number of samples for calibration
    calibration_threshold: float = 0.1  # Torque variance threshold for calibration

    # Joint inertia parameters (for velocity compensation)
    # These are approximate inertia values for each joint (kg*m^2)
    joint_inertia: dict[str, float] = field(default_factory=lambda: {
        "left_arm_joint_1": 0.5,
        "left_arm_joint_2": 0.4,
        "left_arm_joint_3": 0.3,
        "left_arm_joint_4": 0.2,
        "left_arm_joint_5": 0.1,
        "left_arm_joint_6": 0.05,
        "left_arm_joint_7": 0.02,  # Gripper
        "right_arm_joint_1": 0.5,
        "right_arm_joint_2": 0.4,
        "right_arm_joint_3": 0.3,
        "right_arm_joint_4": 0.2,
        "right_arm_joint_5": 0.1,
        "right_arm_joint_6": 0.05,
        "right_arm_joint_7": 0.02,  # Gripper
    })
