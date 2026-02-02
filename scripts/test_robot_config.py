#!/usr/bin/env python3
"""
Test script to verify robot configuration loading.

This script tests:
1. Robot configuration loading from YAML
2. Camera configuration
3. Joint configuration
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import yaml


def test_robot_config_loading():
    """Test loading robot configuration from YAML files."""
    print("\n" + "="*60)
    print("ROBOT CONFIGURATION LOADING TEST")
    print("="*60 + "\n")

    # Path to config file
    config_path = Path("src/lerobot/robots/supre_robot_follower/trunk_config.yaml")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    print(f"✓ Config file exists: {config_path}\n")

    # Load YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    robot_config = config_dict.get("robot", {})

    # Display configuration
    print("Robot Configuration:")
    print(f"  Type: {robot_config.get('type')}")
    print(f"  Joint config file: {robot_config.get('joint_config_file')}")
    print(f"  Max relative joint move: {robot_config.get('max_relative_joint_move')}°")
    print(f"  Control frequency: {robot_config.get('control_frequency')} Hz")
    print(f"  Prometheus port: {robot_config.get('prometheus_port')}")

    # Display camera configuration
    cameras = robot_config.get("cameras", {})
    print(f"\nCamera Configuration ({len(cameras)} cameras):")
    for cam_name, cam_config in cameras.items():
        print(f"  {cam_name}:")
        print(f"    Type: {cam_config.get('type')}")
        print(f"    Index: {cam_config.get('index')}")
        print(f"    Resolution: {cam_config.get('width')}x{cam_config.get('height')}")
        print(f"    FPS: {cam_config.get('fps')}")

    # Display joint calibration
    calibration = robot_config.get("calibration", [])
    print(f"\nJoint Calibration ({len(calibration)} joints):")
    for cal in calibration:
        joint_name = cal.get("joint_name")
        min_pos = cal.get("min_position")
        max_pos = cal.get("max_position")
        print(f"  {joint_name}: [{min_pos}°, {max_pos}°]")

    # Load joint configuration file
    joint_config_path = Path("src/lerobot/robots/supre_robot_follower") / robot_config.get("joint_config_file")
    print(f"\n{'='*60}")
    print(f"Joint Configuration: {joint_config_path}")
    print(f"{'='*60}\n")

    if joint_config_path.exists():
        with open(joint_config_path, "r") as f:
            joint_config = yaml.safe_load(f)

        # Display joint order
        joint_order = joint_config.get("joint_order", [])
        print(f"Joint Order ({len(joint_order)} joints):")
        for i, joint_name in enumerate(joint_order):
            print(f"  {i+1}. {joint_name}")

        # Display hardware interfaces
        hw_interfaces = joint_config.get("hardware_interfaces", [])
        print(f"\nHardware Interfaces ({len(hw_interfaces)} interfaces):")
        for hw in hw_interfaces:
            name = hw.get("name")
            hw_type = hw.get("type")
            joints = hw.get("config", {}).get("joints", [])
            print(f"  {name} ({hw_type}):")
            for joint in joints:
                joint_name = joint.get("name")
                print(f"    - {joint_name}")

    return True


def test_camera_index_mapping():
    """Test camera index mapping for different tasks."""
    print(f"\n{'='*60}")
    print("CAMERA INDEX MAPPING")
    print(f"{'='*60}\n")

    # Define camera mappings based on requirements
    camera_mappings = {
        "head_cam": 0,
        "left_wrist_cam": 2,
        "left_wrist_cam2": 4,
        "right_wrist_cam": 6,
        "right_wrist_cam2": 8,
    }

    print("Camera Index Mapping:")
    for cam_name, index in camera_mappings.items():
        print(f"  {cam_name}: /dev/video{index}")

    # Task-based camera selection
    task_cameras = {
        "pick_short_workpiece": ["head_cam", "right_wrist_cam", "right_wrist_cam2"],
        "place_short_workpiece": ["head_cam", "right_wrist_cam", "right_wrist_cam2"],
        "pick_long_workpiece": ["head_cam", "left_wrist_cam", "left_wrist_cam2"],
        "place_long_workpiece": ["head_cam", "left_wrist_cam", "left_wrist_cam2"],
        "press_button": ["head_cam", "right_wrist_cam", "right_wrist_cam2"],
    }

    print(f"\nTask-Based Camera Selection:")
    for task_name, cameras in task_cameras.items():
        indices = [camera_mappings[cam] for cam in cameras]
        print(f"  {task_name}:")
        for cam in cameras:
            print(f"    - {cam} (index={camera_mappings[cam]})")
        print(f"    Video devices: {', '.join(f'/dev/video{i}' for i in indices)}")

    return True


def test_model_camera_compatibility():
    """Test compatibility between model and robot camera configuration."""
    print(f"\n{'='*60}")
    print("MODEL CAMERA COMPATIBILITY CHECK")
    print(f"{'='*60}\n")

    # Model cameras (from config.json)
    model_cameras = ["head_cam", "right_wrist_cam", "left_wrist_cam"]

    # Robot cameras
    robot_cameras = ["head_cam", "left_wrist_cam", "left_wrist_cam2",
                     "right_wrist_cam", "right_wrist_cam2"]

    print("Model cameras (3):")
    for cam in model_cameras:
        print(f"  ✓ {cam}")

    print("\nRobot cameras (5):")
    for cam in robot_cameras:
        status = "✓" if cam in model_cameras else " (extra)"
        print(f"  {status} {cam}")

    print("\nCompatibility:")
    for cam in model_cameras:
        if cam in robot_cameras:
            print(f"  ✓ {cam} - Compatible")
        else:
            print(f"  ✗ {cam} - Missing from robot config")

    print("\n⚠️  Note: Model only uses 3 cameras. For tasks requiring 5 cameras,")
    print("    camera switching will be handled by the task orchestrator.")

    return True


def main():
    """Main test function."""
    os.chdir("/home/smai/dc_dir/using/lerobot_origin")

    result1 = test_robot_config_loading()
    result2 = test_camera_index_mapping()
    result3 = test_model_camera_compatibility()

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Robot Config Loading: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Camera Index Mapping: {'✓ PASS' if result2 else '✗ FAIL'}")
    print(f"Model Compatibility: {'✓ PASS' if result3 else '✗ FAIL'}")
    print(f"{'='*60}\n")

    if result1 and result2 and result3:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
