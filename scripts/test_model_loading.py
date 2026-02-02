#!/usr/bin/env python3
"""
Test script to verify model loading with the task agent.

This script tests:
1. Policy loading from pretrained model
2. Configuration parsing
3. Camera switching logic
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.tasks.config import load_config_from_yaml, TaskConfig, CameraConfig


def test_model_loading(model_path: str):
    """Test loading an ACT policy model.

    Args:
        model_path: Path to the pretrained model directory.
    """
    print(f"\n{'='*60}")
    print(f"Testing Model Loading: {model_path}")
    print(f"{'='*60}\n")

    # Check if model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False

    print(f"✓ Model path exists")

    # List model files
    print(f"\nModel files:")
    for f in model_path.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")

    # Try to load the policy
    print(f"\n{'='*60}")
    print("Loading ACT Policy...")
    print(f"{'='*60}\n")

    try:
        policy = ACTPolicy.from_pretrained(str(model_path))
        print("✓ Policy loaded successfully!")

        # Print policy info
        print(f"\nPolicy Configuration:")
        print(f"  Type: {policy.config.type}")
        print(f"  Device: {policy.config.device}")
        print(f"  Chunk size: {policy.config.chunk_size}")
        print(f"  Action steps: {policy.config.n_action_steps}")
        print(f"  Vision backbone: {policy.config.vision_backbone}")

        # Check input features
        print(f"\nInput Features:")
        for key, feature in policy.config.input_features.items():
            print(f"  {key}: type={feature.type}, shape={feature.shape}")

        # Check output features
        print(f"\nOutput Features:")
        for key, feature in policy.config.output_features.items():
            print(f"  {key}: type={feature.type}, shape={feature.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size: {total_params * 4 / (1024**2):.1f} MB (float32)")

        return True

    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_config():
    """Test camera configuration for tasks."""
    print(f"\n{'='*60}")
    print("Testing Camera Configuration")
    print(f"{'='*60}\n")

    # Define camera configurations for different tasks
    task_cameras = {
        "pick_short_workpiece": [
            CameraConfig(name="head_cam", type="opencv", index=0, width=640, height=480, fps=30),
            CameraConfig(name="right_wrist_cam", type="opencv", index=6, width=640, height=480, fps=30),
            CameraConfig(name="right_wrist_cam2", type="opencv", index=8, width=640, height=480, fps=30),
        ],
        "pick_long_workpiece": [
            CameraConfig(name="head_cam", type="opencv", index=0, width=640, height=480, fps=30),
            CameraConfig(name="left_wrist_cam", type="opencv", index=2, width=640, height=480, fps=30),
            CameraConfig(name="left_wrist_cam2", type="opencv", index=4, width=640, height=480, fps=30),
        ],
    }

    for task_name, cameras in task_cameras.items():
        print(f"\n{task_name}:")
        for cam in cameras:
            print(f"  - {cam.name}: index={cam.index}, size={cam.width}x{cam.height}, fps={cam.fps}")

    print("\n✓ Camera configuration test passed")


def test_config_loading(config_path: str):
    """Test loading task agent configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file.
    """
    print(f"\n{'='*60}")
    print(f"Testing Configuration Loading: {config_path}")
    print(f"{'='*60}\n")

    try:
        config = load_config_from_yaml(config_path)
        print("✓ Configuration loaded successfully!")

        print(f"\nConfiguration Summary:")
        print(f"  Tasks: {len(config.tasks)}")
        print(f"  Robot type: {config.robot_config.type}")
        print(f"  Camera enabled: {config.robot_config.camera_enabled}")

        for i, task in enumerate(config.tasks):
            print(f"\n  Task {i+1}: {task.name}")
            print(f"    Policy: {task.policy_path}")
            print(f"    Cameras: {len(task.cameras)}")
            for cam in task.cameras:
                print(f"      - {cam.name} (index={cam.index})")

        return True

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    model_path = "/home/smai/dc_dir/lerobot_0901_pybullet/outputs/train/act_1121_3/checkpoints/last/pretrained_model"
    config_path = "configs/task_agent_tasks_test.yaml"

    # Change to the correct directory
    os.chdir("/home/smai/dc_dir/using/lerobot_origin")

    print("\n" + "="*60)
    print("TASK AGENT MODEL LOADING TEST")
    print("="*60)

    # Test 1: Model loading
    result1 = test_model_loading(model_path)

    # Test 2: Camera configuration
    test_camera_config()

    # Test 3: Configuration loading
    result3 = test_config_loading(config_path)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Model Loading: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Configuration Loading: {'✓ PASS' if result3 else '✗ FAIL'}")
    print(f"{'='*60}\n")

    if result1 and result3:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
