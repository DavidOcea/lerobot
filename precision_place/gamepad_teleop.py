#!/usr/bin/env python3
"""
手柄笛卡尔空间遥操作 — 独立启动脚本

直接启动手柄控制，无需经过 run.py 菜单。
用法: python precision_place/gamepad_teleop.py [--arm left|right] [--urdf /path/to/robot.urdf]

示例:
  python precision_place/gamepad_teleop.py
  python precision_place/gamepad_teleop.py --arm left
  python precision_place/gamepad_teleop.py --urdf /home/smai/dc_dir/urdf/RJ2506/urdf/RJ2506.urdf
"""

import sys
import time
import argparse
from pathlib import Path

# 路径设置
LEROBOT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(LEROBOT_ROOT))

from lerobot.robots.supre_robot_follower import SupreRobotFollower
from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from precision_place.models.calibration_data import ARM_CONFIGS, DEFAULT_URDF_PATH
from precision_place.dual_point_alignment import PrecisionPlaceController
from precision_place.input.gamepad_controller import GamepadRobotController


class MinimalSystem:
    """精简版 System — 仅包含手柄控制所需的最小上下文"""

    def __init__(self):
        self.robot = None
        self.cameras = {}
        self.controller = None
        self.current_arm = "right"
        self.forward_kinematics = None
        self.urdf_path = None
        self._fk_left = None
        self._fk_right = None


def main():
    parser = argparse.ArgumentParser(description="手柄笛卡尔空间遥操作")
    parser.add_argument("--arm", default="right", choices=["left", "right"],
                        help="初始手臂 (默认: right, 运行时可用SELECT切换)")
    parser.add_argument("--urdf", default=None,
                        help="URDF文件路径 (用于FK后备，无SimpleIBVS标定时需要)")
    parser.add_argument("--no-fk", action="store_true",
                        help="禁用FK后备 (仅使用SimpleIBVS标定)")
    args = parser.parse_args()

    system = MinimalSystem()
    system.current_arm = args.arm

    print("\n" + "=" * 60)
    print("手柄笛卡尔空间遥操作 (独立模式)")
    print("=" * 60)

    # 1. 连接机器人
    print("\n[1/3] 连接机器人...")
    robot_config = SupreRobotFollowerConfig(
        joint_config_file="trunk_config_supre_robot_joint.yaml"
    )
    system.robot = SupreRobotFollower(robot_config)
    system.robot.connect()
    print("  ✓ 机器人已连接")

    # 2. 连接相机 (可选，SimpleIBVS需要)
    print("\n[2/3] 连接相机...")
    camera_indices = {
        'head': 0,
        'left_wrist': 2,
        'right_wrist': 4,
    }
    for name, idx in camera_indices.items():
        try:
            from lerobot.cameras.opencv.configuration_opencv import ColorMode
            config = OpenCVCameraConfig(
                index_or_path=idx, fps=30, width=640, height=480,
                color_mode=ColorMode.BGR
            )
            system.cameras[name] = OpenCVCamera(config)
            system.cameras[name].connect()
            print(f"  ✓ {name} (索引{idx})")
        except Exception as e:
            print(f"  ✗ {name} (索引{idx}): {e}")

    # 创建控制器
    arm_config = ARM_CONFIGS.get(args.arm)
    if arm_config.camera_name in system.cameras:
        system.controller = PrecisionPlaceController(
            robot=system.robot,
            camera=system.cameras[arm_config.camera_name],
            arm_config=arm_config,
            camera2=system.cameras.get(arm_config.camera2_name),
            passive=False,
        )
        print(f"  ✓ 控制器已创建 ({args.arm}手)")
    else:
        print(f"  ✗ 相机 {arm_config.camera_name} 未连接, 无法创建控制器")
        system.robot.disconnect()
        return

    # 3. 初始化 FK (作为后备)
    print("\n[3/3] 初始化正运动学...")
    if args.no_fk:
        print("  - 已禁用FK (--no-fk)")
    else:
        urdf_path = args.urdf or DEFAULT_URDF_PATH
        if urdf_path:
            try:
                from precision_place.calibration.forward_kinematics import create_fk_from_urdf
                system.urdf_path = urdf_path
                for arm in ['left', 'right']:
                    try:
                        fk = create_fk_from_urdf(urdf_path, arm)
                        setattr(system, f'_fk_{arm}', fk)
                        print(f"  ✓ FK已初始化 ({arm}臂)")
                    except Exception as e:
                        setattr(system, f'_fk_{arm}', None)
                        print(f"  ✗ FK失败 ({arm}臂): {e}")
                # 设置默认 FK 用于兼容
                system.forward_kinematics = system._fk_right or system._fk_left
            except ImportError:
                print("  ✗ FK模块不可用, 仅SimpleIBVS标定可用")
        else:
            print("  - 未提供URDF, XY控制需要SimpleIBVS标定")
            print("    设置默认路径: 编辑 models/calibration_data.py → DEFAULT_URDF_PATH")
            print("    或在命令行指定: --urdf /path/to/robot.urdf")

    # 4. 启动手柄控制
    print("\n" + "=" * 60)
    try:
        ctrl = GamepadRobotController(system)
        ctrl.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n✗ 手柄控制异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.robot.disconnect()
        for cam in system.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        print("\n已断开连接")


if __name__ == "__main__":
    main()
