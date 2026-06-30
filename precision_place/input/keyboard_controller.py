#!/usr/bin/env python3
"""
键盘读取器 — 通过命名管道(FIFO)接收按键, 模拟 GamepadState 接口

设计:
  1. 本程序创建 /tmp/robot_keys FIFO, 后台线程读取
  2. 用户在另一个 SSH 终端运行 key_sender.sh 发送按键

用法 (两个终端):
  终端1: python precision_place/input/keyboard_controller.py [--arm right]
  终端2: bash precision_place/input/key_sender.sh

键位:
  W/A/S/D   → XY 平移
  ↑/↓       → Z 升降
  ←/→       → Yaw 偏航
  Q/E       → Roll 滚转
  1/2       → 夹爪 开/关
  M         → 切换模式
  ,/.       → 调速
  ESC       → 退出
"""

import sys
import os
import time
import threading
import copy
import argparse
from pathlib import Path

LEROBOT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(LEROBOT_ROOT))

from precision_place.input.gamepad_controller import GamepadState, GamepadRobotController

FIFO_PATH = "/tmp/robot_keys"


class FifoKeyboardReader:
    """通过 FIFO 读取按键, 输出 GamepadState"""

    KEY_MAP = {
        'w': ('left_stick_y', 1.0),
        's': ('left_stick_y', -1.0),
        'a': ('left_stick_x', -1.0),
        'd': ('left_stick_x', 1.0),
        'up':    ('right_stick_y', 1.0),    # ↑
        'down':  ('right_stick_y', -1.0),   # ↓
        'left':  ('right_stick_x', -1.0),   # ←
        'right': ('right_stick_x', 1.0),    # →
        'q': ('l1', True),
        'e': ('r1', True),
        'z': ('l2', 1.0),
        'x': ('r2', 1.0),
        '1': ('triangle', True),
        '2': ('cross', True),
        '3': ('square', True),
        'm': ('circle', True),
        ',': ('dpad_up', True),
        '.': ('dpad_down', True),
    }

    EXCLUSIVE_GROUPS = [
        {'w', 's'}, {'a', 'd'},
        {'up', 'down'}, {'left', 'right'},
        {'q', 'e'}, {'z', 'x'},
    ]

    KEY_HOLD = 0.15  # 按键保持时间 (秒)

    def __init__(self):
        self._state = GamepadState()
        self._lock = threading.Lock()
        self._active = {}
        self._running = False
        self._thread = None
        self._last_raw_keys = set()
        self._pkt_count = 0

    def start(self):
        if os.path.exists(FIFO_PATH):
            os.unlink(FIFO_PATH)
        os.mkfifo(FIFO_PATH)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"  ✓ FIFO 已创建: {FIFO_PATH}")
        print(f"  → 在另一个终端运行: bash precision_place/input/key_sender.sh")

    def stop(self):
        self._running = False
        if os.path.exists(FIFO_PATH):
            os.unlink(FIFO_PATH)

    def get_state(self) -> GamepadState:
        with self._lock:
            return copy.deepcopy(self._state)

    def get_debug_info(self):
        self._pkt_count += 1
        keys_str = ''.join(sorted(self._last_raw_keys)) if self._last_raw_keys else '-'
        return self._pkt_count, 0, 0, "", f"[keys:{keys_str}]"

    def _read_loop(self):
        """后台线程: 阻塞读取 FIFO"""
        while self._running:
            try:
                fd = os.open(FIFO_PATH, os.O_RDONLY)
                while self._running:
                    data = os.read(fd, 1024)
                    if not data:
                        break  # writer closed, reopen
                    for line in data.decode('latin-1', errors='replace').strip().split('\n'):
                        for token in line.strip().split():
                            if token:
                                self._handle_token(token)
                os.close(fd)
            except OSError:
                time.sleep(0.5)

    def _handle_token(self, token: str):
        """处理单个按键token: 格式 'key' 按下, 'key-' 松开"""
        name = token.rstrip('-')
        is_release = token.endswith('-')

        self._last_raw_keys.add(name)

        if name == 'esc':
            self._running = False
            return

        if name not in self.KEY_MAP:
            return

        with self._lock:
            if is_release:
                self._active.pop(name, None)
                self._last_raw_keys.discard(name)
            else:
                for group in self.EXCLUSIVE_GROUPS:
                    if name in group:
                        for old in list(self._active):
                            if old in group and old != name:
                                del self._active[old]
                        break
                attr, val = self.KEY_MAP[name]
                self._active[name] = (attr, val, time.time() + self.KEY_HOLD)
            self._rebuild_state()

    def _rebuild_state(self):
        s = GamepadState()
        now = time.time()
        for name in list(self._active):
            attr, val, expire = self._active[name]
            if now >= expire:
                del self._active[name]
                self._last_raw_keys.discard(name)
                continue
            current = getattr(s, attr)
            setattr(s, attr, True if isinstance(current, bool) else val)
        self._state = s


def _help_text():
    return """
  ┌──────────────────────────────────────────────────────┐
  │  键盘笛卡尔控制 (FIFO方案)                             │
  ├──────────────────────────────────────────────────────┤
  │  在另一个SSH终端运行:                                  │
  │    bash precision_place/input/key_sender.sh           │
  ├──────────────────────────────────────────────────────┤
  │  SINGLE: W/A/S/D=XY  ↑↓=Z  ←→=Yaw  Q/E=Roll         │
  │          1=夹爪开  2=夹爪关  3=记录                     │
  │  DUAL:   W/A/S/D=左手XY  ↑↓←→=右手XY                  │
  │          Z/X=左手Z  ,/.=右手Z                          │
  │  全局:  M=切换模式  ,=加速  .=减速  ESC=退出           │
  └──────────────────────────────────────────────────────┘
"""


def main():
    parser = argparse.ArgumentParser(description="键盘笛卡尔空间遥操作 (FIFO)")
    parser.add_argument("--arm", default="right", choices=["left", "right"])
    parser.add_argument("--urdf", default=None, help="URDF 路径 (FK 后备, SimpleIBVS 标定可用时不需要)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("键盘笛卡尔空间遥操作 (FIFO方案)")
    print("=" * 60)

    from precision_place.gamepad_teleop import MinimalSystem
    from lerobot.robots.supre_robot_follower import SupreRobotFollower
    from lerobot.robots.supre_robot_follower.supre_robot_follower_config import SupreRobotFollowerConfig
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import ColorMode, OpenCVCameraConfig
    from precision_place.models.calibration_data import ARM_CONFIGS, DEFAULT_URDF_PATH
    from precision_place.dual_point_alignment import PrecisionPlaceController

    system = MinimalSystem()
    system.current_arm = args.arm

    print("\n[1/3] 连接机器人...")
    robot_config = SupreRobotFollowerConfig(
        joint_config_file="trunk_config_supre_robot_joint.yaml"
    )
    system.robot = SupreRobotFollower(robot_config)
    system.robot.connect()
    print("  ✓ 机器人已连接")

    print("\n[2/3] 连接相机...")
    camera_indices = {'head': 0, 'left_wrist': 2, 'right_wrist': 4}
    for name, idx in camera_indices.items():
        try:
            config = OpenCVCameraConfig(
                index_or_path=idx, fps=30, width=640, height=480,
                color_mode=ColorMode.BGR
            )
            system.cameras[name] = OpenCVCamera(config)
            system.cameras[name].connect()
            print(f"  ✓ {name} (索引{idx})")
        except Exception as e:
            print(f"  ✗ {name} (索引{idx}): {e}")

    arm_config = ARM_CONFIGS.get(args.arm)
    if arm_config.camera_name in system.cameras:
        system.controller = PrecisionPlaceController(
            robot=system.robot,
            camera=system.cameras[arm_config.camera_name],
            arm=args.arm,
            camera2=system.cameras.get(arm_config.camera2_name),
            passive_mode=False,
        )
        print(f"  ✓ 控制器已创建 ({args.arm}手)")
    else:
        print(f"  ✗ 相机 {arm_config.camera_name} 未连接")
        system.robot.disconnect()
        return

    print("\n[3/3] 初始化正运动学...")
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
            system.forward_kinematics = system._fk_right or system._fk_left
        except ImportError:
            print("  ✗ FK模块不可用")

    print("\n" + "=" * 60)
    print(_help_text())
    reader = FifoKeyboardReader()
    reader.start()

    try:
        ctrl = GamepadRobotController(system)
        ctrl.run(reader=reader)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n✗ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.stop()
        system.robot.disconnect()
        for cam in system.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        print("\n已断开连接")


if __name__ == "__main__":
    main()
