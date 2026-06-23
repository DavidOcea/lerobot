#!/usr/bin/env python3
"""
游戏手柄笛卡尔空间遥操作模块

支持 PS/Xbox 手柄通过 evdev 控制机械臂末端在笛卡尔空间运动。
SINGLE 模式: 单臂 5DOF (XY + Z + Yaw + Roll) + 夹爪
DUAL 模式: 双臂各自 3DOF (XY + Z)
"""

import os
import time
import threading
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ==================== Gamepad State ====================

@dataclass
class GamepadState:
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    l2: float = 0.0
    r2: float = 0.0
    cross: bool = False
    circle: bool = False
    square: bool = False
    triangle: bool = False
    l1: bool = False
    r1: bool = False
    select: bool = False
    start: bool = False
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False
    l3: bool = False
    r3: bool = False
    ps_button: bool = False


# ==================== Gamepad Reader ====================

class GamepadReader:
    """后台线程读取 evdev 游戏手柄事件"""

    # PS 手柄按钮映射 (evdev 事件码)
    PS_BTN_MAP = {
        304: 'cross',      # BTN_SOUTH
        305: 'circle',     # BTN_EAST
        307: 'triangle',   # BTN_NORTH
        308: 'square',     # BTN_WEST
        310: 'l1',         # BTN_TL
        311: 'r1',         # BTN_TR
        312: 'l2_btn',     # BTN_TL2
        313: 'r2_btn',     # BTN_TR2
        314: 'select',     # BTN_SELECT
        315: 'start',      # BTN_START
        317: 'l3',         # BTN_THUMBL
        318: 'r3',         # BTN_THUMBR
        316: 'ps_button',  # BTN_MODE
    }

    PS_AXIS_MAP = {
        0: 'left_stick_x',
        1: 'left_stick_y',
        2: 'right_stick_x',  # ABS_RX on PS4, ABS_Z on PS3
        3: 'l2',             # ABS_Z/L2 trigger
        4: 'r2',             # ABS_RZ/R2 trigger
        5: 'right_stick_y',  # ABS_RY
    }

    # D-pad: ABS_HAT0X (16), ABS_HAT0Y (17)
    DPAD_AXES = {16: 'dpad_x', 17: 'dpad_y'}

    def __init__(self, device_path: str = None):
        self._device_path = device_path
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._state = GamepadState()
        self._lock = threading.Lock()
        self._device_name = ""
        self._pyusb_device = None  # (usb_dev, (vid, pid)) 后备
        self._report_size = 20
        self._use_pyusb = False

    def _find_device(self) -> Optional[str]:
        """自动检测 /dev/input/event* 中的游戏手柄"""
        if self._device_path and os.path.exists(self._device_path):
            return self._device_path

        for i in range(32):
            path = f"/dev/input/event{i}"
            if not os.path.exists(path):
                continue
            try:
                with open(f"/sys/class/input/event{i}/device/name") as f:
                    name = f.read().strip()
                if any(kw in name.lower() for kw in
                       ['sony', 'playstation', 'ps4', 'ps5', 'dualsense',
                        'xbox', 'gamepad', 'joystick', 'wireless controller',
                        '8bitdo', 'nintendo', 'pro controller']):
                    self._device_name = name
                    return path
            except (OSError, PermissionError):
                continue
        return None

    def start(self):
        # 方案1: evdev (标准路径)
        device = self._find_device()
        if device is not None:
            self._device_path = device
            self._use_pyusb = False
            print(f"✓ 检测到手柄(evdev): {self._device_name} ({device})")
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return

        # 方案2: pyusb 后备 (绕过内核驱动问题, 如 Jetson 的 logitech 驱动冲突)
        pyusb_dev = self._find_device_pyusb()
        if pyusb_dev is not None:
            self._pyusb_device = pyusb_dev
            self._use_pyusb = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop_pyusb, daemon=True)
            self._thread.start()
            return

        raise RuntimeError(
            "未检测到游戏手柄。请连接手柄后重试。\n"
            "  如果已连接, 可能需要 sudo usermod -a -G input $USER 后重新登录。"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_state(self) -> GamepadState:
        with self._lock:
            return GamepadState(
                left_stick_x=self._state.left_stick_x,
                left_stick_y=self._state.left_stick_y,
                right_stick_x=self._state.right_stick_x,
                right_stick_y=self._state.right_stick_y,
                l2=self._state.l2,
                r2=self._state.r2,
                cross=self._state.cross,
                circle=self._state.circle,
                square=self._state.square,
                triangle=self._state.triangle,
                l1=self._state.l1,
                r1=self._state.r1,
                select=self._state.select,
                start=self._state.start,
                dpad_up=self._state.dpad_up,
                dpad_down=self._state.dpad_down,
                dpad_left=self._state.dpad_left,
                dpad_right=self._state.dpad_right,
                l3=self._state.l3,
                r3=self._state.r3,
                ps_button=self._state.ps_button,
            )

    def _read_loop(self):
        """后台线程: 持续读取 evdev 事件"""
        import evdev
        try:
            device = evdev.InputDevice(self._device_path)
        except PermissionError:
            print("✗ 无权限读取手柄设备, 请运行: sudo usermod -a -G input $USER")
            self._running = False
            return

        for event in device.read_loop():
            if not self._running:
                break

            with self._lock:
                if event.type == evdev.ecodes.EV_ABS:
                    self._handle_axis(event.code, event.value)
                elif event.type == evdev.ecodes.EV_KEY:
                    self._handle_button(event.code, event.value)

    # ==================== pyusb 后备 ====================

    # 已知游戏手柄 VID/PID (按优先级排列)
    _KNOWN_GAMEPADS = [
        (0x046D, 0xC219),  # F710 DirectInput
        (0x046D, 0xC21F),  # F710 XInput (用 hid-generic 时)
        (0x054C, 0x09CC),  # PS4 DualShock 4
        (0x054C, 0x05C4),  # PS4 DualShock 4 (alternate)
        (0x054C, 0x0CE6),  # PS5 DualSense
        (0x045E, 0x028E),  # Xbox 360
    ]

    def _find_device_pyusb(self):
        """用 pyusb 查找游戏手柄 (绕过内核驱动问题)"""
        try:
            import usb.core
        except ImportError:
            return None

        for vid, pid in self._KNOWN_GAMEPADS:
            try:
                dev = usb.core.find(idVendor=vid, idProduct=pid)
                if dev is not None:
                    return dev, (vid, pid)
            except Exception:
                continue
        return None

    def _unbind_kernel_driver(self, usb_dev):
        """分离所有接口的内核驱动"""
        import usb.core
        import usb.util
        for cfg in usb_dev:
            for intf in cfg:
                try:
                    if usb_dev.is_kernel_driver_active(intf.bInterfaceNumber):
                        usb_dev.detach_kernel_driver(intf.bInterfaceNumber)
                except Exception:
                    pass

    def _read_loop_pyusb(self):
        """后台线程: 用 pyusb 直接读 USB 中断端点"""
        import usb.core
        import usb.util

        usb_dev, (vid, pid) = self._pyusb_device

        try:
            self._unbind_kernel_driver(usb_dev)
        except usb.core.USBError as e:
            if "Access denied" in str(e) or "LIBUSB_ERROR_ACCESS" in str(e):
                print("✗ USB 设备访问被拒绝, pyusb 后备需要 root 权限")
                print("  运行: sudo python precision_place/gamepad_teleop.py ...")
            self._running = False
            return

        try:
            usb_dev.set_configuration()
        except Exception:
            pass

        # 查找 IN 端点
        try:
            cfg = usb_dev.get_active_configuration()
            intf = cfg[(0, 0)]
            ep = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
            )
        except usb.core.USBError as e:
            if "Access" in str(e) or "LIBUSB_ERROR_ACCESS" in str(e):
                print("✗ USB 访问被拒绝, pyusb 需要 root 权限")
                print("  请用 sudo 运行: sudo python3 ... gamepad_teleop.py ...")
            else:
                print(f"✗ USB 错误: {e}")
            self._running = False
            return
        except Exception as e:
            print(f"✗ 无法读取 USB 配置: {e}")
            self._running = False
            return
        if ep is None:
            print("✗ 未找到 USB IN 端点")
            return

        self._report_size = ep.wMaxPacketSize
        self._device_name = f"Gamepad ({vid:04X}:{pid:04X} via pyusb)"
        print(f"✓ 通过 USB 直连手柄: {self._device_name} (端点0x{ep.bEndpointAddress:02X}, {self._report_size}B)")

        # 调试: 打印前10个原始 HID 报告以确认字节布局
        self._debug_packet_count = 0

        while self._running:
            try:
                data = usb_dev.read(ep.bEndpointAddress, self._report_size, timeout=200)
                if self._debug_packet_count < 10:
                    print(f"  [DEBUG] raw({len(data)}B): {' '.join(f'{b:02X}' for b in data)}")
                    self._debug_packet_count += 1
                with self._lock:
                    self._parse_hid_report(data)
            except usb.core.USBTimeoutError:
                pass
            except Exception:
                time.sleep(0.1)

    def _parse_hid_report(self, data: bytes):
        """解析 HID 游戏手柄原始报告 → GamepadState

        F710 D-mode: 8字节紧凑格式 (report ID 0x01)
          [0] report_id, [1-2] X_LE, [3-4] Y_LE, [5] dpad, [6-7] buttons_LE

        标准 HID gamepad: 20+ 字节
          [0] report_id, [1-2] X, [3-4] Y, [5-6] Z/Rx, [7-8] Rz/Ry, [9-10] hat, [11+] buttons
        """
        s = self._state
        raw = list(data)
        size = len(raw)

        # 跳过可能的 report ID
        offset = 0
        report_id = 0
        if size > 0 and raw[0] <= 0x08:
            report_id = raw[0]
            offset = 1

        payload = raw[offset:]
        psize = len(payload)

        def read_u16_le(idx):
            if idx + 1 < psize:
                return payload[idx] | (payload[idx + 1] << 8)
            return 0

        def read_s16_le(idx):
            """有符号 16-bit LE (用于以 0 为中心的轴)"""
            val = read_u16_le(idx)
            if val >= 0x8000:
                return val - 0x10000
            return val

        def axis_to_float(val, center=32768, dead=1500):
            """16-bit 轴值 (0-65535, center≈32768) → [-1, 1]"""
            if val >= center:
                v = (val - center) / (65535.0 - center)
            else:
                v = -(center - val) / float(center)
            if abs(v) < dead / float(center):
                return 0.0
            return max(-1.0, min(1.0, v))

        def signed_axis_to_float(val, dead=1000):
            """有符号 16-bit 轴值 (-32768~32767, center=0) → [-1, 1]"""
            v = val / 32768.0
            if abs(val) < dead:
                return 0.0
            return max(-1.0, min(1.0, v))

        # === F710 D-mode 8字节格式 (report ID 0x01) ===
        if psize == 7:
            # bytes 0-1: left stick X (16-bit LE, center=0x8000)
            s.left_stick_x = axis_to_float(read_u16_le(0))
            # bytes 2-3: left stick Y (16-bit LE, center=0x8000)
            s.left_stick_y = axis_to_float(read_u16_le(2))
            # byte 4: d-pad (0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW, 8/0xF=center)
            dpad = payload[4]
            s.dpad_up = dpad in (0, 1, 7)
            s.dpad_right = dpad in (1, 2, 3)
            s.dpad_down = dpad in (3, 4, 5)
            s.dpad_left = dpad in (5, 6, 7)
            # bytes 5-6: buttons bitmask (16-bit LE)
            btns = read_u16_le(5)
            s.cross = bool(btns & (1 << 0))
            s.circle = bool(btns & (1 << 1))
            s.triangle = bool(btns & (1 << 2))
            s.square = bool(btns & (1 << 3))
            s.l1 = bool(btns & (1 << 4))
            s.r1 = bool(btns & (1 << 5))
            s.l2 = 1.0 if bool(btns & (1 << 6)) else 0.0
            s.r2 = 1.0 if bool(btns & (1 << 7)) else 0.0
            s.select = bool(btns & (1 << 8))
            s.start = bool(btns & (1 << 9))
            s.l3 = bool(btns & (1 << 10))
            s.r3 = bool(btns & (1 << 11))
            s.ps_button = bool(btns & (1 << 12))
            return

        # === F710 D-mode right stick report (report ID 0x02 or 0x03) ===
        if psize == 7 and report_id in (2, 3):
            # 右摇杆可能在其他 report ID 中
            s.right_stick_x = axis_to_float(read_u16_le(0))
            s.right_stick_y = axis_to_float(read_u16_le(2))
            return

        # === 标准 HID gamepad (≥10 字节 payload) ===
        if psize >= 10:
            s.left_stick_x = axis_to_float(read_u16_le(0))
            s.left_stick_y = axis_to_float(read_u16_le(2))
            s.right_stick_x = axis_to_float(read_u16_le(4))
            s.right_stick_y = axis_to_float(read_u16_le(6))
            # Hat switch / D-pad
            hat = read_u16_le(8)
            s.dpad_up = (hat >= 0 and hat <= 1000) or (hat >= 34000)
            s.dpad_right = (hat >= 900 and hat <= 2700)
            s.dpad_down = (hat >= 1800 and hat <= 9000)
            s.dpad_left = (hat >= 2700 and hat <= 8100)

            # Buttons bitmask
            btn_offset = 10
            if btn_offset + 1 < psize:
                btns = payload[btn_offset] | (payload[btn_offset + 1] << 8)
                s.cross = bool(btns & (1 << 0))
                s.circle = bool(btns & (1 << 1))
                s.triangle = bool(btns & (1 << 2))
                s.square = bool(btns & (1 << 3))
                s.l1 = bool(btns & (1 << 4))
                s.r1 = bool(btns & (1 << 5))
                s.l2 = 1.0 if bool(btns & (1 << 6)) else 0.0
                s.r2 = 1.0 if bool(btns & (1 << 7)) else 0.0
                s.select = bool(btns & (1 << 8))
                s.start = bool(btns & (1 << 9))
                s.l3 = bool(btns & (1 << 10))
                s.r3 = bool(btns & (1 << 11))
                s.ps_button = bool(btns & (1 << 12))

    def _handle_axis(self, code: int, value: int):
        """处理摇杆/扳机事件"""
        s = self._state
        if code in self.DPAD_AXES:
            axis_name = self.DPAD_AXES[code]
            if axis_name == 'dpad_x':
                s.dpad_left = (value < 0)
                s.dpad_right = (value > 0)
            elif axis_name == 'dpad_y':
                s.dpad_up = (value < 0)
                s.dpad_down = (value > 0)
            return

        norm = value / 32767.0  # normalize to [-1, 1]
        norm = max(-1.0, min(1.0, norm))

        axis_name = self.PS_AXIS_MAP.get(code)
        if axis_name == 'left_stick_x':
            s.left_stick_x = norm
        elif axis_name == 'left_stick_y':
            s.left_stick_y = norm
        elif axis_name == 'right_stick_x':
            s.right_stick_x = norm
        elif axis_name == 'right_stick_y':
            s.right_stick_y = norm
        elif axis_name == 'l2':
            s.l2 = (value / 255.0) if value <= 255 else (value / 1023.0)
        elif axis_name == 'r2':
            s.r2 = (value / 255.0) if value <= 255 else (value / 1023.0)

    def _handle_button(self, code: int, value: int):
        """处理按钮事件"""
        s = self._state
        btn_name = self.PS_BTN_MAP.get(code)
        if btn_name:
            pressed = (value != 0)
            if btn_name == 'l2_btn':
                pass  # PS4/5 L2 is analog axis, button is redundant
            elif btn_name == 'r2_btn':
                pass
            else:
                setattr(s, btn_name, pressed)


# ==================== Robot Controller ====================

class GamepadRobotController:
    """手柄 → 机器人笛卡尔空间遥操作"""

    # 速度档位: (mm/s, deg/s)
    SPEED_PROFILES = {
        'fine':   (2.0, 1.0),
        'medium': (10.0, 5.0),
        'coarse': (30.0, 15.0),
    }

    # DUAL 模式每臂的关节索引
    ARM_JOINTS = {
        'left':  [0, 1, 2, 3, 4, 5],
        'right': [7, 8, 9, 10, 11, 12],
    }

    DEAD_ZONE = 0.10

    def __init__(self, system):
        """
        Args:
            system: PrecisionPlaceSystem 实例, 需已连接
        """
        self.system = system
        self.controller = system.controller
        self.mode = system.current_arm  # "left" / "right" / "dual"
        self.step_profile = 'medium'
        self._last_dpad_up = False
        self._last_dpad_down = False
        self._last_select = False
        self._last_triangle = False
        self._last_cross = False
        self._last_square = False
        self._running = False

    # ==================== 主循环 ====================

    def run(self):
        reader = GamepadReader()
        try:
            reader.start()
        except RuntimeError as e:
            print(f"✗ {e}")
            return

        self._print_help()
        self._running = True
        dt = 1.0 / 30.0  # 30Hz
        self._debug_counter = 0

        print(f"\n  手柄控制已启动 — 当前模式: {self.mode.upper()}")
        print("  [SELECT]切换模式  [PS键]退出\n")

        while self._running:
            loop_start = time.time()
            state = reader.get_state()

            # 调试: 每秒打印一次解析后的摇杆/按钮状态
            self._debug_counter += 1
            if self._debug_counter % 30 == 0:
                print(f"  [STATE] LX:{state.left_stick_x:+.2f} LY:{state.left_stick_y:+.2f} "
                      f"RX:{state.right_stick_x:+.2f} RY:{state.right_stick_y:+.2f} "
                      f"L2:{state.l2:.1f} R2:{state.r2:.1f} "
                      f"btn:△{state.triangle} ×{state.cross} □{state.square} ○{state.circle} "
                      f"L1:{state.l1} R1:{state.r1} SEL:{state.select}")

            # PS 键退出
            if state.ps_button:
                print("\n  手柄控制结束")
                break

            # 模式切换
            self._check_mode_switch(state)

            # 步长切换
            self._check_step_switch(state)

            # 夹爪控制
            self._check_gripper(state)

            # 记录位姿
            self._check_record(state)

            # 计算并发送运动命令
            self._apply_velocity(state, dt)

            # 维持频率
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        reader.stop()
        self._running = False

    # ==================== 模式/步长 ====================

    def _check_mode_switch(self, state: GamepadState):
        if state.select and not self._last_select:
            if self.mode == 'left':
                self.mode = 'right'
            elif self.mode == 'right':
                self.mode = 'dual'
            else:
                self.mode = 'left'
            print(f"  → 模式: {self.mode.upper()}")
        self._last_select = state.select

    def _check_step_switch(self, state: GamepadState):
        profiles = list(self.SPEED_PROFILES.keys())
        idx = profiles.index(self.step_profile)
        if state.dpad_up and not self._last_dpad_up:
            idx = min(idx + 1, len(profiles) - 1)
            self.step_profile = profiles[idx]
            v, w = self.SPEED_PROFILES[self.step_profile]
            print(f"  → 步长: {self.step_profile} ({v}mm/s, {w}deg/s)")
        elif state.dpad_down and not self._last_dpad_down:
            idx = max(idx - 1, 0)
            self.step_profile = profiles[idx]
            v, w = self.SPEED_PROFILES[self.step_profile]
            print(f"  → 步长: {self.step_profile} ({v}mm/s, {w}deg/s)")
        self._last_dpad_up = state.dpad_up
        self._last_dpad_down = state.dpad_down

    # ==================== 速度 → 关节 ====================

    def _apply_velocity(self, state: GamepadState, dt: float):
        v_mm, v_deg = self.SPEED_PROFILES[self.step_profile]

        if self.mode == 'dual':
            self._apply_dual_velocity(state, dt, v_mm)
        else:
            self._apply_single_velocity(state, dt, v_mm, v_deg, self.mode)

    def _apply_single_velocity(self, state: GamepadState, dt: float,
                                v_mm: float, v_deg: float, arm: str):
        """SINGLE 模式: 左摇杆XY, 右摇杆Z+Yaw, L1/R1=Roll"""
        lx = self._deadzone(state.left_stick_x)
        ly = self._deadzone(state.left_stick_y)
        rx = self._deadzone(state.right_stick_x)
        ry = self._deadzone(state.right_stick_y)

        # 笛卡尔增量 (相机坐标系)
        dx_mm = lx * v_mm * dt
        dy_mm = -ly * v_mm * dt   # 摇杆Y: ↑为正, 屏幕Y: ↓为正, 取反
        dz_mm = -ry * v_mm * dt   # 右摇杆Y: ↑=Z+, ↓=Z-
        dyaw = rx * v_deg * dt
        droll = 0.0
        if state.l1:
            droll -= v_deg * dt
        if state.r1:
            droll += v_deg * dt

        if abs(dx_mm) < 0.001 and abs(dy_mm) < 0.001 and \
           abs(dz_mm) < 0.001 and abs(dyaw) < 0.01 and abs(droll) < 0.01:
            return  # 静止

        deltas = self._compute_joint_deltas(dx_mm, dy_mm, dz_mm, dyaw, droll, arm)
        if not deltas:
            return

        self._send_joint_deltas(deltas)

    def _apply_dual_velocity(self, state: GamepadState, dt: float, v_mm: float):
        """DUAL 模式: 左摇杆→左手XY, 右摇杆→右手XY, L2/R2→Z"""
        lx = self._deadzone(state.left_stick_x)
        ly = self._deadzone(state.left_stick_y)
        rx = self._deadzone(state.right_stick_x)
        ry = self._deadzone(state.right_stick_y)

        # 左手: 左摇杆 + L2
        l_dx = lx * v_mm * dt
        l_dy = -ly * v_mm * dt
        l_dz = -state.l2 * v_mm * dt
        if state.l1:
            l_dz = state.l2 * v_mm * dt  # L1+L2 = Z+

        # 右手: 右摇杆 + R2
        r_dx = rx * v_mm * dt
        r_dy = -ry * v_mm * dt
        r_dz = -state.r2 * v_mm * dt
        if state.r1:
            r_dz = state.r2 * v_mm * dt  # R1+R2 = Z+

        all_deltas = {}
        if any(abs(v) > 0.001 for v in [l_dx, l_dy, l_dz]):
            l_deltas = self._compute_joint_deltas(l_dx, l_dy, l_dz, 0, 0, 'left')
            if l_deltas:
                all_deltas.update(l_deltas)
        if any(abs(v) > 0.001 for v in [r_dx, r_dy, r_dz]):
            r_deltas = self._compute_joint_deltas(r_dx, r_dy, r_dz, 0, 0, 'right')
            if r_deltas:
                all_deltas.update(r_deltas)

        if all_deltas:
            self._send_joint_deltas(all_deltas)

    def _compute_joint_deltas(self, dx_mm: float, dy_mm: float, dz_mm: float,
                               dyaw: float, droll: float,
                               arm: str) -> Optional[Dict[int, float]]:
        """
        笛卡尔增量 → 关节增量 (阻尼最小二乘)
        """
        joints = self.controller.get_joint_states()
        if joints is None:
            return None

        j_indices = self.ARM_JOINTS[arm]

        # 构造 Jacobian: 维度 × N_joints
        # 优先级: SimpleIBVS灵敏度 > FK雅可比 > 启发式
        rows = []
        errors = []

        # XY (总是尝试)
        J_xy, _ = self._build_xy_jacobian(joints, j_indices, arm)
        if J_xy is not None:
            rows.append(J_xy[0])  # dx
            rows.append(J_xy[1])  # dy
            errors.extend([dx_mm, dy_mm])

        # Z
        if abs(dz_mm) > 0.001:
            j_z = self._build_z_jacobian(joints, j_indices, arm)
            if j_z is not None:
                rows.append(j_z)
                errors.append(dz_mm)

        # Yaw
        if abs(dyaw) > 0.01:
            j_yaw = self._build_rot_jacobian(joints, j_indices, arm, axis='z')
            if j_yaw is not None:
                rows.append(j_yaw)
                errors.append(dyaw)

        # Roll
        if abs(droll) > 0.01:
            j_roll = self._build_rot_jacobian(joints, j_indices, arm, axis='x')
            if j_roll is not None:
                rows.append(j_roll)
                errors.append(droll)

        if not rows:
            return None

        J = np.array(rows)  # (dim, N)
        error = np.array(errors)

        # 阻尼最小二乘
        error_norm = float(np.linalg.norm(error))
        damping = max(0.1, error_norm * 0.03)
        n = J.shape[1]
        JJT = J @ J.T
        try:
            z = np.linalg.solve(JJT + damping**2 * np.eye(J.shape[0]), -error)
            delta = J.T @ z
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(J) @ (-error)

        # 裁剪 + 增益
        gain = 0.6
        delta = np.clip(delta * gain, -1.5, 1.5)

        result = {}
        for i, jidx in enumerate(j_indices):
            if abs(delta[i]) > 0.005:
                result[jidx] = float(delta[i])
        return result

    def _get_fk(self, arm: str):
        """获取指定手臂的 FK 计算器"""
        # 优先使用 arm-specific FK (run.py 中 _ensure_fk_for_arms 设置的)
        fk_attr = f'_fk_{arm}'
        if hasattr(self.system, fk_attr):
            fk = getattr(self.system, fk_attr)
            if fk is not None:
                return fk
        # 回退到通用 FK
        return self.system.forward_kinematics

    # ==================== Jacobian 构造 ====================

    def _build_xy_jacobian(self, joints: np.ndarray, j_indices: List[int],
                            arm: str) -> Optional[Tuple[np.ndarray, List[int]]]:
        """XY Jacobian (mm/deg) — 优先 SimpleIBVS 灵敏度, 其次 FK 数值微分"""
        # 方案1: SimpleIBVS 灵敏度标定
        try:
            from precision_place.calibration.simple_ibvs import SimpleIBVSController
            ctrl = SimpleIBVSController(arm=arm)
            sens_list, _ = ctrl.get_interpolated_sensitivities(joints)
            if sens_list:
                sens_map = {s.joint_idx: s for s in sens_list}
                has_mm_data = any(abs(s.mm_dx_per_deg) > 0.001 or abs(s.mm_dy_per_deg) > 0.001
                                  for s in sens_list)
                if has_mm_data:
                    J = np.zeros((2, len(j_indices)))
                    for i, jidx in enumerate(j_indices):
                        if jidx in sens_map:
                            J[0, i] = sens_map[jidx].mm_dx_per_deg
                            J[1, i] = sens_map[jidx].mm_dy_per_deg
                    return J, j_indices
        except Exception:
            pass

        # 方案2: FK 数值微分 (后备, 无需标定)
        fk = self._get_fk(arm)
        if fk is not None:
            J_base = self._fk_numerical_xy_jacobian(fk, joints, j_indices)
            if J_base is not None:
                # 旋转到相机坐标系
                R_base_to_cam = self._get_base_to_cam_rotation(fk, joints, arm)
                if R_base_to_cam is not None:
                    # J_cam = R_cam_base @ J_base
                    J_cam = (R_base_to_cam.T @ np.vstack([J_base, np.zeros(len(j_indices))]))[:2]
                    return J_cam, j_indices
                else:
                    # 没有手眼标定, 直接用 base-frame XY
                    return J_base[:2], j_indices
        return None

    def _fk_numerical_xy_jacobian(self, fk, joints: np.ndarray,
                                   j_indices: List[int]) -> Optional[np.ndarray]:
        """FK 数值微分: 计算 flange 位置 (base frame, mm) 对每个关节的偏导"""
        try:
            pose_base = fk.compute(joints)
            pos_base = pose_base.get_position() * 1000.0  # m → mm
            J = np.zeros((3, len(j_indices)))
            delta = 0.2  # 度
            for i, jidx in enumerate(j_indices):
                perturbed = joints.copy().astype(float)
                perturbed[jidx] += delta
                pose = fk.compute(perturbed)
                pos = pose.get_position() * 1000.0
                J[:, i] = (pos - pos_base) / delta
            return J
        except Exception:
            return None

    def _get_base_to_cam_rotation(self, fk, joints: np.ndarray,
                                   arm: str) -> Optional[np.ndarray]:
        """获取 base→camera 的旋转矩阵, 用于旋转 FK Jacobian"""
        try:
            import yaml
            he_path = (Path(__file__).parent.parent /
                       f"hand_eye_extrinsic_{arm}.yaml")
            if not he_path.exists():
                return None
            with open(he_path, 'r') as f:
                data = yaml.safe_load(f)
            T_flange_cam = np.array(data['extrinsic_matrix']['data']).reshape(4, 4)
            R_flange_cam = T_flange_cam[:3, :3]

            pose = fk.compute(joints)
            R_base_flange = pose.rotation_matrix  # 3×3

            # R_base→cam = R_base→flange @ R_flange→cam
            return R_base_flange @ R_flange_cam
        except Exception:
            return None

    def _build_z_jacobian(self, joints: np.ndarray, j_indices: List[int],
                           arm: str) -> Optional[np.ndarray]:
        """Z轴(深度) Jacobian (mm/deg) — 优先 3D/4D 灵敏度, 其次 FK"""
        # 尝试 3D/4D 灵敏度
        try:
            from precision_place.calibration.simple_ibvs import SimpleIBVSController
            ctrl = SimpleIBVSController(arm=arm, dimension=4)
            sens_list, _ = ctrl.get_interpolated_sensitivities(joints)
            if sens_list:
                sens_map = {s.joint_idx: s for s in sens_list}
                has_dz = any(abs(s.depth_dz_per_deg) > 0.001 for s in sens_list)
                if has_dz:
                    j_z = np.zeros(len(j_indices))
                    for i, jidx in enumerate(j_indices):
                        if jidx in sens_map:
                            j_z[i] = sens_map[jidx].depth_dz_per_deg
                    return j_z
        except Exception:
            pass

        # FK 数值雅可比
        fk = self._get_fk(arm)
        if fk is not None:
            return self._fk_numerical_jacobian(
                fk, joints, j_indices, lambda pos: pos[2] * 1000.0)
        return None

    def _build_rot_jacobian(self, joints: np.ndarray, j_indices: List[int],
                             arm: str, axis: str = 'z') -> Optional[np.ndarray]:
        """旋转 Jacobian (deg/deg) — 优先 3D/4D 灵敏度, 其次 FK"""
        # 尝试 3D/4D 灵敏度
        try:
            from precision_place.calibration.simple_ibvs import SimpleIBVSController
            ctrl = SimpleIBVSController(arm=arm, dimension=4)
            sens_list, _ = ctrl.get_interpolated_sensitivities(joints)
            if sens_list:
                sens_map = {s.joint_idx: s for s in sens_list}
                has_rot = any(abs(s.rotation_ddeg_per_deg) > 0.001 for s in sens_list)
                if has_rot:
                    j_rot = np.zeros(len(j_indices))
                    for i, jidx in enumerate(j_indices):
                        if jidx in sens_map:
                            j_rot[i] = sens_map[jidx].rotation_ddeg_per_deg
                    return j_rot
        except Exception:
            pass

        # FK 数值雅可比
        fk = self._get_fk(arm)
        if fk is not None and axis == 'z':
            return self._fk_numerical_jacobian(
                fk, joints, j_indices,
                lambda j: np.arctan2(
                    fk.compute(j).rotation_matrix[1, 0],
                    fk.compute(j).rotation_matrix[0, 0]
                ) * 180.0 / np.pi
            )
        return None

    def _fk_numerical_jacobian(self, fk, joints: np.ndarray,
                                j_indices: List[int],
                                metric_fn) -> Optional[np.ndarray]:
        """FK 数值微分 Jacobian"""
        try:
            base_val = metric_fn(joints)
            j = np.zeros(len(j_indices))
            delta = 0.15
            for i, jidx in enumerate(j_indices):
                perturbed = joints.copy().astype(float)
                perturbed[jidx] += delta
                j[i] = (metric_fn(perturbed) - base_val) / delta
            return j
        except Exception:
            return None

    # ==================== 发送命令 ====================

    def _send_joint_deltas(self, deltas: Dict[int, float]):
        """将关节增量叠加到当前位置并发送"""
        joints = self.controller.get_joint_states()
        if joints is None:
            return

        target = joints.copy()
        for jidx, d in deltas.items():
            target[jidx] += d

        try:
            self.controller._smooth_move_all_joints(target, steps=3)
        except Exception:
            pass

    # ==================== 夹爪 / 记录 ====================

    def _check_gripper(self, state: GamepadState):
        arm_config = self.controller.arm_config
        gripper_idx = arm_config.gripper_idx

        if state.triangle and not self._last_triangle:
            try:
                joints = self.controller.get_joint_states()
                if joints is not None:
                    joints[gripper_idx] = arm_config.gripper_open
                    action = self.controller._build_action_from_16dim(joints)
                    self.controller.robot.send_action(action)
                    print("  夹爪: 开")
            except Exception:
                pass

        if state.cross and not self._last_cross:
            try:
                joints = self.controller.get_joint_states()
                if joints is not None:
                    joints[gripper_idx] = arm_config.gripper_close
                    action = self.controller._build_action_from_16dim(joints)
                    self.controller.robot.send_action(action)
                    print("  夹爪: 关")
            except Exception:
                pass

        self._last_triangle = state.triangle
        self._last_cross = state.cross

    def _check_record(self, state: GamepadState):
        if state.square and not self._last_square:
            joints = self.controller.get_joint_states()
            if joints is not None:
                pose = None
                if self.system.forward_kinematics:
                    try:
                        pose = self.system.forward_kinematics.compute(joints)
                    except Exception:
                        pass

                if pose:
                    print(f"  [记录] 位姿: pos=({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})m  "
                          f"mode={self.mode}")
                else:
                    print(f"  [记录] 关节: {[f'{joints[i]:.2f}' for i in self.ARM_JOINTS.get(self.mode, self.ARM_JOINTS['right'])]}  "
                          f"mode={self.mode}")
        self._last_square = state.square

    # ==================== 工具 ====================

    @staticmethod
    def _deadzone(val: float) -> float:
        if abs(val) < GamepadRobotController.DEAD_ZONE:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - GamepadRobotController.DEAD_ZONE) / \
               (1.0 - GamepadRobotController.DEAD_ZONE)

    def _print_help(self):
        print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  手柄笛卡尔控制                                       │
  ├──────────────────────────────────────────────────────┤
  │  SINGLE 模式:                                         │
  │    左摇杆    → XY 平移                                │
  │    右摇杆↑↓  → Z 升降                                │
  │    右摇杆←→  → Yaw 旋转                              │
  │    L1/R1     → Roll 滚转                             │
  │    △         → 夹爪 开                               │
  │    ×         → 夹爪 关                               │
  │    □         → 记录当前位姿                           │
  │    十字键↑↓  → 步长 +/-                              │
  ├──────────────────────────────────────────────────────┤
  │  DUAL 模式:                                           │
  │    左摇杆    → 左手 XY                                │
  │    右摇杆    → 右手 XY                                │
  │    L2        → 左手 Z↓                               │
  │    R2        → 右手 Z↓                               │
  │    L1+L2     → 左手 Z↑                               │
  │    R1+R2     → 右手 Z↑                               │
  ├──────────────────────────────────────────────────────┤
  │  [SELECT] 切换模式 (LEFT → RIGHT → DUAL)              │
  │  [PS键]   退出                                        │
  └──────────────────────────────────────────────────────┘
  """)
