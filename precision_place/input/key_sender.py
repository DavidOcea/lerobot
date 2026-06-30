#!/usr/bin/env python3
"""
按键发送器 — 在真实终端中运行, 将按键写入 /tmp/robot_keys FIFO

用法:
  python precision_place/input/key_sender.py

支持:
  - 普通键: w/a/s/d/q/e/z/x/1/2/3/m/,/.
  - 方向键: ↑ ↓ ← →
  - ESC 退出

按键格式:
  按下: key
  松开: key-
"""

import sys
import os
import select
import termios
import tty

FIFO_PATH = "/tmp/robot_keys"

# ANSI escape → key name
ESCAPE_MAP = {
    '[A': 'up',
    '[B': 'down',
    '[C': 'right',
    '[D': 'left',
}

# 有效按键
VALID_KEYS = set('wasdqezx123m,.')
VALID_KEYS.update(ESCAPE_MAP.values())


def parse_key(data: bytes) -> tuple:
    """解析按键, 返回 (key_name, bytes_consumed)"""
    if not data:
        return None, 0
    b0 = data[0]
    if b0 == 0x1b:
        if len(data) >= 2 and data[1] == 0x1b:
            return 'esc', 2
        seq = data[1:].decode('latin-1', errors='replace')
        for code, name in ESCAPE_MAP.items():
            if seq.startswith(code):
                return name, 1 + len(code)
        return 'esc', 1
    if 0x20 <= b0 < 0x7f:
        return chr(b0).lower(), 1
    return None, 1


def main():
    # 检查 FIFO
    if not os.path.exists(FIFO_PATH):
        print(f"✗ FIFO 不存在: {FIFO_PATH}")
        print("  请先启动键盘控制程序: python precision_place/input/keyboard_controller.py")
        sys.exit(1)

    # 检查是否是终端
    if not sys.stdin.isatty():
        print("✗ 需要在真实终端中运行 (SSH / VS Code 终端)")
        sys.exit(1)

    # 打开 FIFO (非阻塞, 用于写入)
    try:
        fifo_fd = os.open(FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        print(f"✗ 无法打开 FIFO: {e}")
        sys.exit(1)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)

    print("=" * 50)
    print("键盘发送器 — 按键将控制机械臂")
    print()
    print("  W/A/S/D = XY    ↑↓ = Z    ←→ = Yaw")
    print("  Q/E = Roll       1/2 = 夹爪开/关")
    print("  M = 切换模式      ,/. = 调速")
    print("  ESC = 退出")
    print()
    print("按任意键开始...")
    print("=" * 50)

    held_keys = set()

    try:
        buf = b''
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                data = os.read(sys.stdin.fileno(), 64)
                if not data:
                    break
                buf += data
                while buf:
                    key, consumed = parse_key(buf)
                    if consumed == 0:
                        break
                    buf = buf[consumed:]
                    if key is None or key not in VALID_KEYS:
                        continue

                    if key == 'esc':
                        os.write(fifo_fd, b'esc\n')
                        print("\n退出")
                        return

                    # 松开之前按下的键
                    for old in list(held_keys):
                        if old != key:
                            msg = f'{old}-\n'.encode()
                            os.write(fifo_fd, msg)
                            held_keys.discard(old)

                    # 发送按键
                    msg = f'{key}\n'.encode()
                    os.write(fifo_fd, msg)
                    held_keys.add(key)

                    # 显示
                    display = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}.get(key, key.upper())
                    active = ' '.join(display if k == key else '_' for k in held_keys)
                    sys.stdout.write(f'\r  当前: {active}    ')
                    sys.stdout.flush()

            else:
                # 超时: 释放所有按键
                if held_keys:
                    for key in held_keys:
                        os.write(fifo_fd, f'{key}-\n'.encode())
                    held_keys.clear()
                    sys.stdout.write('\r  当前: (无)    ')
                    sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        # 释放所有键
        for key in held_keys:
            try:
                os.write(fifo_fd, f'{key}-\n'.encode())
            except Exception:
                pass
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        os.close(fifo_fd)
        print("\n已断开")


if __name__ == "__main__":
    main()
