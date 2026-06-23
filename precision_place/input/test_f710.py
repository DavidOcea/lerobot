#!/usr/bin/env python3
"""F710 HID 原始数据测试 — 独立脚本，不连接机器人"""

import sys
import time
import usb.core
import usb.util

VID, PID = 0x046D, 0xC219

dev = usb.core.find(idVendor=VID, idProduct=PID)
if dev is None:
    print("✗ 未找到 F710")
    sys.exit(1)

# 不 detach, 不 set_configuration
try:
    cfg = dev.get_active_configuration()
except Exception as e:
    print(f"✗ 无法获取配置: {e}")
    print("  可能需要: sudo chmod 666 /dev/bus/usb/009/003")
    sys.exit(1)

ep = None
for intf in cfg:
    ep = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
    )
    if ep:
        break

if ep is None:
    print("✗ 未找到 IN 端点")
    sys.exit(1)

print(f"F710 已连接 (端点 0x{ep.bEndpointAddress:02X}, max={ep.wMaxPacketSize}B)")
print("推摇杆/按键看数据变化, Ctrl+C 退出\n")

pkt = 0
try:
    while True:
        data = dev.read(ep.bEndpointAddress, ep.wMaxPacketSize, timeout=500)
        pkt += 1
        raw_hex = ' '.join(f'{b:02X}' for b in data)
        # 解析
        payload = data[1:] if data[0] <= 0x08 else data  # 跳过 report ID
        lx = payload[0] | (payload[1] << 8)  # LE
        ly = payload[2] | (payload[3] << 8)
        dpad = payload[4]
        btns = payload[5] | (payload[6] << 8)
        print(f"[{pkt:03d}] raw={raw_hex}  "
              f"LX={lx:5d}({lx-32768:+5d}) LY={ly:5d}({ly-32768:+5d})  "
              f"DPAD={dpad:#04x} BTN={btns:#06x}")
except KeyboardInterrupt:
    print(f"\n收到 {pkt} 个包")
