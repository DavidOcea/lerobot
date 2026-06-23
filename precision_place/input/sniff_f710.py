#!/usr/bin/env python3
"""F710 全量 USB 包嗅探 — 发现所有 report ID 和右摇杆数据

用法: python precision_place/input/sniff_f710.py
按 Ctrl+C 退出，会打印每种 report ID 的统计。
"""

import sys
import time
import usb.core
import usb.util
from collections import defaultdict

VID, PID = 0x046D, 0xC219

dev = usb.core.find(idVendor=VID, idProduct=PID)
if dev is None:
    print("✗ 未找到 F710")
    sys.exit(1)

try:
    cfg = dev.get_active_configuration()
except Exception as e:
    print(f"✗ 无法获取配置: {e}")
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
print("请操作所有摇杆和按键，Ctrl+C 退出\n")

# 统计每种 report ID
report_stats = defaultdict(lambda: {"count": 0, "sizes": set(), "examples": []})

pkt = 0
try:
    while True:
        try:
            data = dev.read(ep.bEndpointAddress, ep.wMaxPacketSize, timeout=200)
        except usb.core.USBTimeoutError:
            continue

        pkt += 1
        raw_hex = ' '.join(f'{b:02X}' for b in data)
        rid = data[0]
        payload = data[1:]
        psize = len(payload)

        stats = report_stats[rid]
        stats["count"] += 1
        stats["sizes"].add(len(data))
        if len(stats["examples"]) < 5:
            stats["examples"].append(raw_hex)

        # 解析 0x01 (左摇杆 + 按键)
        if rid == 0x01 and psize >= 7:
            lx = payload[0] | (payload[1] << 8)
            ly = payload[2] | (payload[3] << 8)
            dpad_lo = payload[4] & 0x0F
            dpad_hi = (payload[4] >> 4) & 0x0F
            btns = payload[5] | (payload[6] << 8)
            # 只打印有变化的数据 (非中心)
            is_centered = (abs(lx - 32768) < 500 and abs(ly - 32768) < 500 and dpad_lo in (8, 0xF))
            marker = "" if is_centered else " ***"
            print(f"[{pkt:04d}] ID=0x{rid:02X} sz={len(data)} | "
                  f"LX={lx:5d}({lx-32768:+5d}) LY={ly:5d}({ly-32768:+5d}) "
                  f"DPAD_LO={dpad_lo:#x} DPAD_HI={dpad_hi:#x} BTN=0x{btns:04x}{marker}")

        # 解析 0x02 (可能是右摇杆)
        elif rid == 0x02 and psize >= 7:
            v0 = payload[0] | (payload[1] << 8)
            v1 = payload[2] | (payload[3] << 8)
            v2 = payload[4] | (payload[5] << 8)
            v3 = payload[6]
            is_centered = abs(v0 - 32768) < 500 and abs(v1 - 32768) < 500
            marker = "" if is_centered else " ***"
            print(f"[{pkt:04d}] ID=0x{rid:02X} sz={len(data)} | "
                  f"V0={v0:5d} V1={v1:5d} V2={v2:5d} V3={v3:3d} raw=[{raw_hex}]{marker}")

        # 其他 report ID
        else:
            print(f"[{pkt:04d}] ID=0x{rid:02X} sz={len(data)} raw=[{raw_hex}]")

except KeyboardInterrupt:
    print(f"\n\n=== 统计 ({pkt} 个包) ===")
    for rid in sorted(report_stats.keys()):
        stats = report_stats[rid]
        sizes = sorted(stats["sizes"])
        print(f"  Report ID 0x{rid:02X}: {stats['count']} 包, 大小={sizes}")
        for ex in stats["examples"]:
            print(f"    示例: {ex}")
    print(f"\n收到 {pkt} 个包")
