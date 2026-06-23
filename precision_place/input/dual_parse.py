#!/usr/bin/env python3
"""F710 双解析对比: 8-bit (HID描述符) vs 16-bit (当前代码)"""
import sys, time, usb.core, usb.util

VID, PID = 0x046D, 0xC219
dev = usb.core.find(idVendor=VID, idProduct=PID)
if dev is None:
    print("✗ 未找到 F710"); sys.exit(1)

cfg = dev.get_active_configuration()
ep = None
for intf in cfg:
    ep = usb.util.find_descriptor(intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)
    if ep: break

print("F710 双解析 — 请只推动右摇杆(不碰左摇杆), Ctrl+C退出\n")
print(f"{'#':>4s} {'8bit_X':>6s} {'8bit_Y':>6s} {'8bit_Z':>6s} {'8bit_Rz':>6s} | {'16bit_LX':>8s} {'16bit_LY':>8s} | DPAD BTN")
print("-" * 90)

pkt = 0
try:
    while True:
        try:
            data = dev.read(ep, ep.wMaxPacketSize, timeout=200)
        except usb.core.USBTimeoutError:
            continue
        pkt += 1
        payload = data[1:]
        # 8-bit 解析 (HID描述符定义: X, Y, Z, Rz 各8bit, range [0,255], center≈128)
        x8, y8, z8, rz8 = payload[0], payload[1], payload[2], payload[3]
        x8_d = x8 - 128; y8_d = y8 - 128; z8_d = z8 - 128; rz8_d = rz8 - 128
        # 16-bit 解析 (当前代码: 2轴各16bit LE, center≈32768)
        lx = payload[0] | (payload[1] << 8)
        ly = payload[2] | (payload[3] << 8)
        lx_d = lx - 32768; ly_d = ly - 32768
        dpad = payload[4] & 0x0F
        btns = payload[5] | (payload[6] << 8)

        # 只打印有变化的数据 (任一轴偏离中心)
        if max(abs(x8_d), abs(y8_d), abs(z8_d), abs(rz8_d)) > 3:
            print(f"{pkt:4d} {x8:4d}({x8_d:+4d}) {y8:4d}({y8_d:+4d}) {z8:4d}({z8_d:+4d}) {rz8:4d}({rz8_d:+4d}) | "
                  f"{lx:6d}({lx_d:+6d}) {ly:6d}({ly_d:+6d}) | "
                  f"D={dpad:#x} B=0x{btns:04x}")
except KeyboardInterrupt:
    print(f"\n收到 {pkt} 个包")
