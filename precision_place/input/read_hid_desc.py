#!/usr/bin/env python3
"""读取 F710 HID Report Descriptor — 查看设备声明的完整输入报告格式

USB HID 标准请求: GET_DESCRIPTOR (0x06) → HID Report Descriptor (0x22)
"""

import sys
import usb.core

VID, PID = 0x046D, 0xC219

dev = usb.core.find(idVendor=VID, idProduct=PID)
if dev is None:
    print("✗ 未找到 F710")
    sys.exit(1)

# 标准 HID 请求: GET_DESCRIPTOR
# bmRequestType = 0x81 (Device-to-Host, Standard, Interface)
# bRequest = 0x06 (GET_DESCRIPTOR)
# wValue = 0x2200 (Report Descriptor, index 0)
# wIndex = 0 (interface 0)
# wLength = 512

try:
    data = dev.ctrl_transfer(
        bmRequestType=0x81,  # IN, Standard, Interface
        bRequest=0x06,       # GET_DESCRIPTOR
        wValue=0x2200,       # Report Descriptor
        wIndex=0,            # Interface 0
        data_or_wLength=512
    )
except usb.core.USBError as e:
    print(f"✗ 无法读取 HID Report Descriptor: {e}")
    print("  可能需要先 detach kernel driver")
    # Try with detach
    try:
        for cfg in dev:
            for intf in cfg:
                try:
                    if dev.is_kernel_driver_active(intf.bInterfaceNumber):
                        dev.detach_kernel_driver(intf.bInterfaceNumber)
                        print(f"  detach interface {intf.bInterfaceNumber}")
                except Exception:
                    pass
        data = dev.ctrl_transfer(0x81, 0x06, 0x2200, 0, 512)
    except usb.core.USBError as e2:
        print(f"✗ detach 后仍然失败: {e2}")
        sys.exit(1)

print(f"✓ HID Report Descriptor ({len(data)} bytes)\n")

# 简单解析 HID Report Descriptor
# 关注: Usage (轴/按钮), Report Size, Report Count, Logical Min/Max
# 以及 Report ID

def parse_hid_rd(data):
    """轻量解析 HID Report Descriptor, 提取报告结构"""
    i = 0
    items = []
    current = {}

    while i < len(data):
        b = data[i]
        if b == 0xC0:  # End Collection
            items.append(('END_COLLECTION', None))
            i += 1
            continue

        bSize = b & 0x03
        bType = (b >> 2) & 0x03
        bTag = (b >> 4) & 0x0F

        if bSize == 0:
            size = 0
        elif bSize == 1:
            size = 1
        elif bSize == 2:
            size = 2
        else:
            size = 4

        if size > 0:
            value = int.from_bytes(data[i+1:i+1+size], 'little', signed=True)
        else:
            value = 0

        # Tag names
        tag_names = {
            0x0: 'INPUT', 0x1: 'OUTPUT', 0x2: 'FEATURE',
            0x3: 'COLLECTION', 0x4: 'END_COLLECTION',
            0x8: 'USAGE', 0x9: 'USAGE_PAGE',
            0xA: 'LOGICAL_MIN', 0xB: 'LOGICAL_MAX',
            0xC: 'PHYSICAL_MIN', 0xD: 'PHYSICAL_MAX',
            0xE: 'REPORT_SIZE', 0xF: 'REPORT_ID',
            0x10: 'REPORT_COUNT',
        }

        tag = tag_names.get(bTag, f'0x{bTag:02X}')

        if tag == 'INPUT' and value > 0:
            flags = []
            if value & 1: flags.append('Data')
            else: flags.append('Const')
            if value & 2: flags.append('Var')
            else: flags.append('Array')
            if value & 4: flags.append('Abs')
            else: flags.append('Rel')
            items.append((tag, f'0x{value:04X} ({", ".join(flags)})'))
        elif tag == 'REPORT_ID':
            items.append((tag, f'0x{value:02X}'))
        elif tag in ('USAGE', 'USAGE_PAGE'):
            items.append((tag, f'0x{value:04X}'))
        elif tag in ('REPORT_SIZE', 'REPORT_COUNT'):
            items.append((tag, value))
        elif tag in ('LOGICAL_MIN', 'LOGICAL_MAX', 'PHYSICAL_MIN', 'PHYSICAL_MAX'):
            items.append((tag, value))
        elif tag == 'COLLECTION':
            coll_types = {0: 'Physical', 1: 'Application', 2: 'Logical'}
            items.append((tag, coll_types.get(value, f'{value}')))
        else:
            items.append((tag, value))

        i += 1 + size

    return items


items = parse_hid_rd(data)

# 先打印原始描述符
print(f"Raw descriptor ({len(data)} bytes):")
print(' '.join(f'{b:02X}' for b in data))

# 按 Report ID 分组显示
current_rid = '0x00'
current_size = 0
current_count = 0
current_lmin = 0
current_lmax = 0
for tag, val in items:
    if tag == 'REPORT_ID':
        current_rid = val
        print(f"\n--- Report ID: {val} ---")
    elif tag == 'USAGE_PAGE':
        page_names = {0x0001: 'Generic Desktop', 0x0009: 'Button'}
        print(f"  Usage Page: {page_names.get(val, val)}")
    elif tag == 'USAGE':
        gd_usages = {
            0x0030: 'X', 0x0031: 'Y', 0x0032: 'Z',
            0x0033: 'Rx', 0x0034: 'Ry', 0x0035: 'Rz',
            0x0036: 'Slider', 0x0037: 'Dial',
            0x0039: 'Hat Switch',
        }
        print(f"  Usage: {gd_usages.get(val, val)}")
    elif tag == 'REPORT_SIZE':
        current_size = val
    elif tag == 'REPORT_COUNT':
        current_count = val
    elif tag == 'LOGICAL_MIN':
        current_lmin = val
    elif tag == 'LOGICAL_MAX':
        current_lmax = val
    elif tag == 'INPUT':
        print(f"  INPUT: {current_size}bit × {current_count}  "
              f"range=[{current_lmin}, {current_lmax}]  flags={val}")
    elif tag == 'COLLECTION':
        print(f"  [{val} Collection]")
    elif tag == 'END_COLLECTION':
        print(f"  [End Collection]")

print(f"\n解析完成")
