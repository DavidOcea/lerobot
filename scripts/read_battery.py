#!/usr/bin/env python3
"""Read battery data via RS-485 and display all interpretations.

Run on the robot: python scripts/read_battery.py
"""
import serial
import time

PORT = '/dev/ttyTHS2'
VER = 0x20
ADDR = 0x01

def read_battery(ser):
    """Send 46 42 (all pack) and return decoded response."""
    info = "FF00"
    frame_body = f"{VER:02X}{ADDR:02X}{0x46:02X}{0x42:02X}{len(info)//2:04X}{info}"
    chksum = sum(frame_body.encode('ascii')) & 0xFFFF
    frame = f"~{frame_body}{chksum:04X}\r"
    ser.write(frame.encode('ascii'))
    ser.flush()
    raw = ser.read(512)
    if not raw:
        return None

    # Decode: SOI(1) + ASCII body + EOI(1)
    ascii_body = raw[1:-1].decode('ascii')
    # Hex-decode to binary
    return bytes.fromhex(ascii_body)

def show(data):
    """Display battery data with multiple interpretations."""
    # Header: 20 01 46 00 C0 6E 11 01 0D 0F (10 bytes)
    info_data = data[10:-2]  # Skip header and last 2 bytes

    print(f"Raw INFO ({len(info_data)}B): {info_data.hex()}")

    # 12 cell-like values (uint16 LE)
    print(f"\n  Cell-like values (uint16 LE, mV?):")
    for i in range(12):
        v = int.from_bytes(info_data[i*2:i*2+2], 'little')
        print(f"    Cell[{i:2d}]: {v} ({v/1000:.3f}V)")

    # Fixed values region
    print(f"\n  Static fields:")
    print(f"    Byte[24] LE: {int.from_bytes(info_data[24:26], 'little')}")

    # Percentage/status fields (uint8 BE)
    print(f"    Pct-like fields (BE): {info_data[27]}%  |  {info_data[29]}%  |  {info_data[31]}%")

    # Flags
    print(f"    Flags byte: 0x{info_data[32]:02X}")

    # Voltage-like (uint16 BE, maybe mV)
    v1 = int.from_bytes(info_data[33:35], 'big')  # B6 C8 → 46792
    v2 = int.from_bytes(info_data[35:37], 'big')  # B2 03 → 45571
    print(f"    Field A (BE uint16): {v1} ({v1/1000:.3f}V if mV)")
    print(f"    Field B (BE uint16): {v2} ({v2/1000:.3f}V if mV)")

    # Current-like (int16 BE)
    i1 = int.from_bytes(info_data[33:35], 'big', signed=True)
    i2 = int.from_bytes(info_data[35:37], 'big', signed=True)
    print(f"    Field A (BE int16): {i1}")
    print(f"    Field B (BE int16): {i2}")

    # More fields
    print(f"\n    Field[37:39] (BE): {int.from_bytes(info_data[37:39], 'big')}")
    print(f"    Field[39:41] (BE): {int.from_bytes(info_data[39:41], 'big')}")
    print(f"    Field[41:43] (BE): {int.from_bytes(info_data[41:43], 'big')} → 0x{int.from_bytes(info_data[41:43], 'big'):04X}")
    print(f"    Field[43:45] (BE): {int.from_bytes(info_data[43:45], 'big')}")
    print(f"    Field[45:47] (BE): {int.from_bytes(info_data[45:47], 'big')}")
    print()

ser = serial.Serial(PORT, 9600, timeout=1.0)
for i in range(5):
    data = read_battery(ser)
    if data:
        print(f"=== Read #{i+1} ===")
        show(data)
    time.sleep(1)
ser.close()