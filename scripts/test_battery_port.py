#!/usr/bin/env python3
"""Test which serial port connects to the battery."""
import serial
import time

ADDR = 0x01  # Try default device address

def send_cmd(ser, cid1, cid2, info=""):
    """Send command frame and return raw response."""
    frame_body = f"02{ADDR:02X}{cid1:02X}{cid2:02X}{len(info)//2:04X}{info}"
    chksum = sum(frame_body.encode('ascii')) & 0xFFFF
    frame = f"~{frame_body}{chksum:04X}\r"
    ser.write(frame.encode('ascii'))
    ser.flush()
    return ser.read(512)

for port in ['/dev/ttyTHS1', '/dev/ttyTHS2']:
    print(f"\n=== Testing {port} ===")
    try:
        ser = serial.Serial(port, 9600, timeout=1.0)
        print(f"  opened OK")

        # Try 46 47: read analog data
        resp = send_cmd(ser, 0x46, 0x47)
        if resp:
            print(f"  46 47 resp ({len(resp)}B): {resp.hex()}")
            print(f"  46 47 ascii: {resp[:120]}")
        else:
            print(f"  46 47: no response")

        time.sleep(0.1)

        # Try 46 42: read pack info (command=0xFF = query all)
        resp = send_cmd(ser, 0x46, 0x42, "FF00")
        if resp:
            print(f"  46 42 resp ({len(resp)}B): {resp.hex()}")
        else:
            print(f"  46 42: no response")

        ser.close()
    except Exception as e:
        print(f"  ERROR: {e}")