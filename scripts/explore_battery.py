#!/usr/bin/env python3
"""Explore battery protocol by trying different commands."""
import serial
import time

PORT = '/dev/ttyTHS2'
ADDR = 0x01
VER = 0x20

def send_cmd(ser, cid1, cid2, info=""):
    frame_body = f"{VER:02X}{ADDR:02X}{cid1:02X}{cid2:02X}{len(info)//2:04X}{info}"
    chksum = sum(frame_body.encode('ascii')) & 0xFFFF
    frame = f"~{frame_body}{chksum:04X}\r"
    print(f"  TX: {frame.strip()}")
    ser.write(frame.encode('ascii'))
    ser.flush()
    resp = ser.read(512)
    if resp:
        print(f"  RX ({len(resp)}B): {resp.hex()}")
        print(f"  RX ascii: {resp}")
        return resp
    else:
        print(f"  RX: no response")
        return None

ser = serial.Serial(PORT, 9600, timeout=1.0)

commands = [
    ("Device Info",        0x46, 0x4F, ""),
    ("Get Analog Data",    0x46, 0x47, ""),
    ("Get System Param",   0x46, 0x51, ""),
    ("Get Pack Info",      0x46, 0x90, ""),
    ("Get Data (all pack)", 0x46, 0x42, "FF00"),
    ("Get Data (pack 1)",  0x46, 0x42, "0100"),
    ("Get Pack Status",    0x46, 0x44, ""),
]

for name, cid1, cid2, info in commands:
    print(f"\n=== {name} (CID={cid1:02X} {cid2:02X}) ===")
    send_cmd(ser, cid1, cid2, info)
    time.sleep(0.15)

ser.close()