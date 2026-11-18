#!/usr/bin/env python3
"""vive_calib/receiver.py —— Jetson（机器人）侧 UDP 接收器，纯标准库。

职责：收 UDP 位姿 → 按「设备名 + 序号」去重/丢弃乱序 → 缓存每台设备最新位姿，
供后续采集主循环读。

用法：
    python3 receiver.py [--bind 0.0.0.0] [--port 12345] [--verbose]

消息格式（与 sender.c 的 pose_msg_t 字节对齐，小端）：
    "<16s d I 7f"  =>  dev(16) ts(double) seq(uint32) pos(3f) quat(4f)  共 56 字节
"""
import argparse
import socket
import struct
import time

MSG_FORMAT = "<16s d I 7f"
MSG_SIZE = struct.calcsize(MSG_FORMAT)  # 56

# seq 大幅回退视为 sender 重启（>2^30），重置该设备的序号水位
_SEQ_RESET_THRESHOLD = 1 << 30


class PoseReceiver:
    def __init__(self, bind_ip="0.0.0.0", port=12345):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_ip, port))
        self.last_seq = {}   # dev -> 已见最大序号
        self.latest = {}     # dev -> (ts, seq, pos, quat)

    def poll(self, timeout=None):
        """收一包。返回 (dev, ts, seq, pos, quat)；超时/坏包/乱序返回 None。"""
        self.sock.settimeout(timeout)
        try:
            data, _ = self.sock.recvfrom(MSG_SIZE * 2)
        except socket.timeout:
            return None

        if len(data) != MSG_SIZE:
            return None  # 长度不对，丢弃

        dev_b, ts, seq, px, py, pz, qx, qy, qz, qw = struct.unpack(MSG_FORMAT, data)
        dev = dev_b.split(b"\0", 1)[0].decode("ascii", "replace")

        last = self.last_seq.get(dev, -1)
        if seq <= last:
            if seq < last - _SEQ_RESET_THRESHOLD:
                self.last_seq[dev] = -1  # sender 重启，序号回零，重置
            else:
                return None  # 乱序或重复，丢弃

        self.last_seq[dev] = seq
        pos, quat = (px, py, pz), (qx, qy, qz, qw)
        self.latest[dev] = (ts, seq, pos, quat)
        return (dev, ts, seq, pos, quat)

    def get_latest(self):
        """当前每台设备的最新位姿 {dev: (ts, seq, pos, quat)}。"""
        return dict(self.latest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="跑 N 秒后退出（0=一直跑）")
    ap.add_argument("--verbose", action="store_true",
                    help="每包都打印（默认每 50 包打印一次）")
    args = ap.parse_args()

    rx = PoseReceiver(args.bind, args.port)
    print(f"listening on {args.bind}:{args.port}, msg={MSG_SIZE}B")
    t0 = time.time()
    n = 0
    while True:
        r = rx.poll(timeout=1.0)
        if r is not None:
            dev, ts, seq, pos, quat = r
            n += 1
            if args.verbose or n % 50 == 1:
                print(f"[{dev}] seq={seq} t={ts:.3f} "
                      f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                      f"quat=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})")
        if args.duration and time.time() - t0 >= args.duration:
            break

    print(f"received {n} valid packets; devices={sorted(rx.last_seq)}")


if __name__ == "__main__":
    main()
