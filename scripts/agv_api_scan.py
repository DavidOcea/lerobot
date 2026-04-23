#!/usr/bin/env python3
"""
Seer AGV TCP API Scanner - Systematic discovery of all available APIs.

Scans API code ranges across all ports (19204-19207) to discover:
- Valid API endpoints and their response field names
- Velocity/speed query APIs (vx, vy, vtheta) - currently unknown
- Position, station, status APIs - verify and supplement existing knowledge
- Any other useful APIs for AGV control

Usage:
    # Full scan (all ranges, all ports)
    python scripts/agv_api_scan.py --host 192.168.2.210

    # Quick scan (only status query port, focused range)
    python scripts/agv_api_scan.py --host 192.168.2.210 --quick

    # Scan specific range on specific port
    python scripts/agv_api_scan.py --host 192.168.2.210 --port 19204 --start 0x03E8 --end 0x0460

    # Search for velocity API specifically
    python scripts/agv_api_scan.py --host 192.168.2.210 --search velocity

    # Save results to file
    python scripts/agv_api_scan.py --host 192.168.2.210 --output /root/workspace/dc_dir/api_scan_results.json
"""

import argparse
import json
import logging
import socket
import struct
import sys
import time
from pathlib import Path

LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noisy logs during scanning
logging.getLogger("lerobot.robots.agv").setLevel(logging.ERROR)

# Known API ranges by port
PORT_RANGES = {
    19204: {  # Status query port
        "name": "Status Query",
        "full_range": (0x03E8, 0x0460),   # 1000-1120
        "quick_range": (0x03E8, 0x0410),   # 1000-1040
        "step": 2,  # APIs seem to increment by 2
    },
    19205: {  # Control port
        "name": "Control",
        "full_range": (0x07D2, 0x07E0),   # 2002-2016
        "quick_range": (0x07D2, 0x07DA),   # 2002-2010
        "step": 1,
    },
    19206: {  # Navigation port
        "name": "Navigation",
        "full_range": (0x07E8, 0x07F0),   # 2024-2032
        "quick_range": (0x07E8, 0x07EC),   # 2024-2028
        "step": 1,
    },
    19207: {  # Task management port
        "name": "Task Management",
        "full_range": (0x07F0, 0x07FA),   # 2032-2042
        "quick_range": (0x07F0, 0x07F4),   # 2032-2036
        "step": 1,
    },
}

# Known valid APIs (for verification and reference)
KNOWN_APIS = {
    0x03E8: "综合状态查询 (系统版本/地图名)",
    0x03EA: "电量查询 (battery_level)",
    0x03EC: "任务状态查询 ✅ (x, y, angle, current_station)",
    0x03F4: "EMC急停状态 (emergency/soft_emc) - 原误标为速度",
    0x044E: "电池/充电状态 - 原误标为站点",
    0x07D2: "急停控制",
    0x07D3: "暂停控制",
    0x07D4: "继续控制",
    0x07D5: "取消任务控制",
    0x07D6: "设置速度控制",
    0x07E8: "导航到站点",
    0x07E9: "导航到坐标",
    0x07EB: "取消导航",
}


def build_packet(api_type: int, seq_num: int, data: dict = None) -> bytes:
    """Build a Seer AGV TCP request packet."""
    json_str = json.dumps(data or {}, ensure_ascii=False)
    json_bytes = json_str.encode('utf-8')
    data_len = len(json_bytes)

    header = bytes([0x5A, 0x01])
    header += struct.pack('<H', seq_num)
    header += struct.pack('>I', data_len)
    header += struct.pack('>H', api_type)
    header += b'\x00' * 6

    return header + json_bytes


def recv_exact(sock: socket.socket, n: int, timeout: float = 2.0) -> bytes:
    """Receive exactly n bytes from socket."""
    data = b''
    sock.settimeout(timeout)
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 4096))
        if not chunk:
            raise ConnectionError(f"Connection closed, got {len(data)}/{n} bytes")
        data += chunk
    return data


def scan_api(sock: socket.socket, api_type: int, seq_num: int) -> dict | None:
    """Scan a single API code and return response if valid."""
    try:
        packet = build_packet(api_type, seq_num, {})
        sock.sendall(packet)

        # Receive header
        header_bytes = recv_exact(sock, 16, timeout=3.0)
        if len(header_bytes) != 16:
            return None

        sync = header_bytes[0]
        if sync != 0x5A:
            return None

        data_len = struct.unpack('>I', header_bytes[4:8])[0]
        api_type_resp = struct.unpack('>H', header_bytes[8:10])[0]

        # Receive payload
        if data_len > 0:
            payload_bytes = recv_exact(sock, data_len, timeout=3.0)
            try:
                payload = json.loads(payload_bytes.decode('utf-8'))
            except json.JSONDecodeError:
                return {"api": api_type, "api_resp": api_type_resp, "raw_payload": payload_bytes.decode('utf-8', errors='replace')[:200]}
        else:
            payload = {}

        return {"api": api_type, "api_resp": api_type_resp, "fields": list(payload.keys()), "data": payload}

    except socket.timeout:
        return None
    except ConnectionError:
        return None
    except Exception as e:
        return None


def search_for_velocity_fields(results: dict) -> list:
    """Find APIs that might contain velocity data."""
    velocity_candidates = []
    velocity_keywords = ['vx', 'vy', 'vtheta', 'velocity', 'speed', 'vel_x', 'vel_y', 'vel_theta',
                         'v_x', 'v_y', 'v_angle', 'linear_speed', 'angular_speed',
                         'odom_x', 'odom_y', 'odom_vx', 'odom_vy']

    for api_code, result in results.items():
        if result is None:
            continue
        fields = result.get('fields', [])
        for kw in velocity_keywords:
            if any(kw in f.lower() for f in fields):
                velocity_candidates.append({
                    "api": api_code,
                    "matched_keyword": kw,
                    "fields": fields,
                    "data": result.get('data', {}),
                })
                break

    return velocity_candidates


def main():
    parser = argparse.ArgumentParser(description="Seer AGV TCP API Scanner")

    parser.add_argument("--host", type=str, required=True, help="AGV IP address")
    parser.add_argument("--port", type=int, default=None, help="Scan specific port only")
    parser.add_argument("--start", type=int, default=None, help="Start API code (hex or decimal, e.g. 0x03E8 or 1000)")
    parser.add_argument("--end", type=int, default=None, help="End API code (hex or decimal)")
    parser.add_argument("--step", type=int, default=None, help="API code increment step")
    parser.add_argument("--quick", action="store_true", help="Quick scan - only scan known nearby ranges")
    parser.add_argument("--search", type=str, default=None, help="Search for specific keyword in response fields (e.g. 'velocity', 'station')")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show all responses including empty ones")
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout per request")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between API probes (seconds)")

    args = parser.parse_args()

    # Determine scan configuration
    ports_to_scan = {}
    if args.port:
        if args.port not in PORT_RANGES:
            print(f"Unknown port {args.port}. Known ports: {list(PORT_RANGES.keys())}")
            sys.exit(1)
        ports_to_scan[args.port] = PORT_RANGES[args.port]
    else:
        ports_to_scan = PORT_RANGES

    # Determine API range
    if args.start is not None and args.end is not None:
        for port, config in ports_to_scan.items():
            config["scan_range"] = (args.start, args.end)
            config["scan_step"] = args.step or config["step"]
    elif args.quick:
        for port, config in ports_to_scan.items():
            config["scan_range"] = config["quick_range"]
            config["scan_step"] = config["step"]
    else:
        for port, config in ports_to_scan.items():
            config["scan_range"] = config["full_range"]
            config["scan_step"] = config["step"]

    print("=" * 70)
    print("Seer AGV TCP API Scanner")
    print("=" * 70)
    print(f"Host: {args.host}")
    print(f"Mode: {'Quick' if args.quick else 'Full'} scan")
    print()

    all_results = {}
    seq_num = 0

    for port, config in ports_to_scan.items():
        port_name = config["name"]
        start, end = config.get("scan_range", config["full_range"])
        step = config.get("scan_step", config["step"])

        print(f"\n{'─' * 70}")
        print(f"Port {port} ({port_name}): Scanning API {start:#06x} - {end:#06x} (step={step})")
        print(f"{'─' * 70}")

        # Connect to this port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((args.host, port))
            print(f"  Connected to {args.host}:{port}")
        except Exception as e:
            print(f"  ✗ Cannot connect to port {port}: {e}")
            continue

        found_count = 0
        port_results = {}

        api_codes = range(start, end + 1, step)
        total = len(list(api_codes))
        api_codes = range(start, end + 1, step)  # Re-create iterator

        for i, api_code in enumerate(api_codes):
            seq_num = (seq_num + 1) % 65536
            result = scan_api(sock, api_code, seq_num)

            if result is not None:
                found_count += 1
                api_key = f"{port}:{api_code:#06x}"
                all_results[api_key] = result
                port_results[api_code] = result

                known_desc = KNOWN_APIS.get(api_code, "")
                fields = result.get('fields', [])
                fields_preview = fields[:10] if len(fields) > 10 else fields

                status_icon = "✓" if known_desc else "?"
                print(f"  {status_icon} API {api_code:#06x} ({api_code}): {fields_preview}")
                if known_desc:
                    print(f"    Known: {known_desc}")
                if len(fields) > 10:
                    print(f"    ... ({len(fields)} total fields)")
            elif args.verbose:
                print(f"  · API {api_code:#06x}: no response")

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  [Progress: {i+1}/{total}, Found: {found_count}]")

            time.sleep(args.delay)

        sock.close()

        print(f"\n  Port {port} summary: {found_count} valid APIs found out of {total} scanned")

    # ========== Analysis ==========
    print(f"\n{'=' * 70}")
    print("SCAN RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # Group by port
    for port, config in PORT_RANGES.items():
        port_results = {k: v for k, v in all_results.items() if k.startswith(f"{port}:")}
        if not port_results:
            continue

        print(f"\nPort {port} ({config['name']}):")
        for api_key, result in sorted(port_results.items()):
            api_code = int(api_key.split(":")[1], 16)
            fields = result.get('fields', [])
            known_desc = KNOWN_APIS.get(api_code, "NEW!")
            print(f"  {api_code:#06x} ({api_code}): {fields[:8]}{'...' if len(fields)>8 else ''} [{known_desc}]")

    # ========== Search for specific keywords ==========
    search_kw = args.search or "velocity"
    print(f"\n{'─' * 70}")
    print(f"Searching for '{search_kw}' related fields...")
    print(f"{'─' * 70}")

    velocity_candidates = []
    for api_key, result in all_results.items():
        if result is None:
            continue
        fields = result.get('fields', [])
        for f in fields:
            if search_kw.lower() in f.lower():
                velocity_candidates.append({
                    "api_key": api_key,
                    "matched_field": f,
                    "all_fields": fields,
                    "sample_data": {f: result['data'].get(f) for f in fields[:5]},
                })

    if velocity_candidates:
        print(f"  Found {len(velocity_candidates)} matches:")
        for c in velocity_candidates:
            print(f"  ✓ {c['api_key']}: field '{c['matched_field']}' in {c['all_fields'][:8]}")
            if c['sample_data']:
                print(f"    Sample: {c['sample_data']}")
    else:
        print(f"  No matches found for '{search_kw}'")

    # Also search specifically for velocity patterns
    if search_kw == "velocity" or not args.search:
        print(f"\n  Extended velocity field search:")
        vel_candidates = search_for_velocity_fields(
            {int(k.split(":")[1], 16): v for k, v in all_results.items()}
        )
        if vel_candidates:
            print(f"  Found {len(vel_candidates)} velocity-related APIs:")
            for c in vel_candidates:
                print(f"  ✓ API {c['api']:#06x}: keyword '{c['matched_keyword']}' found in {c['fields']}")
                print(f"    Data: {c['data']}")
        else:
            print("  ✗ No velocity-related APIs found in scanned range")
            print("  → Consider expanding scan range or checking API documentation")

    # ========== Save results ==========
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert api_key format for JSON
        json_results = {}
        for api_key, result in all_results.items():
            if result:
                json_results[api_key] = {
                    "api_code": result.get("api"),
                    "api_resp": result.get("api_resp"),
                    "fields": result.get("fields", []),
                    "data_preview": {k: v for k, v in result.get("data", {}).items() if k in (result.get("fields", [])[:20])},
                }

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {output_path}")

    # ========== Unknown API identification ==========
    new_apis = []
    for api_key, result in all_results.items():
        api_code = int(api_key.split(":")[1], 16)
        if api_code not in KNOWN_APIS and result is not None:
            new_apis.append((api_key, result))

    if new_apis:
        print(f"\n{'─' * 70}")
        print(f"NEW APIs (not in known list): {len(new_apis)}")
        print(f"{'─' * 70}")
        for api_key, result in new_apis:
            fields = result.get('fields', [])
            print(f"  {api_key}: {fields[:10]}{'...' if len(fields)>10 else ''}")
            if result.get('data'):
                # Show a preview of the data
                preview_keys = fields[:5]
                preview = {k: result['data'].get(k) for k in preview_keys}
                print(f"    Preview: {preview}")

    print(f"\n{'=' * 70}")
    print(f"Total APIs found: {len(all_results)}")
    print(f"Known APIs verified: {len([k for k in all_results if int(k.split(':')[1], 16) in KNOWN_APIS])}")
    print(f"New APIs discovered: {len(new_apis)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()