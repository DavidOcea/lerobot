#!/usr/bin/env python3
"""
Navigation & Control API Deep Scanner for Seer AGV.

The previous scan only sent empty {} requests, which returned "error api type"
for navigation/control APIs. This scanner sends real parameters to discover
the correct API codes and request formats.

Strategy:
1. Scan wider API range on ports 19205-19207
2. Send multiple request payload variants for each API code
3. Use real station names from the AGV (LM8, etc.)
"""

import json
import socket
import struct
import sys
import time
from pathlib import Path

LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

# ========== TCP Protocol ==========
SYNC_BYTE = 0x5A
VERSION_BYTE = 0x01


def build_packet(api_type: int, seq_num: int, data: dict = None) -> bytes:
    json_str = json.dumps(data or {}, ensure_ascii=False)
    json_bytes = json_str.encode('utf-8')
    data_len = len(json_bytes)
    header = bytes([SYNC_BYTE, VERSION_BYTE])
    header += struct.pack('<H', seq_num)
    header += struct.pack('>I', data_len)
    header += struct.pack('>H', api_type)
    header += b'\x00' * 6
    return header + json_bytes


def recv_exact(sock: socket.socket, n: int, timeout: float = 3.0) -> bytes:
    data = b''
    sock.settimeout(timeout)
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 4096))
        if not chunk:
            raise ConnectionError(f"Connection closed, got {len(data)}/{n} bytes")
        data += chunk
    return data


def send_and_recv(sock: socket.socket, api_type: int, seq_num: int,
                  data: dict = None, timeout: float = 3.0) -> dict | None:
    """Send a request and receive the response."""
    try:
        packet = build_packet(api_type, seq_num, data or {})
        sock.sendall(packet)

        header_bytes = recv_exact(sock, 16, timeout=timeout)
        sync = header_bytes[0]
        if sync != SYNC_BYTE:
            return None

        data_len = struct.unpack('>I', header_bytes[4:8])[0]
        api_type_resp = struct.unpack('>H', header_bytes[8:10])[0]

        if data_len > 0:
            payload_bytes = recv_exact(sock, data_len, timeout=timeout)
            try:
                payload = json.loads(payload_bytes.decode('utf-8'))
            except json.JSONDecodeError:
                return {"api_resp": api_type_resp, "raw": payload_bytes.decode('utf-8', errors='replace')[:300]}
        else:
            payload = {}

        return {"api_resp": api_type_resp, "payload": payload}

    except socket.timeout:
        return None
    except (ConnectionError, ConnectionResetError):
        # Reconnect needed
        return {"error": "connection_reset"}
    except Exception as e:
        return None


def reconnect(host: str, port: int) -> socket.socket:
    """Create a fresh socket connection."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((host, port))
    return sock


# ========== Request payload variants ==========
# Different possible parameter names the AGV might expect
STATION_PAYLOADS = [
    {"station_id": "LM8"},
    {"location_id": "LM8"},
    {"target_station": "LM8"},
    {"target_location": "LM8"},
    {"name": "LM8"},
    {"id": "LM8"},
    {"dest": "LM8"},
    {"goal": "LM8"},
    {"target_id": "LM8"},
    {"station": "LM8"},
    {"location": "LM8"},
    {"point": "LM8"},
    {"pose_id": "LM8"},
    {"task_location": "LM8"},
    {"end_station": "LM8"},
]

POSITION_PAYLOADS = [
    {"x": 0.0, "y": 0.0, "theta": 0.0},
    {"x": 0.0, "y": 0.0},
    {"position": {"x": 0.0, "y": 0.0, "theta": 0.0}},
    {"target_x": 0.0, "target_y": 0.0},
    {"goal_x": 0.0, "goal_y": 0.0},
    {"dest_x": 0.0, "dest_y": 0.0},
]

CONTROL_PAYLOADS = [
    {},  # Some control APIs accept empty request (like stop)
    {"enable": True},
    {"enable": False},
    {"state": 1},
    {"state": 0},
    {"command": "stop"},
    {"command": "pause"},
    {"command": "resume"},
    {"action": "stop"},
    {"action": "pause"},
    {"action": "resume"},
    {"speed": 0.5},
    {"max_speed": 0.5},
    {"velocity": 0.5},
]


def scan_port_with_payloads(host: str, port: int, port_name: str,
                           api_range: tuple, payloads: list, seq_start: int = 0):
    """Scan API codes on a port using multiple payload variants."""
    print(f"\n{'=' * 70}")
    print(f"Port {port} ({port_name}): Scanning {api_range[0]:#06x} - {api_range[1]:#06x}")
    print(f"Payloads: {len(payloads)} variants")
    print(f"{'=' * 70}")

    sock = reconnect(host, port)
    seq_num = seq_start
    results = {}

    for api_code in range(api_range[0], api_range[1] + 1):
        best_result = None
        best_payload_idx = -1

        for p_idx, payload in enumerate(payloads):
            seq_num = (seq_num + 1) % 65536
            result = send_and_recv(sock, api_code, seq_num, payload)

            if result is None:
                # Connection probably broken, reconnect
                try:
                    sock.close()
                except:
                    pass
                sock = reconnect(host, port)
                continue

            if result.get("error") == "connection_reset":
                sock = reconnect(host, port)
                continue

            resp_payload = result.get("payload", {})
            ret_code = resp_payload.get("ret_code", None)

            # Check if this is a meaningful response (not just "error api type")
            err_msg = resp_payload.get("err_msg", "")

            # A successful or parameter-error response is more useful than "error api type"
            if ret_code is not None and "error api type" not in str(err_msg).lower():
                best_result = {
                    "api_code": api_code,
                    "api_resp": result.get("api_resp"),
                    "payload_sent": payload,
                    "payload_idx": p_idx,
                    "ret_code": ret_code,
                    "err_msg": err_msg,
                    "response_fields": list(resp_payload.keys()),
                    "response_data": resp_payload,
                }
                best_payload_idx = p_idx
                break  # Found a working payload, no need to try more

            time.sleep(0.02)

        if best_result is not None:
            results[api_code] = best_result
            ret_code = best_result["ret_code"]
            err_msg = best_result["err_msg"]
            payload_keys = list(best_result["payload_sent"].keys())
            resp_fields = best_result["response_fields"][:10]
            status = "OK" if ret_code == 0 else f"ret={ret_code}"
            print(f"  {api_code:#06x}: {status} | payload={payload_keys} | resp_fields={resp_fields}")
            if err_msg and ret_code != 0:
                print(f"    err_msg: {err_msg}")
        elif result is not None and result.get("payload"):
            # Only got "error api type" - still record it
            resp = result.get("payload", {})
            err_msg = resp.get("err_msg", "")
            if "error api type" in str(err_msg).lower():
                # API exists but wrong type code - record for reference
                results[api_code] = {
                    "api_code": api_code,
                    "api_resp": result.get("api_resp"),
                    "payload_sent": {},
                    "ret_code": resp.get("ret_code"),
                    "err_msg": err_msg,
                    "response_fields": list(resp.keys()),
                    "note": "API exists but rejected request - likely different API category",
                }

    sock.close()
    return results


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.2.210"

    print("=" * 70)
    print("Seer AGV Navigation & Control API Deep Scanner")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Strategy: Send real parameters to find correct API codes")
    print()

    all_results = {}

    # 1. Scan 19205 (Control) with control payloads
    print("\n[Phase 1] Control port 19205")
    ctrl_results = scan_port_with_payloads(
        host, 19205, "Control",
        (0x07D0, 0x07FF),  # Wider range: 2000-2047
        CONTROL_PAYLOADS,
        seq_start=0,
    )
    for code, r in ctrl_results.items():
        all_results[f"19205:{code:#06x}"] = r

    # 2. Scan 19206 (Navigation) with station payloads
    print("\n[Phase 2] Navigation port 19206 - Station navigation payloads")
    nav_station_results = scan_port_with_payloads(
        host, 19206, "Navigation (station)",
        (0x07D0, 0x07FF),  # Wider range: 2000-2047
        STATION_PAYLOADS,
        seq_start=10000,
    )
    for code, r in nav_station_results.items():
        all_results[f"19206:{code:#06x}_station"] = r

    # 3. Scan 19206 (Navigation) with position payloads
    print("\n[Phase 3] Navigation port 19206 - Position navigation payloads")
    nav_pos_results = scan_port_with_payloads(
        host, 19206, "Navigation (position)",
        (0x07D0, 0x07FF),
        POSITION_PAYLOADS,
        seq_start=20000,
    )
    for code, r in nav_pos_results.items():
        all_results[f"19206:{code:#06x}_position"] = r

    # 4. Scan 19207 (Task management) with station payloads
    print("\n[Phase 4] Task management port 19207")
    task_results = scan_port_with_payloads(
        host, 19207, "Task Management",
        (0x07D0, 0x07FF),
        STATION_PAYLOADS + CONTROL_PAYLOADS + POSITION_PAYLOADS,
        seq_start=30000,
    )
    for code, r in task_results.items():
        all_results[f"19207:{code:#06x}"] = r

    # 5. Also try the status query port range with broader scan
    #    The API_NAVIGATE_STATION (0x07E8) might actually belong to 19204
    print("\n[Phase 5] Extended status query port 19204 (2000+ range)")
    sock = reconnect(host, 19204)
    seq_num = 40000
    status_extended = {}
    for api_code in range(0x07D0, 0x07FF + 1):
        seq_num = (seq_num + 1) % 65536
        result = send_and_recv(sock, api_code, seq_num, {"station_id": "LM8"})
        if result is not None and result.get("payload"):
            resp = result.get("payload", {})
            ret_code = resp.get("ret_code")
            err_msg = resp.get("err_msg", "")
            if ret_code is not None and "error api type" not in str(err_msg).lower():
                status_extended[api_code] = {
                    "api_code": api_code,
                    "api_resp": result.get("api_resp"),
                    "payload_sent": {"station_id": "LM8"},
                    "ret_code": ret_code,
                    "err_msg": err_msg,
                    "response_fields": list(resp.keys()),
                    "response_data": resp,
                }
                print(f"  {api_code:#06x}: ret={ret_code} err={err_msg[:50]} fields={list(resp.keys())[:10]}")
        time.sleep(0.02)
    sock.close()
    for code, r in status_extended.items():
        all_results[f"19204:{code:#06x}"] = r

    # ========== Summary ==========
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # Filter: only show results that are NOT "error api type"
    meaningful_results = {}
    for key, r in all_results.items():
        err_msg = str(r.get("err_msg", ""))
        if "error api type" not in err_msg.lower():
            meaningful_results[key] = r

    print(f"\nTotal API codes tested: {len(all_results)}")
    print(f"Meanful responses (not 'error api type'): {len(meaningful_results)}")

    if meaningful_results:
        print(f"\n{'─' * 70}")
        print("MEANINGFUL RESULTS (successful or parameter-error):")
        print(f"{'─' * 70}")
        for key, r in sorted(meaningful_results.items()):
            api_code = r.get("api_code", 0)
            ret_code = r.get("ret_code")
            err_msg = r.get("err_msg", "")
            payload_sent = r.get("payload_sent", {})
            fields = r.get("response_fields", [])
            print(f"\n  {key}:")
            print(f"    API: {api_code:#06x} ({api_code})")
            print(f"    Payload sent: {payload_sent}")
            print(f"    ret_code: {ret_code}")
            print(f"    err_msg: {err_msg[:80]}")
            print(f"    Fields: {fields[:15]}{'...' if len(fields) > 15 else ''}")
            if ret_code == 0:
                print(f"    *** SUCCESS! ***")

    # ========== Save results ==========
    output_path = "/root/workspace/dc_dir/nav_api_scan_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()