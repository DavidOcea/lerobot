#!/usr/bin/env python3
"""
Test script for Seer AGV TCP Controller.

This script tests basic AGV connectivity and functionality:
- TCP connection
- Status query
- Battery level
- Position query
- Navigation commands (optional)

Usage:
    # Basic connectivity test (safe, no movement)
    python test_agv_controller.py --host 192.168.1.100

    # Full test including navigation (requires AGV to be available)
    python test_agv_controller.py --host 192.168.1.100 --test-navigation --target-station station_B

    # Debug mode
    python test_agv_controller.py --host 192.168.1.100 --debug
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports using relative path
LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.robots.agv.seer_agv_controller import SeerAGVController, AGVStatus


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_connection(controller: SeerAGVController) -> bool:
    """Test TCP connection to AGV."""
    print("\n" + "=" * 50)
    print("TEST 1: TCP Connection")
    print("=" * 50)

    try:
        success = controller.connect()
        if success:
            print("✓ Connection successful")
            print(f"  Connected ports: {list(controller._sockets.keys())}")
            return True
        else:
            print("✗ Connection failed")
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False


def test_status_query(controller: SeerAGVController) -> bool:
    """Test status query."""
    print("\n" + "=" * 50)
    print("TEST 2: Status Query")
    print("=" * 50)

    try:
        status = controller.get_status(use_cache=False)

        print(f"✓ Status retrieved:")
        print(f"  Battery: {status.battery}%")
        print(f"  Status code: {status.status_code} ({_status_code_to_str(status.status_code)})")
        print(f"  Current station: {status.current_station}")
        print(f"  Position: x={status.position.x:.3f}, y={status.position.y:.3f}, θ={status.position.theta:.3f}")
        print(f"  Is moving: {status.is_moving}")
        print(f"  Error code: {status.error_code}")

        if status.error_code != 0:
            print(f"  Error message: {status.error_message}")

        return True

    except Exception as e:
        print(f"✗ Status query failed: {e}")
        return False


def test_battery_query(controller: SeerAGVController) -> bool:
    """Test battery level query."""
    print("\n" + "=" * 50)
    print("TEST 3: Battery Level")
    print("=" * 50)

    try:
        battery = controller.get_battery()
        print(f"✓ Battery level: {battery}%")

        if battery < 20:
            print("  ⚠ Warning: Low battery!")
        elif battery < 50:
            print("  ⚠ Notice: Battery below 50%")
        else:
            print("  ✓ Battery OK")

        return True

    except Exception as e:
        print(f"✗ Battery query failed: {e}")
        return False


def test_position_query(controller: SeerAGVController) -> bool:
    """Test position query."""
    print("\n" + "=" * 50)
    print("TEST 4: Position Query")
    print("=" * 50)

    try:
        position = controller.get_position()
        print(f"✓ Current position:")
        print(f"  X: {position.x:.3f} m")
        print(f"  Y: {position.y:.3f} m")
        print(f"  Theta: {position.theta:.3f} rad ({position.theta * 180 / 3.14159:.1f}°)")

        return True

    except Exception as e:
        print(f"✗ Position query failed: {e}")
        return False


def test_velocity_query(controller: SeerAGVController) -> bool:
    """Test velocity query."""
    print("\n" + "=" * 50)
    print("TEST 5: Velocity Query")
    print("=" * 50)

    try:
        vx, vy, vtheta = controller.get_velocity()
        print(f"✓ Current velocity:")
        print(f"  Vx: {vx:.3f} m/s")
        print(f"  Vy: {vy:.3f} m/s")
        print(f"  Vtheta: {vtheta:.3f} rad/s")

        if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vtheta) > 0.01:
            print("  → AGV is moving")
        else:
            print("  → AGV is stationary")

        return True

    except Exception as e:
        print(f"✗ Velocity query failed: {e}")
        return False


def test_navigation(controller: SeerAGVController, target_station: str, wait: bool = True) -> bool:
    """Test navigation to a station.

    WARNING: This test will move the AGV!
    """
    print("\n" + "=" * 50)
    print("TEST 6: Navigation (WARNING: AGV WILL MOVE)")
    print("=" * 50)

    print(f"Target station: {target_station}")
    print("Starting navigation in 3 seconds...")
    print("Press Ctrl+C to cancel")
    time.sleep(3)

    try:
        # Start navigation
        print(f"\n→ Sending navigation command...")
        success = controller.move_to_station(target_station)

        if not success:
            print("✗ Navigation command failed")
            return False

        print("✓ Navigation command sent successfully")

        if not wait:
            print("  (Not waiting for arrival - check AGV status manually)")
            return True

        # Wait for arrival
        print(f"\n→ Waiting for arrival (timeout: 60s)...")

        # Poll status
        start_time = time.time()
        while time.time() - start_time < 60:
            status = controller.get_status(use_cache=False)
            print(f"  Station: {status.current_station}, Moving: {status.is_moving}")

            if status.current_station == target_station:
                print(f"\n✓ Arrived at {target_station}")
                return True

            if not status.is_moving and time.time() - start_time > 10:
                # AGV stopped but not at target
                print(f"\n⚠ AGV stopped but not at target station")
                print(f"  Current station: {status.current_station}")
                return False

            time.sleep(2)

        print(f"\n✗ Timeout - did not arrive at {target_station}")
        return False

    except KeyboardInterrupt:
        print("\n\n⚠ Navigation cancelled by user")
        controller.stop()
        return False

    except Exception as e:
        print(f"\n✗ Navigation failed: {e}")
        controller.stop()
        return False


def test_emergency_stop(controller: SeerAGVController) -> bool:
    """Test emergency stop (only if AGV is moving)."""
    print("\n" + "=" * 50)
    print("TEST 7: Emergency Stop")
    print("=" * 50)

    try:
        success = controller.stop()
        if success:
            print("✓ Emergency stop executed")
            return True
        else:
            print("✗ Emergency stop failed")
            return False

    except Exception as e:
        print(f"✗ Emergency stop error: {e}")
        return False


def _status_code_to_str(code: int) -> str:
    """Convert status code to human-readable string."""
    codes = {
        0: "IDLE",
        1: "EXECUTING_TASK",
        2: "CHARGING",
        3: "ERROR",
        4: "PAUSED",
    }
    return codes.get(code, f"UNKNOWN({code})")


def main():
    parser = argparse.ArgumentParser(description="Test Seer AGV TCP Controller")

    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="AGV IP address",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=19204,
        help="AGV port (default: 19204)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--test-navigation",
        action="store_true",
        help="Test navigation (WARNING: will move AGV)",
    )

    parser.add_argument(
        "--target-station",
        type=str,
        default="station_B",
        help="Target station for navigation test",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for arrival during navigation test",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    print("=" * 50)
    print("Seer AGV TCP Controller Test")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("=" * 50)

    # Create controller
    controller = SeerAGVController(
        host=args.host,
        port=args.port,
        connection_timeout=5.0,
        read_timeout=2.0,
    )

    # Run tests
    results = []

    # Test 1: Connection
    results.append(("Connection", test_connection(controller)))

    if not results[0][1]:
        print("\n✗ Cannot proceed - connection failed")
        print("Please check:")
        print("  - AGV IP address is correct")
        print("  - AGV is powered on")
        print("  - Network connection is available")
        sys.exit(1)

    # Test 2-5: Queries (safe tests)
    results.append(("Status Query", test_status_query(controller)))
    results.append(("Battery Query", test_battery_query(controller)))
    results.append(("Position Query", test_position_query(controller)))
    results.append(("Velocity Query", test_velocity_query(controller)))

    # Test 6: Navigation (optional, dangerous)
    if args.test_navigation:
        print("\n⚠ WARNING: Navigation test will move the AGV!")
        print("Make sure:")
        print("  - AGV path is clear")
        print("  - No obstacles on the route")
        print("  - You can observe AGV movement")
        print("\nPress Ctrl+C at any time to emergency stop")

        results.append(("Navigation", test_navigation(
            controller,
            args.target_station,
            wait=not args.no_wait
        )))

    # Disconnect
    print("\n" + "=" * 50)
    print("Disconnecting...")
    controller.disconnect()
    print("✓ Disconnected")

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for name, result in results:
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}")

    print("=" * 50)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed == 0:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()