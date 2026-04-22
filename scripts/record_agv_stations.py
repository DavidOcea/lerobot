#!/usr/bin/env python3
"""
AGV Map Station Coordinate Recorder Tool.

Record station coordinates during map building and generate YAML config.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Use relative path based on script location for portability
LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.robots.agv.seer_agv_controller import SeerAGVController

STATION_DATA_FILE = LEROBOT_ROOT / "configs" / "agv_stations.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_stations():
    if STATION_DATA_FILE.exists():
        with open(STATION_DATA_FILE) as f:
            return json.load(f)
    return {}


def save_stations(stations):
    STATION_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATION_DATA_FILE, 'w') as f:
        json.dump(stations, f, indent=2)
    logger.info(f"Saved {len(stations)} stations")


def record_station(controller, station_id, description=""):
    position = controller.get_position()
    current_station = controller.get_current_station()
    battery = controller.get_battery()

    station_data = {
        "id": station_id,
        "x": position.x,
        "y": position.y,
        "theta": position.theta,
        "description": description,
        "current_station_name": current_station,
        "battery_at_record": battery,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info(f"Recorded station '{station_id}':")
    logger.info(f"  Position: x={position.x:.3f}, y={position.y:.3f}, theta={position.theta:.3f}")
    return station_data


def interactive_record(controller):
    stations = load_stations()

    print("\n" + "=" * 60)
    print("AGV Station Coordinate Recorder - Interactive Mode")
    print("=" * 60)
    print("Drive AGV to position, then enter station ID to record.")

    while True:
        try:
            print("\nCurrent AGV Status:")
            position = controller.get_position()
            current_station = controller.get_current_station()
            print(f"  Position: ({position.x:.3f}, {position.y:.3f}) theta={position.theta:.3f}")
            print(f"  Current station: {current_station}")

            station_id = input("\nEnter station ID (or 'q' to quit): ").strip()
            if station_id.lower() == 'q':
                break
            if not station_id:
                continue

            if station_id in stations:
                print(f"  Station '{station_id}' exists. Overwrite? (y/n): ", end="")
                if input().strip().lower() != 'y':
                    continue

            description = input("Enter description (optional): ").strip()
            station_data = record_station(controller, station_id, description)
            stations[station_id] = station_data
            save_stations(stations)
            print(f"  Station '{station_id}' recorded")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def list_stations(controller=None):
    stations = load_stations()

    print("\n" + "=" * 60)
    print("Recorded AGV Stations")
    print("=" * 60)

    if not stations:
        print("No stations recorded.")
        return

    print(f"\nTotal: {len(stations)} stations\n")

    for station_id, data in stations.items():
        print(f"{station_id}:")
        print(f"  Position: x={data['x']:.3f}, y={data['y']:.3f}, theta={data['theta']:.3f}")
        if data.get('description'):
            print(f"  Description: {data['description']}")
        print()

    if controller and controller.is_connected():
        position = controller.get_position()
        print(f"\nCurrent AGV: ({position.x:.3f}, {position.y:.3f})")


def generate_yaml_config():
    stations = load_stations()

    if not stations:
        print("No stations recorded.")
        return

    print("\n" + "=" * 60)
    print("Generated YAML Configuration")
    print("=" * 60)
    print("\nagv_config:")
    print("  station_map:")
    for station_id, data in stations.items():
        print(f"    {station_id}: [{data['x']:.3f}, {data['y']:.3f}, {data['theta']:.3f}]")

    yaml_path = STATION_DATA_FILE.parent / 'agv_station_map.yaml'
    with open(yaml_path, 'w') as f:
        f.write("# AGV Station Map\n\nagv_config:\n  station_map:\n")
        for station_id, data in stations.items():
            f.write(f"    {station_id}: [{data['x']:.3f}, {data['y']:.3f}, {data['theta']:.3f}]\n")
    print(f"\nSaved to: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="AGV Station Coordinate Recorder")

    parser.add_argument("--host", type=str, default="192.168.192.5", help="AGV IP")
    parser.add_argument("--port", type=int, default=19204, help="AGV TCP port")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--station", type=str, help="Record current position as station")
    parser.add_argument("--desc", type=str, default="", help="Station description")
    parser.add_argument("--list", action="store_true", help="List recorded stations")
    parser.add_argument("--generate-yaml", action="store_true", help="Generate YAML config")
    parser.add_argument("--delete", type=str, help="Delete a station")
    parser.add_argument("--clear", action="store_true", help="Clear all stations")

    args = parser.parse_args()

    if args.generate_yaml:
        generate_yaml_config()
        return

    stations = load_stations()

    if args.delete:
        if args.delete in stations:
            del stations[args.delete]
            save_stations(stations)
            print(f"Deleted: {args.delete}")
        return

    if args.clear:
        stations.clear()
        save_stations(stations)
        print("All stations cleared")
        return

    if args.list and not args.host:
        list_stations()
        return

    controller = SeerAGVController(host=args.host, port=args.port)

    print(f"\nConnecting to AGV at {args.host}:{args.port}...")
    if not controller.connect():
        print("Failed to connect")
        if args.list:
            list_stations()
        return

    print("Connected")

    try:
        if args.interactive:
            interactive_record(controller)
        elif args.station:
            station_data = record_station(controller, args.station, args.desc)
            stations[args.station] = station_data
            save_stations(stations)
            print(f"\nStation '{args.station}' recorded")
        elif args.list:
            list_stations(controller)
        else:
            status = controller.get_status()
            print(f"\nCurrent AGV: ({status.position.x:.3f}, {status.position.y:.3f})")
            print(f"Station: {status.current_station}, Battery: {status.battery}%")
            print("\nUse --interactive, --station, or --list")
    finally:
        controller.disconnect()
        print("\nDisconnected")


if __name__ == "__main__":
    main()