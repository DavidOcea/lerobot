"""
Robot state monitoring and anomaly detection.

This module provides monitoring capabilities for robotic task execution,
including state tracking, anomaly detection, and optional Prometheus metrics.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot
    from lerobot.scripts.server.robot_client import RobotClient

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """A snapshot of robot state at a point in time."""

    timestamp: float
    observation: dict[str, Any]
    action: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection check."""

    is_anomaly: bool = False
    anomaly_type: str | None = None  # "position", "force", "velocity", "collision"
    severity: str = "none"  # "none", "low", "medium", "high"
    affected_joints: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MonitoringStats:
    """Statistics for monitoring system."""

    total_updates: int = 0
    anomaly_count: int = 0
    last_update_time: float = 0
    observation_history_size: int = 0
    average_update_frequency: float = 0


class StateMonitor:
    """Monitors robot state and detects anomalies.

    The monitor tracks:
    - Observation and action history
    - Joint position and force trends
    - Anomaly patterns
    - Optional Prometheus metrics

    Usage:
        monitor = StateMonitor(robot, prometheus_port=8000)
        monitor.update(observation, action)
        anomalies = monitor.detect_anomalies()
    """

    def __init__(
        self,
        robot: "Robot | RobotClient",
        prometheus_port: int = 8000,
        history_size: int = 1000,
        enable_prometheus: bool = True,
    ):
        """Initialize the state monitor.

        Args:
            robot: Robot instance or RobotClient to monitor.
            prometheus_port: Port for Prometheus metrics server.
            history_size: Maximum number of state snapshots to keep.
            enable_prometheus: Whether to start Prometheus metrics server.
        """
        self.robot = robot
        self.prometheus_port = prometheus_port
        self.history_size = history_size
        self.enable_prometheus = enable_prometheus

        # State history
        self.history: deque[StateSnapshot] = deque(maxlen=history_size)

        # Statistics
        self._start_time = time.time()
        self._total_updates = 0
        self._anomaly_count = 0

        # Prometheus exporter (optional)
        self._prometheus_server = None
        if self.enable_prometheus:
            self._setup_prometheus()

    def _setup_prometheus(self):
        """Set up Prometheus metrics exporter."""
        try:
            from prometheus_client import Counter, Gauge, start_http_server

            # Define metrics with unique names to avoid conflicts
            self._metrics = {
                "lerobot_observations_total": Counter(
                    "lerobot_robot_observations_total",
                    "Total number of observations received",
                ),
                "lerobot_actions_total": Counter(
                    "lerobot_robot_actions_total",
                    "Total number of actions executed",
                ),
                "lerobot_anomalies_total": Counter(
                    "lerobot_robot_anomalies_total",
                    "Total number of anomalies detected",
                ),
                "lerobot_joint_position": Gauge(
                    "lerobot_joint_position",
                    "Current joint position",
                    ["joint_name"],
                ),
                "lerobot_joint_force": Gauge(
                    "lerobot_joint_force",
                    "Current joint force/torque",
                    ["joint_name"],
                ),
            }

            # Start HTTP server
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")

        except ImportError:
            logger.warning("prometheus_client not installed, disabling Prometheus metrics")
            self.enable_prometheus = False
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            self.enable_prometheus = False

    def update(
        self,
        observation: dict[str, Any],
        action: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Update monitor with new observation and action.

        Args:
            observation: Current observation dict.
            action: Current action dict (optional).
            metadata: Additional metadata to store (optional).
        """
        snapshot = StateSnapshot(
            timestamp=time.time(),
            observation=observation.copy(),
            action=action.copy() if action else None,
            metadata=metadata.copy() if metadata else {},
        )

        self.history.append(snapshot)
        self._total_updates += 1

        # Update Prometheus metrics
        if self.enable_prometheus:
            self._update_prometheus_metrics(snapshot)

    def _update_prometheus_metrics(self, snapshot: StateSnapshot):
        """Update Prometheus metrics with snapshot data."""
        if not hasattr(self, "_metrics"):
            return

        try:
            # Update counters
            self._metrics["observations_total"].inc()
            if snapshot.action:
                self._metrics["actions_total"].inc()

            # Update joint positions
            for key, value in snapshot.observation.items():
                if ".pos" in key:
                    joint_name = key.replace(".pos", "")
                    self._metrics["joint_position"].labels(joint_name=joint_name).set(value)

                # Update joint forces
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    self._metrics["joint_force"].labels(joint_name=joint_name).set(value)

        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")

    def detect_anomalies(self) -> AnomalyDetectionResult:
        """Analyze recent state history for anomalies.

        Returns:
            AnomalyDetectionResult with any detected anomalies.
        """
        if len(self.history) < 10:
            return AnomalyDetectionResult()

        result = AnomalyDetectionResult(timestamp=time.time())

        # Get recent snapshots
        recent = list(self.history)[-50:]

        # Check for position anomalies
        position_anomalies = self._detect_position_anomalies(recent)
        if position_anomalies:
            result.is_anomaly = True
            result.anomaly_type = "position"
            result.affected_joints.extend(position_anomalies)

        # Check for force anomalies
        force_anomalies = self._detect_force_anomalies(recent)
        if force_anomalies:
            result.is_anomaly = True
            result.anomaly_type = "force" if result.anomaly_type is None else result.anomaly_type
            result.affected_joints.extend(force_anomalies)

        # Determine severity
        if result.is_anomaly:
            if len(result.affected_joints) > 3:
                result.severity = "high"
            elif len(result.affected_joints) > 1:
                result.severity = "medium"
            else:
                result.severity = "low"

            self._anomaly_count += 1

            if self.enable_prometheus and hasattr(self, "_metrics"):
                self._metrics["anomalies_total"].inc()

        return result

    def _detect_position_anomalies(self, snapshots: list[StateSnapshot]) -> list[str]:
        """Detect position anomalies in recent snapshots."""
        anomalous_joints = []

        if len(snapshots) < 2:
            return anomalous_joints

        # Get joint names
        joint_names = set()
        for snapshot in snapshots:
            for key in snapshot.observation.keys():
                if ".pos" in key:
                    joint_names.add(key.replace(".pos", ""))

        for joint_name in joint_names:
            positions = []
            for snapshot in snapshots:
                pos_key = f"{joint_name}.pos"
                if pos_key in snapshot.observation:
                    positions.append(snapshot.observation[pos_key])

            if len(positions) < 2:
                continue

            # Check for sudden jumps or high variance
            positions_array = np.array(positions)
            diffs = np.abs(np.diff(positions_array))

            # Threshold for sudden jump detection
            jump_threshold = 0.5  # radians
            max_diff = np.max(diffs) if len(diffs) > 0 else 0

            if max_diff > jump_threshold:
                anomalous_joints.append(joint_name)

        return anomalous_joints

    def _detect_force_anomalies(self, snapshots: list[StateSnapshot]) -> list[str]:
        """Detect force anomalies in recent snapshots."""
        anomalous_joints = []

        if len(snapshots) < 2:
            return anomalous_joints

        # Get joint names
        joint_names = set()
        for snapshot in snapshots:
            for key in snapshot.observation.keys():
                if ".force" in key:
                    joint_names.add(key.replace(".force", ""))

        for joint_name in joint_names:
            forces = []
            for snapshot in snapshots:
                force_key = f"{joint_name}.force"
                if force_key in snapshot.observation:
                    forces.append(snapshot.observation[force_key])

            if len(forces) < 5:
                continue

            # Check for high force values or spikes
            forces_array = np.array(forces)
            max_force = np.max(np.abs(forces_array))
            mean_force = np.mean(np.abs(forces_array))

            # Threshold for high force detection
            force_threshold = 3.0  # Nm

            if max_force > force_threshold:
                anomalous_joints.append(joint_name)

            # Check for sudden spikes
            diffs = np.abs(np.diff(forces_array))
            if len(diffs) > 0 and np.max(diffs) > 2.0:
                anomalous_joints.append(joint_name)

        return anomalous_joints

    def get_statistics(self) -> MonitoringStats:
        """Get monitoring statistics.

        Returns:
            MonitoringStats with current statistics.
        """
        elapsed = time.time() - self._start_time
        avg_freq = self._total_updates / elapsed if elapsed > 0 else 0

        return MonitoringStats(
            total_updates=self._total_updates,
            anomaly_count=self._anomaly_count,
            last_update_time=self.history[-1].timestamp if self.history else 0,
            observation_history_size=len(self.history),
            average_update_frequency=avg_freq,
        )

    def get_recent_history(self, n: int = 10) -> list[StateSnapshot]:
        """Get the n most recent state snapshots.

        Args:
            n: Number of recent snapshots to return.

        Returns:
            List of recent StateSnapshot objects.
        """
        return list(self.history)[-n:]

    def clear_history(self):
        """Clear the state history."""
        self.history.clear()

    def stop(self):
        """Stop the monitor and clean up resources."""
        logger.info("Stopping state monitor...")
        self.clear_history()

    def export_history(self, filepath: str):
        """Export state history to a file.

        Args:
            filepath: Path to save the history (supports .json, .npz).
        """
        if not self.history:
            logger.warning("No history to export")
            return

        filepath = str(filepath)

        if filepath.endswith(".npz"):
            # Export as numpy archive
            import pickle

            data = {
                "snapshots": [s.__dict__ for s in self.history],
                "statistics": self.get_statistics().__dict__,
            }
            np.savez(filepath, data=data)
            logger.info(f"History exported to {filepath}")

        elif filepath.endswith(".json"):
            # Export as JSON
            import json

            data = {
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "observation": s.observation,
                        "action": s.action,
                        "metadata": s.metadata,
                    }
                    for s in self.history
                ],
                "statistics": self.get_statistics().__dict__,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"History exported to {filepath}")

        else:
            logger.error(f"Unsupported file format: {filepath}")
