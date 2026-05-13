"""
Monitoring module for robotic task execution.

This module provides:
- State monitoring and logging
- Anomaly detection
- Prometheus metrics integration
- Observation and action history tracking
- Real-time HTTP dashboard (MonitorCollector + HTTPDashboard)
"""

from .config import MonitoringConfig
from .state_monitor import (
    AnomalyDetectionResult,
    MonitoringStats,
    StateMonitor,
    StateSnapshot,
)
from .dashboard import (
    MonitorCollector,
    HTTPDashboard,
    RobotSnapshot,
    AGVSnapshot,
    TaskSnapshot,
)

__all__ = [
    "MonitoringConfig",
    "AnomalyDetectionResult",
    "MonitoringStats",
    "StateMonitor",
    "StateSnapshot",
    "MonitorCollector",
    "HTTPDashboard",
    "RobotSnapshot",
    "AGVSnapshot",
    "TaskSnapshot",
]
