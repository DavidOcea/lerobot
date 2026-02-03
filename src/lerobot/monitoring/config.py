"""
Configuration classes for the monitoring module.

Defines monitoring and logging configuration settings.
"""

from dataclasses import dataclass


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""

    enable_prometheus: bool = True
    prometheus_port: int = 8000
    log_level: str = "INFO"
    log_observations: bool = False
    log_actions: bool = False
    state_history_size: int = 1000
