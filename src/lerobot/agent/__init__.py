"""
Agent module for robotic task orchestration.

This module provides the main orchestrator that coordinates:
- Task scheduling and execution
- Collision detection and handling
- State monitoring and logging
- Policy switching and management
"""

from .config import OrchestratorConfig
from .orchestrator import TaskAgentOrchestrator

__all__ = [
    "OrchestratorConfig",
    "TaskAgentOrchestrator",
]
