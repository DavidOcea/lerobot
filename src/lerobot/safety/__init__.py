"""
Safety module for robot collision detection and handling.

This module provides:
- Collision detection based on torque sensing
- Collision handling and recovery strategies
- Safety monitoring and emergency stop functionality
"""

from .collision_detector import (
    CollisionConfig,
    CollisionDetector,
    CollisionEvent,
    CollisionResult,
)
from .collision_handler import (
    CollisionHandler,
    RecoveryAction,
    RecoveryStrategy,
)

__all__ = [
    # Collision detection
    "CollisionConfig",
    "CollisionDetector",
    "CollisionEvent",
    "CollisionResult",
    # Collision handling
    "CollisionHandler",
    "RecoveryAction",
    "RecoveryStrategy",
]
