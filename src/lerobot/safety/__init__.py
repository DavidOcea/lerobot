"""
Safety module for robot collision detection and handling.

This module provides:
- Collision detection based on torque sensing
- Enhanced collision detection with multiple strategies
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
from .enhanced_collision_detector import (
    EnhancedCollisionConfig,
    EnhancedCollisionDetector,
    create_enhanced_collision_config,
)

__all__ = [
    # Collision detection
    "CollisionConfig",
    "CollisionDetector",
    "CollisionEvent",
    "CollisionResult",
    # Enhanced collision detection
    "EnhancedCollisionConfig",
    "EnhancedCollisionDetector",
    "create_enhanced_collision_config",
    # Collision handling
    "CollisionHandler",
    "RecoveryAction",
    "RecoveryStrategy",
]
