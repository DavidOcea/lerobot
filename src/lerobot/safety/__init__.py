"""
Safety module for robot collision detection and handling.

This module provides:
- Collision detection based on torque sensing
- Enhanced collision detection with multiple strategies
- Temporal collision detection with pattern analysis
- Adaptive collision detection with motion-aware thresholds
- Collision handling and recovery strategies
- Safety monitoring and emergency stop functionality
"""

from .adaptive_collision_detector import (
    AdaptiveCollisionDetector,
    AdaptiveThresholdConfig,
    MotionPhase,
    create_adaptive_collision_config,
)
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
from .temporal_collision_detector import (
    OscillationDetector,
    TemporalCollisionConfig,
    TemporalCollisionDetector,
    TrendAnalyzer,
    create_temporal_collision_config,
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
    # Temporal collision detection
    "TemporalCollisionConfig",
    "TemporalCollisionDetector",
    "OscillationDetector",
    "TrendAnalyzer",
    "create_temporal_collision_config",
    # Adaptive collision detection
    "AdaptiveThresholdConfig",
    "AdaptiveCollisionDetector",
    "MotionPhase",
    "create_adaptive_collision_config",
    # Collision handling
    "CollisionHandler",
    "RecoveryAction",
    "RecoveryStrategy",
]
