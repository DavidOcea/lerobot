"""
Collision handling and recovery strategies.

This module implements collision response strategies including:
- Emergency stop
- Safe retreat to previous state
- Task continuation assessment
- Recovery action execution
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .collision_detector import CollisionEvent, CollisionResult
from .config import CollisionConfig

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies after collision."""

    STOP = "stop"  # Simply stop, do nothing else
    STOP_AND_RETREAT = "stop_and_retreat"  # Stop and move back slightly
    RETRY = "retry"  # Stop, retreat, and retry the action


@dataclass
class RecoveryAction:
    """Represents a recovery action to execute."""

    strategy: RecoveryStrategy
    retreat_distance: float  # Radians to retreat per joint
    wait_time: float  # Seconds to wait after retreat
    retry_allowed: bool  # Whether task retry is allowed


class CollisionHandler:
    """Handles collision events and executes recovery strategies.

    The handler is responsible for:
    1. Executing emergency stop when collision is detected
    2. Optionally retreating to a safe state
    3. Assessing whether the task can continue
    4. Returning appropriate status for task resumption

    Usage:
        handler = CollisionHandler(config, robot)
        result = handler.handle_collision(collision_event, previous_observation)
        if result.should_continue:
            # Continue with task
        else:
            # Abort or restart
    """

    def __init__(self, config: CollisionConfig, robot: "Robot"):
        """Initialize the collision handler.

        Args:
            config: Collision configuration containing recovery settings.
            robot: Robot instance for executing recovery actions.
        """
        self.config = config
        self.robot = robot

        # State tracking
        self._pre_collision_state: dict[str, float] = {}
        self._collision_count = 0
        self._recovery_count = 0

        # Parse recovery strategy
        self.strategy = RecoveryStrategy(config.recovery_strategy)

    def handle_collision(
        self,
        collision_result: CollisionResult,
        previous_observation: dict[str, Any],
        previous_action: dict[str, Any] | None = None,
    ) -> "CollisionHandlerResult":
        """Handle a detected collision event.

        Args:
            collision_result: Result from collision detector.
            previous_observation: Observation from before the collision.
            previous_action: Action being executed when collision occurred.

        Returns:
            CollisionHandlerResult with handling outcome.
        """
        self._collision_count += 1

        logger.warning(
            f"Handling collision #{self._collision_count}. "
            f"Severity: {collision_result.severity}, "
            f"Affected joints: {list(collision_result.affected_joints.keys())}"
        )

        # Store pre-collision state for potential retreat
        self._store_pre_collision_state(previous_observation)

        # Execute emergency stop
        stop_success = self.stop_robot()

        # Determine recovery action
        recovery = self._get_recovery_action(collision_result)

        # Execute recovery based on strategy
        recovery_success = False
        if recovery.strategy == RecoveryStrategy.STOP:
            recovery_success = True  # Stop is always successful

        elif recovery.strategy == RecoveryStrategy.STOP_AND_RETREAT:
            recovery_success = self._retreat_to_safe_state(recovery.retreat_distance)

        elif recovery.strategy == RecoveryStrategy.RETRY:
            recovery_success = self._retreat_to_safe_state(recovery.retreat_distance)

        # Assess task continuation
        can_continue = self.assess_task_continuation(collision_result, recovery_success)

        result = CollisionHandlerResult(
            collision_count=self._collision_count,
            recovery_attempted=recovery.strategy != RecoveryStrategy.STOP,
            recovery_success=recovery_success,
            can_continue=can_continue,
            strategy_used=recovery.strategy,
            should_retry=recovery.retry_allowed and can_continue,
        )

        if recovery_success:
            self._recovery_count += 1
            logger.info(f"Recovery completed. Can continue: {can_continue}")

        return result

    def stop_robot(self) -> bool:
        """Execute emergency stop on the robot.

        Returns:
            True if stop was successful, False otherwise.
        """
        try:
            # Check if robot has emergency_stop method
            if hasattr(self.robot, "emergency_stop"):
                self.robot.emergency_stop()
                logger.info("Emergency stop executed via robot.emergency_stop()")
            else:
                # Fallback: deactivate hardware
                if hasattr(self.robot, "_hardware_manager"):
                    self.robot._hardware_manager.deactivate()
                    logger.info("Emergency stop executed via hardware_manager.deactivate()")
                else:
                    logger.warning("No emergency stop method available")

            return True

        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            return False

    def retreat_to_safe_state(self, distance: float | None = None) -> bool:
        """Retreat robot joints to a safer state after collision.

        Moves each joint slightly in the opposite direction of its last movement.

        Args:
            distance: Distance to retreat in radians (uses config if None).

        Returns:
            True if retreat was successful, False otherwise.
        """
        if distance is None:
            distance = self.config.recovery_retreat_distance

        if not self._pre_collision_state:
            logger.warning("No pre-collision state stored, cannot retreat")
            return False

        try:
            # Build retreat action by moving back from current position
            current_obs = self.robot.get_observation()

            retreat_action = {}
            for joint_key in self._pre_collision_state:
                if ".pos" in joint_key:
                    joint_name = joint_key.replace(".pos", "")

                    # Get current position
                    current_pos = current_obs.get(joint_key, 0)

                    # Move back by retreat distance
                    retreat_pos = current_pos - distance

                    # Clamp to safe range if needed
                    retreat_pos = max(-3.14, min(3.14, retreat_pos))

                    retreat_action[joint_name] = retreat_pos

            if retreat_action:
                # Send retreat action
                self.robot.send_action(retreat_action)
                logger.info(f"Retreating with distance: {distance} rad")

                # Wait for retreat to complete
                time.sleep(0.5)

                return True

        except Exception as e:
            logger.error(f"Failed to retreat: {e}")
            return False

        return False

    def _retreat_to_safe_state(self, distance: float) -> bool:
        """Internal retreat method (alias for retreat_to_safe_state)."""
        return self.retreat_to_safe_state(distance)

    def assess_task_continuation(
        self, collision_result: CollisionResult, recovery_success: bool
    ) -> bool:
        """Assess whether the task can continue after collision.

        Args:
            collision_result: The collision that occurred.
            recovery_success: Whether recovery action was successful.

        Returns:
            True if task can continue, False if it should be aborted.
        """
        # High severity collisions may require abort
        if collision_result.severity == "high":
            logger.warning("High severity collision - task continuation not recommended")
            return False

        # If recovery failed, don't continue
        if not recovery_success:
            logger.error("Recovery failed - cannot continue task")
            return False

        # Check if too many collisions have occurred
        if self._collision_count > 5:
            logger.error(f"Too many collisions ({self._collision_count}) - aborting task")
            return False

        # Otherwise, allow continuation
        return True

    def _get_recovery_action(self, collision_result: CollisionResult) -> RecoveryAction:
        """Determine the appropriate recovery action based on collision severity."""
        # For high severity, always stop
        if collision_result.severity == "high":
            return RecoveryAction(
                strategy=RecoveryStrategy.STOP,
                retreat_distance=0.0,
                wait_time=0.0,
                retry_allowed=False,
            )

        # Use configured strategy
        retreat_distance = self.config.recovery_retreat_distance
        retry_allowed = self.strategy == RecoveryStrategy.RETRY

        return RecoveryAction(
            strategy=self.strategy,
            retreat_distance=retreat_distance,
            wait_time=self.config.recovery_timeout,
            retry_allowed=retry_allowed,
        )

    def _store_pre_collision_state(self, observation: dict[str, Any]):
        """Store the robot state before collision for potential retreat."""
        self._pre_collision_state = {
            k: float(v) for k, v in observation.items() if ".pos" in k
        }

    def reset(self):
        """Reset handler state."""
        self._pre_collision_state.clear()
        self._collision_count = 0
        self._recovery_count = 0

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "collision_count": self._collision_count,
            "recovery_count": self._recovery_count,
            "recovery_rate": (
                self._recovery_count / self._collision_count
                if self._collision_count > 0
                else 0
            ),
            "strategy": self.strategy.value,
        }


@dataclass
class CollisionHandlerResult:
    """Result of collision handling operation."""

    collision_count: int
    recovery_attempted: bool
    recovery_success: bool
    can_continue: bool
    strategy_used: RecoveryStrategy
    should_retry: bool
