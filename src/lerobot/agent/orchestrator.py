"""
Task agent orchestrator for robotic task execution.

This module provides the main orchestrator class that coordinates all
subsystems for autonomous robotic task execution.

Supports two execution modes:
1. Local mode (recommended): Direct policy execution without Policy Server
2. Remote mode: Uses Policy Server via gRPC for remote inference

New Features:
- Interactive task selection with user prompting
- Emergency stop with action history and rollback
- Task completion detection
- State monitoring and logging
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from lerobot.monitoring.state_monitor import StateMonitor
from lerobot.safety import (
    CollisionDetector,
    CollisionHandler,
    EnhancedCollisionDetector,
    create_enhanced_collision_config,
)
from lerobot.safety.collision_detector import CollisionConfig
from lerobot.tasks.completion_detector import TaskCompletionDetector
from lerobot.tasks.local_policy_executor import LocalPolicyExecutor
from lerobot.tasks.task_scheduler import ExecutionSummary, TaskScheduler, TaskStatus

# New imports
from lerobot.tasks.interactive_task_selector import (
    InteractiveTaskSelector,
    TaskSelection,
    ExecutionMode as ExecutionMode,
)
from lerobot.safety.emergency_stop_controller import (
    EmergencyStopController,
    DangerDetectionConfig,
    StopEvent,
    StopReason,
)

from .config import OrchestratorConfig

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot
    from lerobot.scripts.server.robot_client import RobotClient

logger = logging.getLogger(__name__)


class TaskAgentOrchestrator:
    """Main orchestrator for task-based robot control.

    The orchestrator brings together all subsystems:
    - Robot control (direct or via gRPC client)
    - Task scheduling and execution
    - Collision detection and handling
    - Task completion detection
    - State monitoring and logging

    Supports two execution modes:
    - Local mode (recommended): Direct policy execution, no Policy Server needed
    - Remote mode: Uses Policy Server via gRPC for remote inference

    Usage:
        # Local mode (recommended)
        config = load_config_from_yaml("configs/task_agent_tasks.yaml")
        config.use_local_execution = True  # Enable local mode
        orchestrator = TaskAgentOrchestrator(config)
        summary = orchestrator.run()

        # Remote mode (requires Policy Server)
        config.use_local_execution = False
        orchestrator = TaskAgentOrchestrator(config)
        summary = orchestrator.run()
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration.
        """
        self.config = config

        # Initialize components (will be connected in initialize())
        self.robot: Robot | None = None  # Direct robot instance (local mode)
        self.robot_client: RobotClient | None = None  # gRPC client (remote mode)
        self.local_executor: LocalPolicyExecutor | None = None  # Local policy executor
        self.task_scheduler: TaskScheduler | None = None
        self.collision_detector: CollisionDetector | None = None
        self.collision_handler: CollisionHandler | None = None
        self.state_monitor: StateMonitor | None = None
        self.completion_detectors: dict[str, TaskCompletionDetector] = {}

        # New components for interactive and emergency stop
        self.interactive_selector: InteractiveTaskSelector | None = None
        self.emergency_controller: EmergencyStopController | None = None

        # Execution state
        self.is_initialized = False
        self.is_running = False
        self.total_collision_count = 0

        # Execution mode
        self.use_local_execution = getattr(config, "use_local_execution", True)

        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on config settings."""
        log_level = self.config.monitoring_config.log_level
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def initialize(self) -> bool:
        """Initialize all subsystems and connect to robot.

        Returns:
            True if initialization was successful, False otherwise.
        """
        logger.info("Initializing TaskAgentOrchestrator...")
        logger.info(f"Execution mode: {'LOCAL (direct)' if self.use_local_execution else 'REMOTE (via Policy Server)'}")

        try:
            # Choose execution mode path
            if self.use_local_execution:
                return self._initialize_local_mode()
            else:
                return self._initialize_remote_mode()

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def _initialize_local_mode(self) -> bool:
        """Initialize for local execution mode (no Policy Server needed).

        Returns:
            True if initialization was successful, False otherwise.
        """
        logger.info("Initializing in LOCAL mode...")

        # 1. Connect directly to robot
        from lerobot.robots import make_robot_from_config
        self.robot = make_robot_from_config(self.config.robot_config)
        self.robot.connect()
        logger.info("Robot connected directly (no gRPC)")

        # 2. Initialize local policy executor
        device = getattr(self.config, "policy_device", "cuda")
        self.local_executor = LocalPolicyExecutor(device=device)
        logger.info(f"Local policy executor initialized on {device}")

        # 3. Initialize collision detector
        collision_cfg = self.config.collision_config
        if getattr(collision_cfg, 'use_enhanced_detector', False):
            # Use enhanced collision detector with multiple detection strategies
            enhanced_cfg = create_enhanced_collision_config(
                collision_threshold=collision_cfg.collision_threshold,
                detection_window=collision_cfg.detection_window,
                adaptive_mode=collision_cfg.adaptive_mode,
                velocity_compensation=collision_cfg.velocity_compensation,
                joint_specific_thresholds=getattr(collision_cfg, 'joint_specific_thresholds', {}),
                joint_inertia=getattr(collision_cfg, 'joint_inertia', {}),
                max_torque_limit=collision_cfg.max_torque_limit,
            )
            self.collision_detector = EnhancedCollisionDetector(enhanced_cfg)
            logger.info("Using Enhanced Collision Detector with multiple detection strategies")
        else:
            self.collision_detector = CollisionDetector(self.config.collision_config)

        if self.config.collision_config.auto_calibrate:
            logger.info("Calibrating collision detector...")
            self.collision_detector.calibrate_base_torques(
                self.robot,
                num_samples=self.config.collision_config.calibration_samples,
            )

        # 4. Initialize collision handler
        self.collision_handler = CollisionHandler(
            self.config.collision_config,
            self.robot,
        )

        # 5. Initialize state monitor
        if self.config.monitoring_config.enable_prometheus:
            self.state_monitor = StateMonitor(
                robot=self.robot,
                prometheus_port=self.config.monitoring_config.prometheus_port,
            )

        # 6. Initialize completion detectors
        for task in self.config.tasks:
            if task.completion_criteria.type != "position":
                self.completion_detectors[task.name] = TaskCompletionDetector(
                    task.completion_criteria
                )

        # 7. Initialize task scheduler with local executor
        from lerobot.tasks.task_scheduler import LocalTaskScheduler
        from lerobot.tasks.adaptive_scheduler import AdaptiveTaskScheduler

        base_scheduler = LocalTaskScheduler(
            tasks=self.config.tasks,
            robot=self.robot,
            policy_executor=self.local_executor,
            completion_detector=None,  # Will be set per task
        )

        # Wrap with adaptive scheduler if enabled
        if getattr(self.config, 'enable_adaptive_scheduler', True):
            gripper_config = getattr(self.config, 'gripper_config', None)
            self.task_scheduler = AdaptiveTaskScheduler(
                scheduler=base_scheduler,
                gripper_config=gripper_config or {}
            )
            logger.info("Using AdaptiveTaskScheduler with force feedback and adaptive speed control")
        else:
            self.task_scheduler = base_scheduler
            logger.info("Using standard LocalTaskScheduler")

        # 8. Initialize interactive task selector if enabled
        if getattr(self.config, 'enable_interactive_mode', False):
            self._init_interactive_selector()

        # 9. Initialize emergency stop controller if enabled
        if getattr(self.config, 'enable_emergency_stop', True):
            self._init_emergency_controller()

        self.is_initialized = True
        logger.info("Initialization complete (LOCAL mode)")
        return True

    def _initialize_remote_mode(self) -> bool:
        """Initialize for remote execution mode (requires Policy Server).

        Returns:
            True if initialization was successful, False otherwise.
        """
        logger.info("Initializing in REMOTE mode...")

        # 1. Connect to robot via gRPC
        self._connect_to_robot()

        # 2. Initialize collision detector
        self.collision_detector = CollisionDetector(self.config.collision_config)
        if self.config.collision_config.auto_calibrate:
            logger.info("Calibrating collision detector...")
            self._calibrate_collision_detector()

        # 3. Initialize collision handler
        self.collision_handler = CollisionHandler(
            self.config.collision_config,
            self.robot_client,
        )

        # 4. Initialize state monitor
        if self.config.monitoring_config.enable_prometheus:
            self.state_monitor = StateMonitor(
                robot=self.robot_client,
                prometheus_port=self.config.monitoring_config.prometheus_port,
            )

        # 5. Initialize completion detectors
        for task in self.config.tasks:
            if task.completion_criteria.type != "position":
                self.completion_detectors[task.name] = TaskCompletionDetector(
                    task.completion_criteria
                )

        # 6. Initialize task scheduler with gRPC client
        self.task_scheduler = TaskScheduler(
            tasks=self.config.tasks,
            robot=self.robot_client,
            policy_client=self.robot_client,
            completion_detector=None,
        )

        self.is_initialized = True
        logger.info("Initialization complete (REMOTE mode)")
        return True

    def _connect_to_robot(self):
        """Connect to the robot via gRPC client."""
        # Import here to avoid circular imports
        from lerobot.scripts.server.configs import RobotClientConfig
        from lerobot.scripts.server.robot_client import RobotClient
        from lerobot.robots import make_robot_from_config

        logger.info(
            f"Connecting to robot at {self.config.policy_server_host}:"
            f"{self.config.policy_server_port}..."
        )

        # Create robot config
        robot_config = self.config.robot_config

        # Create RobotClient config
        client_config = RobotClientConfig(
            robot=robot_config,
            server_address=f"{self.config.policy_server_host}:{self.config.policy_server_port}",
            task="task_agent",
            policy_type="act",
            pretrained_name_or_path="",  # Will be set per task
            policy_device="cuda",
            actions_per_chunk=50,
            environment_dt=self.config.environment_dt,
            fps=self.config.robot_config.control_frequency,
            verify_robot_cameras=False,
        )

        # Create and start client
        self.robot_client = RobotClient(client_config)

        if not self.robot_client.start():
            raise ConnectionError("Failed to start robot client")

        logger.info("Robot connected successfully")

    def _calibrate_collision_detector(self):
        """Calibrate the collision detector with the robot at rest."""
        if self.collision_detector is None:
            return

        logger.info("Calibrating collision detector base torques...")

        # Get initial observation for calibration
        observation = self.robot_client.get_latest_observation()

        # Use the robot's internal calibration if available
        if hasattr(self.robot_client, "robot") and self.robot_client.robot is not None:
            self.collision_detector.calibrate_base_torques(
                self.robot_client.robot,
                num_samples=self.config.collision_config.calibration_samples,
            )
        else:
            # Fallback: use current observation as base
            logger.warning(
                "Robot not directly accessible, using single observation for calibration"
            )
            for key, value in observation.items():
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    self.collision_detector.set_base_torque(joint_name, float(value))

    def _switch_cameras_for_task(self, task):
        """Switch active cameras for the given task.

        Args:
            task: TaskConfig with cameras list specifying which cameras to use.
        """
        if not task.cameras:
            logger.info(f"No cameras specified for task {task.name}, using default cameras")
            return

        camera_names = [cam.name for cam in task.cameras]
        logger.info(f"Switching to cameras for task {task.name}: {camera_names}")

        try:
            # Get robot reference (local or remote mode)
            robot = None
            if self.use_local_execution and self.robot is not None:
                robot = self.robot
            elif hasattr(self.robot_client, "robot") and self.robot_client.robot is not None:
                robot = self.robot_client.robot

            if robot is None:
                logger.warning("Cannot switch cameras - robot not accessible")
                return

            # Disable all cameras first
            for cam_name, cam in robot.cameras.items():
                if hasattr(cam, "is_enabled"):
                    cam.is_enabled = False

            # Enable specified cameras
            for cam_config in task.cameras:
                if cam_config.name in robot.cameras:
                    cam = robot.cameras[cam_config.name]
                    if hasattr(cam, "is_enabled"):
                        cam.is_enabled = True
                        logger.info(f"  Enabled camera: {cam_config.name} (index={cam_config.index})")
                    else:
                        logger.warning(f"  Camera {cam_config.name} does not support enable/disable")
                else:
                    logger.warning(f"  Camera {cam_config.name} not found in robot")

            # Log active cameras
            active = [name for name, cam in robot.cameras.items() if getattr(cam, "is_enabled", True)]
            logger.info(f"Active cameras: {active}")

        except Exception as e:
            logger.error(f"Failed to switch cameras: {e}")

    def _init_interactive_selector(self):
        """Initialize the interactive task selector."""
        if self.interactive_selector is not None:
            logger.info("Interactive selector already initialized")
            return

        self.interactive_selector = InteractiveTaskSelector(
            tasks=self.config.tasks,
            exit_handler=self._handle_exit_request,
        )
        logger.info("Interactive task selector initialized")

    def _init_emergency_controller(self):
        """Initialize the emergency stop controller."""
        if self.emergency_controller is not None:
            logger.info("Emergency controller already initialized")
            return

        # Get robot reference
        robot = self.robot if self.use_local_execution else self.robot_client
        if robot is None:
            logger.warning("Cannot initialize emergency controller - no robot available")
            return

        # Create danger detection config
        danger_config = DangerDetectionConfig(
            force_threshold=getattr(self.config, 'emergency_force_threshold', 2.5),
            total_force_threshold=getattr(self.config, 'emergency_total_force_threshold', 5.0),
            max_joint_force=getattr(self.config, 'emergency_max_joint_force', 1.5),
            max_velocity=getattr(self.config, 'emergency_max_velocity', 5.0),
            velocity_change_threshold=getattr(self.config, 'emergency_velocity_change_threshold', 2.0),
            max_action_delta=getattr(self.config, 'emergency_max_action_delta', 0.5),
            detection_window=getattr(self.config, 'emergency_detection_window', 5),
        )

        # Create rollback config
        from lerobot.safety.emergency_stop_controller import RollbackConfig
        rollback_config = RollbackConfig(
            max_rollback_steps=getattr(self.config, 'emergency_max_rollback_steps', 100),
            rollback_step_delay=getattr(self.config, 'emergency_rollback_step_delay', 0.02),
            safe_state_confirm_steps=getattr(self.config, 'emergency_safe_confirm_steps', 10),
        )

        # Create emergency controller
        self.emergency_controller = EmergencyStopController(
            robot=robot,
            history_size=getattr(self.config, 'emergency_history_size', 1000),
            danger_config=danger_config,
            rollback_config=rollback_config,
        )

        # Set stop callback
        self.emergency_controller.set_stop_callback(self._on_emergency_stop)

        logger.info("Emergency stop controller initialized")

    def _handle_exit_request(self) -> bool:
        """Handle user request to exit.

        Returns:
            True if exit should proceed.
        """
        logger.info("Exit requested by user")
        return True

    def _on_emergency_stop(self, stop_event):
        """Callback when emergency stop is triggered.

        Args:
            stop_event: StopEvent with details of the stop.
        """
        logger.warning(f"Emergency stop triggered: {stop_event.reason.value}")

        # Update collision count for tracking
        self.total_collision_count += 1

    def _check_emergency_stop(self, observation: dict[str, Any], action: dict[str, float]) -> bool:
        """Check if emergency stop should be triggered.

        Args:
            observation: Current observation from robot.
            action: Current action being executed.

        Returns:
            True if emergency stop was triggered, False otherwise.
        """
        if self.emergency_controller is None:
            return False

        # Check for dangerous action
        is_dangerous, reason = self.emergency_controller.check_action_danger(action, observation)

        if is_dangerous:
            logger.warning(f"Dangerous action detected: {reason.value if reason else 'unknown'}")

            # Trigger emergency stop
            auto_rollback = getattr(self.config, 'auto_rollback_on_stop', True)
            self.emergency_controller.trigger_stop(
                reason=reason or StopReason.DANGEROUS_ACTION,
                auto_rollback=auto_rollback
            )

            return True

        return False

    def _handle_emergency_resume(self) -> bool:
        """Handle resuming after emergency stop.

        Returns:
            True if successfully resumed, False otherwise.
        """
        if self.emergency_controller is None:
            return True  # No emergency controller, nothing to resume

        # Resume execution
        success = self.emergency_controller.resume()

        if success:
            logger.info("Successfully resumed after emergency stop")

        return success

    def run(self) -> ExecutionSummary:
        """Execute the complete task sequence.

        Returns:
            ExecutionSummary with results for all tasks.
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize orchestrator")

        self.is_running = True
        self.total_collision_count = 0

        logger.info(f"Starting task sequence with {len(self.config.tasks)} tasks")

        try:
            # Execute tasks with safety monitoring
            summary = self._execute_with_safety()

            return summary

        finally:
            self.is_running = False
            self._cleanup()

    def _execute_with_safety(self) -> ExecutionSummary:
        """Execute task sequence with collision monitoring and interactive selection."""
        results = []

        for i, task in enumerate(self.config.tasks):
            if not task.enabled:
                logger.info(f"Skipping disabled task: {task.name}")
                continue

            # Interactive task selection before each task
            if self.interactive_selector is not None:
                selection = self.interactive_selector.prompt_next_task()

                # Handle exit request
                if selection.exit_requested:
                    logger.info("User requested exit, stopping task sequence")
                    break

                # Handle custom task creation
                if selection.custom_task_name is not None:
                    logger.info(f"Custom task requested: {selection.custom_task_name}")
                    # Would need to execute custom task here
                    continue

                # Handle specific task selection
                if selection.selected_task:
                    # Find and execute selected task
                    for t in self.config.tasks:
                        if t.name.lower() == selection.selected_task.lower():
                            task = t
                            i = self.config.tasks.index(t)
                            break
                    logger.info(f"User selected task: {task.name}")

            logger.info(f"Executing task {i + 1}/{len(self.config.tasks)}: {task.name}")

            # Switch cameras for this task
            self._switch_cameras_for_task(task)

            # Set completion detector for this task
            self.task_scheduler.completion_detector = self.completion_detectors.get(
                task.name
            )

            # Override settings if specified
            original_max_retries = task.max_retries
            original_max_duration = task.max_duration

            if self.config.override_max_retries is not None:
                task.max_retries = self.config.override_max_retries
            if self.config.override_max_duration is not None:
                task.max_duration = self.config.override_max_duration

            # Execute task
            # Check if using adaptive scheduler
            if hasattr(self.task_scheduler, 'execute_task_adaptive'):
                result = self.task_scheduler.execute_task_adaptive(
                    task,
                    collision_detector=self.collision_detector,
                    collision_handler=self.collision_handler,
                    state_monitor=self.state_monitor,
                )
            else:
                result = self.task_scheduler.execute_task_with_safety(
                    task,
                    collision_detector=self.collision_detector,
                    collision_handler=self.collision_handler,
                    state_monitor=self.state_monitor,
                )

            # Restore original settings
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration

            results.append(result)

            # Update collision count
            if result.collision_detected:
                self.total_collision_count += 1

            # Check if we should continue
            if result.status == TaskStatus.FATAL_FAILURE:
                logger.error(f"Fatal failure in task {task.name}, aborting sequence")
                break

            if self.total_collision_count >= self.config.max_total_collisions:
                logger.error(
                    f"Maximum collision count reached ({self.config.max_total_collisions}), aborting"
                )
                break

        # Build summary
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == TaskStatus.FAILED)
        fatal = sum(1 for r in results if r.status == TaskStatus.FATAL_FAILURE)

        return ExecutionSummary(
            total_tasks=len(self.config.tasks),
            completed_tasks=completed,
            failed_tasks=failed + fatal,
            skipped_tasks=0,
            total_duration=sum(r.duration for r in results),
            task_results=results,
            overall_success=(fatal == 0 and completed > 0),
            collision_count=self.total_collision_count,
            total_retries=sum(r.attempts - 1 for r in results),
        )

    def _cleanup(self):
        """Clean up resources after execution."""
        logger.info("Cleaning up...")

        # Stop emergency controller
        if self.emergency_controller is not None:
            logger.info("Stopping emergency stop controller")

        # Stop state monitor
        if self.state_monitor is not None:
            self.state_monitor.stop()

        # Disconnect from robot (local mode)
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")

        # Stop robot client (remote mode)
        if self.robot_client is not None:
            try:
                self.robot_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping robot client: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status.

        Returns:
            Dictionary with status information.
        """
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "total_collisions": self.total_collision_count,
            "collision_detector_stats": (
                self.collision_detector.get_statistics()
                if self.collision_detector
                else None
            ),
            "collision_handler_stats": (
                self.collision_handler.get_statistics() if self.collision_handler else None
            ),
            "scheduler_state": (
                self.task_scheduler.get_current_state() if self.task_scheduler else None
            ),
        }
