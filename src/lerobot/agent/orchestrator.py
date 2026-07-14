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
- AGV navigation task support (NEW)
"""

import logging
import os
import select
import sys
import time
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
from lerobot.tasks.config import TaskConfig, parse_task_dict
from lerobot.tasks.local_policy_executor import LocalPolicyExecutor
from lerobot.tasks.task_scheduler import ExecutionSummary, TaskResult, TaskScheduler, TaskStatus

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
    RecoveryAction,
)

# AGV imports (NEW)
from lerobot.robots.agv.seer_agv_controller import SeerAGVController, AGVPosition
from lerobot.tasks.agv_executor import AGVTaskExecutor, AGVExecutionResult, create_task_result_from_agv_result
from lerobot.tasks.classifier import make_classifier, reset_classify_counters, _counted_label

# Monitoring imports (NEW)
from lerobot.monitoring.dashboard import MonitorCollector, HTTPDashboard
import yaml

from .config import OrchestratorConfig, AGVGlobalConfig

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

        # AGV components (NEW)
        self.agv_controller: SeerAGVController | None = None
        self.agv_executor: AGVTaskExecutor | None = None

        # Monitoring dashboard (NEW)
        self.monitor_collector: "MonitorCollector | None" = None
        self.http_dashboard: "HTTPDashboard | None" = None

        # Execution state
        self.is_initialized = False
        self.is_running = False
        self.total_collision_count = 0

        # Execution mode
        self.use_local_execution = getattr(config, "use_local_execution", True)

        # Track builtin skill names so we can skip them during auto-advance
        self._builtin_task_names: set[str] = set()

        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on config settings."""
        log_level = self.config.monitoring_config.log_level
        if '--debug' in sys.argv:
            log_level = 'DEBUG'
        level = getattr(logging, log_level)
        # setLevel works even after basicConfig, unlike basicConfig itself
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.getLogger().setLevel(level)
        logging.getLogger('lerobot').setLevel(level)

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
        print(f"[Orchestrator] Initializing completion detectors for {len(self.config.tasks)} tasks...")
        for task in self.config.tasks:
            print(f"[Orchestrator] Task: {task.name}, criteria type: {task.completion_criteria.type}")
            if task.completion_criteria.type != "position" and task.task_type != "position_sequence":
                self.completion_detectors[task.name] = TaskCompletionDetector(
                    task.completion_criteria
                )
                print(f"[Orchestrator] Created detector for {task.name}")
        print(f"[Orchestrator] Total detectors created: {len(self.completion_detectors)}")

        # 7. Initialize emergency stop controller FIRST (needed by adaptive scheduler)
        if getattr(self.config, 'enable_emergency_stop', True):
            self._init_emergency_controller()

        # 8. Initialize task scheduler with local executor
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
                gripper_config=gripper_config or {},
                enable_action_smoothing=getattr(self.config, 'enable_action_smoothing', True),
                smoothing_level=getattr(self.config, 'action_smoothing_level', 'medium'),
                emergency_check_callback=self._check_emergency_stop if self.emergency_controller else None,
                emergency_controller=self.emergency_controller,  # Pass controller for auto recording
            )
            logger.info("Using AdaptiveTaskScheduler with force feedback and adaptive speed control")
        else:
            self.task_scheduler = base_scheduler
            logger.info("Using standard LocalTaskScheduler")

        # 9. Initialize interactive task selector if enabled
        if getattr(self.config, 'enable_interactive_mode', False):
            self._init_interactive_selector()

        # 10. Initialize AGV components if enabled (NEW)
        agv_config = getattr(self.config, 'agv_config', None)
        if agv_config and agv_config.enabled:
            self._init_agv(agv_config)

        # 11. Initialize monitoring dashboard (NEW)
        self._init_monitoring()

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
            if task.completion_criteria.type != "position" and task.task_type != "position_sequence":
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

        tasks = list(self.config.tasks)  # copy before possibly prepending

        # Load and prepend builtin generic skills (AGV moves, robot home, etc.)
        # We do this here — not in load_config_from_yaml — because only the
        # orchestrator knows whether interactive mode is truly enabled.
        builtin_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "builtin_skills.yaml"
        if builtin_path.exists():
            try:
                builtin = yaml.safe_load(builtin_path.read_text())
                builtin_tasks = builtin.get("tasks", [])
                if builtin_tasks:
                    named_positions = self.config.named_positions
                    default_arm_safe = self.config.agv_config.default_arm_safe_positions if self.config.agv_config else {}
                    default_arm_home = self.config.agv_config.default_arm_home_positions if self.config.agv_config else {}
                    builtin_configs = []
                    for td in builtin_tasks:
                        try:
                            tc = parse_task_dict(td, named_positions, default_arm_safe, default_arm_home)
                            builtin_configs.append(tc)
                        except Exception as e:
                            logger.warning(
                                f"Skipping builtin skill '{td.get('name', td)}': {e}"
                            )
                    tasks = builtin_configs + tasks
                    self._builtin_task_names = {tc.name for tc in builtin_configs}
                    logger.info(
                        f"Merged {len(builtin_configs)} builtin skill(s) from {builtin_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load builtin skills: {e}")

        initial_mode = ExecutionMode.INTERACTIVE if self.config.enable_interactive_mode else ExecutionMode.AUTOMATIC
        self.interactive_selector = InteractiveTaskSelector(
            tasks=tasks,
            exit_handler=self._handle_exit_request,
            monitor_collector=self.monitor_collector,
            robot=self.robot,
            initial_mode=initial_mode,
        )
        logger.info(f"Interactive task selector initialized (mode: {initial_mode.value})")

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
            force_threshold=getattr(self.config, 'emergency_force_threshold', 5.0),  # Increased from 2.5
            total_force_threshold=getattr(self.config, 'emergency_total_force_threshold', 15.0),  # Increased from 5.0
            max_joint_force=getattr(self.config, 'emergency_max_joint_force', 3.0),  # Increased from 1.5
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

    def _init_agv(self, agv_config: AGVGlobalConfig) -> bool:
        """Initialize AGV controller and executor.

        Args:
            agv_config: AGV global configuration.

        Returns:
            True if AGV initialized successfully, False otherwise.
        """
        try:
            logger.info(f"Initializing AGV controller: host={agv_config.host}")

            # Create AGV TCP controller
            self.agv_controller = SeerAGVController(
                host=agv_config.host,
                port=agv_config.port,
                connection_timeout=agv_config.connection_timeout,
                read_timeout=agv_config.read_timeout,
                auto_reconnect=agv_config.auto_reconnect,
            )

            # Attempt connection
            if not self.agv_controller.connect():
                logger.warning("AGV connection failed, AGV tasks will be skipped")
                self.agv_controller = None
                return False

            # Set station map if provided
            if agv_config.station_map:
                station_map = {
                    station_id: tuple(coords)  # Convert list to tuple
                    for station_id, coords in agv_config.station_map.items()
                }
                self.agv_controller.set_station_map(station_map)

            # Create AGV executor
            self.agv_executor = AGVTaskExecutor(
                agv_controller=self.agv_controller,
                robot=self.robot,
                enable_safety_check=agv_config.check_arm_before_move,
            )

            logger.info("AGV initialized successfully")
            return True

        except Exception as e:
            logger.error(f"AGV initialization failed: {e}")
            self.agv_controller = None
            self.agv_executor = None
            return False

    def _init_monitoring(self):
        """Initialize the real-time monitoring dashboard.

        Creates a MonitorCollector (background AGV polling) and HTTPDashboard
        (lightweight HTTP server).  The monitor_collector is attached to the
        task_scheduler so the main action loop can push robot state updates.
        """
        dashboard_port = getattr(self.config, 'monitoring_dashboard_port', 8080)
        enable_dashboard = getattr(self.config, 'enable_monitoring_dashboard', True)

        if not enable_dashboard:
            logger.info("Monitoring dashboard disabled")
            return

        try:
            # Create collector (AGV polling in background thread)
            self.monitor_collector = MonitorCollector(
                agv_controller=self.agv_controller,
                robot=self.robot,
                cameras=getattr(self.robot, 'cameras', {}),
            )
            self.monitor_collector.start()

            # Attach to task scheduler for frame-level robot state updates
            if self.task_scheduler and hasattr(self.task_scheduler, 'monitor_collector'):
                self.task_scheduler.monitor_collector = self.monitor_collector

            # Start HTTP dashboard
            self.http_dashboard = HTTPDashboard(self.monitor_collector, port=dashboard_port)
            self.http_dashboard.start()

            logger.info(f"Monitoring dashboard initialized on port {dashboard_port}")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring dashboard: {e}", exc_info=True)
            self.monitor_collector = None
            self.http_dashboard = None

        # Backfill monitor_collector into interactive selector (init order: selector before monitoring)
        if self.monitor_collector is not None and self.interactive_selector is not None:
            self.interactive_selector._monitor_collector = self.monitor_collector
            self.interactive_selector._robot = self.robot
            logger.info("Interactive selector connected to MonitorCollector")

    def _wait_for_input(
        self,
        terminal_prompt: str,
        prompt_data: dict,
        timeout: float = 30.0,
    ) -> str | None:
        """Wait for input from terminal OR dashboard frontend via select() multiplexing.

        Publishes prompt_data to MonitorCollector (renders buttons on dashboard),
        then select()s on both sys.stdin and the command pipe.  Whichever fires
        first wins — terminal readline or frontend button click.

        Args:
            terminal_prompt: Text prompt printed to terminal.
            prompt_data: Structured prompt for dashboard frontend.
                {"type": str, "message": str, "options": [...], "timeout_default": str}
            timeout: Seconds until auto-select timeout_default (0 = no timeout).

        Returns:
            User input string, or None on error/exit.
        """
        mc = self.monitor_collector
        if mc is None:
            # No dashboard — fall back to plain input()
            try:
                return input(terminal_prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return None

        mc.set_pending_prompt({**prompt_data, "timeout": timeout})
        pipe_fd = mc.command_pipe_r
        stdin_fd = sys.stdin.fileno()
        deadline = time.time() + timeout if timeout > 0 else None

        sys.stdout.write(terminal_prompt)
        sys.stdout.flush()

        # Proactive Modbus refresh before select loop.
        if self.robot:
            try:
                self.robot.get_observation()
            except Exception:
                pass

        last_keepalive = time.time()
        while True:
            wait = max(0.1, deadline - time.time()) if deadline else 0.5
            if deadline and time.time() >= deadline:
                default = prompt_data.get("timeout_default", "")
                sys.stdout.write(f"\n[timeout] auto-selecting: {default}\n")
                sys.stdout.flush()
                mc.clear_pending_prompt()
                return default

            try:
                readable, _, _ = select.select([stdin_fd, pipe_fd], [], [], wait)
            except (ValueError, OSError):
                break

            for fd in readable:
                if fd == stdin_fd:
                    line = sys.stdin.readline()
                    mc.clear_pending_prompt()
                    return line.strip() if line else None
                elif fd == pipe_fd:
                    os.read(pipe_fd, 256)
                    cmd = mc.get_last_command()
                    if cmd is not None:
                        sys.stdout.write(f"\n[frontend] {cmd}\n")
                        sys.stdout.flush()
                        mc.clear_pending_prompt()
                        return cmd

            # Gripper Modbus keepalive: prevent connection timeout during long prompts.
            # The JodellGripper's C++ Modbus library times out if no commands are sent
            # for several seconds.  A get_observation() pings all hardware interfaces.
            if self.robot and time.time() - last_keepalive > 2.0:
                try:
                    self.robot.get_observation()
                except Exception:
                    pass
                last_keepalive = time.time()

        mc.clear_pending_prompt()
        return None

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

    def _check_emergency_stop(self, observation: dict[str, Any], action: dict[str, float], task_name: str = None) -> RecoveryAction | None:
        """Check if emergency stop should be triggered and handle recovery.

        Args:
            observation: Current observation from robot.
            action: Current action being executed.
            task_name: Name of the current task (optional).

        Returns:
            RecoveryAction if emergency stop was triggered, None otherwise.
        """
        if self.emergency_controller is None:
            return None

        # Check for dangerous action
        is_dangerous, reason = self.emergency_controller.check_action_danger(action, observation)

        if is_dangerous:
            logger.warning(f"Dangerous action detected: {reason.value if reason else 'unknown'}")

            # Trigger emergency stop (without auto-rollback)
            self.emergency_controller.trigger_stop(
                reason=reason or StopReason.DANGEROUS_ACTION,
                auto_rollback=False  # We'll handle rollback after user selection
            )

            # Publish emergency prompt to dashboard if monitoring is active
            mc = self.monitor_collector
            pipe_fd = mc.command_pipe_r if mc else None
            if mc:
                mc.set_pending_prompt({
                    "type": "recovery",
                    "message": f"EMERGENCY STOP: {reason.value if reason else 'dangerous action'}",
                    "options": [
                        {"key": "1", "label": "Stop program completely"},
                        {"key": "2", "label": "Rollback and continue"},
                        {"key": "3", "label": "Rollback and retry with new model"},
                    ],
                    "timeout_default": "2",
                })

            # Prompt user for recovery action (terminal + dashboard)
            recovery_action = self.emergency_controller.prompt_recovery_action(
                task_name, pipe_fd=pipe_fd, robot=self.robot
            )

            # Check for frontend command override
            if mc:
                cmd = mc.get_last_command()
                if cmd:
                    action_map = {"1": RecoveryAction.STOP_PROGRAM,
                                  "2": RecoveryAction.ROLLBACK_AND_CONTINUE,
                                  "3": RecoveryAction.ROLLBACK_AND_RETRY_MODEL}
                    recovery_action = action_map.get(cmd, recovery_action)
                mc.clear_pending_prompt()

            # Handle the selected recovery action
            return self._handle_recovery_action(recovery_action, task_name)

        return None

    def _handle_recovery_action(self, recovery_action: RecoveryAction, task_name: str = None) -> RecoveryAction:
        """Handle the user-selected recovery action.

        Args:
            recovery_action: The recovery action selected by user.
            task_name: Name of the task that was interrupted.

        Returns:
            The recovery action for further processing by caller.
        """
        if recovery_action == RecoveryAction.STOP_PROGRAM:
            logger.info("User selected to stop the program")
            # Stop the program
            self.is_running = False

        elif recovery_action == RecoveryAction.ROLLBACK_AND_CONTINUE:
            logger.info("User selected to rollback and continue")
            # Get suggested rollback steps
            steps = self.emergency_controller.get_suggested_rollback_steps()
            logger.info(f"Rolling back {steps} steps...")

            # Execute rollback
            success = self.emergency_controller.rollback(steps=steps, confirm_before_resume=False)
            if success:
                logger.info("Rollback complete, resuming task")
                # Clear stop state to allow continuation
                self.emergency_controller.resume()
                # Reset policy executor to clear action queue and temporal ensemble state
                # This prevents large corrective actions after rollback
                if self.local_executor is not None:
                    self.local_executor.reset()
                    logger.debug("Policy executor reset after rollback")

        elif recovery_action == RecoveryAction.ROLLBACK_AND_RETRY_MODEL:
            logger.info("User selected to rollback and retry with new model")
            # Get suggested rollback steps
            steps = self.emergency_controller.get_suggested_rollback_steps()
            logger.info(f"Rolling back {steps} steps...")

            # Execute rollback
            success = self.emergency_controller.rollback(steps=steps, confirm_before_resume=False)
            if success:
                logger.info("Rollback complete, ready for new model selection")
                # Clear stop state
                self.emergency_controller.resume()
                # Reset policy executor to clear action queue and temporal ensemble state
                # This prevents large corrective actions after rollback
                if self.local_executor is not None:
                    self.local_executor.reset()
                    logger.debug("Policy executor reset before model switch")

                # Prompt for alternative model
                new_model_path = self._prompt_alternative_model(task_name)
                if new_model_path:
                    # Update task with new model path
                    # This will be handled by the task scheduler
                    logger.info(f"New model selected: {new_model_path}")
                    return recovery_action
                else:
                    logger.info("No new model selected, stopping")
                    return RecoveryAction.STOP_PROGRAM

        return recovery_action

    def _prompt_alternative_model(self, task_name: str = None, timeout: float = 30.0) -> str | None:
        """Prompt user to select an alternative model for task retry.

        Supports both terminal input and dashboard frontend buttons via
        _wait_for_input / select() multiplexing.

        Args:
            task_name: Name of the task that failed.
            timeout: Maximum time to wait for user input.

        Returns:
            Path to the selected model, or None if user cancelled.
        """
        print("\n" + "=" * 60)
        print("SELECT ALTERNATIVE MODEL")
        print("=" * 60)
        if task_name:
            print(f"Task: {task_name}")
        print("")
        print("Available models:")
        print("  1 - Default model (from config)")
        print("  2 - Enter custom model path")
        print("  0 - Cancel (stop program)")
        print("=" * 60)
        print(f"Auto-selecting default model in {timeout:.0f} seconds...", flush=True)

        user_input = self._wait_for_input(
            terminal_prompt=">>> ",
            prompt_data={
                "type": "model_selection",
                "message": f"Select model for: {task_name or 'unknown task'}",
                "options": [
                    {"key": "1", "label": "Default model (from config)"},
                    {"key": "2", "label": "Enter custom model path"},
                    {"key": "0", "label": "Cancel (stop program)"},
                ],
                "timeout_default": "1",
            },
            timeout=timeout,
        )

        # Timeout / None → use default
        if not user_input:
            user_input = "1"

        try:
            if user_input == "1":
                if task_name:
                    for task in self.config.tasks:
                        if task.name == task_name or task.name == f"{task_name}_retry":
                            logger.info(f"Using default model: {task.policy_path}")
                            return task.policy_path
                return None
            elif user_input == "2":
                print("Enter model path: ", flush=True)
                custom_path = self._wait_for_input(
                    terminal_prompt="",
                    prompt_data={
                        "type": "model_selection",
                        "message": "Enter custom model path",
                        "options": [
                            {"key": "__cancel__", "label": "Cancel"},
                        ],
                        "timeout_default": "__cancel__",
                    },
                    timeout=30.0,
                )
                if custom_path and custom_path != "__cancel__":
                    logger.info(f"Using custom model: {custom_path}")
                    return custom_path
                return None
            elif user_input == "0":
                return None
            else:
                print(f"Invalid input '{user_input}', using default model")
                return None

        except (KeyboardInterrupt, EOFError):
            logger.info("Input interrupted, cancelling model selection")
            return None

    def run(self) -> ExecutionSummary:
        """Execute the complete task sequence with optional cycle support.

        Returns:
            ExecutionSummary with results for all tasks.
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize orchestrator")

        self.is_running = True
        self.total_collision_count = 0

        # Get cycle configuration
        max_cycles = getattr(self.config, 'max_cycles', 1)
        cycle_delay = getattr(self.config, 'cycle_delay', 2.0)
        enable_cycle_prompt = getattr(self.config, 'enable_cycle_prompt', True)

        logger.info(f"Starting task sequence with {len(self.config.tasks)} tasks")
        if max_cycles > 1:
            logger.info(f"Cycle mode enabled: max_cycles={max_cycles}")
        elif max_cycles == -1:
            logger.info(f"Infinite cycle mode enabled (press Ctrl+C to stop)")

        cycle_count = 0
        all_results = []

        try:
            # Execute tasks with cycle support
            while True:
                cycle_count += 1
                self.current_cycle = cycle_count
                self.total_cycles = max_cycles

                # Check cycle limit
                if max_cycles != -1 and cycle_count > max_cycles:
                    logger.info(f"Completed {max_cycles} cycles, stopping")
                    break

                # Log cycle start
                if max_cycles > 1 or max_cycles == -1:
                    logger.info(f"=== Starting Cycle {cycle_count} ===")
                    self._add_monitor_event("info", "Orchestrator",
                        f"Cycle {cycle_count}/{max_cycles if max_cycles != -1 else '∞'} started")

                # Reset task index for new cycle
                if self.interactive_selector is not None:
                    self.interactive_selector.current_task_index = 0

                # Execute one cycle
                summary = self._execute_with_safety()
                all_results.extend(summary.task_results)

                # Check for fatal failure - don't continue cycles
                if summary.overall_success == False and any(
                    r.status == TaskStatus.FATAL_FAILURE for r in summary.task_results
                ):
                    logger.error("Fatal failure occurred, stopping cycles")
                    break

                # Cycle complete prompt (interactive mode)
                if max_cycles > 1 or max_cycles == -1:
                    logger.info(f"=== Cycle {cycle_count} Completed ===")
                    logger.info(f"  Completed: {summary.completed_tasks}, Failed: {summary.failed_tasks}")
                    self._add_monitor_event("info", "Orchestrator",
                        f"Cycle {cycle_count} done — {summary.completed_tasks} ok, {summary.failed_tasks} failed")

                    # Prompt for next cycle if interactive
                    if enable_cycle_prompt and self.config.enable_interactive_mode:
                        try:
                            prompt_msg = f"\n循环 {cycle_count} 完成。"
                            if max_cycles != -1:
                                prompt_msg += f" 还有 {max_cycles - cycle_count} 个循环待执行。"
                            prompt_msg += "\n按 Enter 继续下一循环，输入 'q' 退出: "

                            remaining = max_cycles - cycle_count if max_cycles != -1 else "?"
                            user_input = self._wait_for_input(
                                terminal_prompt=prompt_msg,
                                prompt_data={
                                    "type": "cycle_prompt",
                                    "message": f"循环 {cycle_count} 完成 ({remaining} 剩余)",
                                    "options": [
                                        {"key": "", "label": "继续下一循环"},
                                        {"key": "q", "label": "退出"},
                                    ],
                                    "timeout_default": "",
                                },
                                timeout=0,  # no timeout for cycle prompt
                            )
                            if user_input and user_input.lower() == 'q':
                                logger.info("User requested to stop cycles")
                                break
                        except (EOFError, KeyboardInterrupt):
                            logger.info("User interrupted, stopping cycles")
                            break

                    # Delay between cycles
                    if cycle_delay > 0:
                        logger.info(f"Waiting {cycle_delay}s before next cycle...")
                        import time
                        time.sleep(cycle_delay)

                # For single cycle mode, exit after one execution
                if max_cycles == 1:
                    break

            # Build final summary
            if all_results:
                completed = sum(1 for r in all_results if r.status == TaskStatus.COMPLETED)
                failed = sum(1 for r in all_results if r.status == TaskStatus.FAILED)
                fatal = sum(1 for r in all_results if r.status == TaskStatus.FATAL_FAILURE)

                return ExecutionSummary(
                    total_tasks=len(self.config.tasks) * cycle_count,
                    completed_tasks=completed,
                    failed_tasks=failed + fatal,
                    skipped_tasks=0,
                    total_duration=sum(r.duration for r in all_results),
                    task_results=all_results,
                    overall_success=(fatal == 0 and completed > 0),
                    collision_count=self.total_collision_count,
                    total_retries=sum(r.attempts - 1 for r in all_results),
                )

            return summary

        finally:
            self.is_running = False
            self._cleanup()

    def _execute_with_safety(self) -> ExecutionSummary:
        """Execute task sequence with collision monitoring and interactive selection."""
        results = []

        has_selector = self.interactive_selector is not None
        logger.warning(
            f"[DIAG] _execute_with_safety: interactive_selector={has_selector}, "
            f"task_count={len(self.config.tasks)}"
        )

        # Use while loop to properly handle interactive task selection
        # The interactive_selector maintains the current task index
        while True:
            # Get current task from interactive selector
            if self.interactive_selector is not None:
                # Non-blocking poll for dashboard pause/resume commands.
                # When paused (INTERACTIVE mode), prompt_next_task() below
                # will block and let the operator choose.  When resumed
                # (AUTOMATIC mode), it returns immediately.
                if self.interactive_selector.check_pause_request():
                    continue  # re-evaluate with new mode

                # Use the selector's task list (includes builtin skills) for
                # index lookups — current_idx is relative to the selector's
                # combined list, NOT self.config.tasks (YAML-only).
                selector_tasks = self.interactive_selector.config_tasks
                current_idx = self.interactive_selector.current_task_index

                # Check if we've reached the end of the task list
                if current_idx >= len(selector_tasks):
                    logger.info("All tasks completed or end of task list reached")
                    break

                task = selector_tasks[current_idx]

                # Prompt user for task selection
                selection = self.interactive_selector.prompt_next_task()

                # Handle exit request
                if selection.exit_requested:
                    logger.info("User requested exit, stopping task sequence")
                    break

                # Handle reset request
                if selection.reset_requested:
                    logger.info("User requested robot reset to zero position")
                    try:
                        # Check if robot has reset_to_zero method
                        if hasattr(self.robot, 'reset_to_zero'):
                            # Use configured reset duration and positions
                            duration = getattr(self.config, 'reset_duration', 3.0)
                            target_positions = getattr(self.config, 'reset_positions', {})
                            self.robot.reset_to_zero(duration=duration, target_positions=target_positions if target_positions else None)
                        else:
                            logger.warning("Robot does not support reset_to_zero method")
                    except Exception as e:
                        logger.error(f"Reset failed: {e}")
                    continue  # Return to selection prompt after reset

                # Handle custom task creation
                if selection.custom_task_name is not None:
                    logger.info(f"Custom task requested: {selection.custom_task_name}")
                    # Would need to execute custom task here
                    continue

                # Handle specific task selection
                if selection.selected_task:
                    # Find and execute selected task (search selector_tasks
                    # which includes both builtin skills and YAML tasks)
                    for t in selector_tasks:
                        if t.name.lower() == selection.selected_task.lower():
                            task = t
                            current_idx = selector_tasks.index(t)
                            # Update the selector's index to match
                            self.interactive_selector.current_task_index = current_idx
                            break
                    logger.info(f"User selected task: {task.name} (index {current_idx})")
                else:
                    # User selected "Execute next task in sequence"
                    # Use the current task from selector
                    logger.info(f"Executing next task in sequence: {task.name} (index {current_idx})")

                # Note: Index increment is now AFTER task execution, based on result
            else:
                # No interactive selector, use indexed while loop with branch support
                logger.warning("[DIAG] Entering non-interactive while loop")
                task_list = self.config.tasks
                idx = 0
                while idx < len(task_list):
                    task = task_list[idx]
                    if not task.enabled:
                        logger.info(f"Skipping disabled task: {task.name}")
                        idx += 1
                        continue

                    logger.info(f"Executing task {idx + 1}/{len(task_list)}: {task.name}")

                    # Execute the task
                    result = self._execute_single_task(task)
                    results.append(result)

                    # Update collision count
                    if result.collision_detected:
                        self.total_collision_count += 1

                    # Check if we should continue
                    if result.status == TaskStatus.FATAL_FAILURE:
                        logger.error(f"Fatal failure in task {task.name}, aborting sequence")
                        break
                    if result.status == TaskStatus.FAILED and task.stop_on_failure:
                        logger.error(f"Task {task.name} failed — stopping cycle")
                        break
                    if result.status == TaskStatus.FAILED and not task.stop_on_failure:
                        logger.warning(f"Task {task.name} failed but stop_on_failure=false — continuing")

                    if self.total_collision_count >= self.config.max_total_collisions:
                        logger.error(
                            f"Maximum collision count reached ({self.config.max_total_collisions}), aborting"
                        )
                        break

                    # Branch routing (same logic as interactive path)
                    logger.warning(
                        f"[DIAG] result processing: task={task.name}, "
                        f"next_task='{result.next_task}', cycle_end={task.cycle_end}"
                    )
                    if result.next_task:
                        found = False
                        for j, t in enumerate(task_list):
                            if t.name == result.next_task:
                                idx = j
                                logger.warning(
                                    f"Branch: {task.name} → {result.next_task} (index {j})"
                                )
                                found = True
                                break
                        if not found:
                            logger.error(
                                f"Branch target '{result.next_task}' not found "
                                f"in task list — falling back to sequential advance"
                            )
                            idx += 1
                    elif task.cycle_end:
                        logger.info(f"Cycle end marker at task {task.name}")
                        break  # end current cycle, loop back via max_cycles
                    else:
                        idx += 1
                # Exit while loop when no interactive selector
                break

            # Skip disabled tasks
            if not task.enabled:
                logger.info(f"Skipping disabled task: {task.name}")
                # Advance index for disabled tasks
                self.interactive_selector.current_task_index += 1
                continue

            logger.info(f"Executing task {current_idx + 1}/{len(self.config.tasks)}: {task.name}")

            # Execute the task
            result = self._execute_single_task(task)

            results.append(result)

            # Record task execution
            if self.interactive_selector is not None:
                self.interactive_selector.record_task_execution(task.name)

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

            # P0 FIX: Only advance task index if task completed successfully
            # If task failed (e.g., after emergency stop with rollback), keep index
            # so user can retry the same task or choose a different action
            if result.status == TaskStatus.COMPLETED:
                # Conditional branching: if the task returned next_task,
                # jump to that task by name instead of sequential advance.
                logger.warning(
                    f"[DIAG] interactive result: task={task.name}, "
                    f"next_task='{result.next_task}', cycle_end={task.cycle_end}"
                )
                if result.next_task and self.interactive_selector is not None:
                    selector_tasks = self.interactive_selector.config_tasks
                    found = False
                    for idx, t in enumerate(selector_tasks):
                        if t.name == result.next_task:
                            self.interactive_selector.current_task_index = idx
                            logger.warning(
                                f"Branch: {task.name} → {result.next_task} "
                                f"(index {idx})"
                            )
                            found = True
                            break
                    if not found:
                        logger.error(
                            f"Branch target '{result.next_task}' not found "
                            f"in task list — falling back to sequential advance"
                        )
                        self.interactive_selector.current_task_index += 1
                else:
                    if task.cycle_end:
                        logger.info(f"Cycle end marker at task {task.name}")
                        break  # end current cycle via while loop
                    self.interactive_selector.current_task_index += 1
                    # Skip builtin skills in AUTOMATIC mode — they are manually
                    # selectable skills, not part of the automatic task sequence.
                    if (
                        self._builtin_task_names
                        and self.interactive_selector._execution_mode == ExecutionMode.AUTOMATIC
                    ):
                        selector_tasks = self.interactive_selector.config_tasks
                        while self.interactive_selector.current_task_index < len(selector_tasks):
                            t = selector_tasks[self.interactive_selector.current_task_index]
                            if t.name in self._builtin_task_names:
                                logger.info(
                                    f"Skipping builtin skill '{t.name}' in AUTOMATIC mode"
                                )
                                self.interactive_selector.current_task_index += 1
                            else:
                                break
                logger.info(f"Task {task.name} completed, moving to next task")
            elif result.status == TaskStatus.FAILED:
                # Task failed but not fatal - keep index to allow retry
                logger.info(f"Task {task.name} failed, keeping index for potential retry")
                # User can manually advance via interactive menu, or retry the same task
            # FATAL_FAILURE already broke out of the loop above
            # SKIPPED is handled before this point

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

    def _execute_single_task(self, task: TaskConfig) -> TaskResult:
        """Execute a single task with all necessary setup and monitoring.

        Supports policy tasks, AGV navigation tasks, and position tasks.

        Args:
            task: The task configuration to execute.

        Returns:
            TaskResult with execution outcome.
        """
        # Override settings if specified
        original_max_retries = task.max_retries
        original_max_duration = task.max_duration

        if self.config.override_max_retries is not None:
            task.max_retries = self.config.override_max_retries
        if self.config.override_max_duration is not None:
            task.max_duration = self.config.override_max_duration

        # Apply global speed_multiplier (clamped to safe range)
        multiplier = max(0.25, min(4.0, self.config.speed_multiplier))
        if multiplier != 1.0:
            task.speed_multiplier = multiplier

        # Handle AGV tasks separately
        if task.task_type == "agv":
            self._add_monitor_event("info", task.name, "AGV navigation started")
            result = self._execute_agv_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)"
                + (f" — {result.error_message}" if result.error_message else ""))
            # Restore original settings
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle position tasks (direct joint position movement)
        if task.task_type == "position":
            self._add_monitor_event("info", task.name, "Moving to target position")
            result = self._execute_position_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)"
                + (f" — {result.error_message}" if result.error_message else ""))
            # Restore original settings
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle position_sequence tasks (multi-step position movement)
        if task.task_type == "position_sequence":
            self._add_monitor_event("info", task.name, f"Executing {len(task.steps)} steps")
            result = self._execute_position_sequence_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)"
                + (f" — {result.error_message}" if result.error_message else ""))
            # Restore original settings
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle visual_align tasks (AprilTag-guided AGV fine alignment)
        if task.task_type == "visual_align":
            self._add_monitor_event("info", task.name, "Visual alignment started")
            result = self._execute_visual_align_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)"
                + (f" — {result.error_message}" if result.error_message else ""))
            # Restore original settings
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle classify tasks (workpiece identification for branching)
        if task.task_type == "classify":
            self._add_monitor_event("info", task.name, "Classify started")
            result = self._execute_classify_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)"
                + (f" — {result.error_message}" if result.error_message else ""))
            logger.warning(
                f"[DIAG] _execute_single_task classify: next_task='{result.next_task}', "
                f"status={result.status.value}"
            )
            # Restore original settings
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle system_command tasks (shell command execution)
        if task.task_type == "system_command":
            self._add_monitor_event("info", task.name, f"Running: {task.command}")
            result = self._execute_system_command_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)")
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle parallel tasks (concurrent sub-tasks)
        if task.task_type == "parallel":
            self._add_monitor_event("info", task.name, f"Parallel: {len(task.parallel_tasks)} sub-tasks")
            result = self._execute_parallel_task(task)
            self._add_monitor_event(
                "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
                f"{result.status.value} ({result.duration:.1f}s)"
                + (f" — {result.error_message}" if result.error_message else ""))
            task.max_retries = original_max_retries
            task.max_duration = original_max_duration
            if hasattr(task, 'speed_multiplier'):
                delattr(task, 'speed_multiplier')
            return result

        # Handle policy tasks (existing logic)
        # Apply speed_multiplier to policy task timeout
        if multiplier != 1.0:
            task.max_duration = max(1.0, task.max_duration / multiplier)

        # Switch cameras for this task
        self._switch_cameras_for_task(task)
        self._add_monitor_event("info", task.name, "Policy task started")

        # Set completion detector for this task
        print(f"[_execute_single_task] Setting completion detector for task: {task.name}")
        print(f"[_execute_single_task] Available detectors: {list(self.completion_detectors.keys())}")
        detector = self.completion_detectors.get(task.name)
        self.task_scheduler.completion_detector = detector

        # Also set on the underlying scheduler if this is an AdaptiveTaskScheduler
        if hasattr(self.task_scheduler, 'scheduler'):
            self.task_scheduler.scheduler.completion_detector = detector
            print(f"[_execute_single_task] Also set on underlying scheduler")

        print(f"[_execute_single_task] Detector assigned: {self.task_scheduler.completion_detector}")

        # Execute task
        # Check if using adaptive scheduler
        if hasattr(self.task_scheduler, 'execute_task_adaptive'):
            result = self.task_scheduler.execute_task_adaptive(
                task,
                collision_detector=self.collision_detector,
                collision_handler=self.collision_handler,
                state_monitor=self.state_monitor,
            )

            # Handle emergency stop with new model retry
            if hasattr(result, 'retry_with_new_model') and result.retry_with_new_model:
                # User wants to retry with a new model
                new_model = self._prompt_alternative_model(task.name)
                if new_model:
                    # Create a new task config with the new model
                    from copy import deepcopy
                    retry_task = deepcopy(task)
                    retry_task.policy_path = new_model
                    retry_task.name = f"{task.name}_retry"

                    logger.info(f"Retrying task with new model: {new_model}")
                    result = self.task_scheduler.execute_task_adaptive(
                        retry_task,
                        collision_detector=self.collision_detector,
                        collision_handler=self.collision_handler,
                        state_monitor=self.state_monitor,
                    )
                else:
                    # No new model selected, mark as failed
                    result.status = TaskStatus.FATAL_FAILURE
                    result.error_message = "Emergency stop: No alternative model selected"
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
        if hasattr(task, 'speed_multiplier'):
            delattr(task, 'speed_multiplier')

        self._add_monitor_event(
            "info" if result.status == TaskStatus.COMPLETED else "warn", task.name,
            f"{result.status.value} ({result.duration:.1f}s)"
            + (f" — {result.error_message}" if result.error_message else ""))

        return result

    def _execute_agv_task(self, task: TaskConfig) -> TaskResult:
        """Execute an AGV navigation task.

        Args:
            task: Task configuration with agv_config.

        Returns:
            TaskResult with execution outcome.
        """
        logger.info(f"Executing AGV task: {task.name}")

        if not self.agv_executor or not self.agv_controller:
            logger.warning("AGV executor not initialized, skipping task")
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.SKIPPED,
                duration=0.0,
                error_message="AGV not initialized",
                collision_detected=False,
                attempts=1,
            )

        if not self.agv_controller.is_connected():
            # Try to reconnect
            if not self.agv_controller.reconnect():
                return TaskResult(
                    task_name=task.name,
                    status=TaskStatus.FAILED,
                    duration=0.0,
                    error_message="AGV connection failed",
                    collision_detected=False,
                    attempts=1,
                )

        agv_config = task.agv_config
        if not agv_config:
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.FAILED,
                duration=0.0,
                error_message="No AGV config provided",
                collision_detected=False,
                attempts=1,
            )

        # Update monitoring with current robot state before AGV movement
        if self.monitor_collector is not None and self.robot:
            try:
                observation = self.robot.get_observation()
                task_info = {
                    "task_name": task.name,
                    "task_type": task.task_type,
                    "cycle": getattr(self, 'current_cycle', 0),
                    "total_cycles": getattr(self, 'total_cycles', 0),
                    "collision_count": self.total_collision_count,
                    "total_tasks": len(self.config.tasks),
                    "completed_tasks": getattr(task, '_completed_tasks', 0),
                    "failed_tasks": getattr(task, '_failed_tasks', 0),
                    "last_error": getattr(task, '_last_error', ""),
                }
                self.monitor_collector.update_robot_state(observation, {}, task_info)
            except Exception:
                pass

        # Apply speed_multiplier to AGV velocities and max_duration
        multiplier = getattr(task, 'speed_multiplier', 1.0)
        agv_max_duration = max(1.0, agv_config.max_duration / multiplier) if multiplier != 1.0 else agv_config.max_duration
        agv_vx = agv_config.translate_vx * multiplier if agv_config.translate_vx is not None and multiplier != 1.0 else agv_config.translate_vx
        agv_vy = agv_config.translate_vy * multiplier if agv_config.translate_vy is not None and multiplier != 1.0 else agv_config.translate_vy
        agv_vw = agv_config.turn_vw * multiplier if agv_config.turn_vw is not None and multiplier != 1.0 else agv_config.turn_vw

        # Execute AGV navigation
        agv_result = self.agv_executor.execute(
            task_name=task.name,
            target_station=agv_config.target_station,
            target_position=agv_config.target_position,
            translate_dist=agv_config.translate_dist,
            translate_vx=agv_vx,
            translate_vy=agv_vy,
            translate_mode=agv_config.translate_mode,
            turn_angle=agv_config.turn_angle,
            turn_vw=agv_vw,
            turn_mode=agv_config.turn_mode,
            max_duration=agv_max_duration,
            wait_for_arrival=agv_config.wait_for_arrival,
            arrival_timeout=agv_config.arrival_timeout,
            arrival_tolerance=agv_config.arrival_tolerance,
            check_arm_safe=agv_config.check_arm_safe_position,
            arm_safe_positions=agv_config.arm_safe_positions,
            arm_home_positions=agv_config.arm_home_positions,
            retry_on_timeout=agv_config.retry_on_timeout,
            retry_count=agv_config.retry_count,
            emergency_stop_on_error=agv_config.emergency_stop_on_error,
            angle_correction=agv_config.angle_correction,
            angle_correction_tolerance=agv_config.angle_correction_tolerance,
            angle_correction_reference=agv_config.angle_correction_reference,
        )

        # Convert to TaskResult
        result = create_task_result_from_agv_result(task.name, agv_result)

        logger.info(
            f"AGV task completed: {task.name} -> "
            f"status={result.status.value}, "
            f"station={agv_result.arrival_station}, "
            f"duration={result.duration:.1f}s"
        )

        return result

    def _execute_visual_align_task(self, task: TaskConfig) -> TaskResult:
        """Execute a visual alignment task — AprilTag-guided AGV fine positioning.

        Uses the head camera to detect an AprilTag marker, then iteratively
        aligns the AGV via turn + translate until within tolerance.

        Args:
            task: Task configuration with visual_align_config.

        Returns:
            TaskResult with execution outcome.
        """
        import time
        from lerobot.agent.visual_align import execute_visual_align
        from lerobot.tasks.config import VisualAlignConfig

        start_time = time.time()
        logger.info(f"Executing visual_align task: {task.name}")

        # Prerequisite checks
        if not self.robot:
            return TaskResult(
                task_name=task.name, status=TaskStatus.SKIPPED,
                duration=0.0, error_message="Robot not connected",
            )
        if not self.agv_controller or not self.agv_controller.is_connected():
            return TaskResult(
                task_name=task.name, status=TaskStatus.SKIPPED,
                duration=0.0, error_message="AGV not connected",
            )

        va_config = task.visual_align_config
        if va_config is None:
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=0.0, error_message="visual_align_config is None",
            )

        # Ensure head camera is enabled for detection
        self._switch_cameras_for_task(task)

        # Arm safety check before AGV movement
        if va_config.check_arm_safe_position and self.robot:
            safe_positions = va_config.arm_safe_positions
            if not safe_positions and hasattr(self, 'agv_executor'):
                # Inherit from global defaults
                safe_positions = self.agv_executor.default_arm_safe_positions

            current_pos = self.robot.get_current_position()
            deviations = {}
            for joint_name, threshold in (safe_positions or {}).items():
                if joint_name in current_pos:
                    actual = abs(current_pos[joint_name])
                    deviations[joint_name] = f"pos={current_pos[joint_name]:.1f}°, threshold={threshold:.1f}°"

            logger.info(f"Arm safety check: {deviations if deviations else 'no thresholds defined'}")

        # Update monitoring with current robot state before visual alignment
        if self.monitor_collector is not None and self.robot:
            try:
                observation = self.robot.get_observation()
                task_info = {
                    "task_name": task.name,
                    "task_type": task.task_type,
                    "cycle": getattr(self, 'current_cycle', 0),
                    "total_cycles": getattr(self, 'total_cycles', 0),
                    "collision_count": self.total_collision_count,
                    "total_tasks": len(self.config.tasks),
                    "completed_tasks": getattr(task, '_completed_tasks', 0),
                    "failed_tasks": getattr(task, '_failed_tasks', 0),
                    "last_error": getattr(task, '_last_error', ""),
                }
                self.monitor_collector.update_robot_state(observation, {}, task_info)
            except Exception:
                pass

        # Execute alignment
        success, message = execute_visual_align(
            robot=self.robot,
            agv_controller=self.agv_controller,
            config=va_config,
            logger=logger,
        )

        duration = time.time() - start_time
        status = TaskStatus.COMPLETED if success else TaskStatus.FAILED

        logger.info(
            f"Visual align {task.name}: {status.value} "
            f"(message: {message}, duration: {duration:.1f}s)"
        )

        return TaskResult(
            task_name=task.name,
            status=status,
            duration=duration,
            error_message=None if success else message,
        )

    def _execute_classify_task(self, task: TaskConfig) -> TaskResult:
        """Classify a workpiece using the head camera.

        Captures an image, runs the configured classifier, and returns
        a TaskResult whose ``next_task`` is set from ``task.next_tasks``
        mapped by the classification label.

        The orchestrator reads ``next_task`` to decide which task to
        execute next (branching).
        """
        logger.info(f"Executing classify task: {task.name}")
        start_time = time.time()

        if not self.robot:
            return TaskResult(
                task_name=task.name, status=TaskStatus.SKIPPED,
                duration=0.0, error_message="Robot not connected",
            )

        cc = task.classify_config
        if cc is None:
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=0.0, error_message="classify_config is None",
            )

        self._switch_cameras_for_task(task)

        # Capture image
        obs = self.robot.get_observation()
        img = obs.get("images", {}).get("head_cam")
        if img is None:
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=time.time() - start_time,
                error_message="No head_cam image",
            )

        import numpy as np
        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)

        # Build classifier
        classifier_kwargs = {
            "marker_id_map": cc.marker_id_map,
            "marker_family": cc.marker_family,
            "marker_size": cc.marker_size,
            "default_label": cc.default_label,
            "default_next_task": cc.default_next_task,
            # YOLO-specific (ignored by apriltag)
            "model_path": cc.model_path,
            "classes": cc.classes,
            "conf_threshold": cc.conf_threshold,
        }
        try:
            classifier = make_classifier(cc.method, **classifier_kwargs)

            # Retry loop for no_detection
            retry_attempts = 0
            max_retries = cc.retry_max_attempts if cc.retry_on_no_detect else 1
            result = None
            for retry_attempts in range(max_retries):
                result = classifier.classify(bgr)
                if result.label == "no_detection" and cc.retry_on_no_detect:
                    logger.warning(
                        f"Classify retry {retry_attempts + 1}/{max_retries}: "
                        f"no detection, waiting {cc.retry_wait_seconds}s ..."
                    )
                    if cc.retry_command:
                        import subprocess as _sp
                        parts = cc.retry_command.split()
                        if parts and parts[0] in ("espeak-ng", "espeak", "aplay", "speaker-test"):
                            try:
                                _sp.run(parts, shell=False, timeout=10)
                            except Exception:
                                pass
                    time.sleep(cc.retry_wait_seconds)
                    # Re-capture image
                    obs = self.robot.get_observation()
                    img = obs.get("images", {}).get("head_cam")
                    if img is not None:
                        bgr = img if img.dtype == np.uint8 else img.astype(np.uint8)
                else:
                    break

            if result is None:
                return TaskResult(
                    task_name=task.name, status=TaskStatus.FAILED,
                    duration=time.time() - start_time,
                    error_message="Classification failed after retries",
                )

            # No-detection even after retries: recovery or fail
            if result.label == "no_detection" and cc.retry_on_no_detect:
                if cc.recovery_task:
                    logger.warning(
                        f"Classify failed after {max_retries} retries — "
                        f"recovery → {cc.recovery_task}"
                    )
                    # Return COMPLETED with next_task pointing to recovery
                    return TaskResult(
                        task_name=task.name,
                        status=TaskStatus.COMPLETED,
                        duration=time.time() - start_time,
                        success=True,
                        next_task=cc.recovery_task,
                    )
                logger.error(
                    f"Classify failed: no_detection after {max_retries} retries — stopping cycle"
                )
                return TaskResult(
                    task_name=task.name, status=TaskStatus.FAILED,
                    duration=time.time() - start_time,
                    error_message=f"No workpiece detected after {max_retries} attempts",
                )

            # Apply label counter for alternating placements
            if cc.label_counter_enable and cc.counter_keywords:
                result.label = _counted_label(result.label, cc.counter_keywords, cc.counter_modulo)
                logger.warning(f"Classify counted label: {result.label}")

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=time.time() - start_time,
                error_message=f"Classification error: {e}",
            )

        # Route to next task based on label
        next_task = task.next_tasks.get(result.label, cc.default_next_task)
        label_str = f"label={result.label}"
        if next_task:
            label_str += f" → next_task={next_task}"
        logger.warning(
            f"Classify result: {label_str} (confidence={result.confidence:.2f})"
        )

        return TaskResult(
            task_name=task.name,
            status=TaskStatus.COMPLETED,
            duration=time.time() - start_time,
            success=True,
            next_task=next_task,
        )

    def _execute_parallel_task(self, task: TaskConfig) -> TaskResult:
        """Execute sub-tasks concurrently in separate threads.

        Each sub-task dict is parsed into a TaskConfig and dispatched to
        _execute_single_task in its own thread.  All threads run independently
        (AGV uses its own TCP socket, robot uses CAN bus — no shared resource).

        If any sub-task returns FATAL_FAILURE the AGV navigation is cancelled
        via cancel_navigation() to halt the other AGV sub-task gracefully.
        """
        import threading
        from lerobot.tasks.config import parse_task_dict

        logger.info(f"Parallel task '{task.name}': launching {len(task.parallel_tasks)} sub-tasks")

        # Parse sub-task dicts into TaskConfig objects
        sub_tasks = []
        named_positions = self.config.named_positions or {}
        default_arm_safe = self.config.agv_config.default_arm_safe_positions if self.config.agv_config else {}
        default_arm_home = self.config.agv_config.default_arm_home_positions if self.config.agv_config else {}

        for st_dict in task.parallel_tasks:
            try:
                st = parse_task_dict(st_dict, named_positions, default_arm_safe, default_arm_home)
                sub_tasks.append(st)
            except Exception as e:
                logger.error(f"Failed to parse parallel sub-task '{st_dict.get('name', '?')}': {e}")

        if not sub_tasks:
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=0.0, error_message="No valid sub-tasks to execute",
            )

        results = [None] * len(sub_tasks)
        fatal_event = threading.Event()

        def _run_subtask(idx: int, st: TaskConfig):
            try:
                results[idx] = self._execute_single_task(st)
            except Exception as e:
                results[idx] = TaskResult(
                    task_name=st.name, status=TaskStatus.FATAL_FAILURE,
                    duration=0.0, error_message=str(e),
                )
            # If fatal, signal other thread to cancel
            if results[idx] and results[idx].status == TaskStatus.FATAL_FAILURE:
                fatal_event.set()

        start_time = time.time()
        threads = []
        for i, st in enumerate(sub_tasks):
            t = threading.Thread(target=_run_subtask, args=(i, st), name=f"parallel_{st.name}")
            t.start()
            threads.append(t)
            logger.info(f"  Sub-task [{i}]: {st.name} ({st.task_type}) started")

        # Wait for all threads — if a fatal occurs kill AGV on any still-running
        for t in threads:
            remaining = max(0, task.max_duration - (time.time() - start_time))
            t.join(timeout=remaining)
            if fatal_event.is_set() and t.is_alive():
                logger.warning(f"  Parallel: fatal in another sub-task, cancelling AGV navigation")
                try:
                    self.agv_controller.cancel_navigation() if self.agv_controller else None
                except Exception:
                    pass
                t.join(timeout=5.0)

        elapsed = time.time() - start_time

        # Aggregate results
        completed = sum(1 for r in results if r and r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r and r.status in (TaskStatus.FAILED, TaskStatus.FATAL_FAILURE))
        fatal = sum(1 for r in results if r and r.status == TaskStatus.FATAL_FAILURE)

        if fatal > 0 or any(not r for r in results):
            status = TaskStatus.FATAL_FAILURE
            success = False
        elif failed > 0:
            status = TaskStatus.FAILED
            success = False
        else:
            status = TaskStatus.COMPLETED
            success = True

        msg = f"{completed}/{len(sub_tasks)} ok"
        if failed > 0:
            msg += f", {failed} failed"
        logger.info(f"Parallel task '{task.name}' done: {msg} ({elapsed:.1f}s)")

        errors = "; ".join(r.error_message for r in results if r and r.error_message)

        return TaskResult(
            task_name=task.name, status=status,
            duration=elapsed, success=success,
            error_message=errors,
        )

    def _execute_system_command_task(self, task: TaskConfig) -> TaskResult:
        """Execute a shell command task (e.g. TTS voice broadcast)."""
        import subprocess
        logger.info(f"Executing system_command: {task.command}")
        start_time = time.time()
        try:
            parts = task.command.split()
            if not parts or parts[0] not in ("espeak-ng", "espeak", "aplay", "speaker-test"):
                return TaskResult(
                    task_name=task.name, status=TaskStatus.FAILED,
                    duration=time.time() - start_time,
                    error_message=f"Command not in allowlist: {task.command}",
                )
            subprocess.run(parts, shell=False, timeout=task.max_duration,
                           capture_output=True)
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.COMPLETED,
                duration=time.time() - start_time,
                success=True,
            )
        except subprocess.TimeoutExpired:
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=time.time() - start_time,
                error_message=f"Command timed out after {task.max_duration}s",
            )
        except Exception as e:
            return TaskResult(
                task_name=task.name, status=TaskStatus.FAILED,
                duration=time.time() - start_time,
                error_message=str(e),
            )

    def _execute_position_task(self, task: TaskConfig) -> TaskResult:
        """Execute a position task - move joints directly to target positions.

        This task type is for moving the robot to specific joint positions
        without requiring a policy. Used for safety positions, home positions,
        etc.

        Args:
            task: Task configuration with target_joint_positions in completion_criteria.

        Returns:
            TaskResult with execution outcome.
        """
        import time
        import math

        logger.info(f"Executing position task: {task.name}")

        if not self.robot:
            logger.warning("Robot not connected, skipping task")
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.SKIPPED,
                duration=0.0,
                error_message="Robot not connected",
                collision_detected=False,
                attempts=1,
            )

        target_positions = task.completion_criteria.target_joint_positions
        if not target_positions:
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.FAILED,
                duration=0.0,
                error_message="No target_joint_positions provided",
                collision_detected=False,
                attempts=1,
            )

        tolerance = task.completion_criteria.position_tolerance  # degrees
        max_duration = task.max_duration
        control_frequency = getattr(self.config.robot_config, 'control_frequency', 30)
        dt = 1.0 / control_frequency

        # Apply speed_multiplier if set
        multiplier = getattr(task, 'speed_multiplier', 1.0)
        if multiplier != 1.0:
            dt = max(0.005, dt / multiplier)

        # Get current positions using the robot's helper method
        # SupreRobotFollower.get_current_position() returns {"joint_name": position} in degrees
        current_positions_dict = self.robot.get_current_position()

        # Get joint names from robot config
        joint_names = self.robot.observation_joint_names if hasattr(self.robot, 'observation_joint_names') else list(target_positions.keys())

        # Calculate movement duration based on max distance
        max_distance = 0.0
        for joint_name in joint_names:
            if joint_name in target_positions and joint_name in current_positions_dict:
                target = target_positions[joint_name]
                current = current_positions_dict[joint_name]
                distance = abs(target - current)
                max_distance = max(max_distance, distance)

        # Estimate time needed — speed constant and floor scale with multiplier
        speed_deg_per_s = 30.0 * max(multiplier, 1.0)
        duration_floor = max(1.0, 3.0 / max(multiplier, 1.0))
        estimated_duration = max_distance / speed_deg_per_s + (1.0 / max(multiplier, 1.0))
        actual_duration = min(max_duration, max(estimated_duration, duration_floor))

        logger.info(f"Moving to target positions, estimated duration: {actual_duration:.1f}s")
        logger.debug(f"Target positions: {target_positions}")

        # Execute smooth movement using interpolated commands
        start_time = time.time()
        step_count = 0
        total_steps = int(actual_duration * control_frequency)

        success = True
        while time.time() - start_time < actual_duration:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / actual_duration)

            # Use smooth interpolation (ease-in-out)
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))

            # Calculate intermediate positions
            target_action = {}
            for joint_name in joint_names:
                if joint_name in target_positions and joint_name in current_positions_dict:
                    current = current_positions_dict[joint_name]
                    target = target_positions[joint_name]
                    # Interpolate: current + (target - current) * progress
                    intermediate = current + (target - current) * smooth_progress
                    # Add .pos suffix for send_action format
                    target_action[f"{joint_name}.pos"] = intermediate

            # Send action to robot (dict format: {"joint_name.pos": value})
            if target_action:
                self.robot.send_action(target_action)

            step_count += 1

            # Get observation for collision and monitoring
            observation = self.robot.get_observation()

            # Update real-time monitoring dashboard (non-blocking)
            if self.monitor_collector is not None:
                try:
                    task_info = {
                        "task_name": task.name,
                        "task_type": task.task_type,
                        "cycle": getattr(self, 'current_cycle', 0),
                        "total_cycles": getattr(self, 'total_cycles', 0),
                        "collision_count": self.total_collision_count,
                        "total_tasks": len(self.config.tasks),
                        "completed_tasks": getattr(task, '_completed_tasks', 0),
                        "failed_tasks": getattr(task, '_failed_tasks', 0),
                        "last_error": getattr(task, '_last_error', ""),
                    }
                    self.monitor_collector.update_robot_state(observation, target_action, task_info)
                except Exception:
                    pass  # Never let monitoring errors affect the control loop

            # Check for collision
            if self.collision_detector:
                collision_result = self.collision_detector.check_collision(observation, target_action)
                if collision_result.is_detected:
                    logger.warning(f"Collision detected during position task {task.name}")
                    self._add_monitor_event("warn", task.name, "Collision detected — stopping")
                    success = False
                    break

            # Pace the control loop to the configured control frequency
            time.sleep(dt)

        # Check if reached target positions
        final_positions_dict = self.robot.get_current_position()

        all_reached = True
        for joint_name, target in target_positions.items():
            if joint_name in final_positions_dict:
                final = final_positions_dict[joint_name]
                if abs(final - target) > tolerance:
                    all_reached = False
                    logger.debug(f"Joint {joint_name}: target={target:.1f}, actual={final:.1f}, diff={abs(final-target):.1f}")

        duration = time.time() - start_time
        # Use COMPLETED status (SUCCESS is not a valid TaskStatus enum value)
        status = TaskStatus.COMPLETED if all_reached and success else TaskStatus.FAILED

        logger.info(
            f"Position task completed: {task.name} -> "
            f"status={status.value}, "
            f"duration={duration:.1f}s, "
            f"reached={all_reached}"
        )

        return TaskResult(
            task_name=task.name,
            status=status,
            duration=duration,
            error_message=None if all_reached else "Did not reach target positions",
            collision_detected=not success,
            attempts=1,
            # TaskResult.__post_init__ calculates duration = end_time - start_time.
            # Without these, it resets duration to 0.0 regardless of the passed value.
            start_time=start_time,
            end_time=start_time + duration,
        )

    def _execute_position_sequence_task(self, task: TaskConfig) -> TaskResult:
        """Execute a position_sequence task - move joints through multiple positions sequentially.

        Non-overlap steps are executed independently via _execute_position_task().
        Consecutive overlap_next steps are merged into a single continuous trajectory
        — one constant-speed path through all waypoints, no re-interpolation at
        boundaries, no control-loop gap between targets.

        Args:
            task: Task configuration with steps list containing PositionSequenceStep objects.

        Returns:
            TaskResult with aggregated execution outcome.
        """
        import time
        import math

        logger.info(f"Executing position_sequence task: {task.name} ({len(task.steps)} steps)")

        joint_names = getattr(self.robot, 'observation_joint_names', None)
        control_frequency = getattr(self.config.robot_config, 'control_frequency', 30)
        dt = 1.0 / control_frequency
        speed_multiplier = getattr(task, 'speed_multiplier', 1.0)
        if speed_multiplier != 1.0:
            dt = max(0.005, dt / speed_multiplier)

        from lerobot.tasks.config import CompletionCriteria

        start_time = time.time()
        i = 0
        while i < len(task.steps):
            step = task.steps[i]
            step_name = step.name or f"step_{i + 1}"

            # ── Merged overlap chain ──────────────────────────────────────
            # Group consecutive overlap_next steps into one continuous
            # trajectory.  The chain stops BEFORE a non-overlap step (or at
            # end of list).  The non-overlap step runs as standalone.
            if step.overlap_next and i + 1 < len(task.steps):
                chain_start = i
                while i < len(task.steps) and task.steps[i].overlap_next:
                    i += 1
                # i now points to the first step WITHOUT overlap_next (or past end)
                chain_steps = task.steps[chain_start:i]
                chain_names = [s.name or f"step_{chain_start + k + 1}" for k, s in enumerate(chain_steps)]

                logger.info(
                    f"  Continuous chain steps {chain_start + 1}-{chain_start + len(chain_steps)}: "
                    f"{' → '.join(chain_names)}"
                )

                # ── Build unified trajectory ─────────────────────────────
                # Catmull-Rom spline through waypoints → C¹ continuous,
                # no sharp corners at via-points.  Control-point padding
                # duplicates endpoints so the curve starts/ends exactly at
                # the first/last waypoint.
                #
                # waypoints: [current_pos, W1, W2, ..., Wn]
                # padded:    [current_pos, current_pos, W1, W2, ..., Wn, Wn]
                current_pos = self.robot.get_current_position()
                waypoints = [current_pos]
                for cs in chain_steps:
                    wp = {}
                    for jn in joint_names:
                        wp[jn] = cs.position.get(jn, current_pos.get(jn, 0.0))
                    waypoints.append(wp)

                # Pad for Catmull-Rom boundary conditions
                padded = [waypoints[0], waypoints[0]] + waypoints + [waypoints[-1]]

                # Helper: Catmull-Rom for a scalar
                def _cr(p0, p1, p2, p3, t):
                    t2 = t * t
                    t3 = t2 * t
                    return 0.5 * (
                        (2.0 * p1) +
                        (-p0 + p2) * t +
                        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                    )

                # Chord-length parameterization
                segment_lengths = []
                for k in range(len(waypoints) - 1):
                    max_d = 0.0
                    for jn in joint_names:
                        max_d = max(max_d, abs(waypoints[k + 1][jn] - waypoints[k][jn]))
                    segment_lengths.append(max_d)

                total_length = sum(segment_lengths)
                if total_length < 0.01:
                    logger.info(f"  Chain has zero-length path, skipping")
                    i = chain_start + len(chain_steps)
                    continue

                cum_lengths = [0.0]
                for sl in segment_lengths:
                    cum_lengths.append(cum_lengths[-1] + sl)

                speed_deg_per_s = 30.0 * max(speed_multiplier, 1.0)
                total_duration = total_length / speed_deg_per_s

                logger.info(
                    f"    path={total_length:.1f}° · speed={speed_deg_per_s:.0f}°/s · "
                    f"est={total_duration:.1f}s (Catmull-Rom spline)"
                )

                chain_success = True
                traj_start = time.time()

                while time.time() - traj_start < total_duration:
                    elapsed = time.time() - traj_start
                    distance = min(speed_deg_per_s * elapsed, total_length)

                    # Find which segment contains this distance
                    seg_idx = 0
                    for s in range(len(cum_lengths) - 1):
                        if cum_lengths[s] <= distance <= cum_lengths[s + 1]:
                            seg_idx = s
                            break

                    slen = segment_lengths[seg_idx]
                    t = (distance - cum_lengths[seg_idx]) / slen if slen > 0 else 1.0

                    # Catmull-Rom: padded indices for this segment
                    # padded = [W0, W0, W1, W2, ..., Wn, Wn]
                    # segment k (waypoint k → k+1) uses padded[k..k+3]
                    p_idx = seg_idx  # → p0=padded[p_idx], p1=padded[p_idx+1], ...

                    target_action = {}
                    for jn in joint_names:
                        p0 = padded[p_idx][jn]
                        p1 = padded[p_idx + 1][jn]
                        p2 = padded[p_idx + 2][jn]
                        p3 = padded[p_idx + 3][jn]
                        target_action[f"{jn}.pos"] = _cr(p0, p1, p2, p3, t)

                    self.robot.send_action(target_action)

                    observation = self.robot.get_observation()

                    # Monitoring
                    if self.monitor_collector is not None:
                        try:
                            task_info = {
                                "task_name": task.name,
                                "task_type": task.task_type,
                                "cycle": getattr(self, 'current_cycle', 0),
                                "total_cycles": getattr(self, 'total_cycles', 0),
                                "collision_count": self.total_collision_count,
                            }
                            self.monitor_collector.update_robot_state(observation, target_action, task_info)
                        except Exception:
                            pass

                    # Collision
                    if self.collision_detector:
                        if self.collision_detector.check_collision(observation, target_action).is_detected:
                            logger.warning(f"Collision during continuous chain")
                            chain_success = False
                            break

                    time.sleep(dt)

                # Send final waypoint explicitly
                final_action = {}
                for jn in joint_names:
                    final_action[f"{jn}.pos"] = waypoints[-1][jn]
                self.robot.send_action(final_action)

                if not chain_success:
                    return TaskResult(
                        task_name=task.name,
                        status=TaskStatus.FAILED,
                        duration=time.time() - start_time,
                        error_message="Continuous chain failed (collision)",
                        collision_detected=True,
                        attempts=1,
                        start_time=start_time,
                        end_time=time.time(),
                    )

                logger.info(
                    f"  Continuous chain steps {chain_start + 1}-{chain_start + len(chain_steps)} "
                    f"completed in {time.time() - start_time:.1f}s"
                )
                # i already points to the non-overlap step after the chain
                continue

            # ── Standalone step (no overlap) ───────────────────────────
            logger.info(f"  Step {i + 1}/{len(task.steps)}: {step_name}")

            step_task = TaskConfig(
                name=f"{task.name}/{step_name}",
                task_type="position",
                max_duration=step.max_duration,
                max_retries=1,
                completion_criteria=CompletionCriteria(
                    type="position",
                    target_joint_positions=step.position,
                    position_tolerance=step.position_tolerance,
                ),
            )
            if speed_multiplier != 1.0:
                step_task.speed_multiplier = speed_multiplier

            result = self._execute_position_task(step_task)
            if result.status != TaskStatus.COMPLETED:
                logger.warning(f"Step {step_name} failed: {result.error_message}")
                return TaskResult(
                    task_name=task.name,
                    status=result.status,
                    duration=time.time() - start_time,
                    error_message=f"Step '{step_name}' failed: {result.error_message}",
                    collision_detected=result.collision_detected,
                    attempts=1,
                    start_time=start_time,
                    end_time=time.time(),
                )

            i += 1

        logger.info(
            f"Position_sequence task completed: {task.name} -> "
            f"status=completed, duration={time.time() - start_time:.1f}s"
        )

        return TaskResult(
            task_name=task.name,
            status=TaskStatus.COMPLETED,
            duration=time.time() - start_time,
            attempts=1,
            start_time=start_time,
            end_time=time.time(),
        )

    def _add_monitor_event(self, level: str, source: str, message: str):
        """Thread-safe helper to log an event to the monitoring dashboard."""
        if self.monitor_collector is not None:
            try:
                self.monitor_collector.add_event(level, source, message)
            except Exception:
                pass

    def _cleanup(self):
        """Clean up resources after execution."""
        logger.info("Cleaning up...")

        # Stop emergency controller
        if self.emergency_controller is not None:
            logger.info("Stopping emergency stop controller")

        # Stop state monitor
        if self.state_monitor is not None:
            self.state_monitor.stop()

        # Stop monitoring dashboard (NEW)
        if self.http_dashboard is not None:
            try:
                self.http_dashboard.stop()
            except Exception as e:
                logger.warning(f"Error stopping dashboard: {e}")
        if self.monitor_collector is not None:
            try:
                self.monitor_collector.stop()
            except Exception as e:
                logger.warning(f"Error stopping monitor collector: {e}")

        # Disconnect AGV (NEW)
        if self.agv_controller is not None:
            agv_config = getattr(self.config, 'agv_config', None)
            if agv_config and agv_config.auto_disconnect_after_task:
                logger.info("Disconnecting AGV")
                self.agv_controller.disconnect()

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
