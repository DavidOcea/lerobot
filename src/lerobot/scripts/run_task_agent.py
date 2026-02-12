#!/usr/bin/env python3
"""
Main execution script for the task agent orchestrator.

This script provides a command-line interface for running robotic task
sequences with collision detection and automatic retry.

Usage:
    # Local mode (recommended - no Policy Server needed)
    python run_task_agent.py --config configs/task_agent_tasks.yaml
    python run_task_agent.py --config configs/task_agent_tasks.yaml --debug

    # Remote mode (requires Policy Server)
    python run_task_agent.py --config configs/task_agent_tasks.yaml --remote
    python run_task_agent.py --config configs/task_agent_tasks.yaml --remote --debug
"""

import argparse
import logging
import sys
from pathlib import Path

import draccus

from lerobot.agent.config import OrchestratorConfig
from lerobot.agent.orchestrator import TaskAgentOrchestrator
from lerobot.tasks.config import load_config_from_yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run robotic task sequences with collision detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks from config (local mode - recommended)
  python run_task_agent.py --config configs/task_agent_tasks.yaml

  # Run with interactive mode (prompt before each task)
  python run_task_agent.py --config configs/task_agent_tasks.yaml --interactive

  # Run with custom emergency stop thresholds
  python run_task_agent.py --config configs/task_agent_tasks.yaml --emergency-force-threshold 2.0 --emergency-max-velocity 3.0

  # Run with debug output
  python run_task_agent.py --config configs/task_agent_tasks.yaml --debug

  # Run only specific tasks
  python run_task_agent.py --config configs/task_agent_tasks.yaml --tasks pick_short,pick_long

  # Disable emergency stop (for testing)
  python run_task_agent.py --config configs/task_agent_tasks.yaml --no-emergency-stop

  # Remote mode (requires Policy Server running)
  python run_task_agent.py --config configs/task_agent_tasks.yaml --remote
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to task configuration YAML file",
    )

    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote mode (connect to Policy Server). Default is local mode (no Policy Server needed).",
    )

    parser.add_argument(
        "--policy-device",
        type=str,
        default="cuda",
        help="Device for policy execution (default: cuda)",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task names to run (default: all enabled tasks)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override maximum retry attempts for all tasks",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Override maximum duration for all tasks (seconds)",
    )

    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=None,
        help="Override collision threshold (Nm)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse configuration and validate without execution",
    )

    parser.add_argument(
        "--save-observations",
        action="store_true",
        help="Save observations to disk for debugging",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive task selection mode (prompt before each task)",
    )

    # Emergency stop settings
    parser.add_argument(
        "--no-emergency-stop",
        action="store_true",
        help="Disable emergency stop and rollback functionality",
    )

    parser.add_argument(
        "--emergency-force-threshold",
        type=float,
        default=None,
        help="Emergency stop force threshold in Nm (default: 2.0)",
    )

    parser.add_argument(
        "--emergency-max-velocity",
        type=float,
        default=None,
        help="Emergency stop maximum velocity in rad/s (default: 3.0)",
    )

    parser.add_argument(
        "--emergency-max-rollback-steps",
        type=int,
        default=None,
        help="Maximum steps to rollback on emergency stop (default: 80)",
    )

    return parser.parse_args()


def main():
    """Main entry point for task agent execution."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {config_path}")

    try:
        config = load_config_from_yaml(config_path)

        # Set execution mode
        config.use_local_execution = not args.remote
        config.policy_device = args.policy_device

        # Log execution mode
        mode_str = "LOCAL (direct)" if config.use_local_execution else "REMOTE (via Policy Server)"
        logger.info(f"Execution mode: {mode_str}")

        if not config.use_local_execution:
            logger.warning("Remote mode requires Policy Server to be running!")
            logger.info("Start Policy Server: ./scripts/start_policy_server.sh")

        # Apply command line overrides
        if args.max_retries is not None:
            config.override_max_retries = args.max_retries
            logger.info(f"Override max_retries: {args.max_retries}")

        if args.max_duration is not None:
            config.override_max_duration = args.max_duration
            logger.info(f"Override max_duration: {args.max_duration}")

        if args.collision_threshold is not None:
            config.collision_config.collision_threshold = args.collision_threshold
            logger.info(f"Override collision_threshold: {args.collision_threshold}")

        if args.debug:
            config.debug_mode = True
            config.monitoring_config.log_level = "DEBUG"

        if args.save_observations:
            config.save_observations = True

        # Apply interactive mode setting
        config.enable_interactive_mode = args.interactive
        if args.interactive:
            logger.info("Interactive mode enabled - will prompt before each task")

        # Apply emergency stop settings
        config.enable_emergency_stop = not args.no_emergency_stop
        if args.no_emergency_stop:
            logger.warning("Emergency stop DISABLED via command line")

        if args.emergency_force_threshold is not None:
            config.emergency_force_threshold = args.emergency_force_threshold
            logger.info(f"Emergency force threshold: {args.emergency_force_threshold} Nm")

        if args.emergency_max_velocity is not None:
            config.emergency_max_velocity = args.emergency_max_velocity
            logger.info(f"Emergency max velocity: {args.emergency_max_velocity} rad/s")

        if args.emergency_max_rollback_steps is not None:
            config.emergency_max_rollback_steps = args.emergency_max_rollback_steps
            logger.info(f"Emergency max rollback steps: {args.emergency_max_rollback_steps}")

        # Filter tasks if specified
        if args.tasks:
            task_names = [t.strip() for t in args.tasks.split(",")]
            enabled_tasks = []
            for task in config.tasks:
                if task.name in task_names:
                    task.enabled = True
                    enabled_tasks.append(task.name)
                else:
                    task.enabled = False

            logger.info(f"Enabled tasks: {enabled_tasks}")

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=args.debug)
        sys.exit(1)

    # Dry run - just validate and print config
    if args.dry_run:
        logger.info("Dry run mode - configuration validation only")
        logger.info(f"Tasks: {len(config.tasks)}")
        for task in config.tasks:
            status = "enabled" if task.enabled else "disabled"
            logger.info(f"  - {task.name}: {status}")
        sys.exit(0)

    # Create and run orchestrator
    logger.info("Creating task agent orchestrator...")

    try:
        orchestrator = TaskAgentOrchestrator(config)

        # Initialize
        if not orchestrator.initialize():
            logger.error("Failed to initialize orchestrator")
            sys.exit(1)

        # Run task sequence
        logger.info("Starting task execution...")
        summary = orchestrator.run()

        # Print results
        print("\n" + "=" * 60)
        print("TASK EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total tasks:        {summary.total_tasks}")
        print(f"Completed:          {summary.completed_tasks}")
        print(f"Failed:             {summary.failed_tasks}")
        print(f"Skipped:            {summary.skipped_tasks}")
        print(f"Total duration:     {summary.total_duration:.2f}s")
        print(f"Collisions:         {summary.collision_count}")
        print(f"Total retries:      {summary.total_retries}")
        print(f"Overall success:    {summary.overall_success}")
        print("=" * 60)

        # Print per-task results
        print("\nPer-task results:")
        for result in summary.task_results:
            status_symbol = "✓" if result.success else "✗"
            print(
                f"  {status_symbol} {result.task_name}: {result.status.value} "
                f"(attempts: {result.attempts}, duration: {result.duration:.2f}s)"
            )

        # Exit with appropriate code
        sys.exit(0 if summary.overall_success else 1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()
