"""Collect trajectory data by replaying position_sequence with noise injection.

Uses position_sequence definitions from YAML task configs as the "expert
trajectory" baseline, adds Gaussian noise per-frame for diversity, executes
on the real robot, and records (observation, action) pairs into a
LeRobotDataset for model training.

Leader correction is stubbed — corrector passes actions through unchanged.
The interface is preserved for future Leader/keyboard correction integration.

Example usage:
```shell
python -m lerobot.scripts.collect_trajectory \
    --task_config_path=configs/agv_pick_place_ABCD.yaml \
    --trajectory_tasks=pickup_at_station_C \
    --robot.type=supre_robot_follower \
    --dataset.repo_id=myuser/pickup_trajectory_v1 \
    --dataset.single_task="Pick workpiece from station C" \
    --dataset.num_episodes=50 \
    --noise.noise_std=1.5
```
"""

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.tasks.config import (
    PositionSequenceStep,
    TaskConfig,
    load_config_from_yaml,
)
from lerobot.trajectory.generator import TrajectoryGenerator
from lerobot.trajectory.noise import add_noise_to_action, SmoothNoiseGenerator
from lerobot.utils.control_utils import is_headless
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say


# ============================================================
# Configuration dataclasses
# ============================================================

@dataclass
class NoiseConfig:
    """Noise injection parameters for trajectory data collection."""
    noise_std: float = 1.5  # Standard deviation in degrees (OU: sigma parameter)
    noise_theta: float = 2.0  # OU decay rate; higher = smoother, smaller peak offset. Half-life ≈ ln(2)/theta seconds
    noise_seed: int | None = None  # Base seed; episode i uses base_seed + i. None = random per episode
    skip_keys: list[str] = field(default_factory=lambda: ["joint_7"])  # Substrings matching these get no noise (gripper)
    noise_mode: str = "ou"  # "ou" (Ornstein-Uhlenbeck smooth) or "white" (i.i.d. Gaussian)


@dataclass
class CorrectorStubConfig:
    """Stub configuration for action correction (Leader/keyboard).

    Currently always enable=False (passthrough). When Leader correction
    is implemented, this will hold leader_adjust and key_adjust sub-configs.
    """
    enable: bool = False  # Stub: always False for now
    # Future expansion points:
    # leader_adjust: LeaderAdjustConfig = LeaderAdjustConfig()
    # key_adjust: KeyAdjustConfig = KeyAdjustConfig()


class CorrectorStub:
    """Stubbed action corrector — passthrough when enable=False.

    Provides the same interface as tep_force's ActionCorrector:
    correct(action, events, observation) -> action, reset() -> None.
    When enable=True in the future, it will apply leader/keyboard corrections.
    """

    def __init__(self, config: CorrectorStubConfig, teleop: Any = None):
        self.config = config
        self.teleop = teleop

    def correct(
        self,
        action: dict[str, float],
        events: dict | None = None,
        observation: dict | None = None,
    ) -> dict[str, float]:
        if not self.config.enable:
            return action  # Passthrough
        # Future: leader/keyboard correction logic goes here
        raise NotImplementedError("Action correction not yet implemented. Set corrector.enable=False.")

    def reset(self):
        """Reset accumulated corrections (no-op in stub mode)."""
        pass


@dataclass
class TrajectoryDatasetConfig:
    """Dataset configuration for trajectory collection."""
    repo_id: str  # Dataset identifier (e.g., "username/pickup_trajectory_v1")
    single_task: str  # Task description for dataset metadata
    root: str | Path | None = None  # Local storage directory
    fps: int = 30  # Frames per second for recording
    num_episodes: int = 50  # Number of episodes to collect
    video: bool = True  # Whether to save images as videos
    num_image_writer_processes: int = 0  # Image writer processes (0 = use threads)
    num_image_writer_threads_per_camera: int = 4  # Threads per camera
    use_actual_timestamp: bool = True  # True=perf_counter, False=ideal (frame_index/fps)
    tolerance_s: float | None = None  # None=auto (0.03 if actual, 1e-4 if ideal)


@dataclass
class CollectTrajectoryConfig:
    """Complete configuration for trajectory data collection."""
    # Robot configuration (draccus polymorphic dispatch) — required, no default
    robot: RobotConfig
    # Task config source (YAML with named_positions and position_sequence tasks) — required
    task_config_path: str = ""
    # Dataset configuration — required fields inside
    dataset: TrajectoryDatasetConfig = TrajectoryDatasetConfig(repo_id="", single_task="")
    # Which position_sequence task(s) to use: "all" or comma-separated names
    trajectory_tasks: str = "all"
    # Noise configuration
    noise: NoiseConfig = NoiseConfig()
    # Corrector configuration (stub)
    corrector: CorrectorStubConfig = CorrectorStubConfig()
    # Execution settings
    control_frequency: int = 30  # Hz for sending actions
    success_key_timeout: float = 30.0  # Seconds to wait for S/F key after episode
    reset_duration: float = 3.0  # Seconds to move to start position before each episode
    play_sounds: bool = True


# ============================================================
# Keyboard listener
# ============================================================

def init_collect_keyboard_listener() -> tuple[Any | None, dict[str, bool]]:
    """Initialize simplified keyboard listener for trajectory collection.

    Keys:
    - S: Mark success, save episode
    - F: Mark fail, discard episode
    - ESC: Stop all recording

    Returns:
        (listener, events) tuple. Listener is None in headless environments.
    """
    events: dict[str, bool] = {
        "mark_success": False,
        "mark_fail": False,
        "stop_recording": False,
    }

    if is_headless():
        logging.warning(
            "Headless environment detected. Keyboard inputs will not be available. "
            "Episodes will auto-save on timeout."
        )
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if hasattr(key, "char") and key.char:
                char = key.char.lower()
                if char == "s":
                    print("[Collect] S pressed: marking SUCCESS")
                    events["mark_success"] = True
                elif char == "f":
                    print("[Collect] F pressed: marking FAIL, discarding episode")
                    events["mark_fail"] = True
            elif key == keyboard.Key.esc:
                print("[Collect] ESC pressed: stopping collection")
                events["stop_recording"] = True
        except Exception as e:
            print(f"[Collect] Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("=" * 60)
    print("Keyboard controls: S=save episode, F=discard, ESC=stop")
    print("=" * 60)

    return listener, events


# ============================================================
# Reset helper
# ============================================================

def reset_to_start_position(
    robot: Robot,
    start_action: dict[str, float],
    duration: float = 3.0,
    control_frequency: int = 30,
) -> None:
    """Move robot to the trajectory's starting position using smooth interpolation.

    Args:
        robot: Connected robot instance.
        start_action: First frame of the trajectory ({"joint_name.pos": float} format).
        duration: Seconds to spend moving to start position.
        control_frequency: Hz for the movement loop.
    """
    current_positions = robot.get_current_position()  # {"joint_name": float} no suffix
    dt = 1.0 / control_frequency
    start_time = time.perf_counter()

    logging.info(f"Resetting robot to start position over {duration:.1f}s")

    while time.perf_counter() - start_time < duration:
        elapsed = time.perf_counter() - start_time
        progress = min(1.0, elapsed / duration)
        smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))

        move_action = {}
        for key, target_value in start_action.items():
            # Strip .pos suffix for lookup in current_positions
            joint_name = key.removesuffix(".pos")
            if joint_name in current_positions:
                current_value = current_positions[joint_name]
                intermediate = current_value + (target_value - current_value) * smooth_progress
                move_action[key] = intermediate
            else:
                move_action[key] = target_value

        if move_action:
            robot.send_action(move_action)
        time.sleep(dt)

    logging.info("Reset complete")


# ============================================================
# Frame loop
# ============================================================

def collect_frame_loop(
    robot: Robot,
    trajectory: list[dict[str, float]],
    dataset: LeRobotDataset,
    corrector: CorrectorStub,
    rng: np.random.Generator,
    events: dict[str, bool],
    cfg: CollectTrajectoryConfig,
    noise_gen: SmoothNoiseGenerator | None = None,
) -> bool:
    """Execute one episode of trajectory collection with noise and recording.

    Args:
        noise_gen: SmoothNoiseGenerator for OU mode. None = use white noise.

    Returns:
        True if episode was saved, False if discarded.
    """
    noise_cfg = cfg.noise
    fps = cfg.dataset.fps
    single_task = cfg.dataset.single_task
    use_actual_timestamp = cfg.dataset.use_actual_timestamp

    episode_start = time.perf_counter()
    frame_index = 0

    for idx, base_action in enumerate(trajectory):
        loop_start = time.perf_counter()

        # 1. Timestamp
        if use_actual_timestamp:
            actual_timestamp = time.perf_counter() - episode_start
            max_timestamp = frame_index / fps
            actual_timestamp = min(actual_timestamp, max_timestamp)
        else:
            actual_timestamp = frame_index / fps

        # 2. Add noise to base action
        if noise_gen is not None:
            noisy_action = noise_gen.perturb(base_action)
        else:
            noisy_action = add_noise_to_action(
                base_action, noise_cfg.noise_std, rng, noise_cfg.skip_keys
            )

        # 3. Apply correction (stub: passthrough)
        final_action = corrector.correct(noisy_action, events=events)

        # 4. Get observation BEFORE sending action
        observation = robot.get_observation()

        # Flatten nested "images" dict for build_dataset_frame compatibility.
        # get_observation() returns {"images": {"head_cam": img, ...}} but
        # build_dataset_frame expects {"head_cam": img, ...} at top level.
        if "images" in observation and isinstance(observation["images"], dict):
            flat_obs = {k: v for k, v in observation.items() if k != "images"}
            flat_obs.update(observation["images"])
            observation = flat_obs

        # 5. Send action and get what robot actually received
        sent_action = robot.send_action(final_action)

        # 6. Record frame
        obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
        frame = {**obs_frame, **action_frame}
        dataset.add_frame(frame, task=single_task, timestamp=actual_timestamp)

        frame_index += 1

        # 7. Check keyboard events mid-episode
        if events["mark_fail"]:
            logging.info("Episode marked as FAIL during execution, discarding...")
            dataset.clear_episode_buffer()
            events["mark_fail"] = False
            corrector.reset()
            return False

        if events["stop_recording"]:
            logging.info("ESC pressed during episode, discarding and stopping...")
            dataset.clear_episode_buffer()
            corrector.reset()
            return False

        # 8. Timing sync
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

    # 9. Episode complete — wait for success/fail key
    logging.info(f"Episode trajectory complete ({frame_index} frames). Press S to save, F to discard...")

    wait_start = time.perf_counter()
    while time.perf_counter() - wait_start < cfg.success_key_timeout:
        if events["mark_success"]:
            logging.info("Episode marked as SUCCESS, saving...")
            dataset.save_episode()
            events["mark_success"] = False
            corrector.reset()
            return True
        if events["mark_fail"]:
            logging.info("Episode marked as FAIL, discarding...")
            dataset.clear_episode_buffer()
            events["mark_fail"] = False
            corrector.reset()
            return False
        if events["stop_recording"]:
            logging.info("ESC pressed, stopping...")
            dataset.clear_episode_buffer()
            corrector.reset()
            return False
        time.sleep(0.1)

    # Timeout — auto-save
    logging.info("Success key timeout, auto-saving episode...")
    dataset.save_episode()
    corrector.reset()
    return True


# ============================================================
# Main entry point
# ============================================================

@draccus.wrap()
def collect_trajectory(cfg: CollectTrajectoryConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # === 1. Load task config (YAML with named_positions + position_sequence) ===
    task_config = load_config_from_yaml(cfg.task_config_path)

    # Filter position_sequence tasks
    seq_tasks = [t for t in task_config.tasks if t.task_type == "position_sequence"]
    if not seq_tasks:
        raise ValueError(
            f"No position_sequence tasks found in {cfg.task_config_path}. "
            "Available task types: " + ", ".join(t.task_type for t in task_config.tasks)
        )

    # Select tasks by name
    if cfg.trajectory_tasks.lower() == "all":
        selected_tasks = seq_tasks
    else:
        requested_names = [n.strip() for n in cfg.trajectory_tasks.split(",")]
        selected_tasks = [t for t in seq_tasks if t.name in requested_names]
        if not selected_tasks:
            available_names = [t.name for t in seq_tasks]
            raise ValueError(
                f"No matching position_sequence tasks for '{cfg.trajectory_tasks}'. "
                f"Available: {available_names}"
            )

    logging.info(f"Selected {len(selected_tasks)} position_sequence tasks: {[t.name for t in selected_tasks]}")

    # === 2. Create robot ===
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # === 3. Create dataset ===
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    # Tolerance: auto based on timestamp mode
    tolerance_s = cfg.dataset.tolerance_s
    if tolerance_s is None:
        tolerance_s = 0.03 if cfg.dataset.use_actual_timestamp else 1e-4

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") and robot.cameras else 0
    image_writer_threads = cfg.dataset.num_image_writer_threads_per_camera * num_cameras

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        tolerance_s=tolerance_s,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    logging.info(f"Dataset created: {cfg.dataset.repo_id}, fps={cfg.dataset.fps}, tolerance_s={tolerance_s}")

    # === 4. Build trajectory ===
    joint_names = robot.observation_joint_names if hasattr(robot, "observation_joint_names") else None
    if joint_names is None:
        # Fallback: derive from first selected task's step positions
        joint_names = list(selected_tasks[0].steps[0].position.keys())

    start_position = robot.get_current_position()  # {"joint_name": float} no suffix

    # Generate per-task trajectories and concatenate
    full_trajectory = []
    for task in selected_tasks:
        generator = TrajectoryGenerator(task.steps, cfg.dataset.fps, joint_names)
        task_trajectory = generator.generate(start_position)
        logging.info(
            f"Task '{task.name}': {generator.total_frames} frames, "
            f"{generator.total_duration:.1f}s"
        )
        full_trajectory.extend(task_trajectory)

        # Update start_position for next task (chain from end of this trajectory)
        # Last frame's values become the next task's start
        last_frame = task_trajectory[-1]
        for key, value in last_frame.items():
            start_position[key.removesuffix(".pos")] = value

    logging.info(f"Total trajectory: {len(full_trajectory)} frames")

    # === 5. Init corrector (stub) ===
    corrector = CorrectorStub(cfg.corrector)

    # === 6. Init keyboard listener ===
    listener, events = init_collect_keyboard_listener()

    # === 7. Episode loop ===
    log_say("Starting trajectory collection", cfg.play_sounds, blocking=True)

    recorded_episodes = 0
    episode_idx = 0

    while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
        # Per-episode noise seed
        if cfg.noise.noise_seed is not None:
            episode_seed = cfg.noise.noise_seed + episode_idx
            rng = np.random.default_rng(episode_seed)
            logging.info(f"Episode {episode_idx + 1}: seed={episode_seed}, mode={cfg.noise.noise_mode}")
        else:
            rng = np.random.default_rng()
            logging.info(f"Episode {episode_idx + 1}: random seed, mode={cfg.noise.noise_mode}")

        corrector.reset()

        # Create noise generator for OU mode
        noise_gen = None
        if cfg.noise.noise_mode == "ou":
            all_keys = list(full_trajectory[0].keys())
            noise_gen = SmoothNoiseGenerator(
                all_keys=all_keys,
                noise_std=cfg.noise.noise_std,
                skip_keys=cfg.noise.skip_keys,
                rng=rng,
                fps=cfg.dataset.fps,
                theta=cfg.noise.noise_theta,
            )

        # Move robot to trajectory start position
        log_say(f"Episode {episode_idx + 1}, resetting to start", cfg.play_sounds, blocking=False)
        reset_to_start_position(
            robot,
            full_trajectory[0],
            duration=cfg.reset_duration,
            control_frequency=cfg.control_frequency,
        )

        # Execute trajectory with noise and recording
        success = collect_frame_loop(
            robot=robot,
            trajectory=full_trajectory,
            dataset=dataset,
            corrector=corrector,
            rng=rng,
            events=events,
            cfg=cfg,
            noise_gen=noise_gen,
        )

        episode_idx += 1
        if success:
            recorded_episodes += 1
            logging.info(f"Episode saved. Total saved: {recorded_episodes}/{cfg.dataset.num_episodes}")
        else:
            logging.info("Episode discarded, retrying...")

        if events["stop_recording"]:
            break

        # Brief pause between episodes
        time.sleep(1.0)

    # === 8. Cleanup ===
    log_say("Stopping collection", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if not is_headless() and listener is not None:
        listener.stop()

    dataset.stop_image_writer()
    logging.info(f"Collection complete. Episodes saved: {recorded_episodes}")

    return dataset


if __name__ == "__main__":
    collect_trajectory()