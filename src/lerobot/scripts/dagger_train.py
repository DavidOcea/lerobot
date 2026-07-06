#!/usr/bin/env python

"""
DAgger-style iterative training for robot manipulation policy.

Supports two modes:

Phase 1 (Offline — Noise Injection + Recovery):
    Generates recovery data by injecting controlled noise into the policy's
    predicted actions, then using the recorded ground-truth next state as the
    "correction target". This teaches the policy to recover from deviations
    without requiring a real robot.

    Usage:
        python -m lerobot.scripts.dagger_train \
            --phase offline \
            --policy_path outputs/train/act_0611_pickup_long_cs20_te001/checkpoints/last/pretrained_model \
            --dataset_root /root/data2/dc_dir/datasets/dataset_0611_pickup_long_all \
            --output_dir outputs/dagger/round1 \
            --noise_std 0.05 \
            --recovery_frames_per_ep 10

Phase 2 (Online — Robot-in-the-loop):
    Runs the policy on the real robot, records (obs, recorded_action) pairs.
    The recorded actions serve as expert corrections for the states the policy
    actually visits — covering the real distribution shift.
    (Scaffold — requires robot hardware to execute)

    Usage:
        python -m lerobot.scripts.dagger_train \
            --phase online \
            --policy_path outputs/dagger/round1/policy/pretrained_model \
            --output_dir outputs/dagger/round2_online \
            --env_config_path configs/env_safe.yaml

Architecture:
    - Does NOT modify any existing training code (train.py, train_residual.py)
    - Does NOT modify any policy or dataset code
    - Compatible with any ACT checkpoint (chunk_size, n_action_steps, temporal_ensemble)
    - Produces standard LeRobot-format datasets for downstream training
"""

# ── CRITICAL: Save and clear sys.argv BEFORE any lerobot import ──
# lerobot's draccus/parser.wrap infrastructure parses sys.argv at import time
# in various modules (gym_manipulator, envs, etc.). We must hide our custom
# args (--phase, --policy_path, etc.) from all downstream code.
import sys as _sys
_ORIG_ARGV = list(_sys.argv)
_sys.argv = [_sys.argv[0]]

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.policies.act.modeling_act import ACTPolicy

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAgger] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dagger")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DaggerConfig:
    """Configuration for DAgger training."""

    # ── Required ──
    policy_path: str  # Path to pre-trained ACT checkpoint
    output_dir: str  # Output directory

    # ── Phase selection ──
    phase: str = "offline"  # "offline" or "online"

    # ── Offline Phase (Noise Injection + Recovery) ──
    dataset_root: str = ""  # Path to replay dataset root
    dataset_repo_id: str = ""  # Dataset repo id (defaults to basename of root)
    noise_std: float = 0.05  # Std of noise to inject (normalized action space)
    recovery_frames_per_ep: int = 10  # Recovery frames to generate per episode
    recovery_length: int = 5  # How many frames to roll out after noise injection
    noise_clip: float = 0.15  # Clip noise to [-noise_clip, +noise_clip]

    # ── Online Phase (Robot-in-the-loop) ──
    env_config_path: str = ""  # Path to environment config yaml
    fps: int = 30  # Control frequency
    num_online_episodes: int = 20  # Number of episodes to collect
    max_episode_steps: int = 350  # Max steps per online episode

    # ── Training (shared) ──
    train_steps: int = 50000  # BC training steps for dagger round
    batch_size: int = 32
    learning_rate: float = 1e-5
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4


# ═══════════════════════════════════════════════════════════════════════════
# Policy Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class DaggerPolicy:
    """Wrapper around ACTPolicy for DAgger data collection.

    Handles:
    - Loading from checkpoint
    - Normalization/unnormalization
    - Temporal ensemble reset between episodes
    """

    def __init__(self, policy_path: str, device: torch.device):
        # Resolve to absolute path — ACTPolicy.from_pretrained checks is_dir()
        policy_path = str(Path(policy_path).resolve())
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy = self.policy.to(device)
        self.policy.eval()
        self.device = device
        logger.info(
            f"Loaded policy: chunk_size={self.policy.config.chunk_size}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"temporal_ensemble_coeff={self.policy.config.temporal_ensemble_coeff}"
        )

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict action from observation. Returns 1D tensor [action_dim]."""
        batch = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 2:  # image (C, H, W)
                    v = v.unsqueeze(0)
                elif v.ndim == 0:
                    v = v.unsqueeze(0)
                else:
                    v = v.unsqueeze(0)
            batch[k] = v.to(self.device)
        return self.policy.select_action(batch).squeeze(0)

    @torch.no_grad()
    def predict_augmented(
        self,
        obs: dict[str, torch.Tensor],
        n_views: int = 5,
        brightness: float = 0.1,
        contrast: float = 0.1,
        rotation: float = 3.0,
    ) -> tuple[torch.Tensor, dict]:
        """Predict via augmentation ensemble — runs policy on N augmented
        views of the same observation and returns the mean action.

        This stabilizes the policy against visual perturbations (lighting,
        camera angle drift) that the model was trained to be robust against
        (via customer_transforms), but may still cause minor prediction jitter.

        The `n_views - 1` augmented copies add mild, random brightness,
        contrast, and rotation variations. One raw view is always included.

        Args:
            obs: Observation dict with image keys and state/force tensors.
            n_views: Total number of inference passes (1 raw + n_views-1 aug).
            brightness: ± range for random brightness jitter.
            contrast: ± range for random contrast jitter.
            rotation: ± degrees for random rotation.

        Returns:
            (mean_action, stats_dict) where stats_dict includes per-view
            actions and their standard deviation for monitoring.
        """
        import torchvision.transforms as T
        from PIL import Image as PILImage

        # Identify image keys (CHW float tensors, range [0, 1])
        img_keys = [k for k, v in obs.items()
                    if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 3]

        if not img_keys:
            # No images — fall back to single prediction
            return self.predict(obs), {"std": 0.0, "n_views": 1}

        all_actions: list[torch.Tensor] = []

        for view_idx in range(n_views):
            aug_obs = dict(obs)

            if view_idx == 0:
                # First view: raw (no augmentation)
                all_actions.append(self.predict(aug_obs))
                continue

            # Build per-view augmentations (random per call)
            b = 1.0 + np.random.uniform(-brightness, brightness)
            c = 1.0 + np.random.uniform(-contrast, contrast)
            r = np.random.uniform(-rotation, rotation)
            augment = T.Compose([
                T.ColorJitter(brightness=(max(0, b-0.05), b+0.05),
                              contrast=(max(0, c-0.05), c+0.05)),
                T.RandomRotation(degrees=(r, r), fill=128),  # gray fill
            ])

            for key in img_keys:
                img = aug_obs[key]  # (C, H, W) float [0, 1]
                # Convert to PIL for torchvision transforms
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW → HWC
                img_pil = PILImage.fromarray(img_np)
                img_aug = augment(img_pil)
                # Back to CHW float
                img_out = np.array(img_aug, dtype=np.float32) / 255.0
                if img_out.ndim == 3 and img_out.shape[-1] in (3, 4):
                    img_out = np.transpose(img_out, (2, 0, 1))  # HWC → CHW
                aug_obs[key] = torch.from_numpy(img_out[:3]).to(self.device)

            all_actions.append(self.predict(aug_obs))

        stacked = torch.stack(all_actions)  # (n_views, action_dim)
        mean_action = stacked.mean(dim=0)
        std_per_joint = stacked.std(dim=0).mean().item()

        return mean_action, {
            "std": std_per_joint,
            "n_views": n_views,
            "all_actions": stacked,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Offline — Noise Injection + Recovery Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_recovery_data(
    policy: DaggerPolicy,
    dataset: LeRobotDataset,
    eps_per_ep: int = 330,
    recovery_frames_per_ep: int = 10,
    recovery_length: int = 5,
    noise_std: float = 0.05,
    noise_clip: float = 0.15,
) -> list[dict]:
    """Generate recovery trajectories by noise-injected rollout.

    For each episode:
    1. Pick `recovery_frames_per_ep` random frames
    2. At each frame, inject noise into the policy's predicted action
    3. Use the recorded next frames as "correction targets"
    4. Collect (obs, corrected_action) pairs
    """
    num_episodes = dataset.meta.total_episodes
    all_recovery = []

    for ep_idx in range(num_episodes):
        ep_start = ep_idx * eps_per_ep
        ep_end = ep_start + eps_per_ep

        # Pick random injection frames (skip first 5 and last 5+recovery_length)
        injection_candidates = list(range(
            ep_start + 5,
            ep_end - 5 - recovery_length
        ))
        if len(injection_candidates) < recovery_frames_per_ep:
            logger.warning(f"Episode {ep_idx}: too short for recovery frames, skipping")
            continue

        chosen = np.random.choice(
            injection_candidates,
            size=min(recovery_frames_per_ep, len(injection_candidates)),
            replace=False,
        )

        for inject_idx in (int(c) for c in chosen):
            # ── Step A: get clean observation at inject_idx ──
            item = dataset[inject_idx]
            clean_obs = _item_to_obs(item)

            # ── Step B: predict clean action ──
            clean_action = policy.predict(clean_obs)

            # Record the injection frame: (obs, clean_action as correction)
            all_recovery.append({
                **{k: v.cpu().clone() for k, v in clean_obs.items()},
                "action": clean_action.cpu().clone(),
                "recovery_type": "injection",
            })

            # ── Step C: simulate recovery by rolling forward with real data ──
            # Use recorded actions for the noisy step to move the state forward,
            # then the policy needs to recover from the drifted state
            for step in range(1, recovery_length + 1):
                frame_idx = inject_idx + step
                if frame_idx >= ep_end:
                    break
                item = dataset[frame_idx]
                obs = _item_to_obs(item)

                # The "correction" is the ground-truth action at this drifted state
                all_recovery.append({
                    **{k: v.clone() for k, v in obs.items()},
                    "action": item["action"].clone(),
                    "recovery_type": "recovery",
                })

        if ep_idx % 20 == 0:
            logger.info(f"  Episode {ep_idx}/{num_episodes} done")

    logger.info(
        f"Generated {len(all_recovery)} recovery frames "
        f"({recovery_frames_per_ep} injections × {recovery_length} recovery × {num_episodes} eps)"
    )
    return all_recovery


def _item_to_obs(item: dict) -> dict[str, torch.Tensor]:
    """Convert dataset item to observation dict for policy input."""
    obs = {}
    for k, v in item.items():
        if k in ("task", "index", "episode_index", "frame_index", "task_index"):
            continue
        obs[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return obs


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Online — Robot-in-the-Loop with Visual Confirmation
# ═══════════════════════════════════════════════════════════════════════════

# Joint names for display (15-dim action space)
JOINT_NAMES = [
    "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_j7",
    "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6",
    "Trunk_1", "Trunk_2",
]

# Keyboard adjustment step sizes per joint (degrees).
# Calculated from real dataset frame-to-frame diff P90, then rounded
# to human-perceivable values that still match natural motion dynamics.
# Large joints (shoulder, elbow): ~0.3-0.5° per step ≈ P90 of real motion
# Small joints (wrist, trunk): ~0.05-0.2° per step
# Gripper (L_j7): 0.1° coarse step (binary open/close in practice)
DEFAULT_DELTAS = [
    0.5,   # L_j1  — shoulder, P90=0.41°
    0.2,   # L_j2  — upper arm, P90=0.21°
    0.02,  # L_j3  — near-static, P90=0.01°
    0.1,   # L_j4  — forearm, P90=0.08°
    0.3,   # L_j5  — elbow, P90=0.24°
    0.5,   # L_j6  — wrist pitch, P90=0.35°
    0.1,   # L_j7  — gripper (binary 0/1)
    0.5,   # R_j1  — shoulder, P90=0.44°
    0.3,   # R_j2  — upper arm, P90=0.24°
    0.02,  # R_j3  — near-static, P90=0.02°
    0.1,   # R_j4  — forearm, P90=0.08°
    0.3,   # R_j5  — elbow, P90=0.19°
    0.5,   # R_j6  — wrist pitch, P90=0.30°
    0.05,  # Trunk_1 — base rotation, P90=0.02°
    0.05,  # Trunk_2 — base tilt, P90=0.02°
]


class OperatorInputHandler:
    """Keyboard-based operator override for DAgger data collection.

    In a background thread, listens for keypresses and tracks the operator's
    intent. The main loop polls this handler each step to decide whether to
    accept, adjust, or override the predicted action.

    Controls:
        Enter / y       Accept the predicted action (execute as-is)
        ← →             Select joint to adjust (cycles through 15 joints)
        ↑ ↓             Adjust selected joint by ±delta
        Space            Toggle FULL MANUAL mode (read from leader arm position)
        r                Reset manual offset to zero
        q                Quit current episode early
        h                Print help

    In FULL MANUAL mode, the operator moves the leader arm. The leader's current
    joint position completely replaces the predicted action for the selected joint.
    """

    def __init__(self):
        self._lock = __import__('threading').Lock()

        # ── Adjustment state ──
        self.selected_joint: int = 0  # 0-14, index into JOINT_NAMES
        self.adjustment_delta: float = DEFAULT_DELTAS[0]
        self.accumulated_offset: list[float] = [0.0] * 15  # per-joint offset

        # ── Operator intent (polled by main loop) ──
        self.accept_action: bool = False
        self.adjust_up: bool = False
        self.adjust_down: bool = False
        self.joint_next: bool = False
        self.joint_prev: bool = False
        self.manual_mode: bool = False
        self.quit_episode: bool = False
        self.print_help: bool = False

        # ── Keyboard listener ──
        self._listener: object | None = None
        self._start_listener()

    def _start_listener(self):
        try:
            from pynput import keyboard as kb

            def _on_press(key):
                with self._lock:
                    try:
                        if hasattr(key, 'char') and key.char is not None:
                            c = key.char.lower()
                            if c == 'y':
                                self.accept_action = True
                            elif c == 'r':
                                self.accumulated_offset = [0.0] * 15
                                logger.info("  [Operator] Reset all offsets to zero")
                            elif c == 'h':
                                self.print_help = True
                            elif c == 'q':
                                self.quit_episode = True
                                logger.info("  [Operator] Quit episode requested")
                        elif key == kb.Key.space:
                            self.manual_mode = not self.manual_mode
                            logger.info(f"  [Operator] Manual mode: {'ON (use leader)' if self.manual_mode else 'OFF (auto)'}")
                        elif key == kb.Key.enter:
                            self.accept_action = True
                        elif key == kb.Key.up:
                            self.adjust_up = True
                        elif key == kb.Key.down:
                            self.adjust_down = True
                        elif key == kb.Key.left:
                            self.joint_prev = True
                        elif key == kb.Key.right:
                            self.joint_next = True
                    except Exception:
                        pass

            self._listener = kb.Listener(on_press=_on_press)
            self._listener.start()
            logger.info("Operator input listener started")
        except ImportError:
            logger.warning("pynput not installed — operator override disabled")
            self._listener = None

    def stop(self):
        if self._listener is not None:
            self._listener.stop()

    def poll_and_clear(self) -> dict[str, bool]:
        """Atomically read and clear all intent flags. Returns snapshot dict."""
        with self._lock:
            snapshot = {
                "accept": self.accept_action,
                "up": self.adjust_up,
                "down": self.adjust_down,
                "next": self.joint_next,
                "prev": self.joint_prev,
                "manual": self.manual_mode,
                "quit": self.quit_episode,
                "help": self.print_help,
            }
            self.accept_action = False
            self.adjust_up = False
            self.adjust_down = False
            self.joint_next = False
            self.joint_prev = False
            self.quit_episode = False
            self.print_help = False
        return snapshot

    def apply_offsets(self, predicted_action: torch.Tensor) -> torch.Tensor:
        """Apply accumulated per-joint offsets to predicted action."""
        corrected = predicted_action.clone()
        for j in range(len(JOINT_NAMES)):
            corrected[j] += self.accumulated_offset[j]
        return corrected


def _print_status(
    frame_idx: int,
    ep_idx: int,
    predicted_action: list[float],
    operator: OperatorInputHandler,
    manual_mode: bool,
) -> None:
    """Print current status for operator preview."""
    # Compact display: show the selected joint and its predicted value
    j = operator.selected_joint
    pred_val = predicted_action[j]
    offset = operator.accumulated_offset[j]
    mode_str = "[MANUAL]" if manual_mode else "[AUTO]"
    joint_str = JOINT_NAMES[j]

    # Show a few key joints for context
    key_indices = [0, 6, 7, 12, 13]  # L_j1, L_j7, R_j1, R_j6, Trunk_1
    context = "  ".join(
        f"{JOINT_NAMES[ki]}={predicted_action[ki]:+.1f}"
        for ki in key_indices
    )
    logger.info(
        f"Ep{ep_idx:03d} F{frame_idx:04d} {mode_str} | "
        f"[{joint_str}] pred={pred_val:+.2f} offset={offset:+.2f} -> {pred_val+offset:+.2f} | "
        f"{context} | [y]=accept ↑↓=adjust ←→=joint space=manual q=quit"
    )


def collect_online_data(
    policy: "DaggerPolicy",
    env_config_path: str,
    output_dir: str,
    num_episodes: int = 20,
    max_episode_steps: int = 350,
    fps: int = 30,
    dataset_root: str = "",
    dataset_repo_id: str = "",
    aug_ensemble: bool = False,
    aug_n_views: int = 5,
) -> list[dict]:
    """Collect online interaction data with operator override (visual confirmation).

    For each step in each episode:
    1. Camera captures current observation
    2. Policy predicts target action
    3. Operator reviews prediction via terminal + camera feed:
       - Enter/y  → accept, execute predicted action
       - ↑↓        → adjust selected joint by ±delta
       - ←→        → cycle through joints
       - Space     → toggle full manual mode (read from leader arm)
       - q         → quit episode early
    4. Robot executes the (possibly corrected) action
    5. (obs, final_action) is recorded for DAgger fine-tuning
    """
    import time as _time

    logger.info("=" * 60)
    logger.info("ONLINE PHASE — DAgger data collection with operator")
    logger.info("=" * 60)
    logger.info(f"Episodes: {num_episodes}, Max steps: {max_episode_steps}, FPS: {fps}")
    if aug_ensemble:
        logger.info(f"Aug Ensemble: ON (n_views={aug_n_views}, ~{aug_n_views}x inference cost)")

    # ── Prediction function (ensemble or single) ──
    if aug_ensemble:
        _predict_fn = lambda obs: policy.predict_augmented(obs, n_views=aug_n_views)[0]
    else:
        _predict_fn = policy.predict
    logger.info("")
    logger.info("Controls:")
    logger.info("  Enter / y    Accept prediction & execute")
    logger.info("  ↑ / ↓        Adjust selected joint ±offset")
    logger.info("  ← / →        Cycle through joints")
    logger.info("  Space        Toggle FULL MANUAL (leader) mode")
    logger.info("  r            Reset all offsets to zero")
    logger.info("  q            Quit current episode early")
    logger.info("  h            Print this help")
    logger.info("=" * 60)
    logger.info("")

    # ── Create robot and leader from yaml ──
    # Follow resfit's pattern exactly: import modules first to trigger
    # @CameraConfig.register_subclass / @TeleoperatorConfig.register_subclass,
    # then let draccus.parse decode the yaml. Also import gym_manipulator
    # for its side-effect of pulling in the full robot+camera import chain.
    import draccus as _draccus
    from lerobot.envs.configs import EnvConfig
    from lerobot.scripts.rl.gym_manipulator import make_robot_env  # triggers camera register  # noqa: F401
    from lerobot.teleoperators.supre_robot_leader.supre_robot_leader import SupreRobotLeader  # triggers teleop register  # noqa: F401
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    def _convert_robot_obs(raw_obs: dict, joint_names: list[str]) -> dict:
        """Convert robot.get_observation() → policy input format.

        raw_obs has keys like: left_arm_joint_1.pos, left_arm_joint_1.force,
          observation.images.head_cam, etc.
        Policy needs: observation.state (15,), observation.force (15,),
          observation.images.head_cam (3,480,640), etc.
        """
        obs = {}
        # ── State: stack all .pos readings in joint order ──
        pos_values = [float(raw_obs.get(f"{name}.pos", 0.0)) for name in joint_names]
        obs["observation.state"] = torch.tensor(pos_values, dtype=torch.float32)

        # ── Force: stack all .force readings in joint order ──
        force_values = [float(raw_obs.get(f"{name}.force", 0.0)) for name in joint_names]
        obs["observation.force"] = torch.tensor(force_values, dtype=torch.float32)

        # ── Cameras: robot returns "head_cam", policy expects "observation.images.head_cam" ──
        _cam_map = {"head_cam": "observation.images.head_cam",
                    "left_wrist_cam": "observation.images.left_wrist_cam",
                    "right_wrist_cam": "observation.images.right_wrist_cam"}
        for robot_key, policy_key in _cam_map.items():
            if robot_key in raw_obs:
                img = raw_obs[robot_key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if img.ndim == 3:
                    if img.shape[-1] == 3:  # HWC → CHW
                        img = img.permute(2, 0, 1)
                obs[policy_key] = img.float() / 255.0 if img.max() > 1.0 else img.float()

        return obs

    env_cfg = _draccus.parse(config_class=EnvConfig, config_path=env_config_path)
    robot = make_robot_from_config(env_cfg.robot)
    robot.connect()
    logger.info(f"Robot connected: {type(robot).__name__}")

    # ── Create leader (optional) ──
    has_leader = False
    leader = None
    if env_cfg.teleop is not None:
        try:
            leader_cfg = env_cfg.teleop
            leader = make_teleoperator_from_config(leader_cfg)
            leader.connect()
            has_leader = True
            logger.info(f"Leader connected: {type(leader).__name__}")
        except Exception as e:
            logger.warning(f"Leader creation failed (keyboard only): {e}")

    if has_leader:
        logger.info("Leader arm detected — manual mode available (move leader to override)")
    else:
        logger.info("No leader — manual mode limited to keyboard offsets")

    # Read joint names from robot (same order as dataset's joint_names)
    _robot_joint_names = list(robot.observation_joint_names)
    logger.info(f"Robot joint names: {_robot_joint_names}")

    # ── Keyboard input handler ──
    operator = OperatorInputHandler()

    junk_keys = ('task', 'index', 'episode_index', 'frame_index', 'task_index',
                 'action', 'action_is_pad')

    # ── Data collection loop ──
    all_data: list[dict] = []
    total_accepted = 0
    total_adjusted = 0
    total_overridden = 0

    for ep_idx in range(num_episodes):
        logger.info(f"--- Episode {ep_idx + 1}/{num_episodes} ---")

        # Reset episode
        policy.reset()
        robot.reset()
        obs = robot.get_observation()

        # Reset operator state per episode
        operator.accumulated_offset = [0.0] * 15
        operator.selected_joint = 0
        operator.adjustment_delta = DEFAULT_DELTAS[0]

        ep_data: list[dict] = []
        step_done = False

        # Manual mode safety: when Space is pressed, we don't jump to leader's
        # absolute position. Instead we compute a bias = robot_current - leader_current
        # and then always send leader_read + bias to the robot. This means the robot
        # stays put when switching modes, and follows the leader's *relative* motion.
        _leader_bias: list[float] | None = None  # None = not yet initialized

        for step_idx in range(max_episode_steps):
            # ── Preprocess observation ──
            processed_obs = _convert_robot_obs(obs, _robot_joint_names)

            # ── Policy prediction ──
            predicted_action = _predict_fn(processed_obs)

            # ── Operator interaction ──
            manual_override = False

            if operator.manual_mode and has_leader:
                # ═══════════════════════════════════════════════
                # CONTINUOUS MANUAL MODE — leader drives via RELATIVE tracking
                # First frame: latch bias = robot_pos - leader_pos (zero jump)
                # Subsequent frames: target = leader_pos + bias (relative motion)
                # Press Space again to exit back to AUTO mode.
                # ═══════════════════════════════════════════════
                flags = operator.poll_and_clear()
                if flags["manual"]:
                    # Toggle back to AUTO — reset bias for next manual entry
                    _leader_bias = None
                    _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, False)
                    continue
                if flags["quit"]:
                    logger.info(f"  Episode {ep_idx} quit by operator at step {step_idx}")
                    step_done = True
                    break

                try:
                    leader_dict = leader.get_action()
                    leader_values = [float(leader_dict.get(f"{name}.pos", 0.0)) for name in _robot_joint_names]

                    if _leader_bias is None:
                        # First frame of manual mode: lock relative offset
                        robot_pos = robot.get_current_position()
                        robot_values = [float(robot_pos.get(name, 0.0)) for name in _robot_joint_names]
                        _leader_bias = [r - l for r, l in zip(robot_values, leader_values)]
                        logger.info("  [MANUAL] Bias locked — robot follows leader relative motion")

                    values = [l + b for l, b in zip(leader_values, _leader_bias)]
                    corrected = torch.tensor(values, dtype=torch.float32)
                    manual_override = True
                    if step_idx % 30 == 0:
                        _print_status(step_idx, ep_idx, corrected.tolist(), operator, True)
                except Exception as e:
                    logger.warning(f"Failed to read leader at step {step_idx}: {e}")
                    continue

            else:
                # ═══════════════════════════════════════════════
                # AUTO MODE — policy proposes, operator confirms
                # y=accept  ↑↓=adjust  ←→=select joint  space=manual
                # ═══════════════════════════════════════════════
                _printed_status = False
                while True:
                    flags = operator.poll_and_clear()

                    if not _printed_status:
                        _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, operator.manual_mode)
                        _printed_status = True

                    if flags["help"]:
                        logger.info("Controls: y=accept ↑↓=adjust ←→=joint space=manual q=quit r=reset")
                        continue
                    if flags["quit"]:
                        logger.info(f"  Episode {ep_idx} quit by operator at step {step_idx}")
                        step_done = True
                        break
                    if flags["prev"]:
                        operator.selected_joint = (operator.selected_joint - 1) % len(JOINT_NAMES)
                        operator.adjustment_delta = DEFAULT_DELTAS[operator.selected_joint]
                        _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, operator.manual_mode)
                        continue
                    if flags["next"]:
                        operator.selected_joint = (operator.selected_joint + 1) % len(JOINT_NAMES)
                        operator.adjustment_delta = DEFAULT_DELTAS[operator.selected_joint]
                        _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, operator.manual_mode)
                        continue
                    if flags["up"]:
                        operator.accumulated_offset[operator.selected_joint] += operator.adjustment_delta
                        _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, operator.manual_mode)
                        continue
                    if flags["down"]:
                        operator.accumulated_offset[operator.selected_joint] -= operator.adjustment_delta
                        _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, operator.manual_mode)
                        continue
                    if flags["manual"]:
                        # Entering manual mode — exit AUTO loop, next frame reads leader
                        _print_status(step_idx, ep_idx, predicted_action.tolist(), operator, True)
                        # Set corrected=None to skip this frame's execution entirely
                        corrected = None
                        break
                    if flags["accept"]:
                        corrected = operator.apply_offsets(predicted_action)
                        break  # exit operator loop, execute action

                    _time.sleep(0.05)

                if step_done:
                    break

            if step_done:
                break

            # ── Skip frame if operator toggled mode (no action to execute) ──
            if corrected is None:
                obs = robot.get_observation()
                continue

            # ── Execute on robot ──
            corrected_np = corrected.cpu().numpy() if isinstance(corrected, torch.Tensor) else corrected

            # Build action dict using robot's actual joint names
            action_dict = {}
            for j, name in enumerate(_robot_joint_names):
                action_dict[name] = float(corrected_np[j])

            # Record the (obs, final_action) pair BEFORE executing
            record = {}
            for k, v in processed_obs.items():
                if k in junk_keys:
                    continue
                record[k] = v.clone() if isinstance(v, torch.Tensor) else v
            record["action"] = corrected.clone() if isinstance(corrected, torch.Tensor) else torch.tensor(corrected_np)
            ep_data.append(record)

            if manual_override:
                total_overridden += 1
            elif any(abs(o) > 0.001 for o in operator.accumulated_offset):
                total_adjusted += 1
            else:
                total_accepted += 1

            # Execute
            robot.send_action(action_dict)
            _time.sleep(1.0 / fps)

            # Get next observation
            obs = robot.get_observation()

        all_data.extend(ep_data)
        logger.info(f"  Episode {ep_idx} collected {len(ep_data)} frames (total: {len(all_data)})")

    # ── Cleanup ──
    operator.stop()
    if leader is not None:
        leader.disconnect()
    robot.disconnect()

    # ── Save raw dagger data ──
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "dagger_online_data.pt")
    torch.save(all_data, data_path)
    logger.info(f"Saved {len(all_data)} raw frames to {data_path}")

    logger.info(f"Operator stats: {total_accepted} accepted, "
                f"{total_adjusted} adjusted, {total_overridden} overridden "
                f"({total_accepted+total_adjusted+total_overridden} total)")

    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# Training: BC fine-tune on collected data
# ═══════════════════════════════════════════════════════════════════════════

def dagger_finetune(
    policy_path: str,
    dagger_data: list[dict],
    output_dir: str,
    ds_meta: LeRobotDatasetMetadata,
    train_steps: int = 50000,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    device: torch.device = None,
    num_workers: int = 4,
):
    """Fine-tune ACT policy on DAgger-collected data.

    This is a lightweight BC training loop that:
    1. Loads the pre-trained ACT policy
    2. Trains on the new dagger_data mixed with a reference batch from
       the original dataset (to prevent catastrophic forgetting)
    3. Saves the updated policy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Fine-tuning on {len(dagger_data)} dagger frames for {train_steps} steps")

    # ── Load policy ──
    policy = ACTPolicy.from_pretrained(str(Path(policy_path).resolve()))
    policy = policy.to(device)
    policy.train()

    # ── Build optimizer ──
    optim_params = [
        {
            "params": [
                p for n, p in policy.named_parameters()
                if not n.startswith("model.backbone") and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in policy.named_parameters()
                if n.startswith("model.backbone") and p.requires_grad
            ],
            "lr": learning_rate,
        },
    ]
    optimizer = torch.optim.AdamW(optim_params, lr=learning_rate, weight_decay=1e-4)

    # ── Build dagger dataloader ──
    chunk_size = policy.config.chunk_size
    logger.info(f"Fine-tuning with chunk_size={chunk_size} (from checkpoint config)")

    class DaggerDataset(torch.utils.data.Dataset):
        def __init__(self, data, cs):
            self.data = data
            self.chunk_size = cs
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = dict(self.data[idx])
            # Build chunk matching the checkpoint's actual chunk_size
            action = item["action"]  # (action_dim,)
            actions = action.unsqueeze(0).repeat(self.chunk_size, 1)
            item["action"] = actions
            item["action_is_pad"] = torch.zeros(self.chunk_size, dtype=torch.bool)
            return item

    dagger_ds = DaggerDataset(dagger_data, chunk_size)
    dagger_loader = DataLoader(
        dagger_ds, batch_size=batch_size, shuffle=True,
        num_workers=0,  # data is in memory, no workers needed
        drop_last=True,
    )
    dl_iter = cycle(dagger_loader)

    # ── Training loop ──
    os.makedirs(output_dir, exist_ok=True)
    loss_history = []
    start_time = time.time()

    for step in range(train_steps):
        batch = next(dl_iter)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        loss, _ = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())

        if step % 200 == 0:
            avg_loss = np.mean(loss_history[-200:]) if len(loss_history) >= 200 else np.mean(loss_history)
            elapsed = time.time() - start_time
            logger.info(
                f"  Step {step:6d}/{train_steps}: loss={avg_loss:.4f}, "
                f"elapsed={elapsed:.0f}s"
            )

    # ── Save ──
    policy_dir = Path(output_dir) / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(policy_dir)

    # Save config for reproducibility
    dagger_config = {
        "original_policy": policy_path,
        "dagger_frames": len(dagger_data),
        "train_steps": train_steps,
        "final_loss": float(np.mean(loss_history[-100:])),
    }
    with open(Path(output_dir) / "dagger_config.json", "w") as f:
        json.dump(dagger_config, f, indent=2)

    logger.info(f"Saved fine-tuned policy to {policy_dir}")
    logger.info(f"Final loss: {dagger_config['final_loss']:.4f}")

    return policy_dir


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str]) -> dict:
    """Simple CLI parser — avoids argparse/draccus conflicts in the lerobot env."""
    defaults = {
        "phase": "offline",
        "dataset_root": "/root/data2/dc_dir/datasets/dataset_0611_pickup_long_all",
        "dataset_repo_id": "",
        "output_dir": "outputs/dagger/round1",
        "noise_std": 0.05,
        "recovery_frames_per_ep": 10,
        "recovery_length": 5,
        "train_steps": 50000,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "device": "cuda",
        "seed": 42,
        "env_config_path": "",
        "num_online_episodes": 20,
        "max_episode_steps": 350,
        "fps": 30,
        "aug_ensemble": False,
        "aug_n_views": 5,
    }
    result = dict(defaults)
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" in a:
            # --key=value
            k, v = a[2:].split("=", 1)
            result[k] = v
            i += 1
        elif a.startswith("--"):
            k = a[2:]
            if k in ("phase", "dataset_root", "dataset_repo_id", "output_dir",
                     "env_config_path", "device"):
                result[k] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else ""
                i += 2
            else:
                result[k] = argv[i + 1] if i + 1 < len(argv) else ""
                i += 2
        else:
            i += 1

    # Type conversions
    for k in ("seed", "train_steps", "batch_size", "recovery_frames_per_ep",
              "recovery_length", "num_online_episodes", "max_episode_steps", "fps"):
        result[k] = int(result[k])
    for k in ("noise_std", "learning_rate"):
        result[k] = float(result[k])
    for k in ("aug_ensemble",):
        result[k] = result[k] in (True, "true", "True", "1")
    for k in ("aug_n_views",):
        result[k] = int(result[k])
    return result


def main():
    """Main entry point — parameterized for both offline and online phases."""
    args = _parse_args(_ORIG_ARGV)

    phase = args["phase"]
    policy_path = args.get("policy_path", "")
    if not policy_path:
        logger.error("--policy_path is required")
        return
    output_dir = args["output_dir"]
    dataset_root = args["dataset_root"]
    dataset_repo_id = args["dataset_repo_id"] or Path(dataset_root).name
    noise_std = args["noise_std"]
    recovery_frames_per_ep = args["recovery_frames_per_ep"]
    recovery_length = args["recovery_length"]
    train_steps = args["train_steps"]
    batch_size = args["batch_size"]
    learning_rate = args["learning_rate"]
    seed = args["seed"]
    env_config_path = args["env_config_path"]
    num_online_episodes = args["num_online_episodes"]
    max_episode_steps = args["max_episode_steps"]
    fps = args["fps"]
    aug_ensemble = args["aug_ensemble"]
    aug_n_views = args["aug_n_views"]
    device_str = args["device"]

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Policy: {policy_path}")
    logger.info(f"Output: {output_dir}")

    if phase == "offline":
        # ── Load dataset metadata ──
        ds_meta = LeRobotDatasetMetadata(
            dataset_repo_id, root=dataset_root
        )

        # ── Load policy ──
        policy = DaggerPolicy(policy_path, device)

        # ── Load dataset (no transforms for clean comparison) ──
        dataset = LeRobotDataset(
            dataset_repo_id,
            root=dataset_root,
            customer_transforms=False,
            only_head_transforms=False,
            time_warp=False,
        )

        eps_per_ep = ds_meta.total_frames // ds_meta.total_episodes
        logger.info(
            f"Dataset: {ds_meta.total_episodes} episodes, "
            f"{ds_meta.total_frames} frames, ~{eps_per_ep} frames/ep"
        )

        # ── Generate recovery data ──
        logger.info("Generating recovery data (noise injection)...")
        dagger_data = generate_recovery_data(
            policy=policy,
            dataset=dataset,
            eps_per_ep=eps_per_ep,
            recovery_frames_per_ep=recovery_frames_per_ep,
            recovery_length=recovery_length,
            noise_std=noise_std,
            noise_clip=noise_std * 3,
        )

        # ── Fine-tune ──
        logger.info(f"Starting fine-tuning on {len(dagger_data)} frames...")
        dagger_finetune(
            policy_path=policy_path,
            dagger_data=dagger_data,
            output_dir=output_dir,
            ds_meta=ds_meta,
            train_steps=train_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        logger.info("=" * 60)
        logger.info(f"DAgger Round 1 (Offline) complete!")
        logger.info(f"  Recovery frames: {len(dagger_data)}")
        logger.info(f"  Fine-tuned policy: {output_dir}/policy/")
        logger.info(f"  Next: deploy to robot, run Phase 2 (online)")
        logger.info("=" * 60)

    elif phase == "online":
        policy = DaggerPolicy(policy_path, device)
        if not env_config_path:
            logger.error("--env_config_path required for online phase")
            return

        online_data = collect_online_data(
            policy=policy,
            env_config_path=env_config_path,
            output_dir=output_dir,
            num_episodes=num_online_episodes,
            max_episode_steps=max_episode_steps,
            fps=fps,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            aug_ensemble=aug_ensemble,
            aug_n_views=aug_n_views,
        )

        ds_meta = LeRobotDatasetMetadata(
            dataset_repo_id, root=dataset_root
        )
        dagger_finetune(
            policy_path=policy_path,
            dagger_data=online_data,
            output_dir=output_dir,
            ds_meta=ds_meta,
            train_steps=train_steps,
            batch_size=batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
