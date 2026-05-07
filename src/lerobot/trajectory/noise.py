"""Noise injection for trajectory-based data collection.

Provides two noise modes:
- "white": i.i.d. Gaussian noise per frame (same as tep_force replay_record)
- "ou": Ornstein-Uhlenbeck process — temporally correlated smooth noise.
  OU noise has much less frame-to-frame jitter than white noise, making the
  perturbed trajectory look like a natural human variation rather than random
  twitching. This is the default for trajectory collection.

OU process: dx = theta * (0 - x) * dt + sigma * dW
  - theta (decay rate): controls how fast noise returns to 0. Higher = smoother.
  - sigma (volatility): controls noise magnitude.
  - dt: 1/fps (time step between frames).

The noise_std config parameter maps to sigma in degrees.
For OU mode, theta=2.0 provides good smoothing at 30fps (noise half-life ~0.35s).

All joint position keys use the ".pos" suffix format (e.g., "left_arm_joint_1.pos").
"""

import math
import numpy as np


class SmoothNoiseGenerator:
    """Ornstein-Uhlenbeck noise generator for temporally smooth perturbations.

    Unlike i.i.d. Gaussian noise which produces violent frame-to-frame jitter,
    OU noise is correlated across time — each frame's noise is a smooth
    continuation of the previous frame's noise. This makes the perturbed
    trajectory resemble natural human variability rather than random twitching.

    Usage:
        rng = np.random.default_rng(seed)
        noise_gen = SmoothNoiseGenerator(all_keys, noise_std=1.5, skip_keys=["joint_7"], rng=rng)
        for frame in trajectory:
            noisy_action = noise_gen.perturb(frame)
        noise_gen.reset()  # Clear accumulated noise before next episode

    The OU process is: dx = theta * (0 - x) * dt + sigma * sqrt(dt) * N(0,1)
    """

    def __init__(
        self,
        all_keys: list[str],
        noise_std: float,
        skip_keys: list[str] | None = None,
        rng: np.random.Generator | None = None,
        theta: float = 2.0,
        fps: int = 30,
    ):
        self.all_keys = all_keys
        self.noise_std = noise_std
        self.skip_keys = skip_keys or ["joint_7"]
        self.rng = rng or np.random.default_rng()
        self.theta = theta
        self.dt = 1.0 / fps

        # Initialize noise state per key (OU process state x)
        self.state: dict[str, float] = {}
        self._init_state()

    def _init_state(self):
        """Initialize noise state to zero for all non-skipped keys."""
        self.state = {}
        for key in self.all_keys:
            if not self._should_skip(key):
                self.state[key] = 0.0

    def _should_skip(self, key: str) -> bool:
        return any(skip_key in key for skip_key in self.skip_keys)

    def perturb(self, action: dict[str, float]) -> dict[str, float]:
        """Add temporally smooth OU noise to an action dict.

        Args:
            action: Action dict with {"joint_name.pos": float} keys.

        Returns:
            New action dict with OU noise added. Original dict is not modified.
        """
        noisy_action = {}
        for key, value in action.items():
            if self._should_skip(key):
                noisy_action[key] = value
            elif key in self.state:
                # OU process update: dx = theta * (0 - x) * dt + sigma * sqrt(dt) * N(0,1)
                x = self.state[key]
                dx = -self.theta * x * self.dt + self.noise_std * math.sqrt(self.dt) * self.rng.normal(0, 1)
                self.state[key] = x + dx
                noisy_action[key] = value + self.state[key]
            else:
                # Key not in state (e.g., new key) — use white noise
                noisy_action[key] = value + self.rng.normal(0, self.noise_std)
        return noisy_action

    def reset(self):
        """Reset noise state to zero. Call before each new episode."""
        self._init_state()


def add_noise_to_action(
    action: dict[str, float],
    noise_std: float,
    rng: np.random.Generator,
    skip_keys: list[str] | None = None,
) -> dict[str, float]:
    """Add i.i.d. Gaussian noise to action values (in degrees).

    This is the "white noise" mode — each frame gets independent noise.
    For temporally smooth noise, use SmoothNoiseGenerator instead.

    Args:
        action: Action dict with {"joint_name.pos": float} keys and degree values.
        noise_std: Standard deviation of noise in **degrees**.
        rng: NumPy random generator instance.
        skip_keys: Substrings to match against action keys. Matched keys get no noise.

    Returns:
        New action dict with noise added. Original dict is not modified.
    """
    if skip_keys is None:
        skip_keys = ["joint_7"]

    noisy_action = {}
    for key, value in action.items():
        if skip_keys and any(skip_key in key for skip_key in skip_keys):
            noisy_action[key] = value
        else:
            noise = rng.normal(0, noise_std)
            noisy_action[key] = value + noise
    return noisy_action