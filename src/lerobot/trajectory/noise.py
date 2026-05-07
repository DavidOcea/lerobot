"""Noise injection for trajectory-based data collection.

Adapted from tep_force branch's add_noise_to_action() but using degrees
instead of radians for noise_std, consistent with the rest of the system
where all position values are in degrees.

All joint position keys use the ".pos" suffix format (e.g., "left_arm_joint_1.pos").
"""

import numpy as np


def add_noise_to_action(
    action: dict[str, float],
    noise_std: float,
    rng: np.random.Generator,
    skip_keys: list[str] | None = None,
) -> dict[str, float]:
    """Add Gaussian noise to action values (in degrees).

    Args:
        action: Action dict with {"joint_name.pos": float} keys and degree values.
        noise_std: Standard deviation of noise in **degrees**.
                   Default 1.5° ≈ tep_force's 0.02 rad expressed in system's native unit.
        rng: NumPy random generator instance (from np.random.default_rng(seed)).
        skip_keys: List of substrings to match against action keys. Keys containing
                   any of these substrings will NOT receive noise. Defaults to ["joint_7"]
                   (gripper joints). Set to empty list [] to add noise to all joints.

    Returns:
        New action dict with noise added. Original dict is not modified.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> add_noise_to_action({"j1.pos": 10.0, "j7.pos": 0.5}, 1.5, rng, ["joint_7"])
        {"j1.pos": 10.xxx, "j7.pos": 0.5}  # j7 preserved, j1 perturbed
    """
    if skip_keys is None:
        skip_keys = ["joint_7"]

    noisy_action = {}
    for key, value in action.items():
        if skip_keys and any(skip_key in key for skip_key in skip_keys):
            noisy_action[key] = value  # No noise for skipped joints
        else:
            noise = rng.normal(0, noise_std)
            noisy_action[key] = value + noise
    return noisy_action