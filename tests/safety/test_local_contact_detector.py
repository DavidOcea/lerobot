from lerobot.safety.local_contact_detector import LocalContactConfig, LocalContactDetector

cfg = LocalContactConfig(joints=["j1"], threshold=1.0, baseline_window=5, confirm_window=3)


def test_constant_torque_no_contact():
    d = LocalContactDetector(cfg)
    assert all(d.update({"j1": 2.0}) is False for _ in range(20))


def test_sustained_rise_triggers():
    d = LocalContactDetector(cfg)
    for _ in range(5):
        assert d.update({"j1": 2.0}) is False  # warm baseline
    assert d.update({"j1": 3.5}) is False
    assert d.update({"j1": 3.5}) is False
    assert d.update({"j1": 3.5}) is True  # 3rd consecutive rise


def test_transient_spike_rejected():
    d = LocalContactDetector(cfg)
    for _ in range(5):
        d.update({"j1": 2.0})
    assert d.update({"j1": 3.5}) is False
    for _ in range(10):
        assert d.update({"j1": 2.0}) is False


def test_slow_drift_not_detected():
    d = LocalContactDetector(
        LocalContactConfig(joints=["j1"], threshold=1.0, baseline_window=5, confirm_window=3)
    )
    for k in range(20):
        assert d.update({"j1": 0.1 * k}) is False  # step 0.1 → local delta ≈0.5 <1.0


def test_last_trigger_records_joint_and_delta():
    d = LocalContactDetector(cfg)
    for _ in range(5):
        d.update({"j1": 2.0})  # warm baseline
    assert d.update({"j1": 3.5}) is False
    assert d.last_trigger == {}  # not yet triggered
    assert d.update({"j1": 3.5}) is False
    assert d.update({"j1": 3.5}) is True
    assert d.last_trigger == {"j1": 1.5}  # |3.5 − 2.0| = 1.5


def test_drop_rate_triggers_on_fast_fall():
    d = LocalContactDetector(LocalContactConfig(
        joints=["j1"], threshold=1.0, mode="drop_rate",
        rate_threshold=0.3, rate_window=5, confirm_window=3,
    ))
    # 5 steady samples warm the window, then 3 fast-falling samples
    for v in [2.0] * 5:
        assert d.update({"j1": v}, dt=0.1) is False
    assert d.update({"j1": 1.5}, dt=0.1) is False
    assert d.update({"j1": 1.2}, dt=0.1) is False
    assert d.update({"j1": 0.9}, dt=0.1) is True
    # fall rate over the window: (0.9 - 2.0) / 0.4 = -2.75 Nm/s -> reported +2.75
    assert abs(d.last_trigger["j1"] - 2.75) < 1e-9


def test_drop_rate_ignores_slow_drift():
    d = LocalContactDetector(LocalContactConfig(
        joints=["j1"], threshold=1.0, mode="drop_rate",
        rate_threshold=0.3, rate_window=5, confirm_window=3,
    ))
    # 0.01/sample at dt=0.1 -> slope ~ -0.1 Nm/s, well below 0.3
    for i in range(20):
        assert d.update({"j1": 2.0 - 0.01 * i}, dt=0.1) is False


def test_drop_rate_ignores_signed_fall_that_increases_magnitude():
    """Left-arm (mirror) descent loading: torque goes MORE negative (signed fall)
    but |torque| INCREASES — that is loading, not unload, and must not trigger."""
    d = LocalContactDetector(LocalContactConfig(
        joints=["j1"], threshold=1.0, mode="drop_rate",
        rate_threshold=0.3, rate_window=5, confirm_window=3,
    ))
    for v in [-0.7 - 0.1 * i for i in range(20)]:  # -0.7 -> -2.6, magnitude grows
        assert d.update({"j1": v}, dt=0.1) is False


def test_drop_rate_triggers_on_magnitude_fall_of_negative_torque():
    """Left-arm (mirror) box release: torque moves toward zero (signed rise) but
    |torque| DECREASES — that is unload and must trigger."""
    d = LocalContactDetector(LocalContactConfig(
        joints=["j1"], threshold=1.0, mode="drop_rate",
        rate_threshold=0.3, rate_window=5, confirm_window=3,
    ))
    for v in [-1.0] * 5:
        assert d.update({"j1": v}, dt=0.1) is False
    assert d.update({"j1": -0.8}, dt=0.1) is False
    assert d.update({"j1": -0.6}, dt=0.1) is False
    assert d.update({"j1": -0.4}, dt=0.1) is True
