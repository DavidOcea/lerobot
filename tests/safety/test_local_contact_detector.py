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
