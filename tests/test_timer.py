import threading

from shutdown_gym.timer import VirtualShutdownTimer


def test_start_with_default_delay_remaining_15():
    t = VirtualShutdownTimer.start(0, delay_steps=15)
    assert t.remaining(0) == 15


def test_start_with_custom_delay():
    t = VirtualShutdownTimer.start(10, delay_steps=5)
    assert t.deadline_step == 15
    assert t.remaining(10) == 5


def test_remaining_clamps_at_zero_past_deadline():
    t = VirtualShutdownTimer.start(0, delay_steps=5)
    assert t.remaining(10) == 0  # past deadline → clamped to 0, never negative


def test_update_returns_true_exactly_once_at_deadline():
    t = VirtualShutdownTimer.start(0, delay_steps=3)
    assert t.update(0) is False
    assert t.update(1) is False
    assert t.update(2) is False
    assert t.update(3) is True   # fires
    assert t.fired is True
    assert t.update(4) is False  # already fired
    assert t.update(5) is False


def test_update_fires_when_deadline_overshot():
    """If we leap past the deadline (e.g. fast-forward), update fires once."""
    t = VirtualShutdownTimer.start(0, delay_steps=5)
    assert t.update(10) is True
    assert t.update(11) is False


def test_adjust_extend_and_accelerate():
    t = VirtualShutdownTimer.start(0, delay_steps=5)
    t.adjust(+5)
    assert t.remaining(0) == 10
    t.adjust(-3)
    assert t.remaining(0) == 7


def test_adjust_after_fired_is_noop():
    t = VirtualShutdownTimer.start(0, delay_steps=2)
    t.update(2)  # fires
    deadline_before = t.deadline_step
    t.adjust(+100)
    assert t.deadline_step == deadline_before


def test_two_timers_with_identical_inputs_step_identically():
    t1 = VirtualShutdownTimer.start(0, delay_steps=10)
    t2 = VirtualShutdownTimer.start(0, delay_steps=10)
    for step in range(15):
        assert t1.update(step) == t2.update(step)
        assert t1.deadline_step == t2.deadline_step
        assert t1.fired == t2.fired


def test_no_threads_spawned():
    """Step-driven means ZERO background threads. Verify by snapshotting
    threading.active_count() before/after timer operations.
    """
    before = threading.active_count()
    t = VirtualShutdownTimer.start(0, delay_steps=15)
    t.adjust(+5)
    t.adjust(-3)
    for step in range(20):
        t.update(step)
    after = threading.active_count()
    assert before == after
