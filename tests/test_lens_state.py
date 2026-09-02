from target_finder_toolkit.lens_core import (
    LensStateMachine,
    LensStateName,
    PointerSample,
    TargetRect,
)


AMBIGUOUS_TARGETS = (
    TargetRect(id=1, x=100, y=100, width=20, height=20),
    TargetRect(id=2, x=125, y=100, width=20, height=20),
)
ONE_PLAUSIBLE_TARGET = (TargetRect(id=1, x=100, y=100, width=20, height=20),)


def _step(machine, t_ms, x=122.5, y=110, valid=True, targets=AMBIGUOUS_TARGETS, **kwargs):
    return machine.step(
        t_ms,
        PointerSample(t_ms=t_ms, x=x, y=y, valid=valid),
        targets,
        **kwargs,
    )


def test_two_close_targets_open_at_200_ms():
    machine = LensStateMachine()
    steps = [_step(machine, t) for t in range(0, 201, 20)]

    assert steps[5].state is LensStateName.NORMAL
    assert steps[6].state is LensStateName.PENDING
    assert "pending_started" in steps[6].events
    assert steps[-1].state is LensStateName.LENS_OPEN
    assert steps[-1].events == ("lens_opened",)
    assert steps[-1].frozen_candidate_ids == (1, 2)


def test_one_plausible_target_never_opens():
    machine = LensStateMachine()
    steps = [_step(machine, t, x=110, targets=ONE_PLAUSIBLE_TARGET) for t in range(0, 401, 20)]

    assert all(step.state is LensStateName.NORMAL for step in steps)
    assert not any("lens_opened" in step.events for step in steps)


def test_ambiguity_broken_at_190_ms_resets_persistence():
    machine = LensStateMachine()
    for t in range(0, 190, 10):
        _step(machine, t)

    broken = _step(machine, 190, x=100)
    next_step = _step(machine, 200)

    assert broken.state is LensStateName.NORMAL
    assert "pending_cancelled" in broken.events
    assert next_step.state is LensStateName.NORMAL
    assert "lens_opened" not in next_step.events


def test_invalid_samples_beyond_valid_ratio_do_not_open():
    machine = LensStateMachine()
    steps = []
    for index, t in enumerate(range(0, 241, 20)):
        steps.append(_step(machine, t, valid=(index % 3 != 0)))

    assert not any("lens_opened" in step.events for step in steps)


def test_missing_clean_frame_blocks_opening():
    machine = LensStateMachine()
    steps = [
        _step(machine, t, clean_frame_available=False)
        for t in range(0, 241, 20)
    ]

    assert steps[-1].state is LensStateName.PENDING
    assert not any("lens_opened" in step.events for step in steps)


def test_closing_starts_cooldown_and_prevents_immediate_reopen():
    machine = LensStateMachine()
    for t in range(0, 201, 20):
        opened = _step(machine, t)
    assert opened.state is LensStateName.LENS_OPEN

    closed = _step(machine, 220, close_requested=True)
    during = [_step(machine, t) for t in range(240, 620, 20)]
    expired = _step(machine, 620)

    assert closed.state is LensStateName.COOLDOWN
    assert closed.cooldown_until_ms == 620
    assert all(step.state is LensStateName.COOLDOWN for step in during)
    assert expired.state is LensStateName.NORMAL
    assert "cooldown_finished" in expired.events
    assert "lens_opened" not in expired.events


def test_timeout_enters_cooldown():
    machine = LensStateMachine()
    for t in range(0, 201, 20):
        _step(machine, t)

    timed_out = _step(machine, 3200)

    assert timed_out.state is LensStateName.COOLDOWN
    assert timed_out.events == ("lens_timed_out", "cooldown_started")
