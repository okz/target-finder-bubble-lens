from target_finder_toolkit.lens_core import (
    LensConfig,
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

    assert steps[-1].state is LensStateName.NORMAL
    assert not any("lens_opened" in step.events for step in steps)
    assert any("lens_suppressed_no_clean_frame" in step.events for step in steps)


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


def test_lens_has_no_human_timeout():
    machine = LensStateMachine()
    for t in range(0, 201, 20):
        _step(machine, t)

    still_open = _step(machine, 20_000)

    assert still_open.state is LensStateName.LENS_OPEN


def test_outside_grace_starts_after_transfer_protection_and_reentry_cancels_it():
    machine = LensStateMachine()
    for t in range(0, 201, 20):
        _step(machine, t)

    protected = _step(machine, 700, pointer_in_interaction_region=False)
    grace = _step(machine, 800, pointer_in_interaction_region=False)
    reentered = _step(machine, 1700, pointer_in_interaction_region=True)

    assert protected.state is LensStateName.LENS_OPEN
    assert grace.state is LensStateName.EXIT_GRACE
    assert grace.events == ("exit_grace_started",)
    assert reentered.state is LensStateName.LENS_OPEN
    assert reentered.events == ("exit_grace_cancelled",)


def test_remaining_outside_for_more_than_grace_closes_lens():
    machine = LensStateMachine()
    for t in range(0, 201, 20):
        _step(machine, t)

    _step(machine, 800, pointer_in_interaction_region=False)
    still_in_grace = _step(machine, 1700, pointer_in_interaction_region=False)
    closed = _step(machine, 2100, pointer_in_interaction_region=False)

    assert still_in_grace.state is LensStateName.EXIT_GRACE
    assert closed.state is LensStateName.COOLDOWN
    assert closed.events == ("outside_region", "cooldown_started")


def test_selection_feedback_is_visible_before_cooldown():
    machine = LensStateMachine()
    for t in range(0, 201, 20):
        _step(machine, t)

    feedback = _step(machine, 220, selection_requested=True)
    active = _step(machine, 400)
    finished = _step(machine, 420)

    assert feedback.state is LensStateName.FEEDBACK
    assert feedback.feedback_until_ms == 420
    assert active.state is LensStateName.FEEDBACK
    assert finished.state is LensStateName.COOLDOWN


def test_optional_watchdog_is_test_only_and_closes_when_configured():
    machine = LensStateMachine(LensConfig(test_watchdog_ms=15_000))
    for t in range(0, 201, 20):
        _step(machine, t)

    closed = _step(machine, 15_200)

    assert closed.state is LensStateName.COOLDOWN
    assert closed.events == ("test_watchdog", "cooldown_started")
