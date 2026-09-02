import pytest

from target_finder_toolkit.lens_core import (
    LensConfig,
    Point,
    PointerSample,
    TargetRect,
    ambiguity_solution,
    evaluate_window,
    filter_and_deduplicate,
)


AMBIGUOUS_TARGETS = (
    TargetRect(id=1, x=100, y=100, width=20, height=20),
    TargetRect(id=2, x=125, y=100, width=20, height=20),
)


def _samples(points, step_ms=20):
    return [PointerSample(index * step_ms, x, y) for index, (x, y) in enumerate(points)]


def test_two_close_targets_are_plausible_and_ambiguous():
    solution = ambiguity_solution(Point(122.5, 110), AMBIGUOUS_TARGETS)

    assert [target.id for target in solution.plausible] == [1, 2]
    assert solution.ambiguity_score == pytest.approx(1.0)


def test_one_plausible_target_is_not_ambiguous():
    far_targets = (
        AMBIGUOUS_TARGETS[0],
        TargetRect(id=2, x=300, y=100, width=20, height=20),
    )

    solution = ambiguity_solution(Point(110, 110), far_targets)

    assert [target.id for target in solution.plausible] == [1]
    assert solution.ambiguity_score == 0.0


def test_stable_window_uses_robust_center_and_r90():
    samples = _samples([(121, 109), (123, 111), (122, 110), (124, 108), (120, 112)])

    decision = evaluate_window(samples, AMBIGUOUS_TARGETS)

    assert decision.stable
    assert decision.center == Point(122, 110)
    assert decision.r90_px < 4
    assert decision.ambiguous


def test_unstable_window_is_rejected():
    samples = _samples([(70, 110), (170, 110)] * 5)

    decision = evaluate_window(samples, AMBIGUOUS_TARGETS)

    assert not decision.stable
    assert not decision.ambiguous
    assert decision.r90_px > 35


def test_invalid_samples_beyond_ratio_are_rejected():
    samples = [
        PointerSample(t, 122.5, 110, valid=(index % 4 != 0))
        for index, t in enumerate(range(0, 201, 20))
    ]

    decision = evaluate_window(samples, AMBIGUOUS_TARGETS)

    assert decision.valid_ratio < 0.8
    assert not decision.stable
    assert not decision.ambiguous


def test_text_detections_do_not_create_ambiguity_by_default():
    button = AMBIGUOUS_TARGETS[0]
    text = TargetRect(id=2, x=125, y=100, width=20, height=20, class_id=3)

    solution = ambiguity_solution(Point(122.5, 110), [button, text])

    assert [target.id for target in solution.plausible] == [1]
    assert not solution.ambiguity_score


def test_text_can_be_explicitly_included():
    button = AMBIGUOUS_TARGETS[0]
    text = TargetRect(id=2, x=125, y=100, width=20, height=20, class_id=3)

    solution = ambiguity_solution(
        Point(122.5, 110), [button, text], LensConfig(include_text_targets=True)
    )

    assert [target.id for target in solution.plausible] == [1, 2]


def test_near_identical_duplicates_collapse_to_higher_confidence():
    low = TargetRect(id=1, x=100, y=100, width=30, height=30, score=0.80)
    high = TargetRect(id=2, x=101, y=101, width=30, height=30, score=0.95)

    filtered = filter_and_deduplicate([low, high])

    assert filtered == (high,)


def test_adjacent_controls_do_not_collapse():
    assert filter_and_deduplicate(AMBIGUOUS_TARGETS) == AMBIGUOUS_TARGETS
