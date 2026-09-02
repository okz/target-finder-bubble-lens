"""Pure geometry and trigger logic for the ambiguity-triggered gaze lens.

This module deliberately has no GUI, detector, capture, or input-device imports.
All times are supplied by the caller in milliseconds so replay tests can be
fully deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import hypot, inf, isfinite
from statistics import median
from typing import Iterable, Sequence


SELECTABLE_CLASS_IDS = frozenset({0, 1, 2, 4, 5})


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class TargetRect:
    id: int
    x: float
    y: float
    width: float
    height: float
    score: float = 1.0
    class_id: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Target width and height must be positive")

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        return (
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x, self.y + self.height),
            Point(self.x + self.width, self.y + self.height),
        )


@dataclass(frozen=True, slots=True)
class PointerSample:
    t_ms: float
    x: float
    y: float
    valid: bool = True

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass(frozen=True, slots=True)
class LensConfig:
    trigger_window_ms: float = 200.0
    min_valid_ratio: float = 0.80
    min_valid_samples: int = 2
    fixation_r90_px: float = 35.0
    uncertainty_radius_px: float = 48.0
    ambiguity_threshold: float = 0.65
    ambiguous_sample_ratio: float = 0.75
    pending_cue_ms: float = 120.0
    lens_timeout_ms: float = 3000.0
    pointer_loss_timeout_ms: float = 500.0
    cooldown_ms: float = 400.0
    confidence_threshold: float = 0.40
    duplicate_iou_threshold: float = 0.85
    duplicate_center_distance_px: float = 5.0
    include_text_targets: bool = False
    selectable_class_ids: frozenset[int] = SELECTABLE_CLASS_IDS

    def __post_init__(self) -> None:
        ratios = {
            "min_valid_ratio": self.min_valid_ratio,
            "ambiguity_threshold": self.ambiguity_threshold,
            "ambiguous_sample_ratio": self.ambiguous_sample_ratio,
            "confidence_threshold": self.confidence_threshold,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
        }
        for name, value in ratios.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.trigger_window_ms <= 0 or self.min_valid_samples < 1:
            raise ValueError("Trigger window and minimum sample count must be positive")
        if self.pending_cue_ms > self.trigger_window_ms:
            raise ValueError("Pending cue cannot begin after the trigger window")
        if self.uncertainty_radius_px <= 0:
            raise ValueError("Uncertainty radius must be positive")


@dataclass(frozen=True, slots=True)
class BubbleSolution:
    primary: TargetRect | None
    secondary: TargetRect | None
    primary_distance: float = inf
    secondary_distance: float = inf
    containment_distance: float = 0.0
    radius: float = 0.0


@dataclass(frozen=True, slots=True)
class AmbiguitySolution:
    primary: TargetRect | None
    ranked: tuple[tuple[float, TargetRect], ...]
    plausible: tuple[TargetRect, ...]
    ambiguity_score: float
    d1: float = inf
    d2: float = inf


@dataclass(frozen=True, slots=True)
class WindowDecision:
    stable: bool
    ambiguous: bool
    center: Point | None = None
    r90_px: float = inf
    valid_ratio: float = 0.0
    ambiguous_ratio: float = 0.0
    target_solution: AmbiguitySolution | None = None
    latest_solution: AmbiguitySolution | None = None
    sample_count: int = 0
    valid_sample_count: int = 0


class LensStateName(str, Enum):
    NORMAL = "NORMAL"
    PENDING = "PENDING"
    LENS_OPEN = "LENS_OPEN"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class LensStep:
    state: LensStateName
    events: tuple[str, ...]
    decision: WindowDecision
    ambiguous_for_ms: float = 0.0
    frozen_candidate_ids: tuple[int, ...] = ()
    cooldown_until_ms: float | None = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def point_rect_distance(point: Point, target: TargetRect) -> float:
    """Shortest Euclidean distance from a point to a closed rectangle."""

    center = target.center
    dx = max(abs(point.x - center.x) - target.width / 2.0, 0.0)
    dy = max(abs(point.y - center.y) - target.height / 2.0, 0.0)
    return hypot(dx, dy)


def containment_distance(point: Point, target: TargetRect) -> float:
    """Radius needed for a point-centred circle to contain the rectangle."""

    return max(hypot(point.x - corner.x, point.y - corner.y) for corner in target.corners)


def intersection_over_union(left: TargetRect, right: TargetRect) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union


def filter_and_deduplicate(
    targets: Iterable[TargetRect], config: LensConfig = LensConfig()
) -> tuple[TargetRect, ...]:
    """Filter non-selectable targets and collapse only near-identical boxes."""

    allowed = set(config.selectable_class_ids)
    if config.include_text_targets:
        allowed.add(3)
    filtered = [
        target
        for target in targets
        if target.class_id in allowed and target.score >= config.confidence_threshold
    ]
    # Considering high-confidence detections first makes the retained box explicit.
    filtered.sort(key=lambda target: (-target.score, target.id))
    kept: list[TargetRect] = []
    for candidate in filtered:
        duplicate = any(
            existing.class_id == candidate.class_id
            and intersection_over_union(existing, candidate) >= config.duplicate_iou_threshold
            and hypot(
                existing.center.x - candidate.center.x,
                existing.center.y - candidate.center.y,
            )
            <= config.duplicate_center_distance_px
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda target: target.id))


def _rank_targets(point: Point, targets: Iterable[TargetRect]) -> tuple[tuple[float, TargetRect], ...]:
    return tuple(
        sorted(
            ((point_rect_distance(point, target), target) for target in targets),
            key=lambda item: (item[0], -item[1].score, item[1].id),
        )
    )


def bubble_solution(
    point: Point,
    raw_targets: Iterable[TargetRect],
    config: LensConfig = LensConfig(),
) -> BubbleSolution:
    targets = filter_and_deduplicate(raw_targets, config)
    ranked = _rank_targets(point, targets)
    if not ranked:
        return BubbleSolution(primary=None, secondary=None)

    d1, primary = ranked[0]
    containment = containment_distance(point, primary)
    if len(ranked) == 1:
        return BubbleSolution(primary, None, d1, inf, containment, containment)

    d2, secondary = ranked[1]
    epsilon = max(0.5, 0.01 * d2)
    radius = max(0.0, min(containment, d2 - epsilon))
    return BubbleSolution(primary, secondary, d1, d2, containment, radius)


def ambiguity_solution(
    point: Point,
    raw_targets: Iterable[TargetRect],
    config: LensConfig = LensConfig(),
) -> AmbiguitySolution:
    targets = filter_and_deduplicate(raw_targets, config)
    ranked = _rank_targets(point, targets)
    if not ranked:
        return AmbiguitySolution(None, (), (), 0.0)

    d1, primary = ranked[0]
    d2 = ranked[1][0] if len(ranked) > 1 else inf
    plausible = tuple(target for distance, target in ranked if distance <= config.uncertainty_radius_px)
    if len(plausible) < 2 or not isfinite(d2):
        score = 0.0
    else:
        delta = d2 - d1
        score = 1.0 - _clamp(delta / config.uncertainty_radius_px, 0.0, 1.0)
    return AmbiguitySolution(primary, ranked, plausible, score, d1, d2)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return inf
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_window(
    samples: Sequence[PointerSample],
    raw_targets: Iterable[TargetRect],
    config: LensConfig = LensConfig(),
) -> WindowDecision:
    """Evaluate fixation and ambiguity for the supplied rolling window."""

    if not samples:
        return WindowDecision(stable=False, ambiguous=False)
    latest_t = samples[-1].t_ms
    window_start = latest_t - config.trigger_window_ms
    window = tuple(sample for sample in samples if window_start <= sample.t_ms <= latest_t)
    valid = tuple(sample for sample in window if sample.valid)
    valid_ratio = len(valid) / len(window) if window else 0.0
    if (
        len(valid) < config.min_valid_samples
        or valid_ratio < config.min_valid_ratio
        or not samples[-1].valid
    ):
        return WindowDecision(
            stable=False,
            ambiguous=False,
            valid_ratio=valid_ratio,
            sample_count=len(window),
            valid_sample_count=len(valid),
        )

    center = Point(median(sample.x for sample in valid), median(sample.y for sample in valid))
    radii = [hypot(sample.x - center.x, sample.y - center.y) for sample in valid]
    r90 = _percentile(radii, 90.0)
    targets = filter_and_deduplicate(raw_targets, config)
    center_solution = ambiguity_solution(center, targets, config)
    latest_solution = ambiguity_solution(valid[-1].point, targets, config)
    if r90 > config.fixation_r90_px:
        return WindowDecision(
            stable=False,
            ambiguous=False,
            center=center,
            r90_px=r90,
            valid_ratio=valid_ratio,
            target_solution=center_solution,
            latest_solution=latest_solution,
            sample_count=len(window),
            valid_sample_count=len(valid),
        )

    per_sample = [ambiguity_solution(sample.point, targets, config) for sample in valid]
    flags = [
        len(solution.plausible) >= 2
        and solution.ambiguity_score >= config.ambiguity_threshold
        for solution in per_sample
    ]
    ambiguous_ratio = sum(flags) / len(flags)
    current_ambiguous = (
        len(center_solution.plausible) >= 2
        and center_solution.ambiguity_score >= config.ambiguity_threshold
        and len(latest_solution.plausible) >= 2
        and latest_solution.ambiguity_score >= config.ambiguity_threshold
        and ambiguous_ratio >= config.ambiguous_sample_ratio
    )
    return WindowDecision(
        stable=True,
        ambiguous=current_ambiguous,
        center=center,
        r90_px=r90,
        valid_ratio=valid_ratio,
        ambiguous_ratio=ambiguous_ratio,
        target_solution=center_solution,
        latest_solution=latest_solution,
        sample_count=len(window),
        valid_sample_count=len(valid),
    )


@dataclass(slots=True)
class LensStateMachine:
    """Deterministic trigger state machine driven entirely by caller time."""

    config: LensConfig = field(default_factory=LensConfig)
    state: LensStateName = LensStateName.NORMAL
    _samples: deque[PointerSample] = field(default_factory=deque, init=False)
    _ambiguous_since_ms: float | None = field(default=None, init=False)
    _opened_at_ms: float | None = field(default=None, init=False)
    _last_valid_ms: float | None = field(default=None, init=False)
    _cooldown_until_ms: float | None = field(default=None, init=False)
    _frozen_candidate_ids: tuple[int, ...] = field(default=(), init=False)

    def _empty_decision(self) -> WindowDecision:
        return WindowDecision(stable=False, ambiguous=False)

    def _current_ambiguous_run_start(
        self, raw_targets: Iterable[TargetRect]
    ) -> float | None:
        """Return the first timestamp in the latest uninterrupted ambiguous run.

        Invalid samples are tolerated by the separate valid-ratio rule. A valid
        non-ambiguous sample, however, breaks persistence immediately.
        """

        targets = filter_and_deduplicate(raw_targets, self.config)
        run_start: float | None = None
        for sample in self._samples:
            if not sample.valid:
                continue
            solution = ambiguity_solution(sample.point, targets, self.config)
            ambiguous = (
                len(solution.plausible) >= 2
                and solution.ambiguity_score >= self.config.ambiguity_threshold
            )
            if ambiguous:
                if run_start is None:
                    run_start = sample.t_ms
            else:
                run_start = None
        return run_start

    def _start_cooldown(self, now_ms: float, reason: str) -> LensStep:
        self.state = LensStateName.COOLDOWN
        self._cooldown_until_ms = now_ms + self.config.cooldown_ms
        self._opened_at_ms = None
        self._ambiguous_since_ms = None
        self._samples.clear()
        frozen = self._frozen_candidate_ids
        self._frozen_candidate_ids = ()
        return LensStep(
            state=self.state,
            events=(reason, "cooldown_started"),
            decision=self._empty_decision(),
            frozen_candidate_ids=frozen,
            cooldown_until_ms=self._cooldown_until_ms,
        )

    def step(
        self,
        now_ms: float,
        sample: PointerSample,
        raw_targets: Iterable[TargetRect],
        *,
        clean_frame_available: bool = True,
        close_requested: bool = False,
    ) -> LensStep:
        if sample.t_ms != now_ms:
            raise ValueError("sample.t_ms must equal now_ms for deterministic stepping")

        if self.state is LensStateName.LENS_OPEN:
            if sample.valid:
                self._last_valid_ms = now_ms
            if close_requested:
                return self._start_cooldown(now_ms, "lens_closed")
            if self._opened_at_ms is not None and now_ms - self._opened_at_ms >= self.config.lens_timeout_ms:
                return self._start_cooldown(now_ms, "lens_timed_out")
            if self._last_valid_ms is not None and now_ms - self._last_valid_ms >= self.config.pointer_loss_timeout_ms:
                return self._start_cooldown(now_ms, "pointer_lost")
            return LensStep(
                self.state,
                (),
                self._empty_decision(),
                frozen_candidate_ids=self._frozen_candidate_ids,
            )

        if self.state is LensStateName.COOLDOWN:
            if self._cooldown_until_ms is not None and now_ms < self._cooldown_until_ms:
                return LensStep(
                    self.state,
                    (),
                    self._empty_decision(),
                    cooldown_until_ms=self._cooldown_until_ms,
                )
            self.state = LensStateName.NORMAL
            self._cooldown_until_ms = None
            self._samples.clear()
            self._ambiguous_since_ms = None
            cooldown_events = ("cooldown_finished",)
        else:
            cooldown_events = ()

        self._samples.append(sample)
        cutoff = now_ms - self.config.trigger_window_ms
        while self._samples and self._samples[0].t_ms < cutoff:
            self._samples.popleft()
        decision = evaluate_window(tuple(self._samples), raw_targets, self.config)

        if not decision.stable or not decision.ambiguous:
            state_changed = self.state is LensStateName.PENDING
            self.state = LensStateName.NORMAL
            self._ambiguous_since_ms = None
            events = cooldown_events + (("pending_cancelled",) if state_changed else ())
            return LensStep(self.state, events, decision)

        if self._ambiguous_since_ms is None:
            self._ambiguous_since_ms = self._current_ambiguous_run_start(raw_targets)
            if self._ambiguous_since_ms is None:
                self._ambiguous_since_ms = now_ms
        ambiguous_for = now_ms - self._ambiguous_since_ms
        candidate_ids = tuple(target.id for target in decision.target_solution.plausible)

        if ambiguous_for >= self.config.trigger_window_ms and clean_frame_available:
            self.state = LensStateName.LENS_OPEN
            self._opened_at_ms = now_ms
            self._last_valid_ms = now_ms
            self._frozen_candidate_ids = candidate_ids
            return LensStep(
                self.state,
                cooldown_events + ("lens_opened",),
                decision,
                ambiguous_for,
                candidate_ids,
            )

        if ambiguous_for >= self.config.pending_cue_ms:
            entered = self.state is not LensStateName.PENDING
            self.state = LensStateName.PENDING
            events = cooldown_events + (("pending_started",) if entered else ())
            return LensStep(self.state, events, decision, ambiguous_for, candidate_ids)

        self.state = LensStateName.NORMAL
        return LensStep(self.state, cooldown_events, decision, ambiguous_for, candidate_ids)


__all__ = [
    "AmbiguitySolution",
    "BubbleSolution",
    "LensConfig",
    "LensStateMachine",
    "LensStateName",
    "LensStep",
    "Point",
    "PointerSample",
    "SELECTABLE_CLASS_IDS",
    "TargetRect",
    "WindowDecision",
    "ambiguity_solution",
    "bubble_solution",
    "containment_distance",
    "evaluate_window",
    "filter_and_deduplicate",
    "intersection_over_union",
    "point_rect_distance",
]
