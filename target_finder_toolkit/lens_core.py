"""Pure geometry and trigger logic for the ambiguity-triggered gaze lens.

This module deliberately has no GUI, detector, capture, or input-device imports.
All times are supplied by the caller in milliseconds so replay tests can be
fully deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
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
class Rect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("Rectangle width and height cannot be negative")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)


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
    fixation_drift_px: float = 50.0
    fixation_jump_floor_px: float = 12.0
    uncertainty_radius_px: float = 48.0
    ambiguity_threshold: float = 0.51
    pending_cue_ms: float = 120.0
    initial_transfer_protection_ms: float = 600.0
    outside_grace_ms: float = 1200.0
    selection_feedback_ms: float = 200.0
    fade_in_ms: float = 100.0
    test_watchdog_ms: float | None = None
    pointer_loss_timeout_ms: float = 500.0
    cooldown_ms: float = 400.0
    lens_size_px: float = 360.0
    lens_scale: float = 3.0
    lens_gap_px: float = 24.0
    source_hull_padding_px: float = 12.0
    interaction_padding_px: float = 24.0
    corridor_radius_px: float = 24.0
    crop_margin_px: float = 20.0
    minimum_lens_scale: float = 2.0
    source_change_pixel_threshold: float = 20.0
    source_change_fraction_threshold: float = 0.20
    confidence_threshold: float = 0.40
    duplicate_iou_threshold: float = 0.85
    duplicate_center_distance_px: float = 5.0
    include_text_targets: bool = False
    selectable_class_ids: frozenset[int] = SELECTABLE_CLASS_IDS

    def __post_init__(self) -> None:
        ratios = {
            "min_valid_ratio": self.min_valid_ratio,
            "ambiguity_threshold": self.ambiguity_threshold,
            "confidence_threshold": self.confidence_threshold,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
        }
        for name, value in ratios.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.trigger_window_ms <= 0 or self.min_valid_samples < 1:
            raise ValueError("Trigger window and minimum sample count must be positive")
        if self.fixation_drift_px <= 0 or self.fixation_jump_floor_px <= 0:
            raise ValueError("Fixation drift and jump limits must be positive")
        if self.pending_cue_ms > self.trigger_window_ms:
            raise ValueError("Pending cue cannot begin after the trigger window")
        if self.uncertainty_radius_px <= 0:
            raise ValueError("Uncertainty radius must be positive")
        if self.lens_size_px <= 0 or self.lens_scale <= 0:
            raise ValueError("Lens size and scale must be positive")
        if self.lens_gap_px < 0 or self.source_hull_padding_px < 0:
            raise ValueError("Lens gap and source padding cannot be negative")
        timings = (
            self.initial_transfer_protection_ms,
            self.outside_grace_ms,
            self.selection_feedback_ms,
            self.fade_in_ms,
            self.pointer_loss_timeout_ms,
            self.cooldown_ms,
        )
        if any(value < 0 for value in timings):
            raise ValueError("Lens lifecycle timings cannot be negative")
        if self.test_watchdog_ms is not None and self.test_watchdog_ms <= 0:
            raise ValueError("Test watchdog must be positive when configured")
        if self.interaction_padding_px < 0 or self.corridor_radius_px < 0:
            raise ValueError("Interaction padding and corridor radius cannot be negative")
        if self.crop_margin_px < 0:
            raise ValueError("Crop margin cannot be negative")
        if not 0 < self.minimum_lens_scale <= self.lens_scale:
            raise ValueError("Minimum lens scale must be positive and no greater than preferred scale")
        if self.source_change_pixel_threshold < 0:
            raise ValueError("Source change pixel threshold cannot be negative")
        if not 0 <= self.source_change_fraction_threshold <= 1:
            raise ValueError("Source change fraction threshold must be between 0 and 1")


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
    fixation_drift_px: float = inf
    latest_jump_px: float = inf
    valid_ratio: float = 0.0
    selection_noise_score: float = 0.0
    target_solution: AmbiguitySolution | None = None
    latest_solution: AmbiguitySolution | None = None
    sample_count: int = 0
    valid_sample_count: int = 0


@dataclass(frozen=True, slots=True)
class LensPlacement:
    rect: Rect
    source_hull: Rect
    side: str
    used_fallback: bool = False


@dataclass(frozen=True, slots=True)
class LensPlacementResult:
    placement: LensPlacement | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LensCropResult:
    crop: Rect | None
    effective_scale: float
    required_hull: Rect
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LensLayoutResult:
    placement: LensPlacement | None = None
    crop: Rect | None = None
    effective_scale: float = 0.0
    required_hull: Rect | None = None
    reason: str | None = None


class LensStateName(str, Enum):
    NORMAL = "NORMAL"
    PENDING = "PENDING"
    LENS_OPEN = "LENS_OPEN"
    EXIT_GRACE = "EXIT_GRACE"
    FEEDBACK = "FEEDBACK"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class LensStep:
    state: LensStateName
    events: tuple[str, ...]
    decision: WindowDecision
    ambiguous_for_ms: float = 0.0
    frozen_candidate_ids: tuple[int, ...] = ()
    cooldown_until_ms: float | None = None
    outside_until_ms: float | None = None
    feedback_until_ms: float | None = None


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


def rect_intersection_area(left: Rect, right: Rect) -> float:
    width = max(0.0, min(left.right, right.right) - max(left.x, right.x))
    height = max(0.0, min(left.bottom, right.bottom) - max(left.y, right.y))
    return width * height


def union_rect(targets: Iterable[TargetRect], padding: float = 0.0) -> Rect:
    targets = tuple(targets)
    if not targets:
        raise ValueError("At least one target is required")
    left = min(target.x for target in targets) - padding
    top = min(target.y for target in targets) - padding
    right = max(target.x + target.width for target in targets) + padding
    bottom = max(target.y + target.height for target in targets) + padding
    return Rect(left, top, right - left, bottom - top)


def _clamp_rect_axis(position: float, size: float, low: float, high: float) -> float:
    return _clamp(position, low, max(low, high - size))


def choose_lens_rect(
    candidates: Iterable[TargetRect],
    screen: Rect,
    config: LensConfig = LensConfig(),
) -> LensPlacementResult:
    """Choose a deterministic, fixed sidecar placement for a candidate cluster."""

    source_hull = union_rect(candidates, config.source_hull_padding_px)
    size = config.lens_size_px
    if size <= screen.width and size <= screen.height:
        positions = (
            (
                "right",
                source_hull.right + config.lens_gap_px,
                _clamp_rect_axis(
                    source_hull.center.y - size / 2.0,
                    size,
                    screen.y,
                    screen.bottom,
                ),
            ),
            (
                "left",
                source_hull.x - config.lens_gap_px - size,
                _clamp_rect_axis(
                    source_hull.center.y - size / 2.0,
                    size,
                    screen.y,
                    screen.bottom,
                ),
            ),
            (
                "below",
                _clamp_rect_axis(
                    source_hull.center.x - size / 2.0,
                    size,
                    screen.x,
                    screen.right,
                ),
                source_hull.bottom + config.lens_gap_px,
            ),
            (
                "above",
                _clamp_rect_axis(
                    source_hull.center.x - size / 2.0,
                    size,
                    screen.x,
                    screen.right,
                ),
                source_hull.y - config.lens_gap_px - size,
            ),
        )
        for side, x, y in positions:
            rect = Rect(x, y, size, size)
            entirely_on_screen = (
                rect.x >= screen.x
                and rect.y >= screen.y
                and rect.right <= screen.right
                and rect.bottom <= screen.bottom
            )
            if entirely_on_screen and rect_intersection_area(rect, source_hull) == 0.0:
                return LensPlacementResult(LensPlacement(rect, source_hull, side, False))

        dock = Rect(
            _clamp_rect_axis(
                screen.center.x - size / 2.0,
                size,
                screen.x,
                screen.right,
            ),
            screen.bottom - size,
            size,
            size,
        )
        if rect_intersection_area(dock, source_hull) == 0.0:
            return LensPlacementResult(
                LensPlacement(dock, source_hull, "fallback_bottom_center", True)
            )
    return LensPlacementResult(None, "lens_suppressed_no_safe_placement")


def choose_candidate_crop(
    candidates: Iterable[TargetRect],
    lens_rect: Rect,
    screen: Rect,
    config: LensConfig = LensConfig(),
) -> LensCropResult:
    """Fit a square crop around every candidate at a useful magnification."""

    padded = union_rect(candidates, config.crop_margin_px)
    required_left = max(screen.x, padded.x)
    required_top = max(screen.y, padded.y)
    required_right = min(screen.right, padded.right)
    required_bottom = min(screen.bottom, padded.bottom)
    if required_left >= required_right or required_top >= required_bottom:
        return LensCropResult(None, 0.0, padded, "lens_suppressed_cluster_too_large")
    required = Rect(
        required_left,
        required_top,
        required_right - required_left,
        required_bottom - required_top,
    )
    minimum_side = max(required.width, required.height)
    preferred_side = lens_rect.width / config.lens_scale
    side = max(minimum_side, preferred_side)
    if side > screen.width or side > screen.height:
        return LensCropResult(None, 0.0, required, "lens_suppressed_cluster_too_large")
    x = _clamp_rect_axis(required.center.x - side / 2.0, side, screen.x, screen.right)
    y = _clamp_rect_axis(required.center.y - side / 2.0, side, screen.y, screen.bottom)
    crop = Rect(x, y, side, side)
    effective_scale = min(
        config.lens_scale,
        lens_rect.width / crop.width,
        lens_rect.height / crop.height,
    )
    if effective_scale < config.minimum_lens_scale:
        return LensCropResult(
            None,
            effective_scale,
            required,
            "lens_suppressed_cluster_too_large",
        )
    return LensCropResult(crop, effective_scale, required)


def prepare_lens_layout(
    candidates: Iterable[TargetRect], screen: Rect, config: LensConfig = LensConfig(),
) -> LensLayoutResult:
    """Shared runtime/evaluation gate for displaying the actual candidate set."""
    candidates = filter_and_deduplicate(candidates, config)
    if len(candidates) < 2:
        return LensLayoutResult(reason="target_invalidated")
    placement = choose_lens_rect(candidates, screen, config)
    if placement.placement is None:
        return LensLayoutResult(reason=placement.reason)
    crop = choose_candidate_crop(candidates, placement.placement.rect, screen, config)
    return LensLayoutResult(
        placement=placement.placement, crop=crop.crop, effective_scale=crop.effective_scale,
        required_hull=crop.required_hull, reason=crop.reason,
    )


def expand_rect(rect: Rect, padding: float) -> Rect:
    if padding < 0:
        raise ValueError("Padding cannot be negative")
    return Rect(
        rect.x - padding,
        rect.y - padding,
        rect.width + 2 * padding,
        rect.height + 2 * padding,
    )


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return hypot(point.x - start.x, point.y - start.y)
    projection = _clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared,
        0.0,
        1.0,
    )
    nearest = Point(start.x + projection * dx, start.y + projection * dy)
    return hypot(point.x - nearest.x, point.y - nearest.y)


def point_in_interaction_region(
    point: Point,
    source_hull: Rect,
    lens_rect: Rect,
    config: LensConfig = LensConfig(),
) -> bool:
    source = expand_rect(source_hull, config.interaction_padding_px)
    lens = expand_rect(lens_rect, config.interaction_padding_px)
    in_source = source.x <= point.x <= source.right and source.y <= point.y <= source.bottom
    in_lens = lens.x <= point.x <= lens.right and lens.y <= point.y <= lens.bottom
    in_corridor = (
        point_segment_distance(point, source_hull.center, lens_rect.center)
        <= config.corridor_radius_px
    )
    return in_source or in_lens or in_corridor


def choose_source_crop(center: Point, lens_rect: Rect, screen: Rect, scale: float) -> Rect:
    if scale <= 0:
        raise ValueError("Scale must be positive")
    width = min(lens_rect.width / scale, screen.width)
    height = min(lens_rect.height / scale, screen.height)
    x = _clamp_rect_axis(center.x - width / 2.0, width, screen.x, screen.right)
    y = _clamp_rect_axis(center.y - height / 2.0, height, screen.y, screen.bottom)
    return Rect(x, y, width, height)


def source_to_lens(point: Point, source_crop: Rect, lens_rect: Rect) -> Point:
    return Point(
        lens_rect.x + (point.x - source_crop.x) * lens_rect.width / source_crop.width,
        lens_rect.y + (point.y - source_crop.y) * lens_rect.height / source_crop.height,
    )


def lens_to_source(point: Point, source_crop: Rect, lens_rect: Rect) -> Point:
    return Point(
        source_crop.x + (point.x - lens_rect.x) * source_crop.width / lens_rect.width,
        source_crop.y + (point.y - lens_rect.y) * source_crop.height / lens_rect.height,
    )


def transform_target_to_lens(
    target: TargetRect, source_crop: Rect, lens_rect: Rect
) -> TargetRect:
    top_left = source_to_lens(Point(target.x, target.y), source_crop, lens_rect)
    bottom_right = source_to_lens(
        Point(target.x + target.width, target.y + target.height), source_crop, lens_rect
    )
    return TargetRect(
        id=target.id,
        x=top_left.x,
        y=top_left.y,
        width=bottom_right.x - top_left.x,
        height=bottom_right.y - top_left.y,
        score=target.score,
        class_id=target.class_id,
    )


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
    split = max(1, len(valid) // 2)
    first_half = valid[:split]
    second_half = valid[split:] or valid[-1:]
    first_center = Point(
        median(sample.x for sample in first_half),
        median(sample.y for sample in first_half),
    )
    second_center = Point(
        median(sample.x for sample in second_half),
        median(sample.y for sample in second_half),
    )
    fixation_drift = hypot(
        second_center.x - first_center.x,
        second_center.y - first_center.y,
    )
    previous = valid[:-1] or valid[-1:]
    previous_center = Point(
        median(sample.x for sample in previous),
        median(sample.y for sample in previous),
    )
    previous_radii = [
        hypot(sample.x - previous_center.x, sample.y - previous_center.y)
        for sample in previous
    ]
    previous_r90 = _percentile(previous_radii, 90.0)
    latest_jump = hypot(
        valid[-1].x - previous_center.x,
        valid[-1].y - previous_center.y,
    )
    jump_limit = max(config.fixation_jump_floor_px, 3.0 * previous_r90)
    targets = filter_and_deduplicate(raw_targets, config)
    center_solution = ambiguity_solution(center, targets, config)
    latest_solution = ambiguity_solution(valid[-1].point, targets, config)
    if fixation_drift > config.fixation_drift_px or latest_jump > jump_limit:
        return WindowDecision(
            stable=False,
            ambiguous=False,
            center=center,
            r90_px=r90,
            fixation_drift_px=fixation_drift,
            latest_jump_px=latest_jump,
            valid_ratio=valid_ratio,
            target_solution=center_solution,
            latest_solution=latest_solution,
            sample_count=len(window),
            valid_sample_count=len(valid),
        )

    distance_margin = max(0.0, center_solution.d2 - center_solution.d1)
    if len(center_solution.plausible) < 2:
        selection_noise_score = 0.0
    elif distance_margin <= 0.5:
        selection_noise_score = 1.0
    else:
        selection_noise_score = r90 / (r90 + distance_margin + 0.5)
    center_solution = replace(center_solution, ambiguity_score=selection_noise_score)
    current_ambiguous = (
        len(center_solution.plausible) >= 2
        and selection_noise_score >= config.ambiguity_threshold
    )
    return WindowDecision(
        stable=True,
        ambiguous=current_ambiguous,
        center=center,
        r90_px=r90,
        fixation_drift_px=fixation_drift,
        latest_jump_px=latest_jump,
        valid_ratio=valid_ratio,
        selection_noise_score=selection_noise_score,
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
    _last_nonambiguous_ms: float | None = field(default=None, init=False)
    _opened_at_ms: float | None = field(default=None, init=False)
    _last_valid_ms: float | None = field(default=None, init=False)
    _outside_since_ms: float | None = field(default=None, init=False)
    _feedback_until_ms: float | None = field(default=None, init=False)
    _cooldown_until_ms: float | None = field(default=None, init=False)
    _frozen_candidate_ids: tuple[int, ...] = field(default=(), init=False)

    def _empty_decision(self) -> WindowDecision:
        return WindowDecision(stable=False, ambiguous=False)

    def _start_cooldown(self, now_ms: float, reason: str) -> LensStep:
        self.state = LensStateName.COOLDOWN
        self._cooldown_until_ms = now_ms + self.config.cooldown_ms
        self._opened_at_ms = None
        self._outside_since_ms = None
        self._feedback_until_ms = None
        self._ambiguous_since_ms = None
        self._last_nonambiguous_ms = None
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
        close_reason: str | None = None,
        pointer_in_interaction_region: bool = True,
        selection_requested: bool = False,
    ) -> LensStep:
        if sample.t_ms != now_ms:
            raise ValueError("sample.t_ms must equal now_ms for deterministic stepping")

        if self.state is LensStateName.FEEDBACK:
            if close_requested or close_reason is not None:
                return self._start_cooldown(now_ms, close_reason or "lens_closed")
            if self._feedback_until_ms is not None and now_ms >= self._feedback_until_ms:
                return self._start_cooldown(now_ms, "selection_feedback_finished")
            return LensStep(
                self.state,
                (),
                self._empty_decision(),
                frozen_candidate_ids=self._frozen_candidate_ids,
                feedback_until_ms=self._feedback_until_ms,
            )

        if (
            selection_requested
            and sample.valid
            and not close_requested
            and close_reason is None
            and self.state in (
                LensStateName.NORMAL, LensStateName.PENDING,
                LensStateName.LENS_OPEN, LensStateName.EXIT_GRACE,
            )
        ):
            self.state = LensStateName.FEEDBACK
            self._feedback_until_ms = now_ms + self.config.selection_feedback_ms
            self._outside_since_ms = None
            return LensStep(
                self.state,
                ("selection_feedback_started",),
                self._empty_decision(),
                frozen_candidate_ids=self._frozen_candidate_ids,
                feedback_until_ms=self._feedback_until_ms,
            )

        if self.state in (LensStateName.LENS_OPEN, LensStateName.EXIT_GRACE):
            if sample.valid:
                self._last_valid_ms = now_ms
            if close_requested or close_reason is not None:
                return self._start_cooldown(now_ms, close_reason or "lens_closed")
            if (
                self.config.test_watchdog_ms is not None
                and self._opened_at_ms is not None
                and now_ms - self._opened_at_ms >= self.config.test_watchdog_ms
            ):
                return self._start_cooldown(now_ms, "test_watchdog")
            if self._last_valid_ms is not None and now_ms - self._last_valid_ms >= self.config.pointer_loss_timeout_ms:
                return self._start_cooldown(now_ms, "pointer_lost")

            protected = (
                self._opened_at_ms is not None
                and now_ms - self._opened_at_ms < self.config.initial_transfer_protection_ms
            )
            if sample.valid and not protected:
                if pointer_in_interaction_region:
                    if self.state is LensStateName.EXIT_GRACE:
                        self.state = LensStateName.LENS_OPEN
                        self._outside_since_ms = None
                        return LensStep(
                            self.state,
                            ("exit_grace_cancelled",),
                            self._empty_decision(),
                            frozen_candidate_ids=self._frozen_candidate_ids,
                        )
                else:
                    if self._outside_since_ms is None:
                        self._outside_since_ms = now_ms
                        self.state = LensStateName.EXIT_GRACE
                        return LensStep(
                            self.state,
                            ("exit_grace_started",),
                            self._empty_decision(),
                            frozen_candidate_ids=self._frozen_candidate_ids,
                            outside_until_ms=now_ms + self.config.outside_grace_ms,
                        )
                    if now_ms - self._outside_since_ms >= self.config.outside_grace_ms:
                        return self._start_cooldown(now_ms, "outside_region")
            return LensStep(
                self.state,
                (),
                self._empty_decision(),
                frozen_candidate_ids=self._frozen_candidate_ids,
                outside_until_ms=(
                    None
                    if self._outside_since_ms is None
                    else self._outside_since_ms + self.config.outside_grace_ms
                ),
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
            self._last_nonambiguous_ms = now_ms
            events = cooldown_events + (("pending_cancelled",) if state_changed else ())
            return LensStep(self.state, events, decision)

        if self._ambiguous_since_ms is None:
            window_start = self._samples[0].t_ms
            self._ambiguous_since_ms = max(
                window_start,
                self._last_nonambiguous_ms
                if self._last_nonambiguous_ms is not None
                else window_start,
            )
        ambiguous_for = now_ms - self._ambiguous_since_ms
        candidate_ids = tuple(target.id for target in decision.target_solution.plausible)

        if ambiguous_for >= self.config.trigger_window_ms:
            if not clean_frame_available:
                self.state = LensStateName.NORMAL
                self._ambiguous_since_ms = now_ms
                return LensStep(
                    self.state,
                    cooldown_events + ("lens_suppressed_no_clean_frame",),
                    decision,
                    ambiguous_for,
                    candidate_ids,
                )
            self.state = LensStateName.LENS_OPEN
            self._opened_at_ms = now_ms
            self._last_valid_ms = now_ms
            self._outside_since_ms = None
            self._feedback_until_ms = None
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
    "LensLayoutResult",
    "LensCropResult",
    "LensPlacement",
    "LensPlacementResult",
    "LensStateMachine",
    "LensStateName",
    "LensStep",
    "Point",
    "Rect",
    "PointerSample",
    "SELECTABLE_CLASS_IDS",
    "TargetRect",
    "WindowDecision",
    "ambiguity_solution",
    "bubble_solution",
    "choose_lens_rect",
    "choose_candidate_crop",
    "choose_source_crop",
    "containment_distance",
    "evaluate_window",
    "filter_and_deduplicate",
    "expand_rect",
    "intersection_over_union",
    "lens_to_source",
    "point_rect_distance",
    "point_in_interaction_region",
    "point_segment_distance",
    "prepare_lens_layout",
    "rect_intersection_area",
    "source_to_lens",
    "transform_target_to_lens",
    "union_rect",
]
