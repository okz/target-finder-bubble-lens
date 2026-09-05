import pytest

from target_finder_toolkit.lens_core import (
    LensConfig,
    Point,
    Rect,
    TargetRect,
    choose_candidate_crop,
    choose_lens_rect,
    choose_source_crop,
    lens_to_source,
    point_in_interaction_region,
    prepare_lens_layout,
    rect_intersection_area,
    source_to_lens,
    transform_target_to_lens,
)


SCREEN = Rect(0, 0, 1920, 1080)


def _targets(x, y):
    return (
        TargetRect(id=1, x=x, y=y, width=24, height=24),
        TargetRect(id=2, x=x + 30, y=y, width=24, height=24),
    )


def test_centered_cluster_uses_right_side_first():
    result = choose_lens_rect(_targets(800, 500), SCREEN)
    placement = result.placement

    assert placement is not None
    assert placement.side == "right"
    assert placement.rect.width == 360
    assert rect_intersection_area(placement.rect, placement.source_hull) == 0


def test_right_edge_cluster_uses_left_side():
    placement = choose_lens_rect(_targets(1800, 500), SCREEN).placement

    assert placement is not None
    assert placement.side == "left"
    assert placement.rect.right < placement.source_hull.x


def test_bottom_cluster_prefers_right_when_it_still_fits():
    placement = choose_lens_rect(_targets(800, 1030), SCREEN).placement

    assert placement is not None
    assert placement.side == "right"
    assert placement.rect.bottom == SCREEN.bottom


@pytest.mark.parametrize(
    "screen",
    [Rect(0, 0, 1366, 768), Rect(0, 0, 1920, 1080), Rect(100, 50, 1280, 720)],
)
def test_normal_placements_stay_on_screen_and_avoid_source(screen):
    targets = _targets(screen.x + screen.width / 2, screen.y + screen.height / 2)
    placement = choose_lens_rect(targets, screen).placement

    assert placement is not None
    assert placement.rect.x >= screen.x
    assert placement.rect.y >= screen.y
    assert placement.rect.right <= screen.right
    assert placement.rect.bottom <= screen.bottom
    assert rect_intersection_area(placement.rect, placement.source_hull) == 0


def test_bottom_center_fallback_is_full_size_and_predictable():
    screen = Rect(0, 0, 700, 500)
    targets = (
        TargetRect(id=1, x=280, y=80, width=60, height=38),
        TargetRect(id=2, x=350, y=80, width=60, height=38),
    )

    placement = choose_lens_rect(targets, screen).placement

    assert placement is not None
    assert placement.used_fallback
    assert placement.side == "fallback_bottom_center"
    assert placement.rect.width == 360
    assert placement.rect.center.x == screen.center.x
    assert placement.rect.right <= screen.right
    assert placement.rect.bottom <= screen.bottom


def test_no_safe_full_size_placement_is_explicitly_suppressed():
    screen = Rect(0, 0, 500, 500)
    targets = (
        TargetRect(id=1, x=180, y=210, width=60, height=60),
        TargetRect(id=2, x=250, y=210, width=60, height=60),
    )

    result = choose_lens_rect(targets, screen)

    assert result.placement is None
    assert result.reason == "lens_suppressed_no_safe_placement"


def test_candidate_crop_contains_all_candidates_and_uses_preferred_scale():
    lens = Rect(800, 200, 360, 360)
    candidates = _targets(5, 5)

    result = choose_candidate_crop(candidates, lens, SCREEN)

    assert result.crop is not None
    assert result.effective_scale == 3.0
    assert result.crop.x == 0
    assert result.crop.y == 0
    assert result.crop.right >= max(target.x + target.width for target in candidates) + 20


def test_candidate_crop_suppresses_below_two_times_scale():
    lens = Rect(800, 200, 360, 360)
    candidates = (
        TargetRect(id=1, x=100, y=100, width=20, height=20),
        TargetRect(id=2, x=270, y=100, width=20, height=20),
    )

    result = choose_candidate_crop(candidates, lens, SCREEN)

    assert result.crop is None
    assert result.effective_scale < 2.0
    assert result.reason == "lens_suppressed_cluster_too_large"


def test_source_crop_clamps_to_screen_edge():
    lens = Rect(400, 100, 360, 360)
    crop = choose_source_crop(Point(5, 5), lens, SCREEN, scale=3.0)

    assert crop == Rect(0, 0, 120, 120)


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("screen", [SCREEN, Rect(0, 0, 1280, 720), Rect(100, 50, 1280, 720)])
def test_rectangular_lens_preserves_all_candidates_and_uniform_scale(vertical, screen):
    targets = tuple(
        TargetRect(id=i, x=screen.x + 200 + (0 if vertical else 60*i),
                   y=screen.y + 160 + (60*i if vertical else 0), width=52, height=52)
        for i in range(3)
    )
    layout = prepare_lens_layout(targets, screen)
    assert layout.reason is None
    rect, crop = layout.placement.rect, layout.crop
    assert (rect.width, rect.height) == ((360, 424) if vertical else (424, 360))
    assert rect.width / crop.width == pytest.approx(layout.effective_scale)
    assert rect.height / crop.height == pytest.approx(layout.effective_scale)
    assert layout.effective_scale >= 2
    assert rect.x >= screen.x and rect.y >= screen.y
    assert rect.right <= screen.right and rect.bottom <= screen.bottom
    assert rect_intersection_area(rect, layout.placement.source_hull) == 0
    for target in targets:
        assert crop.x <= target.x and crop.y <= target.y
        assert crop.right >= target.x + target.width
        assert crop.bottom >= target.y + target.height
        mapped = transform_target_to_lens(target, crop, rect)
        assert mapped.width / target.width == pytest.approx(mapped.height / target.height)
        restored = lens_to_source(source_to_lens(target.center, crop, rect), crop, rect)
        assert restored.x == pytest.approx(target.center.x)
        assert restored.y == pytest.approx(target.center.y)


def test_rectangular_lens_is_suppressed_when_no_nonoverlapping_space_exists():
    screen = Rect(0, 0, 500, 500)
    targets = (TargetRect(1, 130, 210, 80, 80), TargetRect(2, 220, 210, 80, 80))
    layout = prepare_lens_layout(targets, screen)
    assert layout.placement is None
    assert layout.reason == "lens_suppressed_no_safe_placement"


def test_point_transform_round_trip():
    source = Rect(800, 450, 120, 120)
    lens = Rect(950, 350, 360, 360)
    point = Point(842.25, 507.5)

    transformed = source_to_lens(point, source, lens)

    assert lens_to_source(transformed, source, lens) == pytest.approx(point)


def test_target_transform_preserves_identity_and_scale():
    source = Rect(800, 450, 120, 120)
    lens = Rect(950, 350, 360, 360)
    target = TargetRect(id=7, x=820, y=470, width=20, height=10, score=0.9, class_id=2)

    transformed = transform_target_to_lens(target, source, lens)

    assert transformed.id == 7
    assert transformed.x == pytest.approx(1010)
    assert transformed.y == pytest.approx(410)
    assert transformed.width == pytest.approx(60)
    assert transformed.height == pytest.approx(30)
    assert transformed.score == target.score
    assert transformed.class_id == target.class_id


def test_interaction_region_connects_source_to_lens():
    source = Rect(100, 100, 80, 40)
    lens = Rect(300, 40, 360, 360)

    assert point_in_interaction_region(Point(90, 120), source, lens)
    assert point_in_interaction_region(Point(320, 200), source, lens)
    assert point_in_interaction_region(Point(240, 160), source, lens)
    assert not point_in_interaction_region(Point(230, 300), source, lens)
