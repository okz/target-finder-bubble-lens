import pytest

from target_finder_toolkit.lens_core import (
    LensConfig,
    Point,
    Rect,
    TargetRect,
    choose_lens_rect,
    choose_source_crop,
    lens_to_source,
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
    placement = choose_lens_rect(_targets(800, 500), SCREEN)

    assert placement.side == "right"
    assert placement.rect.width == 360
    assert rect_intersection_area(placement.rect, placement.source_hull) == 0


def test_right_edge_cluster_uses_left_side():
    placement = choose_lens_rect(_targets(1800, 500), SCREEN)

    assert placement.side == "left"
    assert placement.rect.right < placement.source_hull.x


def test_bottom_cluster_prefers_right_when_it_still_fits():
    placement = choose_lens_rect(_targets(800, 1030), SCREEN)

    assert placement.side == "right"
    assert placement.rect.bottom == SCREEN.bottom


@pytest.mark.parametrize(
    "screen",
    [Rect(0, 0, 1366, 768), Rect(0, 0, 1920, 1080), Rect(100, 50, 1280, 720)],
)
def test_normal_placements_stay_on_screen_and_avoid_source(screen):
    targets = _targets(screen.x + screen.width / 2, screen.y + screen.height / 2)
    placement = choose_lens_rect(targets, screen)

    assert placement.rect.x >= screen.x
    assert placement.rect.y >= screen.y
    assert placement.rect.right <= screen.right
    assert placement.rect.bottom <= screen.bottom
    assert rect_intersection_area(placement.rect, placement.source_hull) == 0


def test_fallback_size_is_used_when_360_does_not_fit_beside_hull():
    screen = Rect(0, 0, 700, 500)
    targets = (
        TargetRect(id=1, x=280, y=210, width=60, height=60),
        TargetRect(id=2, x=350, y=210, width=60, height=60),
    )

    placement = choose_lens_rect(targets, screen)

    assert placement.used_fallback
    assert placement.rect.width <= 320
    assert placement.rect.right <= screen.right
    assert placement.rect.bottom <= screen.bottom


def test_source_crop_clamps_to_screen_edge():
    lens = Rect(400, 100, 360, 360)
    crop = choose_source_crop(Point(5, 5), lens, SCREEN, scale=3.0)

    assert crop == Rect(0, 0, 120, 120)


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
