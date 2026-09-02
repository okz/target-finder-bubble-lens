import math

import pytest

from target_finder_toolkit.lens_core import (
    Point,
    TargetRect,
    bubble_solution,
    containment_distance,
    point_rect_distance,
)


RECT = TargetRect(id=1, x=10, y=20, width=30, height=40)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (Point(20, 30), 0.0),
        (Point(10, 20), 0.0),
        (Point(40, 60), 0.0),
        (Point(25, 20), 0.0),
        (Point(5, 40), 5.0),
        (Point(25, 70), 10.0),
        (Point(4, 12), 10.0),
    ],
)
def test_point_rectangle_distance(point, expected):
    assert point_rect_distance(point, RECT) == pytest.approx(expected)


def test_containment_distance_uses_farthest_corner():
    assert containment_distance(Point(10, 20), RECT) == pytest.approx(50.0)


def test_bubble_selects_nearest_and_second_nearest():
    point = Point(22, 10)
    nearest = TargetRect(id=1, x=10, y=20, width=20, height=20)
    second = TargetRect(id=2, x=35, y=20, width=20, height=20)
    far = TargetRect(id=3, x=100, y=20, width=20, height=20)

    solution = bubble_solution(point, [far, second, nearest])

    assert solution.primary == nearest
    assert solution.secondary == second
    assert solution.primary_distance == pytest.approx(10.0)


def test_bubble_radius_stays_strictly_clear_of_second_target():
    point = Point(22, 10)
    targets = [
        TargetRect(id=1, x=10, y=20, width=20, height=20),
        TargetRect(id=2, x=35, y=20, width=20, height=20),
    ]

    solution = bubble_solution(point, targets)

    assert solution.radius >= 0
    assert solution.radius < solution.secondary_distance


def test_equal_distance_tie_breaks_by_score_then_id():
    point = Point(20, 20)
    lower_score = TargetRect(id=1, x=0, y=0, width=10, height=10, score=0.8)
    higher_score = TargetRect(id=2, x=30, y=30, width=10, height=10, score=0.9)

    solution = bubble_solution(point, [lower_score, higher_score])

    assert solution.primary == higher_score
