import os
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from target_finder_toolkit.bubblegazelens import (
    LensOverlay,
    ReplayPointerProvider,
    SnapshotStore,
    TargetSnapshot,
    load_replay_samples,
    source_change_fraction,
)
from target_finder_toolkit.lens_core import (
    LensConfig,
    LensStateName,
    PointerSample,
    Rect,
    TargetRect,
    source_to_lens,
)


def _frame(width=1280, height=720):
    frame = np.full((height, width, 3), 242, dtype=np.uint8)
    frame[80:640, 80:1200] = (232, 232, 232)
    return frame


def _targets():
    return (
        TargetRect(id=1, x=400, y=300, width=28, height=28, score=0.95, class_id=0),
        TargetRect(id=2, x=434, y=300, width=28, height=28, score=0.94, class_id=0),
    )


def _open_overlay(qtbot):
    snapshot = TargetSnapshot(1, _targets(), _frame())
    trace = [PointerSample(t, 431, 314) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
    )
    qtbot.addWidget(overlay)
    for _ in trace:
        overlay.tick()
    assert overlay.machine.state is LensStateName.LENS_OPEN
    assert overlay.frozen_lens is not None
    return overlay, snapshot


def test_snapshot_store_copies_frame_and_keeps_generation_consistent():
    store = SnapshotStore()
    frame = _frame(20, 10)
    detections = [
        {"id": 7, "x": 1, "y": 2, "width": 3, "height": 4, "score": 0.9, "class_id": 0}
    ]

    store.update(detections, [], [], frame)
    frame[:] = 0
    snapshot = store.read()

    assert snapshot.generation == 1
    assert snapshot.targets[0].id == 7
    assert snapshot.frame[0, 0, 0] == 242


def test_offscreen_lens_render_is_nonempty_and_fixed(qtbot, tmp_path):
    overlay, snapshot = _open_overlay(qtbot)
    original_rect = overlay.frozen_lens.placement.rect

    image = overlay.render_to_image()
    output = tmp_path / "lens.png"
    assert image.save(str(output))
    assert output.stat().st_size > 1000
    assert image.hasAlphaChannel()

    destination = source_to_lens(
        snapshot.targets[0].center,
        overlay.frozen_lens.source_crop,
        overlay.frozen_lens.placement.rect,
    )
    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(220, destination.x, destination.y)]
    )
    overlay.tick()

    assert overlay.frozen_lens.placement.rect == original_rect
    assert overlay.selected_target_id == 1


def test_pending_preview_and_lens_fade_are_deterministic(qtbot):
    snapshot = TargetSnapshot(1, _targets(), _frame())
    pending_trace = [PointerSample(t, 431, 314) for t in range(0, 121, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(pending_trace),
        LensConfig(),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
    )
    qtbot.addWidget(overlay)
    for _ in pending_trace:
        overlay.tick()
    assert overlay.machine.state is LensStateName.PENDING
    assert overlay._pending_placement is not None

    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(t, 431, 314) for t in (140, 160, 180, 200)]
    )
    for _ in range(4):
        overlay.tick()
    lens = overlay.frozen_lens.placement.rect
    probe_x = int(lens.x + 50)
    probe_y = int(lens.y + lens.height - 50)
    assert overlay.render_to_image().pixelColor(probe_x, probe_y).alpha() == 0

    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(250, 431, 314), PointerSample(300, 431, 314)]
    )
    overlay.tick()
    half_alpha = overlay.render_to_image().pixelColor(probe_x, probe_y).alpha()
    overlay.tick()
    full_alpha = overlay.render_to_image().pixelColor(probe_x, probe_y).alpha()

    assert 165 <= half_alpha <= 200
    assert full_alpha >= 250


def test_dry_run_confirmation_logs_selection_and_never_clicks(qtbot, tmp_path):
    from target_finder_toolkit.bubblegazelens import JsonlLogger

    log_path = tmp_path / "events.jsonl"
    logger = JsonlLogger(log_path, LensConfig())
    snapshot = TargetSnapshot(1, _targets(), _frame())
    trace = [PointerSample(t, 431, 314) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(),
        logger,
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
    )
    qtbot.addWidget(overlay)
    for _ in trace:
        overlay.tick()
    destination = source_to_lens(
        snapshot.targets[1].center,
        overlay.frozen_lens.source_crop,
        overlay.frozen_lens.placement.rect,
    )
    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(220, destination.x, destination.y)]
    )
    overlay.tick()
    overlay.confirm_selection()
    logger.close()

    content = log_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in content.splitlines()]
    assert '"event": "selection_dry_run"' in content
    assert '"target_id": 2' in content
    assert all(isinstance(event, dict) for event in events)
    opened = next(event for event in events if event.get("event") == "lens_opened")
    assert opened["selection_diagnostic"]["interpretation"] == (
        "fixation-center offset from current Bubble winner"
    )


def test_changed_candidate_set_invalidates_open_lens(qtbot):
    current = [TargetSnapshot(1, _targets(), _frame())]
    trace = [PointerSample(t, 431, 314) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: current[0],
        ReplayPointerProvider(trace),
        LensConfig(),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
    )
    qtbot.addWidget(overlay)
    for _ in trace:
        overlay.tick()
    assert overlay.machine.state is LensStateName.LENS_OPEN

    current[0] = TargetSnapshot(2, (_targets()[0],), _frame())
    overlay.pointer_provider = ReplayPointerProvider([PointerSample(220, 431, 314)])
    overlay.tick()

    assert overlay.machine.state is LensStateName.COOLDOWN
    assert "target_invalidated" in overlay._last_step.events


def test_significant_source_change_invalidates_open_lens(qtbot):
    current = [TargetSnapshot(1, _targets(), _frame())]
    trace = [PointerSample(t, 431, 314) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: current[0],
        ReplayPointerProvider(trace),
        LensConfig(),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
    )
    qtbot.addWidget(overlay)
    for _ in trace:
        overlay.tick()

    current[0] = TargetSnapshot(2, _targets(), np.zeros_like(_frame()))
    overlay.pointer_provider = ReplayPointerProvider([PointerSample(220, 431, 314)])
    overlay.tick()

    assert overlay.machine.state is LensStateName.COOLDOWN
    assert "source_changed" in overlay._last_step.events


def test_scroll_signal_invalidates_without_consuming_a_wheel_event(qtbot):
    overlay, _snapshot = _open_overlay(qtbot)

    overlay.request_scroll_close()
    overlay.pointer_provider = ReplayPointerProvider([PointerSample(220, 431, 314)])
    overlay.tick()

    assert overlay.machine.state is LensStateName.COOLDOWN
    assert "source_scrolled" in overlay._last_step.events


def test_selection_enters_feedback_before_closing(qtbot):
    overlay, snapshot = _open_overlay(qtbot)
    destination = source_to_lens(
        snapshot.targets[0].center,
        overlay.frozen_lens.source_crop,
        overlay.frozen_lens.placement.rect,
    )
    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(220, destination.x, destination.y)]
    )
    overlay.tick()
    overlay.confirm_selection()
    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(240, destination.x, destination.y), PointerSample(440, destination.x, destination.y)]
    )

    overlay.tick()
    assert overlay.machine.state is LensStateName.FEEDBACK
    assert overlay.selected_target_id == 1
    overlay.tick()
    assert overlay.machine.state is LensStateName.COOLDOWN


def test_source_change_fraction_uses_per_pixel_mean_bgr_threshold():
    frozen = np.zeros((10, 10, 3), dtype=np.uint8)
    current = frozen.copy()
    current[:2, :, :] = 30

    fraction = source_change_fraction(
        frozen,
        current,
        Rect(0, 0, 10, 10),
        Rect(0, 0, 10, 10),
        20,
    )

    assert fraction == pytest.approx(0.20)


def test_opening_is_suppressed_when_candidates_cannot_reach_two_times_scale(qtbot):
    targets = (
        TargetRect(id=1, x=350, y=280, width=80, height=80, score=0.95, class_id=0),
        TargetRect(id=2, x=440, y=280, width=80, height=80, score=0.94, class_id=0),
    )
    snapshot = TargetSnapshot(1, targets, _frame())
    trace = [PointerSample(t, 435, 320) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
        interaction_mode="forced-lens",
    )
    qtbot.addWidget(overlay)
    for _ in trace:
        overlay.tick()

    assert overlay.machine.state is LensStateName.COOLDOWN
    assert overlay.frozen_lens is None
    assert "lens_suppressed_cluster_too_large" in overlay._last_step.events


def test_tracking_loss_closes_an_open_lens(qtbot):
    snapshot = TargetSnapshot(1, _targets(), _frame())
    trace = [PointerSample(t, 431, 314) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(pointer_loss_timeout_ms=500),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
    )
    qtbot.addWidget(overlay)
    for _ in trace:
        overlay.tick()
    assert overlay.machine.state is LensStateName.LENS_OPEN

    invalid_trace = [PointerSample(t, 431, 314, valid=False) for t in range(220, 701, 20)]
    overlay.pointer_provider = ReplayPointerProvider(invalid_trace)
    for _ in invalid_trace:
        overlay.tick()

    assert overlay.machine.state is LensStateName.COOLDOWN
    assert "pointer_lost" in overlay._last_step.events


def test_bubble_mode_never_opens_the_lens(qtbot):
    snapshot = TargetSnapshot(1, _targets(), _frame())
    trace = [PointerSample(t, 431, 314) for t in range(0, 241, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
        interaction_mode="bubble",
    )
    qtbot.addWidget(overlay)

    for _ in trace:
        overlay.tick()

    assert overlay.machine.state is LensStateName.NORMAL
    assert overlay.frozen_lens is None


def test_forced_lens_ignores_nearest_target_dominance(qtbot):
    snapshot = TargetSnapshot(1, _targets(), _frame())
    trace = [PointerSample(t, 428, 314) for t in range(0, 201, 20)]
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(ambiguity_threshold=1.0),
        screen_rect=Rect(0, 0, 1280, 720),
        start_timer=False,
        interaction_mode="forced-lens",
    )
    qtbot.addWidget(overlay)

    for _ in trace:
        overlay.tick()

    assert overlay.machine.state is LensStateName.LENS_OPEN
    assert overlay.frozen_lens is not None


def test_load_replay_samples_validates_the_coordinate_contract(tmp_path):
    replay = tmp_path / "trace.jsonl"
    replay.write_text(
        '{"t_ms": 0, "x": 200, "y": 300, "valid": true, "screen": "primary"}\n'
        '{"t_ms": 20, "x": 201, "y": 301, "valid": false, "screen": "primary"}\n',
        encoding="utf-8",
    )

    samples = load_replay_samples(replay)

    assert samples == (
        PointerSample(0, 200, 300, True),
        PointerSample(20, 201, 301, False),
    )

    replay.write_text(
        '{"t_ms": 0, "x": 0.5, "y": 0.5, "valid": true, "screen": "primary"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="logical pixels"):
        load_replay_samples(replay)
