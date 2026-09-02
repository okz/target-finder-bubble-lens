import os
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from target_finder_toolkit.bubblegazelens import (
    LensOverlay,
    ReplayPointerProvider,
    SnapshotStore,
    TargetSnapshot,
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
