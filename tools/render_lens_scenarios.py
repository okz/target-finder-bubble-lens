"""Render a deterministic Bubble Gaze Lens scenario contact sheet."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from PyQt6 import QtCore, QtGui, QtWidgets

from target_finder_toolkit.bubblegazelens import (
    LensOverlay,
    ReplayPointerProvider,
    TargetSnapshot,
)
from target_finder_toolkit.lens_core import (
    LensConfig,
    LensStateName,
    PointerSample,
    Rect,
    TargetRect,
    rect_intersection_area,
    source_to_lens,
)


def _targets(count: int, x: float, y: float, size: float, gap: float) -> tuple[TargetRect, ...]:
    class_ids = (0, 1, 2, 4, 5, 0)
    return tuple(
        TargetRect(
            id=index + 1,
            x=x + index * (size + gap),
            y=y,
            width=size,
            height=size,
            score=0.98 - index * 0.01,
            class_id=class_ids[index % len(class_ids)],
        )
        for index in range(count)
    )


def render_presets() -> tuple[dict, ...]:
    return (
        {"name": "center_1920x1080_two", "screen": (1920, 1080), "targets": _targets(2, 860, 510, 24, 8)},
        {"name": "left_1366x768_three", "screen": (1366, 768), "targets": _targets(3, 20, 350, 24, 7)},
        {"name": "right_1366x768_three", "screen": (1366, 768), "targets": _targets(3, 1260, 350, 24, 7)},
        {"name": "top_1920x1080_six", "screen": (1920, 1080), "targets": _targets(6, 820, 18, 20, 2)},
        {"name": "bottom_1920x1080_six", "screen": (1920, 1080), "targets": _targets(6, 820, 1040, 20, 2)},
        {"name": "top_right_1280x720_two", "screen": (1280, 720), "targets": _targets(2, 1190, 24, 26, 8)},
        {"name": "fallback_700x500_large", "screen": (700, 500), "targets": _targets(2, 280, 80, 38, 32)},
        {"name": "logical_1280x720_dpr_1_5", "screen": (1280, 720), "targets": _targets(3, 530, 330, 22, 7)},
        {"name": "rectangular_horizontal_1280x720", "screen": (1280, 720), "targets": _targets(2, 350, 280, 80, 10)},
        {"name": "rectangular_vertical_1280x720", "screen": (1280, 720), "targets": (
            TargetRect(1, 350, 220, 80, 80, .98, 0),
            TargetRect(2, 350, 310, 80, 80, .97, 1),
        )},
    )


def _frame(width: int, height: int, targets: tuple[TargetRect, ...]) -> np.ndarray:
    frame = np.full((height, width, 3), (244, 244, 244), dtype=np.uint8)
    frame[: min(54, height)] = (42, 38, 34)
    if height > 120 and width > 120:
        frame[75 : height - 40, 40 : width - 40] = (235, 235, 235)
    colors = ((72, 126, 235), (92, 185, 92), (230, 150, 54), (190, 100, 210), (200, 180, 70))
    for index, target in enumerate(targets):
        x1 = max(0, int(target.x))
        y1 = max(0, int(target.y))
        x2 = min(width, int(target.x + target.width))
        y2 = min(height, int(target.y + target.height))
        frame[y1:y2, x1:x2] = colors[index % len(colors)]
    return frame


def _qimage(frame: np.ndarray) -> QtGui.QImage:
    contiguous = np.ascontiguousarray(frame)
    height, width, _ = contiguous.shape
    return QtGui.QImage(
        contiguous.data,
        width,
        height,
        contiguous.strides[0],
        QtGui.QImage.Format.Format_BGR888,
    ).copy()


def _render_one(preset: dict, output_dir: Path) -> dict:
    width, height = preset["screen"]
    targets = preset["targets"]
    frame = _frame(width, height, targets)
    first, second = targets[0], targets[1]
    point_x = (first.center.x + second.center.x) / 2.0
    point_y = (first.center.y + second.center.y) / 2.0
    trace = [PointerSample(t, point_x, point_y) for t in range(0, 201, 20)]
    snapshot = TargetSnapshot(1, targets, frame)
    overlay = LensOverlay(
        lambda: snapshot,
        ReplayPointerProvider(trace),
        LensConfig(),
        screen_rect=Rect(0, 0, width, height),
        start_timer=False,
    )
    for _ in trace:
        overlay.tick()
    if overlay.machine.state is not LensStateName.LENS_OPEN or overlay.frozen_lens is None:
        raise RuntimeError(f"Scenario did not open the lens: {preset['name']}")

    frozen = overlay.frozen_lens
    destination = source_to_lens(first.center, frozen.source_crop, frozen.placement.rect)
    overlay.pointer_provider = ReplayPointerProvider(
        [PointerSample(300, destination.x, destination.y)]
    )
    overlay.tick()

    canvas = _qimage(frame)
    painter = QtGui.QPainter(canvas)
    painter.drawImage(0, 0, overlay.render_to_image())
    painter.end()
    output_path = output_dir / f"{preset['name']}.png"
    if not canvas.save(str(output_path)):
        raise RuntimeError(f"Could not save {output_path}")

    overlap = rect_intersection_area(frozen.placement.rect, frozen.placement.source_hull)
    result = {
        "name": preset["name"],
        "passed": overlay.selected_target_id == first.id and overlap == 0,
        "screen": [width, height],
        "placement": frozen.placement.side,
        "lens_rect": [
            frozen.placement.rect.x,
            frozen.placement.rect.y,
            frozen.placement.rect.width,
            frozen.placement.rect.height,
        ],
        "source_hull_overlap_px2": overlap,
        "effective_scale": frozen.effective_scale,
        "selected_id": overlay.selected_target_id,
        "expected_selected_id": first.id,
        "image": str(output_path),
    }
    overlay.close()
    return result


def _contact_sheet(results: list[dict], output_path: Path) -> None:
    tile_width, tile_height = 640, 410
    columns = 2
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), (24, 27, 33))
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        row, column = divmod(index, columns)
        left, top = column * tile_width, row * tile_height
        with Image.open(result["image"]) as source:
            preview = ImageOps.contain(source.convert("RGB"), (tile_width - 24, tile_height - 62))
        x = left + (tile_width - preview.width) // 2
        y = top + 38 + (tile_height - 50 - preview.height) // 2
        sheet.paste(preview, (x, y))
        label = (
            f"{result['name']}  |  {result['placement']}  |  "
            f"selected {result['selected_id']}"
        )
        draw.text((left + 12, top + 12), label, fill=(245, 245, 245))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def render_scenarios(contact_sheet: Path, output_dir: Path) -> list[dict]:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [_render_one(preset, output_dir) for preset in render_presets()]
    _contact_sheet(results, contact_sheet)
    app.processEvents()
    return results
