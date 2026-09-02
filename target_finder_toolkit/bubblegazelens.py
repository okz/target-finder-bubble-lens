"""Dry-run ambiguity-triggered Bubble Gaze Lens Qt prototype."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from target_finder_toolkit.lens_core import (
    LensConfig,
    LensPlacement,
    LensStateMachine,
    LensStateName,
    LensStep,
    Point,
    PointerSample,
    Rect,
    TargetRect,
    bubble_solution,
    choose_lens_rect,
    choose_source_crop,
    filter_and_deduplicate,
    intersection_over_union,
    transform_target_to_lens,
)
AMBER = QtGui.QColor(255, 176, 32, 235)
GREEN = QtGui.QColor(35, 225, 95, 220)
WHITE = QtGui.QColor(250, 250, 250, 235)
DARK = QtGui.QColor(12, 15, 20, 225)


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    generation: int
    targets: tuple[TargetRect, ...]
    frame: np.ndarray | None


class SnapshotStore:
    """Thread-safe handoff from the detector callback to the Qt thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = TargetSnapshot(0, (), None)

    def update(self, detections, _added, _removed, frame) -> None:
        targets = tuple(
            TargetRect(
                id=int(item["id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item["width"]),
                height=float(item["height"]),
                score=float(item["score"]),
                class_id=int(item["class_id"]),
            )
            for item in detections
            if item.get("id") is not None and item.get("width", 0) > 0 and item.get("height", 0) > 0
        )
        frame_copy = None if frame is None else np.ascontiguousarray(frame).copy()
        with self._lock:
            self._snapshot = TargetSnapshot(
                self._snapshot.generation + 1,
                targets,
                frame_copy,
            )

    def read(self) -> TargetSnapshot:
        with self._lock:
            return self._snapshot


class PointerProvider(Protocol):
    def get_sample(self) -> PointerSample | None: ...


class MousePointerProvider:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()

    def get_sample(self) -> PointerSample:
        position = QtGui.QCursor.pos()
        return PointerSample(
            t_ms=(self._clock() - self._started_at) * 1000.0,
            x=float(position.x()),
            y=float(position.y()),
            valid=True,
        )


class ReplayPointerProvider:
    def __init__(self, samples: Sequence[PointerSample]) -> None:
        self._samples = iter(samples)

    def get_sample(self) -> PointerSample | None:
        return next(self._samples, None)


class JsonlLogger:
    def __init__(self, path: Path | None, config: LensConfig) -> None:
        self._stream = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("a", encoding="utf-8")
            self.write({"event": "session_started", "config": asdict(config)})

    def write(self, payload: dict) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(payload, default=sorted, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


@dataclass(frozen=True, slots=True)
class FrozenLens:
    generation: int
    frame: np.ndarray
    candidates: tuple[TargetRect, ...]
    placement: LensPlacement
    source_crop: Rect


def _contains(rect: Rect, point: Point) -> bool:
    return rect.x <= point.x <= rect.right and rect.y <= point.y <= rect.bottom


def _qrect(rect: Rect) -> QtCore.QRectF:
    return QtCore.QRectF(rect.x, rect.y, rect.width, rect.height)


def _frame_image(frame: np.ndarray) -> QtGui.QImage:
    contiguous = np.ascontiguousarray(frame)
    height, width, channels = contiguous.shape
    if channels != 3:
        raise ValueError("Expected a three-channel detector frame")
    image = QtGui.QImage(
        contiguous.data,
        width,
        height,
        contiguous.strides[0],
        QtGui.QImage.Format.Format_BGR888,
    )
    return image.copy()


class LensOverlay(QtWidgets.QWidget):
    """Single click-through overlay that owns the complete lens visual state."""

    def __init__(
        self,
        snapshot_reader: Callable[[], TargetSnapshot],
        pointer_provider: PointerProvider,
        config: LensConfig | None = None,
        logger: JsonlLogger | None = None,
        *,
        screen: QtGui.QScreen | None = None,
        screen_rect: Rect | None = None,
        start_timer: bool = True,
    ) -> None:
        super().__init__()
        self.snapshot_reader = snapshot_reader
        self.pointer_provider = pointer_provider
        self.config = config or LensConfig()
        self.logger = logger or JsonlLogger(None, self.config)
        self.machine = LensStateMachine(self.config)
        self._snapshot = TargetSnapshot(0, (), None)
        self._last_sample: PointerSample | None = None
        self._last_step: LensStep | None = None
        self._frozen: FrozenLens | None = None
        self._selected_target: TargetRect | None = None
        self._close_requested = False

        if screen_rect is None:
            screen = screen or QtGui.QGuiApplication.primaryScreen()
            if screen is None:
                raise RuntimeError("No screen is available")
            geometry = screen.geometry()
            self.screen_rect = Rect(0, 0, geometry.width(), geometry.height())
            self.screen_geometry = geometry
            self.setScreen(screen)
            self.setGeometry(geometry)
        else:
            self.screen_rect = screen_rect
            self.screen_geometry = QtCore.QRect(
                int(screen_rect.x),
                int(screen_rect.y),
                int(screen_rect.width),
                int(screen_rect.height),
            )
            self.resize(int(screen_rect.width), int(screen_rect.height))

        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowTransparentForInput
            | QtCore.Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self._timer: QtCore.QTimer | None = None
        if start_timer:
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.tick)
            self._timer.start(16)

    @property
    def frozen_lens(self) -> FrozenLens | None:
        return self._frozen

    @property
    def selected_target_id(self) -> int | None:
        return None if self._selected_target is None else self._selected_target.id

    @QtCore.pyqtSlot()
    def request_close(self) -> None:
        self._close_requested = True

    @QtCore.pyqtSlot()
    def confirm_selection(self) -> None:
        if self._frozen is None or self._selected_target is None or self._last_sample is None:
            return
        self.logger.write(
            {
                "t_ms": self._last_sample.t_ms,
                "event": "selection_dry_run",
                "state": LensStateName.LENS_OPEN.value,
                "target_id": self._selected_target.id,
                "snapshot_generation": self._frozen.generation,
            }
        )
        self._close_requested = True

    def _log_step(self, step: LensStep, sample: PointerSample) -> None:
        if not step.events:
            return
        decision = step.decision
        for event in step.events:
            payload = {
                "t_ms": sample.t_ms,
                "event": event,
                "state": step.state.value,
                "pointer": {"x": sample.x, "y": sample.y, "valid": sample.valid},
                "fixation": {
                    "r90_px": _json_number(decision.r90_px),
                    "valid_ratio": decision.valid_ratio,
                },
                "ambiguity": {
                    "score": (
                        decision.target_solution.ambiguity_score
                        if decision.target_solution is not None
                        else 0.0
                    ),
                    "sample_ratio": decision.ambiguous_ratio,
                    "candidate_ids": list(step.frozen_candidate_ids),
                },
                "snapshot_generation": self._snapshot.generation,
            }
            if event == "lens_opened" and self._frozen is not None:
                payload["lens"] = {
                    "rect": asdict(self._frozen.placement.rect),
                    "source_crop": asdict(self._frozen.source_crop),
                    "scale": self.config.lens_scale,
                    "placement": self._frozen.placement.side,
                }
            self.logger.write(payload)

    def _freeze_lens(self, step: LensStep) -> None:
        if self._snapshot.frame is None or step.decision.center is None:
            return
        candidate_ids = set(step.frozen_candidate_ids)
        candidates = tuple(
            target
            for target in filter_and_deduplicate(self._snapshot.targets, self.config)
            if target.id in candidate_ids
        )
        if len(candidates) < 2:
            self._close_requested = True
            return
        placement = choose_lens_rect(candidates, self.screen_rect, self.config)
        source_crop = choose_source_crop(
            step.decision.center,
            placement.rect,
            self.screen_rect,
            self.config.lens_scale,
        )
        self._frozen = FrozenLens(
            generation=self._snapshot.generation,
            frame=self._snapshot.frame.copy(),
            candidates=candidates,
            placement=placement,
            source_crop=source_crop,
        )

    def _update_lens_selection(self, sample: PointerSample) -> None:
        self._selected_target = None
        if self._frozen is None or not sample.valid:
            return
        point = sample.point
        if not _contains(self._frozen.placement.rect, point):
            return
        transformed = tuple(
            transform_target_to_lens(
                target,
                self._frozen.source_crop,
                self._frozen.placement.rect,
            )
            for target in self._frozen.candidates
        )
        self._selected_target = bubble_solution(point, transformed, self.config).primary

    def _frozen_candidates_are_valid(self) -> bool:
        if self._frozen is None or self._snapshot.generation == self._frozen.generation:
            return True
        current = {
            target.id: target
            for target in filter_and_deduplicate(self._snapshot.targets, self.config)
        }
        for frozen_target in self._frozen.candidates:
            candidate = current.get(frozen_target.id)
            if (
                candidate is None
                or candidate.class_id != frozen_target.class_id
                or intersection_over_union(candidate, frozen_target) < 0.50
            ):
                return False
        return True

    @QtCore.pyqtSlot()
    def tick(self) -> None:
        sample = self.pointer_provider.get_sample()
        if sample is None:
            return
        self._snapshot = self.snapshot_reader()
        self._last_sample = sample
        close_reason = None
        if self.machine.state is LensStateName.LENS_OPEN and not self._frozen_candidates_are_valid():
            close_reason = "target_invalidated"
        step = self.machine.step(
            sample.t_ms,
            sample,
            self._snapshot.targets,
            clean_frame_available=self._snapshot.frame is not None,
            close_requested=self._close_requested,
            close_reason=close_reason,
        )
        self._close_requested = False
        if "lens_opened" in step.events:
            self._freeze_lens(step)
        if step.state is LensStateName.LENS_OPEN:
            self._update_lens_selection(sample)
        elif step.state is not LensStateName.LENS_OPEN:
            self._frozen = None
            self._selected_target = None
        self._last_step = step
        self._log_step(step, sample)
        self.update()

    def _draw_cursor(self, painter: QtGui.QPainter, point: Point, color: QtGui.QColor) -> None:
        painter.setPen(QtGui.QPen(color, 2))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QtCore.QPointF(point.x, point.y), 6, 6)
        painter.drawLine(QtCore.QPointF(point.x - 11, point.y), QtCore.QPointF(point.x - 7, point.y))
        painter.drawLine(QtCore.QPointF(point.x + 7, point.y), QtCore.QPointF(point.x + 11, point.y))
        painter.drawLine(QtCore.QPointF(point.x, point.y - 11), QtCore.QPointF(point.x, point.y - 7))
        painter.drawLine(QtCore.QPointF(point.x, point.y + 7), QtCore.QPointF(point.x, point.y + 11))

    def _draw_bubble(self, painter: QtGui.QPainter) -> None:
        if self._last_sample is None or not self._last_sample.valid:
            return
        solution = bubble_solution(self._last_sample.point, self._snapshot.targets, self.config)
        if solution.primary is None:
            return
        color = AMBER if self._last_step and self._last_step.state is LensStateName.PENDING else GREEN
        painter.setPen(QtGui.QPen(color, 3))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        point = self._last_sample.point
        painter.drawEllipse(QtCore.QPointF(point.x, point.y), solution.radius, solution.radius)
        target = solution.primary
        painter.drawRoundedRect(
            QtCore.QRectF(target.x, target.y, target.width, target.height),
            min(target.width, target.height) / 4.0,
            min(target.width, target.height) / 4.0,
        )
        self._draw_cursor(painter, point, WHITE)

    def _draw_lens(self, painter: QtGui.QPainter) -> None:
        if self._frozen is None:
            return
        frozen = self._frozen
        lens_rect = frozen.placement.rect
        source_hull = frozen.placement.source_hull
        painter.setPen(QtGui.QPen(AMBER, 3))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(_qrect(source_hull), 8, 8)
        painter.drawLine(
            QtCore.QPointF(source_hull.center.x, source_hull.center.y),
            QtCore.QPointF(lens_rect.center.x, lens_rect.center.y),
        )

        painter.fillRect(_qrect(lens_rect), DARK)
        painter.drawImage(_qrect(lens_rect), _frame_image(frozen.frame), _qrect(frozen.source_crop))
        painter.save()
        painter.setClipRect(_qrect(lens_rect))
        for candidate in frozen.candidates:
            transformed = transform_target_to_lens(candidate, frozen.source_crop, lens_rect)
            selected = self._selected_target is not None and candidate.id == self._selected_target.id
            painter.setPen(QtGui.QPen(AMBER if selected else WHITE, 4 if selected else 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                QtCore.QRectF(
                    transformed.x,
                    transformed.y,
                    transformed.width,
                    transformed.height,
                ),
                5,
                5,
            )
        if self._last_sample is not None and _contains(lens_rect, self._last_sample.point):
            self._draw_cursor(painter, self._last_sample.point, AMBER)
        painter.restore()

        painter.setPen(QtGui.QPen(AMBER, 4))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(_qrect(lens_rect), 12, 12)
        label_rect = QtCore.QRectF(lens_rect.x, lens_rect.y - 28, lens_rect.width, 24)
        painter.fillRect(label_rect, DARK)
        painter.setPen(WHITE)
        painter.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "DRY RUN  •  look into lens  •  Enter confirms")

    def _paint(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._last_step is not None and self._last_step.state is LensStateName.LENS_OPEN:
            self._draw_lens(painter)
        else:
            self._draw_bubble(painter)

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        self._paint(painter)
        painter.end()

    def render_to_image(self) -> QtGui.QImage:
        image = QtGui.QImage(
            int(self.screen_rect.width),
            int(self.screen_rect.height),
            QtGui.QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        self._paint(painter)
        painter.end()
        return image


class LiveKeyboardController:
    def __init__(self, overlay: LensOverlay, stop_callback: Callable[[], None]) -> None:
        from pynput import keyboard

        self._keyboard = keyboard
        self._overlay = overlay
        self._stop_callback = stop_callback
        self._listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key) -> None:
        if key == self._keyboard.Key.esc:
            QtCore.QMetaObject.invokeMethod(
                self._overlay, "request_close", QtCore.Qt.ConnectionType.QueuedConnection
            )
        elif key == self._keyboard.Key.enter:
            QtCore.QMetaObject.invokeMethod(
                self._overlay, "confirm_selection", QtCore.Qt.ConnectionType.QueuedConnection
            )
        else:
            try:
                if key.char == "q":
                    self._stop_callback()
            except AttributeError:
                pass

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the dry-run Bubble Gaze Lens overlay")
    parser.add_argument(
        "--model",
        default="yolo26n-640",
        help="Bundled YOLO26 model name or path to custom .pt weights",
    )
    parser.add_argument("--change-thresh", type=int, default=100)
    parser.add_argument("--capture-interval", type=float, default=1 / 30)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--trigger-window-ms", type=float, default=200)
    parser.add_argument("--fixation-r90-px", type=float, default=35)
    parser.add_argument("--uncertainty-radius-px", type=float, default=48)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.65)
    parser.add_argument("--ambiguous-sample-ratio", type=float, default=0.75)
    parser.add_argument("--lens-size-px", type=float, default=360)
    parser.add_argument("--lens-scale", type=float, default=3.0)
    parser.add_argument("--lens-gap-px", type=float, default=24)
    parser.add_argument("--lens-timeout-ms", type=float, default=3000)
    parser.add_argument("--cooldown-ms", type=float, default=400)
    parser.add_argument("--include-text-targets", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("artifacts/lens-events.jsonl"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    from target_finder_toolkit.targetfinder import TargetFinder

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if len(app.screens()) != 1:
        raise SystemExit("Bubble Gaze Lens MVP supports exactly one monitor")

    config = LensConfig(
        trigger_window_ms=args.trigger_window_ms,
        fixation_r90_px=args.fixation_r90_px,
        uncertainty_radius_px=args.uncertainty_radius_px,
        ambiguity_threshold=args.ambiguity_threshold,
        ambiguous_sample_ratio=args.ambiguous_sample_ratio,
        lens_size_px=args.lens_size_px,
        lens_scale=args.lens_scale,
        lens_gap_px=args.lens_gap_px,
        lens_timeout_ms=args.lens_timeout_ms,
        cooldown_ms=args.cooldown_ms,
        confidence_threshold=args.confidence,
        include_text_targets=args.include_text_targets,
    )
    logger = JsonlLogger(args.log, config)
    store = SnapshotStore()
    detector = TargetFinder(
        args.model,
        args.change_thresh,
        args.capture_interval,
        args.confidence,
        args.iou,
    )
    detector.set_callback(store.update, with_frame=True)
    overlay = LensOverlay(store.read, MousePointerProvider(), config, logger)
    detector.overlay_window["bubble-gaze-lens"] = overlay

    keyboard_controller: LiveKeyboardController | None = None

    def stop() -> None:
        QtCore.QMetaObject.invokeMethod(app, "quit", QtCore.Qt.ConnectionType.QueuedConnection)

    keyboard_controller = LiveKeyboardController(overlay, stop)
    app.aboutToQuit.connect(detector.stop)
    app.aboutToQuit.connect(keyboard_controller.stop)
    app.aboutToQuit.connect(logger.close)
    signal.signal(signal.SIGINT, lambda *_: stop())

    print("Bubble Gaze Lens: single monitor, logical coordinates, Text excluded by default")
    print("Selection mode: DRY RUN (no operating-system click path exists)")
    print(f"Effective configuration: {json.dumps(asdict(config), default=sorted)}")
    print("Keys: Escape closes the lens, Enter logs a dry-run selection, q quits")
    overlay.show()
    keyboard_controller.start()
    detector.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
