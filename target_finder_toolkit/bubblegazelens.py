"""Dry-run ambiguity-triggered Bubble Gaze Lens Qt prototype."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
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
    filter_and_deduplicate,
    intersection_over_union,
    point_in_interaction_region,
    prepare_lens_layout,
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
    label: str

    def get_sample(self) -> PointerSample | None: ...

    def diagnostics(self) -> dict: ...

    def close(self) -> None: ...


class MousePointerProvider:
    label = "MOUSE PROXY"

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

    def diagnostics(self) -> dict:
        return {"provider": "mouse", "coordinate_space": "logical_px"}

    def close(self) -> None:
        pass


class ReplayPointerProvider:
    label = "REPLAY"

    def __init__(self, samples: Sequence[PointerSample]) -> None:
        self._samples = iter(samples)

    def get_sample(self) -> PointerSample | None:
        return next(self._samples, None)

    def diagnostics(self) -> dict:
        return {"provider": "replay", "coordinate_space": "logical_px"}

    def close(self) -> None:
        pass


def load_replay_samples(path: Path) -> tuple[PointerSample, ...]:
    samples: list[PointerSample] = []
    previous_t_ms: float | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            t_ms = float(payload["t_ms"])
            x = float(payload["x"])
            y = float(payload["y"])
            valid = payload.get("valid", True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid replay sample at {path}:{line_number}") from error
        if not all(math.isfinite(value) for value in (t_ms, x, y)):
            raise ValueError(f"Non-finite replay sample at {path}:{line_number}")
        if not isinstance(valid, bool):
            raise ValueError(f"Replay validity must be boolean at {path}:{line_number}")
        if payload.get("screen", "primary") != "primary":
            raise ValueError(f"Replay samples must use the primary screen at {path}:{line_number}")
        if valid and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            raise ValueError(f"Replay coordinates must be logical pixels at {path}:{line_number}")
        if previous_t_ms is not None and t_ms < previous_t_ms:
            raise ValueError(f"Replay timestamps moved backwards at {path}:{line_number}")
        samples.append(PointerSample(t_ms=t_ms, x=x, y=y, valid=valid))
        previous_t_ms = t_ms
    if not samples:
        raise ValueError(f"Replay file has no samples: {path}")
    return tuple(samples)


class UdpGazeProvider:
    """Receive a generic logical-pixel gaze stream on localhost only.

    Datagrams are UTF-8 JSON objects such as::

        {"t_ms": 1234.5, "x": 812.2, "y": 498.1,
         "valid": true, "screen": "primary"}

    Local receipt time drives the state machine, avoiding assumptions about the
    tracker process's clock. The supplied timestamp is retained for diagnostics.
    """

    label = "UDP GAZE"

    def __init__(
        self,
        port: int = 4242,
        *,
        hold_valid_ms: float = 50.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("UDP port must be between 0 and 65535")
        if hold_valid_ms < 0:
            raise ValueError("hold_valid_ms cannot be negative")
        self._clock = clock
        self._started_at = clock()
        self._hold_valid_ms = hold_valid_ms
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("127.0.0.1", port))
        self._socket.setblocking(False)
        self.port = int(self._socket.getsockname()[1])
        self._last_point = Point(0.0, 0.0)
        self._last_packet_valid = False
        self._last_receipt_s: float | None = None
        self._last_source_t_ms: float | None = None
        self._received_packets = 0
        self._dropped_packets = 0
        self._last_drop_reason: str | None = None
        self._closed = False

    @staticmethod
    def _number(value) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _drop(self, reason: str) -> None:
        self._dropped_packets += 1
        self._last_drop_reason = reason

    def _accept_payload(self, payload: object, received_at: float) -> None:
        if not isinstance(payload, dict):
            self._drop("payload_not_object")
            return
        if payload.get("screen") != "primary":
            self._drop("screen_must_be_primary")
            return
        valid = payload.get("valid")
        if not isinstance(valid, bool):
            self._drop("valid_must_be_boolean")
            return
        x = self._number(payload.get("x"))
        y = self._number(payload.get("y"))
        if valid and (x is None or y is None):
            self._drop("valid_sample_requires_finite_xy")
            return
        if x is not None and y is not None and valid and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            self._drop("normalized_coordinates_not_supported")
            return
        source_t_ms = self._number(payload.get("t_ms"))
        if source_t_ms is None:
            self._drop("t_ms_must_be_finite")
            return
        if self._last_source_t_ms is not None and source_t_ms < self._last_source_t_ms:
            self._drop("source_timestamp_moved_backwards")
            return
        if x is not None and y is not None:
            self._last_point = Point(x, y)
        self._last_packet_valid = valid
        self._last_receipt_s = received_at
        self._last_source_t_ms = source_t_ms
        self._received_packets += 1
        self._last_drop_reason = None

    def _drain(self, now_s: float) -> None:
        while not self._closed:
            try:
                data, _address = self._socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._drop("invalid_json")
                continue
            self._accept_payload(payload, now_s)

    def get_sample(self) -> PointerSample:
        now_s = self._clock()
        self._drain(now_s)
        age_ms = (
            math.inf
            if self._last_receipt_s is None
            else (now_s - self._last_receipt_s) * 1000.0
        )
        return PointerSample(
            t_ms=(now_s - self._started_at) * 1000.0,
            x=self._last_point.x,
            y=self._last_point.y,
            valid=self._last_packet_valid and age_ms <= self._hold_valid_ms,
        )

    def diagnostics(self) -> dict:
        now_s = self._clock()
        age_ms = (
            None
            if self._last_receipt_s is None
            else max(0.0, (now_s - self._last_receipt_s) * 1000.0)
        )
        return {
            "provider": "udp",
            "bind": f"127.0.0.1:{self.port}",
            "coordinate_space": "logical_px",
            "source_t_ms": self._last_source_t_ms,
            "packet_age_ms": age_ms,
            "received_packets": self._received_packets,
            "dropped_packets": self._dropped_packets,
            "last_drop_reason": self._last_drop_reason,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._socket.close()


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
    effective_scale: float
    opened_at_ms: float


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


def source_change_fraction(
    frozen_frame: np.ndarray,
    current_frame: np.ndarray,
    region: Rect,
    screen: Rect,
    pixel_threshold: float,
) -> float:
    """Fraction of logical source pixels whose mean BGR difference crosses a threshold."""

    if frozen_frame.shape != current_frame.shape:
        return 1.0
    height, width = frozen_frame.shape[:2]
    left = max(0, int(math.floor(region.x - screen.x)))
    top = max(0, int(math.floor(region.y - screen.y)))
    right = min(width, int(math.ceil(region.right - screen.x)))
    bottom = min(height, int(math.ceil(region.bottom - screen.y)))
    if left >= right or top >= bottom:
        return 1.0
    frozen = frozen_frame[top:bottom, left:right].astype(np.int16)
    current = current_frame[top:bottom, left:right].astype(np.int16)
    mean_difference = np.abs(current - frozen).mean(axis=2)
    return float(np.count_nonzero(mean_difference >= pixel_threshold) / mean_difference.size)


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
        interaction_mode: str = "auto-lens",
    ) -> None:
        super().__init__()
        self.snapshot_reader = snapshot_reader
        self.pointer_provider = pointer_provider
        self.config = config or LensConfig()
        if interaction_mode not in {"bubble", "forced-lens", "auto-lens"}:
            raise ValueError(f"Unsupported interaction mode: {interaction_mode}")
        self.interaction_mode = interaction_mode
        self.logger = logger or JsonlLogger(None, self.config)
        machine_config = (
            replace(self.config, ambiguity_threshold=0.0)
            if interaction_mode == "forced-lens"
            else self.config
        )
        self.machine = LensStateMachine(machine_config)
        self._snapshot = TargetSnapshot(0, (), None)
        self._last_sample: PointerSample | None = None
        self._last_step: LensStep | None = None
        self._frozen: FrozenLens | None = None
        self._pending_placement: LensPlacement | None = None
        self._selected_target: TargetRect | None = None
        self._close_requested = False
        self._close_reason: str | None = None
        self._selection_requested = False
        self._confirmation_target: TargetRect | None = None
        self._lens_entered = False
        self._scroll_requested = False
        self._last_source_change_fraction: float | None = None
        self._last_suppression_details: dict | None = None
        self._last_tracking_valid: bool | None = None

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
        self.logger.write({"event": "overlay_started", "interaction_mode": interaction_mode})

    @property
    def frozen_lens(self) -> FrozenLens | None:
        return self._frozen

    @property
    def selected_target_id(self) -> int | None:
        return None if self._selected_target is None else self._selected_target.id

    @QtCore.pyqtSlot()
    def request_close(self) -> None:
        self._close_requested = True
        self._close_reason = "lens_closed"

    @QtCore.pyqtSlot()
    def request_scroll_close(self) -> None:
        self._scroll_requested = True

    @QtCore.pyqtSlot()
    def confirm_selection(self) -> None:
        if (
            self._selection_requested
            or self.machine.state in (LensStateName.FEEDBACK, LensStateName.COOLDOWN)
            or self._last_sample is None
            or not self._last_sample.valid
        ):
            return
        target = (
            self._selected_target if self._frozen is not None
            else bubble_solution(self._last_sample.point, self._snapshot.targets, self.config).primary
        )
        if target is not None:
            # A request is committed only after the next pointer update/snapshot is
            # checked. Never turn a changed highlight into a different selection.
            self._confirmation_target = target
            self._selection_requested = True

    def _log_step(self, step: LensStep, sample: PointerSample) -> None:
        if not step.events:
            return
        decision = step.decision
        for event in step.events:
            selection_diagnostic = None
            if decision.center is not None and decision.target_solution is not None:
                primary = decision.target_solution.primary
                if primary is not None:
                    dx = primary.center.x - decision.center.x
                    dy = primary.center.y - decision.center.y
                    selection_diagnostic = {
                        "target_id": primary.id,
                        "dx_px": dx,
                        "dy_px": dy,
                        "distance_px": math.hypot(dx, dy),
                        "interpretation": "fixation-center offset from current Bubble winner",
                    }
            payload = {
                "t_ms": sample.t_ms,
                "event": event,
                "state": step.state.value,
                "pointer": {"x": sample.x, "y": sample.y, "valid": sample.valid},
                "fixation": {
                    "r90_px": _json_number(decision.r90_px),
                    "drift_px": _json_number(decision.fixation_drift_px),
                    "latest_jump_px": _json_number(decision.latest_jump_px),
                    "valid_ratio": decision.valid_ratio,
                },
                "ambiguity": {
                    "score": (
                        decision.target_solution.ambiguity_score
                        if decision.target_solution is not None
                        else 0.0
                    ),
                    "selection_noise_score": decision.selection_noise_score,
                    "candidate_ids": list(step.frozen_candidate_ids),
                },
                "snapshot_generation": self._snapshot.generation,
                "pointer_diagnostics": self.pointer_provider.diagnostics(),
                "selection_diagnostic": selection_diagnostic,
            }
            if event == "lens_opened" and self._frozen is not None:
                payload["lens"] = {
                    "rect": asdict(self._frozen.placement.rect),
                    "source_crop": asdict(self._frozen.source_crop),
                    "scale": self._frozen.effective_scale,
                    "placement": self._frozen.placement.side,
                }
            if event == "source_changed":
                payload["source_change_fraction"] = self._last_source_change_fraction
            if event.startswith("lens_suppressed_"):
                payload["suppression"] = self._last_suppression_details
            self.logger.write(payload)

    def _log_tracking_transition(self, sample: PointerSample) -> None:
        if self._last_tracking_valid is sample.valid:
            return
        self._last_tracking_valid = sample.valid
        self.logger.write(
            {
                "t_ms": sample.t_ms,
                "event": "tracking_valid" if sample.valid else "tracking_lost",
                "pointer": {"x": sample.x, "y": sample.y, "valid": sample.valid},
                "pointer_diagnostics": self.pointer_provider.diagnostics(),
            }
        )

    def _freeze_lens(self, step: LensStep, sample: PointerSample) -> str | None:
        self._last_suppression_details = None
        if self._snapshot.frame is None or step.decision.center is None:
            return "lens_suppressed_no_clean_frame"
        candidate_ids = set(step.frozen_candidate_ids)
        candidates = tuple(
            target
            for target in filter_and_deduplicate(self._snapshot.targets, self.config)
            if target.id in candidate_ids
        )
        layout = prepare_lens_layout(candidates, self.screen_rect, self.config)
        if layout.reason is not None:
            self._last_suppression_details = {
                "reason": layout.reason,
                "required_hull": None if layout.required_hull is None else asdict(layout.required_hull),
                "effective_scale": layout.effective_scale,
                "minimum_scale": self.config.minimum_lens_scale,
            }
            return layout.reason
        self._frozen = FrozenLens(
            generation=self._snapshot.generation,
            frame=self._snapshot.frame.copy(),
            candidates=candidates,
            placement=layout.placement,
            source_crop=layout.crop,
            effective_scale=layout.effective_scale,
            opened_at_ms=sample.t_ms,
        )
        self._lens_entered = False
        return None

    def _update_pending_preview(self, step: LensStep) -> None:
        candidate_ids = set(step.frozen_candidate_ids)
        candidates = tuple(
            target
            for target in filter_and_deduplicate(self._snapshot.targets, self.config)
            if target.id in candidate_ids
        )
        if len(candidates) < 2:
            self._pending_placement = None
            return
        self._pending_placement = choose_lens_rect(
            candidates, self.screen_rect, self.config
        ).placement

    def _update_lens_selection(self, sample: PointerSample) -> None:
        self._selected_target = None
        if self._frozen is None or not sample.valid:
            return
        point = sample.point
        if not _contains(self._frozen.placement.rect, point):
            return
        if not self._lens_entered:
            self._lens_entered = True
            self.logger.write({
                "event": "lens_first_entry",
                "t_ms": sample.t_ms,
                "interaction_mode": self.interaction_mode,
                "transfer_time_ms": sample.t_ms - self._frozen.opened_at_ms,
                "snapshot_generation": self._frozen.generation,
            })
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

    def _source_change_fraction(self) -> float:
        if self._frozen is None or self._snapshot.frame is None:
            return 1.0
        return source_change_fraction(
            self._frozen.frame,
            self._snapshot.frame,
            self._frozen.source_crop,
            self.screen_rect,
            self.config.source_change_pixel_threshold,
        )

    @QtCore.pyqtSlot()
    def tick(self) -> None:
        sample = self.pointer_provider.get_sample()
        if sample is None:
            return
        self._snapshot = self.snapshot_reader()
        self._last_sample = sample
        self._log_tracking_transition(sample)
        close_reason = None
        active_states = {
            LensStateName.LENS_OPEN,
            LensStateName.EXIT_GRACE,
            LensStateName.FEEDBACK,
        }
        if self.machine.state in active_states:
            if self._scroll_requested:
                close_reason = "source_scrolled"
            elif not self._frozen_candidates_are_valid():
                close_reason = "target_invalidated"
            elif self._frozen is not None and self._snapshot.generation != self._frozen.generation:
                self._last_source_change_fraction = self._source_change_fraction()
                if (
                    self._last_source_change_fraction
                    >= self.config.source_change_fraction_threshold
                ):
                    close_reason = "source_changed"
        pointer_in_region = True
        if self._frozen is not None and sample.valid:
            pointer_in_region = point_in_interaction_region(
                sample.point,
                self._frozen.placement.source_hull,
                self._frozen.placement.rect,
                self.config,
            )
        trigger_targets = () if self.interaction_mode == "bubble" else self._snapshot.targets
        state_before = self.machine.state
        confirmation = None
        if self._selection_requested and sample.valid:
            if self._frozen is not None:
                self._update_lens_selection(sample)
                candidate = self._selected_target
            else:
                candidate = bubble_solution(sample.point, self._snapshot.targets, self.config).primary
            requested = self._confirmation_target
            if (
                candidate is not None and requested is not None
                and candidate.id == requested.id
                and candidate.class_id == requested.class_id
                and intersection_over_union(candidate, requested) >= 0.50
                and not self._scroll_requested
            ):
                confirmation = candidate
        step = self.machine.step(
            sample.t_ms,
            sample,
            trigger_targets,
            clean_frame_available=self._snapshot.frame is not None,
            close_requested=self._close_requested,
            close_reason=close_reason or self._close_reason,
            pointer_in_interaction_region=pointer_in_region,
            selection_requested=confirmation is not None,
        )
        if "selection_feedback_started" in step.events and confirmation is not None:
            self._selected_target = confirmation
            self.logger.write({
                "event": "selection_dry_run",
                "t_ms": sample.t_ms,
                "state": state_before.value,
                "accepted_state": step.state.value,
                "interaction_mode": self.interaction_mode,
                "selection_space": "lens" if self._frozen is not None else "source",
                "target_id": confirmation.id,
                "snapshot_generation": self._snapshot.generation,
                "frozen_generation": None if self._frozen is None else self._frozen.generation,
                "pointer": {"x": sample.x, "y": sample.y, "valid": sample.valid},
            })
        elif self._selection_requested:
            self.logger.write({
                "event": "selection_rejected",
                "t_ms": sample.t_ms,
                "interaction_mode": self.interaction_mode,
                "reason": close_reason or self._close_reason or (
                    "source_scrolled" if self._scroll_requested else
                    "invalid_tracking" if not sample.valid else "selection_no_longer_available"
                ),
            })
        self._close_requested = False
        self._close_reason = None
        self._selection_requested = False
        self._confirmation_target = None
        self._scroll_requested = False
        if "lens_opened" in step.events:
            suppression_reason = self._freeze_lens(step, sample)
            if suppression_reason is not None:
                step = self.machine.step(
                    sample.t_ms,
                    sample,
                    (),
                    close_reason=suppression_reason,
                )
        if step.state is LensStateName.PENDING:
            self._update_pending_preview(step)
        else:
            self._pending_placement = None
        if step.state in (LensStateName.LENS_OPEN, LensStateName.EXIT_GRACE):
            self._update_lens_selection(sample)
        elif step.state not in active_states:
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
        if self.machine.state is LensStateName.FEEDBACK and self._selected_target is not None:
            target = self._selected_target
            painter.setPen(QtGui.QPen(AMBER, 4))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                QtCore.QRectF(target.x, target.y, target.width, target.height), 5, 5
            )
            painter.drawText(
                QtCore.QPointF(target.x, max(20, target.y - 8)), "DRY RUN  •  selected"
            )
            return
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

        if self._last_step and self._last_step.state is LensStateName.PENDING:
            decision = self._last_step.decision
            plausible = (
                () if decision.target_solution is None else decision.target_solution.plausible
            )
            if plausible:
                source = choose_lens_rect(plausible, self.screen_rect, self.config)
                if source.placement is not None:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 176, 32, 90), 2))
                    painter.drawRoundedRect(_qrect(source.placement.source_hull), 8, 8)
            if self._pending_placement is not None:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 176, 32, 80), 2))
                painter.drawRoundedRect(_qrect(self._pending_placement.rect), 12, 12)

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

        painter.save()
        if self._last_sample is not None and self.config.fade_in_ms > 0:
            fade = min(
                1.0,
                max(0.0, (self._last_sample.t_ms - frozen.opened_at_ms) / self.config.fade_in_ms),
            )
            painter.setOpacity(fade)
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
        label_rect = QtCore.QRectF(lens_rect.x + 8, lens_rect.y + 8, lens_rect.width - 16, 24)
        painter.fillRect(label_rect, DARK)
        painter.setPen(WHITE)
        label = (
            "DRY RUN  •  selected"
            if self._last_step and self._last_step.state is LensStateName.FEEDBACK
            else "DRY RUN  •  look into lens  •  Enter confirms"
        )
        painter.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()

    def _draw_tracking_status(self, painter: QtGui.QPainter) -> None:
        valid = self._last_sample is not None and self._last_sample.valid
        color = GREEN if valid else AMBER
        state = "VALID" if valid else "LOST"
        label = getattr(self.pointer_provider, "label", "POINTER")
        status_rect = QtCore.QRectF(12, 12, 180, 26)
        painter.fillRect(status_rect, DARK)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QtCore.QPointF(27, 25), 5, 5)
        painter.setPen(WHITE)
        painter.drawText(
            QtCore.QRectF(40, 12, 146, 26),
            QtCore.Qt.AlignmentFlag.AlignVCenter,
            f"{label}  {state}",
        )

    def _paint(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._frozen is not None and self._last_step is not None and self._last_step.state in {
            LensStateName.LENS_OPEN,
            LensStateName.EXIT_GRACE,
            LensStateName.FEEDBACK,
        }:
            self._draw_lens(painter)
        else:
            self._draw_bubble(painter)
        self._draw_tracking_status(painter)

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
        from pynput import keyboard, mouse

        self._keyboard = keyboard
        self._overlay = overlay
        self._stop_callback = stop_callback
        self._enter_down = False
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._scroll_listener = mouse.Listener(on_scroll=self._on_scroll)

    def _on_press(self, key) -> None:
        if key == self._keyboard.Key.esc:
            QtCore.QMetaObject.invokeMethod(
                self._overlay, "request_close", QtCore.Qt.ConnectionType.QueuedConnection
            )
        elif key == self._keyboard.Key.enter:
            if self._enter_down:
                return
            self._enter_down = True
            QtCore.QMetaObject.invokeMethod(
                self._overlay, "confirm_selection", QtCore.Qt.ConnectionType.QueuedConnection
            )
        else:
            try:
                if key.char == "q":
                    self._stop_callback()
            except AttributeError:
                pass

    def _on_release(self, key) -> None:
        if key == self._keyboard.Key.enter:
            self._enter_down = False

    def _on_scroll(self, _x, _y, _dx, _dy) -> None:
        QtCore.QMetaObject.invokeMethod(
            self._overlay,
            "request_scroll_close",
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def start(self) -> None:
        self._listener.start()
        self._scroll_listener.start()

    def stop(self) -> None:
        self._listener.stop()
        self._scroll_listener.stop()


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
    parser.add_argument("--fixation-drift-px", type=float, default=50)
    parser.add_argument("--fixation-jump-floor-px", type=float, default=12)
    parser.add_argument("--uncertainty-radius-px", type=float, default=48)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.51)
    parser.add_argument("--lens-size-px", type=float, default=360)
    parser.add_argument("--lens-scale", type=float, default=3.0)
    parser.add_argument("--lens-gap-px", type=float, default=24)
    parser.add_argument("--transfer-protection-ms", type=float, default=600)
    parser.add_argument("--outside-grace-ms", type=float, default=1200)
    parser.add_argument("--feedback-ms", type=float, default=200)
    parser.add_argument("--crop-margin-px", type=float, default=20)
    parser.add_argument("--minimum-lens-scale", type=float, default=2.0)
    parser.add_argument("--cooldown-ms", type=float, default=400)
    parser.add_argument("--include-text-targets", action="store_true")
    parser.add_argument(
        "--pointer",
        choices=("mouse", "replay", "udp"),
        default="mouse",
        help="Logical-pixel pointer source (default: mouse)",
    )
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument(
        "--udp-port",
        type=int,
        default=4242,
        help="Localhost UDP gaze port when --pointer=udp (default: 4242)",
    )
    parser.add_argument(
        "--mode",
        choices=("bubble", "forced-lens", "auto-lens"),
        default="auto-lens",
        help="Gate B comparison condition (default: auto-lens)",
    )
    parser.add_argument("--log", type=Path, default=Path("artifacts/lens-events.jsonl"))
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.pointer == "replay" and args.replay_file is None:
        parser.error("--replay-file is required when --pointer=replay")
    from target_finder_toolkit.targetfinder import TargetFinder

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if len(app.screens()) != 1:
        raise SystemExit("Bubble Gaze Lens MVP supports exactly one monitor")

    config = LensConfig(
        trigger_window_ms=args.trigger_window_ms,
        fixation_drift_px=args.fixation_drift_px,
        fixation_jump_floor_px=args.fixation_jump_floor_px,
        uncertainty_radius_px=args.uncertainty_radius_px,
        ambiguity_threshold=args.ambiguity_threshold,
        lens_size_px=args.lens_size_px,
        lens_scale=args.lens_scale,
        lens_gap_px=args.lens_gap_px,
        initial_transfer_protection_ms=args.transfer_protection_ms,
        outside_grace_ms=args.outside_grace_ms,
        selection_feedback_ms=args.feedback_ms,
        crop_margin_px=args.crop_margin_px,
        minimum_lens_scale=args.minimum_lens_scale,
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
    pointer_provider: PointerProvider
    if args.pointer == "udp":
        pointer_provider = UdpGazeProvider(args.udp_port)
    elif args.pointer == "replay":
        try:
            pointer_provider = ReplayPointerProvider(load_replay_samples(args.replay_file))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        pointer_provider = MousePointerProvider()
    overlay = LensOverlay(
        store.read,
        pointer_provider,
        config,
        logger,
        interaction_mode=args.mode,
    )
    detector.overlay_window["bubble-gaze-lens"] = overlay

    keyboard_controller: LiveKeyboardController | None = None

    def stop() -> None:
        QtCore.QMetaObject.invokeMethod(app, "quit", QtCore.Qt.ConnectionType.QueuedConnection)

    keyboard_controller = LiveKeyboardController(overlay, stop)
    app.aboutToQuit.connect(detector.stop)
    app.aboutToQuit.connect(keyboard_controller.stop)
    app.aboutToQuit.connect(pointer_provider.close)
    app.aboutToQuit.connect(logger.close)
    signal.signal(signal.SIGINT, lambda *_: stop())

    text_status = "included" if args.include_text_targets else "excluded"
    print(f"Bubble Gaze Lens: single monitor, logical coordinates, Text {text_status}")
    print("Selection mode: DRY RUN (no operating-system click path exists)")
    print(f"Interaction condition: {args.mode}")
    if args.pointer == "udp":
        print(f"Gaze input: UDP JSON on 127.0.0.1:{pointer_provider.port}")
    elif args.pointer == "replay":
        print(f"Gaze input: JSONL replay from {args.replay_file}")
    else:
        print("Gaze input: mouse proxy")
    print(f"Effective configuration: {json.dumps(asdict(config), default=sorted)}")
    print("Keys: Escape closes the lens, Enter logs a dry-run selection, q quits")
    overlay.show()
    keyboard_controller.start()
    detector.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
