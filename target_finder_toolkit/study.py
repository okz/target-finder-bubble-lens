"""Local controlled acquisition study; page geometry is scoring ground truth only."""

from __future__ import annotations

import csv
import io
import json
import math
import random
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .lens_core import TargetRect, intersection_over_union


MODES = ("bubble", "forced-lens", "auto-lens")


def make_trials(width: float, height: float, seed: int) -> list[dict]:
    """Same 24 tasks in each condition, with reproducible condition/task ordering."""
    if width < 1000 or height < 600:
        raise ValueError("The controlled study requires at least 1000×600 logical pixels")
    rng = random.Random(seed)
    modes = list(MODES)
    rng.shuffle(modes)
    tasks = []
    for density, count, size, gap in (("easy", 1, 52, 0), ("pair", 2, 32, 8), ("dense", 4, 28, 6)):
        for repetition in range(8):
            total_width = count * size + (count - 1) * gap
            center_x = width * (0.35 if repetition % 2 == 0 else 0.65)
            center_y = height * (0.42 if repetition % 4 < 2 else 0.68)
            targets = [
                {"id": i + 1, "x": center_x - total_width / 2 + i * (size + gap),
                 "y": center_y - size / 2, "width": size, "height": size}
                for i in range(count)
            ]
            tasks.append({"task_id": f"{density}-{repetition + 1}", "density": density,
                          "intended_target_id": repetition % count + 1, "targets": targets})
    trials = []
    for mode in modes:
        ordered = list(tasks)
        rng.shuffle(ordered)
        for task in ordered:
            trials.append(dict(task, mode=mode, trial_id=len(trials) + 1))
    return trials


class AcquisitionStudy:
    """One session, one backend clock, single acceptance per presented trial."""

    def __init__(self, width: int, height: int, *, participant: str, seed: int,
                 pointer: str, emit: Callable[[dict], None], clock=time.monotonic):
        self.width, self.height = width, height
        self.trials = make_trials(width, height, seed)
        self.participant, self.seed, self.pointer = participant, seed, pointer
        self.emit, self.clock = emit, clock
        self.lock = threading.RLock()
        self.phase = "setup"
        self.index = 0
        self.results: list[dict] = []
        self.started_ms: float | None = None
        self.deadline_ms = 0.0
        self.last_page_ms: float | None = None
        self.last_overlay_ms: float | None = None
        self.lens_opened = False
        self.first_entry_ms: float | None = None
        self.cancel_count = 0
        self.interruption: str | None = None

    def _now(self):
        return self.clock() * 1000.0

    def _event(self, event, **fields):
        self.emit(dict(event=event, study_time_ms=self._now(), **fields))

    def _advance(self):
        now = self._now()
        if self.phase in ("between", "presenting", "active", "feedback"):
            if self.last_page_ms is not None and now - self.last_page_ms > 2000:
                self.interrupt("page_connection_lost")
                return
        if self.phase == "feedback" and now >= self.deadline_ms:
            self.index += 1
            if self.index == len(self.trials):
                self.phase = "complete"
                self._event("study_completed", completed_trials=len(self.results))
            else:
                self.phase = "between"
                self.deadline_ms = now + 800
        elif self.phase == "between" and now >= self.deadline_ms:
            self.phase = "presenting"

    def control_state(self):
        """Called by the Qt thread; only an active presented trial enables interaction."""
        with self.lock:
            self.last_overlay_ms = self._now()
            self._advance()
            trial = self.trials[min(self.index, len(self.trials) - 1)]
            return trial["mode"], trial["trial_id"] if self.phase in ("active", "feedback") else None

    def state(self):
        with self.lock:
            self._advance()
            return {"phase": self.phase, "trial": self.trials[self.index] if self.index < len(self.trials) else None,
                    "completed": len(self.results), "total": len(self.trials),
                    "width": self.width, "height": self.height, "pointer": self.pointer,
                    "participant": self.participant, "interruption": self.interruption,
                    "last_result": self.results[-1] if self.results else None,
                    "overlay_connected": self.last_overlay_ms is not None and self._now() - self.last_overlay_ms <= 1000}

    def heartbeat(self):
        with self.lock:
            self._advance()  # Do not revive a connection that already exceeded the deadline.
            self.last_page_ms = self._now()
            if self.phase not in ("setup", "complete", "interrupted") and (
                self.last_overlay_ms is None or self._now() - self.last_overlay_ms > 2000
            ):
                self.interrupt("overlay_connection_lost")
            return self.state()

    def start(self):
        with self.lock:
            if self.phase != "setup":
                raise ValueError("A study session can be started only once")
            if self.last_overlay_ms is None or self._now() - self.last_overlay_ms > 1000:
                raise ValueError("The lens overlay is not connected")
            self.last_page_ms = self._now()
            self.phase = "between"
            self.deadline_ms = self._now() + 800
            self._event("study_started", participant=self.participant, seed=self.seed,
                        pointer=self.pointer, trials=self.trials, screen=[self.width, self.height])
            return self.state()

    def presented(self, trial_id: int, viewport_width: float, viewport_height: float):
        with self.lock:
            self._advance()
            trial = self.trials[min(self.index, len(self.trials) - 1)]
            if self.phase != "presenting" or trial_id != trial["trial_id"]:
                raise ValueError("Presentation acknowledgement is stale or out of order")
            if (isinstance(viewport_width, bool) or isinstance(viewport_height, bool)
                or not isinstance(viewport_width, (int, float)) or not isinstance(viewport_height, (int, float))
                or not math.isfinite(viewport_width) or not math.isfinite(viewport_height)
                or viewport_width <= 0 or viewport_height <= 0 or abs(
                viewport_width / viewport_height - self.width / self.height
            ) > 0.01):
                raise ValueError("Fullscreen viewport must match the study screen aspect ratio")
            self.started_ms = self._now()
            self.lens_opened, self.first_entry_ms, self.cancel_count = False, None, 0
            self.phase = "active"
            self._event("trial_started", trial=trial, viewport=[viewport_width, viewport_height],
                        timing_basis="backend_receipt_after_two_animation_frames")
            return self.state()

    def observe(self, event: dict):
        with self.lock:
            self._advance()
            if self.phase != "active":
                return
            trial = self.trials[self.index]
            # Events from a previous trial or another condition must never score.
            if event.get("study_trial_id") != trial["trial_id"] or event.get("interaction_mode") != trial["mode"]:
                return
            name = event.get("event")
            if name == "lens_opened":
                self.lens_opened = True
            elif name == "lens_first_entry" and self.first_entry_ms is None:
                self.first_entry_ms = event["transfer_time_ms"]
            elif name in ("lens_closed", "outside_region", "pointer_lost", "source_scrolled", "target_invalidated", "source_changed"):
                self.cancel_count += 1
            elif name == "selection_dry_run":
                box = event.get("source_target")
                matches = []
                if box:
                    detection = TargetRect(0, box["x"], box["y"], box["width"], box["height"])
                    for target in trial["targets"]:
                        if intersection_over_union(detection, TargetRect(**target)) >= 0.5:
                            matches.append(target["id"])
                selected = matches[0] if len(matches) == 1 else None
                result = {
                    "trial_id": trial["trial_id"], "task_id": trial["task_id"], "mode": trial["mode"],
                    "density": trial["density"], "intended_target_id": trial["intended_target_id"],
                    "selected_target_id": selected, "detector_target_id": event["target_id"],
                    "mapping_status": "matched" if selected is not None else "unmatched_or_ambiguous",
                    "correct": selected == trial["intended_target_id"],
                    "acquisition_time_ms": self._now() - self.started_ms,
                    "selection_space": event["selection_space"], "lens_opened": self.lens_opened,
                    "first_lens_entry_ms": self.first_entry_ms, "lens_cancellations": self.cancel_count,
                }
                self.results.append(result)
                self.phase = "feedback"
                # Leave time for the overlay's 200 ms selection feedback before hiding the board.
                self.deadline_ms = self._now() + 400
                self._event("trial_completed", **result)

    def interrupt(self, reason):
        with self.lock:
            if self.phase in ("complete", "interrupted"):
                return
            self.interruption = reason
            self.phase = "interrupted"
            self._event("study_interrupted", reason=reason, trial_index=self.index,
                        completed_trials=len(self.results))

    def report(self):
        with self.lock:
            summary = {}
            for mode in MODES:
                rows = [r for r in self.results if r["mode"] == mode]
                correct_rows = [r for r in rows if r["correct"]]
                summary[mode] = {
                    "planned_trials": 24, "completed_trials": len(rows),
                    "correct_trials": sum(r["correct"] for r in rows),
                    "accuracy_among_completed": sum(r["correct"] for r in rows) / len(rows) if rows else None,
                    "unmatched_selections": sum(r["selected_target_id"] is None for r in rows),
                    "median_acquisition_time_ms": statistics.median(r["acquisition_time_ms"] for r in rows) if rows else None,
                    "median_correct_acquisition_time_ms": statistics.median(r["acquisition_time_ms"] for r in correct_rows) if correct_rows else None,
                    "by_density": {
                        density: {
                            "planned_trials": 8,
                            "completed_trials": sum(r["density"] == density for r in rows),
                            "correct_trials": sum(r["density"] == density and r["correct"] for r in rows),
                        } for density in ("easy", "pair", "dense")
                    },
                }
            return {"participant": self.participant, "seed": self.seed, "pointer": self.pointer,
                    "phase": self.phase, "interruption": self.interruption,
                    "timing_basis": "backend_monotonic_clock; start_after_presentation_ack; end_at_accepted_selection",
                    "mapping": "unique_ground_truth_target_with_IoU_at_least_0.5",
                    "screen": [self.width, self.height], "schedule": self.trials,
                    "summary": summary, "trials": list(self.results)}


class StudyServer:
    """Same-origin loopback page and study controls; no endpoint can select a target."""

    def __init__(self, study: AcquisitionStudy, port=0):
        if not 0 <= port <= 65535:
            raise ValueError("Study port must be between 0 and 65535")
        self.study = study
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def respond(self, status, data, content_type="application/json"):
                data = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.headers.get("Host") != owner.authority:
                    return self.respond(403, {"error": "Invalid host"})
                if self.path == "/":
                    return self.respond(200, Path(__file__).with_name("study_page.html").read_bytes(), "text/html; charset=utf-8")
                if self.path == "/api/state":
                    return self.respond(200, study.heartbeat())
                if self.path == "/results.json":
                    return self.respond(200, study.report())
                if self.path == "/results.csv":
                    report = study.report()
                    rows = [dict(participant=report["participant"], pointer=report["pointer"], seed=report["seed"], **r)
                            for r in report["trials"]]
                    output = io.StringIO(newline="")
                    if rows:
                        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    return self.respond(200, output.getvalue().encode(), "text/csv; charset=utf-8")
                return self.respond(404, {"error": "Not found"})

            def do_POST(self):
                if self.headers.get("Host") != owner.authority or self.headers.get("Origin") != owner.url.rstrip("/"):
                    return self.respond(403, {"error": "Same-origin requests only"})
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Invalid request length")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("Request body must be an object")
                    if self.path == "/api/start":
                        result = study.start()
                    elif self.path == "/api/presented":
                        if body.get("fullscreen") is not True:
                            raise ValueError("Fullscreen is required")
                        result = study.presented(body["trial_id"], body["width"], body["height"])
                    elif self.path == "/api/interrupt":
                        reason = body.get("reason", "page_interrupted")
                        allowed = {"page_interrupted", "fullscreen_exited", "page_hidden", "viewport_changed", "presentation_failed"}
                        study.interrupt(reason if reason in allowed else "page_interrupted")
                        result = study.state()
                    else:
                        return self.respond(404, {"error": "Not found"})
                except (ValueError, KeyError, TypeError) as error:
                    return self.respond(400, {"error": str(error)})
                self.respond(200, result)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.authority = f"127.0.0.1:{self.server.server_port}"
        self.url = f"http://{self.authority}/"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.study.interrupt("application_closed")
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class StudyLogger:
    """Forward overlay events and synchronously score accepted selections."""

    def __init__(self, base, study):
        self.base, self.study = base, study

    def write(self, event):
        self.base.write(event)
        self.study.observe(event)

    def close(self):
        self.base.close()
