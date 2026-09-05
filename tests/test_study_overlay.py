import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from target_finder_toolkit.bubblegazelens import LensOverlay, ReplayPointerProvider, TargetSnapshot
from target_finder_toolkit.lens_core import PointerSample, Rect, TargetRect
from target_finder_toolkit.study import AcquisitionStudy, StudyLogger


class Log:
    def __init__(self): self.events = []
    def write(self, event): self.events.append(event)


def test_overlay_selection_scores_ground_truth_and_trial_change_resets_requests(qtbot):
    now = [10.]
    log = Log()
    study = AcquisitionStudy(1280, 720, participant="integration", seed=42, pointer="mouse", emit=log.write, clock=lambda: now[0])
    snapshot = [TargetSnapshot(1, (), np.zeros((720, 1280, 3), dtype=np.uint8))]
    overlay = LensOverlay(lambda: snapshot[0], ReplayPointerProvider([PointerSample(0, 0, 0)]),
                          logger=StudyLogger(log, study), study=study,
                          screen_rect=Rect(0, 0, 1280, 720), start_timer=False)
    qtbot.addWidget(overlay)
    overlay.tick()
    study.start()
    now[0] += .801
    trial = study.state()["trial"]
    study.presented(trial["trial_id"], 1280, 720)
    target = next(t for t in trial["targets"] if t["id"] == trial["intended_target_id"])
    # Detector identity is deliberately different from ground truth identity.
    detected = TargetRect(987, target["x"], target["y"], target["width"], target["height"])
    snapshot[0] = TargetSnapshot(2, (detected,), snapshot[0].frame)
    overlay.pointer_provider = ReplayPointerProvider([PointerSample(20, detected.center.x, detected.center.y), PointerSample(40, detected.center.x, detected.center.y)])
    overlay.tick()
    overlay.confirm_selection()
    now[0] += .2
    overlay.tick()
    assert study.results[0]["correct"]
    assert study.results[0]["selected_target_id"] == target["id"]
    assert study.results[0]["detector_target_id"] == 987
    now[0] += .401
    overlay.pointer_provider = ReplayPointerProvider([PointerSample(60, detected.center.x, detected.center.y)])
    overlay.tick()
    overlay.confirm_selection()
    assert overlay._study_trial_id is None
    assert not overlay._selection_requested
