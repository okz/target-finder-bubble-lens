import json
import urllib.error
import urllib.request
from collections import Counter

import pytest

from target_finder_toolkit.study import AcquisitionStudy, MODES, StudyServer, make_trials


class Clock:
    value = 10.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def setup_study():
    clock, events = Clock(), []
    study = AcquisitionStudy(1280, 720, participant="test", seed=42, pointer="mouse", emit=events.append, clock=clock)
    study.control_state()
    study.start()
    return study, clock, events


def present(study, clock):
    clock.advance(.801)
    study.control_state()
    state = study.heartbeat()
    assert state["phase"] == "presenting"
    study.presented(state["trial"]["trial_id"], 1280, 720)
    return state["trial"]


def selection(trial, target_id=None):
    target_id = target_id or trial["intended_target_id"]
    target = next(t for t in trial["targets"] if t["id"] == target_id)
    return {"event": "selection_dry_run", "study_trial_id": trial["trial_id"],
            "interaction_mode": trial["mode"], "target_id": 987, "selection_space": "source",
            "source_target": {k: target[k] for k in ("x", "y", "width", "height")}}


def test_trials_are_balanced_reproducible_and_use_identical_tasks_across_conditions():
    trials = make_trials(1280, 720, 42)
    assert trials == make_trials(1280, 720, 42)
    assert trials != make_trials(1280, 720, 43)
    assert len(trials) == 72
    assert Counter((t["mode"], t["density"]) for t in trials) == Counter({(m, d): 8 for m in MODES for d in ("easy", "pair", "dense")})
    for task_id in {t["task_id"] for t in trials}:
        matching = [t for t in trials if t["task_id"] == task_id]
        assert all(t["targets"] == matching[0]["targets"] for t in matching)
        assert all(t["intended_target_id"] == matching[0]["intended_target_id"] for t in matching)


def test_correct_selection_has_backend_timing_and_cannot_complete_twice():
    study, clock, events = setup_study()
    trial = present(study, clock)
    clock.advance(.275)
    event = selection(trial)
    study.observe(dict(event, study_trial_id=999))
    assert not study.results
    study.observe(event)
    study.observe(event)
    assert len(study.results) == 1
    assert study.results[0]["correct"]
    assert study.results[0]["acquisition_time_ms"] == pytest.approx(275)
    assert study.results[0]["detector_target_id"] == 987
    assert study.control_state()[1] == trial["trial_id"]  # Retain visual feedback.
    clock.advance(.401)
    assert study.control_state()[1] is None
    assert study.phase == "between"
    assert len([e for e in events if e["event"] == "trial_completed"]) == 1


def test_unmatched_selection_is_an_error_not_nearest_target_guess():
    study, clock, _ = setup_study()
    trial = present(study, clock)
    event = selection(trial)
    event["source_target"] = {"x": 0, "y": 0, "width": 30, "height": 30}
    study.observe(event)
    assert study.results[0]["selected_target_id"] is None
    assert not study.results[0]["correct"]
    assert study.report()["summary"][trial["mode"]]["unmatched_selections"] == 1


def test_all_conditions_complete_and_score_wrong_answers_without_retry_bias():
    study, clock, _ = setup_study()
    wrong = 0
    for _ in range(72):
        trial = present(study, clock)
        selected = trial["intended_target_id"]
        if len(trial["targets"]) > 1:
            selected = selected % len(trial["targets"]) + 1
            wrong += 1
        clock.advance(.2)
        study.observe(selection(trial, selected))
        clock.advance(.401)
        study.control_state()
        study.heartbeat()
    assert study.phase == "complete"
    report = study.report()
    assert sum(not row["correct"] for row in report["trials"]) == wrong
    assert all(row["completed_trials"] == 24 for row in report["summary"].values())


def test_lost_page_interrupts_without_manufacturing_a_completed_trial():
    study, clock, events = setup_study()
    trial = present(study, clock)
    clock.advance(2.1)
    study.control_state()
    study.observe(selection(trial))
    assert study.phase == "interrupted"
    assert study.interruption == "page_connection_lost"
    assert not study.results
    assert any(e["event"] == "study_interrupted" for e in events)


@pytest.mark.parametrize("dimensions", [(0, 720), (1280, float("nan")), (800, 800), (True, 720)])
def test_invalid_viewport_cannot_start_a_trial(dimensions):
    study, clock, _ = setup_study()
    clock.advance(.801)
    trial = study.state()["trial"]
    with pytest.raises(ValueError):
        study.presented(trial["trial_id"], *dimensions)
    assert study.phase == "presenting"


def test_http_serves_page_and_results_but_has_no_selection_endpoint():
    clock, events = Clock(), []
    study = AcquisitionStudy(1280, 720, participant="http-test", seed=2, pointer="udp", emit=events.append, clock=clock)
    server = StudyServer(study)
    server.start()
    try:
        with urllib.request.urlopen(server.url) as response:
            assert b"requestFullscreen" in response.read()
        study.control_state()
        request = urllib.request.Request(server.url + "api/start", data=b"{}", headers={"Origin": server.url.rstrip("/"), "Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["phase"] == "between"
        with urllib.request.urlopen(server.url + "results.json") as response:
            assert json.load(response)["pointer"] == "udp"
        for endpoint, origin, status in [("api/select", server.url.rstrip("/"), 404), ("api/interrupt", "https://example.com", 403)]:
            request = urllib.request.Request(server.url + endpoint, data=b"{}", headers={"Origin": origin})
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == status
    finally:
        server.close()
