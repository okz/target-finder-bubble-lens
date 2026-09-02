import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "evaluate_synthetic_ambiguity.py"
SPEC = importlib.util.spec_from_file_location("synthetic_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_small_evaluation_is_deterministic_and_complete():
    arguments = {
        "seeds": 4,
        "sigmas": (3.0, 15.0),
        "gaps": (0.0, 32.0),
        "widths": (20.0,),
        "layouts": ("pair", "toolbar"),
    }

    first = MODULE.evaluate(**arguments)
    second = MODULE.evaluate(**arguments)

    assert first == second
    assert first["summary"]["cell_count"] == 8
    assert first["summary"]["trial_count"] == 32
    assert first["summary"]["placement_success"] == 1.0
    assert first["summary"]["mapping_accuracy"] == 1.0


def test_target_layouts_have_stable_intended_targets():
    pair, pair_intended = MODULE._targets("pair", 32, 8)
    toolbar, toolbar_intended = MODULE._targets("toolbar", 32, 8)

    assert len(pair) == 2
    assert pair_intended.id == 1
    assert len(toolbar) == 4
    assert toolbar_intended.id == 2


def test_evaluation_contains_only_noise_driven_selection_ambiguity():
    report = MODULE.evaluate(
        seeds=20,
        sigmas=(25.0,),
        gaps=(0.0,),
        widths=(20.0,),
        layouts=("toolbar",),
    )
    cell = report["cells"][0]

    assert "biases_px" not in report["parameters"]
    assert "calibration_bias_px" not in cell
    assert report["parameters"]["excluded_failure_modes"] == [
        "known_or_user_compensated_offset",
        "unknown_calibration_failure",
    ]
    assert sum(cell["winner_counts"].values()) == 20
    assert len(cell["winner_counts"]) >= 2
    assert cell["selection_ambiguous"] == (cell["baseline_error"] >= 0.20)


def test_trace_is_centered_on_the_intended_target_without_an_offset_parameter():
    _targets, intended = MODULE._targets("pair", 32, 8)
    samples = [
        sample
        for seed in range(200)
        for sample in MODULE._trace(intended, sigma=8, seed=seed)
    ]
    mean_x = sum(sample.x for sample in samples) / len(samples)
    mean_y = sum(sample.y for sample in samples) / len(samples)

    assert abs(mean_x - intended.center.x) < 0.5
    assert abs(mean_y - intended.center.y) < 0.5
