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
        "biases": (0.0, 30.0),
        "gaps": (0.0, 32.0),
        "widths": (20.0,),
        "layouts": ("pair", "toolbar"),
    }

    first = MODULE.evaluate(**arguments)
    second = MODULE.evaluate(**arguments)

    assert first == second
    assert first["summary"]["cell_count"] == 16
    assert first["summary"]["trial_count"] == 64
    assert first["summary"]["placement_success"] == 1.0
    assert first["summary"]["mapping_accuracy"] == 1.0


def test_target_layouts_have_stable_intended_targets():
    pair, pair_intended = MODULE._targets("pair", 32, 8)
    toolbar, toolbar_intended = MODULE._targets("toolbar", 32, 8)

    assert len(pair) == 2
    assert pair_intended.id == 1
    assert len(toolbar) == 4
    assert toolbar_intended.id == 2
