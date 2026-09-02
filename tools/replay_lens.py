"""Run deterministic ambiguity-trigger scenarios and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from target_finder_toolkit.lens_core import (
    LensConfig,
    LensStateMachine,
    PointerSample,
    TargetRect,
)


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    scenarios: list[dict[str, Any]] = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            scenarios.extend(payload)
        else:
            scenarios.append(payload)
    return scenarios


def _config(values: dict[str, Any]) -> LensConfig:
    values = dict(values)
    if "selectable_class_ids" in values:
        values["selectable_class_ids"] = frozenset(values["selectable_class_ids"])
    return LensConfig(**values)


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    config = _config(scenario.get("config", {}))
    targets = tuple(TargetRect(**target) for target in scenario["targets"])
    machine = LensStateMachine(config)
    transitions: list[dict[str, Any]] = []
    open_time_ms: float | None = None

    for sample_values in scenario["trace"]:
        sample = PointerSample(**sample_values)
        step = machine.step(
            sample.t_ms,
            sample,
            targets,
            clean_frame_available=scenario.get("clean_frame_available", True),
        )
        if step.events:
            transitions.append(
                {
                    "t_ms": sample.t_ms,
                    "state": step.state.value,
                    "events": list(step.events),
                    "ambiguous_for_ms": step.ambiguous_for_ms,
                    "candidate_ids": list(step.frozen_candidate_ids),
                }
            )
        if "lens_opened" in step.events and open_time_ms is None:
            open_time_ms = sample.t_ms

    expected = scenario["expected"]
    opens = open_time_ms is not None
    passed = opens == expected["lens_opens"]
    if passed and opens:
        minimum = expected.get("open_time_min_ms", expected.get("open_time_ms", open_time_ms))
        maximum = expected.get("open_time_max_ms", expected.get("open_time_ms", open_time_ms))
        passed = minimum <= open_time_ms <= maximum

    return {
        "name": scenario["name"],
        "passed": passed,
        "lens_opens": opens,
        "open_time_ms": open_time_ms,
        "expected": expected,
        "transitions": transitions,
        "effective_config": asdict(config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    results = [run_scenario(scenario) for scenario in _load_scenarios(args.scenarios)]
    report = {
        "passed": all(result["passed"] for result in results),
        "scenario_count": len(results),
        "passed_count": sum(result["passed"] for result in results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=sorted), encoding="utf-8")
    print(
        f"{report['passed_count']}/{report['scenario_count']} scenarios passed; "
        f"report: {args.report}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
