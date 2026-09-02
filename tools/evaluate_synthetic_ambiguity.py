"""Run the deterministic Monte Carlo ambiguity-trigger evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable

from target_finder_toolkit.lens_core import (
    LensConfig,
    LensStateMachine,
    Point,
    PointerSample,
    Rect,
    TargetRect,
    bubble_solution,
    choose_lens_rect,
    choose_source_crop,
    rect_intersection_area,
    source_to_lens,
    transform_target_to_lens,
)


DEFAULT_SIGMAS = (3.0, 8.0, 15.0, 25.0)
DEFAULT_GAPS = (0.0, 8.0, 16.0, 32.0, 64.0)
DEFAULT_WIDTHS = (20.0, 32.0, 52.0)
DEFAULT_LAYOUTS = ("pair", "toolbar")
SCREEN = Rect(0, 0, 1920, 1080)


def _targets(layout: str, width: float, gap: float) -> tuple[tuple[TargetRect, ...], TargetRect]:
    count = 2 if layout == "pair" else 4
    total_width = count * width + (count - 1) * gap
    left = (SCREEN.width - total_width) / 2.0
    top = (SCREEN.height - width) / 2.0
    targets = tuple(
        TargetRect(
            id=index + 1,
            x=left + index * (width + gap),
            y=top,
            width=width,
            height=width,
            score=0.98 - index * 0.01,
            class_id=index % 3,
        )
        for index in range(count)
    )
    intended = targets[0] if layout == "pair" else targets[1]
    return targets, intended


def _trace(
    intended: TargetRect,
    sigma: float,
    seed: int,
    end_ms: int = 300,
) -> tuple[PointerSample, ...]:
    rng = random.Random(seed)
    mean = intended.center
    return tuple(
        PointerSample(
            t_ms=float(t_ms),
            x=rng.gauss(mean.x, sigma),
            y=rng.gauss(mean.y, sigma),
            valid=True,
        )
        for t_ms in range(0, end_ms + 1, 20)
    )


def _mapping_checks(targets: tuple[TargetRect, ...], config: LensConfig) -> tuple[int, int, bool]:
    placement = choose_lens_rect(targets, SCREEN, config)
    placement_ok = rect_intersection_area(placement.rect, placement.source_hull) == 0.0
    crop = choose_source_crop(
        placement.source_hull.center,
        placement.rect,
        SCREEN,
        config.lens_scale,
    )
    transformed = tuple(transform_target_to_lens(target, crop, placement.rect) for target in targets)
    passed = 0
    for source_target, lens_target in zip(targets, transformed):
        destination = source_to_lens(source_target.center, crop, placement.rect)
        winner = bubble_solution(destination, transformed, config).primary
        passed += int(winner is not None and winner.id == lens_target.id)
    return passed, len(targets), placement_ok


def evaluate(
    *,
    seeds: int = 100,
    sigmas: Iterable[float] = DEFAULT_SIGMAS,
    gaps: Iterable[float] = DEFAULT_GAPS,
    widths: Iterable[float] = DEFAULT_WIDTHS,
    layouts: Iterable[str] = DEFAULT_LAYOUTS,
    config: LensConfig | None = None,
) -> dict:
    config = config or LensConfig()
    cells = []
    mapping_passed = 0
    mapping_total = 0
    placement_passed = 0
    placement_total = 0
    cell_index = 0

    for layout in layouts:
        for width in widths:
            for gap in gaps:
                targets, intended = _targets(layout, width, gap)
                mapped, mapping_count, placement_ok = _mapping_checks(targets, config)
                mapping_passed += mapped
                mapping_total += mapping_count
                placement_passed += int(placement_ok)
                placement_total += 1
                for sigma in sigmas:
                    errors = 0
                    opens = 0
                    winner_counts: dict[int | None, int] = {}
                    open_times: list[float] = []
                    candidate_sizes: list[int] = []
                    for seed in range(seeds):
                        trace = _trace(intended, sigma, cell_index * 100_000 + seed)
                        winner = bubble_solution(trace[-1].point, targets, config).primary
                        winner_id = None if winner is None else winner.id
                        winner_counts[winner_id] = winner_counts.get(winner_id, 0) + 1
                        errors += int(winner_id != intended.id)
                        machine = LensStateMachine(config)
                        for sample in trace:
                            step = machine.step(
                                sample.t_ms,
                                sample,
                                targets,
                                clean_frame_available=True,
                            )
                            if "lens_opened" in step.events:
                                opens += 1
                                open_times.append(sample.t_ms)
                                candidate_sizes.append(len(step.frozen_candidate_ids))
                                break
                    baseline_error = errors / seeds
                    probabilities = [count / seeds for count in winner_counts.values()]
                    winner_entropy_bits = -sum(
                        probability * math.log2(probability)
                        for probability in probabilities
                        if probability > 0.0
                    )
                    cells.append(
                        {
                            "layout": layout,
                            "target_width_px": width,
                            "target_gap_px": gap,
                            "noise_sigma_px": sigma,
                            "trials": seeds,
                            "intended_target_id": intended.id,
                            "winner_counts": {
                                "none" if target_id is None else str(target_id): count
                                for target_id, count in sorted(
                                    winner_counts.items(),
                                    key=lambda item: (-1 if item[0] is None else item[0]),
                                )
                            },
                            "winner_entropy_bits": winner_entropy_bits,
                            "baseline_error": baseline_error,
                            "selection_ambiguous": baseline_error >= 0.20
                            and len(winner_counts) >= 2,
                            "trigger_open_rate": opens / seeds,
                            "median_open_time_ms": (
                                statistics.median(open_times) if open_times else None
                            ),
                            "mean_candidate_set_size": (
                                statistics.fmean(candidate_sizes) if candidate_sizes else None
                            ),
                        }
                    )
                    cell_index += 1

    ambiguous = [cell for cell in cells if cell["selection_ambiguous"]]
    easy = [cell for cell in cells if cell["baseline_error"] <= 0.05]
    ambiguous_trials = sum(cell["trials"] for cell in ambiguous)
    easy_trials = sum(cell["trials"] for cell in easy)
    selection_ambiguity_recall = (
        sum(cell["trigger_open_rate"] * cell["trials"] for cell in ambiguous)
        / ambiguous_trials
        if ambiguous_trials
        else None
    )
    false_open_rate = (
        sum(cell["trigger_open_rate"] * cell["trials"] for cell in easy) / easy_trials
        if easy_trials
        else None
    )
    ambiguous_open_times = [
        cell["median_open_time_ms"]
        for cell in ambiguous
        if cell["median_open_time_ms"] is not None
    ]
    median_open_time = (
        statistics.median(ambiguous_open_times) if ambiguous_open_times else None
    )
    candidate_sizes = [
        cell["mean_candidate_set_size"]
        for cell in cells
        if cell["mean_candidate_set_size"] is not None
    ]
    mapping_accuracy = mapping_passed / mapping_total if mapping_total else None
    placement_success = placement_passed / placement_total if placement_total else None

    gates = {
        "selection_ambiguity_recall_at_least_0_80": selection_ambiguity_recall is not None
        and selection_ambiguity_recall >= 0.80,
        "easy_false_open_at_most_0_10": false_open_rate is not None and false_open_rate <= 0.10,
        "median_open_time_190_to_260_ms": median_open_time is not None
        and 190 <= median_open_time <= 260,
        "placement_success_1_00": placement_success == 1.0,
        "mapping_accuracy_1_00": mapping_accuracy == 1.0,
    }
    return {
        "passed": all(gates.values()),
        "parameters": {
            "seeds_per_cell": seeds,
            "sigmas_px": list(sigmas),
            "gaps_px": list(gaps),
            "target_widths_px": list(widths),
            "layouts": list(layouts),
            "trace_end_ms": 300,
            "trace_interval_ms": 20,
            "baseline_selection": "ordinary Bubble winner at final trace sample",
            "selection_ambiguity_definition": (
                "zero-bias noise produces at least two competing Bubble winners "
                "and intended-target error >= 0.20"
            ),
            "excluded_failure_modes": [
                "known_or_user_compensated_offset",
                "unknown_calibration_failure",
            ],
            "trigger_config": {
                "fixation_r90_px": config.fixation_r90_px,
                "uncertainty_radius_px": config.uncertainty_radius_px,
                "ambiguity_threshold": config.ambiguity_threshold,
                "ambiguous_sample_ratio": config.ambiguous_sample_ratio,
            },
        },
        "summary": {
            "cell_count": len(cells),
            "trial_count": len(cells) * seeds,
            "ambiguous_cell_count": len(ambiguous),
            "easy_cell_count": len(easy),
            "selection_ambiguity_recall": selection_ambiguity_recall,
            "false_open_rate": false_open_rate,
            "median_open_time_ms": median_open_time,
            "mean_candidate_set_size": (
                statistics.fmean(candidate_sizes) if candidate_sizes else None
            ),
            "placement_success": placement_success,
            "mapping_accuracy": mapping_accuracy,
        },
        "gates": gates,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--fixation-r90-px", type=float, default=35.0)
    parser.add_argument("--uncertainty-radius-px", type=float, default=48.0)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.65)
    parser.add_argument("--ambiguous-sample-ratio", type=float, default=0.75)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    config = LensConfig(
        fixation_r90_px=args.fixation_r90_px,
        uncertainty_radius_px=args.uncertainty_radius_px,
        ambiguity_threshold=args.ambiguity_threshold,
        ambiguous_sample_ratio=args.ambiguous_sample_ratio,
    )
    report = evaluate(seeds=args.seeds, config=config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(f"Gates: {json.dumps(report['gates'])}")
    print(f"Report: {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
