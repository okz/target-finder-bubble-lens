# Synthetic selection-ambiguity evaluation

Run on 3 September 2026 after restricting the experiment to selection
ambiguity. Known or user-compensated offsets and unknown calibration failures
are explicitly excluded.

The deterministic matrix contains 120 cells and 12,000 trials across
two-target and toolbar layouts, 20/32/52 px targets, 0/8/16/32/64 px gaps, and
3/8/15/25 px zero-mean Gaussian measurement noise. Every trace is centered on
the intended target; no calibration-bias input or code path exists.

```powershell
& '.\.venv-mamba\python.exe' tools\evaluate_synthetic_ambiguity.py `
  --seeds 100 `
  --report artifacts\synthetic-evaluation.json
```

## Ground truth

For each cell, ordinary Bubble selection is evaluated at the final gaze sample
over 100 seeds. A cell is selection-ambiguous only when:

- zero-bias measurement noise produces at least two different Bubble winners;
  and
- the intended-target error rate is at least 20%.

The report records winner counts and Shannon entropy for auditability. A
consistently displaced but confident winner cannot satisfy this definition.
Easy cells have at most 5% intended-target error.

## Trigger correction

The original trigger treated a large gaze `r90` as pointer movement. That
rejected stationary but noisy fixations—the exact cases the lens should help.
The corrected trigger separates two observations:

- **Movement:** drift between the robust centers of the first and second halves
  of the window, plus a sudden-jump guard relative to preceding noise.
- **Selection uncertainty:** gaze `r90` relative to the distance advantage of
  the nearest target over the second-nearest target.

For two plausible targets, the selection-noise score is:

```text
r90 / (r90 + (d2 - d1) + 0.5)
```

An exact tie scores 1.0. The default threshold is 0.51, fixation drift is
limited to 50 px, and a sudden jump must remain within the larger of 12 px or
three times the preceding `r90`. The full 200 ms window must remain ambiguous;
a broken window restarts persistence.

## Automated gate result

| Gate | Result | Pass? |
|---|---:|:---:|
| Recall on noise-driven selection-ambiguous cells | 80.68% | Yes |
| False-open rate on easy cells | 1.39% | Yes |
| Median open time | 240 ms | Yes |
| Placement success | 100% | Yes |
| Lens mapping accuracy | 100% | Yes |
| Full-layout suppression below the 2× minimum | 40% | Informational |

There are 19 selection-ambiguous cells and 75 easy cells. All five automated
gates pass without calibration-offset cases or relaxed acceptance thresholds.
Mapping accuracy is measured only when the full candidate layout can be shown
at the configured 2× minimum. The separate 40% suppression figure is an
intentionally conservative stress check over complete synthetic toolbars,
including distant controls that would not normally be in a runtime plausible
candidate set; suppression produces no selectable lens.
The 100-seed full matrix is also a regression test, preventing future code from
silently restoring the previous failure mode.

The earlier calibration-contaminated 20.51% result and the first corrected but
structurally flawed 14.26% result are superseded. Neither should be used for
product decisions.

These results authorize mouse/replay evaluation and preparation for Human Gate
B. They do not authorize operating-system click execution.
