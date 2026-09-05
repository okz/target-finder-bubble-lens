# Synthetic selection-ambiguity evaluation

Updated on 5 September 2026 to count runtime display availability as well as
trigger attempts. Known or user-compensated offsets and unknown calibration
failures remain explicitly excluded.

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

## Corrected display-availability gate

Each trigger attempt now passes its actual candidate set through
`prepare_lens_layout`, the same placement/crop gate used by the overlay. A
successful assistance opportunity requires a displayable lens containing the
intended target. The screen remains the original 1920×1080 logical-pixel
synthetic screen; other screen sizes still need evaluation.

| Gate | Result | Pass? |
|---|---:|:---:|
| Displayed lens with intended target on ambiguous cells | 72.26% | **No: requires 80%** |
| Displayed false-open rate on easy cells | 1.36% | Yes |
| Median display time over displayed ambiguous-cell trials | 240 ms | Yes |
| Placement success | 100% | Yes |
| Lens mapping accuracy | 100% | Yes |
| Full-layout suppression below the 2× minimum | 40% | Informational |

There are 19 selection-ambiguous cells and 75 easy cells. Across the full
12,000-trial matrix there are 2,532 trigger attempts, 2,270 displayable lenses,
and 262 runtime suppressions. Every displayed lens includes the intended
target in this matrix. The unchanged recall requirement fails; thresholds have
not been relaxed. The historical trigger-only recall remains 80.68%, with
1.39% trigger-only false opens, and is retained as a separate regression metric.

Mapping accuracy still measures exact transformed target centers only when a
complete layout fits. It does not measure noisy lens selection or acquisition.
The 40% full-layout stress suppression is separate from runtime suppression of
actual plausible candidate sets. For a reproducible dense cell (52 px toolbar,
8 px gap, sigma 25, seeds 0–99), 82 trigger attempts yield 15 displayable lenses
and 67 runtime suppressions.

The report explicitly marks acquisition as unevaluated, with null completed
selection and acquisition-accuracy values. `passed` describes only the listed
display/geometry gates. CLI execution returns nonzero when a product gate
fails. The regression suite verifies this known shortfall as correct evaluator
output; passing software tests must not be reported as passing the product gate.

The earlier calibration-contaminated 20.51% result and the first corrected but
structurally flawed 14.26% result are superseded. Neither should be used for
product decisions.

Use these results to resolve geometry and measurement gaps before interpreting
Human Gate B as evidence of improved acquisition. They do not authorize
operating-system click execution.
