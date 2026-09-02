# Synthetic selection-ambiguity evaluation

Run on 2 September 2026 after restricting the experiment to selection
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

## Corrected gate result

| Gate | Result | Pass? |
|---|---:|:---:|
| Recall on noise-driven selection-ambiguous cells | 14.26% | No |
| False-open rate on easy cells | 3.96% | Yes |
| Median open time | 200 ms | Yes |
| Placement success | 100% | Yes |
| Lens mapping accuracy | 100% | Yes |

There are 19 selection-ambiguous cells and 75 easy cells. The previous
calibration-contaminated 20.51% recall value is superseded and must not be used.

The remaining failure is genuine: most ambiguous cells occur at 25 px noise,
and the current trigger often classifies that noisy fixation as movement because
its fixation `r90` exceeds 35 px. This is a selection-ambiguity problem, not an
offset problem.

## Limited in-scope sensitivity check

Two 12,000-trial alternatives were checked without calibration bias:

| Fixation r90 | Uncertainty radius | Ambiguity threshold | Recall | False opens |
|---:|---:|---:|---:|---:|
| 35 px | 48 px | 0.65 | 14.26% | 3.96% |
| 65 px | 64 px | 0.65 | 44.68% | 8.16% |
| 65 px | 64 px | 0.50 | 67.53% | 25.88% |

The first relaxed candidate remains below the 80% recall gate. Relaxing the
dominance threshold further still misses recall while exceeding the 10%
false-open gate. Production defaults therefore remain unchanged pending a
better fixation-versus-measurement-noise discriminator. No acceptance threshold
has been relaxed, and no operating-system click path is enabled.
