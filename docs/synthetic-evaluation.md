# Synthetic ambiguity evaluation

Run on 2 September 2026 with the initial trigger configuration. The deterministic
matrix contains 360 cells and 36,000 trials across two-target and toolbar layouts,
20/32/52 px targets, 0/8/16/32/64 px gaps, 3/8/15/25 px Gaussian noise, and
0/15/30 px calibration bias.

```powershell
& '.\.venv-mamba\python.exe' tools\evaluate_synthetic_ambiguity.py `
  --seeds 100 `
  --report artifacts\synthetic-evaluation.json
```

## Initial gate result

| Gate | Result | Pass? |
|---|---:|:---:|
| Recall on cells with baseline error >= 20% | 20.51% | No |
| False-open rate on cells with baseline error <= 5% | 4.85% | Yes |
| Median open time | 200 ms | Yes |
| Placement success | 100% | Yes |
| Lens mapping accuracy | 100% | Yes |

There were 168 ambiguous cells and 136 easy cells. The automatic trigger gate is
therefore **failed**, even though timing, placement, mapping, and easy-cell
specificity pass. The acceptance threshold has not been relaxed.

## One preliminary threshold sweep

The following 20-seed-per-cell sweep was used only to check whether a simple
threshold adjustment resolves the miss rate. It does not replace the 100-seed
gate run.

| Fixation r90 | Uncertainty radius | Ambiguity threshold | Recall | False opens |
|---:|---:|---:|---:|---:|
| 45 px | 48 px | 0.65 | 21.55% | 4.97% |
| 55 px | 48 px | 0.65 | 22.21% | 4.97% |
| 55 px | 64 px | 0.65 | 42.53% | 10.61% |
| 55 px | 64 px | 0.50 | 67.64% | 26.52% |
| 60 px | 80 px | 0.65 | 61.61% | 18.82% |
| 60 px | 80 px | 0.50 | 82.04% | 37.80% |

None of these candidates satisfies both the 80% recall and 10% false-open gates.
The evidence currently supports keeping the initial parameters for live logging
and evaluating Bubble-only, forced-lens, and automatic-lens conditions separately
at Human Gate B. It does not support enabling operating-system clicks.
