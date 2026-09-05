# Human Gate B runbook

This gate evaluates the interaction with real gaze. It must remain dry-run: the
prototype has no operating-system click path.

## Prerequisites

- One monitor only.
- The tracker-specific adapter emits primary-screen **logical pixels** as UDP
  JSON to `127.0.0.1:4242`.
- A valid packet has this shape:

  ```json
  {"t_ms": 1234.5, "x": 812.2, "y": 498.1, "valid": true, "screen": "primary"}
  ```

- Invalid tracking may omit `x` and `y`, but must retain `t_ms`, `valid: false`,
  and `screen: "primary"`.
- At the recorded Windows 150% display scaling, 1920x1080 physical pixels map
  to 1280x720 Qt logical pixels. The tracker adapter must perform this conversion;
  the lens intentionally rejects normalized input.

## Conditions

Choose a randomized condition order before starting. Use a different log file
for each condition:

```powershell
bubblegazelens --pointer udp --udp-port 4242 --mode bubble `
  --log artifacts/gate-b-bubble.jsonl
bubblegazelens --pointer udp --udp-port 4242 --mode forced-lens `
  --log artifacts/gate-b-forced-lens.jsonl
bubblegazelens --pointer udp --udp-port 4242 --mode auto-lens `
  --log artifacts/gate-b-auto-lens.jsonl
```

The forced-lens condition still requires a stable fixation near at least two
plausible targets; it removes only the nearest-target dominance threshold. The
automatic condition uses the unchanged initial trigger parameters.

Run 24 target tasks per condition:

- 8 easy or isolated targets;
- 8 close pairs;
- 8 dense toolbar or menu targets.

Start with the design owner. Continue to four formative users only if that pass
finds no blocking issue.

## Record per task

Enter uses a common dry-run confirmation path in all three conditions, even
when no lens opens. A selection is logged only after the next pointer update and
the current detector snapshot still support the requested target. Feedback and
cooldown prevent duplicate acceptance; held Enter does not auto-repeat.
The accepted event contains the interaction mode, source/lens selection space,
target ID, snapshot generations and acceptance time. Rejected requests are
separate events. `lens_first_entry` provides first-entry latency directly.
Task starts and intended target IDs still need to be recorded by the study
harness or observer; the application does not infer trial boundaries.

- participant, condition, randomized order, task ID, and target-density class;
- intended and dry-run-selected target IDs, correctness, and acquisition time;
- whether the lens opened and whether it was needed;
- false open or missed open;
- lens-appearance to first gaze-entry time;
- cancellation;
- surprise rating from 1 to 5;
- yes/no response to “I knew where to look next”; and
- a brief observation when behavior is unexpected.

Retain the application JSONL logs. They include tracking transitions, rejected
packet counts, trigger state, candidate IDs, fixation spread, ambiguity score,
and the fixation-center offset from the current Bubble winner.

## Go criteria

- Forced lens improves close/dense accuracy over Bubble-only.
- Automatic lens approaches forced-lens accuracy.
- Easy-task false opens are at most 15%.
- Median surprise is at most 2 after the first three lens exposures.
- At least 80% of lens appearances are followed by gaze entry within 700 ms.
- Invalid tracking closes an open lens safely.

The corrected synthetic display-availability gate currently fails: 72.26% of
ambiguous-cell trials display a lens containing the intended target, below the
unchanged 80% criterion. The older 80.68% result counts trigger attempts before
runtime geometry suppression. Displayed easy-cell false opens are 1.36%.
Calibration offsets remain excluded; completed acquisition is not simulated.
Resolve the display shortfall and trial-recording workflow before using this
comparison for a product decision. Exploratory dry-run observations may inform
that design work. Gate B does not enable clicks; make an explicit go, pivot, or
stop decision before any selection-execution work.
