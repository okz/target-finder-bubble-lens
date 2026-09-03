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

The corrected synthetic automatic-trigger gate passes with 80.68%
selection-ambiguity recall and 1.39% easy-cell false opens. Calibration offsets
are excluded from those statistics. Gate B is still required and is not
permission to enable clicks. After the run, make an explicit go, pivot, or stop
decision before any selection-execution work.
