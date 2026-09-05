# Controlled acquisition study

This local task page measures completed dry-run acquisition with the actual
TargetFinder detections and Bubble Gaze Lens overlay. Page geometry is used
only as scoring ground truth; it is never supplied to the target selector.
The existing static `tools/lens_test_page.html` remains a simple visual demo.

## Run

Use one monitor with at least 1000×600 Qt logical pixels. From the repository:

```powershell
& '.\.venv-mamba\python.exe' -m target_finder_toolkit.bubblegazelens `
  --study --study-participant P01 --study-seed 1 --pointer mouse `
  --log artifacts/study-P01-mouse.jsonl
```

For a tracker emitting the existing localhost logical-pixel protocol, replace
`--pointer mouse` with `--pointer udp --udp-port 4242`. Mouse and UDP sessions
are identified separately in the event log and results. Replay input is not
accepted in the live study. No OS clicks are executed in either case.

Open the printed localhost URL in a desktop browser on that monitor. Press
**Enter fullscreen & start**. Do not use an embedded browser panel for actual
measurement: the study assumes that browser fullscreen covers the same physical
monitor as the Qt overlay. Browser CSS coordinates are scaled into the recorded
Qt logical screen size. The viewport must have the same aspect ratio; resizing,
zoom changes that resize the viewport, switching away, or leaving fullscreen
interrupts the session. Initial fullscreen resizing settles before the first
presentation acknowledgement.

Look at the requested numbered control. When a lens opens, look into it and
press Enter to confirm. When there is no lens, Enter confirms the source Bubble
winner. The first accepted selection ends the trial, including a wrong or
unmatched selection. Escape ends the study session (browser fullscreen also
uses Escape); outside the study it retains its normal lens-close behavior.
Moving outside the lens interaction region can still cancel the lens during
a task. Press `q` to stop the application after exporting results.

## Schedule and measurement

- 72 tasks: 24 per condition, with 8 easy/isolated, 8 close-pair and 8 dense tasks.
- The same task layouts and intended IDs occur in every condition. Condition
  order and within-condition task order are shuffled reproducibly by the seed.
  Use different recorded seeds across participants; this is randomization, not
  a guarantee of counterbalancing in a small sample.
- Conditions switch automatically. `--mode` does not override the study's
  schedule. Trial changes clear pending confirmations, fixation evidence,
  cooldown and frozen lens state. Feedback remains visible before advancement.
- Each task follows an 800 ms centre cue. The cue controls presentation timing;
  it does not verify a central gaze fixation. There is no acquisition timeout.
- The page renders the number cue and target board and acknowledges presentation
  after two animation frames. The backend's monotonic clock starts timing when
  that acknowledgement arrives and ends timing when the overlay emits an
  accepted selection. Browser/loopback acknowledgement delay is not calibrated;
  these timings are suitable for formative comparison, not display-latency
  metrology. They include subsequent detector/interaction delays.
- A detector rectangle matches a page target only if exactly one ground-truth
  rectangle has IoU ≥0.5. Unmatched or ambiguous matches are recorded as errors;
  no nearest-target guess or automatic retry improves the score. Detector IDs
  and ground-truth task target IDs are logged separately.

## Results and interruption

The result page exports JSON and CSV with correctness, acquisition time,
condition, density, intended and selected IDs, detector ID, source/lens selection
space, lens openings, first lens-entry latency, and cancellation count. JSON
also records the schedule, screen size, seed, input source, and per-condition
summaries. Accuracy denominators are completed trials; planned and completed
counts are both shown. Median time over all completions and median time over
correct completions are distinct fields.

The append-only application JSONL log contains session metadata, presentation
starts, overlay events and trial completions as they happen. Downloads remain
available while the application is running. If the browser or overlay stops
responding for over two seconds, or the page signals an interruption, the
session ends without turning the in-flight task into a completed selection.
Incomplete results are explicitly labeled. Start a new process/log for a new
session; the runner does not silently resume or recycle completed trials.

Keep raw logs. The page is a measurement instrument, not proof of improved gaze
acquisition. Directional ambiguity, tracker sampling/freshness and dynamic
target identity remain open implementation issues. Human surprise ratings and
whether a lens was needed still require participant/observer input.
