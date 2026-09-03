# Agent instructions

Work autonomously in this repository and optimize for useful progress per token.

## Cost and delegation

- Do not switch models, enable a faster or more expensive mode, or raise reasoning effort unless the user explicitly requests it.
- Use one agent by default. Do not invoke optional subagents, Copilot, Oracle, or another model unless the user explicitly asks or a higher-priority instruction requires it.
- When work is delegated, do not repeat the same investigation or implementation locally. Integrate and verify the delegated result once.
- Ask the user only for decisions that materially change product behavior, safety, or scope. With full workspace access, run ordinary commands yourself.

## Read narrowly

- Start with `rg --files` and `rg -n`. Read only the relevant line ranges; do not dump whole large source files, logs, generated reports, or diffs.
- Before opening a named path, confirm it exists. Do not retry guessed filenames.
- Keep command output bounded. Prefer `git status --short`, `git diff --stat`, targeted diffs, and `pytest -q`.
- After the first exploration pass, retain a short working summary and do not re-read unchanged material.
- Never paste credentials, authentication records, raw Codex databases, or unrelated session history into output.

## Edit and verify incrementally

- Use `apply_patch` for edits. Group related changes instead of making many tiny patches.
- Run the smallest relevant test first after a change. Run the complete validation stack once at the final milestone, not after every edit.
- `python -m pytest -q` already executes the full synthetic regression through `tests/test_synthetic_evaluation.py`; do not also run the standalone 12,000-trial evaluator unless its JSON report must be refreshed.
- `tools/replay_lens.py --contact-sheet ...` already runs the replay and render scenarios; do not invoke the render helper separately.
- For parameter tuning, use a small smoke matrix first. Run the full 100-seed matrix only for the selected configuration.
- Reuse `.venv-mamba` when it exists. Do not recreate the environment or reinstall dependencies unless a required import or command is missing.

## Git and remote checks

- Preserve unrelated changes. Check status before editing and once before commit.
- For this repository, push with the authorized `okz` account using `git -c credential.username=okz`; never print or retrieve the token. If non-interactive use is needed, set `GCM_INTERACTIVE=Never` for that command.
- After pushing, verify the remote SHA once. Poll GitHub Actions no more frequently than every 20–30 seconds and stop once the matching commit completes.
- Do not repeat a hanging push. Stop its exact Git/GCM processes, diagnose account selection, then make one corrected retry.

## Communication

- Keep progress updates short and milestone-based. Do not narrate routine commands or unchanged polls.
- Final responses should lead with the outcome, list only material validation results and remaining risks, and avoid replaying the implementation history.
