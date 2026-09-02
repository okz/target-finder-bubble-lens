# Bubble Gaze Lens baseline

Recorded on 2 September 2026 for the ambiguity-triggered lens vertical slice.

## Repository

- Upstream: `ahmedbenakouche/target_finder_toolkit`
- Baseline commit: `b02ee2f7e077523df425fe948c76f06daf9c78be`
- Package version: `0.2.0`
- Model default: `yolo26n-640`
- Detector defaults: confidence `0.40`, IoU `0.30`, capture interval `1/30 s`

## Local environment

- OS: Windows 11 Enterprise, build 26200, 64-bit
- Qt screen geometry: 1280 x 720 logical pixels
- Qt device pixel ratio: 1.5 (1920 x 1080 physical pixels)
- Qt available geometry: 1280 x 672 logical pixels
- Python: 3.11.16, managed by Mamba
- Prefix: `.venv-mamba` (covered by the `.venv*` ignore rule)

Create or refresh the environment from the repository root:

```powershell
& 'C:\oz\git\.env\Library\bin\mamba.exe' create --yes `
  --prefix '.venv-mamba' --channel conda-forge python=3.11 pip
& '.\.venv-mamba\python.exe' -m pip install --editable '.[dev]'
```

Run the deterministic milestone checks:

```powershell
& '.\.venv-mamba\python.exe' -m pytest -q `
  tests\test_lens_geometry.py `
  tests\test_lens_ambiguity.py `
  tests\test_lens_state.py
& '.\.venv-mamba\python.exe' tools\replay_lens.py `
  --scenarios tests\fixtures\lens `
  --report artifacts\replay-report.json `
  --contact-sheet artifacts\contact-sheet.png
```

## Safety status

The upstream `bubblecursor` demo was inspected but not launched unattended. It
registers a global mouse listener and redirects clicks through `pyautogui`; it
does not provide a dry-run mode. The new lens code contains no GUI, global input
listener, or click-injection path. Live baseline timing and visual validation
remain part of the later human-gated mouse-proxy milestone.
