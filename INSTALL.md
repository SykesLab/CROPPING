# Install Guide

End-to-end setup for the CROPPING pipeline (Preprocessing, Calibration,
Training, Inference).

## TL;DR

From the repo root:

```bash
python install.py
```

This creates a `venv/`, installs every Python dependency, runs an editable
install of the project, and prints how to launch each GUI. CPU PyTorch by
default; pass `--cuda 12.1` (or 11.8 / 12.4 / 12.6) for the matching wheel.

When it finishes, activate the venv and you're done:

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

Then any of:

```bash
cropping-preprocess          # Preprocessing GUI
cropping-calibrate           # Calibration GUI
cropping-train               # Training GUI
cropping-infer               # Inference GUI
```

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10–3.13 | 3.11.x required if you also need the Phantom SDK (.cine reading) |
| NumPy | < 2.0 | Pinned for Phantom SDK compatibility |
| PyTorch | 1.9.0+ | CUDA strongly recommended for training (see below) |

## Three install paths

Pick whichever fits your workflow.

### 1. One-shot installer — `python install.py`

The recommended path for a fresh machine.

```bash
python install.py                # CPU PyTorch
python install.py --cuda 12.1    # CUDA 12.1 PyTorch (also: 11.8 / 12.4 / 12.6)
python install.py --check        # check existing env, print what's missing
python install.py --install      # install missing packages into the CURRENT env
```

What it does:
1. Verifies Python ≥ 3.10
2. Creates `venv/` (skips if it already exists)
3. Installs PyTorch from the appropriate index URL
4. Installs all other dependencies
5. Installs the project in editable mode (`pip install -e .`)
6. Verifies imports
7. Prints launch commands

### 2. Editable install (you already have a venv)

```bash
pip install -e ".[all]"
```

`[all]` pulls in `[training]` (torch, torchvision, tensorboard) and `[dev]`
(pytest, pytest-timeout). Use `[training]` or `[dev]` individually if you
want a slimmer install.

### 3. Plain `requirements.txt`

```bash
pip install -r requirements.txt
```

This installs every Python dep but does NOT install the project itself, so
the `cropping-*` console scripts won't be on your PATH. You'll need to run
the modules from the repo root (`python -m Inference` etc.).

## CUDA / GPU setup

Training is 10–100× faster with CUDA. Two ways to get a CUDA-enabled torch:

**Easy** — let the installer pick the wheel:

```bash
python install.py --cuda 12.1
```

**Manual** — if you want to add CUDA to an existing install:

```bash
# Step 1: check your CUDA version
nvidia-smi
# (look at "CUDA Version: X.X" in the top-right banner)

# Step 2: uninstall the CPU build
pip uninstall torch torchvision

# Step 3: install the matching CUDA build (pick ONE)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 4: verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expect: CUDA: True
```

## Optional / hardware-only dependencies

These are NOT installed by default — only needed if you're driving the
specific hardware.

### Phantom SDK (`pyphantom`) — required for .cine reading

Used by Preprocessing and the Inference `.cine input` mode. Not on PyPI;
obtain the wheel from Vision Research / your institution.

```bash
pip install "path/to/pyphantom-3.11.11.806-py311-none-any.whl"
```

You may also need to add the SDK's DLL directory to PATH (Windows:
`...\PhantomSDK\Bin\Win64`) and install the bundled Visual C++
Redistributable. **Python 3.11.x is required** by the wheel above.

Verify:

```bash
python -c "import pyphantom; print('OK')"
```

If you only ever feed the Inference GUI pre-cropped PNGs, you can skip
pyphantom entirely.

### Lab calibration capture (Step 0 only)

[Calibration/lab_capture/](Calibration/lab_capture/) drives a Phantom
camera + ThorLabs KDC101 stage + Arduino to record a calibration z-stack.
Hardware deps live in their own file:

```bash
pip install -r Calibration/lab_capture/requirements.txt
```

This adds `pythonnet` (.NET interop for the ThorLabs Kinesis DLLs) and
`pyfirmata` (Arduino Firmata protocol). Kinesis DLLs must exist at
`C:\Program Files\Thorlabs\Kinesis\`.

A convenience batch file is also provided:
[Calibration/lab_capture/setup_lab_env.bat](Calibration/lab_capture/setup_lab_env.bat)
creates a dedicated `phantom_env` venv against Python 3.11 and installs
pyphantom + the lab_capture deps in one go.

## Verify the install

```bash
python install.py --check
```

Prints every package the pipeline expects, with versions and a CUDA flag
on torch. Exits 0 if everything is present, 1 otherwise.

You can also run the test suite:

```bash
python -m pytest Extras/tests/ -v
```

## Troubleshooting

- **`pip install -e .` errors with a torch / wheel-builder failure.**
  Install torch first (`python install.py --cuda 12.1` or the manual
  steps above), then re-run the editable install.
- **`ModuleNotFoundError: pyphantom`.** You're running Preprocessing or
  the Inference `.cine` mode without the Phantom SDK. Install pyphantom
  (above) or use the Inference GUI's `Precropped PNG` mode instead.
- **`pyfirmata` errors with `inspect.getargspec`.** Known issue on
  Python 3.11+; the lab_capture scripts patch this automatically — no
  manual fix needed. If you see it elsewhere, you've imported pyfirmata
  outside one of the patched scripts.
- **`cropping-infer` (or the other console scripts) not found.** You ran
  `pip install -r requirements.txt` instead of an editable install. Run
  `pip install -e .` from the repo root, or use
  `python -m Inference` instead.

## Layout summary

| File | Purpose |
|---|---|
| [pyproject.toml](pyproject.toml) | Source of truth for deps + console scripts |
| [requirements.txt](requirements.txt) | Flat mirror for `pip install -r` users |
| [install.py](install.py) | One-shot installer (creates venv, picks CUDA, etc.) |
| [Calibration/lab_capture/requirements.txt](Calibration/lab_capture/requirements.txt) | Hardware-only deps (pythonnet, pyfirmata) |
| [Calibration/lab_capture/setup_lab_env.bat](Calibration/lab_capture/setup_lab_env.bat) | Auto-creates the dedicated lab venv on Windows |
