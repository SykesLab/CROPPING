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

## Requirements — read this carefully

This pipeline requires **Python 3.11 specifically.** Not 3.10. Not 3.12, 3.13, or 3.14.
Both `install.py` and `pyproject.toml` enforce this. Other deps cascade from
this constraint.

| Requirement | Version | Why this constraint exists |
|---|---|---|
| Python | **== 3.11.x** | The `pyphantom` wheel is built `py311-none-any` and will not install on any other Python version. `install.py` uses `py -3.11` on Windows to find it; if 3.11 isn't present the installer aborts with instructions. `pyproject.toml`'s `requires-python = ">=3.11,<3.12"` enforces the same constraint at `pip install` time. |
| NumPy | **< 2.0** | NumPy 2.x breaks the `pyphantom` binary ABI. Pinning numpy `<2.0` cascades through `pip`'s resolver to compatible scipy/opencv/pandas versions automatically. |
| PyTorch | 1.9.0+ | Installed by `install.py` from the CUDA-specific index URL, not from PyPI. CUDA strongly recommended for training (10–100× speedup); CPU is fine for inference. |
| Other deps | see `pyproject.toml` | Lower bounds only. Pip's resolver picks compatible upper bounds via the numpy `<2.0` cascade. |

If you have multiple Python versions installed, **the installer will find 3.11
automatically via `py -3.11` (Windows) or `python3.11` (Linux/Mac).** You don't
need to remove other Python versions or change PATH.

If 3.11 isn't on the system, get it from
<https://www.python.org/downloads/release/python-3119/>. The default install
options are fine.

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

Used by Preprocessing and the Inference `.cine input` mode. Not on PyPI; obtain
the wheel from your institution's Phantom SDK distribution. The wheel filename
looks like `pyphantom-3.11.11.806-py311-none-any.whl` (Phantom SDK 11) — **the
`py311` suffix is the Python-version constraint**: this wheel only installs on
Python 3.11. Newer SDK releases may carry wheels for newer Python versions; if
your wheel filename has a different `pyXXX` suffix, the rest of the pipeline
will need to be installed against that version.

#### Step 1 — Install the wheel

If you put `pyphantom-*.whl` next to `install.py` (or in `./wheels/`),
`install.py` will pick it up and install it automatically. Otherwise:

```bash
pip install "path/to/pyphantom-3.11.11.806-py311-none-any.whl"
```

#### Step 2 — Add the SDK's DLL directory to PATH (Windows)

The wheel installs the Python bindings, but `pyphantom` also needs the SDK's
native DLLs at runtime. They live in your SDK install at e.g.
`C:\Users\<you>\OneDrive\My Documents\Phantom\PhSDK11\Bin\Win64` (path varies
by where you unpacked the SDK).

In a regular `cmd` (not Powershell), run:

```cmd
setx PATH "%PATH%;C:\Users\justi\OneDrive\My Documents\Phantom\PhSDK11\Bin\Win64"
```

Replace the path with your actual SDK location. **Then close and reopen any
terminal/IDE** — `setx` only takes effect in new processes.

User PATH is fine; you do NOT need admin / system PATH for this. To check the
update worked, open a fresh terminal and run `echo %PATH%` — you should see
the SDK Bin directory at the end.

#### Step 3 — Install the Visual C++ Redistributable

The Phantom SDK depends on the Microsoft VC++ runtime. If the SDK package you
received includes a `vcredist_x64.exe`, run it. If not, install the latest
from <https://aka.ms/vs/17/release/vc_redist.x64.exe> (Microsoft's permalink).

#### Step 4 — Verify

```bash
python -c "import pyphantom; print('OK')"
```

You should see `OK`. If you see `ImportError: DLL load failed`, the PATH
update from Step 2 hasn't reached this terminal — close and reopen.

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

- **`install.py` aborts with "Python 3.11 not found".** You need Python 3.11
  installed on the system. Get it from
  <https://www.python.org/downloads/release/python-3119/>. The default install
  options are fine — leave PATH alone and don't uncheck the `py` launcher.
  After installing, re-run `install.py`; it'll find 3.11 via the launcher.
- **`pip install -e .` errors with `Requires Python ==3.11.*`.** Your active
  interpreter isn't 3.11. Either run `py -3.11 -m pip install -e .` or use
  `install.py` which builds a fresh 3.11 venv automatically.
- **`pip install -e .` errors with a torch / wheel-builder failure.**
  Install torch first (`python install.py --cuda 12.1` or the manual steps
  above), then re-run the editable install.
- **`ModuleNotFoundError: pyphantom`.** You're running Preprocessing or the
  Inference `.cine` mode without the Phantom SDK. Drop `pyphantom-*.whl` next
  to `install.py` and re-run; or install manually per the Phantom SDK section
  above; or use Inference's `Precropped PNG` mode and skip pyphantom entirely.
- **`pyphantom` imports but `ImportError: DLL load failed` at runtime.** The
  Phantom SDK's native DLLs aren't on PATH. See "Phantom SDK Step 2" above —
  add the SDK's `Bin\Win64` directory with `setx`, then close and reopen the
  terminal.
- **`pyfirmata` errors with `inspect.getargspec`.** Known issue on Python
  3.11+; the lab_capture scripts patch this automatically — no manual fix
  needed. If you see it elsewhere, you've imported pyfirmata outside one of
  the patched scripts.
- **`cropping-infer` (or the other console scripts) not found.** You ran
  `pip install -r requirements.txt` instead of an editable install. Run
  `pip install -e .` from the repo root, or use `python -m Inference.inference_gui`
  instead.

## Layout summary

| File | Purpose |
|---|---|
| [pyproject.toml](pyproject.toml) | Source of truth for deps + console scripts |
| [requirements.txt](requirements.txt) | Flat mirror for `pip install -r` users |
| [install.py](install.py) | One-shot installer (creates venv, picks CUDA, etc.) |
| [Calibration/lab_capture/requirements.txt](Calibration/lab_capture/requirements.txt) | Hardware-only deps (pythonnet, pyfirmata) |
| [Calibration/lab_capture/setup_lab_env.bat](Calibration/lab_capture/setup_lab_env.bat) | Auto-creates the dedicated lab venv on Windows |
