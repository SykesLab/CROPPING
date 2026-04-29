#!/usr/bin/env python3
"""
CROPPING Pipeline — Environment Installer

Cross-platform installer. Creates a virtual environment, installs all
dependencies, verifies imports, and prints launch instructions.

Usage:
    python install.py                # Full setup with CPU PyTorch
    python install.py --cuda 12.1    # Full setup with CUDA 12.1 PyTorch
    python install.py --check        # Check existing environment only
    python install.py --install      # Install missing packages into current env
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
VENV_NAME = "venv"
VENV_DIR = REPO_ROOT / VENV_NAME

# Python 3.11 is REQUIRED, not optional. The pyphantom wheel is built
# `py311-none-any` and won't install on any other version. Other deps work
# on more versions but this one constrains the whole environment.
REQUIRED_PYTHON = (3, 11)

# All packages the pipeline needs, mapped to their import name.
# Pin rationale lives in pyproject.toml — keep this list in sync with it.
# torch + torchvision are installed separately in Step 3 from the CUDA-specific
# index URL — DO NOT add them here, or step 4's PyPI install will overwrite
# the CUDA build with the CPU build.
CORE_PACKAGES: Dict[str, str] = {
    "numpy>=1.21.0,<2.0": "numpy",
    "scipy>=1.7.0,<1.13": "scipy",
    "opencv-python>=4.5.0,<5": "cv2",
    "Pillow>=8.0.0,<11": "PIL",
    "PyYAML>=5.4.0,<7": "yaml",
    "matplotlib>=3.4.0,<4": "matplotlib",
    "pandas>=1.3.0,<3": "pandas",
    "tqdm>=4.60.0,<5": "tqdm",
}

ML_PACKAGES: Dict[str, str] = {
    "tensorboard>=2.10.0,<3": "tensorboard",
    # torch + torchvision intentionally absent — installed in Step 3 from CUDA index.
}

DEV_PACKAGES: Dict[str, str] = {
    "pytest>=7.0.0,<9": "pytest",
    "pytest-timeout>=2.0.0,<3": "pytest_timeout",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "pyphantom": "pyphantom",
}

# Modules each GUI imports at startup (used to verify install completeness).
# Each tuple is (gui_module, list_of_required_imports_to_test).
GUI_VERIFICATION: List[Tuple[str, List[str]]] = [
    ("Preprocessing", ["tkinter", "PIL.ImageTk", "yaml", "cv2", "numpy", "pandas",
                        "scipy", "matplotlib", "tqdm"]),
    ("Calibration",   ["tkinter", "PIL.ImageTk", "yaml", "cv2", "numpy", "pandas",
                        "scipy", "matplotlib"]),
    ("Training",      ["tkinter", "torch", "torchvision", "yaml", "cv2", "numpy",
                        "pandas", "matplotlib", "tqdm", "tensorboard"]),
    ("Inference",     ["tkinter", "PIL.ImageTk", "torch", "yaml", "cv2", "numpy",
                        "matplotlib"]),
]

CUDA_TORCH_URLS = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.6": "https://download.pytorch.org/whl/cu126",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_step(n: int, total: int, text: str):
    print(f"\n[{n}/{total}] {text}")


def run(cmd: List[str], desc: str = "", **kwargs) -> bool:
    """Run a command, return True on success."""
    if desc:
        print(f"  {desc}...")
    try:
        subprocess.check_call(cmd, **kwargs)
        return True
    except subprocess.CalledProcessError:
        return False


def get_python() -> str:
    """Get the path to the Python executable (venv-aware)."""
    if VENV_DIR.exists():
        if platform.system() == "Windows":
            return str(VENV_DIR / "Scripts" / "python.exe")
        return str(VENV_DIR / "bin" / "python")
    return sys.executable


def find_python_3_11() -> Optional[str]:
    """Locate a Python 3.11 interpreter on the system.

    Tries (in order):
      1. The currently-running interpreter, if it's 3.11
      2. Windows: `py -3.11` launcher
      3. Common command names (`python3.11`, `python3.11.exe`)

    Returns the path to the executable, or None if no 3.11 is found.
    """
    # 1. Current interpreter
    if sys.version_info[:2] == REQUIRED_PYTHON:
        return sys.executable

    # 2. Windows py launcher
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["py", "-3.11", "-c",
                 "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                exe = result.stdout.strip()
                if exe and Path(exe).is_file():
                    return exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 3. Common command names
    for cmd in ("python3.11", "python3.11.exe"):
        try:
            result = subprocess.run(
                [cmd, "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                exe = result.stdout.strip()
                if exe and Path(exe).is_file():
                    return exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def check_package(import_name: str) -> Tuple[bool, str]:
    """Check if a package can be imported."""
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "ok")
        return True, version
    except ImportError:
        return False, "not installed"


# ── Check ─────────────────────────────────────────────────────────────────

def check_environment() -> Tuple[List[str], List[str], List[str]]:
    """Check all packages. Returns (installed, missing_required, missing_optional)."""
    installed = []
    missing = []
    optional_missing = []

    print("\n  Core packages:")
    for spec, imp in CORE_PACKAGES.items():
        ok, ver = check_package(imp)
        name = spec.split(">=")[0].split("<")[0]
        if ok:
            print(f"    {name:<20s} {ver}")
            installed.append(name)
        else:
            print(f"    {name:<20s} MISSING")
            missing.append(spec)

    print("\n  ML packages:")
    # torch is special — check separately
    ok, ver = check_package("torch")
    if ok:
        import torch
        cuda_str = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU"
        print(f"    {'torch':<20s} {ver} ({cuda_str})")
        installed.append("torch")
    else:
        print(f"    {'torch':<20s} MISSING")
        missing.append("torch>=1.9.0")

    for spec, imp in ML_PACKAGES.items():
        ok, ver = check_package(imp)
        name = spec.split(">=")[0]
        if ok:
            print(f"    {name:<20s} {ver}")
            installed.append(name)
        else:
            print(f"    {name:<20s} MISSING")
            missing.append(spec)

    print("\n  Optional:")
    for spec, imp in OPTIONAL_PACKAGES.items():
        ok, ver = check_package(imp)
        if ok:
            print(f"    {spec:<20s} {ver}")
            installed.append(spec)
        else:
            print(f"    {spec:<20s} not installed (needed for .cine files)")
            optional_missing.append(spec)

    return installed, missing, optional_missing


def check_only():
    """Run environment check and print summary."""
    print_header("CROPPING Pipeline — Environment Check")

    ver = sys.version_info
    print(f"\n  Python: {ver.major}.{ver.minor}.{ver.micro}")
    if ver[:2] != REQUIRED_PYTHON:
        print(f"  ERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} required "
              f"(found {ver.major}.{ver.minor}).")
        print(f"  The pyphantom wheel is built py311-none-any and won't install on any other version.")
        print(f"  Install Python 3.11 from https://www.python.org/downloads/release/python-3119/")
        return False

    venv_active = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"  Virtual env: {'active' if venv_active else 'not active'}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    installed, missing, optional = check_environment()

    print(f"\n{'=' * 60}")
    if not missing:
        print(f"  READY — {len(installed)} packages installed")
        if optional:
            print(f"  Note: {', '.join(optional)} not installed (optional)")
        return True
    else:
        print(f"  NOT READY — missing: {', '.join(m.split('>=')[0] for m in missing)}")
        print(f"  Run: python install.py")
        return False


# ── Pyphantom local-wheel auto-detect ─────────────────────────────────────

def _find_local_pyphantom_wheel() -> Optional[Path]:
    """Look for a `pyphantom-*.whl` file in the script dir or ./wheels/.

    Returns the path to the first wheel found, or None.
    """
    search_roots = [REPO_ROOT, REPO_ROOT / "wheels"]
    for root in search_roots:
        if not root.is_dir():
            continue
        for whl in sorted(root.glob("pyphantom-*.whl")):
            return whl
    return None


def _install_pyphantom_from_local_wheel(pip: str) -> None:
    """If a local pyphantom wheel exists, install it. Otherwise print instructions."""
    # Already installed?
    result = subprocess.run(
        [pip, "-c", "import pyphantom"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  pyphantom already installed — skipping.")
        return

    # Local wheel?
    wheel = _find_local_pyphantom_wheel()
    if wheel is not None:
        print(f"  Found local wheel: {wheel.name}")
        if run([pip, "-m", "pip", "install", str(wheel)],
               desc="installing pyphantom from local wheel"):
            print("  pyphantom installed.")
            return
        print("  WARNING: local wheel install failed. See error above.")
        return

    # Neither installed nor wheel present — instruct the user
    print("  pyphantom (Phantom SDK) is NOT installed and no local wheel was found.")
    print("  This is optional — only needed for .cine file support.")
    print()
    print("  To install:")
    print(f"    1. Obtain `pyphantom-*.whl` from your institution's Phantom SDK distribution.")
    print(f"    2. Place the wheel in {REPO_ROOT} (or {REPO_ROOT / 'wheels'}/).")
    print(f"    3. Re-run `python install.py` — it will detect and install automatically.")
    print()
    print(f"  Or install manually:")
    print(f"    {pip} -m pip install path/to/pyphantom-*.whl")
    print()
    print(f"  After installing, you'll also need the SDK's DLL directory on PATH.")
    print(f"  See INSTALL.md \"Phantom SDK\" section for the setx command.")


# ── Install verification ──────────────────────────────────────────────────

def _verify_installation(pip: str) -> None:
    """Verify every dep imports AND every GUI's import chain is loadable.

    Reports per-GUI what's missing, not just an aggregate failure.
    """
    # First: aggregate import smoke test
    verify_cmd = (
        "import torch, torchvision, cv2, numpy, scipy, pandas, "
        "matplotlib, yaml, tqdm, PIL, tensorboard; "
        "print(f'  torch       {torch.__version__} (CUDA: {torch.cuda.is_available()})'); "
        "print(f'  torchvision {torchvision.__version__}'); "
        "print(f'  numpy       {numpy.__version__}'); "
        "print(f'  opencv      {cv2.__version__}'); "
        "print(f'  matplotlib  {matplotlib.__version__}'); "
        "print('  Aggregate import: OK')"
    )
    if not run([pip, "-c", verify_cmd]):
        print("  WARNING: aggregate import failed — see error above.")

    # Per-GUI import check: each GUI must be loadable.
    print()
    print("  Per-GUI verification:")
    all_passed = True
    for gui_name, required in GUI_VERIFICATION:
        check_lines = [f"import {imp}" for imp in required]
        check_script = "; ".join(check_lines)
        result = subprocess.run(
            [pip, "-c", check_script],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"    {gui_name:14s} OK")
        else:
            all_passed = False
            err = (result.stderr or "").strip().splitlines()
            err_short = err[-1] if err else "unknown error"
            print(f"    {gui_name:14s} FAILED — {err_short}")

    if all_passed:
        print()
        print("  All 4 GUIs are loadable. Pipeline is ready.")
    else:
        print()
        print("  WARNING: at least one GUI cannot import. The pipeline is not fully ready.")
        print("  Re-run `python install.py --check` after fixing the missing import.")


# ── Install ───────────────────────────────────────────────────────────────

def full_setup(cuda_version: str = "cpu"):
    """Full environment setup."""
    print_header("CROPPING Pipeline — Setup")
    total_steps = 6

    # Step 1: Find a Python 3.11 interpreter (REQUIRED — wheel constraint)
    print_step(1, total_steps, "Locating Python 3.11")
    py311 = find_python_3_11()
    if py311 is None:
        print(f"  ERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} not found on this system.")
        print(f"  This pipeline requires Python 3.11 specifically — the pyphantom wheel")
        print(f"  is built `py311-none-any` and won't install on 3.10, 3.12, 3.13, or 3.14.")
        print(f"")
        print(f"  Install Python 3.11.9 from:")
        print(f"    https://www.python.org/downloads/release/python-3119/")
        print(f"")
        print(f"  After installing, re-run this script. On Windows the `py -3.11` launcher")
        print(f"  will pick it up automatically; you don't need to change PATH or remove other")
        print(f"  Python versions.")
        sys.exit(1)
    print(f"  Found: {py311}")
    if py311 != sys.executable:
        print(f"  (currently running on {sys.executable} — venv will be built from the 3.11 above)")

    # Step 2: Virtual environment built from the 3.11 interpreter
    print_step(2, total_steps, "Virtual environment")
    if VENV_DIR.exists():
        print(f"  {VENV_NAME}/ already exists — reusing")
        # Sanity-check the existing venv is 3.11
        try:
            result = subprocess.run(
                [get_python(), "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True, text=True, timeout=10,
            )
            if "(3, 11)" not in (result.stdout or ""):
                print(f"  WARNING: existing venv is not Python 3.11. Delete {VENV_DIR}/ and re-run.")
                sys.exit(1)
        except Exception:
            pass
    else:
        print(f"  Creating {VENV_NAME}/ from {py311}...")
        if not run([py311, "-m", "venv", str(VENV_DIR)]):
            print("  ERROR: Failed to create virtual environment")
            sys.exit(1)
        print(f"  Created: {VENV_DIR}")

    pip = get_python()

    # Step 3: PyTorch (pinned to the supported torch 2.x generation)
    print_step(
        3, total_steps,
        f"Installing PyTorch ({'CUDA ' + cuda_version if cuda_version != 'cpu' else 'CPU'})")
    torch_url = CUDA_TORCH_URLS.get(cuda_version, CUDA_TORCH_URLS["cpu"])
    if not run([pip, "-m", "pip", "install",
                "torch>=2.0.0,<3", "torchvision>=0.15.0,<1",
                "--index-url", torch_url],
               desc=f"torch from {torch_url}"):
        print("  WARNING: PyTorch install failed. Try manually.")

    # Step 4: All other packages
    print_step(4, total_steps, "Installing dependencies")

    all_specs = list(CORE_PACKAGES.keys()) + list(ML_PACKAGES.keys()) + list(DEV_PACKAGES.keys())
    if not run([pip, "-m", "pip", "install"] + all_specs,
               desc="Core + ML + dev packages"):
        print("  WARNING: Some packages may have failed")

    # Editable install of the project itself
    run([pip, "-m", "pip", "install", "-e", str(REPO_ROOT)],
        desc="Editable install of CROPPING")

    # Step 5: Auto-install pyphantom if a local wheel is present
    print_step(5, total_steps, "Pyphantom (Phantom SDK, optional)")
    _install_pyphantom_from_local_wheel(pip)

    # Step 6: Verify the install — every dep imports + every GUI is loadable
    print_step(6, total_steps, "Verifying installation")
    _verify_installation(pip)

    # Done
    print_header("Setup Complete")

    if platform.system() == "Windows":
        activate = f"{VENV_NAME}\\Scripts\\activate"
    else:
        activate = f"source {VENV_NAME}/bin/activate"

    print(f"""
  Activate the environment:
    {activate}

  Launch the pipeline (module form):
    python -m Preprocessing        # Preprocessing GUI
    python -m Calibration          # Calibration GUI
    python -m Training             # Training GUI
    python -m Inference            # Inference GUI

  Or via console scripts (after editable install above):
    cropping-preprocess
    cropping-calibrate
    cropping-train
    cropping-infer

  Run automated tests:
    python -m pytest Extras/tests/ -v

  Lab-capture hardware deps (only needed for Step 0 — calibration capture):
    pip install -r Calibration/lab_capture/requirements.txt
""")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CROPPING Pipeline — Environment Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                 Full setup with CPU PyTorch
  python install.py --cuda 12.1     Full setup with CUDA 12.1
  python install.py --check         Check existing environment
  python install.py --install       Install missing packages only
        """,
    )
    parser.add_argument("--check", action="store_true",
                        help="Check environment without installing")
    parser.add_argument("--install", action="store_true",
                        help="Install missing packages into current environment")
    parser.add_argument("--cuda", type=str, default="cpu",
                        choices=list(CUDA_TORCH_URLS.keys()),
                        help="CUDA version for PyTorch (default: cpu)")

    args = parser.parse_args()

    if args.check:
        ok = check_only()
        sys.exit(0 if ok else 1)

    if args.install:
        # In-place install into the CURRENT interpreter — must already be 3.11.
        ver = sys.version_info
        if ver[:2] != REQUIRED_PYTHON:
            print(f"ERROR: --install needs to run on Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} "
                  f"(current: {ver.major}.{ver.minor}).")
            print("Either re-run with the right Python (e.g. `py -3.11 install.py --install`)")
            print("or run plain `python install.py` to build a fresh 3.11 venv.")
            sys.exit(1)
        pip = sys.executable
        all_specs = list(CORE_PACKAGES.keys()) + list(ML_PACKAGES.keys()) + list(DEV_PACKAGES.keys())
        print("Installing missing packages...")
        run([pip, "-m", "pip", "install"] + all_specs)
        _install_pyphantom_from_local_wheel(pip)
        _verify_installation(pip)
        sys.exit(0)

    full_setup(cuda_version=args.cuda)


if __name__ == "__main__":
    main()
