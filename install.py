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
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
VENV_NAME = "venv"
VENV_DIR = REPO_ROOT / VENV_NAME

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 14)

# All packages the pipeline needs, mapped to their import name
CORE_PACKAGES: Dict[str, str] = {
    "numpy>=1.21.0,<2.0": "numpy",
    "scipy>=1.7.0": "scipy",
    "opencv-python>=4.5.0": "cv2",
    "Pillow>=8.0.0": "PIL",
    "PyYAML>=5.4.0": "yaml",
    "matplotlib>=3.4.0": "matplotlib",
    "pandas>=1.3.0": "pandas",
    "tqdm>=4.60.0": "tqdm",
}

ML_PACKAGES: Dict[str, str] = {
    "torchvision>=0.10.0": "torchvision",
    "tensorboard>=2.5.0": "tensorboard",
}

DEV_PACKAGES: Dict[str, str] = {
    "pytest>=7.0.0": "pytest",
    "pytest-timeout>=2.0.0": "pytest_timeout",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "pyphantom": "pyphantom",
}

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
    if ver[:2] < MIN_PYTHON:
        print(f"  ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
        return False
    if ver[:2] > MAX_PYTHON:
        print(
            f"  WARNING: Python {ver.major}.{ver.minor} is newer than tested ({MAX_PYTHON[0]}.{MAX_PYTHON[1]})")

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


# ── Install ───────────────────────────────────────────────────────────────

def full_setup(cuda_version: str = "cpu"):
    """Full environment setup."""
    print_header("CROPPING Pipeline — Setup")
    total_steps = 5

    # Step 1: Python version
    print_step(1, total_steps, "Checking Python")
    ver = sys.version_info
    print(f"  Python {ver.major}.{ver.minor}.{ver.micro}")
    if ver[:2] < MIN_PYTHON:
        print(f"  ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.")
        print(f"  Download from: https://www.python.org/downloads/")
        sys.exit(1)

    # Step 2: Virtual environment
    print_step(2, total_steps, "Virtual environment")
    if VENV_DIR.exists():
        print(f"  {VENV_NAME}/ already exists — reusing")
    else:
        print(f"  Creating {VENV_NAME}/...")
        if not run([sys.executable, "-m", "venv", str(VENV_DIR)]):
            print("  ERROR: Failed to create virtual environment")
            sys.exit(1)
        print(f"  Created: {VENV_DIR}")

    pip = get_python()

    # Step 3: PyTorch
    print_step(
        3, total_steps,
        f"Installing PyTorch ({'CUDA ' + cuda_version if cuda_version != 'cpu' else 'CPU'})")
    torch_url = CUDA_TORCH_URLS.get(cuda_version, CUDA_TORCH_URLS["cpu"])
    if not run([pip, "-m", "pip", "install", "torch", "torchvision",
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

    # Step 5: Verify
    print_step(5, total_steps, "Verifying installation")
    verify_cmd = (
        "import torch, cv2, numpy, scipy, pandas, matplotlib, yaml, tqdm, PIL, tensorboard; "
        "print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})'); "
        "print(f'  numpy {numpy.__version__}'); "
        "print(f'  opencv {cv2.__version__}'); "
        "print('  All imports OK')"
    )
    if not run([pip, "-c", verify_cmd]):
        print("  WARNING: Some imports failed — check output above")

    # Check pyphantom
    result = subprocess.run(
        [pip, "-c", "import pyphantom; print('pyphantom available')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("\n  pyphantom (Phantom SDK) is NOT installed.")
        print("  This is optional — only needed to read .cine files directly.")
        print("  To install: place the .whl file in this directory and run:")
        print(f"    {pip} -m pip install pyphantom-*.whl")

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
        pip = sys.executable
        all_specs = list(CORE_PACKAGES.keys()) + list(ML_PACKAGES.keys())
        print("Installing missing packages...")
        run([pip, "-m", "pip", "install"] + all_specs)
        check_only()
        sys.exit(0)

    full_setup(cuda_version=args.cuda)


if __name__ == "__main__":
    main()
