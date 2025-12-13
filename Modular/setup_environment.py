"""Environment setup and verification for Droplet Preprocessing Pipeline.

Run this script to set up and verify your environment:
    python setup_environment.py

Or import and call from main_runner:
    from setup_environment import verify_environment
    verify_environment()
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Minimum Python version
MIN_PYTHON = (3, 8)

# Required packages and their import names
REQUIRED_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "opencv-python": "cv2",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "customtkinter": "customtkinter",
}

# Optional packages
OPTIONAL_PACKAGES: Dict[str, str] = {
    "pyphantom": "pyphantom",  # Phantom SDK for .cine files
}


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets minimum requirements.
    
    Returns:
        Tuple of (success, message).
    """
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        return True, f"✓ Python {current[0]}.{current[1]} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required)"
    else:
        return False, f"✗ Python {current[0]}.{current[1]} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"


def check_package(import_name: str) -> Tuple[bool, str]:
    """Check if a package can be imported.
    
    Args:
        import_name: The name used to import the package.
        
    Returns:
        Tuple of (success, version_or_error).
    """
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_all_packages() -> Tuple[List[str], List[str], List[str]]:
    """Check all required and optional packages.
    
    Returns:
        Tuple of (installed, missing_required, missing_optional).
    """
    installed = []
    missing_required = []
    missing_optional = []
    
    print("\n--- Checking Required Packages ---")
    for pkg_name, import_name in REQUIRED_PACKAGES.items():
        success, info = check_package(import_name)
        if success:
            print(f"  ✓ {pkg_name} ({info})")
            installed.append(pkg_name)
        else:
            print(f"  ✗ {pkg_name} - NOT FOUND")
            missing_required.append(pkg_name)
    
    print("\n--- Checking Optional Packages ---")
    for pkg_name, import_name in OPTIONAL_PACKAGES.items():
        success, info = check_package(import_name)
        if success:
            print(f"  ✓ {pkg_name} ({info})")
            installed.append(pkg_name)
        else:
            print(f"  ⚠ {pkg_name} - NOT FOUND (optional)")
            missing_optional.append(pkg_name)
    
    return installed, missing_required, missing_optional


def install_requirements(requirements_path: Path = None) -> bool:
    """Install packages from requirements.txt.
    
    Args:
        requirements_path: Path to requirements.txt (default: same directory).
        
    Returns:
        True if successful.
    """
    if requirements_path is None:
        requirements_path = Path(__file__).parent / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"✗ requirements.txt not found at {requirements_path}")
        return False
    
    print(f"\nInstalling from {requirements_path}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False


def check_phantom_sdk() -> Tuple[bool, str]:
    """Check if Phantom SDK (pyphantom) is installed and working.
    
    Returns:
        Tuple of (success, message).
    """
    try:
        import pyphantom
        # Try to access the Cine class
        if hasattr(pyphantom, "Cine"):
            return True, "pyphantom installed and Cine class available"
        else:
            return False, "pyphantom installed but Cine class not found"
    except ImportError:
        return False, "pyphantom not installed"
    except Exception as e:
        return False, f"pyphantom error: {e}"


def print_phantom_instructions():
    """Print instructions for installing Phantom SDK."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    PHANTOM SDK INSTALLATION                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  The Phantom SDK (pyphantom) is required to read .cine files.    ║
║                                                                   ║
║  Installation steps:                                              ║
║                                                                   ║
║  1. Download the Phantom SDK from Vision Research:                ║
║     https://www.phantomhighspeed.com/resourcesandsupport/         ║
║                                                                   ║
║  2. Install the SDK following their instructions                  ║
║                                                                   ║
║  3. The Python bindings (pyphantom) should be available after    ║
║     installation. If not, check the SDK documentation.           ║
║                                                                   ║
║  Alternative: If you have .cine files but no SDK access,         ║
║  contact your lab's Phantom camera administrator.                ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def create_directories() -> None:
    """Create necessary output directories."""
    try:
        from config_modular import OUTPUT_ROOT, CINE_ROOT
        
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output directory: {OUTPUT_ROOT}")
        
        if not CINE_ROOT.exists():
            print(f"  ⚠ CINE_ROOT does not exist: {CINE_ROOT}")
            print(f"    Update config_modular.py with your data path")
        else:
            print(f"  ✓ CINE_ROOT exists: {CINE_ROOT}")
            
    except ImportError:
        print("  ⚠ Could not import config_modular - skipping directory check")


def verify_environment(auto_install: bool = False) -> bool:
    """Verify the environment is set up correctly.
    
    Args:
        auto_install: If True, automatically install missing packages.
        
    Returns:
        True if environment is ready.
    """
    print("=" * 60)
    print("DROPLET PREPROCESSING PIPELINE - ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check Python version
    py_ok, py_msg = check_python_version()
    print(f"\n{py_msg}")
    
    if not py_ok:
        print("\n✗ Python version too old. Please upgrade Python.")
        return False
    
    # Check packages
    installed, missing_required, missing_optional = check_all_packages()
    
    # Handle missing required packages
    if missing_required:
        print(f"\n⚠ Missing required packages: {', '.join(missing_required)}")
        
        if auto_install:
            print("\nAttempting automatic installation...")
            if install_requirements():
                # Re-check after installation
                _, still_missing, _ = check_all_packages()
                if still_missing:
                    print(f"\n✗ Still missing: {', '.join(still_missing)}")
                    return False
            else:
                return False
        else:
            print("\nTo install missing packages, run:")
            print(f"  pip install -r requirements.txt")
            print("\nOr run this script with --install flag:")
            print(f"  python {Path(__file__).name} --install")
            return False
    
    # Check Phantom SDK
    print("\n--- Checking Phantom SDK ---")
    phantom_ok, phantom_msg = check_phantom_sdk()
    if phantom_ok:
        print(f"  ✓ {phantom_msg}")
    else:
        print(f"  ⚠ {phantom_msg}")
        print_phantom_instructions()
    
    # Check directories
    print("\n--- Checking Directories ---")
    create_directories()
    
    # Summary
    print("\n" + "=" * 60)
    if missing_required:
        print("STATUS: ✗ Environment NOT ready")
        return False
    elif not phantom_ok:
        print("STATUS: ⚠ Environment ready (but Phantom SDK missing)")
        print("        You can process existing crops but not read .cine files")
        return True  # Partial success
    else:
        print("STATUS: ✓ Environment ready!")
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up environment for Droplet Preprocessing Pipeline"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Automatically install missing packages",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check, don't prompt for installation",
    )
    
    args = parser.parse_args()
    
    success = verify_environment(auto_install=args.install)
    
    if not success and not args.check_only:
        print("\nWould you like to install missing packages now? [y/N] ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                install_requirements()
                verify_environment()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipped.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
