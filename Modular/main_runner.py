"""Entry point for the modular cropping pipeline.

Provides streamlined interactive configuration.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import config_modular
from config_modular import CINE_ROOT
from phantom_silence_modular import cine
from pipeline_folder import process_per_folder
from pipeline_global import process_global


def _check_sdk() -> bool:
    """Quick SDK validation."""
    cine_files = list(CINE_ROOT.rglob("*.cine"))
    if not cine_files:
        return False
    try:
        c = cine.Cine.from_filepath(str(cine_files[0]))
        return c is not None and c.range is not None
    except Exception:
        return False


def _count_data() -> Dict[str, int]:
    """Count folders and approximate droplets."""
    folders = [p for p in CINE_ROOT.iterdir() if p.is_dir()]
    total_cines = len(list(CINE_ROOT.rglob("*.cine")))
    return {
        "folders": len(folders),
        "cines": total_cines,
        "droplets": total_cines // 2,
    }


def _header() -> bool:
    """Print header and return SDK status."""
    print("\n" + "=" * 50)
    print("  DROPLET CROPPING PIPELINE")
    print("=" * 50)
    
    # Check SDK
    sdk_ok = _check_sdk()
    status = "Ready" if sdk_ok else "SDK Error"
    
    # Count data
    counts = _count_data()
    
    print(f"  Source:   {CINE_ROOT}")
    print(f"  Found:    {counts['folders']} folders, ~{counts['droplets']} droplets")
    print(f"  Status:   {status}")
    print("=" * 50)
    
    return sdk_ok


def _prompt_choice(question: str, options: Dict[str, str], default: str) -> str:
    """Prompt user to select from options.
    
    Args:
        question: Question text
        options: {key: description} mapping
        default: Default key (shown with *)
    
    Returns:
        Selected key (uppercase)
    """
    print(f"\n  {question}")
    for key, desc in options.items():
        marker = " *" if key == default else ""
        print(f"    [{key}] {desc}{marker}")
    
    choice = input(f"  [{default}] > ").strip().upper()
    
    if choice == "":
        return default
    if choice in options:
        return choice
    
    print(f"    Invalid, using: {default}")
    return default


def _prompt_int(question: str, default: int, min_val: int = 1) -> int:
    """Prompt for integer input."""
    choice = input(f"  {question} [{default}] > ").strip()
    
    if choice == "":
        return default
    try:
        val = int(choice)
        return max(min_val, val)
    except ValueError:
        print(f"    Invalid, using: {default}")
        return default


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    """Yes/no prompt."""
    hint = "[Y/n]" if default else "[y/N]"
    choice = input(f"  {question} {hint} > ").strip().lower()
    
    if choice == "":
        return default
    return choice in ("y", "yes")


def configure() -> Dict[str, Any]:
    """Run interactive configuration.
    
    Returns:
        Configuration dict with all settings.
    """
    config = {}
    
    # 1. Scope
    scope = _prompt_choice(
        "What to process?",
        {
            "F": "Full run (all selected droplets)",
            "Q": "Quick test (1 droplet per folder)",
        },
        default="F"
    )
    config["quick_test"] = (scope == "Q")
    
    # 2. Sampling (skip if quick test)
    if config["quick_test"]:
        config["step"] = 1
        print("\n  Sampling: N/A (quick test)")
    else:
        sampling = _prompt_choice(
            "Sampling rate?",
            {
                "A": "All droplets (step=1)",
                "T": "Every 10th (step=10)",
                "C": "Custom step",
            },
            default="T"
        )
        if sampling == "A":
            config["step"] = 1
        elif sampling == "T":
            config["step"] = 10
        else:
            config["step"] = _prompt_int("Enter step size:", default=10, min_val=1)
    
    # 3. Crop mode
    crop = _prompt_choice(
        "Crop calibration?",
        {
            "G": "Global (one size for all folders)",
            "P": "Per-folder (separate size each folder)",
        },
        default="G"
    )
    config["global_mode"] = (crop == "G")
    
    # 4. Execution
    execution = _prompt_choice(
        "Execution mode?",
        {
            "F": "Fast (multiprocessing)",
            "S": "Safe (single-process, for debugging)",
        },
        default="F"
    )
    config["safe_mode"] = (execution == "S")
    
    # 5. Outputs
    output = _prompt_choice(
        "Outputs to generate?",
        {
            "C": "Crops only (fastest)",
            "A": "All (crops + darkness + overlay plots)",
        },
        default="C"
    )
    config["full_output"] = (output == "A")
    
    # 6. Profiling (optional add-on)
    config["profile"] = _prompt_yes_no("Save profiling JSON?", default=False)
    
    return config


def _print_summary(config: Dict[str, Any]) -> None:
    """Print configuration summary."""
    counts = _count_data()
    
    # Calculate estimated droplets
    if config["quick_test"]:
        est_droplets = counts["folders"]
    else:
        est_droplets = counts["droplets"] // config["step"]
    
    print("\n" + "-" * 50)
    print("  CONFIGURATION")
    print("-" * 50)
    print(f"  Scope:       {'Quick test' if config['quick_test'] else 'Full run'}")
    print(f"  Sampling:    Every {config['step']} {'(N/A)' if config['quick_test'] else ''}")
    print(f"  Crop mode:   {'Global' if config['global_mode'] else 'Per-folder'}")
    print(f"  Execution:   {'Safe' if config['safe_mode'] else 'Fast'}")
    print(f"  Outputs:     {'All plots' if config['full_output'] else 'Crops only'}")
    print(f"  Profiling:   {'Yes' if config['profile'] else 'No'}")
    print("-" * 50)
    print(f"  Estimated:   ~{est_droplets} droplets to process")
    print("-" * 50)


def main() -> None:
    """Main entry point."""
    # Header + SDK check
    if not _header():
        print("\n[ERROR] SDK check failed. Cannot load .cine files.")
        sys.exit(1)
    
    # Interactive configuration
    config = configure()
    
    # Show summary
    _print_summary(config)
    
    # Confirm
    if not _prompt_yes_no("Proceed?", default=True):
        print("\n  Cancelled.\n")
        sys.exit(0)
    
    # Apply sampling rate to config module
    config_modular.CINE_STEP = config["step"]
    
    # Run pipeline
    print("\n" + "=" * 50)
    
    if config["global_mode"]:
        process_global(
            safe_mode=config["safe_mode"],
            profile=config["profile"],
            quick_test=config["quick_test"],
            full_output=config["full_output"],
        )
    else:
        process_per_folder(
            safe_mode=config["safe_mode"],
            profile=config["profile"],
            quick_test=config["quick_test"],
            full_output=config["full_output"],
        )
    
    print("\n=== COMPLETE ===\n")


if __name__ == "__main__":
    main()
