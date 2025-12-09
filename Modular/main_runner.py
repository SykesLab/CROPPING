# main_runner.py
#
# Entry point for the modular cropping pipeline.
# Lets you choose: global vs per-folder crop, execution mode, and output options.

import sys
from pathlib import Path

from config_modular import CINE_ROOT
from pipeline_global import process_global
from pipeline_folder import process_per_folder

# Use SILENT cine import (pre-loads SDK once, suppresses banner)
from phantom_silence_modular import cine


def check_sdk_and_cines(root_folder: Path) -> bool:
    """Verify SDK works and .cine files exist."""
    cine_files = list(root_folder.rglob("*.cine"))

    if not cine_files:
        print(f"\n[SDK CHECK] No .cine files under {root_folder}")
        return False

    test_path = cine_files[0]
    print(f"\n[SDK CHECK] Testing SDK on: {test_path}")

    try:
        c = cine.Cine.from_filepath(str(test_path))
        if c is None:
            print("[SDK CHECK] Cine returned None.")
            return False

        _ = c.range
        print("[SDK CHECK] SUCCESS — Cine loading works.\n")
        return True

    except Exception as e:
        print("[SDK CHECK] FAILED — Cine could not be opened.")
        print("Details:", e)
        return False


def _ask_crop_mode() -> str:
    """Ask user for global vs per-folder crop mode."""
    print("\n" + "=" * 40)
    print(" MODULAR CROPPING PIPELINE")
    print("=" * 40)
    print(f"CINE ROOT: {CINE_ROOT}")
    print("=" * 40)
    print("1 = Global crop size (all folders)")
    print("2 = Per-folder crop size")
    print("=" * 40 + "\n")

    return input("Enter 1 or 2: ").strip()


def _ask_execution_mode():
    """
    Ask for execution mode.
    
    Returns:
        (safe_mode, profile, quick_test)
    """
    print("\n" + "=" * 40)
    print(" EXECUTION MODE")
    print("=" * 40)
    print("1 = Fast (multiprocessing)")
    print("2 = Safe (single-process, for debugging)")
    print("3 = Safe + Profiling (full dataset)")
    print("4 = Quick Test (first droplet per folder)")
    print("=" * 40 + "\n")

    mode = input("Enter 1, 2, 3, or 4: ").strip()

    if mode == "1":
        return False, False, False  # safe_mode, profile, quick_test
    elif mode == "2":
        return True, False, False
    elif mode == "3":
        return True, True, False
    elif mode == "4":
        return True, True, True
    else:
        print("Invalid execution mode.")
        sys.exit(1)


def _ask_output_options() -> bool:
    """
    Ask user what outputs they want.
    
    Returns:
        full_output: If True, generate all plots. If False, crops only.
    """
    print("\n" + "=" * 40)
    print(" OUTPUT OPTIONS")
    print("=" * 40)
    print("1 = Crops only (fastest, minimal memory)")
    print("2 = Full output (crops + darkness + overlay plots)")
    print("=" * 40 + "\n")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        return False  # crops only
    elif choice == "2":
        return True   # full output
    else:
        print("Invalid choice, defaulting to crops only.")
        return False


def main():
    crop_choice = _ask_crop_mode()

    if not check_sdk_and_cines(CINE_ROOT):
        print("[ABORT] SDK or cine load failure.")
        sys.exit(1)

    safe_mode, profile, quick_test = _ask_execution_mode()
    
    # Ask for output options (applies to all modes)
    # Safe mode and quick test default to full output for debugging
    if safe_mode or quick_test:
        full_output = True
        print("\n[INFO] Safe/Quick mode → full output enabled by default.")
    else:
        full_output = _ask_output_options()

    # Print summary
    print("\n" + "=" * 40)
    print(" CONFIGURATION SUMMARY")
    print("=" * 40)
    print(f"  Crop mode:    {'Global' if crop_choice == '1' else 'Per-folder'}")
    print(f"  Execution:    {'Safe (single-process)' if safe_mode else 'Fast (multiprocessing)'}")
    print(f"  Profiling:    {'Yes' if profile else 'No'}")
    print(f"  Quick test:   {'Yes' if quick_test else 'No'}")
    print(f"  Full output:  {'Yes (crops + all plots)' if full_output else 'No (crops only)'}")
    print("=" * 40)

    if crop_choice == "1":
        print("\n[MODE] Global crop mode\n")
        process_global(
            safe_mode=safe_mode,
            profile=profile,
            quick_test=quick_test,
            full_output=full_output,
        )
    elif crop_choice == "2":
        print("\n[MODE] Per-folder crop mode\n")
        process_per_folder(
            safe_mode=safe_mode,
            profile=profile,
            quick_test=quick_test,
            full_output=full_output,
        )
    else:
        print("Invalid crop mode choice.")
        sys.exit(1)

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
