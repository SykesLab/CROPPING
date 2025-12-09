# main_modular_runner.py
#
# Entry point for the modular cropping pipeline.
# - Lets you choose: global vs per-folder crop.
# - Then lets you choose: fast (multiprocessing), safe, or safe+profiling.

import sys
from pathlib import Path

from config_modular import CINE_ROOT
from pipeline_global_every10_parallel import process_global_every_10_parallel
from pipeline_folder_every10_parallel import process_per_folder_every_10_parallel

# Use SILENT cine import (pre-loads SDK once, suppresses banner)
from phantom_silence_modular import cine


def check_sdk_and_cines(root_folder: Path) -> bool:
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

        _ = c.range  # force handle
        print("[SDK CHECK] SUCCESS — Cine loading works.\n")
        return True

    except Exception as e:
        print("[SDK CHECK] FAILED — Cine could not be opened.")
        print("Details:", e)
        return False


def _ask_crop_mode() -> str:
    print("\n====================================")
    print(" MODULAR CROPPING PIPELINE")
    print("====================================")
    print(f"CINE ROOT: {CINE_ROOT}")
    print("====================================")
    print("1 = Global crop size (all folders)")
    print("2 = Per-folder crop size")
    print("====================================\n")

    choice = input("Enter 1 or 2: ").strip()
    return choice


def _ask_execution_mode():
    """
    Ask for execution mode:

    1 = Fast (multiprocessing)
    2 = Safe (single-process)
    3 = Safe + Profiling (whole dataset)
    4 = QUICK TEST — first droplet per folder + profiling
    """

    print("\n====================================")
    print(" EXECUTION MODE")
    print("====================================")
    print("1 = Fast (multiprocessing)")
    print("2 = Safe (single-process)")
    print("3 = Safe + Profiling (full dataset)")
    print("4 = QUICK TEST (first droplet per folder + profiling)")
    print("====================================\n")

    mode = input("Enter 1, 2, 3, or 4: ").strip()

    if mode == "1":
        return False, False, False  # (safe_mode, profile, quick_test)

    elif mode == "2":
        return True, False, False

    elif mode == "3":
        return True, True, False

    elif mode == "4":
        return True, True, True   # QUICK TEST: safe + profiling + quick_test

    else:
        print("Invalid execution mode.")
        sys.exit(1)



def main():
    crop_choice = _ask_crop_mode()

    if not check_sdk_and_cines(CINE_ROOT):
        print("[ABORT] SDK or cine load failure.")
        sys.exit(1)

    safe_mode, profile, quick_test = _ask_execution_mode()


    if crop_choice == "1":
        print("\n[MODE] Global crop mode\n")
        process_global_every_10_parallel(safe_mode=safe_mode, profile=profile, quick_test=quick_test)
    elif crop_choice == "2":
        print("\n[MODE] Per-folder crop mode\n")
        process_per_folder_every_10_parallel(safe_mode=safe_mode, profile=profile, quick_test=quick_test)
    else:
        print("Invalid crop mode choice.")
        sys.exit(1)

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
