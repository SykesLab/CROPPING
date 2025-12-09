# main_modular_runner.py

import sys
from pathlib import Path

from config_modular import CINE_ROOT
from pipeline_global_every10_parallel import process_global_every_10_parallel
from pipeline_folder_every10_parallel import process_per_folder_every_10_parallel

# Use SILENT cine import
from phantom_silence_modular import cine


def check_sdk_and_cines(root_folder: Path):
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


def main():
    print("\n====================================")
    print(" MODULAR CROPPING PIPELINE")
    print("====================================")
    print(f"CINE ROOT: {CINE_ROOT}")
    print("====================================")
    print("1 = Global crop size (all folders)")
    print("2 = Per-folder crop size")
    print("====================================\n")

    choice = input("Enter 1 or 2: ").strip()

    if not check_sdk_and_cines(CINE_ROOT):
        print("[ABORT] SDK or cine load failure.")
        sys.exit(1)

    if choice == "1":
        print("\n[MODE] Global crop mode (Parallel)\n")
        process_global_every_10_parallel()
    elif choice == "2":
        print("\n[MODE] Per-folder crop mode (Parallel)\n")
        process_per_folder_every_10_parallel()
    else:
        print("Invalid choice.")
        sys.exit(1)

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
