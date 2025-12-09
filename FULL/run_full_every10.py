# run_full_every10.py

import sys
from pathlib import Path

# SDK import test
try:
    from pyphantom import cine
except Exception as e:
    print("\n[SDK ERROR] Could not import pyphantom.cine")
    print("Ensure Phantom SDK and pyphantom are installed correctly.")
    print("Details:", e)
    sys.exit(1)


def check_sdk_and_cines(root_folder):
    root = Path(root_folder)
    cine_files = list(root.rglob("*.cine"))

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


from cine_iterator_full_every10_global import process_every_10_global
from cine_iterator_full_every10_perfolder import process_every_10_perfolder


root = r"F:\spheres"

print("\n====================================")
print(" CHOOSE CROPPING MODE")
print("====================================")
print("1 = Global crop size (all folders)")
print("2 = Per-folder crop size")
print("====================================\n")

choice = input("Enter 1 or 2: ").strip()

if not check_sdk_and_cines(root):
    print("[ABORT] SDK or cine load failure.")
    sys.exit(1)

if choice == "1":
    print("\n[MODE] Global crop mode\n")
    process_every_10_global(root)
elif choice == "2":
    print("\n[MODE] Per-folder crop mode\n")
    process_every_10_perfolder(root)
else:
    print("Invalid choice.")
    sys.exit(1)

print("\n=== DONE ===\n")
