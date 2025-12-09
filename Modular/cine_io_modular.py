# cine_io_modular.py
#
# Safely loads .cine files and groups them by droplet ID.
# Uses the central silent Phantom importer (phantom_silence_modular.py),
# so no file in this project ever imports pyphantom directly.

import re
from pathlib import Path

from phantom_silence_modular import cine  # <-- centralised silent import


# ============================================================
#  SAFE CINE LOADER
# ============================================================
def safe_load_cine(path: Path):
    """
    Safely load a .cine file using pyphantom via the silent cine handle.
    Returns a cine object or None on failure.
    """
    try:
        c = cine.Cine.from_filepath(str(path))
        if c is None:
            return None

        # Accessing c.range forces the internal handle to initialize
        _ = c.range
        return c

    except Exception as e:
        print(f"[CINE LOAD ERROR] {path.name}: {e}")
        return None


# ============================================================
#  ITERATE SUBFOLDERS
# ============================================================
def iter_subfolders(root: Path):
    """
    Returns sorted list of subdirectories under the given root.
    """
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


# ============================================================
#  GROUP CINE FILES BY DROPLET ID (g/v pairs)
# ============================================================
def group_cines_by_droplet(folder: Path):
    """
    Group .cine files by droplet ID and camera ('g' or 'v').

    Expected file pattern: sphere0843g.cine → droplet ID = 0843, cam = g
    Returns: list of (droplet_id, {"g": path_or_None, "v": path_or_None})
    """
    folder = Path(folder)
    groups = {}

    for path in sorted(folder.glob("*.cine")):
        # Extract: numeric ID + camera letter
        # Example: sphere0843g → ID=0843, cam=g
        stem = path.stem
        m = re.search(r"(\d+)([gv])$", stem, re.IGNORECASE)
        if not m:
            continue

        droplet_id = m.group(1)
        cam = m.group(2).lower()

        if droplet_id not in groups:
            groups[droplet_id] = {"g": None, "v": None}

        groups[droplet_id][cam] = path

    # Sort droplet IDs numerically
    sorted_ids = sorted(groups.keys(), key=lambda x: int(x))
    return [(did, groups[did]) for did in sorted_ids]
