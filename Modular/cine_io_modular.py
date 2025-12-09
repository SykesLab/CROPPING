# cine_io_modular.py
#
# Safely loads .cine files and groups them by droplet ID.
# Fully suppresses Phantom SDK ("phpy: version...") banner output,
# including inside multiprocessing workers.

import os
import sys
import re
from pathlib import Path
from contextlib import contextmanager


# ============================================================
#  SILENCE stdout temporarily (used for pyphantom imports)
# ============================================================
@contextmanager
def suppress_stdout():
    """Temporarily silence all stdout."""
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = saved_stdout


# ============================================================
#  IMPORT pyphantom WITH STDOUT SUPPRESSED (critical fix)
# ============================================================

# Silence banner while importing pyphantom.cine
with suppress_stdout():
    from pyphantom import cine

# Silence banner while importing Phantom (camera init)
ph = None
try:
    with suppress_stdout():
        from pyphantom import Phantom
        ph = Phantom(init_camera=False)
except Exception:
    try:
        with suppress_stdout():
            ph = Phantom()
    except Exception:
        ph = None


# ============================================================
#  SAFE CINE LOADER
# ============================================================
def safe_load_cine(path: Path):
    """
    Safely load a .cine file using pyphantom.
    Returns a cine object or None on failure.
    """
    try:
        with suppress_stdout():  # ensure no Phantom noise appears
            c = cine.Cine.from_filepath(str(path))
        if c is None:
            return None

        # Accessing c.range forces the handle to initialize properly
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
