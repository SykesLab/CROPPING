"""
CINE file I/O and grouping utilities.

Handles loading .cine files via the Phantom SDK and grouping them
by droplet ID based on filename patterns (e.g. sphere0843g.cine).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phantom_silence_modular import cine


def safe_load_cine(path: Path) -> Optional[Any]:
    """Load a .cine file, returning None on failure."""
    try:
        cine_obj = cine.Cine.from_filepath(str(path))
        if cine_obj is None:
            return None
        _ = cine_obj.range  # Force handle init
        return cine_obj
    except Exception as e:
        print(f"[CINE LOAD ERROR] {path.name}: {e}")
        return None


def iter_subfolders(root: Path) -> List[Path]:
    """Get sorted list of subdirectories."""
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def get_cine_folders(root: Path) -> List[Path]:
    """
    Get folders containing .cine files.

    If root has subfolders with cines, returns those.
    Otherwise if root itself has cines, returns [root].
    """
    root = Path(root)

    subfolders = iter_subfolders(root)
    if subfolders:
        cine_folders = [sf for sf in subfolders if list(sf.glob("*.cine"))]
        if cine_folders:
            return cine_folders

    if list(root.glob("*.cine")):
        return [root]

    return []


def group_cines_by_droplet(folder: Path) -> List[Tuple[str, Dict[str, Optional[Path]]]]:
    """
    Group .cine files by droplet ID and camera.

    Filename pattern: sphere0843g.cine -> droplet_id=0843, camera=g
    Returns list of (droplet_id, {"g": path, "v": path}) sorted by ID.
    """
    folder = Path(folder)
    groups: Dict[str, Dict[str, Optional[Path]]] = {}

    for path in sorted(folder.glob("*.cine")):
        match = re.search(r"(\d+)([gv])$", path.stem, re.IGNORECASE)
        if not match:
            continue

        droplet_id = match.group(1)
        cam = match.group(2).lower()

        if droplet_id not in groups:
            groups[droplet_id] = {"g": None, "v": None}
        groups[droplet_id][cam] = path

    sorted_ids = sorted(groups.keys(), key=lambda x: int(x))
    return [(did, groups[did]) for did in sorted_ids]
