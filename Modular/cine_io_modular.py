"""Cine file I/O and grouping utilities.

Handles safe loading of .cine files and grouping by droplet ID.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phantom_silence_modular import cine


def safe_load_cine(path: Path) -> Optional[Any]:
    """Safely load a .cine file.

    Args:
        path: Path to the .cine file.

    Returns:
        Loaded cine object, or None if loading failed.
    """
    try:
        cine_obj = cine.Cine.from_filepath(str(path))
        if cine_obj is None:
            return None

        # Access range to force handle initialisation
        _ = cine_obj.range
        return cine_obj

    except Exception as e:
        print(f"[CINE LOAD ERROR] {path.name}: {e}")
        return None


def iter_subfolders(root: Path) -> List[Path]:
    """Get sorted list of subdirectories.

    Args:
        root: Root directory to scan.

    Returns:
        Sorted list of subdirectory paths.
    """
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def group_cines_by_droplet(
    folder: Path,
) -> List[Tuple[str, Dict[str, Optional[Path]]]]:
    """Group .cine files by droplet ID and camera.

    Expected filename pattern: sphere0843g.cine -> ID=0843, cam=g

    Args:
        folder: Folder containing .cine files.

    Returns:
        List of (droplet_id, {"g": path_or_None, "v": path_or_None}) tuples,
        sorted numerically by droplet ID.
    """
    folder = Path(folder)
    groups: Dict[str, Dict[str, Optional[Path]]] = {}

    for path in sorted(folder.glob("*.cine")):
        stem = path.stem
        match = re.search(r"(\d+)([gv])$", stem, re.IGNORECASE)
        if not match:
            continue

        droplet_id = match.group(1)
        cam = match.group(2).lower()

        if droplet_id not in groups:
            groups[droplet_id] = {"g": None, "v": None}

        groups[droplet_id][cam] = path

    # Sort numerically by droplet ID
    sorted_ids = sorted(groups.keys(), key=lambda x: int(x))
    return [(did, groups[did]) for did in sorted_ids]
