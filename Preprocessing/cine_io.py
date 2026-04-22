"""
CINE file I/O and grouping utilities.

Handles loading .cine files via the Phantom SDK and grouping them
by droplet ID based on filename patterns (e.g. sphere0843g.cine).

Includes silent pyphantom import to suppress SDK banner on worker spawn.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phantom_utils import init_pyphantom

logger = logging.getLogger(__name__)

# Load pyphantom components silently (optional)
cine, utils, ph, PYPHANTOM_AVAILABLE, PHANTOM_INITIALIZED = init_pyphantom()


# --- CINE loading and grouping ---

def safe_load_cine(path: Path) -> Optional[Any]:
    """Load a .cine file, returning None on failure."""
    if not PYPHANTOM_AVAILABLE or cine is None:
        logger.error(f"[CINE LOAD ERROR] {path.name}: pyphantom not available")
        return None

    try:
        cine_obj = cine.Cine.from_filepath(str(path))
        if cine_obj is None:
            return None
        _ = cine_obj.range  # Force handle init
        return cine_obj
    except Exception as e:
        logger.error(f"[CINE LOAD ERROR] {path.name}: {e}")
        return None


def get_camera_model(cine_obj: Any) -> Optional[str]:
    """
    Extract camera model name from a loaded .cine object.

    Tries various attribute names that pyphantom may expose for camera info.

    Args:
        cine_obj: A loaded pyphantom Cine object

    Returns:
        Camera model string if found, None otherwise
    """
    if cine_obj is None:
        return None

    # Try various attribute names that pyphantom may use
    attr_names = [
        'camera',
        'cameraModel',
        'camera_model',
        'CameraModel',
        'camera_name',
        'cameraName',
    ]

    for attr in attr_names:
        if hasattr(cine_obj, attr):
            val = getattr(cine_obj, attr)
            if val is not None and str(val).strip():
                return str(val).strip()

    # Try accessing via header/metadata dict if available
    for meta_attr in ['header', 'metadata', 'info', 'setup']:
        if hasattr(cine_obj, meta_attr):
            meta = getattr(cine_obj, meta_attr)
            if isinstance(meta, dict):
                for key in ['camera', 'Camera', 'cameraModel', 'CameraModel', 'camera_model']:
                    if key in meta and meta[key]:
                        return str(meta[key]).strip()

    return None


def get_camera_model_from_path(path: Path) -> Optional[str]:
    """
    Load a .cine file and extract the camera model.

    Args:
        path: Path to .cine file

    Returns:
        Camera model string if found, None otherwise
    """
    cine_obj = safe_load_cine(path)
    if cine_obj is None:
        return None
    return get_camera_model(cine_obj)


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
    Supports camera suffixes: g (green), v (violet), m (mono/main)
    Returns list of (droplet_id, {cam: path, ...}) sorted by ID.
    """
    folder = Path(folder)
    groups: Dict[str, Dict[str, Optional[Path]]] = {}

    for path in sorted(folder.glob("*.cine")):
        match = re.search(r"(\d+)([gmv])$", path.stem, re.IGNORECASE)
        if not match:
            continue

        droplet_id = match.group(1)
        cam = match.group(2).lower()

        if droplet_id not in groups:
            groups[droplet_id] = {}
        groups[droplet_id][cam] = path

    sorted_ids = sorted(groups.keys(), key=lambda x: int(x))
    return [(did, groups[did]) for did in sorted_ids]
