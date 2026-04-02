"""
CINE file I/O and grouping utilities.

Handles loading .cine files via the Phantom SDK and grouping them
by droplet ID based on filename patterns (e.g. sphere0843g.cine).

Includes silent pyphantom import to suppress SDK banner on worker spawn.
"""

import os
import re
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


# --- Silent pyphantom import ---

@contextmanager
def _suppress_output() -> Generator[None, None, None]:
    """Context manager to silence stdout and stderr at the OS level.

    Uses low-level file descriptor manipulation to suppress output from
    C extensions that bypass Python's sys.stdout/stderr.
    """
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# Load pyphantom components silently (optional)
cine: Optional[Any] = None
utils: Optional[Any] = None
ph: Optional[Any] = None
PYPHANTOM_AVAILABLE = False
PHANTOM_INITIALIZED = False

try:
    with _suppress_output():
        from pyphantom import cine, utils
        PYPHANTOM_AVAILABLE = True

        try:
            from pyphantom import Phantom
            ph = Phantom(init_camera=False)
            PHANTOM_INITIALIZED = ph is not None
        except Exception:
            try:
                ph = Phantom()
                PHANTOM_INITIALIZED = ph is not None
            except Exception:
                ph = None
                PHANTOM_INITIALIZED = False
except ImportError:
    # pyphantom not installed - this is OK, just can't read .cine files
    pass


# --- CINE loading and grouping ---

def safe_load_cine(path: Path) -> Optional[Any]:
    """Load a .cine file, returning None on failure."""
    if not PYPHANTOM_AVAILABLE or cine is None:
        logging.error(f"[CINE LOAD ERROR] {path.name}: pyphantom not available")
        return None

    try:
        cine_obj = cine.Cine.from_filepath(str(path))
        if cine_obj is None:
            return None
        _ = cine_obj.range  # Force handle init
        return cine_obj
    except Exception as e:
        print(f"[CINE LOAD ERROR] {path.name}: {e}")
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
