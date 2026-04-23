"""Path utilities for timestamped training datasets, runs, and validation results.

Folder layout:
    training_output/
        datasets/<YYYYMMDD_HHMMSS>_<name>/
        runs/<YYYYMMDD_HHMMSS>_<name>/
        synthetic_validation/<run_name>/test_<YYYYMMDD_HHMMSS>_<variant>/
        real_crop_validation/<run_name>/
            test_<YYYYMMDD_HHMMSS>_<variant>/
            edits/<edited_checkpoint>.pth
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

DATASETS_SUBDIR = "datasets"
RUNS_SUBDIR = "runs"
SYNTHETIC_VALIDATION_SUBDIR = "synthetic_validation"
REAL_CROP_VALIDATION_SUBDIR = "real_crop_validation"
EDITS_SUBDIR = "edits"
TEST_PREFIX = "test"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

# Folder-name pattern: YYYYMMDD_HHMMSS_<rest>
_RUN_FOLDER_RE = re.compile(r"^\d{8}_\d{6}(_.+)?$")


def sanitise_run_name(name: str) -> str:
    """Replace illegal path characters with underscores; fall back to 'unnamed'."""
    if not name:
        return "unnamed"
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip("._")
    return cleaned or "unnamed"


def make_run_folder_name(name: Optional[str], default: str = "run") -> str:
    """Compose a timestamped folder name: <YYYYMMDD_HHMMSS>_<sanitised>."""
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    safe = sanitise_run_name(name) if name else default
    return f"{timestamp}_{safe}"


def datasets_root(training_output_root: Path) -> Path:
    return Path(training_output_root) / DATASETS_SUBDIR


def runs_root(training_output_root: Path) -> Path:
    return Path(training_output_root) / RUNS_SUBDIR


def list_datasets(training_output_root: Path) -> List[Path]:
    """All dataset folders under training_output/datasets/, newest first."""
    root = datasets_root(training_output_root)
    if not root.is_dir():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and _RUN_FOLDER_RE.match(p.name)]
    return sorted(folders, key=lambda p: p.name, reverse=True)


def list_runs(training_output_root: Path) -> List[Path]:
    """All run folders under training_output/runs/, newest first."""
    root = runs_root(training_output_root)
    if not root.is_dir():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and _RUN_FOLDER_RE.match(p.name)]
    return sorted(folders, key=lambda p: p.name, reverse=True)


def find_latest_dataset(training_output_root: Path) -> Optional[Path]:
    items = list_datasets(training_output_root)
    return items[0] if items else None


def find_latest_run(training_output_root: Path) -> Optional[Path]:
    items = list_runs(training_output_root)
    return items[0] if items else None


def validate_dataset(path: Path) -> Tuple[bool, str]:
    """Check a folder looks like a usable dataset. Returns (ok, message)."""
    p = Path(path)
    if not p.is_dir():
        return False, f"Not a folder: {p}"
    if not (p / "metadata.csv").is_file():
        return False, "Missing metadata.csv"
    if not (p / "blur").is_dir():
        return False, "Missing blur/ subfolder"
    return True, "OK"


# ── Validation results layout ────────────────────────────────────────────

def synthetic_validation_root(training_output_root: Path) -> Path:
    return Path(training_output_root) / SYNTHETIC_VALIDATION_SUBDIR


def real_crop_validation_root(training_output_root: Path) -> Path:
    return Path(training_output_root) / REAL_CROP_VALIDATION_SUBDIR


def synthetic_validation_dir(training_output_root: Path, run_name: str) -> Path:
    """Per-model folder under synthetic_validation/<run_name>/."""
    return synthetic_validation_root(training_output_root) / run_name


def real_crop_validation_dir(training_output_root: Path, run_name: str) -> Path:
    """Per-model folder under real_crop_validation/<run_name>/."""
    return real_crop_validation_root(training_output_root) / run_name


def edits_dir(training_output_root: Path, run_name: str) -> Path:
    """edits/ folder under real_crop_validation/<run_name>/edits/."""
    return real_crop_validation_dir(training_output_root, run_name) / EDITS_SUBDIR


def make_test_folder_name(variant: str = "original") -> str:
    """Compose a timestamped test-result folder name: test_<YYYYMMDD_HHMMSS>_<variant>."""
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    safe_variant = sanitise_run_name(variant) if variant else "original"
    return f"{TEST_PREFIX}_{timestamp}_{safe_variant}"


def detect_run_name(checkpoint_path: Path) -> Optional[str]:
    """Walk up a checkpoint's path to find the enclosing run name.

    Recognised locations:
      * .../runs/<run_name>/checkpoints/<file>.pth
      * .../real_crop_validation/<run_name>/edits/<file>.pth
    Returns None if the checkpoint is elsewhere on disk.
    """
    parts = Path(checkpoint_path).resolve().parts
    for anchor in (RUNS_SUBDIR, REAL_CROP_VALIDATION_SUBDIR):
        if anchor in parts:
            idx = parts.index(anchor)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def detect_variant(checkpoint_path: Path) -> str:
    """Short label describing which model variant a checkpoint represents.

    Convention:
      * under runs/<run>/checkpoints/     → 'original'
      * under .../edits/<stem>_v<N>.pth   → 'v<N>'
      * under .../edits/<other>.pth       → sanitised stem
      * anywhere else                     → sanitised stem
    """
    p = Path(checkpoint_path)
    parts = p.resolve().parts
    if EDITS_SUBDIR in parts:
        stem = p.stem
        m = re.search(r'_v(\d+)$', stem)
        if m:
            return f"v{m.group(1)}"
        return sanitise_run_name(stem)
    if RUNS_SUBDIR in parts:
        return 'original'
    return sanitise_run_name(p.stem)


