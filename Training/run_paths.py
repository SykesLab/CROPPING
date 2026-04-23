"""Path utilities for timestamped training datasets and runs.

Folder layout:
    training_output/
        datasets/<YYYYMMDD_HHMMSS>_<name>/
        runs/<YYYYMMDD_HHMMSS>_<name>/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

DATASETS_SUBDIR = "datasets"
RUNS_SUBDIR = "runs"
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


