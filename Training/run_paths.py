"""Path utilities for timestamped datasets, models, and test outputs.

Folder layout::

    training_output/
        datasets/<YYYYMMDD_HHMMSS>_<name>/
            blur/ sharp/ blur_map/
            metadata.csv  generation_config.yaml  dataset_summary.json

        models/<YYYYMMDD_HHMMSS>_<name>/
            checkpoints/              ← .pth files
            logs/                     ← tensorboard events +
                run_metadata.json        per-run artifacts live here
                training_history.yaml
                training_curves.png
            training_config.yaml      ← resolved config (model root)
            tests/
                synthetic/<test_<ts>>/
                real_crop/<test_<ts>>/
            edits/
                <user_named>/
                    dme_best.pth
                    tests/
                        synthetic/<test_<ts>>/
                        real_crop/<test_<ts>>/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ── Layout constants ─────────────────────────────────────────────────────

DATASETS_SUBDIR = "datasets"
MODELS_SUBDIR = "models"
CHECKPOINTS_SUBDIR = "checkpoints"
TESTS_SUBDIR = "tests"
SYNTHETIC_TESTS_SUBDIR = "synthetic"
REAL_CROP_TESTS_SUBDIR = "real_crop"
EDITS_SUBDIR = "edits"
TEST_PREFIX = "test"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

# Folder-name pattern: YYYYMMDD_HHMMSS_<rest>
_RUN_FOLDER_RE = re.compile(r"^\d{8}_\d{6}(_.+)?$")

# Crop/frame filenames encode their true z-position as 'z<signed_value>mm',
# e.g. 'sphere4_z-6.20mm.png' or 'frame_042_z+1.50mm.png'.
DEFOCUS_FILENAME_PATTERN = re.compile(r"z([+-]?\d+\.?\d*)mm")


def parse_true_z_from_filename(name) -> Optional[float]:
    """Extract the true z-position (mm, signed) from a filename.

    Returns None if the filename doesn't match the expected pattern.
    """
    m = DEFOCUS_FILENAME_PATTERN.search(str(name))
    return float(m.group(1)) if m else None


# ── Naming helpers ───────────────────────────────────────────────────────

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


def make_test_folder_name() -> str:
    """Compose a timestamped test-result folder name: test_<YYYYMMDD_HHMMSS>.

    No variant suffix — the variant is always implied by location:
    under ``models/<run>/tests/`` for the original, under
    ``models/<run>/edits/<edit>/tests/`` for edited versions.
    """
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    return f"{TEST_PREFIX}_{timestamp}"


# ── Root paths ───────────────────────────────────────────────────────────

def datasets_root(training_output_root: Path) -> Path:
    return Path(training_output_root) / DATASETS_SUBDIR


def models_root(training_output_root: Path) -> Path:
    return Path(training_output_root) / MODELS_SUBDIR


# ── Listing + latest ─────────────────────────────────────────────────────

def list_datasets(training_output_root: Path) -> List[Path]:
    """All dataset folders under training_output/datasets/, newest first."""
    root = datasets_root(training_output_root)
    if not root.is_dir():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and _RUN_FOLDER_RE.match(p.name)]
    return sorted(folders, key=lambda p: p.name, reverse=True)


def list_models(training_output_root: Path) -> List[Path]:
    """All model folders under training_output/models/, newest first."""
    root = models_root(training_output_root)
    if not root.is_dir():
        return []
    folders = [p for p in root.iterdir() if p.is_dir() and _RUN_FOLDER_RE.match(p.name)]
    return sorted(folders, key=lambda p: p.name, reverse=True)


def find_latest_dataset(training_output_root: Path) -> Optional[Path]:
    items = list_datasets(training_output_root)
    return items[0] if items else None


def find_latest_model(training_output_root: Path) -> Optional[Path]:
    items = list_models(training_output_root)
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


# ── Per-model paths ──────────────────────────────────────────────────────

def model_dir(training_output_root: Path, model_name: str) -> Path:
    """training_output/models/<model_name>/"""
    return models_root(training_output_root) / model_name


def model_checkpoints_dir(training_output_root: Path, model_name: str) -> Path:
    """training_output/models/<model_name>/checkpoints/"""
    return model_dir(training_output_root, model_name) / CHECKPOINTS_SUBDIR


def model_edits_dir(training_output_root: Path, model_name: str) -> Path:
    """training_output/models/<model_name>/edits/"""
    return model_dir(training_output_root, model_name) / EDITS_SUBDIR


def edit_dir(training_output_root: Path, model_name: str, edit_name: str) -> Path:
    """training_output/models/<model_name>/edits/<edit_name>/"""
    return model_edits_dir(training_output_root, model_name) / sanitise_run_name(edit_name)


def tests_dir(training_output_root: Path, model_name: str, kind: str,
              edit_name: Optional[str] = None) -> Path:
    """Test-results folder for a given model + kind (+ optional edit).

    kind must be 'synthetic' or 'real_crop'. If edit_name is given, the path
    lands under edits/<edit_name>/tests/<kind>/; otherwise under
    tests/<kind>/ at the model root (i.e. tests of the original checkpoint).
    """
    if kind == 'synthetic':
        leaf = SYNTHETIC_TESTS_SUBDIR
    elif kind == 'real_crop':
        leaf = REAL_CROP_TESTS_SUBDIR
    else:
        raise ValueError(f"kind must be 'synthetic' or 'real_crop', got {kind!r}")

    if edit_name:
        base = edit_dir(training_output_root, model_name, edit_name)
    else:
        base = model_dir(training_output_root, model_name)
    return base / TESTS_SUBDIR / leaf


# ── Checkpoint-path introspection ────────────────────────────────────────

def detect_model_name(checkpoint_path: Path) -> Optional[str]:
    """Walk up a checkpoint's path to find the enclosing model-folder name.

    Recognised location: ``.../models/<model_name>/...`` (including under
    checkpoints/, edits/<edit>/, tests/, etc.)
    Returns None if the checkpoint isn't in a recognised location.
    """
    parts = Path(checkpoint_path).resolve().parts
    if MODELS_SUBDIR in parts:
        idx = parts.index(MODELS_SUBDIR)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def detect_variant(checkpoint_path: Path) -> str:
    """Label describing which model variant a checkpoint represents.

    * under ``models/<m>/checkpoints/<file>`` → 'original'
    * under ``models/<m>/edits/<edit>/<file>`` → '<edit>'
    * anywhere else → the file stem (sanitised)
    """
    p = Path(checkpoint_path)
    parts = p.resolve().parts
    if EDITS_SUBDIR in parts:
        idx = parts.index(EDITS_SUBDIR)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if MODELS_SUBDIR in parts:
        return 'original'
    return sanitise_run_name(p.stem)
