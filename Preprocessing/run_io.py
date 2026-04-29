"""
Run-folder management for timestamped Preprocessing outputs.

Each pipeline invocation creates a fresh run directory under
`<output_root>/runs/<YYYY-MM-DD_HHMMSS>_<label>/` so re-runs never
overwrite each other. The downstream contract (Training reading
`Focus/sharp_crops.csv`) is preserved verbatim inside each run dir.

The active run dir is propagated to worker subprocesses via the
`CROPPING_RUN_ROOT` environment variable, which `config.py` reads
on import.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


RUN_ROOT_ENV = "CROPPING_RUN_ROOT"
OUTPUT_ROOT_ENV = "CROPPING_OUTPUT_ROOT"


def _safe_label(text: str) -> str:
    """Strip characters that don't belong in a folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return cleaned.strip("-_.") or "run"


def default_label(cine_root: Path) -> str:
    """Default run label = basename of the cine root, sanitised."""
    return _safe_label(Path(cine_root).name)


def make_run_dir(
    output_root: Path,
    cine_root: Path,
    label: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Create a fresh timestamped run directory and return its Path.

    Layout: <output_root>/runs/<YYYY-MM-DD_HHMMSS>_<label>/

    Raises FileExistsError if the dir somehow already exists (timestamp
    collision — bumps to the next second only on a real clash, callers
    can retry).
    """
    output_root = Path(output_root)
    label = _safe_label(label) if label else default_label(cine_root)
    if timestamp is None:
        timestamp = datetime.now()
    stamp = timestamp.strftime("%Y-%m-%d_%H%M%S")
    run_dir = output_root / "runs" / f"{stamp}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def find_latest_run(output_root: Path) -> Optional[Path]:
    """Return the most recently modified run dir, or None if none exist."""
    runs_dir = Path(output_root) / "runs"
    if not runs_dir.is_dir():
        return None
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def write_run_metadata(run_dir: Path, **fields: Any) -> Path:
    """Write run_metadata.yaml with the supplied fields. Returns its path."""
    return _write(run_dir, fields, merge=False)


def update_run_metadata(run_dir: Path, **fields: Any) -> Path:
    """Merge fields into run_metadata.yaml (creating it if missing)."""
    return _write(run_dir, fields, merge=True)


def _write(run_dir: Path, fields: Dict[str, Any], merge: bool) -> Path:
    path = Path(run_dir) / "run_metadata.yaml"
    existing: Dict[str, Any] = {}
    if merge and path.exists():
        with open(path) as f:
            existing = yaml.safe_load(f) or {}
    serialisable = {k: (str(v) if isinstance(v, Path) else v) for k, v in fields.items()}
    existing.update(serialisable)
    with open(path, "w") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)
    return path


def set_env_run_root(run_dir: Path) -> None:
    """Publish the active run dir so subprocesses pick it up on import."""
    os.environ[RUN_ROOT_ENV] = str(run_dir)


def clear_env_run_root() -> None:
    """Remove the env var (used after a run completes, optional)."""
    os.environ.pop(RUN_ROOT_ENV, None)


def get_env_run_root() -> Optional[Path]:
    """Read the active run dir from env, if set."""
    val = os.environ.get(RUN_ROOT_ENV)
    return Path(val) if val else None
