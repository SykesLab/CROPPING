"""Disk caching for computed fingerprint DataFrames.

The fingerprint computation is the slow step (~60+ seconds for 200
samples with all metrics). Caching to CSV gives ~100x speedup on
subsequent runs of the same dataset.

Cache invalidation: a cache file is considered stale if any file under
the source folder has a more recent mtime than the cache itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _short_hash(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:10]


def cache_filename(
    label: str,
    source_path: Path,
    n_samples: Optional[int] = None,
    extra_key: str = "",
) -> str:
    """Build a deterministic cache filename keyed by source path + n_samples."""
    h = _short_hash(str(Path(source_path).resolve()))
    n_part = f"_n{n_samples}" if n_samples else "_all"
    extra = f"_{extra_key}" if extra_key else ""
    return f"{label}_{h}{n_part}{extra}.csv"


def cache_path_for(
    cache_dir: Path,
    label: str,
    source_path: Path,
    n_samples: Optional[int] = None,
    extra_key: str = "",
) -> Path:
    """Full cache file path."""
    return Path(cache_dir) / cache_filename(label, source_path, n_samples, extra_key)


def is_cache_valid(cache_path: Path, source_path: Path) -> bool:
    """Cache is valid iff it exists AND no file under source has a newer mtime.

    Folder-source: walks the tree to find the maximum mtime.
    File-source: compares directly.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return False
    cache_mtime = cache_path.stat().st_mtime
    src = Path(source_path)
    if src.is_file():
        return src.stat().st_mtime <= cache_mtime
    if src.is_dir():
        latest = 0.0
        # Limit walk to common file types to avoid scanning huge irrelevant trees
        for pattern in ("metadata.csv", "*.png", "*.cine", "*.tif", "*.jpg"):
            for p in src.rglob(pattern):
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if m > latest:
                    latest = m
        return latest <= cache_mtime
    return False


def load_cache(cache_path: Path) -> pd.DataFrame:
    """Load a cached fingerprint DataFrame."""
    return pd.read_csv(cache_path)


def save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Write a fingerprint DataFrame to disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)


def clear_cache(cache_dir: Path) -> int:
    """Delete all .csv files under cache_dir. Returns count removed."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return 0
    n = 0
    for p in cache_dir.glob("*.csv"):
        try:
            p.unlink()
            n += 1
        except OSError:
            continue
    return n
