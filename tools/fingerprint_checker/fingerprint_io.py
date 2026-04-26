"""I/O for the fingerprint checker.

Loads:
  - Synthetic datasets — a `training_output/datasets/<ts>_<name>/` folder
    with `metadata.csv`, `dataset_summary.json`, `blur/`, `sharp/`, etc.
  - Calibration z-stacks — a folder of `.cine` files with an optional
    `positions.csv` providing stage positions.

All loaders are read-only. Subsampling is deterministic (seed=42) and
stratified by defocus so a small subset still spans the full blur range.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class SyntheticSample:
    """One row from a synthetic dataset's metadata.csv plus its image path."""
    index: int
    blur_path: Path
    defocus_mm: float
    sigma_px: float                       # primary label (model space)
    sigma_measured_erf: Optional[float]   # may be NaN (not measured for all)
    erf_r_squared: Optional[float]
    diameter_px: Optional[float]
    camera: Optional[str]
    extra: dict = field(default_factory=dict)  # full metadata row for advanced use


@dataclass
class SyntheticDataset:
    """Loaded view of a synthetic dataset folder."""
    root: Path
    summary: dict                         # dataset_summary.json contents
    metadata: pd.DataFrame                # metadata.csv (full)
    samples: List[SyntheticSample]        # one entry per (subsampled) row
    sample_count_total: int               # full dataset size
    sample_count_used: int                # after subsampling


@dataclass
class CalibrationFrame:
    """A single calibration .cine frame with its stage position."""
    file_path: Path
    stage_position_mm: float        # raw stage position (from positions.csv)
    defocus_mm: float               # signed defocus (= stage - focus_offset)
    frame_idx: int = 0


@dataclass
class CalibrationStack:
    """A calibration z-stack."""
    root: Path
    frames: List[CalibrationFrame]
    focus_offset_mm: float = 0.0    # what stage position was at focus


# ── Synthetic loader ─────────────────────────────────────────────────────


def find_synthetic_dataset(folder: Path) -> Path:
    """Resolve a path to the actual dataset root.

    Accepts either a `<ts>_<name>` dataset folder directly OR a `datasets/`
    parent (returns the most recent dataset inside).
    """
    folder = Path(folder)
    if (folder / "metadata.csv").is_file() and (folder / "blur").is_dir():
        return folder
    # Maybe it's the datasets/ parent
    candidates = [p for p in folder.iterdir() if p.is_dir()
                  and (p / "metadata.csv").is_file()] if folder.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"No synthetic dataset found at {folder} (need metadata.csv + blur/)")
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_synthetic_dataset(
    folder: Path,
    n_samples: Optional[int] = None,
    stratify_bins: int = 20,
    seed: int = 42,
) -> SyntheticDataset:
    """Load metadata + image paths for a synthetic dataset.

    Args:
        folder: dataset root (a `<ts>_<name>` folder).
        n_samples: how many to keep. None = all. Smaller than total = stratified
            subsample by `defocus_mm` so coverage stays uniform across the
            blur range.
        stratify_bins: number of |defocus| bins for stratification.
        seed: deterministic sampling.
    """
    root = find_synthetic_dataset(folder)
    metadata_path = root / "metadata.csv"
    summary_path = root / "dataset_summary.json"
    blur_dir = root / "blur"

    metadata = pd.read_csv(metadata_path)
    if not blur_dir.is_dir():
        raise FileNotFoundError(f"Missing {blur_dir}")
    summary = {}
    if summary_path.is_file():
        with open(summary_path) as f:
            summary = json.load(f)

    total = len(metadata)
    if n_samples is None or n_samples >= total:
        chosen = metadata
    else:
        chosen = _stratified_sample(metadata, n_samples, stratify_bins, seed)

    samples: List[SyntheticSample] = []
    for _, row in chosen.iterrows():
        idx = int(row.get('index', 0))
        # blur images are saved with index_str padded — match dataset writer's '06d' pattern
        # but be tolerant of any width
        blur_path = _resolve_blur_path(blur_dir, idx)
        if blur_path is None:
            continue
        samples.append(SyntheticSample(
            index=idx,
            blur_path=blur_path,
            defocus_mm=float(row.get('defocus_mm', float('nan'))),
            sigma_px=float(row.get('sigma_px', float('nan'))),
            sigma_measured_erf=_nan_or_float(row.get('sigma_measured_erf')),
            erf_r_squared=_nan_or_float(row.get('erf_r_squared')),
            diameter_px=_nan_or_float(row.get('diameter_px')),
            camera=row.get('camera') if pd.notna(row.get('camera')) else None,
            extra=row.to_dict(),
        ))

    return SyntheticDataset(
        root=root,
        summary=summary,
        metadata=metadata,
        samples=samples,
        sample_count_total=total,
        sample_count_used=len(samples),
    )


def _nan_or_float(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _resolve_blur_path(blur_dir: Path, idx: int) -> Optional[Path]:
    """Find the blur image for a given index. The writer uses '06d' padding."""
    candidates = [
        blur_dir / f"{idx:06d}.png",
        blur_dir / f"{idx}.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _stratified_sample(
    metadata: pd.DataFrame, n_samples: int, n_bins: int, seed: int,
) -> pd.DataFrame:
    """Sample `n_samples` rows evenly across |defocus_mm| bins.

    If a bin is too small to hit the per-bin quota, take what's there;
    remaining quota gets distributed across other bins.
    """
    if 'defocus_mm' not in metadata.columns:
        # No defocus column — fall back to random uniform sample
        return metadata.sample(n=n_samples, random_state=seed)
    df = metadata.copy()
    df['_abs_z'] = df['defocus_mm'].abs()
    try:
        df['_bin'] = pd.qcut(df['_abs_z'], q=n_bins, duplicates='drop')
    except (ValueError, TypeError):
        df['_bin'] = pd.cut(df['_abs_z'], bins=n_bins)
    per_bin = max(1, n_samples // n_bins)
    sampled = (
        df.groupby('_bin', observed=True, group_keys=False)
        .apply(lambda g: g.sample(min(per_bin, len(g)), random_state=seed),
               include_groups=False)
    )
    # `include_groups=False` drops the bin column from the result; restore
    # the metadata columns by re-aligning on the index
    sampled = df.loc[sampled.index]
    # Top up if we under-shot due to small bins
    if len(sampled) < n_samples:
        leftover = df.drop(sampled.index)
        if len(leftover) > 0:
            extra_n = min(n_samples - len(sampled), len(leftover))
            sampled = pd.concat(
                [sampled, leftover.sample(n=extra_n, random_state=seed)])
    return sampled.drop(columns=['_abs_z', '_bin']).reset_index(drop=True)


def load_blur_image(path: Path) -> np.ndarray:
    """Load a blur image as a 2D float32 array in [0,1]."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


# ── Calibration loader ───────────────────────────────────────────────────


def load_calibration_stack(
    folder: Path,
    focus_offset_mm: float = 0.0,
    frame_idx: int = 0,
) -> CalibrationStack:
    """Load a calibration z-stack folder.

    Expects: a folder containing `.cine` files and an optional
    `positions.csv` mapping `filename → stage_position_mm`. Without the
    CSV, frames are still listed but their `defocus_mm` is NaN.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Calibration folder does not exist: {folder}")
    cine_files = sorted(folder.glob("*.cine"), key=_cine_sort_key)
    if not cine_files:
        raise FileNotFoundError(f"No .cine files in {folder}")

    positions = _load_positions_csv(folder)
    frames: List[CalibrationFrame] = []
    for cine in cine_files:
        stage = positions.get(cine.name, positions.get(cine.stem, float('nan')))
        defocus = (stage - focus_offset_mm) if not np.isnan(stage) else float('nan')
        frames.append(CalibrationFrame(
            file_path=cine,
            stage_position_mm=stage,
            defocus_mm=defocus,
            frame_idx=frame_idx,
        ))
    return CalibrationStack(
        root=folder, frames=frames, focus_offset_mm=focus_offset_mm)


def _cine_sort_key(p: Path) -> tuple:
    """Sort .cine files by trailing integer in stem, falling back to name."""
    import re
    m = re.search(r'_(\d+)$', p.stem)
    if m:
        return (0, int(m.group(1)))
    return (1, p.name)


def _load_positions_csv(folder: Path) -> dict:
    """Load filename→stage-position map from positions.csv."""
    csv = folder / "positions.csv"
    if not csv.is_file():
        return {}
    df = pd.read_csv(csv)
    if df.shape[1] < 2:
        return {}
    out = {}
    for _, row in df.iterrows():
        fn = str(row.iloc[0])
        try:
            pos = float(row.iloc[1])
        except (ValueError, TypeError):
            continue
        out[fn] = pos
        out[Path(fn).stem] = pos
    return out


def load_cine_frame(
    cine_path: Path, frame_idx: int = 0,
) -> Optional[np.ndarray]:
    """Load one frame from a .cine file as float32 [0,1].

    Reuses Calibration's cine_loader infrastructure. Returns None if
    pyphantom isn't installed or the file can't be read.
    """
    try:
        from cine_loader import CineLoader  # Calibration/cine_loader.py
    except ImportError:
        return None
    try:
        loader = CineLoader([cine_path])
        img = loader._extract_frame_from_cine(cine_path, frame_idx, average_frames=1)
    except Exception:
        return None
    if img is None:
        return None
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


# ── Iteration helpers ────────────────────────────────────────────────────


def iterate_synthetic_images(
    dataset: SyntheticDataset,
) -> Iterator[Tuple[SyntheticSample, np.ndarray]]:
    """Yield (sample_metadata, image array) pairs for the loaded subset."""
    for sample in dataset.samples:
        try:
            img = load_blur_image(sample.blur_path)
        except IOError:
            continue
        yield sample, img


def iterate_calibration_frames(
    stack: CalibrationStack,
) -> Iterator[Tuple[CalibrationFrame, np.ndarray]]:
    """Yield (frame_metadata, image array) pairs for the calibration stack."""
    for frame in stack.frames:
        img = load_cine_frame(frame.file_path, frame.frame_idx)
        if img is None:
            continue
        yield frame, img


# ── Generic image folder loader (for real crops + inference crops) ───────


@dataclass
class CropSample:
    """A single crop image with optional parsed defocus."""
    file_path: Path
    defocus_mm: Optional[float] = None     # parsed from filename if available


@dataclass
class CropFolder:
    """A folder of crop images for fingerprinting."""
    root: Path
    label: str                              # 'real' / 'inference' / etc.
    samples: List[CropSample]
    sample_count_total: int
    sample_count_used: int


def load_crop_folder(
    folder: Path,
    label: str = 'real',
    n_samples: Optional[int] = None,
    parse_defocus_from_filename: bool = True,
    seed: int = 42,
) -> CropFolder:
    """Load a folder of PNG crops. Recurses into subfolders.

    If ``parse_defocus_from_filename`` is True, extracts defocus from
    filenames matching ``z<signed_value>mm`` (the project's standard
    convention emitted by `Training.run_paths.parse_true_z_from_filename`).
    Samples without a parseable defocus get ``defocus_mm=None``.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Crop folder does not exist: {folder}")

    pngs = sorted(folder.rglob("*.png"))
    if not pngs:
        # Try other common extensions
        pngs = sorted(list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff"))
                      + list(folder.rglob("*.jpg")))
    if not pngs:
        raise FileNotFoundError(f"No image files found in {folder}")
    total = len(pngs)

    # Subsample if requested
    if n_samples is not None and n_samples < total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=n_samples, replace=False)
        idx.sort()
        pngs = [pngs[i] for i in idx]

    parser = None
    if parse_defocus_from_filename:
        try:
            from run_paths import parse_true_z_from_filename
            parser = parse_true_z_from_filename
        except ImportError:
            parser = None

    samples = []
    for p in pngs:
        z = parser(p.name) if parser else None
        samples.append(CropSample(file_path=p, defocus_mm=z))

    return CropFolder(
        root=folder,
        label=label,
        samples=samples,
        sample_count_total=total,
        sample_count_used=len(samples),
    )


def iterate_crop_folder(
    folder: CropFolder,
) -> Iterator[Tuple[CropSample, np.ndarray]]:
    """Yield (sample, image) pairs from a crop folder."""
    for s in folder.samples:
        try:
            img = load_blur_image(s.file_path)
        except IOError:
            continue
        yield s, img


def load_sample_image_by_row(row) -> Optional[np.ndarray]:
    """Universal image loader: dispatch on source_path's extension.

    Pass a row from any of the per-source fingerprint DataFrames (i.e.
    ``result.synthetic_fingerprints.iloc[k]``). Returns the float32 [0,1]
    image array, or None if loading fails (e.g. .cine without pyphantom).
    """
    if row is None:
        return None
    path_str = row.get('source_path')
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    if path.suffix.lower() == '.cine':
        return load_cine_frame(path)
    try:
        return load_blur_image(path)
    except IOError:
        return None
