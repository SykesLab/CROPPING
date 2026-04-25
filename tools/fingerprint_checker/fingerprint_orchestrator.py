"""Composes io + metrics + analyses into one run.

The orchestrator handles the typical workflow:
  1. Load synthetic dataset (with optional subsampling)
  2. Compute fingerprints for synthetic samples
  3. Run Check A (scale-chain, configs only)
  4. Run Check B-internal (synthetic sigma trend — does measured ERF
     correlate with metadata sigma?)
  5. If calibration provided: load + compute fingerprints, run Check B
     (synthetic vs calibration alignment via NN matching)
  6. (Check C — coverage of real crops — deferred to Phase 2)

Designed for both the GUI and headless / scripted use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .fingerprint_analyses import (
    AlignmentResult,
    alignment_check_nn,
    flag_alignment,
)
from .fingerprint_io import (
    SyntheticDataset,
    SyntheticSample,
    iterate_calibration_frames,
    iterate_synthetic_images,
    load_calibration_stack,
    load_synthetic_dataset,
)
from .fingerprint_metrics import compute_fingerprint, FingerprintRecord
from .scale_chain import RoundTripResult, round_trip_check


ProgressCallback = Optional[Callable[[str, float], None]]
"""Signature: progress_callback(message: str, fraction_done: float in [0,1])."""


@dataclass
class AllChecksResult:
    """Aggregate result from one run of all three checks."""
    config_summary: dict
    synthetic_dataset_root: Optional[str] = None
    calibration_root: Optional[str] = None

    # Computed fingerprints (one row per sample)
    synthetic_fingerprints: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_fingerprints: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Check A
    scale_chain: Optional[RoundTripResult] = None

    # Check B-internal: how do measured image features track metadata sigma?
    sigma_trend_correlations: dict = field(default_factory=dict)

    # Check B: alignment between synthetic and calibration (if calibration provided)
    alignment_synth_vs_calib: Optional[AlignmentResult] = None
    alignment_flags: dict = field(default_factory=dict)

    # Misc / diagnostics
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            'config_summary': self.config_summary,
            'synthetic_dataset_root': self.synthetic_dataset_root,
            'calibration_root': self.calibration_root,
            'synthetic_fingerprint_count': len(self.synthetic_fingerprints),
            'calibration_fingerprint_count': len(self.calibration_fingerprints),
            'scale_chain': self.scale_chain.to_dict() if self.scale_chain else None,
            'sigma_trend_correlations': self.sigma_trend_correlations,
            'alignment_synth_vs_calib': (
                self.alignment_synth_vs_calib.to_dict()
                if self.alignment_synth_vs_calib else None
            ),
            'alignment_flags': self.alignment_flags,
            'diagnostics': self.diagnostics,
        }
        return d


# ── Fingerprint extraction loops (use the IO iterators) ─────────────────


def fingerprint_synthetic(
    dataset: SyntheticDataset,
    progress: ProgressCallback = None,
    skip_uniformity: bool = False,
    skip_symmetry: bool = False,
) -> pd.DataFrame:
    """Compute fingerprints for every sample in the loaded subset.

    Returns a DataFrame with one row per sample and one column per metric,
    plus pass-through metadata columns (`defocus_mm`, `sigma_px`, `camera`).
    """
    rows = []
    n = dataset.sample_count_used
    for i, (sample, image) in enumerate(iterate_synthetic_images(dataset)):
        if progress and (i % max(1, n // 50) == 0):
            progress(f"Synthetic fingerprint {i+1}/{n}", i / max(1, n))
        rec = compute_fingerprint(
            image,
            erf_sigma_precomputed=sample.sigma_measured_erf,
            erf_r_squared_precomputed=sample.erf_r_squared,
            skip_uniformity=skip_uniformity,
            skip_symmetry=skip_symmetry,
        )
        row = rec.to_dict()
        row['source_path'] = str(sample.blur_path)
        row['source_type'] = 'synthetic'
        row['index'] = sample.index
        row['defocus_mm'] = sample.defocus_mm
        row['sigma_px_metadata'] = sample.sigma_px
        row['camera'] = sample.camera
        # Drop the raw nested-dict 'metadata' field for tabular output
        row.pop('metadata', None)
        rows.append(row)
    if progress:
        progress("Synthetic fingerprint done", 1.0)
    return pd.DataFrame(rows)


def fingerprint_calibration(
    stack,
    progress: ProgressCallback = None,
) -> pd.DataFrame:
    """Compute fingerprints for every calibration frame.

    Returns a DataFrame with one row per frame. NaN-padded if the .cine
    can't be loaded (e.g. pyphantom missing on this machine).
    """
    rows = []
    n = len(stack.frames)
    loaded = 0
    failed = 0
    for i, (frame, image) in enumerate(iterate_calibration_frames(stack)):
        if progress and (i % max(1, n // 10) == 0):
            progress(f"Calibration fingerprint {i+1}/{n}", i / max(1, n))
        rec = compute_fingerprint(image)  # full set, no skips — only ~60 frames
        row = rec.to_dict()
        row['source_path'] = str(frame.file_path)
        row['source_type'] = 'calibration'
        row['index'] = i
        row['defocus_mm'] = frame.defocus_mm
        row['stage_position_mm'] = frame.stage_position_mm
        row.pop('metadata', None)
        rows.append(row)
        loaded += 1
    failed = n - loaded
    if progress:
        progress(f"Calibration done ({loaded}/{n} loaded, {failed} failed)", 1.0)
    return pd.DataFrame(rows)


# ── Check B-internal: synthetic sigma trend ─────────────────────────────


def synthetic_sigma_trend_correlations(
    df: pd.DataFrame,
    metadata_sigma_col: str = 'sigma_px_metadata',
) -> dict:
    """How well do measured features track the metadata-known sigma?

    Returns a dict of {metric → {'pearson_r': float, 'n_finite': int,
    'expected_direction': '+' | '-' | '?'}}.
    """
    expected = {
        # Should INCREASE with sigma:
        'erf_sigma_px': '+',
        'edge_transition_width': '+',
        # Should DECREASE with sigma (more blur → less sharpness):
        'edge_gradient_max': '-',
        'laplacian_variance': '-',
        'tenengrad': '-',
        'high_freq_energy_ratio': '-',
    }
    out = {}
    if metadata_sigma_col not in df.columns:
        return out
    s_meta = df[metadata_sigma_col].to_numpy(dtype=float)
    for metric, sign in expected.items():
        if metric not in df.columns:
            continue
        s_meas = df[metric].to_numpy(dtype=float)
        mask = np.isfinite(s_meta) & np.isfinite(s_meas)
        if mask.sum() < 5:
            out[metric] = {
                'pearson_r': float('nan'),
                'n_finite': int(mask.sum()),
                'expected_direction': sign,
            }
            continue
        r = float(np.corrcoef(s_meta[mask], s_meas[mask])[0, 1])
        out[metric] = {
            'pearson_r': r,
            'n_finite': int(mask.sum()),
            'expected_direction': sign,
        }
    return out


# ── Top-level entry ──────────────────────────────────────────────────────


def run_all_checks(
    *,
    config_path: Optional[Path] = None,
    config_dict: Optional[dict] = None,
    synthetic_dataset_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    n_synthetic_samples: Optional[int] = None,
    calibration_focus_offset_mm: float = 0.0,
    k_neighbours: int = 20,
    progress: ProgressCallback = None,
    skip_uniformity: bool = False,
    skip_symmetry: bool = False,
) -> AllChecksResult:
    """Run Check A + Check B end-to-end.

    Either ``config_path`` or ``config_dict`` must be given (used for Check A).
    ``synthetic_dataset_path`` is required for any image-based check.
    ``calibration_path`` is optional; when omitted, only the synthetic-only
    sigma-trend correlation analysis runs (no NN alignment).
    """
    diagnostics: List[str] = []

    # Resolve config
    if config_dict is None:
        if config_path is None:
            raise ValueError("Either config_path or config_dict required")
        from .scale_chain import load_config_auto
        config_dict = load_config_auto(Path(config_path))

    if progress:
        progress("Running Check A — scale-chain round-trip", 0.0)

    # Check A
    scale_result = round_trip_check(config_dict)

    # Without a synthetic dataset, that's all we can do
    if synthetic_dataset_path is None:
        if progress:
            progress("Done (Check A only — no synthetic dataset provided)", 1.0)
        return AllChecksResult(
            config_summary={'config_keys': list(config_dict.keys())},
            scale_chain=scale_result,
            diagnostics=diagnostics + [
                "No synthetic dataset provided — only Check A ran."
            ],
        )

    if progress:
        progress("Loading synthetic dataset metadata", 0.05)
    synth_dataset = load_synthetic_dataset(
        Path(synthetic_dataset_path),
        n_samples=n_synthetic_samples,
    )

    if progress:
        progress(
            f"Computing fingerprints for {synth_dataset.sample_count_used} "
            f"synthetic samples", 0.1)

    def _synth_progress(msg: str, frac: float):
        if progress:
            # Map [0,1] to [0.1, 0.55] of overall progress
            progress(msg, 0.1 + frac * 0.45)

    synth_fp = fingerprint_synthetic(
        synth_dataset, progress=_synth_progress,
        skip_uniformity=skip_uniformity, skip_symmetry=skip_symmetry,
    )

    sigma_trends = synthetic_sigma_trend_correlations(synth_fp)

    # If no calibration, return now
    if calibration_path is None:
        if progress:
            progress("Done (no calibration — alignment skipped)", 1.0)
        return AllChecksResult(
            config_summary={'config_keys': list(config_dict.keys())},
            synthetic_dataset_root=str(synth_dataset.root),
            synthetic_fingerprints=synth_fp,
            scale_chain=scale_result,
            sigma_trend_correlations=sigma_trends,
            diagnostics=diagnostics + [
                "No calibration provided — sigma-trend done; alignment skipped."
            ],
        )

    # Load + fingerprint calibration
    if progress:
        progress("Loading calibration z-stack", 0.6)
    calib_stack = load_calibration_stack(
        Path(calibration_path),
        focus_offset_mm=calibration_focus_offset_mm,
    )

    def _calib_progress(msg: str, frac: float):
        if progress:
            progress(msg, 0.6 + frac * 0.3)

    calib_fp = fingerprint_calibration(calib_stack, progress=_calib_progress)

    if calib_fp.empty:
        diagnostics.append(
            "Calibration fingerprint DataFrame is empty — likely pyphantom "
            "missing or all .cine reads failed. Skipping alignment.")
        alignment_result = None
        flags = {}
    else:
        if progress:
            progress("Running Check B — synthetic↔calibration alignment", 0.92)
        alignment_result = alignment_check_nn(
            anchor_df=calib_fp,
            other_df=synth_fp,
            anchor_label='calibration',
            other_label='synthetic',
            k=k_neighbours,
        )
        flags = flag_alignment(alignment_result)

    if progress:
        progress("Done", 1.0)

    return AllChecksResult(
        config_summary={'config_keys': list(config_dict.keys())},
        synthetic_dataset_root=str(synth_dataset.root),
        calibration_root=str(calib_stack.root),
        synthetic_fingerprints=synth_fp,
        calibration_fingerprints=calib_fp,
        scale_chain=scale_result,
        sigma_trend_correlations=sigma_trends,
        alignment_synth_vs_calib=alignment_result,
        alignment_flags=flags,
        diagnostics=diagnostics,
    )
