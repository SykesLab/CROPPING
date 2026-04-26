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
    CoverageResult,
    alignment_check_nn,
    coverage_check,
    flag_alignment,
    flag_coverage,
    joint_coverage,
)
from .fingerprint_cache import (
    cache_path_for, is_cache_valid, load_cache, save_cache,
)
from .fingerprint_io import (
    CropFolder,
    SyntheticDataset,
    SyntheticSample,
    iterate_calibration_frames,
    iterate_crop_folder,
    iterate_synthetic_images,
    load_calibration_stack,
    load_crop_folder,
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

    # Check B: alignment between inference and calibration (Phase 2 addition)
    alignment_inference_vs_calib: Optional[AlignmentResult] = None
    alignment_inference_flags: dict = field(default_factory=dict)

    # Check C: distribution coverage (Phase 2 addition)
    real_fingerprints: pd.DataFrame = field(default_factory=pd.DataFrame)
    inference_fingerprints: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage_synth_vs_real: Optional[CoverageResult] = None
    coverage_synth_vs_inference: Optional[CoverageResult] = None
    coverage_real_flags: dict = field(default_factory=dict)
    coverage_inference_flags: dict = field(default_factory=dict)
    # Phase 4: joint multivariate coverage (single-number supplement)
    joint_coverage_real: dict = field(default_factory=dict)
    joint_coverage_inference: dict = field(default_factory=dict)

    # Misc / diagnostics
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            'config_summary': self.config_summary,
            'synthetic_dataset_root': self.synthetic_dataset_root,
            'calibration_root': self.calibration_root,
            'synthetic_fingerprint_count': len(self.synthetic_fingerprints),
            'calibration_fingerprint_count': len(self.calibration_fingerprints),
            'real_fingerprint_count': len(self.real_fingerprints),
            'inference_fingerprint_count': len(self.inference_fingerprints),
            'scale_chain': self.scale_chain.to_dict() if self.scale_chain else None,
            'sigma_trend_correlations': self.sigma_trend_correlations,
            'alignment_synth_vs_calib': (
                self.alignment_synth_vs_calib.to_dict()
                if self.alignment_synth_vs_calib else None
            ),
            'alignment_flags': self.alignment_flags,
            'alignment_inference_vs_calib': (
                self.alignment_inference_vs_calib.to_dict()
                if self.alignment_inference_vs_calib else None
            ),
            'alignment_inference_flags': self.alignment_inference_flags,
            'coverage_synth_vs_real': (
                self.coverage_synth_vs_real.to_dict()
                if self.coverage_synth_vs_real else None
            ),
            'coverage_synth_vs_inference': (
                self.coverage_synth_vs_inference.to_dict()
                if self.coverage_synth_vs_inference else None
            ),
            'coverage_real_flags': self.coverage_real_flags,
            'coverage_inference_flags': self.coverage_inference_flags,
            'joint_coverage_real': self.joint_coverage_real,
            'joint_coverage_inference': self.joint_coverage_inference,
            'diagnostics': self.diagnostics,
        }
        return d


# ── Fingerprint extraction loops (use the IO iterators) ─────────────────


def fingerprint_synthetic(
    dataset: SyntheticDataset,
    progress: ProgressCallback = None,
    skip_uniformity: bool = False,
    skip_symmetry: bool = False,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Compute fingerprints for every sample in the loaded subset.

    Returns a DataFrame with one row per sample and one column per metric,
    plus pass-through metadata columns (`defocus_mm`, `sigma_px`, `camera`).
    """
    # Cache check
    if cache_dir is not None and not force_recompute:
        cp = cache_path_for(
            cache_dir, 'synthetic', dataset.root, dataset.sample_count_used)
        if is_cache_valid(cp, dataset.root):
            if progress:
                progress(f"Synthetic cache hit ({cp.name})", 1.0)
            return load_cache(cp)
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
    df = pd.DataFrame(rows)
    if cache_dir is not None and not df.empty:
        cp = cache_path_for(
            cache_dir, 'synthetic', dataset.root, dataset.sample_count_used)
        save_cache(df, cp)
    return df


def fingerprint_crop_folder(
    folder: CropFolder,
    progress: ProgressCallback = None,
    skip_uniformity: bool = False,
    skip_symmetry: bool = False,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Compute fingerprints for every PNG in a CropFolder (real or inference).

    Returns a DataFrame with one row per crop. NaN-padded when an image
    fails to load.
    """
    if cache_dir is not None and not force_recompute:
        cp = cache_path_for(
            cache_dir, folder.label, folder.root, folder.sample_count_used)
        if is_cache_valid(cp, folder.root):
            if progress:
                progress(f"{folder.label.capitalize()} cache hit ({cp.name})", 1.0)
            return load_cache(cp)
    rows = []
    n = folder.sample_count_used
    for i, (sample, image) in enumerate(iterate_crop_folder(folder)):
        if progress and (i % max(1, n // 50) == 0):
            progress(f"{folder.label.capitalize()} fingerprint {i+1}/{n}",
                     i / max(1, n))
        rec = compute_fingerprint(
            image,
            skip_uniformity=skip_uniformity,
            skip_symmetry=skip_symmetry,
        )
        row = rec.to_dict()
        row['source_path'] = str(sample.file_path)
        row['source_type'] = folder.label
        row['index'] = i
        row['defocus_mm'] = (
            sample.defocus_mm if sample.defocus_mm is not None else float('nan'))
        row.pop('metadata', None)
        rows.append(row)
    if progress:
        progress(f"{folder.label.capitalize()} done", 1.0)
    df = pd.DataFrame(rows)
    if cache_dir is not None and not df.empty:
        cp = cache_path_for(
            cache_dir, folder.label, folder.root, folder.sample_count_used)
        save_cache(df, cp)
    return df


def fingerprint_calibration(
    stack,
    progress: ProgressCallback = None,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Compute fingerprints for every calibration frame.

    Returns a DataFrame with one row per frame. NaN-padded if the .cine
    can't be loaded (e.g. pyphantom missing on this machine).
    """
    if cache_dir is not None and not force_recompute:
        cp = cache_path_for(cache_dir, 'calibration', stack.root,
                            n_samples=len(stack.frames),
                            extra_key=f"focus{stack.focus_offset_mm}")
        if is_cache_valid(cp, stack.root):
            if progress:
                progress(f"Calibration cache hit ({cp.name})", 1.0)
            return load_cache(cp)
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
    df = pd.DataFrame(rows)
    if cache_dir is not None and not df.empty:
        cp = cache_path_for(cache_dir, 'calibration', stack.root,
                            n_samples=len(stack.frames),
                            extra_key=f"focus{stack.focus_offset_mm}")
        save_cache(df, cp)
    return df


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
    real_crops_path: Optional[Path] = None,
    inference_crops_path: Optional[Path] = None,
    n_synthetic_samples: Optional[int] = None,
    n_real_samples: Optional[int] = None,
    n_inference_samples: Optional[int] = None,
    calibration_focus_offset_mm: float = 0.0,
    k_neighbours: int = 20,
    progress: ProgressCallback = None,
    skip_uniformity: bool = False,
    skip_symmetry: bool = False,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> AllChecksResult:
    """Run all available checks end-to-end.

    Either ``config_path`` or ``config_dict`` must be given (used for Check A).
    ``synthetic_dataset_path`` is required for any image-based check.

    Optional inputs:
      - ``calibration_path`` enables synthetic↔calibration alignment AND
        inference↔calibration alignment (when inference_crops_path also given)
      - ``real_crops_path`` enables synth↔real distribution coverage (Check C)
      - ``inference_crops_path`` enables synth↔inference distribution coverage
        AND inference↔calibration alignment (with calibration also given)
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
        cache_dir=cache_dir, force_recompute=force_recompute,
    )

    sigma_trends = synthetic_sigma_trend_correlations(synth_fp)

    # ── Optional: load calibration ──────────────────────────────────────
    calib_stack = None
    calib_fp = pd.DataFrame()
    if calibration_path is not None:
        if progress:
            progress("Loading calibration z-stack", 0.55)
        calib_stack = load_calibration_stack(
            Path(calibration_path),
            focus_offset_mm=calibration_focus_offset_mm,
        )
        calib_fp = fingerprint_calibration(
            calib_stack,
            progress=lambda m, f: progress(m, 0.55 + f * 0.15) if progress else None,
            cache_dir=cache_dir, force_recompute=force_recompute,
        )
        if calib_fp.empty:
            diagnostics.append(
                "Calibration fingerprint DataFrame is empty — likely "
                "pyphantom missing or all .cine reads failed.")

    # ── Optional: load real crops ───────────────────────────────────────
    real_fp = pd.DataFrame()
    real_root = None
    if real_crops_path is not None:
        if progress:
            progress("Loading real crops", 0.70)
        real_folder = load_crop_folder(
            Path(real_crops_path), label='real', n_samples=n_real_samples)
        real_root = str(real_folder.root)
        real_fp = fingerprint_crop_folder(
            real_folder,
            progress=lambda m, f: progress(m, 0.70 + f * 0.10) if progress else None,
            skip_uniformity=skip_uniformity, skip_symmetry=skip_symmetry,
            cache_dir=cache_dir, force_recompute=force_recompute,
        )

    # ── Optional: load inference crops ──────────────────────────────────
    inference_fp = pd.DataFrame()
    inference_root = None
    if inference_crops_path is not None:
        if progress:
            progress("Loading inference crops", 0.80)
        inf_folder = load_crop_folder(
            Path(inference_crops_path), label='inference',
            n_samples=n_inference_samples)
        inference_root = str(inf_folder.root)
        inference_fp = fingerprint_crop_folder(
            inf_folder,
            progress=lambda m, f: progress(m, 0.80 + f * 0.10) if progress else None,
            skip_uniformity=skip_uniformity, skip_symmetry=skip_symmetry,
            cache_dir=cache_dir, force_recompute=force_recompute,
        )

    # ── Check B: synthetic ↔ calibration alignment ──────────────────────
    alignment_result = None
    alignment_flags = {}
    if not calib_fp.empty:
        if progress:
            progress("Check B — synthetic↔calibration alignment", 0.90)
        alignment_result = alignment_check_nn(
            anchor_df=calib_fp,
            other_df=synth_fp,
            anchor_label='calibration',
            other_label='synthetic',
            k=k_neighbours,
        )
        alignment_flags = flag_alignment(alignment_result)

    # ── Check B (Phase 2): inference ↔ calibration alignment ─────────────
    inf_alignment_result = None
    inf_alignment_flags = {}
    if not calib_fp.empty and not inference_fp.empty:
        # Filter to inference samples that have a defocus value
        inf_with_z = inference_fp[inference_fp['defocus_mm'].notna()]
        if len(inf_with_z) > 0:
            if progress:
                progress("Check B — inference↔calibration alignment", 0.93)
            inf_alignment_result = alignment_check_nn(
                anchor_df=calib_fp,
                other_df=inf_with_z,
                anchor_label='calibration',
                other_label='inference',
                k=min(k_neighbours, len(inf_with_z)),
            )
            inf_alignment_flags = flag_alignment(inf_alignment_result)
        else:
            diagnostics.append(
                "Inference crops have no parseable defocus from filename — "
                "skipping inference↔calibration alignment.")

    # ── Check C: synthetic ↔ real / inference distribution coverage ─────
    coverage_real = None
    coverage_real_flags = {}
    joint_real = {}
    if not real_fp.empty and not synth_fp.empty:
        if progress:
            progress("Check C — synth↔real coverage", 0.96)
        coverage_real = coverage_check(
            reference_df=synth_fp,
            test_df=real_fp,
            reference_label='synthetic',
            test_label='real',
        )
        coverage_real_flags = flag_coverage(coverage_real)
        joint_real = joint_coverage(synth_fp, real_fp)

    coverage_inference = None
    coverage_inference_flags = {}
    joint_inference = {}
    if not inference_fp.empty and not synth_fp.empty:
        if progress:
            progress("Check C — synth↔inference coverage", 0.98)
        coverage_inference = coverage_check(
            reference_df=synth_fp,
            test_df=inference_fp,
            reference_label='synthetic',
            test_label='inference',
        )
        coverage_inference_flags = flag_coverage(coverage_inference)
        joint_inference = joint_coverage(synth_fp, inference_fp)

    if progress:
        progress("Done", 1.0)

    return AllChecksResult(
        config_summary={'config_keys': list(config_dict.keys())},
        synthetic_dataset_root=str(synth_dataset.root),
        calibration_root=str(calib_stack.root) if calib_stack else None,
        synthetic_fingerprints=synth_fp,
        calibration_fingerprints=calib_fp,
        real_fingerprints=real_fp,
        inference_fingerprints=inference_fp,
        scale_chain=scale_result,
        sigma_trend_correlations=sigma_trends,
        alignment_synth_vs_calib=alignment_result,
        alignment_flags=alignment_flags,
        alignment_inference_vs_calib=inf_alignment_result,
        alignment_inference_flags=inf_alignment_flags,
        coverage_synth_vs_real=coverage_real,
        coverage_real_flags=coverage_real_flags,
        coverage_synth_vs_inference=coverage_inference,
        coverage_inference_flags=coverage_inference_flags,
        joint_coverage_real=joint_real,
        joint_coverage_inference=joint_inference,
        diagnostics=diagnostics,
    )
