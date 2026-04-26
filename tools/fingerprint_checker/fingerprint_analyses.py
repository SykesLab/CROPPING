"""Cross-pipeline fingerprint analyses.

Check B — image-domain alignment between two pipelines, anchored on the
SMALLER pipeline (typically calibration with ~60 frames vs synthetic with
~100k). For each anchor sample, find its K nearest neighbours in the
LARGER pipeline by |defocus_mm|, average their fingerprints, compare to
the anchor's own fingerprint, report per-feature deltas.

Check C — for each fingerprint feature, count how many "test" samples
fall inside the [5th, 95th] percentile range of the "reference" samples.
Per-feature percentile coverage. Multivariate version is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .fingerprint_metrics import (
    ALL_NUMERIC_METRICS,
    SUBJECT_INDEPENDENT_METRICS,
)


# Metrics used in Check B (cross-pipeline alignment) — exclude string
# polarity since it's categorical, not delta-able
_ALIGNMENT_NUMERIC_METRICS = tuple(
    m for m in SUBJECT_INDEPENDENT_METRICS if m != 'polarity'
)


# ── Check B — alignment via calibration-anchored NN ──────────────────────


@dataclass
class AnchorComparison:
    """One anchor (typically a calibration frame) compared against its K
    nearest neighbours in the larger pipeline."""
    anchor_index: int
    anchor_defocus_mm: float
    anchor_fingerprint: dict             # one column per metric
    neighbour_indices: List[int]         # indices into larger pipeline's df
    neighbour_defocus_mean_mm: float
    neighbour_fingerprint_mean: dict     # averaged across K neighbours
    deltas: dict                         # neighbour_mean - anchor, per metric


@dataclass
class AlignmentResult:
    """Full alignment result for one pipeline pair."""
    anchor_label: str                    # 'calibration'
    other_label: str                     # 'synthetic'
    k_neighbours: int
    comparisons: List[AnchorComparison]
    per_metric_summary: Dict[str, dict] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'anchor_label': self.anchor_label,
            'other_label': self.other_label,
            'k_neighbours': self.k_neighbours,
            'comparisons': [
                {
                    'anchor_index': c.anchor_index,
                    'anchor_defocus_mm': c.anchor_defocus_mm,
                    'neighbour_defocus_mean_mm': c.neighbour_defocus_mean_mm,
                    'neighbour_indices': c.neighbour_indices,
                    'deltas': c.deltas,
                }
                for c in self.comparisons
            ],
            'per_metric_summary': self.per_metric_summary,
            'diagnostics': self.diagnostics,
        }


def alignment_check_nn(
    anchor_df: pd.DataFrame,
    other_df: pd.DataFrame,
    *,
    anchor_label: str = 'calibration',
    other_label: str = 'synthetic',
    k: int = 20,
    defocus_col: str = 'defocus_mm',
    metrics: Sequence[str] = _ALIGNMENT_NUMERIC_METRICS,
) -> AlignmentResult:
    """Calibration-anchored nearest-neighbour alignment.

    For each row in `anchor_df`, find the K rows in `other_df` whose
    `|defocus_mm|` are closest to the anchor's `|defocus_mm|`, average
    their numeric fingerprints, compute per-feature delta vs the anchor.

    Both DataFrames must have one column per metric in `metrics` plus a
    `defocus_mm` column. NaN values in either side propagate to NaN deltas
    (kept for interpretation rather than dropped silently).
    """
    diagnostics: List[str] = []
    if defocus_col not in anchor_df.columns:
        diagnostics.append(f"anchor_df missing {defocus_col}")
        return AlignmentResult(anchor_label, other_label, k, [],
                               diagnostics=diagnostics)
    if defocus_col not in other_df.columns:
        diagnostics.append(f"other_df missing {defocus_col}")
        return AlignmentResult(anchor_label, other_label, k, [],
                               diagnostics=diagnostics)

    other_abs_z = other_df[defocus_col].abs().to_numpy()
    other_indices = other_df.index.to_numpy()

    comparisons: List[AnchorComparison] = []
    for ai, arow in anchor_df.iterrows():
        anchor_z = float(arow[defocus_col])
        if not np.isfinite(anchor_z):
            continue
        # K-nearest by |z| distance — argsort and take first k
        dist = np.abs(other_abs_z - abs(anchor_z))
        k_eff = min(k, len(other_abs_z))
        if k_eff == 0:
            continue
        nn_pos = np.argsort(dist)[:k_eff]
        nn_idx = other_indices[nn_pos]
        nn_rows = other_df.loc[nn_idx]
        # Averages — nanmean so missing measurements don't poison the bin
        nn_fp = {}
        deltas = {}
        anchor_fp = {}
        for m in metrics:
            if m not in anchor_df.columns or m not in other_df.columns:
                continue
            a_val = float(arow[m]) if pd.notna(arow[m]) else float('nan')
            anchor_fp[m] = a_val
            with np.errstate(invalid='ignore'):
                n_mean = float(np.nanmean(nn_rows[m].to_numpy(dtype=float)))
            nn_fp[m] = n_mean
            deltas[m] = n_mean - a_val
        comparisons.append(AnchorComparison(
            anchor_index=int(ai),
            anchor_defocus_mm=anchor_z,
            anchor_fingerprint=anchor_fp,
            neighbour_indices=[int(i) for i in nn_idx],
            neighbour_defocus_mean_mm=float(
                np.nanmean(other_df.loc[nn_idx, defocus_col].abs())),
            neighbour_fingerprint_mean=nn_fp,
            deltas=deltas,
        ))

    # Per-metric summary across all anchors
    per_metric: Dict[str, dict] = {}
    for m in metrics:
        if m not in anchor_df.columns or m not in other_df.columns:
            continue
        all_deltas = np.array([c.deltas.get(m, np.nan)
                               for c in comparisons], dtype=float)
        finite = all_deltas[np.isfinite(all_deltas)]
        per_metric[m] = {
            'n_finite': int(len(finite)),
            'mean_delta': float(finite.mean()) if len(finite) else float('nan'),
            'median_delta': float(np.median(finite)) if len(finite) else float('nan'),
            'std_delta': float(finite.std()) if len(finite) else float('nan'),
            'abs_mean_delta': float(np.mean(np.abs(finite))) if len(finite) else float('nan'),
        }

    return AlignmentResult(
        anchor_label=anchor_label,
        other_label=other_label,
        k_neighbours=k,
        comparisons=comparisons,
        per_metric_summary=per_metric,
        diagnostics=diagnostics,
    )


# ── Check C — distribution coverage ──────────────────────────────────────


@dataclass
class CoverageResult:
    """Per-feature coverage of test samples within reference's [p_lo, p_hi]."""
    reference_label: str               # 'synthetic'
    test_label: str                    # 'real'
    percentile_low: float              # default 5
    percentile_high: float             # default 95
    n_reference: int
    n_test: int
    per_feature: Dict[str, dict] = field(default_factory=dict)  # see structure below

    def to_dict(self) -> dict:
        return {
            'reference_label': self.reference_label,
            'test_label': self.test_label,
            'percentile_low': self.percentile_low,
            'percentile_high': self.percentile_high,
            'n_reference': self.n_reference,
            'n_test': self.n_test,
            'per_feature': self.per_feature,
        }


def coverage_check(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    reference_label: str = 'synthetic',
    test_label: str = 'real',
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    metrics: Sequence[str] = ALL_NUMERIC_METRICS,
) -> CoverageResult:
    """For each feature, fraction of test samples inside the reference's
    [p_lo, p_hi] percentile range.

    NaN values are excluded from BOTH percentile estimation and coverage
    counting. A feature with too few finite reference values produces NaN
    bounds and a coverage of NaN for that feature.
    """
    out: Dict[str, dict] = {}
    for m in metrics:
        if m not in reference_df.columns or m not in test_df.columns:
            continue
        ref = reference_df[m].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        test = test_df[m].to_numpy(dtype=float)
        test = test[np.isfinite(test)]
        if len(ref) < 5 or len(test) == 0:
            out[m] = {
                'p_lo': float('nan'), 'p_hi': float('nan'),
                'coverage_pct': float('nan'),
                'n_in': 0, 'n_total': int(len(test)),
            }
            continue
        lo = float(np.percentile(ref, percentile_low))
        hi = float(np.percentile(ref, percentile_high))
        n_in = int(((test >= lo) & (test <= hi)).sum())
        out[m] = {
            'p_lo': lo, 'p_hi': hi,
            'coverage_pct': 100.0 * n_in / len(test),
            'n_in': n_in, 'n_total': int(len(test)),
        }
    return CoverageResult(
        reference_label=reference_label,
        test_label=test_label,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        n_reference=len(reference_df),
        n_test=len(test_df),
        per_feature=out,
    )


# ── Pass/Warn/Fail flagging ──────────────────────────────────────────────


def flag_alignment(
    result: AlignmentResult,
    abs_delta_warn_pct: float = 10.0,
    abs_delta_fail_pct: float = 25.0,
) -> Dict[str, str]:
    """Per-feature PASS/WARN/FAIL based on |mean_delta| / |mean| of anchor."""
    flags = {}
    for m, summary in result.per_metric_summary.items():
        anchor_means = [c.anchor_fingerprint.get(m, np.nan)
                        for c in result.comparisons]
        anchor_means = np.array([v for v in anchor_means if np.isfinite(v)])
        if len(anchor_means) == 0:
            flags[m] = 'NODATA'
            continue
        ref_scale = max(abs(np.nanmean(anchor_means)), 1e-9)
        rel_pct = 100.0 * abs(summary['mean_delta']) / ref_scale
        if not np.isfinite(rel_pct):
            flags[m] = 'NODATA'
        elif rel_pct >= abs_delta_fail_pct:
            flags[m] = 'FAIL'
        elif rel_pct >= abs_delta_warn_pct:
            flags[m] = 'WARN'
        else:
            flags[m] = 'PASS'
    return flags


def flag_coverage(
    result: CoverageResult,
    coverage_warn_below_pct: float = 80.0,
    coverage_fail_below_pct: float = 50.0,
) -> Dict[str, str]:
    """Per-feature PASS/WARN/FAIL based on coverage percentage."""
    flags = {}
    for m, info in result.per_feature.items():
        cov = info['coverage_pct']
        if not np.isfinite(cov):
            flags[m] = 'NODATA'
        elif cov < coverage_fail_below_pct:
            flags[m] = 'FAIL'
        elif cov < coverage_warn_below_pct:
            flags[m] = 'WARN'
        else:
            flags[m] = 'PASS'
    return flags


# ── Multivariate joint coverage (Phase 4 supplement to Check C) ──────────


def joint_coverage(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Sequence[str] = ALL_NUMERIC_METRICS,
    k: int = 5,
) -> Dict[str, float]:
    """Joint k-NN coverage of test in the reference's feature space.

    Per-feature percentile coverage misses combinations: a test sample whose
    individual features all fall in the reference's marginal ranges may
    still be in a region of feature space the reference never visits. This
    function quantifies that.

    Algorithm:
      1. z-score reference + test by reference's mean/std per feature
      2. Compute each REF sample's mean distance to its k nearest REF
         neighbours → defines a typical local-density scale
      3. For each TEST sample, compute its mean distance to k nearest
         reference neighbours
      4. "Covered" = test's k-NN distance ≤ 95th percentile of ref's own
      5. Report joint_coverage_pct = % of test samples covered.

    Returns a dict with: 'joint_coverage_pct', 'n_features_used',
    'n_test_finite', 'n_reference_finite', 'reference_p95_distance'.
    """
    feature_cols = [
        m for m in metrics
        if m in reference_df.columns and m in test_df.columns
    ]
    if not feature_cols:
        return {
            'joint_coverage_pct': float('nan'), 'n_features_used': 0,
            'n_test_finite': 0, 'n_reference_finite': 0,
            'reference_p95_distance': float('nan'),
        }
    ref = reference_df[feature_cols].to_numpy(dtype=float)
    test = test_df[feature_cols].to_numpy(dtype=float)

    means = np.nanmean(ref, axis=0)
    stds = np.nanstd(ref, axis=0)
    stds = np.where((stds < 1e-9) | ~np.isfinite(stds), 1.0, stds)

    ref_z = (ref - means) / stds
    test_z = (test - means) / stds

    # Drop rows with any NaN in EITHER side (joint comparison needs finite all)
    ref_finite_mask = np.all(np.isfinite(ref_z), axis=1)
    test_finite_mask = np.all(np.isfinite(test_z), axis=1)
    ref_clean = ref_z[ref_finite_mask]
    test_clean = test_z[test_finite_mask]

    if len(ref_clean) <= k or len(test_clean) == 0:
        return {
            'joint_coverage_pct': float('nan'),
            'n_features_used': len(feature_cols),
            'n_test_finite': int(test_finite_mask.sum()),
            'n_reference_finite': int(ref_finite_mask.sum()),
            'reference_p95_distance': float('nan'),
        }

    def _knn_mean_distance(query: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
        """For each query, mean Euclidean distance to k nearest ref points."""
        # n_query × n_ref distance matrix
        diff = query[:, None, :] - ref_pts[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        # Sort each row, take the k smallest (excluding self if same set)
        d.sort(axis=1)
        return d[:, :k].mean(axis=1)

    # Reference's typical k-NN distance — exclude self (column 0 = distance 0)
    ref_self_d = _knn_mean_distance(ref_clean, ref_clean)
    # Note: includes self at distance 0, so effectively averaging over k-1
    # nearest others + self. Subtract self's contribution: weight 0/k pulls
    # the mean down by 1/k * 0; correct by * k/(k-1).
    ref_self_d = ref_self_d * (k / max(k - 1, 1))
    ref_p95 = float(np.percentile(ref_self_d, 95))

    test_d = _knn_mean_distance(test_clean, ref_clean)
    n_covered = int((test_d <= ref_p95).sum())

    return {
        'joint_coverage_pct': 100.0 * n_covered / len(test_clean),
        'n_features_used': len(feature_cols),
        'n_test_finite': int(len(test_clean)),
        'n_reference_finite': int(len(ref_clean)),
        'reference_p95_distance': ref_p95,
    }


# ── Nearest-match lookup (Sample Viewer, Phase 3) ────────────────────────


def find_nearest_match(
    source_row: pd.Series,
    target_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> tuple:
    """Find the target row most similar to source_row in z-scored feature space.

    Z-scoring uses the TARGET distribution's mean and std per feature so that
    distances are unit-free regardless of metric scale. Source-NaN dimensions
    are skipped (the source's missing measurements don't count against any
    target). Target-NaN dimensions contribute zero distance for that target on
    that dimension (treats missing = no info).

    Returns ``(target_index, distance)``. Returns ``(None, None)`` if the
    source has no usable features in common with the target.
    """
    if target_df is None or target_df.empty:
        return None, None
    if feature_cols is None:
        feature_cols = [
            m for m in ALL_NUMERIC_METRICS
            if m in source_row.index and m in target_df.columns
        ]
    feature_cols = [c for c in feature_cols if c in target_df.columns]
    if not feature_cols:
        return None, None

    target_arr = target_df[feature_cols].to_numpy(dtype=float)
    means = np.nanmean(target_arr, axis=0)
    stds = np.nanstd(target_arr, axis=0)
    stds = np.where((stds < 1e-9) | ~np.isfinite(stds), 1.0, stds)

    source_vec = np.array(
        [source_row.get(c, np.nan) for c in feature_cols], dtype=float)
    finite = np.isfinite(source_vec)
    if finite.sum() == 0:
        return None, None

    src_z = (source_vec - means) / stds
    tgt_z = (target_arr - means) / stds

    src_z = src_z[finite]
    tgt_z = tgt_z[:, finite]

    diff_sq = (tgt_z - src_z) ** 2
    diff_sq = np.where(np.isnan(diff_sq), 0.0, diff_sq)
    distances = np.sqrt(diff_sq.sum(axis=1))

    best_pos = int(np.argmin(distances))
    return int(target_df.index[best_pos]), float(distances[best_pos])
