"""Matplotlib plot functions for the fingerprint checker.

Each function returns a ``Figure`` so the GUI can embed it via
``FigureCanvasTkAgg`` and the report writer can call ``fig.savefig`` to
persist a PNG. No function writes to disk on its own — that's the
caller's choice.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# Standard figure sizes for the GUI (inches)
DEFAULT_FIGSIZE = (8, 5)
LARGE_FIGSIZE = (10, 6)

# Colour palette per flag
FLAG_COLOURS = {
    'PASS': '#2ca02c',
    'WARN': '#ff7f0e',
    'FAIL': '#d62728',
    'NODATA': '#777777',
}


def plot_scale_chain_residuals(scale_result) -> "matplotlib.figure.Figure":
    """Bar chart of |delta_mm| per test defocus from Check A."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    if scale_result is None or not scale_result.points:
        ax.text(0.5, 0.5, 'No Check A result', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    z_in = [p.defocus_in_mm for p in scale_result.points]
    deltas = [p.delta_mm for p in scale_result.points]
    colours = ['#2ca02c' if p.passed else '#d62728' for p in scale_result.points]
    ax.bar(z_in, deltas, color=colours, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Test defocus z_in (mm)')
    ax.set_ylabel('|z_in − z_recovered|  (mm)')
    title = 'Check A — Scale Chain Round-Trip Residuals'
    title += '   PASS' if scale_result.overall_passed else '   FAIL'
    ax.set_title(title, fontweight='bold',
                 color='#2ca02c' if scale_result.overall_passed else '#d62728')
    ax.axhline(scale_result.config_summary.get('tolerance_mm', 0.001),
               color='gray', linestyle='--', alpha=0.5,
               label=f"tolerance = {scale_result.config_summary.get('tolerance_mm', 0.001)} mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sigma_trends(
    df: pd.DataFrame,
    metadata_sigma_col: str = 'sigma_px_metadata',
    correlations: Optional[dict] = None,
) -> "matplotlib.figure.Figure":
    """4-panel scatter: metadata sigma vs (ERF sigma, edge width,
    Laplacian variance, Tenengrad)."""
    fig, axes = plt.subplots(2, 2, figsize=LARGE_FIGSIZE)
    panels = [
        ('erf_sigma_px', 'Measured ERF σ (px)', '+', axes[0, 0]),
        ('edge_transition_width', 'Edge width 10–90% (px)', '+', axes[0, 1]),
        ('laplacian_variance', 'Laplacian variance', '−', axes[1, 0]),
        ('tenengrad', 'Tenengrad score', '−', axes[1, 1]),
    ]
    if metadata_sigma_col not in df.columns:
        for _, _, _, ax in panels:
            ax.text(0.5, 0.5, f'No {metadata_sigma_col} column',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_axis_off()
        return fig

    s = df[metadata_sigma_col].to_numpy(dtype=float)

    for col, ylabel, expected_dir, ax in panels:
        if col not in df.columns:
            ax.text(0.5, 0.5, f'No column {col}',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_axis_off()
            continue
        y = df[col].to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(y)
        ax.scatter(s[mask], y[mask], s=10, alpha=0.4, color='steelblue')
        ax.set_xlabel(f'metadata σ_px ({mask.sum()} samples)')
        ax.set_ylabel(ylabel)
        title = f"{ylabel}  vs metadata σ"
        if correlations and col in correlations:
            r = correlations[col]['pearson_r']
            title += f"   r={r:+.3f}  (expect {expected_dir})"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Synthetic sigma-trend correlations',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def plot_alignment_per_metric_deltas(
    alignment_result, flags: Optional[dict] = None,
) -> "matplotlib.figure.Figure":
    """Horizontal bar of mean |delta| per feature, coloured by flag."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    if alignment_result is None or not alignment_result.per_metric_summary:
        ax.text(0.5, 0.5, 'No alignment result', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    metrics = list(alignment_result.per_metric_summary.keys())
    abs_means = [alignment_result.per_metric_summary[m]['abs_mean_delta']
                 for m in metrics]
    if flags is None:
        flags = {}
    colours = [FLAG_COLOURS.get(flags.get(m, 'NODATA'), '#777777')
               for m in metrics]

    y = np.arange(len(metrics))
    ax.barh(y, abs_means, color=colours, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel('Mean |delta|  (synthetic−calibration, units vary)')
    ax.set_title(
        f'Check B — Alignment {alignment_result.other_label} vs '
        f'{alignment_result.anchor_label}  '
        f'(K={alignment_result.k_neighbours} neighbours)',
        fontweight='bold')
    # Legend: flag colours
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=l) for l, c in FLAG_COLOURS.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    return fig


def plot_alignment_per_anchor_heatmap(
    alignment_result,
) -> "matplotlib.figure.Figure":
    """Heatmap of (anchors × metrics) showing per-anchor delta per feature."""
    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    if alignment_result is None or not alignment_result.comparisons:
        ax.text(0.5, 0.5, 'No alignment result', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    metrics = list(alignment_result.per_metric_summary.keys())
    if not metrics:
        ax.text(0.5, 0.5, 'No metric summary', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    n_anchors = len(alignment_result.comparisons)
    matrix = np.full((n_anchors, len(metrics)), np.nan)
    for i, c in enumerate(alignment_result.comparisons):
        for j, m in enumerate(metrics):
            matrix[i, j] = c.deltas.get(m, np.nan)

    # Anchors sorted by their own |z| for readable plot
    z_order = np.argsort([abs(c.anchor_defocus_mm)
                          for c in alignment_result.comparisons])
    matrix = matrix[z_order]

    # Symmetric colour scale around 0
    finite_max = np.nanmax(np.abs(matrix)) if np.any(np.isfinite(matrix)) else 1.0
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-finite_max, vmax=finite_max,
                   interpolation='nearest')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Calibration anchor (sorted by |defocus|)')
    ax.set_title('Per-anchor deltas (synthetic − calibration)',
                 fontweight='bold')
    fig.colorbar(im, ax=ax, label='delta')
    fig.tight_layout()
    return fig


def plot_coverage_bars(
    coverage_result, flags: Optional[dict] = None,
) -> "matplotlib.figure.Figure":
    """Horizontal bars of per-feature coverage %."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    if coverage_result is None or not coverage_result.per_feature:
        ax.text(0.5, 0.5, 'No coverage result', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    metrics = list(coverage_result.per_feature.keys())
    coverages = [coverage_result.per_feature[m]['coverage_pct']
                 for m in metrics]
    if flags is None:
        flags = {}
    colours = [FLAG_COLOURS.get(flags.get(m, 'NODATA'), '#777777')
               for m in metrics]
    y = np.arange(len(metrics))
    ax.barh(y, coverages, color=colours, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel('Coverage %  '
                  f'(test inside reference [p{coverage_result.percentile_low}'
                  f'-p{coverage_result.percentile_high}])')
    ax.set_title(
        f'Check C — {coverage_result.test_label} samples covered by '
        f'{coverage_result.reference_label} distribution',
        fontweight='bold')
    ax.set_xlim(0, 100)
    ax.axvline(80, color='gray', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


# ── Convenience: save Figure to disk ────────────────────────────────────


def save_figure(fig, path, dpi: int = 150) -> None:
    """Save a Figure as a PNG. Caller passes the path."""
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
