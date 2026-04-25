"""Persistence: write JSON, markdown, fingerprint CSVs, and plot PNGs.

Only this module touches the output filesystem on the tool's behalf.
GUI / orchestrator can call its functions to materialise reports.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .fingerprint_plots import (
    plot_alignment_per_anchor_heatmap,
    plot_alignment_per_metric_deltas,
    plot_coverage_bars,
    plot_scale_chain_residuals,
    plot_sigma_trends,
    save_figure,
)


def _json_safe(obj):
    """Recursively coerce a dict to JSON-serialisable scalars."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if isinstance(v, float) and (v != v or v == float('inf') or v == -float('inf')):
            return None
        return v
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == -float('inf'):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def write_json_report(result, output_dir: Path) -> Path:
    """Write blur_fingerprint_report.json with the full result tree."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / 'blur_fingerprint_report.json'
    payload = _json_safe(result.to_dict())
    payload['written_at'] = datetime.now().isoformat(timespec='seconds')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    return out


def write_markdown_report(result, output_dir: Path) -> Path:
    """Write blur_fingerprint_report.md, human-readable summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / 'blur_fingerprint_report.md'
    lines = []
    ts = datetime.now().isoformat(timespec='seconds')
    lines.append(f"# Blur / Domain Fingerprint Report")
    lines.append("")
    lines.append(f"_Generated {ts}_")
    lines.append("")

    # Inputs
    lines.append("## Inputs")
    lines.append(f"- Synthetic dataset: `{result.synthetic_dataset_root or '(none)'}`")
    lines.append(f"- Calibration stack: `{result.calibration_root or '(none)'}`")
    lines.append(f"- Synthetic samples fingerprinted: "
                 f"{len(result.synthetic_fingerprints)}")
    lines.append(f"- Calibration frames fingerprinted: "
                 f"{len(result.calibration_fingerprints)}")
    lines.append("")

    # Check A
    lines.append("## Check A — Scale-chain round-trip")
    if result.scale_chain is None:
        lines.append("_(skipped — no config provided)_")
    else:
        sc = result.scale_chain
        flag = "✅ PASS" if sc.overall_passed else "❌ FAIL"
        lines.append(f"**Status:** {flag}")
        lines.append("")
        lines.append("Configuration:")
        lines.append("```")
        for k, v in sc.config_summary.items():
            lines.append(f"  {k}: {v}")
        lines.append("```")
        if sc.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            for d in sc.diagnostics:
                lines.append(f"- {d}")
        lines.append("")
        lines.append("| z_in (mm) | sigma_model | label | z_recovered | |Δ| (mm) | status |")
        lines.append("|---|---|---|---|---|---|")
        for p in sc.points:
            status = "pass" if p.passed else "**FAIL**"
            lines.append(
                f"| {p.defocus_in_mm:.3f} | {p.sigma_model_px:.4f} | "
                f"{p.label:.4f} | {p.defocus_recovered_mm:.4f} | "
                f"{p.delta_mm:.6f} | {status} |"
            )
        lines.append("")

    # Check B-internal: sigma trend
    if result.sigma_trend_correlations:
        lines.append("## Check B-internal — Synthetic sigma trends")
        lines.append("")
        lines.append("Pearson correlation between metadata σ_px and measured features.")
        lines.append("Expected direction: + means feature should grow with σ; − means decrease.")
        lines.append("")
        lines.append("| Metric | n | Pearson r | expected | flag |")
        lines.append("|---|---|---|---|---|")
        for m, info in result.sigma_trend_correlations.items():
            r = info['pearson_r']
            expected = info['expected_direction']
            ok = (
                (expected == '+' and r > 0.7) or
                (expected == '-' and r < -0.7)
            )
            flag = "PASS" if ok else ("FAIL" if abs(r) < 0.5 else "WARN")
            lines.append(f"| {m} | {info['n_finite']} | {r:+.3f} | {expected} | {flag} |")
        lines.append("")

    # Check B: synthetic vs calibration
    lines.append("## Check B — Synthetic ↔ Calibration alignment")
    if result.alignment_synth_vs_calib is None:
        lines.append("_(skipped — calibration not provided)_")
    else:
        ar = result.alignment_synth_vs_calib
        lines.append(
            f"K = {ar.k_neighbours} nearest synthetic samples per calibration "
            f"anchor (matched on |defocus_mm|).")
        lines.append("")
        if ar.diagnostics:
            for d in ar.diagnostics:
                lines.append(f"- {d}")
            lines.append("")
        lines.append("| Metric | n | mean Δ | median Δ | |mean Δ| | flag |")
        lines.append("|---|---|---|---|---|---|")
        for m, summary in ar.per_metric_summary.items():
            flag = result.alignment_flags.get(m, 'NODATA')
            lines.append(
                f"| {m} | {summary['n_finite']} | "
                f"{summary['mean_delta']:+.4f} | "
                f"{summary['median_delta']:+.4f} | "
                f"{summary['abs_mean_delta']:.4f} | {flag} |"
            )
        lines.append("")

    if result.diagnostics:
        lines.append("## Diagnostics")
        for d in result.diagnostics:
            lines.append(f"- {d}")
        lines.append("")

    out.write_text("\n".join(lines), encoding='utf-8')
    return out


def write_fingerprint_csvs(result, output_dir: Path) -> dict:
    """Save the synthetic + calibration fingerprint DataFrames as CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if not result.synthetic_fingerprints.empty:
        p = output_dir / 'fingerprints_synthetic.csv'
        result.synthetic_fingerprints.to_csv(p, index=False)
        written['synthetic'] = p
    if not result.calibration_fingerprints.empty:
        p = output_dir / 'fingerprints_calibration.csv'
        result.calibration_fingerprints.to_csv(p, index=False)
        written['calibration'] = p
    return written


def write_plots(result, output_dir: Path) -> dict:
    """Save all relevant plots as PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if result.scale_chain is not None:
        fig = plot_scale_chain_residuals(result.scale_chain)
        p = output_dir / 'check_a_scale_residuals.png'
        save_figure(fig, p)
        written['scale_residuals'] = p
    if not result.synthetic_fingerprints.empty:
        fig = plot_sigma_trends(
            result.synthetic_fingerprints,
            correlations=result.sigma_trend_correlations)
        p = output_dir / 'check_b_internal_sigma_trends.png'
        save_figure(fig, p)
        written['sigma_trends'] = p
    if result.alignment_synth_vs_calib is not None:
        fig = plot_alignment_per_metric_deltas(
            result.alignment_synth_vs_calib, result.alignment_flags)
        p = output_dir / 'check_b_alignment_per_metric.png'
        save_figure(fig, p)
        written['alignment_per_metric'] = p
        fig = plot_alignment_per_anchor_heatmap(result.alignment_synth_vs_calib)
        p = output_dir / 'check_b_alignment_heatmap.png'
        save_figure(fig, p)
        written['alignment_heatmap'] = p
    return written


def write_full_report(result, output_dir: Path) -> dict:
    """Write all outputs in one go: JSON + Markdown + CSVs + plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'json': write_json_report(result, output_dir),
        'markdown': write_markdown_report(result, output_dir),
    }
    files.update(write_fingerprint_csvs(result, output_dir))
    files.update(write_plots(result, output_dir))
    return files
