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
    def _alignment_table(ar, flags):
        if ar is None:
            return ["_(skipped)_", ""]
        out = [
            f"K = {ar.k_neighbours} nearest {ar.other_label} samples per "
            f"{ar.anchor_label} anchor (matched on |defocus_mm|).",
            "",
        ]
        if ar.diagnostics:
            for d in ar.diagnostics:
                out.append(f"- {d}")
            out.append("")
        out.append("| Metric | n | mean Δ | median Δ | |mean Δ| | flag |")
        out.append("|---|---|---|---|---|---|")
        for m, summary in ar.per_metric_summary.items():
            flag = flags.get(m, 'NODATA')
            out.append(
                f"| {m} | {summary['n_finite']} | "
                f"{summary['mean_delta']:+.4f} | "
                f"{summary['median_delta']:+.4f} | "
                f"{summary['abs_mean_delta']:.4f} | {flag} |"
            )
        out.append("")
        return out

    lines.append("## Check B — Synthetic ↔ Calibration alignment")
    lines.extend(_alignment_table(
        result.alignment_synth_vs_calib, result.alignment_flags))

    lines.append("## Check B — Inference ↔ Calibration alignment")
    lines.extend(_alignment_table(
        result.alignment_inference_vs_calib, result.alignment_inference_flags))

    # Check C — distribution coverage
    def _coverage_table(cov, flags):
        if cov is None:
            return ["_(skipped)_", ""]
        out = [
            f"For each metric: % of {cov.test_label} samples falling within "
            f"the {cov.reference_label} distribution's "
            f"[p{cov.percentile_low}, p{cov.percentile_high}] range. "
            f"({cov.n_test} test, {cov.n_reference} reference)",
            "",
            "| Metric | reference range [lo, hi] | n_in / n_total | coverage % | flag |",
            "|---|---|---|---|---|",
        ]
        for m, info in cov.per_feature.items():
            flag = flags.get(m, 'NODATA')
            out.append(
                f"| {m} | "
                f"[{info['p_lo']:.4f}, {info['p_hi']:.4f}] | "
                f"{info['n_in']} / {info['n_total']} | "
                f"{info['coverage_pct']:.1f}% | {flag} |"
            )
        out.append("")
        return out

    lines.append("## Check C — Synthetic ↔ Real distribution coverage")
    lines.extend(_coverage_table(
        result.coverage_synth_vs_real, result.coverage_real_flags))

    lines.append("## Check C — Synthetic ↔ Inference distribution coverage")
    lines.extend(_coverage_table(
        result.coverage_synth_vs_inference, result.coverage_inference_flags))

    if result.diagnostics:
        lines.append("## Diagnostics")
        for d in result.diagnostics:
            lines.append(f"- {d}")
        lines.append("")

    out.write_text("\n".join(lines), encoding='utf-8')
    return out


def write_fingerprint_csvs(result, output_dir: Path) -> dict:
    """Save the per-source fingerprint DataFrames as CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for label, df in (
        ('synthetic', result.synthetic_fingerprints),
        ('calibration', result.calibration_fingerprints),
        ('real', result.real_fingerprints),
        ('inference', result.inference_fingerprints),
    ):
        if not df.empty:
            p = output_dir / f'fingerprints_{label}.csv'
            df.to_csv(p, index=False)
            written[label] = p
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
        p = output_dir / 'check_b_synth_vs_calib_per_metric.png'
        save_figure(fig, p)
        written['synth_vs_calib_per_metric'] = p
        fig = plot_alignment_per_anchor_heatmap(result.alignment_synth_vs_calib)
        p = output_dir / 'check_b_synth_vs_calib_heatmap.png'
        save_figure(fig, p)
        written['synth_vs_calib_heatmap'] = p
    if result.alignment_inference_vs_calib is not None:
        fig = plot_alignment_per_metric_deltas(
            result.alignment_inference_vs_calib,
            result.alignment_inference_flags)
        p = output_dir / 'check_b_inference_vs_calib_per_metric.png'
        save_figure(fig, p)
        written['inference_vs_calib_per_metric'] = p
        fig = plot_alignment_per_anchor_heatmap(
            result.alignment_inference_vs_calib)
        p = output_dir / 'check_b_inference_vs_calib_heatmap.png'
        save_figure(fig, p)
        written['inference_vs_calib_heatmap'] = p
    if result.coverage_synth_vs_real is not None:
        fig = plot_coverage_bars(
            result.coverage_synth_vs_real, result.coverage_real_flags)
        p = output_dir / 'check_c_synth_vs_real_coverage.png'
        save_figure(fig, p)
        written['synth_vs_real_coverage'] = p
    if result.coverage_synth_vs_inference is not None:
        fig = plot_coverage_bars(
            result.coverage_synth_vs_inference,
            result.coverage_inference_flags)
        p = output_dir / 'check_c_synth_vs_inference_coverage.png'
        save_figure(fig, p)
        written['synth_vs_inference_coverage'] = p
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
