"""
CROPPING Pipeline Diagnostic GUI

Visual diagnostics for every stage of the depth-from-defocus pipeline.
Set your data paths once at the top, then run any diagnostic to see
charts, intermediate values, and pass/fail status.

Usage:
    python test_gui.py
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _m in ("Calibration", "Training", "Preprocessing"):
    _p = str(_REPO_ROOT / _m)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════════
# Input loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_calibration_yaml(path: Path) -> Optional[Dict]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_checkpoint(path: Path) -> Optional[Dict]:
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _load_metadata(data_dir: Path):
    import pandas as pd
    csv = data_dir / "metadata.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["index"] = df["index"].astype(str).str.zfill(6)
    return df.set_index("index")


def _placeholder(msg: str) -> Tuple[Figure, str]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=13, color="gray")
    ax.axis("off")
    return fig, msg


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic functions
# ═══════════════════════════════════════════════════════════════════════════

# --- Category 1: Calibration Health ---

def diag_calibration_fit(inputs: Dict) -> Tuple[Figure, str]:
    """Calibration fit quality and parameter plausibility."""
    cal = inputs["calibration"]
    direct = cal.get("direct", {})
    rho = direct.get("rho_px_per_mm", 0)
    sigma_0 = direct.get("sigma_0", 0)
    r2 = direct.get("r_squared", 0)
    n_pts = direct.get("num_points", 0)
    loo = direct.get("loo_cv", {})

    from calibration_core import validate_calibration, CalibrationResultA
    result = CalibrationResultA(rho_px_per_mm=rho, sigma_0=sigma_0,
                                r_squared=r2, num_points=n_pts)
    is_valid, warnings = validate_calibration(result)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: reconstructed calibration curve
    z_range = np.linspace(0, 8, 100)
    sigma_fit = rho * z_range + sigma_0
    axes[0].plot(z_range, sigma_fit, "r-", linewidth=2, label=f"rho={rho:.4f}, sigma_0={sigma_0:.3f}")
    axes[0].set_xlabel("Defocus |z| (mm)")
    axes[0].set_ylabel("Blur sigma (px)")
    axes[0].set_title("Calibration Model")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: parameter plausibility
    params = ["R^2", "rho", "sigma_0", "n_points"]
    values = [r2, rho, sigma_0, n_pts]
    thresholds_good = [0.95, 0.5, 0, 10]
    thresholds_bad = [0.9, 0.1, 5, 5]
    colours = []
    for val, good, bad in zip(values, thresholds_good, thresholds_bad):
        if params[len(colours)] == "R^2":
            colours.append("#2d8a2d" if val > good else ("#cc8800" if val > bad else "#cc3333"))
        elif params[len(colours)] == "sigma_0":
            colours.append("#2d8a2d" if val < bad else "#cc8800")
        elif params[len(colours)] == "n_points":
            colours.append("#2d8a2d" if val >= good else ("#cc8800" if val >= bad else "#cc3333"))
        else:
            colours.append("#2d8a2d" if val > good else "#cc8800")

    bars = axes[1].barh(params, values, color=colours, edgecolor="black")
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_width() + 0.02 * max(values), bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}" if isinstance(val, float) else str(val),
                     va="center", fontsize=9)
    axes[1].set_title("Parameter Plausibility")
    axes[1].set_xlim(0, max(values) * 1.3)

    # Panel 3: LOO-CV uncertainty
    if loo.get("rho_std", 0) > 0:
        labels = ["rho", "sigma_0"]
        means = [rho, sigma_0]
        stds = [loo["rho_std"], loo["sigma_0_std"]]
        cvs = [s/m*100 if m > 0 else 0 for s, m in zip(stds, means)]
        axes[2].bar(labels, cvs, color="steelblue", edgecolor="black")
        for i, (cv, std, mean) in enumerate(zip(cvs, stds, means)):
            axes[2].text(i, cv + 0.5, f"{mean:.4f} +/- {std:.4f}\n(CV={cv:.1f}%)",
                         ha="center", fontsize=8)
        axes[2].set_ylabel("Coefficient of Variation (%)")
        axes[2].set_title("LOO-CV Stability")
        axes[2].grid(axis="y", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No LOO-CV data\nin calibration YAML",
                     ha="center", va="center", fontsize=12, color="gray", transform=axes[2].transAxes)
        axes[2].set_title("LOO-CV Stability")

    fig.tight_layout()
    status = "PASS" if is_valid else "FAIL"
    warn_str = "\n  ".join(warnings) if warnings else "None"
    summary = (
        f"Calibration Fit Quality\n"
        f"  rho = {rho:.4f} px/mm, sigma_0 = {sigma_0:.3f} px\n"
        f"  R^2 = {r2:.4f}, n = {n_pts} points\n"
        f"  Warnings: {warn_str}\n"
        f"  {status}"
    )
    return fig, summary


def diag_uncertainty_budget(inputs: Dict) -> Tuple[Figure, str]:
    """Calibration uncertainty propagation across defocus range."""
    from physics import ScalingParams, invert_prediction_with_uncertainty

    cal = inputs["calibration"]
    direct = cal.get("direct", {})
    rho = direct.get("rho_px_per_mm", 1.0)
    sigma_0 = direct.get("sigma_0", 0.0)
    s_cal = direct.get("scale_calib_px_per_mm", 1.0) or 1.0
    loo = direct.get("loo_cv", {})
    rho_std = loo.get("rho_std", 0.0)
    sigma_0_std = loo.get("sigma_0_std", 0.0)

    if rho_std == 0:
        return _placeholder("No LOO-CV uncertainties in calibration YAML.\n"
                            "Re-run calibration to generate them.")

    defocus_range = cal.get("defocus_range_mm", [0, 8])
    params = ScalingParams(rho=rho, sigma_0=sigma_0, s_calib=s_cal, s_inference=s_cal,
                           max_blur=rho * abs(defocus_range[1]) + sigma_0 + 2, model_size=256)

    pred_range = np.linspace(0.05, 0.95, 50)
    defocus_vals, unc_vals = [], []
    for p in pred_range:
        r = invert_prediction_with_uncertainty(p, params, 299, rho_std, sigma_0_std)
        defocus_vals.append(r.defocus_mm)
        unc_vals.append(r.defocus_uncertainty_mm)

    d = np.array(defocus_vals)
    u = np.array(unc_vals)
    rel = np.where(d > 0.01, u / d * 100, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(d, u, "b-", linewidth=1.5)
    axes[0].fill_between(d, 0, u, alpha=0.2, color="steelblue")
    axes[0].set_xlabel("Defocus (mm)")
    axes[0].set_ylabel("Uncertainty (mm)")
    axes[0].set_title("Absolute Uncertainty vs Defocus")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(d, rel, "b-", linewidth=1.5)
    axes[1].set_xlabel("Defocus (mm)")
    axes[1].set_ylabel("Relative Uncertainty (%)")
    axes[1].set_title("Relative Uncertainty (higher near focus)")
    axes[1].set_ylim(0, min(100, rel.max() * 1.2) if rel.max() > 0 else 10)
    axes[1].grid(True, alpha=0.3)

    # Depth resolution limit
    resolution_idx = np.argmax(rel < 100) if (rel < 100).any() else 0
    min_resolvable = d[resolution_idx] if resolution_idx > 0 else d[0]
    axes[2].plot(pred_range, d, "b-", linewidth=1.5, label="Defocus")
    axes[2].fill_between(pred_range, d - u, d + u, alpha=0.3, color="steelblue",
                         label="Confidence band")
    axes[2].axhline(min_resolvable, color="red", linestyle="--", linewidth=1,
                    label=f"Min resolvable: {min_resolvable:.2f} mm")
    axes[2].set_xlabel("Model prediction")
    axes[2].set_ylabel("Defocus (mm)")
    axes[2].set_title("Confidence Band")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    unc_at_2mm = np.interp(2.0, d, u)
    summary = (
        f"Uncertainty Budget\n"
        f"  rho = {rho:.4f} +/- {rho_std:.4f}, sigma_0 = {sigma_0:.3f} +/- {sigma_0_std:.3f}\n"
        f"  At 2mm defocus: +/- {unc_at_2mm:.3f} mm ({np.interp(2.0, d, rel):.1f}%)\n"
        f"  Min resolvable defocus: {min_resolvable:.2f} mm\n"
        f"  {'PASS' if unc_at_2mm < 0.5 else 'WARN'}: uncertainty at 2mm {'<' if unc_at_2mm < 0.5 else '>'} 0.5 mm"
    )
    return fig, summary


def diag_config_consistency(inputs: Dict) -> Tuple[Figure, str]:
    """Check calibration YAML vs checkpoint config for mismatches."""
    from physics import ScalingParams, defocus_to_label, label_to_defocus, SATURATION_THRESHOLD

    cal = inputs["calibration"]
    ckpt = inputs["checkpoint"]
    direct = cal.get("direct", {})
    config = ckpt.get("config", {})
    training_cfg = config.get("training", {})

    # Parameter comparison
    comparisons = {
        "rho": (direct.get("rho_px_per_mm"), training_cfg.get("rho_direct")),
        "sigma_0": (direct.get("sigma_0"), training_cfg.get("sigma_0")),
        "scale_calib": (direct.get("scale_calib_px_per_mm"), training_cfg.get("scale_calib_px_per_mm")),
        "training_mode": (cal.get("calibration_mode"), ckpt.get("training_mode")),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: parameter comparison table
    axes[0].axis("off")
    rows = []
    all_match = True
    for key, (yaml_val, ckpt_val) in comparisons.items():
        match = yaml_val == ckpt_val or (yaml_val is not None and ckpt_val is not None
                                          and abs(float(yaml_val) - float(ckpt_val)) < 1e-6
                                          if isinstance(yaml_val, (int, float)) else yaml_val == ckpt_val)
        if not match and yaml_val is not None and ckpt_val is not None:
            all_match = False
        status = "match" if match or yaml_val is None or ckpt_val is None else "MISMATCH"
        rows.append([key, str(yaml_val), str(ckpt_val), status])

    table = axes[0].table(cellText=rows,
                          colLabels=["Param", "Calib YAML", "Checkpoint", "Status"],
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for i, row in enumerate(rows):
        colour = "#d0ffd0" if row[3] == "match" else "#ffd0d0"
        for j in range(4):
            table[i + 1, j].set_facecolor(colour)
    axes[0].set_title("Parameter Consistency")

    # Panel 2: round-trip test
    rho = direct.get("rho_px_per_mm", 1.0)
    sigma_0 = direct.get("sigma_0", 0.0)
    s_cal = direct.get("scale_calib_px_per_mm", 1.0) or 1.0
    max_blur = ckpt.get("max_blur", ckpt.get("max_coc", 20.0))
    model_size = config.get("data", {}).get("image_size_px", 256)

    params = ScalingParams(rho=rho, sigma_0=sigma_0, s_calib=s_cal, s_inference=s_cal,
                           max_blur=max_blur, model_size=model_size)
    z_range = np.linspace(0.1, 8, 50)
    errors = []
    for z in z_range:
        label = defocus_to_label(z, params, 299)
        z_rec, _, _ = label_to_defocus(label, params, 299)
        errors.append(z_rec - z)

    axes[1].plot(z_range, errors, "b.", markersize=4)
    axes[1].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_xlabel("z true (mm)")
    axes[1].set_ylabel("Round-trip error (mm)")
    max_err = max(abs(e) for e in errors)
    axes[1].set_title(f"Round-Trip (max err: {max_err:.2e})")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: saturation map
    labels = [defocus_to_label(z, params, 299) for z in z_range]
    axes[2].plot(z_range, labels, "b-", linewidth=1.5)
    axes[2].axhline(SATURATION_THRESHOLD, color="red", linewidth=1, linestyle="--", alpha=0.5)
    axes[2].axhline(1 - SATURATION_THRESHOLD, color="red", linewidth=1, linestyle="--", alpha=0.5)
    axes[2].fill_between(z_range, SATURATION_THRESHOLD, 1 - SATURATION_THRESHOLD,
                         alpha=0.1, color="green", label="Trustworthy range")
    axes[2].set_xlabel("Defocus (mm)")
    axes[2].set_ylabel("Normalised label")
    axes[2].set_title("Saturation Map")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    status = "PASS" if all_match and max_err < 1e-6 else "FAIL"
    summary = (
        f"Config Consistency\n"
        f"  Parameters: {'all match' if all_match else 'MISMATCH DETECTED'}\n"
        f"  Round-trip max error: {max_err:.2e} mm\n"
        f"  max_blur: {max_blur:.2f}, model_size: {model_size}\n"
        f"  {status}"
    )
    return fig, summary


# --- Category 2: Synthetic Data Quality ---

def diag_label_distribution(inputs: Dict) -> Tuple[Figure, str]:
    """Audit training label distribution for gaps or pile-ups."""
    df = inputs["metadata"]
    blur_col = "sigma_px" if "sigma_px" in df.columns else "coc_px"
    blur_vals = df[blur_col].values
    max_blur = blur_vals.max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: raw blur histogram
    axes[0].hist(blur_vals, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel(f"{blur_col} (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Blur Distribution (n={len(blur_vals)})")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: normalised label histogram
    labels = blur_vals / max_blur
    axes[1].hist(labels, bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(0.99, color="red", linewidth=1, linestyle="--", label="Saturation zone")
    axes[1].set_xlabel("Normalised label (sigma/max)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Label Distribution [0, 1]")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: blur vs defocus
    if "defocus_mm" in df.columns:
        axes[2].scatter(df["defocus_mm"].values, blur_vals, s=5, alpha=0.3, c="steelblue")
        axes[2].set_xlabel("Defocus (mm)")
        axes[2].set_ylabel(f"{blur_col} (px)")
        axes[2].set_title("Blur vs Defocus")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No defocus_mm column", ha="center", va="center",
                     fontsize=12, color="gray", transform=axes[2].transAxes)

    fig.tight_layout()

    # Check bin coverage
    hist, edges = np.histogram(labels, bins=4)
    expected = len(labels) / 4
    min_ratio = hist.min() / expected if expected > 0 else 0
    max_ratio = hist.max() / expected if expected > 0 else 0
    empty_bins = (hist == 0).sum()

    status = "PASS" if empty_bins == 0 and min_ratio > 0.25 else "FAIL" if empty_bins > 0 else "WARN"
    summary = (
        f"Label Distribution Audit ({len(blur_vals)} samples)\n"
        f"  {blur_col} range: [{blur_vals.min():.2f}, {blur_vals.max():.2f}] px\n"
        f"  4-bin coverage: min={hist.min()}, max={hist.max()} (ratio {max_ratio:.1f}x)\n"
        f"  Empty bins: {empty_bins}\n"
        f"  {status}"
    )
    return fig, summary


def diag_blur_trace(inputs: Dict) -> Tuple[Figure, str]:
    """Audit quadrature subtraction in blur trace metadata."""
    df = inputs["metadata"]

    if "quadrature_error_pct" not in df.columns:
        return _placeholder("No blur trace columns in metadata.\n"
                            "Regenerate with save_blur_trace_metadata: true")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    qe = df["quadrature_error_pct"].dropna().values

    # Panel 1: quadrature error distribution
    axes[0].hist(qe, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Quadrature error (%)")
    axes[0].set_ylabel("Count")
    p95 = np.percentile(np.abs(qe), 95)
    axes[0].set_title(f"Quadrature Error (95th pct: {p95:.2f}%)")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: applied vs expected
    if "sigma_applied_px" in df.columns and "sigma_model_expected_px" in df.columns:
        applied = df["sigma_applied_px"].dropna().values
        expected = df["sigma_model_expected_px"].dropna().values
        n = min(len(applied), len(expected))
        axes[1].scatter(expected[:n], applied[:n], s=5, alpha=0.3, c="steelblue")
        mx = max(expected[:n].max(), applied[:n].max()) * 1.05
        axes[1].plot([0, mx], [0, mx], "r--", linewidth=1)
        axes[1].set_xlabel("Expected sigma (px)")
        axes[1].set_ylabel("Applied sigma (px)")
        axes[1].set_title("Applied vs Expected Blur")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Missing columns", ha="center", va="center",
                     fontsize=12, color="gray", transform=axes[1].transAxes)

    # Panel 3: native blur distribution
    if "native_blur_model_px" in df.columns:
        native = df["native_blur_model_px"].dropna().values
        axes[2].hist(native, bins=40, color="coral", edgecolor="black", alpha=0.7)
        axes[2].set_xlabel("Native blur at model scale (px)")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Native Blur Distribution (median={np.median(native):.2f})")
        axes[2].grid(axis="y", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Missing native_blur column", ha="center", va="center",
                     fontsize=12, color="gray", transform=axes[2].transAxes)

    fig.tight_layout()
    status = "PASS" if p95 < 1.0 else "WARN" if p95 < 5.0 else "FAIL"
    summary = (
        f"Blur Trace Audit ({len(qe)} samples)\n"
        f"  Quadrature error 95th pct: {p95:.2f}%\n"
        f"  Median: {np.median(qe):.2f}%, Max: {np.max(np.abs(qe)):.2f}%\n"
        f"  {status}: 95th pct {'<' if p95 < 1.0 else '>'} 1%"
    )
    return fig, summary


# --- Category 3: Model Quality ---

def diag_checkpoint_sanity(inputs: Dict) -> Tuple[Figure, str]:
    """Checkpoint loading, architecture, and weight health."""
    import torch
    from model import DefocusNet

    ckpt = inputs["checkpoint"]
    config = ckpt.get("config", {})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: checkpoint info table
    axes[0].axis("off")
    info = [
        ["epoch", str(ckpt.get("epoch", "?"))],
        ["training_mode", str(ckpt.get("training_mode", "?"))],
        ["max_blur", f"{ckpt.get('max_blur', ckpt.get('max_coc', '?')):.2f}"
         if isinstance(ckpt.get("max_blur"), (int, float)) else "?"],
        ["val_loss", f"{ckpt.get('val_loss', '?'):.4f}"
         if isinstance(ckpt.get("val_loss"), (int, float)) else "?"],
        ["val_mae_px", f"{ckpt.get('val_mae_px', '?'):.4f}"
         if isinstance(ckpt.get("val_mae_px"), (int, float)) else "?"],
    ]
    net_cfg = config.get("network", {}).get("dme", {})
    info.append(["initial_filters", str(net_cfg.get("initial_filters", "?"))])
    info.append(["num_res_blocks", str(net_cfg.get("num_res_blocks", "?"))])

    table = axes[0].table(cellText=info, colLabels=["Key", "Value"],
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    axes[0].set_title("Checkpoint Contents")

    # Panel 2: weight distributions
    model = DefocusNet.from_config(config)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "dme_state_dict" in ckpt:
        state = ckpt["dme_state_dict"]
        sample_key = next(iter(state))
        if not sample_key.startswith("dme_subnet."):
            state = {f"dme_subnet.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()

    all_weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            all_weights.extend(param.detach().cpu().numpy().flatten().tolist())

    axes[1].hist(all_weights, bins=100, color="steelblue", edgecolor="none", alpha=0.7)
    axes[1].set_xlabel("Weight value")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Weight Distribution ({sum(p.numel() for p in model.parameters()):,} params)")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: response to varying intensity
    intensities = np.linspace(-1, 1, 20)
    responses = []
    with torch.no_grad():
        for val in intensities:
            x = torch.full((1, 1, 256, 256), float(val))
            out = model(x).squeeze().item()
            responses.append(out)

    axes[2].plot(intensities, responses, "o-", color="steelblue", markersize=4)
    axes[2].set_xlabel("Input intensity (constant)")
    axes[2].set_ylabel("Model output")
    axes[2].set_title("Response to Flat Images")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    has_nan = any(np.isnan(w) for w in all_weights[:10000])
    in_range = all(0 <= r <= 1 for r in responses)
    has_keys = "config" in ckpt and ("dme_state_dict" in ckpt or "model_state_dict" in ckpt)
    status = "PASS" if has_keys and not has_nan and in_range else "FAIL"

    summary = (
        f"Checkpoint Sanity\n"
        f"  Epoch: {ckpt.get('epoch', '?')}, Mode: {ckpt.get('training_mode', '?')}\n"
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
        f"  Required keys present: {has_keys}\n"
        f"  NaN in weights: {has_nan}\n"
        f"  Outputs in [0,1]: {in_range}\n"
        f"  {status}"
    )
    return fig, summary


def diag_model_vs_synthetic(inputs: Dict) -> Tuple[Figure, str]:
    """Model prediction accuracy on its own synthetic training data."""
    import torch
    import cv2
    from model import DefocusNet

    ckpt = inputs["checkpoint"]
    df = inputs["metadata"]
    data_dir = inputs["synth_dir"]
    blur_dir = data_dir / "blur"
    blur_col = "sigma_px" if "sigma_px" in df.columns else "coc_px"

    config = ckpt.get("config", {})
    max_blur = ckpt.get("max_blur", ckpt.get("max_coc", 20.0))
    model_size = config.get("data", {}).get("image_size_px", 256)

    model = DefocusNet.from_config(config)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "dme_state_dict" in ckpt:
        state = ckpt["dme_state_dict"]
        sample_key = next(iter(state))
        if not sample_key.startswith("dme_subnet."):
            state = {f"dme_subnet.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()

    # Sample up to 500 crops
    sample = df.head(500)
    gt_vals, pred_vals = [], []

    with torch.no_grad():
        for idx in sample.index:
            img_path = blur_dir / f"{idx}.png"
            if not img_path.exists():
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            resized = cv2.resize(gray, (model_size, model_size))
            tensor = torch.from_numpy(resized.astype(np.float32) / 255.0 * 2.0 - 1.0
                                      ).unsqueeze(0).unsqueeze(0)
            pred_norm = model(tensor).squeeze().item()
            pred_px = pred_norm * max_blur
            gt_px = sample.loc[idx, blur_col]
            gt_vals.append(gt_px)
            pred_vals.append(pred_px)

    gt = np.array(gt_vals)
    pred = np.array(pred_vals)
    errors = pred - gt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: scatter
    max_val = max(gt.max(), pred.max()) * 1.05
    axes[0].scatter(gt, pred, s=8, alpha=0.4, c="steelblue")
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
    corr = np.corrcoef(gt, pred)[0, 1]
    axes[0].set_xlabel(f"GT {blur_col} (px)")
    axes[0].set_ylabel(f"Predicted (px)")
    axes[0].set_title(f"Predicted vs GT (R^2={corr**2:.4f})")
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: error histogram
    mae = np.mean(np.abs(errors))
    axes[1].hist(errors, bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Error (px)")
    axes[1].set_title(f"Error Distribution (MAE={mae:.3f} px)")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: binned MAE
    n_bins = 4
    bin_edges = np.linspace(gt.min(), gt.max(), n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]
    bin_maes = []
    for i in range(n_bins):
        mask = (gt >= bin_edges[i]) & (gt < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (gt >= bin_edges[i]) & (gt <= bin_edges[i + 1])
        bin_maes.append(np.mean(np.abs(errors[mask])) if mask.sum() > 0 else 0)

    axes[2].bar(range(n_bins), bin_maes, color="steelblue", edgecolor="black")
    axes[2].set_xticks(range(n_bins))
    axes[2].set_xticklabels(bin_labels, fontsize=8, rotation=20)
    axes[2].set_ylabel("MAE (px)")
    axes[2].set_title("MAE by Blur Level")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    status = "PASS" if mae < 0.5 else "WARN" if mae < 1.0 else "FAIL"
    summary = (
        f"Model vs Synthetic ({len(gt)} crops)\n"
        f"  MAE: {mae:.4f} px, RMSE: {np.sqrt(np.mean(errors**2)):.4f} px\n"
        f"  R^2: {corr**2:.4f}, Bias: {np.mean(errors):+.4f} px\n"
        f"  {status}: MAE {'<' if mae < 0.5 else '>'} 0.5 px"
    )
    return fig, summary


def diag_real_crop_monotonicity(inputs: Dict) -> Tuple[Figure, str]:
    """Check CNN predictions correlate with classical focus metrics on real crops."""
    import torch
    import cv2
    from model import DefocusNet
    from focus_metrics import compute_all_focus_metrics
    from inference_engine import boundary_normalise

    ckpt = inputs["checkpoint"]
    crops_dir = inputs["crops_dir"]
    config = ckpt.get("config", {})
    max_blur = ckpt.get("max_blur", ckpt.get("max_coc", 20.0))
    model_size = config.get("data", {}).get("image_size_px", 256)

    model = DefocusNet.from_config(config)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "dme_state_dict" in ckpt:
        state = ckpt["dme_state_dict"]
        sample_key = next(iter(state))
        if not sample_key.startswith("dme_subnet."):
            state = {f"dme_subnet.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()

    # Find crop images
    crop_files = sorted(crops_dir.rglob("*_crop.png"))
    if not crop_files:
        crop_files = sorted(crops_dir.rglob("*.png"))
    crop_files = crop_files[:200]

    cnn_preds, lap_scores = [], []
    with torch.no_grad():
        for crop_path in crop_files:
            gray = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            metrics = compute_all_focus_metrics(gray)
            lap_scores.append(metrics["laplacian_var"])

            norm = boundary_normalise(gray)
            resized = cv2.resize(norm, (model_size, model_size))
            tensor = torch.from_numpy(resized.astype(np.float32) * 2.0 - 1.0
                                      ).unsqueeze(0).unsqueeze(0)
            pred = model(tensor).squeeze().item()
            cnn_preds.append(pred * max_blur)

    cnn = np.array(cnn_preds)
    lap = np.array(lap_scores)

    from scipy.stats import spearmanr
    rho_s, p_val = spearmanr(cnn, lap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: CNN vs laplacian
    axes[0].scatter(lap, cnn, s=10, alpha=0.4, c="steelblue")
    axes[0].set_xlabel("Laplacian Variance (higher=sharper)")
    axes[0].set_ylabel("CNN blur prediction (px)")
    axes[0].set_title(f"CNN vs Laplacian (Spearman={rho_s:.3f})")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: CNN prediction histogram
    axes[1].hist(cnn, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("CNN prediction (px)")
    axes[1].set_ylabel("Count")
    med = np.median(cnn)
    axes[1].axvline(med, color="red", linewidth=1.5, linestyle="--", label=f"Median: {med:.2f}")
    axes[1].set_title(f"Prediction Distribution (n={len(cnn)})")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: thumbnail grid — 3 sharpest + 3 blurriest
    sort_idx = np.argsort(cnn)
    sharp_idx = sort_idx[:3]
    blurry_idx = sort_idx[-3:]
    for i, (label, idxs) in enumerate([("Sharpest", sharp_idx), ("Blurriest", blurry_idx)]):
        for j, ci in enumerate(idxs):
            ax = fig.add_axes([0.68 + j * 0.1, 0.55 - i * 0.45, 0.09, 0.35])
            img = cv2.imread(str(crop_files[ci]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ax.imshow(img, cmap="gray")
            ax.set_title(f"{cnn[ci]:.1f}px", fontsize=7)
            ax.axis("off")

    axes[2].axis("off")
    axes[2].set_title("Sharpest (top) vs Blurriest (bottom)")

    fig.tight_layout(rect=[0, 0, 0.67, 1])
    status = "PASS" if rho_s < -0.5 else "WARN" if rho_s < -0.3 else "FAIL"
    summary = (
        f"Real Crop Monotonicity ({len(cnn)} crops)\n"
        f"  Spearman(CNN, Laplacian): {rho_s:.3f} (p={p_val:.2e})\n"
        f"  CNN prediction median: {med:.2f} px\n"
        f"  Expected: negative correlation (higher laplacian = sharper = lower CNN)\n"
        f"  {status}: correlation {'<' if rho_s < -0.5 else '>'} -0.5"
    )
    return fig, summary


def diag_full_inverse_chain(inputs: Dict) -> Tuple[Figure, str]:
    """Full pipeline on real crops: crop -> normalise -> model -> defocus mm."""
    import torch
    import cv2
    from model import DefocusNet
    from inference_engine import boundary_normalise
    from physics import ScalingParams, invert_prediction_with_uncertainty

    ckpt = inputs["checkpoint"]
    cal = inputs["calibration"]
    crops_dir = inputs["crops_dir"]

    config = ckpt.get("config", {})
    max_blur = ckpt.get("max_blur", ckpt.get("max_coc", 20.0))
    model_size = config.get("data", {}).get("image_size_px", 256)
    direct = cal.get("direct", {})
    rho = direct.get("rho_px_per_mm", 1.0)
    sigma_0 = direct.get("sigma_0", 0.0)
    s_cal = direct.get("scale_calib_px_per_mm", 1.0) or 1.0
    loo = direct.get("loo_cv", {})
    rho_std = loo.get("rho_std", 0.0)
    sigma_0_std = loo.get("sigma_0_std", 0.0)

    params = ScalingParams(rho=rho, sigma_0=sigma_0, s_calib=s_cal, s_inference=s_cal,
                           max_blur=max_blur, model_size=model_size)

    model = DefocusNet.from_config(config)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "dme_state_dict" in ckpt:
        state = ckpt["dme_state_dict"]
        sample_key = next(iter(state))
        if not sample_key.startswith("dme_subnet."):
            state = {f"dme_subnet.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()

    crop_files = sorted(crops_dir.rglob("*_crop.png"))
    if not crop_files:
        crop_files = sorted(crops_dir.rglob("*.png"))
    crop_files = crop_files[:100]

    defocus_vals, unc_vals = [], []
    n_saturated, n_clamped = 0, 0

    with torch.no_grad():
        for path in crop_files:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            norm = boundary_normalise(gray)
            native_size = max(gray.shape)
            resized = cv2.resize(norm, (model_size, model_size))
            tensor = torch.from_numpy(resized.astype(np.float32) * 2.0 - 1.0
                                      ).unsqueeze(0).unsqueeze(0)
            pred = model(tensor).squeeze().item()

            r = invert_prediction_with_uncertainty(pred, params, native_size, rho_std, sigma_0_std)
            defocus_vals.append(r.defocus_mm)
            unc_vals.append(r.defocus_uncertainty_mm)
            if r.saturated:
                n_saturated += 1
            if r.clamped:
                n_clamped += 1

    d = np.array(defocus_vals)
    u = np.array(unc_vals)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: defocus histogram
    defocus_range = cal.get("defocus_range_mm", [0, 8])
    axes[0].hist(d, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(abs(defocus_range[0]), color="red", linewidth=1, linestyle="--", alpha=0.5)
    axes[0].axvline(abs(defocus_range[1]), color="red", linewidth=1, linestyle="--", alpha=0.5,
                    label=f"Calib range: {defocus_range}")
    axes[0].set_xlabel("Defocus (mm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Defocus Distribution (n={len(d)})")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: defocus with error bars (subset)
    subset = min(30, len(d))
    sort_idx = np.argsort(d)[:subset]
    axes[1].errorbar(range(subset), d[sort_idx], yerr=u[sort_idx], fmt="o",
                     markersize=4, color="steelblue", ecolor="gray", capsize=2)
    axes[1].set_xlabel("Crop (sorted)")
    axes[1].set_ylabel("Defocus (mm)")
    axes[1].set_title("Predictions with Uncertainty")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: status breakdown
    total = len(d)
    in_range = np.sum((d >= 0) & (d <= abs(defocus_range[1]) * 1.2))
    labels_pie = ["In range", "Saturated", "Clamped", "Other"]
    other = total - in_range - n_saturated - n_clamped
    if other < 0:
        other = 0
    sizes = [in_range, n_saturated, n_clamped, other]
    sizes = [s for s in sizes if s > 0]
    labels_pie = [l for l, s in zip(labels_pie, [in_range, n_saturated, n_clamped, other]) if s > 0]
    colours_pie = ["#2d8a2d", "#cc3333", "#cc8800", "#888888"][:len(sizes)]
    axes[2].pie(sizes, labels=labels_pie, colors=colours_pie, autopct="%1.0f%%", startangle=90)
    axes[2].set_title("Prediction Status")

    fig.tight_layout()
    pct_ok = in_range / total * 100 if total > 0 else 0
    status = "PASS" if pct_ok > 90 and n_saturated / total < 0.05 else "WARN"
    summary = (
        f"Full Inverse Chain ({total} crops)\n"
        f"  In range: {in_range} ({pct_ok:.0f}%), Saturated: {n_saturated}, Clamped: {n_clamped}\n"
        f"  Defocus range: [{d.min():.2f}, {d.max():.2f}] mm\n"
        f"  Mean uncertainty: +/- {u.mean():.3f} mm\n"
        f"  {status}"
    )
    return fig, summary


def diag_scale_sensitivity(inputs: Dict) -> Tuple[Figure, str]:
    """How sensitive is defocus to scale and rho uncertainty."""
    from physics import ScalingParams, invert_prediction

    cal = inputs["calibration"]
    ckpt = inputs["checkpoint"]
    direct = cal.get("direct", {})
    rho = direct.get("rho_px_per_mm", 1.0)
    sigma_0 = direct.get("sigma_0", 0.0)
    s_cal = direct.get("scale_calib_px_per_mm", 1.0) or 1.0
    max_blur = ckpt.get("max_blur", ckpt.get("max_coc", 20.0))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: sweep s_inference
    pred_fixed = 0.5
    scale_sweep = np.linspace(0.8, 1.2, 30) * s_cal
    defocus_sweep = []
    for s_inf in scale_sweep:
        params = ScalingParams(rho=rho, sigma_0=sigma_0, s_calib=s_cal, s_inference=s_inf,
                               max_blur=max_blur, model_size=256)
        r = invert_prediction(pred_fixed, params, 299)
        defocus_sweep.append(r.defocus_mm)

    axes[0].plot(scale_sweep / s_cal, defocus_sweep, "b-", linewidth=1.5)
    axes[0].axvline(1.0, color="red", linewidth=1, linestyle="--", label="Nominal")
    axes[0].set_xlabel("Scale ratio (s_inf / s_calib)")
    axes[0].set_ylabel("Defocus (mm)")
    axes[0].set_title("Sensitivity to Camera Scale")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: sweep rho
    rho_sweep = np.linspace(0.8, 1.2, 30) * rho
    defocus_rho = []
    for rho_val in rho_sweep:
        params = ScalingParams(rho=rho_val, sigma_0=sigma_0, s_calib=s_cal, s_inference=s_cal,
                               max_blur=max_blur, model_size=256)
        r = invert_prediction(pred_fixed, params, 299)
        defocus_rho.append(r.defocus_mm)

    axes[1].plot(rho_sweep / rho * 100 - 100, defocus_rho, "b-", linewidth=1.5)
    axes[1].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_xlabel("rho perturbation (%)")
    axes[1].set_ylabel("Defocus (mm)")
    axes[1].set_title("Sensitivity to rho")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: 2D heatmap rho vs scale
    n = 20
    rho_pct = np.linspace(-10, 10, n)
    scale_pct = np.linspace(-10, 10, n)
    grid = np.zeros((n, n))
    for i, sp in enumerate(scale_pct):
        for j, rp in enumerate(rho_pct):
            params = ScalingParams(
                rho=rho * (1 + rp / 100), sigma_0=sigma_0,
                s_calib=s_cal, s_inference=s_cal * (1 + sp / 100),
                max_blur=max_blur, model_size=256,
            )
            r = invert_prediction(pred_fixed, params, 299)
            grid[i, j] = r.defocus_mm

    im = axes[2].imshow(grid, origin="lower", aspect="auto",
                        extent=[-10, 10, -10, 10], cmap="RdYlGn_r")
    axes[2].set_xlabel("rho perturbation (%)")
    axes[2].set_ylabel("Scale perturbation (%)")
    axes[2].set_title("Defocus Sensitivity Map")
    plt.colorbar(im, ax=axes[2], label="Defocus (mm)", shrink=0.8)

    fig.tight_layout()

    nominal = defocus_sweep[len(defocus_sweep) // 2]
    d_at_5pct = abs(defocus_sweep[-5] - nominal)
    summary = (
        f"Scale Sensitivity (at pred=0.5)\n"
        f"  Nominal defocus: {nominal:.2f} mm\n"
        f"  5% scale error -> {d_at_5pct:.3f} mm change\n"
        f"  Informational — check heatmap for dominant parameter"
    )
    return fig, summary


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic registry
# ═══════════════════════════════════════════════════════════════════════════

DIAGNOSTICS = [
    {"id": "cal_fit", "name": "Calibration Fit Quality", "func": diag_calibration_fit,
     "requires": ["calibration"], "category": "Calibration",
     "description": "Checks rho, sigma_0, R^2, point count against plausibility thresholds. "
                    "Shows LOO-CV stability if available."},

    {"id": "cal_unc", "name": "Uncertainty Budget", "func": diag_uncertainty_budget,
     "requires": ["calibration"], "category": "Calibration",
     "description": "Propagates LOO-CV calibration uncertainty through the inverse chain. "
                    "Shows confidence bands and minimum resolvable defocus."},

    {"id": "config_check", "name": "Config Consistency", "func": diag_config_consistency,
     "requires": ["calibration", "checkpoint"], "category": "Calibration",
     "description": "Compares calibration YAML against checkpoint config for parameter mismatches. "
                    "Tests forward-inverse round-trip and maps saturation zones."},

    {"id": "label_dist", "name": "Label Distribution", "func": diag_label_distribution,
     "requires": ["metadata"], "category": "Synthetic Data",
     "description": "Audits training label distribution for gaps, pile-ups, or empty bins "
                    "that could bias the model."},

    {"id": "blur_trace", "name": "Blur Trace Audit", "func": diag_blur_trace,
     "requires": ["metadata"], "category": "Synthetic Data",
     "description": "Checks quadrature subtraction accuracy — verifies that applied blur "
                    "matches expected blur after native blur accounting."},

    {"id": "ckpt_sanity", "name": "Checkpoint Sanity", "func": diag_checkpoint_sanity,
     "requires": ["checkpoint"], "category": "Model",
     "description": "Verifies checkpoint loads correctly, weights are healthy (no NaN), "
                    "and outputs stay in [0,1] for constant inputs."},

    {"id": "model_synth", "name": "Model vs Synthetic", "func": diag_model_vs_synthetic,
     "requires": ["checkpoint", "metadata", "synth_dir"], "category": "Model",
     "description": "Runs the model on its own training data and compares against ground truth. "
                    "Shows scatter, error histogram, and per-bin MAE."},

    {"id": "monotonicity", "name": "Real Crop Monotonicity", "func": diag_real_crop_monotonicity,
     "requires": ["checkpoint", "crops_dir"], "category": "End-to-End",
     "description": "Checks CNN predictions correlate with classical focus metrics on real crops. "
                    "Shows Spearman correlation and thumbnail comparison."},

    {"id": "full_chain", "name": "Full Inverse Chain", "func": diag_full_inverse_chain,
     "requires": ["checkpoint", "calibration", "crops_dir"], "category": "End-to-End",
     "description": "Runs the complete pipeline on real crops: boundary normalise, model forward, "
                    "inverse chain to mm. Shows defocus distribution and prediction status."},

    {"id": "sensitivity", "name": "Scale Sensitivity", "func": diag_scale_sensitivity,
     "requires": ["checkpoint", "calibration"], "category": "End-to-End",
     "description": "Sweeps rho and camera scale to show how sensitive defocus estimates are "
                    "to calibration parameter uncertainty."},
]


# ═══════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CROPPING Pipeline Diagnostics")
        self.minsize(1200, 800)
        self.resizable(True, True)

        self._running = False
        self._current_fig: Optional[Figure] = None
        self._canvas_widget: Optional[FigureCanvasTkAgg] = None
        self._results: Dict[str, Tuple] = {}

        # Loaded data
        self._inputs: Dict[str, Any] = {}

        self._build_ui()

    def _build_ui(self):
        # ── Top: input paths ──
        paths_frame = ttk.LabelFrame(self, text="Data Paths", padding=8)
        paths_frame.pack(fill="x", padx=10, pady=(10, 4))

        self._path_vars = {}
        path_defs = [
            ("synth_dir", "Synthetic Data:", "folder"),
            ("checkpoint_path", "Model Checkpoint:", "file"),
            ("calibration_path", "Calibration YAML:", "file"),
            ("crops_dir", "Sharp Crops:", "folder"),
        ]
        for i, (key, label, kind) in enumerate(path_defs):
            ttk.Label(paths_frame, text=label, width=18).grid(row=i, column=0, sticky="w", pady=1)
            var = tk.StringVar()
            self._path_vars[key] = var
            ttk.Entry(paths_frame, textvariable=var, width=65).grid(row=i, column=1, sticky="ew", padx=4)
            cmd = (lambda k=key, kd=kind: self._browse(k, kd))
            ttk.Button(paths_frame, text="Browse", command=cmd).grid(row=i, column=2, padx=2)
            var.trace_add("write", lambda *_, k=key: self._on_path_change(k))

        paths_frame.columnconfigure(1, weight=1)

        ttk.Button(paths_frame, text="Load All", command=self._load_all).grid(
            row=len(path_defs), column=2, pady=(4, 0))

        self._load_status_var = tk.StringVar(value="Set paths and click Load All")
        ttk.Label(paths_frame, textvariable=self._load_status_var, foreground="gray",
                  font=("Segoe UI", 8)).grid(row=len(path_defs), column=0, columnspan=2,
                                              sticky="w", pady=(4, 0))

        # ── Main: diagnostics list + results ──
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=4)

        # Left: diagnostic tree
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        ttk.Label(left, text="Diagnostics", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=8, pady=(4, 2))

        columns = ("status", "name")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings",
                                 selectmode="browse")
        self.tree.heading("status", text="")
        self.tree.heading("name", text="")
        self.tree.column("#0", width=0, stretch=False)
        self.tree.column("status", width=50, anchor="center", stretch=False)
        self.tree.column("name", width=230)

        # Group by category
        categories_seen = []
        for d in DIAGNOSTICS:
            cat = d.get("category", "Other")
            if cat not in categories_seen:
                self.tree.insert("", "end", iid=f"cat_{cat}", values=("", cat), open=True,
                                 tags=("category",))
                categories_seen.append(cat)
            self.tree.insert(f"cat_{cat}", "end", iid=d["id"], values=("--", d["name"]))

        self.tree.tag_configure("category", font=("Segoe UI", 9, "bold"))
        self.tree.tag_configure("pass", foreground="#2d8a2d")
        self.tree.tag_configure("fail", foreground="#cc3333")
        self.tree.tag_configure("warn", foreground="#cc8800")
        self.tree.tag_configure("disabled", foreground="#aaaaaa")

        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_run = ttk.Button(btn_frame, text="Run Selected", command=self._run_selected)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Run All", command=self._run_all).pack(
            side="left", fill="x", expand=True, padx=(2, 0))

        # Right: results
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._name_var = tk.StringVar(value="Select a diagnostic")
        ttk.Label(right, textvariable=self._name_var,
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 0))

        self._desc_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._desc_var, wraplength=700, justify="left",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(0, 4))

        self._canvas_frame = ttk.Frame(right)
        self._canvas_frame.pack(fill="both", expand=True, padx=8)

        self._summary = tk.Text(right, height=7, wrap="word", font=("Consolas", 9),
                                bg="#1e1e1e", fg="#d4d4d4", state="disabled")
        self._summary.pack(fill="x", padx=8, pady=(4, 8))
        self._summary.tag_configure("pass", foreground="#4ec94e")
        self._summary.tag_configure("fail", foreground="#f44747")
        self._summary.tag_configure("warn", foreground="#ccaa00")

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self._status_var,
                  font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(0, 4))

        self._update_tree_availability()

    # ── Path handling ──

    def _browse(self, key: str, kind: str):
        if kind == "folder":
            path = filedialog.askdirectory(title=f"Select {key}")
        else:
            ftypes = [("All", "*.*")]
            if "checkpoint" in key:
                ftypes = [("PyTorch", "*.pth *.pt"), ("All", "*.*")]
            elif "calibration" in key:
                ftypes = [("YAML", "*.yaml *.yml"), ("All", "*.*")]
            path = filedialog.askopenfilename(title=f"Select {key}", filetypes=ftypes)
        if path:
            self._path_vars[key].set(path)

    def _on_path_change(self, key: str):
        self._update_tree_availability()

    def _load_all(self):
        self._inputs = {}
        status = []

        synth = self._path_vars["synth_dir"].get()
        if synth and Path(synth).is_dir():
            md = _load_metadata(Path(synth))
            if md is not None:
                self._inputs["metadata"] = md
                self._inputs["synth_dir"] = Path(synth)
                status.append(f"Synth: {len(md)} samples")

        ckpt_path = self._path_vars["checkpoint_path"].get()
        if ckpt_path and Path(ckpt_path).is_file():
            try:
                self._inputs["checkpoint"] = _load_checkpoint(Path(ckpt_path))
                status.append("Checkpoint: loaded")
            except Exception as e:
                status.append(f"Checkpoint: {e}")

        cal_path = self._path_vars["calibration_path"].get()
        if cal_path and Path(cal_path).is_file():
            try:
                self._inputs["calibration"] = _load_calibration_yaml(Path(cal_path))
                status.append("Calibration: loaded")
            except Exception as e:
                status.append(f"Calibration: {e}")

        crops = self._path_vars["crops_dir"].get()
        if crops and Path(crops).is_dir():
            self._inputs["crops_dir"] = Path(crops)
            n = len(list(Path(crops).rglob("*.png")))
            status.append(f"Crops: {n} images")

        self._load_status_var.set(" | ".join(status) if status else "Nothing loaded")
        self._update_tree_availability()

    def _update_tree_availability(self):
        for d in DIAGNOSTICS:
            available = all(r in self._inputs for r in d["requires"])
            if d["id"] not in self._results:
                tag = () if available else ("disabled",)
                self.tree.item(d["id"], values=("--" if available else "N/A", d["name"]), tags=tag)

    # ── Selection ──

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel or sel[0].startswith("cat_"):
            return
        diag = next((d for d in DIAGNOSTICS if d["id"] == sel[0]), None)
        if not diag:
            return
        self._name_var.set(diag["name"])
        self._desc_var.set(diag["description"])

        cached = self._results.get(diag["id"])
        if cached:
            fig, summary = cached
            if fig:
                self._show_result(fig, summary)
            else:
                self._show_error(summary)

    # ── Running ──

    def _run_selected(self):
        sel = self.tree.selection()
        if not sel or sel[0].startswith("cat_") or self._running:
            return
        diag = next((d for d in DIAGNOSTICS if d["id"] == sel[0]), None)
        if not diag:
            return

        missing = [r for r in diag["requires"] if r not in self._inputs]
        if missing:
            self._show_error(f"Missing inputs: {', '.join(missing)}\nLoad them in Data Paths above.")
            return

        self._run_single(diag)

    def _run_single(self, diag: Dict):
        self._running = True
        self.btn_run.config(state="disabled")
        self._status_var.set(f"Running: {diag['name']}...")
        self._set_tree_status(diag["id"], "...")

        def worker():
            try:
                fig, summary = diag["func"](self._inputs)
                self._results[diag["id"]] = (fig, summary)
                status = "FAIL" if "FAIL" in summary else "WARN" if "WARN" in summary else "PASS"
                self.after(0, self._set_tree_status, diag["id"], status)
                self.after(0, self._show_result, fig, summary)
            except Exception as e:
                import traceback
                msg = f"{e}\n\n{traceback.format_exc()}"
                self._results[diag["id"]] = (None, msg)
                self.after(0, self._set_tree_status, diag["id"], "FAIL")
                self.after(0, self._show_error, msg)
            self.after(0, self._done)

        threading.Thread(target=worker, daemon=True).start()

    def _run_all(self):
        if self._running:
            return
        self._running = True
        self.btn_run.config(state="disabled")

        runnable = [d for d in DIAGNOSTICS
                    if all(r in self._inputs for r in d["requires"])]

        def worker():
            for i, diag in enumerate(runnable):
                self.after(0, self._status_var.set,
                           f"Running {i + 1}/{len(runnable)}: {diag['name']}...")
                self.after(0, self._set_tree_status, diag["id"], "...")
                try:
                    fig, summary = diag["func"](self._inputs)
                    self._results[diag["id"]] = (fig, summary)
                    status = "FAIL" if "FAIL" in summary else "WARN" if "WARN" in summary else "PASS"
                    self.after(0, self._set_tree_status, diag["id"], status)
                except Exception as e:
                    import traceback
                    self._results[diag["id"]] = (None, str(e))
                    self.after(0, self._set_tree_status, diag["id"], "FAIL")

            self.after(0, self._done)
            self.after(0, self._status_var.set,
                       f"Done: {len(runnable)} diagnostics run — click any to view")
            if runnable:
                self.after(0, self.tree.selection_set, runnable[0]["id"])
                self.after(0, self._on_select, None)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self):
        self._running = False
        self.btn_run.config(state="normal")

    def _set_tree_status(self, diag_id: str, status: str):
        diag = next(d for d in DIAGNOSTICS if d["id"] == diag_id)
        self.tree.item(diag_id, values=(status, diag["name"]))
        tag_map = {"PASS": "pass", "FAIL": "fail", "WARN": "warn", "...": "disabled"}
        self.tree.item(diag_id, tags=(tag_map.get(status, ()),))

    # ── Display ──

    def _show_result(self, fig: Figure, summary: str):
        if self._canvas_widget:
            self._canvas_widget.get_tk_widget().destroy()
        if self._current_fig:
            plt.close(self._current_fig)

        self._current_fig = fig
        self._canvas_widget = FigureCanvasTkAgg(fig, master=self._canvas_frame)
        self._canvas_widget.draw()
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        self._summary.config(state="normal")
        self._summary.delete("1.0", "end")
        for line in summary.split("\n"):
            if "PASS" in line:
                self._summary.insert("end", line + "\n", "pass")
            elif "FAIL" in line:
                self._summary.insert("end", line + "\n", "fail")
            elif "WARN" in line:
                self._summary.insert("end", line + "\n", "warn")
            else:
                self._summary.insert("end", line + "\n")
        self._summary.config(state="disabled")

    def _show_error(self, msg: str):
        if self._canvas_widget:
            self._canvas_widget.get_tk_widget().destroy()
            self._canvas_widget = None
        self._summary.config(state="normal")
        self._summary.delete("1.0", "end")
        self._summary.insert("1.0", f"ERROR:\n{msg}", "fail")
        self._summary.config(state="disabled")

    def destroy(self):
        if self._current_fig:
            plt.close(self._current_fig)
        super().destroy()


if __name__ == "__main__":
    app = DiagnosticApp()
    app.mainloop()
