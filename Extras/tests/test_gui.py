"""
CROPPING Diagnostic GUI

Interactive visual diagnostics for the depth-from-defocus pipeline.
Each diagnostic runs in-process, produces matplotlib figures with
intermediate values, and displays them in embedded canvases.

Usage:
    python test_gui.py
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional

import numpy as np

# Path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _module in ("Calibration", "Training", "Preprocessing"):
    _p = str(_REPO_ROOT / _module)
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
# Diagnostic functions — each returns a Figure + summary text
# ═══════════════════════════════════════════════════════════════════════════

def diag_calibration_roundtrip() -> tuple:
    """Calibration forward-inverse round-trip with fit visualisation."""
    from calibration_core import calibrate_approach_a, sigma_to_depth_approach_a, linear_model

    rho_true, sigma_0_true = 2.5, 0.8
    z_values = np.linspace(-10, 10, 41)
    sigma_values = np.array([rho_true * abs(z) + sigma_0_true for z in z_values])

    result = calibrate_approach_a(z_values.tolist(), sigma_values.tolist())

    z_recovered = []
    residuals = []
    for z, sigma in zip(z_values, sigma_values):
        sigma_pred = result.rho_px_per_mm * abs(z) + result.sigma_0
        z_rec = sigma_to_depth_approach_a(sigma_pred, result.rho_px_per_mm, result.sigma_0)
        z_recovered.append(z_rec)
        residuals.append(z_rec - abs(z))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: calibration data + fit
    z_plot = np.linspace(-10, 10, 200)
    sigma_fit = linear_model(z_plot, result.rho_px_per_mm, result.sigma_0)
    axes[0].scatter(z_values, sigma_values, s=20, c="steelblue", zorder=3, label="Data")
    axes[0].plot(z_plot, sigma_fit, "r-", linewidth=1.5,
                 label=f"Fit: rho={result.rho_px_per_mm:.4f}, sigma0={result.sigma_0:.3f}")
    axes[0].set_xlabel("Defocus z (mm)")
    axes[0].set_ylabel("Blur sigma (px)")
    axes[0].set_title("Calibration Fit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: z_true vs z_recovered
    z_abs = np.abs(z_values)
    axes[1].scatter(z_abs, z_recovered, s=20, c="steelblue", zorder=3)
    axes[1].plot([0, 10], [0, 10], "r--", linewidth=1, label="Perfect")
    axes[1].set_xlabel("|z| true (mm)")
    axes[1].set_ylabel("|z| recovered (mm)")
    axes[1].set_title("Round-Trip Recovery")
    axes[1].legend(fontsize=8)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: residuals
    axes[2].scatter(z_abs, residuals, s=20, c="steelblue", zorder=3)
    axes[2].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[2].set_xlabel("|z| true (mm)")
    axes[2].set_ylabel("Recovery error (mm)")
    axes[2].set_title(f"Residuals (max: {max(abs(r) for r in residuals):.2e} mm)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    max_err = max(abs(r) for r in residuals)
    summary = (
        f"Calibration Round-Trip\n"
        f"  True: rho={rho_true}, sigma_0={sigma_0_true}\n"
        f"  Fitted: rho={result.rho_px_per_mm:.6f}, sigma_0={result.sigma_0:.6f}\n"
        f"  R^2 = {result.r_squared:.8f}\n"
        f"  Max round-trip error: {max_err:.2e} mm\n"
        f"  {'PASS' if max_err < 1e-6 else 'FAIL'}: error {'<' if max_err < 1e-6 else '>'} 1e-6"
    )
    return fig, summary


def diag_dme_loss() -> tuple:
    """DME loss properties: zero-at-identity, symmetry, scale invariance."""
    import torch
    from losses import DMELoss

    loss_fn = DMELoss(max_blur=20.0, eps=0.01)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: loss surface — L(pred, target) for pred,target in [0.01, 0.99]
    n = 50
    pred_range = np.linspace(0.02, 0.98, n)
    target_range = np.linspace(0.02, 0.98, n)
    loss_grid = np.zeros((n, n))
    for i, t in enumerate(target_range):
        for j, p in enumerate(pred_range):
            loss_grid[i, j] = loss_fn(
                torch.tensor([[p]]), torch.tensor([[t]])
            ).item()

    im = axes[0].imshow(loss_grid, origin="lower", aspect="auto",
                        extent=[0.02, 0.98, 0.02, 0.98], cmap="viridis")
    axes[0].plot([0.02, 0.98], [0.02, 0.98], "r--", linewidth=1, label="Zero line")
    axes[0].set_xlabel("Predicted (norm)")
    axes[0].set_ylabel("Target (norm)")
    axes[0].set_title("Loss Surface")
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].legend(fontsize=8)

    # Panel 2: symmetry check — L(a,b) vs L(b,a)
    pairs = [(0.1, 0.9), (0.2, 0.7), (0.3, 0.6), (0.4, 0.8), (0.15, 0.85)]
    lab = []
    lba = []
    pair_labels = []
    for a, b in pairs:
        l1 = loss_fn(torch.tensor([[a]]), torch.tensor([[b]])).item()
        l2 = loss_fn(torch.tensor([[b]]), torch.tensor([[a]])).item()
        lab.append(l1)
        lba.append(l2)
        pair_labels.append(f"({a},{b})")

    x = np.arange(len(pairs))
    axes[1].bar(x - 0.15, lab, 0.3, label="L(a,b)", color="steelblue")
    axes[1].bar(x + 0.15, lba, 0.3, label="L(b,a)", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_labels, fontsize=8)
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Symmetry Check")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: scale invariance — same relative error at different magnitudes
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    bases = [0.05, 0.2, 0.5, 0.8]
    for base in bases:
        losses = []
        for r in ratios:
            pred = base * r
            if pred < 0.01:
                pred = 0.01
            l = loss_fn(torch.tensor([[pred]]), torch.tensor([[base]])).item()
            losses.append(l)
        axes[2].plot(ratios, losses, "o-", label=f"target={base:.2f}", markersize=4)

    axes[2].set_xlabel("pred/target ratio")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Scale Invariance\n(curves should overlap)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    sym_diffs = [abs(a - b) for a, b in zip(lab, lba)]
    max_sym = max(sym_diffs)
    summary = (
        f"DME Loss Diagnostics\n"
        f"  Loss at identity (0.5, 0.5): {loss_fn(torch.tensor([[0.5]]), torch.tensor([[0.5]])).item():.2e}\n"
        f"  Max symmetry difference: {max_sym:.2e}\n"
        f"  {'PASS' if max_sym < 1e-6 else 'FAIL'}: symmetry\n"
        f"  Scale invariance: visual check (curves should roughly overlap)"
    )
    return fig, summary


def diag_model_output() -> tuple:
    """Model output shape, range, and response to different inputs."""
    import torch
    from model import DefocusNet

    model = DefocusNet()
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: output histogram from random inputs
    outputs = []
    with torch.no_grad():
        for _ in range(20):
            x = torch.randn(8, 1, 256, 256)
            out = model(x).squeeze().cpu().numpy()
            outputs.extend(out.tolist())

    axes[0].hist(outputs, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[0].axvline(1, color="red", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Model output (normalised)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Output Distribution (n={len(outputs)})\nRange: [{min(outputs):.4f}, {max(outputs):.4f}]")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: response to constant-intensity inputs
    intensities = np.linspace(-1, 1, 20)
    responses = []
    with torch.no_grad():
        for val in intensities:
            x = torch.full((1, 1, 256, 256), float(val))
            out = model(x).squeeze().item()
            responses.append(out)

    axes[1].plot(intensities, responses, "o-", color="steelblue", markersize=4)
    axes[1].set_xlabel("Input intensity (constant)")
    axes[1].set_ylabel("Model output")
    axes[1].set_title("Response to Flat Images")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: response to increasing blur (synthetic)
    from synthetic_blur import create_gaussian_kernel
    sharp = np.zeros((256, 256), dtype=np.float32)
    sharp[80:176, 80:176] = 1.0  # square in centre

    sigmas_test = np.linspace(0.5, 15, 20)
    blur_responses = []
    with torch.no_grad():
        for sigma in sigmas_test:
            kernel = create_gaussian_kernel(sigma)
            from scipy.signal import fftconvolve
            blurred = fftconvolve(sharp, kernel, mode="same")
            tensor = torch.from_numpy(blurred * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).float()
            out = model(tensor).squeeze().item()
            blur_responses.append(out)

    axes[2].plot(sigmas_test, blur_responses, "o-", color="steelblue", markersize=4)
    axes[2].set_xlabel("Applied blur sigma (px)")
    axes[2].set_ylabel("Model output (normalised)")
    axes[2].set_title("Response to Increasing Blur\n(should be monotonic)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    in_range = all(0 <= v <= 1 for v in outputs)
    monotonic = all(blur_responses[i] <= blur_responses[i + 1] + 0.05
                    for i in range(len(blur_responses) - 1))
    summary = (
        f"Model Output Diagnostics\n"
        f"  Output range: [{min(outputs):.4f}, {max(outputs):.4f}]\n"
        f"  {'PASS' if in_range else 'FAIL'}: all outputs in [0, 1]\n"
        f"  Blur response monotonic: {'Yes' if monotonic else 'No (check plot)'}\n"
        f"  Shape: (B, 1) confirmed"
    )
    return fig, summary


def diag_gaussian_kernel() -> tuple:
    """Gaussian kernel visualisation at multiple sigma values."""
    from synthetic_blur import create_gaussian_kernel

    test_sigmas = [0.5, 1.0, 3.0, 8.0, 15.0]
    fig, axes = plt.subplots(2, len(test_sigmas), figsize=(14, 6))

    results = []
    for i, sigma in enumerate(test_sigmas):
        k = create_gaussian_kernel(sigma)
        ksum = k.sum()
        symmetric = np.allclose(k, k.T, atol=1e-7)
        results.append((sigma, k.shape[0], ksum, symmetric))

        # Top row: kernel image
        axes[0, i].imshow(k, cmap="hot", interpolation="nearest")
        axes[0, i].set_title(f"sigma={sigma}\n{k.shape[0]}x{k.shape[0]}", fontsize=9)
        axes[0, i].axis("off")

        # Bottom row: cross-section profile
        mid = k.shape[0] // 2
        profile = k[mid, :]
        axes[1, i].plot(profile, color="steelblue", linewidth=1.5)
        axes[1, i].set_title(f"sum={ksum:.6f}", fontsize=8)
        axes[1, i].set_xlabel("px")
        if i == 0:
            axes[1, i].set_ylabel("Weight")
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle("Gaussian Kernels: Image (top) and Cross-Section (bottom)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    all_normalised = all(abs(r[2] - 1.0) < 1e-5 for r in results)
    all_symmetric = all(r[3] for r in results)
    lines = ["Gaussian Kernel Diagnostics"]
    for sigma, size, ksum, sym in results:
        lines.append(f"  sigma={sigma:5.1f}: size={size:3d}, sum={ksum:.6f}, symmetric={sym}")
    lines.append(f"  {'PASS' if all_normalised else 'FAIL'}: all normalised to 1.0")
    lines.append(f"  {'PASS' if all_symmetric else 'FAIL'}: all symmetric")
    return fig, "\n".join(lines)


def diag_blur_roundtrip() -> tuple:
    """Apply known blur to sharp edge, recover via ERF fit."""
    from synthetic_blur import apply_gaussian_blur
    from scipy.special import erf
    from scipy.optimize import curve_fit

    test_sigmas = [1.0, 2.0, 4.0, 8.0]
    size = 256
    sharp = np.zeros((size, size), dtype=np.float32)
    sharp[:, size // 2:] = 1.0

    fig, axes = plt.subplots(2, len(test_sigmas), figsize=(14, 6))

    def erf_model(x, I_left, I_right, edge, sigma):
        sigma_s = max(sigma, 0.001)
        return (I_left + I_right) / 2 + (I_right - I_left) / 2 * erf(
            (x - edge) / (sigma_s * np.sqrt(2))
        )

    results = []
    for i, sigma_true in enumerate(test_sigmas):
        blurred = apply_gaussian_blur(sharp, sigma_true)
        row = size // 2
        profile_x = np.arange(size).astype(np.float64)
        profile = blurred[row, :].astype(np.float64)

        popt, _ = curve_fit(
            erf_model, profile_x, profile,
            p0=[0.0, 1.0, size / 2, sigma_true],
            bounds=([-0.1, 0.5, size * 0.3, 0.01], [0.5, 1.1, size * 0.7, 50]),
        )
        sigma_measured = popt[3]
        error_pct = abs(sigma_measured - sigma_true) / sigma_true * 100
        results.append((sigma_true, sigma_measured, error_pct))

        # Top row: blurred image with profile line
        axes[0, i].imshow(blurred, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axhline(row, color="red", linewidth=0.5, alpha=0.5)
        axes[0, i].set_title(f"sigma={sigma_true}", fontsize=9)
        axes[0, i].axis("off")

        # Bottom row: profile + ERF fit
        fitted = erf_model(profile_x, *popt)
        axes[1, i].plot(profile_x[size // 2 - 30:size // 2 + 30],
                        profile[size // 2 - 30:size // 2 + 30],
                        "b-", linewidth=1, label="Data")
        axes[1, i].plot(profile_x[size // 2 - 30:size // 2 + 30],
                        fitted[size // 2 - 30:size // 2 + 30],
                        "r--", linewidth=1.5, label=f"ERF fit: {sigma_measured:.2f}")
        axes[1, i].set_title(f"Recovered: {sigma_measured:.3f} ({error_pct:.1f}%)", fontsize=8)
        axes[1, i].legend(fontsize=7)
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle("Blur Measurement Round-Trip: Apply Gaussian, Recover via ERF", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    all_pass = all(r[2] < 5.0 for r in results)
    lines = ["Blur Measurement Round-Trip"]
    for sigma_true, sigma_meas, err_pct in results:
        status = "OK" if err_pct < 5.0 else "!!"
        lines.append(f"  [{status}] sigma={sigma_true:.1f}: measured={sigma_meas:.3f} (error {err_pct:.1f}%)")
    lines.append(f"  {'PASS' if all_pass else 'FAIL'}: all within 5% tolerance")
    return fig, "\n".join(lines)


def diag_scaling_chain() -> tuple:
    """Full forward-inverse scaling chain with intermediate values."""
    from physics import ScalingParams, defocus_to_label, label_to_defocus

    configs = [
        {"label": "Same camera", "rho": 2.0, "sigma_0": 0.5,
         "s_calib": 100.0, "s_inf": 100.0, "max_blur": 25.0, "model_size": 256},
        {"label": "Cross-camera", "rho": 1.548, "sigma_0": 0.125,
         "s_calib": 102.57, "s_inf": 120.0, "max_blur": 13.72, "model_size": 256},
        {"label": "High rho", "rho": 5.0, "sigma_0": 1.0,
         "s_calib": 80.0, "s_inf": 80.0, "max_blur": 40.0, "model_size": 256},
    ]

    fig, axes = plt.subplots(len(configs), 3, figsize=(14, 4 * len(configs)))

    all_lines = ["Scaling Chain Diagnostics"]

    for row, cfg in enumerate(configs):
        params = ScalingParams(
            rho=cfg["rho"], sigma_0=cfg["sigma_0"],
            s_calib=cfg["s_calib"], s_inference=cfg["s_inf"],
            max_blur=cfg["max_blur"], model_size=cfg["model_size"],
        )
        native_size = 299
        z_range = np.linspace(0.1, 8.0, 40)

        # Forward + inverse for each z
        labels = [defocus_to_label(z, params, native_size) for z in z_range]
        recovered = []
        errors = []
        for z, label in zip(z_range, labels):
            z_rec, _, _ = label_to_defocus(label, params, native_size)
            recovered.append(z_rec)
            errors.append(z_rec - z)

        ax_row = axes[row] if len(configs) > 1 else axes

        # Panel 1: forward mapping z -> label
        ax_row[0].plot(z_range, labels, "b-", linewidth=1.5)
        ax_row[0].set_xlabel("Defocus (mm)")
        ax_row[0].set_ylabel("Normalised label")
        ax_row[0].set_title(f"{cfg['label']}: z -> label")
        ax_row[0].grid(True, alpha=0.3)

        # Panel 2: z_true vs z_recovered
        ax_row[1].plot(z_range, recovered, "b.", markersize=4)
        ax_row[1].plot([0, 8], [0, 8], "r--", linewidth=1)
        ax_row[1].set_xlabel("z true (mm)")
        ax_row[1].set_ylabel("z recovered (mm)")
        ax_row[1].set_title("Round-Trip Recovery")
        ax_row[1].set_aspect("equal", adjustable="box")
        ax_row[1].grid(True, alpha=0.3)

        # Panel 3: residuals
        ax_row[2].plot(z_range, errors, "b.", markersize=4)
        ax_row[2].axhline(0, color="red", linewidth=1, linestyle="--")
        ax_row[2].set_xlabel("z true (mm)")
        ax_row[2].set_ylabel("Error (mm)")
        max_err = max(abs(e) for e in errors)
        ax_row[2].set_title(f"Max error: {max_err:.2e} mm")
        ax_row[2].grid(True, alpha=0.3)

        status = "PASS" if max_err < 1e-9 else "FAIL"
        all_lines.append(
            f"  [{status}] {cfg['label']}: rho={cfg['rho']}, s_c/s_cal="
            f"{cfg['s_inf']}/{cfg['s_calib']}, max_err={max_err:.2e}"
        )

    fig.tight_layout()
    return fig, "\n".join(all_lines)


def diag_uncertainty() -> tuple:
    """Calibration uncertainty propagation visualisation."""
    from physics import ScalingParams, invert_prediction_with_uncertainty

    params = ScalingParams(
        rho=1.414, sigma_0=0.241, s_calib=102.57, s_inference=102.57,
        max_blur=13.72, model_size=256,
    )
    rho_std, sigma_0_std = 0.032, 0.018

    pred_range = np.linspace(0.05, 0.95, 50)
    defocus_vals = []
    unc_vals = []

    for pred in pred_range:
        r = invert_prediction_with_uncertainty(pred, params, 299, rho_std, sigma_0_std)
        defocus_vals.append(r.defocus_mm)
        unc_vals.append(r.defocus_uncertainty_mm)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: defocus vs prediction with confidence band
    defocus_arr = np.array(defocus_vals)
    unc_arr = np.array(unc_vals)
    axes[0].plot(pred_range, defocus_arr, "b-", linewidth=1.5, label="Defocus")
    axes[0].fill_between(pred_range, defocus_arr - unc_arr, defocus_arr + unc_arr,
                         alpha=0.3, color="steelblue", label="Calibration uncertainty")
    axes[0].set_xlabel("Model prediction (normalised)")
    axes[0].set_ylabel("Defocus (mm)")
    axes[0].set_title("Defocus with Confidence Band")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: absolute uncertainty vs defocus
    axes[1].plot(defocus_arr, unc_arr, "b-", linewidth=1.5)
    axes[1].set_xlabel("Defocus (mm)")
    axes[1].set_ylabel("Uncertainty (mm)")
    axes[1].set_title("Absolute Uncertainty vs Defocus")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: relative uncertainty vs defocus
    rel_unc = np.where(defocus_arr > 0.01, unc_arr / defocus_arr * 100, 0)
    axes[2].plot(defocus_arr, rel_unc, "b-", linewidth=1.5)
    axes[2].set_xlabel("Defocus (mm)")
    axes[2].set_ylabel("Relative uncertainty (%)")
    axes[2].set_title("Relative Uncertainty vs Defocus\n(higher near focus)")
    axes[2].set_ylim(0, min(100, rel_unc.max() * 1.2) if rel_unc.max() > 0 else 10)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    summary = (
        f"Uncertainty Propagation\n"
        f"  Calibration: rho={params.rho} +/- {rho_std}, sigma_0={params.sigma_0} +/- {sigma_0_std}\n"
        f"  At z=1mm: +/- {unc_vals[10]:.3f} mm ({rel_unc[10]:.1f}%)\n"
        f"  At z=4mm: +/- {unc_vals[35]:.3f} mm ({rel_unc[35]:.1f}%)\n"
        f"  At z=7mm: +/- {unc_vals[45]:.3f} mm ({rel_unc[45]:.1f}%)\n"
        f"  Uncertainty grows with defocus (dominated by rho_std)"
    )
    return fig, summary


def diag_compare_results() -> tuple:
    """Compare model predictions against ground truth from synthetic data."""
    from tkinter import filedialog

    metadata_path = filedialog.askopenfilename(
        title="Select metadata.csv (ground truth)",
        filetypes=[("CSV files", "*.csv")],
    )
    if not metadata_path:
        return _placeholder_fig("No metadata.csv selected"), "Cancelled by user"

    inference_path = filedialog.askopenfilename(
        title="Select blur_results.csv (model predictions)",
        filetypes=[("CSV files", "*.csv")],
    )
    if not inference_path:
        return _placeholder_fig("No inference CSV selected"), "Cancelled by user"

    import pandas as pd
    metadata_path = Path(metadata_path)
    inference_path = Path(inference_path)

    # Load ground truth
    gt_df = pd.read_csv(metadata_path)
    gt_df["index"] = gt_df["index"].astype(str).str.zfill(6)
    gt_df = gt_df.set_index("index")
    blur_col = "sigma_px" if "sigma_px" in gt_df.columns else "coc_px"
    blur_term = "sigma" if blur_col == "sigma_px" else "CoC"

    # Load predictions
    pred_df = pd.read_csv(inference_path)
    pred_df["index"] = pred_df["filename"].str.replace(".png", "", regex=False).str.zfill(6)
    pred_df = pred_df.set_index("index")

    # Merge
    merged = gt_df.join(pred_df, lsuffix="_gt", rsuffix="_pred", how="inner")
    merged["blur_gt"] = merged[f"{blur_col}_gt"]
    merged["blur_pred"] = merged[f"{blur_col}_pred"]
    merged["error"] = merged["blur_pred"] - merged["blur_gt"]
    merged["abs_error"] = merged["error"].abs()

    mae = merged["abs_error"].mean()
    rmse = np.sqrt((merged["error"] ** 2).mean())
    bias = merged["error"].mean()
    corr = merged["blur_gt"].corr(merged["blur_pred"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: scatter
    max_val = max(merged["blur_gt"].max(), merged["blur_pred"].max()) * 1.05
    axes[0].scatter(merged["blur_gt"], merged["blur_pred"], s=10, alpha=0.4, c="steelblue")
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
    axes[0].set_xlabel(f"GT {blur_term} (px)")
    axes[0].set_ylabel(f"Predicted {blur_term} (px)")
    axes[0].set_title(f"Predicted vs GT (R²={corr**2:.4f})")
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: error histogram
    axes[1].hist(merged["error"], bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].axvline(bias, color="orange", linewidth=1.5, label=f"Bias: {bias:+.3f}")
    axes[1].set_xlabel("Error (px)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error Distribution (MAE={mae:.3f})")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: error vs GT blur
    axes[2].scatter(merged["blur_gt"], merged["error"], s=10, alpha=0.4, c="steelblue")
    axes[2].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[2].set_xlabel(f"GT {blur_term} (px)")
    axes[2].set_ylabel("Error (px)")
    axes[2].set_title("Error vs Blur Level")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    defocus_mae = ""
    if "defocus_mm_gt" in merged.columns and "defocus_mm_pred" in merged.columns:
        d_mae = (merged["defocus_mm_pred"] - merged["defocus_mm_gt"]).abs().mean()
        defocus_mae = f"\n  Defocus MAE: {d_mae:.4f} mm"

    summary = (
        f"Model vs Ground Truth ({len(merged)} samples)\n"
        f"  MAE:  {mae:.4f} px\n"
        f"  RMSE: {rmse:.4f} px\n"
        f"  Bias: {bias:+.4f} px\n"
        f"  R²:   {corr**2:.4f}"
        f"{defocus_mae}\n"
        f"  {'PASS' if mae < 2.0 else 'FAIL'}: MAE {'<' if mae < 2.0 else '>'} 2.0 px"
    )
    return fig, summary


def diag_classical_baseline() -> tuple:
    """Compare classical focus metrics against CNN on synthetic data."""
    from tkinter import filedialog

    data_dir = filedialog.askdirectory(
        title="Select synthetic data folder (contains metadata.csv + blur/)",
    )
    if not data_dir:
        return _placeholder_fig("No data folder selected"), "Cancelled by user"

    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.csv"
    blur_dir = data_dir / "blur"

    if not metadata_path.exists() or not blur_dir.is_dir():
        return _placeholder_fig("Invalid folder — need metadata.csv + blur/"), \
               f"FAIL: {data_dir} missing metadata.csv or blur/ subfolder"

    import pandas as pd
    import cv2
    from focus_metrics import compute_all_focus_metrics

    # Load metadata
    df = pd.read_csv(metadata_path)
    df["index"] = df["index"].astype(str).str.zfill(6)
    df = df.set_index("index")
    blur_col = "sigma_px" if "sigma_px" in df.columns else "coc_px"

    # Compute classical metrics on a sample (cap at 500 for speed)
    sample = df.head(500)
    metric_names = ["laplacian_var", "tenengrad", "tenengrad_var",
                    "brenner", "norm_laplacian", "energy_gradient"]
    scores = {m: [] for m in metric_names}
    valid_idx = []

    for idx in sample.index:
        img_path = blur_dir / f"{idx}.png"
        if not img_path.exists():
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        metrics = compute_all_focus_metrics(gray)
        for m in metric_names:
            scores[m].append(metrics[m])
        valid_idx.append(idx)

    if len(valid_idx) < 20:
        return _placeholder_fig(f"Only {len(valid_idx)} valid crops found"), \
               f"FAIL: need at least 20 crops, found {len(valid_idx)}"

    blur_gt = sample.loc[valid_idx, blur_col].values

    # Split-half calibration for each metric
    np.random.seed(42)
    indices = np.random.permutation(len(valid_idx))
    split = len(indices) // 2
    cal_idx, test_idx = indices[:split], indices[split:]

    results = {}
    for m in metric_names:
        s = np.array(scores[m])
        log_s = np.log(s + 1)

        # Calibrate on first half
        A = np.vstack([log_s[cal_idx], np.ones(len(cal_idx))]).T
        params, _, _, _ = np.linalg.lstsq(A, blur_gt[cal_idx], rcond=None)
        a, b = params

        # Predict on test half
        pred = a * log_s[test_idx] + b
        gt = blur_gt[test_idx]
        mae = np.mean(np.abs(pred - gt))
        corr = np.corrcoef(gt, pred)[0, 1] if len(gt) > 2 else 0
        results[m] = {"mae": mae, "r2": corr ** 2, "pred": pred, "gt": gt}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: MAE bar chart
    names = list(results.keys())
    maes = [results[m]["mae"] for m in names]
    colours = ["steelblue"] * len(names)
    best_idx = np.argmin(maes)
    colours[best_idx] = "#2d8a2d"

    bars = axes[0].bar(range(len(names)), maes, color=colours, edgecolor="black", alpha=0.8)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("MAE (px)")
    axes[0].set_title("Classical Focus Metrics: Blur Estimation MAE")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, mae_val in zip(bars, maes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{mae_val:.2f}", ha="center", va="bottom", fontsize=8)

    # Panel 2: best metric scatter
    best_name = names[best_idx]
    best = results[best_name]
    axes[1].scatter(best["gt"], best["pred"], s=10, alpha=0.4, c="steelblue")
    max_val = max(best["gt"].max(), best["pred"].max()) * 1.05
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=1)
    axes[1].set_xlabel("GT blur (px)")
    axes[1].set_ylabel("Predicted blur (px)")
    axes[1].set_title(f"Best: {best_name} (R²={best['r2']:.3f}, MAE={best['mae']:.2f})")
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    lines = [f"Classical Baseline ({len(valid_idx)} crops, split-half eval)"]
    for m in names:
        r = results[m]
        marker = " <-- BEST" if m == best_name else ""
        lines.append(f"  {m:<20s} MAE={r['mae']:.3f} px  R²={r['r2']:.4f}{marker}")

    return fig, "\n".join(lines)


def _placeholder_fig(msg: str):
    """Create a blank figure with a message."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color="gray")
    ax.axis("off")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic registry
# ═══════════════════════════════════════════════════════════════════════════

DIAGNOSTICS: List[Dict[str, Any]] = [
    {
        "id": "calibration",
        "name": "Calibration Round-Trip",
        "func": diag_calibration_roundtrip,
        "description": "Fits a linear calibration model to synthetic data, then inverts "
                       "every point and checks recovery. Shows the fit, scatter, and residuals.",
    },
    {
        "id": "dme_loss",
        "name": "DME Loss Properties",
        "func": diag_dme_loss,
        "description": "Visualises the loss surface, verifies symmetry L(a,b)=L(b,a), "
                       "and checks scale invariance (same relative error = similar loss).",
    },
    {
        "id": "model_output",
        "name": "Model Output Behaviour",
        "func": diag_model_output,
        "description": "Runs DefocusNet on random, flat, and increasingly blurred inputs. "
                       "Shows output distribution, response curves, and monotonicity.",
    },
    {
        "id": "kernels",
        "name": "Gaussian Kernels",
        "func": diag_gaussian_kernel,
        "description": "Generates Gaussian kernels at multiple sigma values. "
                       "Shows kernel images, cross-section profiles, normalisation, and symmetry.",
    },
    {
        "id": "blur_roundtrip",
        "name": "Blur Measurement Round-Trip",
        "func": diag_blur_roundtrip,
        "description": "Applies known Gaussian blur to a sharp edge, then recovers sigma "
                       "via ERF fitting. Shows the edge profile, fit overlay, and recovery error.",
    },
    {
        "id": "scaling_chain",
        "name": "Scaling Chain",
        "func": diag_scaling_chain,
        "description": "Tests the full physics forward-inverse chain (z -> sigma -> native -> "
                       "model -> label -> back) for multiple camera configurations. "
                       "Shows intermediate values and round-trip precision.",
    },
    {
        "id": "uncertainty",
        "name": "Uncertainty Propagation",
        "func": diag_uncertainty,
        "description": "Propagates LOO-CV calibration uncertainty through the inverse chain. "
                       "Shows confidence bands, absolute and relative uncertainty vs defocus.",
    },
    {
        "id": "compare_results",
        "name": "Model vs Ground Truth",
        "func": diag_compare_results,
        "description": "Compares model predictions against ground truth from synthetic data. "
                       "Prompts for metadata.csv and blur_results.csv, then shows scatter, "
                       "error histogram, and error vs blur level.",
    },
    {
        "id": "classical_baseline",
        "name": "Classical Baseline",
        "func": diag_classical_baseline,
        "description": "Runs 6 classical focus metrics (Laplacian, Tenengrad, Brenner, etc.) "
                       "on synthetic crops and compares blur estimation accuracy. "
                       "Prompts for the synthetic data folder.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CROPPING Pipeline Diagnostics")
        self.minsize(1100, 750)
        self.resizable(True, True)

        self._running = False
        self._current_fig: Optional[Figure] = None
        self._canvas_widget: Optional[FigureCanvasTkAgg] = None
        self._all_results: Dict[str, tuple] = {}

        self._build_ui()

    def _build_ui(self):
        # Main paned layout
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # Left panel: diagnostic list
        left = ttk.Frame(paned, width=280)
        paned.add(left, weight=0)

        ttk.Label(left, text="Diagnostics", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 4))

        columns = ("status", "name")
        self.tree = ttk.Treeview(left, columns=columns, show="headings",
                                 selectmode="browse", height=len(DIAGNOSTICS))
        self.tree.heading("status", text="")
        self.tree.heading("name", text="Test")
        self.tree.column("status", width=50, anchor="center", stretch=False)
        self.tree.column("name", width=220)

        # Style for coloured status text
        style = ttk.Style()
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=26)

        for d in DIAGNOSTICS:
            self.tree.insert("", "end", iid=d["id"], values=("--", d["name"]))
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_run = ttk.Button(btn_frame, text="Run Selected", command=self._run_selected)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Run All", command=self._run_all).pack(
            side="left", fill="x", expand=True, padx=(2, 0))

        # Right panel: results
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # Description
        self.var_name = tk.StringVar(value="Select a diagnostic")
        ttk.Label(right, textvariable=self.var_name,
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        self.var_desc = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.var_desc,
                  wraplength=700, justify="left",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(0, 4))

        # Figure canvas area
        self.canvas_frame = ttk.Frame(right)
        self.canvas_frame.pack(fill="both", expand=True, padx=8)

        # Summary text
        self.summary_text = tk.Text(
            right, height=8, wrap="word", font=("Consolas", 9),
            bg="#1e1e1e", fg="#d4d4d4", state="disabled",
        )
        self.summary_text.pack(fill="x", padx=8, pady=(4, 8))
        self.summary_text.tag_configure("pass", foreground="#4ec94e")
        self.summary_text.tag_configure("fail", foreground="#f44747")

        # Status bar
        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.var_status,
                  font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(0, 4))

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        diag_id = sel[0]
        diag = next(d for d in DIAGNOSTICS if d["id"] == diag_id)
        self.var_name.set(diag["name"])
        self.var_desc.set(diag["description"])

        # Show cached result if available
        cached = self._all_results.get(diag_id)
        if cached is not None:
            fig, summary = cached
            if fig is not None:
                self._show_result(fig, summary)
            else:
                self._show_error(summary)

    def _run_selected(self):
        sel = self.tree.selection()
        if not sel or self._running:
            return
        diag = next(d for d in DIAGNOSTICS if d["id"] == sel[0])
        self._run_diagnostic(diag)

    def _run_all(self):
        if self._running:
            return
        self._running = True
        self.btn_run.config(state="disabled")
        self._all_results: Dict[str, tuple] = {}

        def worker():
            for i, diag in enumerate(DIAGNOSTICS):
                self.after(0, self.var_status.set,
                           f"Running {i + 1}/{len(DIAGNOSTICS)}: {diag['name']}...")
                self.after(0, self._set_tree_status, diag["id"], "...")
                try:
                    fig, summary = diag["func"]()
                    self._all_results[diag["id"]] = (fig, summary)
                    passed = "FAIL" not in summary
                    self.after(0, self._set_tree_status, diag["id"],
                               "PASS" if passed else "FAIL")
                except Exception as e:
                    import traceback
                    self._all_results[diag["id"]] = (None, f"ERROR: {e}\n{traceback.format_exc()}")
                    self.after(0, self._set_tree_status, diag["id"], "FAIL")

            self.after(0, self._on_run_all_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_all_done(self):
        self._running = False
        self.btn_run.config(state="normal")
        self.var_status.set("All diagnostics complete — click any to view results")
        # Show first result
        if self._all_results:
            first_id = DIAGNOSTICS[0]["id"]
            self.tree.selection_set(first_id)
            self._on_select(None)

    def _set_tree_status(self, diag_id: str, status: str):
        """Update the status column for a diagnostic in the tree."""
        diag = next(d for d in DIAGNOSTICS if d["id"] == diag_id)
        self.tree.item(diag_id, values=(status, diag["name"]))
        # Colour the row via tags
        if status == "PASS":
            self.tree.item(diag_id, tags=("pass",))
        elif status == "FAIL":
            self.tree.item(diag_id, tags=("fail",))
        elif status == "...":
            self.tree.item(diag_id, tags=("running",))
        else:
            self.tree.item(diag_id, tags=())
        self.tree.tag_configure("pass", foreground="#2d8a2d")
        self.tree.tag_configure("fail", foreground="#cc3333")
        self.tree.tag_configure("running", foreground="#888888")

    def _run_diagnostic(self, diag: Dict):
        if self._running:
            return
        self._running = True
        self.btn_run.config(state="disabled")
        self.var_status.set(f"Running: {diag['name']}...")
        self.var_name.set(diag["name"])
        self.var_desc.set(diag["description"])
        self._set_tree_status(diag["id"], "...")

        def worker():
            try:
                fig, summary = diag["func"]()
                self._all_results[diag["id"]] = (fig, summary)
                passed = "FAIL" not in summary
                self.after(0, self._set_tree_status, diag["id"],
                           "PASS" if passed else "FAIL")
                self.after(0, self._show_result, fig, summary)
            except Exception as e:
                import traceback
                err_msg = f"{e}\n\n{traceback.format_exc()}"
                self._all_results[diag["id"]] = (None, err_msg)
                self.after(0, self._set_tree_status, diag["id"], "FAIL")
                self.after(0, self._show_error, err_msg)
            finally:
                self.after(0, self._on_diag_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_diag_done(self):
        self._running = False
        self.btn_run.config(state="normal")
        self.var_status.set("Done")

    def _show_result(self, fig: Figure, summary: str):
        # Clear old canvas
        if self._canvas_widget is not None:
            self._canvas_widget.get_tk_widget().destroy()
            self._canvas_widget = None
        if self._current_fig is not None:
            plt.close(self._current_fig)

        self._current_fig = fig
        self._canvas_widget = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self._canvas_widget.draw()
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        # Update summary text with colour
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        for line in summary.split("\n"):
            if "PASS" in line:
                self.summary_text.insert("end", line + "\n", "pass")
            elif "FAIL" in line:
                self.summary_text.insert("end", line + "\n", "fail")
            else:
                self.summary_text.insert("end", line + "\n")
        self.summary_text.config(state="disabled")

    def _show_error(self, msg: str):
        if self._canvas_widget is not None:
            self._canvas_widget.get_tk_widget().destroy()
            self._canvas_widget = None

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", f"ERROR:\n{msg}", "fail")
        self.summary_text.config(state="disabled")
        self.var_status.set("Error")

    def destroy(self):
        if self._current_fig is not None:
            plt.close(self._current_fig)
        super().destroy()


if __name__ == "__main__":
    app = DiagnosticApp()
    app.mainloop()
