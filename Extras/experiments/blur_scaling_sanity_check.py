"""
blur_scaling_sanity_check.py
============================
Standalone sanity-check for the blur-scaling bug in the droplet-defocus pipeline.

Compares two competing blur-generation chains on the same near-focus source crop:

  CURRENT CODE:
    sigma_model = rho * |z| * (scale_px/scale_calib) * (model_size / calib_ref_res)
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                         128/860  <- uses calibration image size

  CORRECTED:
    sigma_model = rho * |z| * (scale_px/scale_calib) * (model_size / crop_size)
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                         128/299  <- uses experiment crop size

Also includes a calibration-target comparison module that measures actual blur
from image content and compares against calibration predictions.

Usage:
    python blur_scaling_sanity_check.py

Edit the PARAMETERS block below to match your setup.
"""

import os
import sys
import csv
import math
import textwrap
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try to import scipy for ERF fitting (optional dependency)
try:
    from scipy.optimize import curve_fit  # type: ignore
    from scipy.special import erf as scipy_erf  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

from paths_config import CROP_BASE

# Path to a near-focus source crop (grayscale PNG/TIFF/etc.).
# Ideally already 299x299. If not, it will be resized to crop_size_px.
# Derive from CROP_BASE or set manually below.
IMAGE_PATH = None  # set manually or derive from CROP_BASE

# Directory where all outputs will be saved (created if absent).
OUTPUT_DIR = str(Path(__file__).parent / "blur_scaling_output")

# Short label appended to output filenames — useful when running multiple configs.
IMAGE_LABEL = "camera_g"

# ── Defocus to test (single-image mode) ──────────────────────────────────────
Z_MM = 4.0  # defocus magnitude in mm (positive; sign does not affect blur magnitude)

# ── Calibration parameters (from calibration_results.yaml) ──────────────────
RHO_DIRECT = 1.413489803011591   # blur slope, calibration-camera px / mm
SIGMA_0    = 0.0                 # blur offset at focus, calibration-camera px
                                 # (set to calibrated value if you want, e.g. 0.2413)

# ── Scale parameters ─────────────────────────────────────────────────────────
SCALE_PX_PER_MM       = 50.2    # experiment camera px/mm  (from sharp_crops.csv, camera g)
SCALE_CALIB_PX_PER_MM = 102.57  # calibration camera px/mm (from calibration_results.yaml)

# ── Resolution parameters ────────────────────────────────────────────────────
CALIB_REFERENCE_RESOLUTION = 860  # px — sphere-crop size at which rho was measured
CROP_SIZE_PX               = 299  # px — experiment droplet crop size (training pipeline)
MODEL_SIZE_PX              = 128  # px — neural network input size

# ── Native blur of the source crop ───────────────────────────────────────────
# sigma of the native lens PSF *in 299px crop-pixel units*.
# Measure from a focused frame with erf fitting, or estimate visually.
# Set to 0.0 if the source image is assumed perfectly sharp.
NATIVE_BLUR_SIGMA_CROP_PX = 0.731  # from sharp_crops.csv: sphere1015g, camera g

# ── Blur kernel parameters ───────────────────────────────────────────────────
RADIUS_FACTOR    = 3.0   # kernel half-width = ceil(radius_factor * sigma)
SIGMA_THRESHOLD  = 0.5   # if kernel sigma <= this, no blur is applied (matches pipeline)

# ── Parameter sweep ──────────────────────────────────────────────────────────
# Set SWEEP_ENABLED = True to also generate a CSV over multiple z values.
SWEEP_ENABLED = True
SWEEP_Z_MM    = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

# Set FULL_SWEEP_IMAGES = True to run the complete image pipeline (blur + ERF
# measurement) for every z in SWEEP_Z_MM and produce a combined comparison
# figure + summary CSV.  This is in addition to the single-z run at Z_MM.
FULL_SWEEP_IMAGES = True

# ── Calibration target parameters ────────────────────────────────────────────
# Whether to include sigma_0 when computing the calibration target total sigma.
# Set True to include the focus-offset sigma_0 in the calibration prediction.
USE_SIGMA0_IN_TARGET = False

# ── Blur estimation (ERF / edge-profile) parameters ──────────────────────────
# Number of rows above/below image centre to average for the analysis profile.
CENTER_BAND_HALF_HEIGHT = 2
# Half-width of the pixel search window around the image centre for edge detection.
EDGE_SEARCH_WINDOW_HALF_WIDTH = 40
# Use scipy ERF curve_fit if scipy is installed.
USE_SCIPY_ERF_FIT_IF_AVAILABLE = True

# =============================================================================
# END OF PARAMETERS
# =============================================================================


# ---------------------------------------------------------------------------
# Core utility functions (unchanged from original)
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_grayscale(image_path: str) -> np.ndarray:
    """Load image as float32 grayscale in [0, 255]."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Source image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cv2.imread could not read: {image_path}")
    return img.astype(np.float32)


def resize_to_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Resize image to crop_size×crop_size if not already that size."""
    h, w = img.shape[:2]
    if h == crop_size and w == crop_size:
        return img.copy()
    interp = cv2.INTER_AREA if (h > crop_size or w > crop_size) else cv2.INTER_CUBIC
    return cv2.resize(img, (crop_size, crop_size), interpolation=interp).astype(np.float32)


def resize_to_model(img: np.ndarray, model_size: int, crop_size: int) -> np.ndarray:
    """Resize crop-space image to model-space exactly as the training pipeline does."""
    interp = cv2.INTER_AREA if model_size < crop_size else cv2.INTER_CUBIC
    return cv2.resize(img, (model_size, model_size), interpolation=interp).astype(np.float32)


def make_gaussian_kernel(sigma: float, radius_factor: float) -> np.ndarray:
    """
    Build a 2D Gaussian kernel with half-width = ceil(radius_factor * sigma).
    Matches the pipeline convention.
    """
    half = int(math.ceil(radius_factor * sigma))
    size = 2 * half + 1
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = kernel_1d @ kernel_1d.T
    return kernel_2d


def apply_gaussian_blur(img: np.ndarray, sigma: float,
                        radius_factor: float, sigma_threshold: float) -> tuple:
    """
    Apply Gaussian blur with given sigma.
    Returns (blurred_image, applied: bool).
    If sigma <= sigma_threshold, returns original image unchanged.
    """
    if sigma <= sigma_threshold:
        return img.copy(), False
    kernel = make_gaussian_kernel(sigma, radius_factor)
    blurred = cv2.filter2D(img, ddepth=-1, kernel=kernel,
                           borderType=cv2.BORDER_REPLICATE)
    return blurred.astype(np.float32), True


def compute_sigma_chain(rho: float, sigma_0: float, z_mm: float,
                        scale_px: float, scale_calib: float,
                        model_size: int, ref_res: int, crop_size: int,
                        native_blur_crop_px: float) -> dict:
    """
    Compute sigma values for both current-code and corrected chains.

    Returns a dict with all intermediate and final sigma values.
    """
    z = abs(z_mm)

    # Step 1: sigma in calibration-camera px (slope + optional offset)
    sigma_calib = rho * z + sigma_0

    # Step 2: cross-camera correction -> experiment raw-frame px
    cross_camera_ratio = scale_px / scale_calib
    sigma_exp = sigma_calib * cross_camera_ratio

    # Step 3a: current-code scaling (128 / 860)
    sigma_target_code = sigma_exp * (model_size / ref_res)

    # Step 3b: corrected scaling (128 / 299)
    sigma_target_correct = sigma_exp * (model_size / crop_size)

    # Native blur in model-pixel space
    native_to_model = model_size / crop_size
    sigma_native_model = native_blur_crop_px * native_to_model

    # Quadrature kernel sigmas
    sigma_kernel_code    = math.sqrt(max(sigma_target_code**2    - sigma_native_model**2, 0.0))
    sigma_kernel_correct = math.sqrt(max(sigma_target_correct**2 - sigma_native_model**2, 0.0))

    return {
        "z_mm":                  z_mm,
        "sigma_calib":           sigma_calib,
        "cross_camera_ratio":    cross_camera_ratio,
        "sigma_exp":             sigma_exp,
        "sigma_target_code":     sigma_target_code,
        "sigma_target_correct":  sigma_target_correct,
        "sigma_native_model":    sigma_native_model,
        "sigma_kernel_code":     sigma_kernel_code,
        "sigma_kernel_correct":  sigma_kernel_correct,
        "target_ratio":          sigma_target_correct / sigma_target_code if sigma_target_code > 0 else float("inf"),
        "kernel_ratio":          sigma_kernel_correct / sigma_kernel_code if sigma_kernel_code > 0 else float("inf"),
        "native_to_model":       native_to_model,
    }


def save_uint8(img_f32: np.ndarray, path: str) -> None:
    """Clip float32 image to [0,255] and save as uint8 PNG."""
    out = np.clip(img_f32, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference of two float32 images, scaled to [0,255]."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if diff.max() > 0:
        diff = diff * (255.0 / diff.max())
    return diff


def build_comparison_panel(images: list, labels: list, tile_size: int = 128) -> np.ndarray:
    """
    Stack a list of uint8 images into one wide panel with text labels above each tile.
    """
    n = len(images)
    label_height = 20
    panel_h = tile_size + label_height
    panel_w = tile_size * n
    panel = np.zeros((panel_h, panel_w), dtype=np.uint8)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        if img.shape[0] != tile_size or img.shape[1] != tile_size:
            img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        x0 = i * tile_size
        panel[label_height:, x0:x0 + tile_size] = img

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.30
        thickness = 1
        (tw, th), _ = cv2.getTextSize(lbl, font, font_scale, thickness)
        tx = x0 + max(0, (tile_size - tw) // 2)
        ty = label_height - 4
        cv2.putText(panel, lbl, (tx, ty), font, font_scale, 255, thickness, cv2.LINE_AA)

    return panel


def centerline_profiles(source: np.ndarray,
                        blurred_code: np.ndarray,
                        blurred_correct: np.ndarray,
                        out_path: str) -> dict:
    """
    Extract horizontal intensity profiles through image centre and save as PNG.
    Also returns a simple edge-width estimate (10–90% rise distance).
    Preserved from original script.
    """
    h, w = source.shape
    cy = h // 2

    prof_src     = source[cy, :].astype(np.float32)
    prof_code    = blurred_code[cy, :].astype(np.float32)
    prof_correct = blurred_correct[cy, :].astype(np.float32)

    xs = np.arange(w)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, prof_src,     color="black",      linewidth=1.2, label="Source")
    ax.plot(xs, prof_code,    color="tab:blue",   linewidth=1.2, linestyle="--",  label="Code blur")
    ax.plot(xs, prof_correct, color="tab:orange", linewidth=1.2, linestyle="-.", label="Corrected blur")
    ax.set_xlabel("Column pixel (model space)")
    ax.set_ylabel("Intensity (0–255)")
    ax.set_title("Horizontal centreline profiles")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    def edge_width_10_90(prof: np.ndarray) -> float:
        grad = np.abs(np.gradient(prof.astype(np.float64)))
        peak = int(np.argmax(grad))
        lo, hi = prof.min(), prof.max()
        span = hi - lo
        if span < 1e-3:
            return float("nan")
        thresh_lo = lo + 0.10 * span
        thresh_hi = lo + 0.90 * span
        left = peak
        for i in range(peak, -1, -1):
            if prof[i] <= thresh_lo:
                left = i
                break
        right = peak
        for i in range(peak, w):
            if prof[i] >= thresh_hi:
                right = i
                break
        if right <= left:
            for i in range(peak, w):
                if prof[i] <= thresh_lo:
                    right = i
                    break
        return float(abs(right - left))

    widths = {
        "source":  edge_width_10_90(prof_src),
        "code":    edge_width_10_90(prof_code),
        "correct": edge_width_10_90(prof_correct),
    }
    return widths


# ---------------------------------------------------------------------------
# NEW: Calibration target computation
# ---------------------------------------------------------------------------

def compute_calibration_targets(rho_direct: float, sigma_0: float, z_mm: float,
                                 use_sigma0: bool,
                                 scale_px_per_mm: float, scale_calib_px_per_mm: float,
                                 model_size_px: int, crop_size_px: int,
                                 calib_ref_res: int) -> dict:
    """
    Compute calibration-predicted TOTAL sigma values at defocus z_mm.

    These are total blur predictions (not the added kernel — native blur not subtracted).

    Notation:
      sigma_calib_total        : total blur in calibration-camera px
      sigma_exp_total          : total blur in experiment raw-frame px
      sigma_target_model_code  : total model-space sigma under current-code scaling
      sigma_target_model_correct: total model-space sigma under corrected scaling
    """
    z = abs(z_mm)
    if use_sigma0:
        sigma_calib_total = rho_direct * z + sigma_0
    else:
        sigma_calib_total = rho_direct * z

    sigma_exp_total = sigma_calib_total * (scale_px_per_mm / scale_calib_px_per_mm)
    sigma_target_model_correct = sigma_exp_total * (model_size_px / crop_size_px)
    sigma_target_model_code    = sigma_exp_total * (model_size_px / calib_ref_res)

    return {
        "sigma_calib_total":           sigma_calib_total,
        "sigma_exp_total":             sigma_exp_total,
        "sigma_target_model_code":     sigma_target_model_code,
        "sigma_target_model_correct":  sigma_target_model_correct,
    }


# ---------------------------------------------------------------------------
# NEW: Blur estimation from image content
# ---------------------------------------------------------------------------

def _make_blur_result(method: str, edge_x: float, sigma: float,
                      profile: np.ndarray, fit_xs, fit_ys, notes: str) -> dict:
    """Build a standardised blur-estimation result dict."""
    return {
        "method_used":       method,
        "edge_center_x":     float(edge_x),
        "estimated_sigma_px": sigma,
        "profile":           profile,
        "fit_xs":            fit_xs,
        "fit_ys":            fit_ys,
        "notes":             notes,
    }


def _erf_model(x, a, b, x0, sigma):
    """
    1D error-function edge model.
    I(x) = a + b * erf((x - x0) / (sigma * sqrt(2)))
    Only called when scipy is available.
    """
    return a + b * scipy_erf((x - x0) / (max(abs(sigma), 1e-6) * math.sqrt(2)))


def _try_erf_fit(profile: np.ndarray, w: int,
                 edge_x: int, search_half_w: int, label: str):
    """
    Attempt a scipy curve_fit of the ERF edge model.
    Returns a blur-result dict on success, None on failure.
    """
    fit_lo = max(0, edge_x - search_half_w)
    fit_hi = min(w - 1, edge_x + search_half_w)
    xs = np.arange(fit_lo, fit_hi + 1, dtype=np.float64)
    ys = profile[fit_lo:fit_hi + 1]
    if len(xs) < 6:
        return None

    a0 = float(np.median(ys))
    b0 = float((ys.max() - ys.min()) / 2.0)
    p0 = [a0, b0, float(edge_x), 2.0]

    try:
        popt, _ = curve_fit(
            _erf_model, xs, ys, p0=p0,
            bounds=([-np.inf, -np.inf, float(fit_lo), 0.01],
                    [np.inf,   np.inf, float(fit_hi), float(w)]),
            maxfev=8000,
        )
        _, _, x0_fit, sigma_fit = popt
        sigma_fit = abs(sigma_fit)
        fit_xs = np.linspace(fit_lo, fit_hi, 300)
        fit_ys = _erf_model(fit_xs, *popt)
        return _make_blur_result(
            "erf_fit", x0_fit, sigma_fit, profile, fit_xs, fit_ys,
            f"[{label}] ERF fit converged: x0={x0_fit:.2f} px, sigma={sigma_fit:.4f} px",
        )
    except Exception:
        return None


def _width_10_90_estimate(profile: np.ndarray, w: int, edge_x: int, label: str) -> dict:
    """
    Fallback blur estimator: 10-90% edge-spread width.
    Converts to approximate sigma via sigma ≈ width / 2.5621
    (derived from the ERF model: 10-90% rise = 2 * erfinv(0.8) * sqrt(2) * sigma ≈ 2.5621 * sigma).
    """
    lo_val = float(profile.min())
    hi_val = float(profile.max())
    span = hi_val - lo_val
    if span < 1e-3:
        return _make_blur_result(
            "failed", edge_x, float("nan"), profile, None, None,
            f"[{label}] Uniform image (span < 1e-3); cannot estimate sigma.",
        )

    thresh_10 = lo_val + 0.10 * span
    thresh_90 = lo_val + 0.90 * span

    # Search left of edge for 10% crossing
    left_x = 0
    for i in range(edge_x, -1, -1):
        if profile[i] <= thresh_10:
            left_x = i
            break

    # Search right of edge for 90% crossing (rising edge)
    right_x = w - 1
    for i in range(edge_x, w):
        if profile[i] >= thresh_90:
            right_x = i
            break

    # Handle falling edge (e.g. dark sphere interior)
    if right_x <= left_x:
        right_x = w - 1
        for i in range(edge_x, w):
            if profile[i] <= thresh_10:
                right_x = i
                break

    width_10_90 = float(abs(right_x - left_x))
    sigma_est = width_10_90 / 2.5621 if width_10_90 > 0 else float("nan")

    method = "width_10_90"
    notes = (
        f"[{label}] 10-90% width={width_10_90:.2f} px -> sigma_approx={sigma_est:.4f} px "
        f"(proxy, not a true ERF fit)"
    )
    return _make_blur_result(method, float(edge_x), sigma_est, profile, None, None, notes)


def estimate_blur_sigma(img: np.ndarray,
                        band_half_height: int,
                        search_window_half_width: int,
                        use_scipy_if_available: bool,
                        label: str = "image") -> dict:
    """
    Estimate the blur sigma (in image-space pixels) from a greyscale image.

    Method:
      1. Average rows in a band of height 2*band_half_height+1 around image centre.
      2. Locate the strongest gradient within search_window_half_width columns of centre.
      3. Fit a 1D ERF model (if scipy available and requested), else use 10-90% width.

    Returns a dict with keys:
      method_used, edge_center_x, estimated_sigma_px, profile,
      fit_xs, fit_ys, notes
    """
    h, w = img.shape[:2]
    cy = h // 2
    cx = w // 2

    # Build averaged centre band
    r0 = max(0, cy - band_half_height)
    r1 = min(h, cy + band_half_height + 1)
    band = img[r0:r1, :].astype(np.float64)
    profile = band.mean(axis=0)   # shape: (w,)

    # Compute gradient and search within window around cx
    x_lo = max(0, cx - search_window_half_width)
    x_hi = min(w - 1, cx + search_window_half_width)
    grad = np.abs(np.gradient(profile))
    search_grad = grad[x_lo:x_hi + 1]

    if search_grad.max() < 1e-4:
        return _make_blur_result(
            "failed", cx, float("nan"), profile, None, None,
            f"[{label}] No significant gradient in search window — image may be uniform.",
        )

    peak_local = int(np.argmax(search_grad))
    edge_x = x_lo + peak_local

    # Try scipy ERF fit first
    if use_scipy_if_available and SCIPY_AVAILABLE:
        result = _try_erf_fit(profile, w, edge_x, search_window_half_width, label)
        if result is not None:
            return result
        # ERF fit failed — fall through to width estimator

    return _width_10_90_estimate(profile, w, edge_x, label)


# ---------------------------------------------------------------------------
# NEW: Edge-fit figures
# ---------------------------------------------------------------------------

def save_edge_fit_profiles(results: dict, out_path: str,
                           calib_targets: dict) -> None:
    """
    Save edge_fit_profiles.png: centre-band profiles for source / code / corrected
    with ERF fit curves overlaid (if available) and target-sigma annotations.

    results: dict with keys 'source', 'code', 'correct', each a blur-estimation result.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colours = {"source": "black", "code": "tab:blue", "correct": "tab:orange"}
    styles  = {"source": "-",     "code": "--",       "correct": "-."}
    display = {"source": "Source", "code": "Code-blur", "correct": "Corrected"}

    for key in ("source", "code", "correct"):
        res = results[key]
        profile = res["profile"]
        w = len(profile)
        xs = np.arange(w)
        ax.plot(xs, profile, color=colours[key], linestyle=styles[key],
                linewidth=1.3, label=f"{display[key]} (measured)")

        # Overlay ERF fit if available
        if res["fit_xs"] is not None and res["fit_ys"] is not None:
            ax.plot(res["fit_xs"], res["fit_ys"], color=colours[key],
                    linestyle=":", linewidth=2.0, alpha=0.8,
                    label=f"{display[key]} ERF fit  sigma={res['estimated_sigma_px']:.3f} px")

        # Mark edge location
        ex = res["edge_center_x"]
        ax.axvline(x=ex, color=colours[key], linewidth=0.7, alpha=0.5)

    # Annotate calibration targets
    sig_code   = calib_targets["sigma_target_model_code"]
    sig_corr   = calib_targets["sigma_target_model_correct"]
    ax.text(0.02, 0.97,
            f"Calib target (code):    {sig_code:.3f} px (total model-space)\n"
            f"Calib target (correct): {sig_corr:.3f} px (total model-space)",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Column pixel (model space)")
    ax.set_ylabel("Intensity (0–255)")
    ax.set_title("ERF edge-fit profiles — centre band")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_edge_fit_diagnostics(images: dict, blur_results: dict,
                               band_half_height: int, out_path: str) -> None:
    """
    Save edge_fit_diagnostics.png: for each of source / code / corrected,
    show (left) image with centre band and edge marker, (right) profile + fit.

    images:       dict  keys 'source', 'code', 'correct' -> float32 np.ndarray
    blur_results: dict  keys 'source', 'code', 'correct' -> blur-estimation result
    """
    keys    = ("source", "code", "correct")
    titles  = ("Source", "Code-blur", "Corrected")
    colours = ("black",  "tab:blue",  "tab:orange")

    fig, axes = plt.subplots(nrows=3, ncols=2,
                             figsize=(10, 9),
                             gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle("Edge-fit diagnostics — centre band analysis", fontsize=11)

    for row, (key, title, colour) in enumerate(zip(keys, titles, colours)):
        img = images[key]
        res = blur_results[key]
        h, w = img.shape[:2]
        cy = h // 2
        r0 = max(0, cy - band_half_height)
        r1 = min(h, cy + band_half_height + 1)

        # --- Left subplot: image with band + edge marker ---
        ax_img = axes[row, 0]
        img_disp = np.clip(img, 0, 255).astype(np.uint8)
        ax_img.imshow(img_disp, cmap="gray", vmin=0, vmax=255,
                      aspect="auto", interpolation="nearest")

        # Draw the analysed band as a transparent rectangle
        band_rect = mpatches.Rectangle(
            (0, r0 - 0.5), w, r1 - r0,
            linewidth=1, edgecolor="lime", facecolor="lime", alpha=0.25,
        )
        ax_img.add_patch(band_rect)

        # Draw vertical line at detected edge
        edge_x = res["edge_center_x"]
        ax_img.axvline(x=edge_x, color="red", linewidth=1.2, linestyle="--", alpha=0.85)

        ax_img.set_title(f"{title}\nedge @ x={edge_x:.1f} px", fontsize=9)
        ax_img.set_xlabel("col px")
        ax_img.set_ylabel("row px")
        ax_img.tick_params(labelsize=7)

        # --- Right subplot: 1D profile + fit ---
        ax_prof = axes[row, 1]
        profile = res["profile"]
        xs = np.arange(len(profile))

        ax_prof.plot(xs, profile, color=colour, linewidth=1.3, label="Band avg")
        ax_prof.axvline(x=edge_x, color="red", linewidth=1.0, linestyle="--",
                        alpha=0.8, label=f"Edge x={edge_x:.1f}")

        if res["fit_xs"] is not None and res["fit_ys"] is not None:
            ax_prof.plot(res["fit_xs"], res["fit_ys"], color="magenta",
                         linewidth=1.8, linestyle=":", alpha=0.9,
                         label=f"ERF fit sigma={res['estimated_sigma_px']:.3f} px")
        elif not math.isnan(res["estimated_sigma_px"]):
            ax_prof.text(0.98, 0.95,
                         f"sigma~{res['estimated_sigma_px']:.3f} px\n(10-90% proxy)",
                         transform=ax_prof.transAxes, fontsize=8,
                         ha="right", va="top",
                         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        ax_prof.set_xlabel("col px (model space)", fontsize=8)
        ax_prof.set_ylabel("Intensity", fontsize=8)
        ax_prof.set_title(f"{title} — centre-band profile\n{res['method_used']}",
                          fontsize=9)
        ax_prof.legend(fontsize=7, loc="lower left")
        ax_prof.grid(True, alpha=0.3)
        ax_prof.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Blur measurement CSV
# ---------------------------------------------------------------------------

def save_blur_measurement_csv(out_path: str,
                               blur_results: dict,
                               calib_targets: dict,
                               chain: dict) -> None:
    """
    Save blur_measurement_report.csv with one row per image.

    blur_results: dict keys 'source', 'code', 'correct'
    calib_targets: dict from compute_calibration_targets
    chain: dict from compute_sigma_chain
    """
    measured_src    = blur_results["source"]["estimated_sigma_px"]
    measured_code   = blur_results["code"]["estimated_sigma_px"]
    measured_correct = blur_results["correct"]["estimated_sigma_px"]

    # Target sigmas for comparison (total model-space sigma from calibration)
    target_code    = calib_targets["sigma_target_model_code"]
    target_correct = calib_targets["sigma_target_model_correct"]

    # Measured added blur (inferred by quadrature from source)
    def added_quadrature(total, source):
        if math.isnan(total) or math.isnan(source):
            return float("nan")
        return math.sqrt(max(total**2 - source**2, 0.0))

    added_meas_code    = added_quadrature(measured_code,    measured_src)
    added_meas_correct = added_quadrature(measured_correct, measured_src)

    def abs_err(meas, tgt):
        if math.isnan(meas):
            return float("nan")
        return abs(meas - tgt)

    def pct_err(meas, tgt):
        if math.isnan(meas) or tgt == 0:
            return float("nan")
        return 100.0 * abs(meas - tgt) / tgt

    fieldnames = [
        "image_name", "method_used", "edge_center_x",
        "measured_sigma_px", "target_sigma_px",
        "absolute_error_px", "percent_error",
        "measured_added_blur_px", "theoretical_added_blur_px",
        "notes",
    ]

    rows = [
        {
            "image_name":             "source",
            "method_used":            blur_results["source"]["method_used"],
            "edge_center_x":          f"{blur_results['source']['edge_center_x']:.2f}",
            "measured_sigma_px":      f"{measured_src:.4f}" if not math.isnan(measured_src) else "nan",
            "target_sigma_px":        "",
            "absolute_error_px":      "",
            "percent_error":          "",
            "measured_added_blur_px": "",
            "theoretical_added_blur_px": "",
            "notes":                  blur_results["source"]["notes"],
        },
        {
            "image_name":             "blurred_code",
            "method_used":            blur_results["code"]["method_used"],
            "edge_center_x":          f"{blur_results['code']['edge_center_x']:.2f}",
            "measured_sigma_px":      f"{measured_code:.4f}" if not math.isnan(measured_code) else "nan",
            "target_sigma_px":        f"{target_code:.4f}",
            "absolute_error_px":      f"{abs_err(measured_code, target_code):.4f}" if not math.isnan(measured_code) else "nan",
            "percent_error":          f"{pct_err(measured_code, target_code):.2f}" if not math.isnan(measured_code) else "nan",
            "measured_added_blur_px": f"{added_meas_code:.4f}" if not math.isnan(added_meas_code) else "nan",
            "theoretical_added_blur_px": f"{chain['sigma_kernel_code']:.4f}",
            "notes":                  blur_results["code"]["notes"],
        },
        {
            "image_name":             "blurred_correct",
            "method_used":            blur_results["correct"]["method_used"],
            "edge_center_x":          f"{blur_results['correct']['edge_center_x']:.2f}",
            "measured_sigma_px":      f"{measured_correct:.4f}" if not math.isnan(measured_correct) else "nan",
            "target_sigma_px":        f"{target_correct:.4f}",
            "absolute_error_px":      f"{abs_err(measured_correct, target_correct):.4f}" if not math.isnan(measured_correct) else "nan",
            "percent_error":          f"{pct_err(measured_correct, target_correct):.2f}" if not math.isnan(measured_correct) else "nan",
            "measured_added_blur_px": f"{added_meas_correct:.4f}" if not math.isnan(added_meas_correct) else "nan",
            "theoretical_added_blur_px": f"{chain['sigma_kernel_correct']:.4f}",
            "notes":                  blur_results["correct"]["notes"],
        },
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Blur measurement CSV saved -> {out_path}")


# ---------------------------------------------------------------------------
# Multi-z full image pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline_for_z(z_mm: float, src_model: np.ndarray) -> dict:
    """
    Run the complete blur + ERF pipeline for a single z value.

    Returns a dict containing all computed/measured values plus the blurred images,
    suitable for building comparison figures and the summary CSV.
    """
    chain = compute_sigma_chain(
        rho=RHO_DIRECT, sigma_0=SIGMA_0, z_mm=z_mm,
        scale_px=SCALE_PX_PER_MM, scale_calib=SCALE_CALIB_PX_PER_MM,
        model_size=MODEL_SIZE_PX, ref_res=CALIB_REFERENCE_RESOLUTION,
        crop_size=CROP_SIZE_PX, native_blur_crop_px=NATIVE_BLUR_SIGMA_CROP_PX,
    )
    calib_targets = compute_calibration_targets(
        rho_direct=RHO_DIRECT, sigma_0=SIGMA_0, z_mm=z_mm,
        use_sigma0=USE_SIGMA0_IN_TARGET,
        scale_px_per_mm=SCALE_PX_PER_MM, scale_calib_px_per_mm=SCALE_CALIB_PX_PER_MM,
        model_size_px=MODEL_SIZE_PX, crop_size_px=CROP_SIZE_PX,
        calib_ref_res=CALIB_REFERENCE_RESOLUTION,
    )

    blurred_code,    code_applied    = apply_gaussian_blur(
        src_model, chain["sigma_kernel_code"],    RADIUS_FACTOR, SIGMA_THRESHOLD)
    blurred_correct, correct_applied = apply_gaussian_blur(
        src_model, chain["sigma_kernel_correct"], RADIUS_FACTOR, SIGMA_THRESHOLD)

    erf_kwargs = dict(
        band_half_height=CENTER_BAND_HALF_HEIGHT,
        search_window_half_width=EDGE_SEARCH_WINDOW_HALF_WIDTH,
        use_scipy_if_available=USE_SCIPY_ERF_FIT_IF_AVAILABLE,
    )
    blur_src     = estimate_blur_sigma(src_model,       label=f"src@{z_mm}",  **erf_kwargs)
    blur_code    = estimate_blur_sigma(blurred_code,    label=f"code@{z_mm}", **erf_kwargs)
    blur_correct = estimate_blur_sigma(blurred_correct, label=f"corr@{z_mm}", **erf_kwargs)

    msrc  = blur_src["estimated_sigma_px"]
    mcode = blur_code["estimated_sigma_px"]
    mcorr = blur_correct["estimated_sigma_px"]

    def added_q(total, src):
        if math.isnan(total) or math.isnan(src):
            return float("nan")
        return math.sqrt(max(total**2 - src**2, 0.0))

    return {
        "z_mm":                       z_mm,
        # chain values
        "sigma_kernel_code":          chain["sigma_kernel_code"],
        "sigma_kernel_correct":       chain["sigma_kernel_correct"],
        "sigma_native_model":         chain["sigma_native_model"],
        "code_threshold_hit":         chain["sigma_kernel_code"] <= SIGMA_THRESHOLD,
        "correct_threshold_hit":      chain["sigma_kernel_correct"] <= SIGMA_THRESHOLD,
        # calibration targets (total model-space sigma)
        "sigma_target_model_code":    calib_targets["sigma_target_model_code"],
        "sigma_target_model_correct": calib_targets["sigma_target_model_correct"],
        # measured sigmas
        "measured_sigma_source":      msrc,
        "measured_sigma_code":        mcode,
        "measured_sigma_correct":     mcorr,
        "method_source":              blur_src["method_used"],
        "method_code":                blur_code["method_used"],
        "method_correct":             blur_correct["method_used"],
        # added blur by quadrature
        "added_meas_code":            added_q(mcode, msrc),
        "added_meas_correct":         added_q(mcorr, msrc),
        # images (float32)
        "img_code":                   blurred_code,
        "img_correct":                blurred_correct,
        "code_applied":               code_applied,
        "correct_applied":            correct_applied,
    }


def save_multi_z_panel(src_model: np.ndarray, z_results: list,
                       out_path: str, tile_size: int = 64) -> None:
    """
    Save a grid figure comparing blurred images across all z values.

    Layout (rows × columns):
      Row 0 : source (repeated, for reference)
      Row 1 : code-blurred at each z
      Row 2 : corrected-blurred at each z
      Row 3 : |corrected - code| diff at each z
    Columns : one per z value in z_results.
    """
    n_z = len(z_results)
    n_rows = 4
    label_h = 18   # px for text header per row
    col_label_h = 22

    fig_h = n_rows * (tile_size + label_h) + col_label_h
    fig_w = n_z * tile_size

    canvas = np.zeros((fig_h, fig_w), dtype=np.uint8)

    def _put(img_f32, row, col):
        """Place a rescaled uint8 tile into the canvas."""
        img8 = np.clip(img_f32, 0, 255).astype(np.uint8)
        if img8.shape[0] != tile_size or img8.shape[1] != tile_size:
            img8 = cv2.resize(img8, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        y0 = col_label_h + row * (tile_size + label_h) + label_h
        x0 = col * tile_size
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = img8

    def _text(text, row, col, dy=0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.28, 1
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        tx = col * tile_size + max(0, (tile_size - tw) // 2)
        ty = col_label_h + row * (tile_size + label_h) + dy + 12
        cv2.putText(canvas, text, (tx, ty), font, scale, 200, thick, cv2.LINE_AA)

    # Column header: z values
    for ci, r in enumerate(z_results):
        z_str = f"z={r['z_mm']:.1f}mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.30, 1
        (tw, _), _ = cv2.getTextSize(z_str, font, scale, thick)
        tx = ci * tile_size + max(0, (tile_size - tw) // 2)
        cv2.putText(canvas, z_str, (tx, 14), font, scale, 220, thick, cv2.LINE_AA)

    # Row labels + tiles
    row_labels = ["source", "code", "corrected", "|corr-code|"]
    for ci, r in enumerate(z_results):
        diff_img = absdiff(r["img_correct"], r["img_code"])

        src_tile = src_model
        code_tile  = r["img_code"]
        corr_tile  = r["img_correct"]
        diff_tile  = diff_img

        for ri, (tile, row_lbl) in enumerate(zip(
                [src_tile, code_tile, corr_tile, diff_tile], row_labels)):
            _put(tile, ri, ci)
            if ci == 0:
                _text(row_lbl, ri, ci, dy=-1)

    out_p = out_path
    cv2.imwrite(out_p, canvas)


def save_multi_z_sigma_plot(z_results: list, out_path: str) -> None:
    """
    Save a plot of sigma vs z comparing:
      - theoretical calibration targets (code and correct)
      - measured sigma from images (code and correct)
      - native model sigma (constant)
      - threshold line
    """
    zs = [r["z_mm"] for r in z_results]

    tgt_code    = [r["sigma_target_model_code"]    for r in z_results]
    tgt_correct = [r["sigma_target_model_correct"] for r in z_results]
    meas_code   = [r["measured_sigma_code"]        for r in z_results]
    meas_correct= [r["measured_sigma_correct"]     for r in z_results]
    native      = [r["sigma_native_model"]         for r in z_results]

    # Replace nan with None so matplotlib skips them
    def _clean(vals):
        return [v if not math.isnan(v) else None for v in vals]

    meas_code_c    = _clean(meas_code)
    meas_correct_c = _clean(meas_correct)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: total model-space sigma ---
    ax = axes[0]
    ax.plot(zs, tgt_code,    color="tab:blue",   linestyle="--", linewidth=1.8,
            label="Target (code formula)")
    ax.plot(zs, tgt_correct, color="tab:orange", linestyle="--", linewidth=1.8,
            label="Target (corrected formula)")
    ax.plot(zs, [v for v in meas_code_c],    color="tab:blue",   linestyle="-",
            marker="o", markersize=6, linewidth=1.4, label="Measured (code)")
    ax.plot(zs, [v for v in meas_correct_c], color="tab:orange", linestyle="-",
            marker="s", markersize=6, linewidth=1.4, label="Measured (corrected)")
    ax.plot(zs, native, color="black", linestyle=":", linewidth=1.2,
            label=f"Native model sigma ({native[0]:.3f} px)")
    ax.axhline(y=SIGMA_THRESHOLD, color="red", linestyle="-.", linewidth=1.0,
               alpha=0.7, label=f"Threshold ({SIGMA_THRESHOLD} px)")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Total model-space sigma (px)")
    ax.set_title("Total sigma: target vs measured")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- Right: error between measured and target ---
    ax2 = axes[1]
    err_code    = [abs(m - t) if m is not None else None
                   for m, t in zip(meas_code_c, tgt_code)]
    err_correct = [abs(m - t) if m is not None else None
                   for m, t in zip(meas_correct_c, tgt_correct)]

    ax2.plot(zs, err_code,    color="tab:blue",   linestyle="-", marker="o",
             markersize=6, linewidth=1.4, label="|measured - target| (code)")
    ax2.plot(zs, err_correct, color="tab:orange", linestyle="-", marker="s",
             markersize=6, linewidth=1.4, label="|measured - target| (corrected)")

    # Also plot the gap between code and corrected targets
    gap = [c - k for c, k in zip(tgt_correct, tgt_code)]
    ax2.plot(zs, gap, color="gray", linestyle="--", linewidth=1.2,
             label="Target gap (correct - code)")

    ax2.set_xlabel("z (mm)")
    ax2.set_ylabel("Absolute error (px)")
    ax2.set_title("Measurement error vs calibration target")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.suptitle("Multi-z blur comparison: calibration target vs measured sigma",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_multi_z_csv(z_results: list, out_path: str) -> None:
    """Save per-z blur measurement summary to CSV."""
    fieldnames = [
        "z_mm",
        "sigma_target_model_code", "sigma_target_model_correct",
        "sigma_native_model",
        "sigma_kernel_code", "sigma_kernel_correct",
        "code_threshold_hit", "correct_threshold_hit",
        "measured_sigma_source", "measured_sigma_code", "measured_sigma_correct",
        "method_source", "method_code", "method_correct",
        "added_meas_code", "added_meas_correct",
        "err_code_vs_target", "err_correct_vs_target",
        "err_code_px", "err_correct_px",
    ]

    rows = []
    for r in z_results:
        ms   = r["measured_sigma_source"]
        mc   = r["measured_sigma_code"]
        mcor = r["measured_sigma_correct"]
        tc   = r["sigma_target_model_code"]
        tcor = r["sigma_target_model_correct"]

        err_c   = abs(mc   - tc)   if not math.isnan(mc)   else float("nan")
        err_cor = abs(mcor - tcor) if not math.isnan(mcor) else float("nan")
        pct_c   = 100.0 * err_c   / tc   if (tc   > 0 and not math.isnan(err_c))   else float("nan")
        pct_cor = 100.0 * err_cor / tcor if (tcor > 0 and not math.isnan(err_cor)) else float("nan")

        rows.append({
            "z_mm":                        f"{r['z_mm']:.4f}",
            "sigma_target_model_code":     f"{tc:.4f}",
            "sigma_target_model_correct":  f"{tcor:.4f}",
            "sigma_native_model":          f"{r['sigma_native_model']:.4f}",
            "sigma_kernel_code":           f"{r['sigma_kernel_code']:.4f}",
            "sigma_kernel_correct":        f"{r['sigma_kernel_correct']:.4f}",
            "code_threshold_hit":          str(r["code_threshold_hit"]),
            "correct_threshold_hit":       str(r["correct_threshold_hit"]),
            "measured_sigma_source":       _fmt(ms),
            "measured_sigma_code":         _fmt(mc),
            "measured_sigma_correct":      _fmt(mcor),
            "method_source":               r["method_source"],
            "method_code":                 r["method_code"],
            "method_correct":              r["method_correct"],
            "added_meas_code":             _fmt(r["added_meas_code"]),
            "added_meas_correct":          _fmt(r["added_meas_correct"]),
            "err_code_vs_target":          f"{pct_c:.2f}%" if not math.isnan(pct_c) else "n/a",
            "err_correct_vs_target":       f"{pct_cor:.2f}%" if not math.isnan(pct_cor) else "n/a",
            "err_code_px":                 _fmt(err_c),
            "err_correct_px":              _fmt(err_cor),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Multi-z summary CSV saved -> {out_path}")


# ---------------------------------------------------------------------------
# Extended parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(rho: float, sigma_0: float,
                    z_list: list,
                    scale_px: float, scale_calib: float,
                    model_size: int, ref_res: int, crop_size: int,
                    native_blur_crop_px: float,
                    sigma_threshold: float,
                    use_sigma0_in_target: bool,
                    out_path: str) -> None:
    """
    Generate a CSV with sigma values across a list of z_mm values.
    Extended with calibration-target columns and predicted total sigma.
    """
    fieldnames = [
        "z_mm",
        # Calibration target columns (total sigma, not kernel)
        "sigma_calib_total",
        "sigma_exp_total",
        "sigma_target_model_code",
        "sigma_target_model_correct",
        # Chain columns (kept for compatibility, same as sigma_target_model_* above
        # but computed with sigma_0 always included)
        "sigma_calib_px",
        "sigma_exp_px",
        # Model-space values
        "sigma_native_model",
        "sigma_kernel_code",
        "sigma_kernel_correct",
        "target_ratio",
        "kernel_ratio",
        "code_threshold_hit",
        "correct_threshold_hit",
        # Predicted total sigma assuming perfect quadrature
        "predicted_total_sigma_code",
        "predicted_total_sigma_correct",
    ]

    rows = []
    for z in z_list:
        c = compute_sigma_chain(rho, sigma_0, z, scale_px, scale_calib,
                                model_size, ref_res, crop_size, native_blur_crop_px)
        ct = compute_calibration_targets(rho, sigma_0, z, use_sigma0_in_target,
                                         scale_px, scale_calib,
                                         model_size, crop_size, ref_res)

        # predicted total = sqrt(native^2 + kernel^2) = target by construction
        pred_total_code    = math.sqrt(c["sigma_native_model"]**2 + c["sigma_kernel_code"]**2)
        pred_total_correct = math.sqrt(c["sigma_native_model"]**2 + c["sigma_kernel_correct"]**2)

        kernel_ratio_str = (f"{c['kernel_ratio']:.4f}"
                            if not math.isinf(c["kernel_ratio"]) else "inf")

        rows.append({
            "z_mm":                       f"{z:.4f}",
            "sigma_calib_total":          f"{ct['sigma_calib_total']:.4f}",
            "sigma_exp_total":            f"{ct['sigma_exp_total']:.4f}",
            "sigma_target_model_code":    f"{ct['sigma_target_model_code']:.4f}",
            "sigma_target_model_correct": f"{ct['sigma_target_model_correct']:.4f}",
            "sigma_calib_px":             f"{c['sigma_calib']:.4f}",
            "sigma_exp_px":               f"{c['sigma_exp']:.4f}",
            "sigma_native_model":         f"{c['sigma_native_model']:.4f}",
            "sigma_kernel_code":          f"{c['sigma_kernel_code']:.4f}",
            "sigma_kernel_correct":       f"{c['sigma_kernel_correct']:.4f}",
            "target_ratio":               f"{c['target_ratio']:.4f}",
            "kernel_ratio":               kernel_ratio_str,
            "code_threshold_hit":         str(c["sigma_kernel_code"] <= sigma_threshold),
            "correct_threshold_hit":      str(c["sigma_kernel_correct"] <= sigma_threshold),
            "predicted_total_sigma_code":    f"{pred_total_code:.4f}",
            "predicted_total_sigma_correct": f"{pred_total_correct:.4f}",
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Sweep CSV saved -> {out_path}")

    # Console summary table
    header = (f"{'z':>5} | {'tgt_code':>9} {'tgt_corr':>9} {'nat':>6} "
              f"{'k_code':>7} {'k_corr':>7} {'t_ratio':>8} | hit_c  hit_ok")
    print()
    print("  Parameter Sweep Summary")
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for r in rows:
        print(
            f"  {float(r['z_mm']):>5.1f} | "
            f"{float(r['sigma_target_model_code']):>9.4f} "
            f"{float(r['sigma_target_model_correct']):>9.4f} "
            f"{float(r['sigma_native_model']):>6.4f} "
            f"{float(r['sigma_kernel_code']):>7.4f} "
            f"{float(r['sigma_kernel_correct']):>7.4f} "
            f"{r['target_ratio']:>8} | "
            f"{r['code_threshold_hit']:<6}  {r['correct_threshold_hit']}"
        )
    print("  " + "-" * len(header))


# ---------------------------------------------------------------------------
# Extended report builder
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".4f"):
    """Format a float or return 'n/a' for nan."""
    if isinstance(val, float) and math.isnan(val):
        return "n/a"
    return format(val, fmt)


def _build_calib_vs_measured_section(calib_targets: dict, chain: dict,
                                      blur_results: dict) -> str:
    """
    Build the 'Calibration Target vs Measured Blur' section of the text report.
    Returns a formatted string.
    """
    measured_src     = blur_results["source"]["estimated_sigma_px"]
    measured_code    = blur_results["code"]["estimated_sigma_px"]
    measured_correct = blur_results["correct"]["estimated_sigma_px"]

    def added_q(total, src):
        if math.isnan(total) or math.isnan(src):
            return float("nan")
        return math.sqrt(max(total**2 - src**2, 0.0))

    added_meas_code    = added_q(measured_code,    measured_src)
    added_meas_correct = added_q(measured_correct, measured_src)

    sk_code    = chain["sigma_kernel_code"]
    sk_correct = chain["sigma_kernel_correct"]
    tm_code    = calib_targets["sigma_target_model_code"]
    tm_correct = calib_targets["sigma_target_model_correct"]

    # --- Interpretation logic ---
    interp = []

    # 1. Corrected formula predicts substantially more blur?
    if tm_correct > 1.5 * tm_code:
        interp.append(
            f"[YES] The corrected formula predicts substantially MORE total model-space blur "
            f"({tm_correct:.4f} px) than the current-code formula ({tm_code:.4f} px) — "
            f"a factor of {tm_correct/tm_code:.2f}x. This is expected from the calib_ref/crop_size "
            f"ratio ({CALIB_REFERENCE_RESOLUTION}/{CROP_SIZE_PX} = "
            f"{CALIB_REFERENCE_RESOLUTION/CROP_SIZE_PX:.3f})."
        )
    elif tm_correct > tm_code:
        interp.append(
            f"[YES] Corrected formula predicts more blur ({tm_correct:.4f} vs {tm_code:.4f} px) "
            f"but the difference is modest (factor {tm_correct/tm_code:.2f}x)."
        )
    else:
        interp.append(
            f"[NO] Corrected formula does NOT predict more blur than current code. "
            f"Check parameter values."
        )

    # 2. Is measured blur of corrected image closer to its target?
    if not (math.isnan(measured_code) or math.isnan(measured_correct)):
        err_code    = abs(measured_code    - tm_code)
        err_correct = abs(measured_correct - tm_correct)
        if err_correct < err_code:
            interp.append(
                f"[YES] Measured blur of the corrected image ({measured_correct:.4f} px) is "
                f"CLOSER to its calibration target ({tm_correct:.4f} px, error={err_correct:.4f} px) "
                f"than the code image ({measured_code:.4f} px) is to its target "
                f"({tm_code:.4f} px, error={err_code:.4f} px)."
            )
        else:
            interp.append(
                f"[NO] Measured blur of the corrected image ({measured_correct:.4f} px) is NOT "
                f"closer to its target ({tm_correct:.4f} px, error={err_correct:.4f} px) than "
                f"the code image ({measured_code:.4f} px, error={err_code:.4f} px). "
                f"This may indicate ERF estimation noise — inspect edge_fit_profiles.png."
            )
    else:
        interp.append("[INFO] Cannot compare targets vs measured: blur estimation returned nan.")

    # 3. Current-code under-blurred?
    if not math.isnan(measured_code):
        if measured_code < tm_correct * 0.7:
            interp.append(
                f"[YES] Current-code image appears UNDER-BLURRED relative to the corrected "
                f"calibration target: measured={measured_code:.4f} px vs target={tm_correct:.4f} px "
                f"({100*(1 - measured_code/tm_correct):.1f}% below target)."
            )
        elif measured_code < tm_correct:
            interp.append(
                f"[MILD] Current-code image is slightly below the corrected target "
                f"({measured_code:.4f} vs {tm_correct:.4f} px)."
            )
        else:
            interp.append(
                f"[NO] Current-code measured blur ({measured_code:.4f} px) is not obviously "
                f"below corrected target ({tm_correct:.4f} px)."
            )

    # 4. Threshold suppressed blur in code path?
    code_thr_hit = chain["sigma_kernel_code"] <= SIGMA_THRESHOLD
    if code_thr_hit:
        interp.append(
            f"[YES] The sigma threshold suppressed ALL additional blur in the current-code path "
            f"(sigma_kernel_code={sk_code:.4f} <= threshold={SIGMA_THRESHOLD}). "
            f"The code-blurred image is IDENTICAL to the source at this z."
        )
    else:
        interp.append(
            f"[NOTE] Code path did apply some blur (sigma_kernel_code={sk_code:.4f} px > threshold)."
        )

    # 5. Overall under-blurring hypothesis
    correct_thr_hit = chain["sigma_kernel_correct"] <= SIGMA_THRESHOLD
    if tm_correct > tm_code and (code_thr_hit or measured_code < 0.8 * tm_correct):
        if not correct_thr_hit:
            interp.append(
                f"\n[CONCLUSION] Results SUPPORT the hypothesis that the training generator "
                f"under-blurs images relative to calibration expectations. "
                f"At z={chain['z_mm']} mm, the corrected pipeline would apply "
                f"sigma_kernel={sk_correct:.4f} px of added blur, whereas the current code "
                f"{'applies zero blur (threshold hit)' if code_thr_hit else f'applies only {sk_code:.4f} px'}. "
                f"Training images generated with the current code will have systematically less "
                f"blur than the physical calibration predicts, biasing the model toward "
                f"over-estimating defocus on real images."
            )
        else:
            interp.append(
                f"\n[CONCLUSION] Both paths hit the sigma threshold at this z. "
                f"Try a larger z_mm to see meaningful blur differences."
            )
    else:
        interp.append(
            f"\n[NOTE] Evidence for under-blurring hypothesis is ambiguous at z={chain['z_mm']} mm. "
            f"Try larger z values or inspect the sweep CSV."
        )

    interp_text = "\n".join("  " + line for line in interp)

    section = textwrap.dedent(f"""\
    Calibration Target vs Measured Blur
    ------------------------------------
    [Calibration target — total model-space sigma at z={chain['z_mm']} mm]
    sigma_calib_total (calib-cam px)       : {_fmt(calib_targets['sigma_calib_total'])}
    sigma_exp_total (exp raw-frame px)     : {_fmt(calib_targets['sigma_exp_total'])}
    sigma_target_model_code (code formula) : {_fmt(calib_targets['sigma_target_model_code'])}
    sigma_target_model_correct (corr form) : {_fmt(calib_targets['sigma_target_model_correct'])}
    USE_SIGMA0_IN_TARGET                   : {USE_SIGMA0_IN_TARGET}

    [Measured sigma from image content — model-space px]
    Source image                           : {_fmt(measured_src)}
      method: {blur_results['source']['method_used']}
    Code-blurred image                     : {_fmt(measured_code)}
      method: {blur_results['code']['method_used']}
    Corrected-blurred image                : {_fmt(measured_correct)}
      method: {blur_results['correct']['method_used']}

    [Added blur — inferred by quadrature relative to source]
    sigma_added_measured_code              : {_fmt(added_meas_code)}
    sigma_added_measured_correct           : {_fmt(added_meas_correct)}

    [Theoretical added kernel sigma]
    sigma_kernel_code  (theoretical)       : {_fmt(sk_code)}
    sigma_kernel_correct (theoretical)     : {_fmt(sk_correct)}

    [Interpretation]
{interp_text}
    """)
    return section


def build_report(params: dict, chain: dict, edge_widths: dict,
                 sigma_threshold: float,
                 calib_targets: dict = None,
                 blur_results: dict = None) -> str:
    """Return a formatted report string. Extended with calibration/measured section."""
    code_threshold_hit    = chain["sigma_kernel_code"]    <= sigma_threshold
    correct_threshold_hit = chain["sigma_kernel_correct"] <= sigma_threshold

    factor = CALIB_REFERENCE_RESOLUTION / CROP_SIZE_PX

    interp_lines = []
    interp_lines.append("INTERPRETATION")
    interp_lines.append("-" * 60)

    if chain["sigma_target_correct"] > chain["sigma_target_code"]:
        interp_lines.append(
            f"[YES] Corrected target sigma ({chain['sigma_target_correct']:.4f} px) is LARGER "
            f"than current-code sigma ({chain['sigma_target_code']:.4f} px) "
            f"by a factor of {chain['target_ratio']:.3f}x."
        )
        interp_lines.append(
            f"      This factor is driven by calib_ref_res / crop_size = "
            f"{CALIB_REFERENCE_RESOLUTION} / {CROP_SIZE_PX} = {factor:.3f}."
        )
    else:
        interp_lines.append(
            f"[NO]  Corrected target sigma is NOT larger. "
            f"Check parameter values — this is unexpected."
        )

    if code_threshold_hit:
        interp_lines.append(
            f"[YES] Current-code kernel sigma ({chain['sigma_kernel_code']:.4f} px) is at or "
            f"below the threshold ({sigma_threshold} px). "
            f"NO blur is applied in the current-code path at z={chain['z_mm']} mm. "
            f"The training image is IDENTICAL to the source."
        )
    else:
        interp_lines.append(
            f"[NOTE] Current-code kernel sigma ({chain['sigma_kernel_code']:.4f} px) is above "
            f"threshold — some blur IS applied, but still {chain['kernel_ratio']:.2f}x less than corrected."
        )

    if correct_threshold_hit:
        interp_lines.append(
            f"[NOTE] Corrected kernel sigma ({chain['sigma_kernel_correct']:.4f} px) is also "
            f"at or below threshold. Even the corrected path applies no extra blur here. "
            f"Consider a larger z_mm value."
        )
    else:
        interp_lines.append(
            f"[YES] Corrected kernel sigma ({chain['sigma_kernel_correct']:.4f} px) is above "
            f"threshold — meaningful blur IS applied in the corrected path."
        )

    if chain["sigma_target_correct"] > chain["sigma_target_code"] and not correct_threshold_hit:
        interp_lines.append(
            f"\n[CONCLUSION] Results SUPPORT the hypothesis that the training generator is "
            f"under-blurring synthetic images. At z={chain['z_mm']} mm, the corrected target "
            f"sigma is {chain['target_ratio']:.3f}x larger than what the current code produces. "
            f"Images generated with the current code logic have significantly less blur than "
            f"the physical calibration predicts, which will cause the trained model to "
            f"systematically over-estimate defocus on real images."
        )
    elif code_threshold_hit and not correct_threshold_hit:
        interp_lines.append(
            f"\n[CONCLUSION] Results STRONGLY support the under-blurring hypothesis. "
            f"Current code applies ZERO additional blur at this defocus, "
            f"while the corrected formula applies sigma_kernel = {chain['sigma_kernel_correct']:.4f} px. "
            f"Any training sample at z={chain['z_mm']} mm or below (where code threshold hits) "
            f"will be an unblurred image paired with a non-zero defocus label."
        )
    else:
        interp_lines.append(
            f"\n[NOTE] Results are ambiguous at this z value. Try larger z_mm values for clearer evidence."
        )

    interpretation = "\n".join("  " + l for l in interp_lines)

    # Optional calibration-target section
    calib_section = ""
    if calib_targets is not None and blur_results is not None:
        calib_section = "\n" + _build_calib_vs_measured_section(
            calib_targets, chain, blur_results)

    report = textwrap.dedent(f"""\
    ================================================================
    Blur Scaling Sanity Check — Report
    ================================================================

    INPUT PARAMETERS
    ----------------
    image_path                 : {params['image_path']}
    z_mm                       : {params['z_mm']} mm
    rho_direct                 : {params['rho_direct']} px/mm  (calibration camera)
    sigma_0                    : {params['sigma_0']} px
    scale_px_per_mm            : {params['scale_px_per_mm']} px/mm  (experiment camera)
    scale_calib_px_per_mm      : {params['scale_calib_px_per_mm']} px/mm  (calibration camera)
    calib_reference_resolution : {params['calib_reference_resolution']} px
    crop_size_px               : {params['crop_size_px']} px
    model_size_px              : {params['model_size_px']} px
    native_blur_sigma_crop_px  : {params['native_blur_sigma_crop_px']} px  (in 299px crop units)
    radius_factor              : {params['radius_factor']}
    sigma_threshold            : {params['sigma_threshold']}

    COMPUTED VALUES
    ---------------
    cross_camera_ratio         : {chain['cross_camera_ratio']:.6f}
      = scale_px_per_mm / scale_calib_px_per_mm
      = {params['scale_px_per_mm']} / {params['scale_calib_px_per_mm']}

    sigma_calib (rho*|z|+s0)   : {chain['sigma_calib']:.6f} px  (calibration-camera px)
    sigma_exp (after correction): {chain['sigma_exp']:.6f} px  (experiment raw-frame px)

    sigma_native_model          : {chain['sigma_native_model']:.6f} px
      = native_blur_sigma_crop_px * (model_size / crop_size)
      = {params['native_blur_sigma_crop_px']} * ({params['model_size_px']} / {params['crop_size_px']})

    CURRENT CODE SCALING  [sigma_exp * (model_size / calib_ref_res)]
    ----------------------------------------------------------------
    sigma_target_code_model     : {chain['sigma_target_code']:.6f} px
      = {chain['sigma_exp']:.6f} * ({params['model_size_px']} / {params['calib_reference_resolution']})
    sigma_kernel_code           : {chain['sigma_kernel_code']:.6f} px
      = sqrt(max({chain['sigma_target_code']:.4f}^2 - {chain['sigma_native_model']:.4f}^2, 0))
    Threshold hit               : {code_threshold_hit}  (kernel <= {params['sigma_threshold']} -> NO blur applied)

    CORRECTED SCALING  [sigma_exp * (model_size / crop_size)]
    ----------------------------------------------------------
    sigma_target_correct_model  : {chain['sigma_target_correct']:.6f} px
      = {chain['sigma_exp']:.6f} * ({params['model_size_px']} / {params['crop_size_px']})
    sigma_kernel_correct        : {chain['sigma_kernel_correct']:.6f} px
      = sqrt(max({chain['sigma_target_correct']:.4f}^2 - {chain['sigma_native_model']:.4f}^2, 0))
    Threshold hit               : {correct_threshold_hit}  (kernel <= {params['sigma_threshold']} -> NO blur applied)

    RATIOS
    ------
    sigma_target_correct / sigma_target_code : {chain['target_ratio']:.6f}x
    sigma_kernel_correct / sigma_kernel_code : {'inf (code kernel is zero/threshold)' if math.isinf(chain['kernel_ratio']) else f"{chain['kernel_ratio']:.6f}x"}
    Expected factor (calib_ref / crop_size)  : {CALIB_REFERENCE_RESOLUTION}/{CROP_SIZE_PX} = {CALIB_REFERENCE_RESOLUTION/CROP_SIZE_PX:.6f}

    EDGE-WIDTH ESTIMATES (10-90% rise, centreline, model-px)
    ---------------------------------------------------------
    Source                      : {edge_widths.get('source', float('nan')):.2f} px
    Code-blurred                : {edge_widths.get('code', float('nan')):.2f} px
    Corrected-blurred           : {edge_widths.get('correct', float('nan')):.2f} px

{interpretation}
{calib_section}
    ================================================================
    """)
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # Force UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Blur Scaling Sanity Check")
    print("=" * 60)
    if SCIPY_AVAILABLE:
        print("  [scipy] Available — ERF fitting enabled.")
    else:
        print("  [scipy] NOT installed — using 10-90% width fallback for blur estimation.")

    ensure_dir(OUTPUT_DIR)

    # ── [1] Load and prepare source image ─────────────────────────────────────
    print(f"\n[1] Loading source image: {IMAGE_PATH}")
    try:
        src_raw = load_grayscale(IMAGE_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"    Raw size: {src_raw.shape[1]}x{src_raw.shape[0]} px")

    src_crop = resize_to_crop(src_raw, CROP_SIZE_PX)
    print(f"    After crop-resize: {src_crop.shape[1]}x{src_crop.shape[0]} px")

    src_model = resize_to_model(src_crop, MODEL_SIZE_PX, CROP_SIZE_PX)
    print(f"    After model-resize: {src_model.shape[1]}x{src_model.shape[0]} px")

    label = f"_{IMAGE_LABEL}" if IMAGE_LABEL else ""
    src_out = os.path.join(OUTPUT_DIR, f"source_crop_resized_{MODEL_SIZE_PX}{label}.png")
    save_uint8(src_model, src_out)
    print(f"    Saved -> {src_out}")

    # ── [2] Compute sigma chain ────────────────────────────────────────────────
    print(f"\n[2] Computing sigma chain for z = {Z_MM} mm ...")
    chain = compute_sigma_chain(
        rho=RHO_DIRECT,
        sigma_0=SIGMA_0,
        z_mm=Z_MM,
        scale_px=SCALE_PX_PER_MM,
        scale_calib=SCALE_CALIB_PX_PER_MM,
        model_size=MODEL_SIZE_PX,
        ref_res=CALIB_REFERENCE_RESOLUTION,
        crop_size=CROP_SIZE_PX,
        native_blur_crop_px=NATIVE_BLUR_SIGMA_CROP_PX,
    )

    print(f"    sigma_target_code    = {chain['sigma_target_code']:.4f} px")
    print(f"    sigma_target_correct = {chain['sigma_target_correct']:.4f} px")
    print(f"    sigma_native_model   = {chain['sigma_native_model']:.4f} px")
    print(f"    sigma_kernel_code    = {chain['sigma_kernel_code']:.4f} px")
    print(f"    sigma_kernel_correct = {chain['sigma_kernel_correct']:.4f} px")
    print(f"    target ratio (corr/code) = {chain['target_ratio']:.4f}x")

    # ── [2b] Compute calibration targets ──────────────────────────────────────
    print(f"\n[2b] Computing calibration targets (USE_SIGMA0_IN_TARGET={USE_SIGMA0_IN_TARGET}) ...")
    calib_targets = compute_calibration_targets(
        rho_direct=RHO_DIRECT,
        sigma_0=SIGMA_0,
        z_mm=Z_MM,
        use_sigma0=USE_SIGMA0_IN_TARGET,
        scale_px_per_mm=SCALE_PX_PER_MM,
        scale_calib_px_per_mm=SCALE_CALIB_PX_PER_MM,
        model_size_px=MODEL_SIZE_PX,
        crop_size_px=CROP_SIZE_PX,
        calib_ref_res=CALIB_REFERENCE_RESOLUTION,
    )
    print(f"    sigma_calib_total        = {calib_targets['sigma_calib_total']:.4f} px  (calib-cam)")
    print(f"    sigma_exp_total          = {calib_targets['sigma_exp_total']:.4f} px  (exp raw-frame)")
    print(f"    sigma_target_model_code  = {calib_targets['sigma_target_model_code']:.4f} px  (model-space, code formula)")
    print(f"    sigma_target_model_corr  = {calib_targets['sigma_target_model_correct']:.4f} px  (model-space, corrected formula)")

    # ── [3] Apply blur ─────────────────────────────────────────────────────────
    print(f"\n[3] Applying blur ...")
    blurred_code, code_applied = apply_gaussian_blur(
        src_model, chain["sigma_kernel_code"], RADIUS_FACTOR, SIGMA_THRESHOLD)
    print(f"    Code path:    sigma_kernel={chain['sigma_kernel_code']:.4f}  applied={code_applied}")

    blurred_correct, correct_applied = apply_gaussian_blur(
        src_model, chain["sigma_kernel_correct"], RADIUS_FACTOR, SIGMA_THRESHOLD)
    print(f"    Correct path: sigma_kernel={chain['sigma_kernel_correct']:.4f}  applied={correct_applied}")

    # ── [4] Save individual images ─────────────────────────────────────────────
    print(f"\n[4] Saving output images ...")

    code_out    = os.path.join(OUTPUT_DIR, f"blurred_code{label}.png")
    correct_out = os.path.join(OUTPUT_DIR, f"blurred_correct{label}.png")
    save_uint8(blurred_code,    code_out)
    save_uint8(blurred_correct, correct_out)
    print(f"    Saved -> {code_out}")
    print(f"    Saved -> {correct_out}")

    diff_code_src     = absdiff(blurred_code,    src_model)
    diff_correct_src  = absdiff(blurred_correct, src_model)
    diff_correct_code = absdiff(blurred_correct, blurred_code)

    d1 = os.path.join(OUTPUT_DIR, f"absdiff_code_vs_source{label}.png")
    d2 = os.path.join(OUTPUT_DIR, f"absdiff_correct_vs_source{label}.png")
    d3 = os.path.join(OUTPUT_DIR, f"absdiff_correct_vs_code{label}.png")
    save_uint8(diff_code_src,     d1)
    save_uint8(diff_correct_src,  d2)
    save_uint8(diff_correct_code, d3)
    print(f"    Saved -> {d1}")
    print(f"    Saved -> {d2}")
    print(f"    Saved -> {d3}")

    # ── [5] Comparison panel ───────────────────────────────────────────────────
    print(f"\n[5] Building comparison panel ...")

    def to_uint8(f: np.ndarray) -> np.ndarray:
        return np.clip(f, 0, 255).astype(np.uint8)

    tiles = [
        to_uint8(src_model),
        to_uint8(blurred_code),
        to_uint8(blurred_correct),
        to_uint8(diff_code_src),
        to_uint8(diff_correct_src),
        to_uint8(diff_correct_code),
    ]
    tile_labels = [
        "Source",
        "Code blur",
        "Corrected",
        "|code-src|",
        "|corr-src|",
        "|corr-code|",
    ]
    panel = build_comparison_panel(tiles, tile_labels, tile_size=MODEL_SIZE_PX)
    panel_out = os.path.join(OUTPUT_DIR, f"comparison_panel{label}.png")
    save_uint8(panel, panel_out)
    print(f"    Saved -> {panel_out}")

    # ── [6] Legacy centreline profile plot (preserved) ────────────────────────
    print(f"\n[6] Generating centreline profile plot ...")
    profile_out = os.path.join(OUTPUT_DIR, f"centerline_profiles{label}.png")
    edge_widths = centerline_profiles(
        src_model, blurred_code, blurred_correct, profile_out)
    print(f"    Edge widths (10-90%, model px):")
    print(f"      Source:    {edge_widths.get('source', float('nan')):.2f} px")
    print(f"      Code:      {edge_widths.get('code',   float('nan')):.2f} px")
    print(f"      Corrected: {edge_widths.get('correct',float('nan')):.2f} px")
    print(f"    Saved -> {profile_out}")

    # ── [6b] ERF blur estimation ───────────────────────────────────────────────
    print(f"\n[6b] Estimating blur sigma from image content (ERF / 10-90% fallback) ...")
    _erf_kwargs = dict(
        band_half_height=CENTER_BAND_HALF_HEIGHT,
        search_window_half_width=EDGE_SEARCH_WINDOW_HALF_WIDTH,
        use_scipy_if_available=USE_SCIPY_ERF_FIT_IF_AVAILABLE,
    )
    blur_src     = estimate_blur_sigma(src_model,      label="source",  **_erf_kwargs)
    blur_code    = estimate_blur_sigma(blurred_code,   label="code",    **_erf_kwargs)
    blur_correct = estimate_blur_sigma(blurred_correct, label="correct", **_erf_kwargs)

    blur_results = {"source": blur_src, "code": blur_code, "correct": blur_correct}

    for key, res in blur_results.items():
        sig = res["estimated_sigma_px"]
        print(f"    {key:<10}: method={res['method_used']:<12} "
              f"sigma={_fmt(sig)} px   edge_x={res['edge_center_x']:.1f}")
        print(f"               {res['notes']}")

    # ── [6c] Edge-fit figure — profiles ───────────────────────────────────────
    print(f"\n[6c] Saving edge-fit profile plot ...")
    erf_profiles_out = os.path.join(OUTPUT_DIR, f"edge_fit_profiles{label}.png")
    save_edge_fit_profiles(blur_results, erf_profiles_out, calib_targets)
    print(f"    Saved -> {erf_profiles_out}")

    # ── [6d] Edge-fit diagnostics figure ──────────────────────────────────────
    print(f"\n[6d] Saving edge-fit diagnostics ...")
    images_dict = {"source": src_model, "code": blurred_code, "correct": blurred_correct}
    erf_diag_out = os.path.join(OUTPUT_DIR, f"edge_fit_diagnostics{label}.png")
    save_edge_fit_diagnostics(images_dict, blur_results, CENTER_BAND_HALF_HEIGHT, erf_diag_out)
    print(f"    Saved -> {erf_diag_out}")

    # ── [6e] Blur measurement CSV ─────────────────────────────────────────────
    print(f"\n[6e] Saving blur measurement CSV ...")
    csv_out = os.path.join(OUTPUT_DIR, f"blur_measurement_report{label}.csv")
    save_blur_measurement_csv(csv_out, blur_results, calib_targets, chain)

    # ── [7] Parameter sweep ───────────────────────────────────────────────────
    if SWEEP_ENABLED:
        print(f"\n[7] Running parameter sweep over z = {SWEEP_Z_MM} mm ...")
        sweep_out = os.path.join(OUTPUT_DIR, f"sweep{label}.csv")
        parameter_sweep(
            rho=RHO_DIRECT,
            sigma_0=SIGMA_0,
            z_list=SWEEP_Z_MM,
            scale_px=SCALE_PX_PER_MM,
            scale_calib=SCALE_CALIB_PX_PER_MM,
            model_size=MODEL_SIZE_PX,
            ref_res=CALIB_REFERENCE_RESOLUTION,
            crop_size=CROP_SIZE_PX,
            native_blur_crop_px=NATIVE_BLUR_SIGMA_CROP_PX,
            sigma_threshold=SIGMA_THRESHOLD,
            use_sigma0_in_target=USE_SIGMA0_IN_TARGET,
            out_path=sweep_out,
        )

    # ── [7b] Full-image sweep across all z values ─────────────────────────────
    if FULL_SWEEP_IMAGES:
        print(f"\n[7b] Running full image pipeline for z = {SWEEP_Z_MM} mm ...")
        z_results = []
        for z in SWEEP_Z_MM:
            print(f"     z = {z:.1f} mm ...", end="  ")
            r = run_full_pipeline_for_z(z, src_model)
            z_results.append(r)
            print(
                f"code: {_fmt(r['measured_sigma_code'])} px  "
                f"correct: {_fmt(r['measured_sigma_correct'])} px  "
                f"(targets: {r['sigma_target_model_code']:.4f} / "
                f"{r['sigma_target_model_correct']:.4f})"
            )

        # Comparison panel
        panel_path = os.path.join(OUTPUT_DIR, f"multi_z_panel{label}.png")
        save_multi_z_panel(src_model, z_results, panel_path, tile_size=96)
        print(f"    Multi-z panel saved -> {panel_path}")

        # Sigma vs z plot
        sigma_plot_path = os.path.join(OUTPUT_DIR, f"multi_z_sigma_comparison{label}.png")
        save_multi_z_sigma_plot(z_results, sigma_plot_path)
        print(f"    Multi-z sigma plot saved -> {sigma_plot_path}")

        # Summary CSV
        multi_z_csv_path = os.path.join(OUTPUT_DIR, f"multi_z_summary{label}.csv")
        save_multi_z_csv(z_results, multi_z_csv_path)

        # Console table
        print()
        hdr = (f"  {'z':>5} | {'tgt_c':>7} {'tgt_ok':>7} | "
               f"{'meas_c':>7} {'meas_ok':>7} | "
               f"{'err_c%':>7} {'err_ok%':>8} | thr_c thr_ok")
        print("  Full-image sweep results")
        print("  " + "-" * (len(hdr) - 2))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in z_results:
            tc   = r["sigma_target_model_code"]
            tcor = r["sigma_target_model_correct"]
            mc   = r["measured_sigma_code"]
            mcor = r["measured_sigma_correct"]
            pct_c   = 100.0 * abs(mc   - tc)   / tc   if (tc   > 0 and not math.isnan(mc))   else float("nan")
            pct_cor = 100.0 * abs(mcor - tcor)  / tcor if (tcor > 0 and not math.isnan(mcor)) else float("nan")
            print(
                f"  {r['z_mm']:>5.1f} | "
                f"{tc:>7.4f} {tcor:>7.4f} | "
                f"{_fmt(mc):>7} {_fmt(mcor):>7} | "
                f"{pct_c:>6.1f}% {pct_cor:>7.1f}% | "
                f"{'Y' if r['code_threshold_hit'] else 'N':^5} "
                f"{'Y' if r['correct_threshold_hit'] else 'N':^5}"
            )
        print("  " + "-" * (len(hdr) - 2))

    # ── [8] Generate and save report ──────────────────────────────────────────
    print(f"\n[8] Writing report ...")
    params_dict = {
        "image_path":                  IMAGE_PATH,
        "z_mm":                        Z_MM,
        "rho_direct":                  RHO_DIRECT,
        "sigma_0":                     SIGMA_0,
        "scale_px_per_mm":             SCALE_PX_PER_MM,
        "scale_calib_px_per_mm":       SCALE_CALIB_PX_PER_MM,
        "calib_reference_resolution":  CALIB_REFERENCE_RESOLUTION,
        "crop_size_px":                CROP_SIZE_PX,
        "model_size_px":               MODEL_SIZE_PX,
        "native_blur_sigma_crop_px":   NATIVE_BLUR_SIGMA_CROP_PX,
        "radius_factor":               RADIUS_FACTOR,
        "sigma_threshold":             SIGMA_THRESHOLD,
    }
    report_text = build_report(
        params_dict, chain, edge_widths, SIGMA_THRESHOLD,
        calib_targets=calib_targets,
        blur_results=blur_results,
    )

    print()
    print(report_text)

    report_path = os.path.join(OUTPUT_DIR, f"report{label}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"    Report saved -> {report_path}")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
