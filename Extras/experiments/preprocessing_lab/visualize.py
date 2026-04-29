"""Shared plotting + helper functions for the preprocessing lab.

Kept separate from the entry-point scripts so both
`run_mode_comparison.py` and `sweep_parameters.py` can use them
without duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

# Path setup so we can import project functions
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from cine_loader import CineLoader


CINE_DIR_DEFAULT = _REPO_ROOT / "calibration spheres" / "9mm"
POSITIONS_CSV_DEFAULT = CINE_DIR_DEFAULT / "positions.csv"
MODELS_DIR = _REPO_ROOT / "Training" / "training_output" / "models"
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "output"


def find_focal_plane(cines: List[Path], pos_map: dict) -> float:
    """Stage position with maximum Laplacian variance (= in focus).

    This is the same focal-plane detection the existing investigation
    scripts used; matches the calibration bundle's reference frame.
    """
    best_var = -np.inf
    best_stage = 0.0
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(
            loader._load_frame(loader.frame_range[0]), dtype=np.float32
        )
        h, w = raw.shape
        cy0, cx0 = h // 2, w // 2
        s = min(h, w) // 3
        lap = cv2.Laplacian(raw[cy0 - s:cy0 + s, cx0 - s:cx0 + s],
                             cv2.CV_32F, ksize=3)
        v = float(lap.var())
        if v > best_var:
            best_var = v
            best_stage = pos_map[cine.name]
    return best_stage


def load_calibration_stack(cine_dir: Path = None,
                            positions_csv: Path = None):
    """Load the user's calibration .cine stack + z positions.

    Returns: (raw_frames: list[np.ndarray], z_positions: list[float],
              cine_paths: list[Path])
    Z positions are referenced to the focal plane (which is auto-found
    via Laplacian variance).
    """
    cine_dir = cine_dir or CINE_DIR_DEFAULT
    positions_csv = positions_csv or POSITIONS_CSV_DEFAULT
    if not cine_dir.is_dir():
        raise FileNotFoundError(f"Calibration .cine folder not found: {cine_dir}")
    if not positions_csv.is_file():
        raise FileNotFoundError(f"positions.csv not found: {positions_csv}")

    positions_df = pd.read_csv(positions_csv)
    pos_map = dict(zip(positions_df["filename"],
                       positions_df["stage_position_mm"]))
    cines = sorted(cine_dir.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float("inf")))
    if not cines:
        raise FileNotFoundError(f"No .cine files in {cine_dir}")

    focus_stage = find_focal_plane(cines, pos_map)

    raw_frames = []
    z_positions = []
    cine_paths = []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(
            loader._load_frame(loader.frame_range[0]), dtype=np.float32)
        raw_frames.append(raw)
        z_positions.append(pos_map[cine.name] - focus_stage)
        cine_paths.append(cine)
    return raw_frames, z_positions, cine_paths


def find_latest_model_checkpoint() -> Path:
    """Return path to the latest dme_best.pth in Training/training_output/models/."""
    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(f"Models dir not found: {MODELS_DIR}")
    candidates = sorted(MODELS_DIR.glob("*/checkpoints/dme_best.pth"),
                        reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No dme_best.pth found under {MODELS_DIR}/*/checkpoints/")
    return candidates[0]


def load_inference_model():
    """Load the latest trained model. Returns a RealCropInference instance."""
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    ckpt = find_latest_model_checkpoint()
    print(f"Using model checkpoint: {ckpt.parent.parent.name}")
    return RealCropInference(model_path=str(ckpt), device="cpu")


def linear_fit(z, sigma, near_focus_threshold: float = 0.5):
    """sigma = rho * |z| + sigma_0 — same parametrisation as Calibration uses.

    Returns (rho, sigma_0, R², n_used). NaN if fit fails or insufficient
    points.
    """
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(sigma) & (np.abs(z) > near_focus_threshold)
    z = z[valid]
    sigma = sigma[valid]
    if len(z) < 4:
        return float("nan"), float("nan"), float("nan"), 0
    abs_z = np.abs(z)
    try:
        popt, _ = curve_fit(
            lambda zz, rho, s0: rho * zz + s0,
            abs_z, sigma,
            p0=[1.0, 0.5],
            bounds=([0.01, 0], [100, 50]),
        )
    except Exception:
        return float("nan"), float("nan"), float("nan"), 0
    rho, s0 = popt
    pred = rho * abs_z + s0
    ss = ((sigma - sigma.mean()) ** 2).sum()
    r2 = 1 - ((sigma - pred) ** 2).sum() / ss if ss > 0 else 0.0
    return rho, s0, r2, len(z)


def plot_mode_curves(
    z_arr: np.ndarray,
    series: dict,  # {label: {'erf': sigmas, 'model': sigmas, 'colour': c}}
    output_path: Path,
    title: str = "Sigma vs z by preprocessing mode",
):
    """Plot ERF + model curves across z, one panel per metric, all modes overlaid."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: ERF measurements
    ax = axs[0]
    for label, data in series.items():
        ax.scatter(z_arr, data["erf"], s=22, alpha=0.7,
                   color=data["colour"], label=label)
        rho, s0, r2, n = linear_fit(z_arr, data["erf"])
        if np.isfinite(rho):
            zfit = np.linspace(z_arr.min() - 0.5, z_arr.max() + 0.5, 100)
            ax.plot(zfit, rho * np.abs(zfit) + s0,
                    color=data["colour"], alpha=0.4, linewidth=1.2)
            data["erf_fit"] = (rho, s0, r2, n)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("ERF measured sigma (px)")
    ax.set_title("ERF measurement per mode")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # Panel 2: Model predictions
    ax = axs[1]
    for label, data in series.items():
        ax.scatter(z_arr, data["model"], s=22, alpha=0.7,
                   color=data["colour"], label=label)
        rho, s0, r2, n = linear_fit(z_arr, data["model"])
        if np.isfinite(rho):
            zfit = np.linspace(z_arr.min() - 0.5, z_arr.max() + 0.5, 100)
            ax.plot(zfit, rho * np.abs(zfit) + s0,
                    color=data["colour"], alpha=0.4, linewidth=1.2)
            data["model_fit"] = (rho, s0, r2, n)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Model predicted sigma (px)")
    ax.set_title("Trained model output per mode")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return series  # mutated with fits


def plot_sample_grid(
    samples: dict,  # {z_value: {mode_label: image_uint8}}
    output_path: Path,
    title: str = "Sample frames per mode",
):
    """Grid of processed frames, rows=z values, columns=modes."""
    z_values = sorted(samples.keys())
    if not z_values:
        return
    mode_labels = list(samples[z_values[0]].keys())

    n_rows = len(z_values)
    n_cols = len(mode_labels)
    fig, axs = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    if n_cols == 1:
        axs = axs.reshape(-1, 1)

    for i, z in enumerate(z_values):
        for j, mode in enumerate(mode_labels):
            ax = axs[i, j]
            img = samples[z][mode]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if i == 0:
                ax.set_title(mode, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"z={z:+.1f}mm", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
