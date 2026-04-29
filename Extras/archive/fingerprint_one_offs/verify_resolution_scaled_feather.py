"""Test the resolution-scaled feather idea across the FULL z range.

If feather is properly scaled so that calibration's effective feather in
model space matches preprocessing's, does the model give z-dependent
predictions on calibration sphere crops?

For each calibration frame:
  - Process with feather = round(3 * source_size / 299) source-pixels
    (so effective feather in model space = 2.6 model-px, matching preprocessing)
  - Run model -> predicted sigma
  - Compare to ERF-truth sigma at that z

If predicted curve tracks ERF curve: the architecture is fundamentally
sound (just needs the resolution-scaled feather). The model would work
on real data with this preprocessing.

If predicted curve is still flat: the issue is deeper than feather width.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from sphere_processing import (
    find_sphere_center, find_consensus_sphere,
    crop_to_square, flatten_sphere_crop,
)
from blur_measurement import measure_blur_erf
from cine_loader import CineLoader


MODEL_SIZE = 256
PREPROCESSING_SOURCE_SIZE = 299  # what preprocessing produces
PREPROCESSING_FEATHER_SOURCE_PX = 3  # what preprocessing uses

CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
POSITIONS_CSV = CINE_DIR / "positions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def find_focal_plane(cines, pos_map):
    best_var = -np.inf
    best_stage = 0.0
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        h, w = raw.shape
        cx0, cy0 = w // 2, h // 2
        s = min(h, w) // 3
        lap = cv2.Laplacian(raw[cy0-s:cy0+s, cx0-s:cx0+s], cv2.CV_32F, ksize=3)
        v = float(lap.var())
        if v > best_var:
            best_var = v
            best_stage = pos_map[cine.name]
    return best_stage


def process_with_resolution_scaled_feather(raw, cx, cy, radius):
    """Crop to square, then flatten with feather scaled to make
    effective feather in model space match preprocessing's 2.6 model-px.

    feather_target_model_px = 3 * (256 / 299) = 2.57
    feather_source_px = feather_target_model_px * (source_size / 256)
    """
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    source_size = max(crop.shape[:2])
    target_feather_model_px = (PREPROCESSING_FEATHER_SOURCE_PX
                                * MODEL_SIZE / PREPROCESSING_SOURCE_SIZE)
    feather_src = max(1, round(target_feather_model_px * source_size / MODEL_SIZE))
    flat, info = flatten_sphere_crop(f, feather=feather_src, inner_margin=0,
                                       flatten_exterior=True)
    if info is None:
        return (f * 255).astype(np.uint8), feather_src
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8), feather_src


def process_default(raw, cx, cy, radius):
    """Existing default mode for ERF measurement — UNCHANGED."""
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    flat, info = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                       flatten_exterior=False)
    if info is None:
        return (f * 255).astype(np.uint8)
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def linear_fit(z, sigma):
    z = np.asarray(z, dtype=float); sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(sigma) & (np.abs(z) > 0.5)
    z = z[valid]; sigma = sigma[valid]
    if len(z) < 4:
        return float('nan'), float('nan'), float('nan'), 0
    abs_z = np.abs(z)
    try:
        popt, _ = curve_fit(lambda zz, rho, s0: rho*zz + s0, abs_z, sigma,
                            p0=[1.0, 0.5], bounds=([0.01, 0], [100, 50]))
    except Exception:
        return float('nan'), float('nan'), float('nan'), 0
    rho, s0 = popt
    pred = rho * abs_z + s0
    ss_res = ((sigma - pred) ** 2).sum()
    ss_tot = ((sigma - sigma.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rho, s0, r2, len(z)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))

    print(f"Found {len(cines)} cines, finding focal plane...")
    focus_stage = find_focal_plane(cines, pos_map)
    print(f"  Focal plane = stage {focus_stage:.2f}mm")

    raw_frames = []
    z_positions = []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]), dtype=np.float32)
        raw_frames.append(raw)
        z_positions.append(pos_map[cine.name] - focus_stage)
    cx, cy, radius = find_consensus_sphere(raw_frames, upper_only=True)
    print(f"  Consensus sphere: center=({cx},{cy}) radius={radius}")

    # Compute the resolution-scaled feather for this dataset
    sample_crop = crop_to_square(raw_frames[0], cx, cy, radius, padding=1.2)
    source_size = max(sample_crop.shape[:2])
    feather_target_model_px = 3 * MODEL_SIZE / PREPROCESSING_SOURCE_SIZE
    feather_src_used = max(1, round(feather_target_model_px * source_size / MODEL_SIZE))
    print(f"\nResolution-scaled feather:")
    print(f"  Calibration source size: {source_size} px")
    print(f"  Preprocessing convention: 3 source-px on 299 source = 2.57 model-px")
    print(f"  For calibration source {source_size}: feather = {feather_src_used} source-px "
          f"= {feather_src_used * MODEL_SIZE / source_size:.2f} model-px")

    # Load model
    print("\nLoading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    # Run ERF on default-mode (truth) and model on resolution-scaled-feather
    print("\nRunning measurement on every frame...")
    rows = []
    for raw, z in zip(raw_frames, z_positions):
        proc_def = process_default(raw, cx, cy, radius)
        proc_scaled, _ = process_with_resolution_scaled_feather(raw, cx, cy, radius)

        # ERF truth (default mode)
        ph, pw = proc_def.shape[:2]
        try:
            m = measure_blur_erf(proc_def, center=(pw//2, ph//2),
                                  radius=min(pw,ph)*4//10, verbose=False)
            erf = float(m.sigma) if m.confidence > 0.5 else float('nan')
        except Exception:
            erf = float('nan')

        # Model on resolution-scaled-feather
        try:
            model_pred = float(inf.estimate_blur_from_image(proc_scaled))
        except Exception:
            model_pred = float('nan')

        rows.append({'z_mm': z, 'erf_truth': erf, 'model_scaled': model_pred})

    df = pd.DataFrame(rows)

    # Fits
    rho_e, s0_e, r2_e, n_e = linear_fit(df['z_mm'], df['erf_truth'])
    rho_m, s0_m, r2_m, n_m = linear_fit(df['z_mm'], df['model_scaled'])
    print("\nLinear fits:")
    print(f"  ERF (truth, default mode):       rho={rho_e:.4f}  s0={s0_e:.4f}  R2={r2_e:.4f}  n={n_e}")
    print(f"  Model (resolution-scaled flatten): rho={rho_m:.4f}  s0={s0_m:.4f}  R2={r2_m:.4f}  n={n_m}")
    if np.isfinite(rho_e) and np.isfinite(rho_m):
        print(f"  rho ratio model/ERF = {rho_m/rho_e:.3f}")
        print(f"  s0 difference        = {s0_m - s0_e:+.3f}")

    # Per-frame agreement summary
    valid = df['erf_truth'].notna() & df['model_scaled'].notna()
    if valid.any():
        d = (df.loc[valid, 'model_scaled'] - df.loc[valid, 'erf_truth'])
        print(f"\n  median gap: {d.median():+.3f} px  mean abs: {d.abs().mean():.3f} px")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    z_arr = df['z_mm'].values
    ax.scatter(z_arr, df['erf_truth'], s=30, color='tab:blue',
               label=f'ERF on default-mode (TRUTH)\nrho={rho_e:.3f} s0={s0_e:.3f} R2={r2_e:.3f}',
               alpha=0.75)
    ax.scatter(z_arr, df['model_scaled'], s=30, color='tab:green', marker='s',
               label=f'Model on resolution-scaled flatten ({feather_src_used}px source)\n'
                     f'rho={rho_m:.3f} s0={s0_m:.3f} R2={r2_m:.3f}',
               alpha=0.75)
    zfit = np.linspace(-8, 6, 100)
    if np.isfinite(rho_e):
        ax.plot(zfit, rho_e * np.abs(zfit) + s0_e, 'b-', alpha=0.5)
    if np.isfinite(rho_m):
        ax.plot(zfit, rho_m * np.abs(zfit) + s0_m, 'g--', alpha=0.5)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Sigma (px, source scale)')
    ax.set_title(f'Resolution-scaled feather test\n'
                 f'feather={feather_src_used} source-px (= {feather_target_model_px:.2f} model-px, '
                 f'matching preprocessing)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "resolution_scaled_feather.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
