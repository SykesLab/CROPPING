"""Test whether calibration parameters (rho, sigma_0) change when we
process calibration .cine frames with simple-mode flatten instead of
default-mode flatten.

Critical question from user: if calibration measures sigma using one
preprocessing recipe but inference data uses another, the calibration
constants would be wrong for inference. So we need to verify that the
SAME flatten recipe used for inference (simple) also produces good
calibration measurements.

Procedure:
  1. Walk all .cine in calibration spheres/9mm/, frame 0 each
  2. For each frame: detect sphere geometry once
  3. Process two ways: default mode AND simple mode (and inference mode for
     completeness)
  4. Run measure_blur_erf on each processed image
  5. Linear-fit z vs sigma for each method to get rho, sigma_0
  6. Compare rho, sigma_0, R2, LOO-CV across methods
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


CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
POSITIONS_CSV = CINE_DIR / "positions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"

# Focus offset — the bundle's z=0 is at stage_position 7.0 (from earlier trace)
# Let me derive from the measurements.csv
BUNDLE_DIR = _REPO_ROOT / "Calibration" / "runs" / "20260426_045138_camera-g"


def linear_fit(z, sigma):
    """sigma = rho * |z| + sigma_0  — same as Calibration uses."""
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(sigma) & (np.abs(z) > 0.5)  # exclude near-focus per default
    z = z[valid]; sigma = sigma[valid]
    if len(z) < 4:
        return float('nan'), float('nan'), float('nan'), float('nan'), len(z)
    abs_z = np.abs(z)
    def model(zz, rho, s0):
        return rho * zz + s0
    try:
        popt, _ = curve_fit(model, abs_z, sigma, p0=[1.0, 0.5],
                            bounds=([0.01, 0], [100, 50]))
    except Exception:
        return float('nan'), float('nan'), float('nan'), float('nan'), len(z)
    rho, s0 = popt
    pred = model(abs_z, rho, s0)
    ss_res = ((sigma - pred) ** 2).sum()
    ss_tot = ((sigma - sigma.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(sigma - pred))
    return rho, s0, r2, mae, len(z)


def loo_cv_fit(z, sigma):
    """Leave-one-out cross-validation."""
    z = np.asarray(z); sigma = np.asarray(sigma)
    valid = np.isfinite(sigma) & (np.abs(z) > 0.5)
    z = z[valid]; sigma = sigma[valid]
    n = len(z)
    if n < 4:
        return float('nan'), float('nan'), float('nan')
    rhos, s0s, residuals = [], [], []
    for i in range(n):
        z_t = np.delete(z, i); s_t = np.delete(sigma, i)
        try:
            popt, _ = curve_fit(
                lambda zz, rho, s0: rho * np.abs(zz) + s0,
                z_t, s_t, p0=[1.0, 0.5],
                bounds=([0.01, 0], [100, 50]))
            rho_i, s0_i = popt
            pred_i = rho_i * np.abs(z[i]) + s0_i
            rhos.append(rho_i); s0s.append(s0_i)
            residuals.append(pred_i - sigma[i])
        except Exception:
            continue
    if not rhos:
        return float('nan'), float('nan'), float('nan')
    return float(np.std(rhos)), float(np.std(s0s)), float(np.mean(np.abs(residuals)))


def process_with_mode(
    raw_frame: np.ndarray, cx: int, cy: int, radius: int,
    mode: str,
) -> np.ndarray:
    """Replicate calibration's _apply_sphere_pipeline with the given flatten
    mode (excluding mirror/blacken since calibration GUI sets those to False)."""
    crop = crop_to_square(raw_frame, cx, cy, radius, padding=1.2)
    if crop.dtype == np.uint8:
        f = crop.astype(np.float32) / 255.0
    else:
        f = crop.astype(np.float32)
        if f.max() > 1.5:
            f = f / 255.0
    if mode == 'default':
        flat, info = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                          flatten_exterior=False)
    elif mode == 'simple':
        flat, info = flatten_sphere_crop(f, feather=3, inner_margin=0,
                                          flatten_exterior=True)
    elif mode == 'inference':
        flat, info = flatten_sphere_crop(f, feather=40, inner_margin=1,
                                          flatten_exterior=True)
    else:
        raise ValueError(mode)
    if info is None:
        return f
    # Match calibration GUI's post-processing: convert float->uint8 with MINMAX
    # (this is what _apply_sphere_pipeline:454 does)
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_with_mode_no_minmax(
    raw_frame: np.ndarray, cx: int, cy: int, radius: int,
    mode: str,
) -> np.ndarray:
    """Same as process_with_mode but skips the MINMAX (proposed Phase 4 fix)."""
    crop = crop_to_square(raw_frame, cx, cy, radius, padding=1.2)
    if crop.dtype == np.uint8:
        f = crop.astype(np.float32) / 255.0
    else:
        f = crop.astype(np.float32)
        if f.max() > 1.5:
            f = f / 255.0
    if mode == 'default':
        flat, info = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                          flatten_exterior=False)
    elif mode == 'simple':
        flat, info = flatten_sphere_crop(f, feather=3, inner_margin=0,
                                          flatten_exterior=True)
    elif mode == 'inference':
        flat, info = flatten_sphere_crop(f, feather=40, inner_margin=1,
                                          flatten_exterior=True)
    else:
        raise ValueError(mode)
    if info is None:
        return (f * 255).astype(np.uint8)
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all .cine files + their stage positions
    print("Loading positions.csv...")
    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))
    print(f"  Found {len(cines)} .cine files")

    # 2. Determine focal plane from the data itself, NOT from bundle's csv.
    # positions.csv has raw stage 0-12; the focal plane could be at any of these.
    # Find by computing Laplacian variance (max sharpness = in-focus).
    print("\nFinding focal plane via Laplacian variance...")
    laplacian_per_frame = []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            laplacian_per_frame.append((cine.name, np.nan, np.nan))
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        # Laplacian variance on the centre crop (where the sphere is)
        h, w = raw.shape
        cx0, cy0 = w // 2, h // 2
        crop_size = min(h, w) // 3
        centre = raw[cy0 - crop_size:cy0 + crop_size,
                     cx0 - crop_size:cx0 + crop_size]
        lap = cv2.Laplacian(centre, cv2.CV_32F, ksize=3)
        lap_var = float(lap.var())
        stage_pos = pos_map.get(cine.name, np.nan)
        laplacian_per_frame.append((cine.name, stage_pos, lap_var))

    # Find the stage position with max Laplacian variance
    df_lap = pd.DataFrame(laplacian_per_frame,
                          columns=['filename', 'stage_mm', 'laplacian_var'])
    df_lap = df_lap.dropna().sort_values('stage_mm').reset_index(drop=True)
    print("\n  Per-frame Laplacian variance (sharpness):")
    for _, r in df_lap.iterrows():
        marker = " <-- in focus" if r['laplacian_var'] == df_lap['laplacian_var'].max() else ""
        print(f"    {r['filename']:<14} stage={r['stage_mm']:>5.1f}mm "
              f"lap_var={r['laplacian_var']:>10.2f}{marker}")

    focus_idx = df_lap['laplacian_var'].idxmax()
    focus_offset = float(df_lap.loc[focus_idx, 'stage_mm'])
    print(f"\n  Focal plane (via max Laplacian): stage = {focus_offset:.2f} mm")
    bundle_focus = pos_map.get('9mm_1.cine', 0) - pd.read_csv(
        BUNDLE_DIR / "measurements.csv").iloc[0]['z_mm']
    print(f"  Bundle assumed focal plane:      stage = {bundle_focus:.2f} mm")
    if abs(focus_offset - bundle_focus) > 0.5:
        print(f"  WARNING: Laplacian-derived focus and bundle-assumed focus "
              f"differ by {abs(focus_offset - bundle_focus):.2f} mm")
        print(f"  -> Using Laplacian-derived focus.")

    # 3. Load all frames + detect consensus sphere
    print("Loading + detecting sphere on all frames...")
    raw_frames = []
    z_positions = []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        stage = pos_map.get(cine.name, np.nan)
        z = stage - focus_offset
        raw_frames.append(raw)
        z_positions.append(z)
    print(f"  Loaded {len(raw_frames)} frames")

    consensus = find_consensus_sphere(raw_frames, upper_only=True)
    if consensus is None:
        print("ERROR: consensus sphere detection failed")
        return
    cx, cy, radius = consensus
    print(f"  Consensus sphere: center=({cx},{cy}) radius={radius}")

    # 4. For each mode, measure sigma at every frame.
    # Pass the CROPPED image's center+radius (same for all frames after crop_to_square).
    modes_to_test = [
        ('default + MINMAX (CURRENT)', 'default', True),
        ('default no MINMAX', 'default', False),
        ('simple + MINMAX', 'simple', True),
        ('simple no MINMAX (PROPOSED)', 'simple', False),
        ('inference + MINMAX', 'inference', True),
        ('inference no MINMAX', 'inference', False),
    ]
    method_results = {}
    print("\nMeasuring blur sigma at every frame, for each method...")
    for label, mode, use_minmax in modes_to_test:
        print(f"\n  Method: {label}")
        sigmas = []
        n_failed_fits = 0
        for raw, z in zip(raw_frames, z_positions):
            if use_minmax:
                proc = process_with_mode(raw, cx, cy, radius, mode)
            else:
                proc = process_with_mode_no_minmax(raw, cx, cy, radius, mode)
            # Cropped image is centered on sphere (radius scaled by crop's aspect)
            ph, pw = proc.shape[:2]
            crop_cx, crop_cy = pw // 2, ph // 2
            # Effective radius after the crop_to_square (image-clamped) — use the
            # min half-dimension as a safe upper bound; measure_blur_erf will
            # refine via radial sweep.
            crop_radius = min(crop_cx, crop_cy) * 8 // 10
            try:
                m = measure_blur_erf(proc, center=(crop_cx, crop_cy),
                                     radius=crop_radius, verbose=False)
                sig = float(m.sigma) if m.confidence > 0.5 else float('nan')
                if sig != sig:  # NaN
                    n_failed_fits += 1
            except Exception:
                sig = float('nan')
                n_failed_fits += 1
            sigmas.append(sig)
        if n_failed_fits:
            print(f"    {n_failed_fits} of {len(sigmas)} frames had no usable ERF fit (R2 < 0.5)")
        # Fit rho, sigma_0
        rho, s0, r2, mae, n_used = linear_fit(z_positions, sigmas)
        rho_std, s0_std, loo_mae = loo_cv_fit(z_positions, sigmas)
        n_finite = int(np.isfinite(np.array(sigmas, dtype=float)).sum())
        print(f"    n_finite_sigmas: {n_finite}/{len(sigmas)}")
        print(f"    rho = {rho:.4f}  sigma_0 = {s0:.4f}  R2 = {r2:.4f}  "
              f"fit_MAE = {mae:.4f} px  (n_used = {n_used})")
        print(f"    LOO-CV: rho_std = {rho_std:.4f}  s0_std = {s0_std:.4f}  "
              f"LOO_MAE = {loo_mae:.4f} px")
        method_results[label] = {
            'sigmas': sigmas, 'rho': rho, 'sigma_0': s0, 'r2': r2,
            'mae': mae, 'rho_std': rho_std, 's0_std': s0_std, 'loo_mae': loo_mae,
            'n_used': n_used, 'n_finite': n_finite,
        }

    # 5. Plot all calibration curves overlaid
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    z_arr = np.array(z_positions)
    colours = plt.cm.tab10.colors
    for i, (label, r) in enumerate(method_results.items()):
        ax.scatter(z_arr, r['sigmas'], s=14, alpha=0.5, color=colours[i],
                   label=f"{label}\nrho={r['rho']:.3f} s0={r['sigma_0']:.3f} "
                         f"R2={r['r2']:.3f} LOO-MAE={r['loo_mae']:.3f}px")
        # Plot fit
        z_fit = np.linspace(-8, 6, 100)
        sig_fit = r['rho'] * np.abs(z_fit) + r['sigma_0']
        ax.plot(z_fit, sig_fit, color=colours[i], linewidth=1.5, alpha=0.8)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('sigma_px (ERF)')
    ax.set_title('Calibration curve under different flatten preprocessing '
                 'recipes\n(does the recipe change the fit?)')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "calibration_invariance.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")

    # 6. Summary comparison
    print("\n" + "=" * 90)
    print("SUMMARY: how do calibration parameters drift with preprocessing?")
    print("=" * 90)
    baseline = method_results['default + MINMAX (CURRENT)']
    print(f"{'method':<32} {'rho':>8} {'s0':>8} {'R2':>8} {'LOO-MAE':>10} | "
          f"{'drho%':>8} {'ds0':>8}")
    print("-" * 92)
    for label, r in method_results.items():
        d_rho = 100.0 * (r['rho'] - baseline['rho']) / baseline['rho']
        d_s0 = r['sigma_0'] - baseline['sigma_0']
        print(f"{label:<32} {r['rho']:>8.4f} {r['sigma_0']:>8.4f} "
              f"{r['r2']:>8.4f} {r['loo_mae']:>10.4f} | "
              f"{d_rho:>+8.2f} {d_s0:>+8.4f}")


if __name__ == '__main__':
    main()
