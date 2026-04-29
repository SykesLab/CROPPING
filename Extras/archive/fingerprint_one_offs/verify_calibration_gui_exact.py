"""Bare verification using ONLY the exact code paths the calibration GUI
uses — no fabricated variants.

Calibration GUI flow (replicated exactly):
  1. Load each .cine via CineLoader (exactly what _load_zstack does)
  2. Run process_sphere_stack(images, blacken=False, mirror=False, flatten=True,
                              flatten_mode="default")
     -> returns processed images at calibration GUI's exact convention
  3. Run measure_blur_auto(img, center=None, radius=None, method='erf')
     -> exactly what _measure_blur does
  4. Run inference.estimate_blur_from_image(img)
     -> what the model would produce if fed the calibration GUI's actual output

Compare ERF vs model. If they disagree, that's the actual cross-pipeline gap
to address — no simple/inference/whatever-mode variants invented by me.
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

from sphere_processing import process_sphere_stack
from blur_measurement import measure_blur_auto
from cine_loader import CineLoader

CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
POSITIONS_CSV = CINE_DIR / "positions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def find_focal_plane(cines, pos_map):
    best = (-np.inf, 0.0)
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        h, w = raw.shape
        cy0, cx0 = h // 2, w // 2
        s = min(h, w) // 3
        v = float(cv2.Laplacian(raw[cy0-s:cy0+s, cx0-s:cx0+s],
                                 cv2.CV_32F, ksize=3).var())
        if v > best[0]:
            best = (v, pos_map[cine.name])
    return best[1]


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
    ss = ((sigma - sigma.mean()) ** 2).sum()
    r2 = 1 - ((sigma - pred) ** 2).sum() / ss if ss > 0 else 0
    return rho, s0, r2, len(z)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Replicate calibration GUI's loading exactly ===
    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))

    print("Finding focal plane via Laplacian...")
    focus_stage = find_focal_plane(cines, pos_map)
    print(f"  Focus stage = {focus_stage:.2f} mm")

    print("Loading all frames...")
    raw_frames = []
    z_positions = []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = loader._load_frame(loader.frame_range[0])
        raw_frames.append(np.asarray(raw, dtype=np.float32))
        z_positions.append(pos_map[cine.name] - focus_stage)

    # === Run the EXACT calibration GUI processing ===
    # calibration_gui.py:1743 uses these exact arguments
    print("\nProcessing with calibration GUI's exact call:")
    print('  process_sphere_stack(..., blacken=False, mirror=False,')
    print('                       flatten=True, flatten_mode="default")')
    processed, sphere_info = process_sphere_stack(
        raw_frames, upper_only=True,
        blacken=False, mirror=False, flatten=True, flatten_mode="default",
    )
    print(f"  Got {len(processed)} processed images, sphere_info = {sphere_info}")

    # === Measure ERF — same call calibration GUI uses ===
    # calibration_gui.py:1944 uses measure_blur_auto(img, center, radius, method, verbose=False)
    # with center/radius BOTH None (auto-detect)
    print("\nMeasuring ERF on each (calibration GUI's exact call)...")
    erf_sigmas = []
    for i, img in enumerate(processed):
        try:
            m = measure_blur_auto(img, center=None, radius=None,
                                   method='erf', verbose=False)
            erf_sigmas.append(float(m.sigma) if m.confidence > 0.5 else float('nan'))
        except Exception:
            erf_sigmas.append(float('nan'))

    # === Run model on each — exactly what would happen if we fed these images
    # to inference_real_crops.py (the actual deployed inference) ===
    print("\nLoading model and running estimate_blur_from_image on each...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    model_sigmas = []
    for img in processed:
        try:
            model_sigmas.append(float(inf.estimate_blur_from_image(img)))
        except Exception:
            model_sigmas.append(float('nan'))

    df = pd.DataFrame({
        'z_mm': z_positions,
        'erf_sigma': erf_sigmas,
        'model_sigma': model_sigmas,
    })
    df.to_csv(OUTPUT_DIR / "calibration_gui_exact.csv", index=False)

    # === Fits ===
    print("\nLinear fits:")
    rho_e, s0_e, r2_e, n_e = linear_fit(df['z_mm'], df['erf_sigma'])
    rho_m, s0_m, r2_m, n_m = linear_fit(df['z_mm'], df['model_sigma'])
    print(f"  ERF (sigma = rho*|z| + s0):")
    print(f"    rho = {rho_e:.4f}  s0 = {s0_e:.4f}  R2 = {r2_e:.4f}  n_used = {n_e}")
    print(f"  Model:")
    print(f"    rho = {rho_m:.4f}  s0 = {s0_m:.4f}  R2 = {r2_m:.4f}  n_used = {n_m}")

    valid = df['erf_sigma'].notna() & df['model_sigma'].notna()
    if valid.any():
        gap = (df.loc[valid, 'model_sigma'] - df.loc[valid, 'erf_sigma'])
        print(f"\n  Model vs ERF: median gap = {gap.median():+.3f} px,  "
              f"mean abs = {gap.abs().mean():.3f} px,  max abs = {gap.abs().max():.3f} px")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['z_mm'], df['erf_sigma'], s=30, color='tab:blue',
               label=f'ERF on calibration GUI output\nrho={rho_e:.3f} s0={s0_e:.3f} R²={r2_e:.3f}')
    ax.scatter(df['z_mm'], df['model_sigma'], s=30, color='tab:red', marker='x',
               label=f'Model on SAME images\nrho={rho_m:.3f} s0={s0_m:.3f} R²={r2_m:.3f}')
    zfit = np.linspace(-8, 6, 100)
    if np.isfinite(rho_e):
        ax.plot(zfit, rho_e * np.abs(zfit) + s0_e, 'b-', alpha=0.5)
    if np.isfinite(rho_m):
        ax.plot(zfit, rho_m * np.abs(zfit) + s0_m, 'r-', alpha=0.5)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Sigma (px)')
    ax.set_title('Calibration GUI exact pipeline:\n'
                 'ERF measurement vs Model prediction on the SAME processed images')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "calibration_gui_exact.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")
    print(f"Wrote: {OUTPUT_DIR / 'calibration_gui_exact.csv'}")


if __name__ == '__main__':
    main()
