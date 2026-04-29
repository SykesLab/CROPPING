"""Test: pass the consensus contour to flatten_sphere_crop instead of
letting it auto-detect per-frame.

The previous test (verify_visual_match.py) revealed that simple-mode
flatten's per-frame Canny detection FAILS on calibration spheres at
mild defocus — it finds the wrong feature and inverts interior/exterior.

Fix: detect the sphere ONCE on the most in-focus frame (already done by
find_consensus_sphere), then pass that contour to every frame's flatten.
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
    find_consensus_sphere, crop_to_square, flatten_sphere_crop,
)
from blur_measurement import measure_blur_erf
from cine_loader import CineLoader

MODEL_SIZE = 256
CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
POSITIONS_CSV = CINE_DIR / "positions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def make_circle_contour(cx_local: int, cy_local: int, radius: int,
                         n_points: int = 360) -> np.ndarray:
    """Build a (n,1,2) cv2-compatible contour for a circle at (cx,cy) with radius."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.stack([
        cx_local + radius * np.cos(angles),
        cy_local + radius * np.sin(angles),
    ], axis=-1).astype(np.int32)
    return pts.reshape(-1, 1, 2)


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

    print("Finding focal plane...")
    focus_stage = find_focal_plane(cines, pos_map)
    print(f"  stage = {focus_stage:.2f}mm")

    raw_frames, z_positions = [], []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        raw_frames.append(raw)
        z_positions.append(pos_map[cine.name] - focus_stage)

    cx, cy, radius = find_consensus_sphere(raw_frames, upper_only=True)
    print(f"Consensus sphere: ({cx},{cy}) r={radius}")

    # Build the consensus contour. After crop_to_square(padding=1.2), the
    # sphere is centered in the cropped image. Crop dimensions = 2*radius*1.2
    # but clamped to image bounds. Estimate effective crop dims and rebuild.
    # Simpler: just recompute per-frame using the cropped image's center.

    # Load model
    print("Loading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    # For each frame: process THREE ways and run model + ERF
    print("\nRunning model on every frame, three flatten variants...")
    rows = []
    for raw, z in zip(raw_frames, z_positions):
        crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
        ph, pw = crop.shape[:2]
        # Sphere center within the crop (it's at the centre after crop_to_square)
        local_cx, local_cy = pw // 2, ph // 2
        local_radius = radius  # same physical pixels
        if crop.dtype == np.uint8:
            f = crop.astype(np.float32) / 255.0
        else:
            f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)

        # 1. Default mode (current calibration measurement) — for ERF truth
        flat_def, _ = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                            flatten_exterior=False)
        proc_def = cv2.normalize(flat_def, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Simple mode AUTO-DETECT (the broken one)
        flat_auto, _ = flatten_sphere_crop(f, feather=3, inner_margin=0,
                                            flatten_exterior=True)
        proc_auto = (np.clip(flat_auto, 0, 1) * 255).astype(np.uint8)

        # 3. Simple mode with CONSENSUS CONTOUR passed in
        contour = make_circle_contour(local_cx, local_cy, local_radius)
        flat_anchor, info = flatten_sphere_crop(f, contour=contour,
                                                  feather=3, inner_margin=0,
                                                  flatten_exterior=True)
        proc_anchor = (np.clip(flat_anchor, 0, 1) * 255).astype(np.uint8) if info else proc_auto

        # ERF on default
        try:
            m = measure_blur_erf(proc_def, center=(pw//2, ph//2),
                                  radius=min(pw,ph)*4//10, verbose=False)
            erf_truth = float(m.sigma) if m.confidence > 0.5 else float('nan')
        except Exception:
            erf_truth = float('nan')

        # Model on each
        try: model_def = float(inf.estimate_blur_from_image(proc_def))
        except Exception: model_def = float('nan')
        try: model_auto = float(inf.estimate_blur_from_image(proc_auto))
        except Exception: model_auto = float('nan')
        try: model_anchor = float(inf.estimate_blur_from_image(proc_anchor))
        except Exception: model_anchor = float('nan')

        rows.append({
            'z_mm': z, 'erf_truth': erf_truth,
            'model_default': model_def,
            'model_simple_autodetect': model_auto,
            'model_simple_anchored': model_anchor,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "anchored_contour.csv", index=False)
    print("\nLinear fits:")
    for col in ('erf_truth', 'model_default', 'model_simple_autodetect',
                'model_simple_anchored'):
        rho, s0, r2, n = linear_fit(df['z_mm'], df[col])
        print(f"  {col:<30}  rho={rho:.4f}  s0={s0:.4f}  R2={r2:.4f}  n={n}")

    # Per-frame agreement: model_anchored vs erf_truth
    valid = df['erf_truth'].notna() & df['model_simple_anchored'].notna()
    if valid.any():
        gap = (df.loc[valid, 'model_simple_anchored'] - df.loc[valid, 'erf_truth'])
        print(f"\n  model_anchored vs erf_truth: median gap = {gap.median():+.3f} px, "
              f"mean abs = {gap.abs().mean():.3f} px")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    z_arr = df['z_mm'].values
    ax.scatter(z_arr, df['erf_truth'], s=30, label='ERF on default-mode (TRUTH)',
               color='tab:blue')
    ax.scatter(z_arr, df['model_simple_autodetect'], s=24, marker='x',
               color='tab:red', label='Model on simple-mode (auto-detect contour)',
               alpha=0.6)
    ax.scatter(z_arr, df['model_simple_anchored'], s=24, marker='^',
               color='tab:green', label='Model on simple-mode (CONSENSUS contour)',
               alpha=0.85)

    # Fit lines
    for col, color in [('erf_truth', 'tab:blue'),
                       ('model_simple_anchored', 'tab:green')]:
        rho, s0, _, _ = linear_fit(df['z_mm'], df[col])
        if np.isfinite(rho):
            zfit = np.linspace(-8, 6, 100)
            ax.plot(zfit, rho * np.abs(zfit) + s0, color=color, alpha=0.5,
                    linewidth=1.4)

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Sigma (px)')
    ax.set_title('Does anchoring the flatten contour fix the model-on-calibration problem?')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "anchored_contour.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
