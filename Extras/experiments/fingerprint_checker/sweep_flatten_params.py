"""Systematic sweep of feather and inner_margin parameters of
flatten_sphere_crop, looking for the combination that makes model output
best match ERF truth on the calibration stack.

Sweep:
  feather       in [3, 6, 10, 15, 20, 25, 30, 40, 50, 60, 80]
  inner_margin  in [0, 5, 20]
  flatten_exterior = True (always; the False case was already shown to be worst)
  MINMAX = False (verified earlier it doesn't change the output meaningfully)

For each combination:
  - Process all 61 frames
  - Run model on each
  - Fit rho/s0 to (z, model_sigma)
  - Compare to ERF truth (rho=0.947, s0=0.461)

Output: heatmap of mean abs gap to truth, plus best-curve overlay plot.
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
    crop_to_square, flatten_sphere_crop, find_consensus_sphere,
)
from blur_measurement import measure_blur_auto
from cine_loader import CineLoader

CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
POSITIONS_CSV = CINE_DIR / "positions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"

# Parameter grid
FEATHERS = [3, 6, 10, 15, 20, 25, 30, 40, 50, 60, 80]
INNER_MARGINS = [0, 5, 20]


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


def process(raw, cx, cy, radius, feather, inner_margin):
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    flat, info = flatten_sphere_crop(f, feather=feather,
                                       inner_margin=inner_margin,
                                       flatten_exterior=True)
    if info is None:
        flat = f
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))

    print("Loading + finding focal plane...")
    focus_stage = find_focal_plane(cines, pos_map)
    raw_frames, z_positions = [], []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        raw_frames.append(raw); z_positions.append(pos_map[cine.name] - focus_stage)
    cx, cy, radius = find_consensus_sphere(raw_frames, upper_only=True)
    print(f"  focus={focus_stage}, sphere=({cx},{cy},r={radius})")
    z_arr = np.array(z_positions)

    # Get ERF truth using calibration GUI's exact processing
    print("\nComputing ERF truth (default mode + MINMAX)...")
    erf_truth = []
    for raw in raw_frames:
        crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
        f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
        flat, _ = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                        flatten_exterior=False)
        proc = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        try:
            m = measure_blur_auto(proc, center=None, radius=None,
                                   method='erf', verbose=False)
            erf_truth.append(float(m.sigma) if m.confidence > 0.5 else float('nan'))
        except Exception:
            erf_truth.append(float('nan'))
    erf_arr = np.array(erf_truth)
    erf_rho, erf_s0, erf_r2, _ = linear_fit(z_positions, erf_truth)
    print(f"  ERF truth: rho={erf_rho:.4f}  s0={erf_s0:.4f}  R²={erf_r2:.4f}")

    print("\nLoading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    # Sweep
    n_combos = len(FEATHERS) * len(INNER_MARGINS)
    print(f"\nSweeping {n_combos} combinations × 61 frames...")
    grid_results = {}  # (feather, inner_margin) -> {sigmas, rho, s0, r2, mean_abs_gap, slope_err, offset}

    for combo_i, (feather, inner_margin) in enumerate(
            [(f, im) for f in FEATHERS for im in INNER_MARGINS]):
        sigmas = []
        for raw in raw_frames:
            proc = process(raw, cx, cy, radius, feather, inner_margin)
            try:
                sigmas.append(float(inf.estimate_blur_from_image(proc)))
            except Exception:
                sigmas.append(float('nan'))
        sigma_arr = np.array(sigmas)
        rho, s0, r2, n_used = linear_fit(z_positions, sigmas)
        valid = np.isfinite(sigma_arr) & np.isfinite(erf_arr)
        gap = sigma_arr[valid] - erf_arr[valid]
        mean_abs = np.mean(np.abs(gap)) if len(gap) else float('nan')
        # Slope error: how far rho ratio is from 1.0
        slope_err = abs(rho - erf_rho) if np.isfinite(rho) else float('inf')
        # Offset: median gap (positive means model overpredicts)
        offset = np.median(gap) if len(gap) else float('nan')
        grid_results[(feather, inner_margin)] = dict(
            sigmas=sigmas, rho=rho, s0=s0, r2=r2,
            mean_abs_gap=mean_abs, slope_err=slope_err, offset=offset,
        )
        print(f"  [{combo_i+1}/{n_combos}] feather={feather:>3}, inner={inner_margin:>2}  "
              f"rho={rho:>6.3f}  s0={s0:>6.3f}  R²={r2:>5.3f}  "
              f"|gap|={mean_abs:>5.2f}  slope_err={slope_err:>5.3f}  off={offset:>+5.2f}")

    # Find best by combined metric: mean_abs_gap (which captures both slope and offset)
    sorted_results = sorted(grid_results.items(),
                             key=lambda kv: kv[1]['mean_abs_gap'])
    print("\n" + "=" * 95)
    print("BEST 5 combinations (by mean abs gap to ERF truth):")
    print("=" * 95)
    for (feather, inner_margin), r in sorted_results[:5]:
        print(f"  feather={feather:>3}, inner={inner_margin:>2}: "
              f"rho={r['rho']:.3f} (vs ERF {erf_rho:.3f}, ratio={r['rho']/erf_rho:.2f})  "
              f"s0={r['s0']:.3f} (vs ERF {erf_s0:.3f}, gap={r['s0']-erf_s0:+.2f})  "
              f"R²={r['r2']:.3f}  |gap|={r['mean_abs_gap']:.2f} px")

    # Heatmap of mean_abs_gap
    grid = np.full((len(INNER_MARGINS), len(FEATHERS)), np.nan)
    for i, im in enumerate(INNER_MARGINS):
        for j, ft in enumerate(FEATHERS):
            grid[i, j] = grid_results[(ft, im)]['mean_abs_gap']

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    ax = axs[0]
    im_h = ax.imshow(grid, aspect='auto', cmap='viridis_r')
    ax.set_xticks(range(len(FEATHERS)))
    ax.set_xticklabels(FEATHERS)
    ax.set_yticks(range(len(INNER_MARGINS)))
    ax.set_yticklabels(INNER_MARGINS)
    ax.set_xlabel('feather (source-px)')
    ax.set_ylabel('inner_margin (source-px)')
    ax.set_title('Mean abs gap (model output vs ERF truth) across parameter sweep\n'
                 'darker = better')
    plt.colorbar(im_h, ax=ax, label='mean abs gap (px)')
    for i in range(len(INNER_MARGINS)):
        for j in range(len(FEATHERS)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                        color='white' if v > grid[np.isfinite(grid)].mean() else 'black',
                        fontsize=8)

    # Top-3 curves overlaid on truth
    ax = axs[1]
    ax.scatter(z_arr, erf_arr, s=40, color='black', label=f'ERF truth (rho={erf_rho:.3f})',
               zorder=10)
    cmap = plt.cm.tab10
    for i, ((feather, inner_margin), r) in enumerate(sorted_results[:5]):
        ax.scatter(z_arr, r['sigmas'], s=20, alpha=0.7, color=cmap(i),
                   label=f"f={feather} im={inner_margin}: rho={r['rho']:.2f} "
                         f"s0={r['s0']:.2f} |gap|={r['mean_abs_gap']:.2f}")
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Sigma (px)')
    ax.set_title('Top 5 parameter combinations (closest to ERF truth)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "sweep_flatten_params.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
