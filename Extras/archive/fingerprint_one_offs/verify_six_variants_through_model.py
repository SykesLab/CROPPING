"""Test ALL six existing preprocessing variants through the model.

The flatten modes already exist in the codebase:
  - "default"  (calibration's current setting: inner_margin=20, no exterior fill)
  - "simple"   (preprocessing's setting: 3px feather, full binary)
  - "inference" (training_gui inference data prep: 40px feather, full binary)
... × MINMAX-applied / not-applied.

For each: process all 61 frames, run model, fit rho/sigma_0. Find the
combination (if any) that makes the model give correct z-dependent
predictions matching ERF truth.

This uses ONLY the calibration codebase's actual functions — no
fabricated padding/feather variants.
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


# Variants — exact mode names from sphere_processing.py
VARIANTS = [
    # (label, flatten_kwargs, apply_minmax)
    ("default + MINMAX (current calibration)",
     dict(feather=3, inner_margin=20, flatten_exterior=False), True),
    ("default no MINMAX",
     dict(feather=3, inner_margin=20, flatten_exterior=False), False),
    ("simple + MINMAX",
     dict(feather=3, inner_margin=0, flatten_exterior=True), True),
    ("simple no MINMAX (matches training pipeline)",
     dict(feather=3, inner_margin=0, flatten_exterior=True), False),
    ("inference + MINMAX",
     dict(feather=40, inner_margin=1, flatten_exterior=True), True),
    ("inference no MINMAX",
     dict(feather=40, inner_margin=1, flatten_exterior=True), False),
]


def process(raw, cx, cy, radius, kwargs, minmax):
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    flat, info = flatten_sphere_crop(f, **kwargs)
    if info is None:
        flat = f
    if minmax:
        return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))

    print("Finding focal plane...")
    focus_stage = find_focal_plane(cines, pos_map)
    print(f"  stage = {focus_stage:.2f}")

    print("Loading frames...")
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

    print("\nLoading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    print("\nRunning model on each variant...")
    results = {}
    erf_truth = []
    for j, (label, kwargs, minmax) in enumerate(VARIANTS):
        print(f"  [{j+1}/{len(VARIANTS)}] {label}")
        sigmas = []
        for raw in raw_frames:
            proc = process(raw, cx, cy, radius, kwargs, minmax)
            try:
                sigmas.append(float(inf.estimate_blur_from_image(proc)))
            except Exception:
                sigmas.append(float('nan'))
        results[label] = sigmas

        if j == 0:  # use default+MINMAX images for ERF truth (calibration GUI default)
            for raw in raw_frames:
                proc = process(raw, cx, cy, radius, kwargs, minmax)
                try:
                    m = measure_blur_auto(proc, center=None, radius=None,
                                           method='erf', verbose=False)
                    erf_truth.append(float(m.sigma) if m.confidence > 0.5 else float('nan'))
                except Exception:
                    erf_truth.append(float('nan'))

    # Print fits
    print("\n" + "=" * 95)
    print(f"{'variant':<48} {'rho':>7} {'s0':>7} {'R2':>7} {'med gap':>10} {'mean abs':>10}")
    print("-" * 95)
    rho_e, s0_e, r2_e, n_e = linear_fit(z_positions, erf_truth)
    erf_arr = np.array(erf_truth, dtype=float)
    print(f"{'ERF on default+MINMAX (TRUTH)':<48} {rho_e:>7.3f} {s0_e:>7.3f} "
          f"{r2_e:>7.3f}")
    for label, sigmas in results.items():
        rho, s0, r2, n = linear_fit(z_positions, sigmas)
        sigma_arr = np.array(sigmas, dtype=float)
        valid = np.isfinite(sigma_arr) & np.isfinite(erf_arr)
        gap = sigma_arr[valid] - erf_arr[valid]
        med = np.median(gap) if len(gap) else float('nan')
        mab = np.mean(np.abs(gap)) if len(gap) else float('nan')
        print(f"{label:<48} {rho:>7.3f} {s0:>7.3f} {r2:>7.3f} "
              f"{med:>+10.3f} {mab:>10.3f}")

    # Plot all variants
    fig, ax = plt.subplots(figsize=(11, 7))
    z_arr = np.array(z_positions)
    ax.scatter(z_arr, erf_truth, s=40, color='black', label='ERF truth', zorder=5)

    colours = plt.cm.tab10.colors
    for i, (label, sigmas) in enumerate(results.items()):
        ax.scatter(z_arr, sigmas, s=18, marker=['o', 's', '^', 'v', 'D', 'P'][i % 6],
                   color=colours[i], alpha=0.7, label=label)

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Sigma (px)')
    ax.set_title('All six existing-codebase preprocessing variants — model output\n'
                 'Black = ERF truth (target curve)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "six_variants_through_model.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
