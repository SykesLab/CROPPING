"""Run ERF on RAW unprocessed .cine frames (no flatten, no MINMAX) to get
the closest thing to a preprocessing-independent sigma measurement.

Then compare against ERF on each preprocessing variant to see how much
each preprocessing biases the sigma measurement.

This addresses the question: 'what is ERF truth, where is it from?'
The answer is — whatever ERF says depends on what preprocessing it ran on.
This script makes that explicit.
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
from blur_measurement import measure_blur_erf
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

    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))

    print("Finding focal plane...")
    focus_stage = find_focal_plane(cines, pos_map)
    print(f"  stage = {focus_stage:.2f} mm")

    # Load all frames
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
    print(f"Consensus sphere on raw frames: ({cx},{cy}) r={radius}")

    # === Method A: ERF on RAW frame (no preprocessing at all) ===
    # We pass the raw float32 image directly to measure_blur_erf with the
    # consensus center+radius.
    print("\nERF on RAW unprocessed frames (no flatten, no MINMAX, no crop)...")
    sigmas_raw = []
    for raw in raw_frames:
        # measure_blur_erf needs uint8 input typically. Convert via dynamic range.
        # Since intensity range varies, scale to 0-255 using global FRAME max
        # (NOT per-image MINMAX — use the SAME scale across all frames so
        # measurements are comparable).
        # Use 0-255 from dataset's global max (computed once below).
        pass

    # Determine global max across all frames so we can scale uniformly
    global_max = max(r.max() for r in raw_frames)
    print(f"  Global max intensity across all 61 raw frames: {global_max:.1f}")

    sigmas_raw = []
    for raw in raw_frames:
        scaled = (raw / global_max * 255).astype(np.uint8)
        try:
            m = measure_blur_erf(scaled, center=(cx, cy), radius=radius,
                                  verbose=False)
            sigmas_raw.append(float(m.sigma) if m.confidence > 0.5 else float('nan'))
        except Exception:
            sigmas_raw.append(float('nan'))

    # === Method B: ERF on RAW with per-image scaling (typical case) ===
    sigmas_raw_minmax = []
    for raw in raw_frames:
        scaled = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        try:
            m = measure_blur_erf(scaled, center=(cx, cy), radius=radius,
                                  verbose=False)
            sigmas_raw_minmax.append(float(m.sigma) if m.confidence > 0.5 else float('nan'))
        except Exception:
            sigmas_raw_minmax.append(float('nan'))

    # === Method C: ERF on default-mode processed (the existing pipeline) ===
    sigmas_default = []
    for raw in raw_frames:
        crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
        f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
        flat, _ = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                        flatten_exterior=False)
        proc = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ph, pw = proc.shape[:2]
        try:
            m = measure_blur_erf(proc, center=(pw//2, ph//2),
                                  radius=min(pw, ph)*4//10, verbose=False)
            sigmas_default.append(float(m.sigma) if m.confidence > 0.5 else float('nan'))
        except Exception:
            sigmas_default.append(float('nan'))

    df = pd.DataFrame({
        'z_mm': z_positions,
        'erf_raw_global_scale': sigmas_raw,
        'erf_raw_minmax': sigmas_raw_minmax,
        'erf_default_pipeline': sigmas_default,
    })
    df.to_csv(OUTPUT_DIR / "raw_erf_comparison.csv", index=False)

    print("\nLinear fits — what each method calls 'rho':")
    for col, label in [
        ('erf_raw_global_scale', 'RAW frame, global-scaled (closest to physical truth)'),
        ('erf_raw_minmax',       'RAW frame, per-image MINMAX'),
        ('erf_default_pipeline', 'Default-mode processed (existing calibration pipeline)'),
    ]:
        rho, s0, r2, n = linear_fit(df['z_mm'], df[col])
        print(f"  {label}")
        print(f"    rho={rho:.4f}  s0={s0:.4f}  R²={r2:.4f}  n_used={n}")

    # Plot all three
    fig, ax = plt.subplots(figsize=(11, 6))
    z_arr = df['z_mm'].values
    for col, color, label in [
        ('erf_raw_global_scale', 'tab:blue', 'RAW frame, global-scaled'),
        ('erf_raw_minmax', 'tab:orange', 'RAW frame, per-image MINMAX'),
        ('erf_default_pipeline', 'tab:green', 'Default-mode pipeline (existing)'),
    ]:
        valid = df[col].notna()
        ax.scatter(z_arr[valid], df[col][valid], s=24, color=color, alpha=0.7,
                   label=label)
        rho, s0, _, _ = linear_fit(z_arr, df[col])
        if np.isfinite(rho):
            zfit = np.linspace(-8, 6, 100)
            ax.plot(zfit, rho * np.abs(zfit) + s0, color=color, alpha=0.4,
                    linewidth=1.4)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('ERF measured sigma (px)')
    ax.set_title('Three ways of measuring sigma per z, all using ERF\n'
                 'They give DIFFERENT rho — there is no single "ERF truth"')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "raw_erf_comparison.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
