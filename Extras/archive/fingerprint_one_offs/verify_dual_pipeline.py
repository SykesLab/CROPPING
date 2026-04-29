"""Comprehensive verification: ERF vs model across all 61 z values, in
default-mode and simple-mode preprocessings.

Question: is the proposed dual-pipeline architecture self-consistent?
That is, does the model's prediction on simple-mode images agree with
the ERF measurement on default-mode images at every z?

If yes (curves overlay): we can safely use ERF on default-mode for
deriving rho/sigma_0, and feed simple-mode crops to the model at
inference time. Both pipelines yield the same effective sigma.

If no (curves diverge): the dual-pipeline is broken and we'd need to
either (a) retrain the model on default-mode images or (b) re-derive
calibration from simple-mode images.
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


def find_focal_plane(cines: list[Path], pos_map: dict) -> float:
    """Return stage_position with maximum Laplacian variance (= in focus)."""
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
        crop_size = min(h, w) // 3
        centre = raw[cy0 - crop_size:cy0 + crop_size,
                     cx0 - crop_size:cx0 + crop_size]
        lap = cv2.Laplacian(centre, cv2.CV_32F, ksize=3)
        v = float(lap.var())
        if v > best_var:
            best_var = v
            best_stage = pos_map[cine.name]
    return best_stage


def process_default(raw, cx, cy, radius):
    """Current calibration GUI preprocessing: default mode + MINMAX."""
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    flat, info = flatten_sphere_crop(f, feather=3, inner_margin=20,
                                       flatten_exterior=False)
    if info is None:
        return (f * 255).astype(np.uint8)
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_simple(raw, cx, cy, radius):
    """Proposed save-for-model preprocessing: simple mode, no MINMAX."""
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    flat, info = flatten_sphere_crop(f, feather=3, inner_margin=0,
                                       flatten_exterior=True)
    if info is None:
        return (f * 255).astype(np.uint8)
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8)


def linear_fit(z, sigma):
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = np.isfinite(sigma) & (np.abs(z) > 0.5)
    z = z[valid]; sigma = sigma[valid]
    if len(z) < 4:
        return float('nan'), float('nan'), float('nan'), 0
    abs_z = np.abs(z)
    def model(zz, rho, s0):
        return rho * zz + s0
    try:
        popt, _ = curve_fit(model, abs_z, sigma, p0=[1.0, 0.5],
                            bounds=([0.01, 0], [100, 50]))
    except Exception:
        return float('nan'), float('nan'), float('nan'), 0
    rho, s0 = popt
    pred = model(abs_z, rho, s0)
    ss_res = ((sigma - pred) ** 2).sum()
    ss_tot = ((sigma - sigma.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rho, s0, r2, len(z)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load .cine files and stage positions
    positions_df = pd.read_csv(POSITIONS_CSV)
    pos_map = dict(zip(positions_df['filename'], positions_df['stage_position_mm']))
    cines = sorted(CINE_DIR.glob("*.cine"),
                   key=lambda p: pos_map.get(p.name, float('inf')))
    print(f"Found {len(cines)} .cine files")

    # 2. Find focal plane
    print("Finding focal plane via Laplacian variance...")
    focus_stage = find_focal_plane(cines, pos_map)
    print(f"  Focal plane: stage = {focus_stage:.2f} mm")

    # 3. Load all frames + consensus geometry
    raw_frames = []
    z_positions = []
    for cine in cines:
        loader = CineLoader(str(cine))
        if loader.cine_obj is None:
            continue
        raw = np.asarray(loader._load_frame(loader.frame_range[0]),
                         dtype=np.float32)
        raw_frames.append(raw)
        z_positions.append(pos_map[cine.name] - focus_stage)
    print(f"Loaded {len(raw_frames)} frames")
    cx, cy, radius = find_consensus_sphere(raw_frames, upper_only=True)
    print(f"Consensus sphere: center=({cx},{cy}) radius={radius}")

    # 4. Load model
    print("\nLoading model...")
    models = sorted(
        (_REPO_ROOT / "Training" / "training_output" / "models")
        .glob("*/checkpoints/dme_best.pth"), reverse=True)
    if not models:
        print("ERROR: no model checkpoint found")
        return
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')
    print(f"  Model: {models[0].parent.parent.name}")

    # 5. Per frame: process two ways, run BOTH ERF and model on EACH
    print("\nRunning ERF + model on every frame in BOTH preprocessings...")
    print("(this is 61 frames * 2 modes * 2 measurement methods = 244 ops)")

    rows = []
    for raw, z in zip(raw_frames, z_positions):
        proc_def = process_default(raw, cx, cy, radius)
        proc_sim = process_simple(raw, cx, cy, radius)

        # ERF on default
        ph, pw = proc_def.shape[:2]
        try:
            m = measure_blur_erf(proc_def, center=(pw // 2, ph // 2),
                                  radius=min(pw, ph) * 4 // 10, verbose=False)
            erf_default = float(m.sigma) if m.confidence > 0.5 else float('nan')
        except Exception:
            erf_default = float('nan')

        # ERF on simple (will mostly fail — binary destroys signal)
        ph, pw = proc_sim.shape[:2]
        try:
            m = measure_blur_erf(proc_sim, center=(pw // 2, ph // 2),
                                  radius=min(pw, ph) * 4 // 10, verbose=False)
            erf_simple = float(m.sigma) if m.confidence > 0.5 else float('nan')
        except Exception:
            erf_simple = float('nan')

        # Model on default
        try:
            model_default = float(inf.estimate_blur_from_image(proc_def))
        except Exception:
            model_default = float('nan')

        # Model on simple
        try:
            model_simple = float(inf.estimate_blur_from_image(proc_sim))
        except Exception:
            model_simple = float('nan')

        rows.append({
            'z_mm': z,
            'erf_default': erf_default,
            'erf_simple': erf_simple,
            'model_default': model_default,
            'model_simple': model_simple,
        })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "dual_pipeline_per_frame.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote per-frame: {csv_path}")

    # 6. Fit rho, sigma_0 for each method
    print("\nLinear fits (sigma = rho * |z| + sigma_0):")
    print(f"{'method':<24} {'rho':>8} {'sigma_0':>9} {'R2':>8} {'n_used':>8}")
    print("-" * 64)
    fits = {}
    for col in ('erf_default', 'erf_simple', 'model_default', 'model_simple'):
        rho, s0, r2, n = linear_fit(df['z_mm'].values, df[col].values)
        fits[col] = (rho, s0, r2, n)
        print(f"{col:<24} {rho:>8.4f} {s0:>9.4f} {r2:>8.4f} {n:>8}")

    # 7. Pairwise sigma agreement at each z
    print("\nAgreement between ERF (default) and model (simple), per frame:")
    print(f"{'z':>6} {'ERF def':>9} {'model sim':>10} {'gap':>7} {'gap%':>7}")
    print("-" * 50)
    agreements = []
    for _, r in df.iterrows():
        if np.isfinite(r['erf_default']) and np.isfinite(r['model_simple']):
            gap = r['model_simple'] - r['erf_default']
            gap_pct = (100 * gap / max(r['erf_default'], 0.1))
            agreements.append((r['z_mm'], gap, gap_pct))
            print(f"{r['z_mm']:>+6.1f} {r['erf_default']:>9.3f} "
                  f"{r['model_simple']:>10.3f} {gap:>+7.3f} {gap_pct:>+6.1f}%")
    if agreements:
        gaps = np.array([a[1] for a in agreements])
        print(f"\n  median gap: {np.median(gaps):+.3f} px")
        print(f"  mean abs gap: {np.mean(np.abs(gaps)):.3f} px")
        print(f"  max abs gap: {np.max(np.abs(gaps)):.3f} px")

    # 8. Plot all four curves
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    z = df['z_mm'].values
    ax = axs[0]
    for col, color, marker, label in [
        ('erf_default', 'tab:blue', 'o', 'ERF on default-mode (current calibration)'),
        ('model_simple', 'tab:green', 's', 'Model on simple-mode (proposed inference)'),
        ('model_default', 'tab:red', 'x', 'Model on default-mode (BROKEN — model never saw this)'),
        ('erf_simple', 'tab:orange', '^', 'ERF on simple-mode (binary destroys signal)'),
    ]:
        valid = np.isfinite(df[col].values)
        ax.scatter(z[valid], df[col].values[valid], s=22, color=color,
                   marker=marker, label=label, alpha=0.75)
        rho, s0, r2, n = fits[col]
        if np.isfinite(rho):
            zfit = np.linspace(-8, 6, 100)
            ax.plot(zfit, rho * np.abs(zfit) + s0, color=color,
                    linewidth=1.4, alpha=0.7,
                    label=f"  fit: rho={rho:.3f} s0={s0:.3f} R²={r2:.3f} n={n}")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Measured sigma (px)")
    ax.set_title("Sigma vs z — all four method×preprocessing combinations")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 40)

    # Zoomed-in agreement plot
    ax = axs[1]
    valid_both = np.isfinite(df['erf_default'].values) & np.isfinite(df['model_simple'].values)
    if valid_both.any():
        z_v = z[valid_both]
        e_v = df['erf_default'].values[valid_both]
        m_v = df['model_simple'].values[valid_both]
        ax.scatter(z_v, e_v, s=40, color='tab:blue', label='ERF on default-mode', alpha=0.7)
        ax.scatter(z_v, m_v, s=40, color='tab:green', marker='s',
                   label='Model on simple-mode', alpha=0.7)
        # Fit lines
        rho_e, s0_e, _, _ = fits['erf_default']
        rho_m, s0_m, _, _ = fits['model_simple']
        zfit = np.linspace(-8, 6, 100)
        ax.plot(zfit, rho_e * np.abs(zfit) + s0_e, 'b-', alpha=0.6,
                label=f'ERF fit: rho={rho_e:.3f}')
        ax.plot(zfit, rho_m * np.abs(zfit) + s0_m, 'g--', alpha=0.6,
                label=f'Model fit: rho={rho_m:.3f}')
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Measured sigma (px)")
    ax.set_title("ZOOM — does ERF(default) agree with Model(simple)?")
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 8)

    fig.tight_layout()
    p = OUTPUT_DIR / "dual_pipeline_verify.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
