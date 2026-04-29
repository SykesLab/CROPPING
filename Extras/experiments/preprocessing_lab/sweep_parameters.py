"""Parameter sweep for the unified preprocessing recipe.

Sweeps `target_feather_model_px` (and optionally other params) over
the calibration stack, runs the trained model on each variant, plots
all curves overlaid so the user can see which feather setting gives
the best slope match to ERF truth.

Usage:
    python -m Extras.experiments.preprocessing_lab.sweep_parameters

Output (under tools/preprocessing_lab/output/):
    sweep_unified_feather.png  — all sweep variants overlaid
    sweep_unified_feather.csv  — per-frame data
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sphere_processing import find_consensus_sphere
from blur_measurement import measure_blur_auto

from Extras.experiments.preprocessing_lab.run_mode_comparison import (
    process_default, MODE_COLOURS,
)
from Extras.experiments.preprocessing_lab.unified_preprocess import unified_preprocess
from Extras.experiments.preprocessing_lab.visualize import (
    OUTPUT_DIR_DEFAULT, linear_fit, load_calibration_stack,
    load_inference_model,
)


# Sweep grid
FEATHER_VALUES = [10, 15, 20, 30, 40, 60]
MODEL_SIZE = 256


def measure_erf(processed_uint8):
    try:
        m = measure_blur_auto(
            processed_uint8, center=None, radius=None,
            method="erf", verbose=False)
        return float(m.sigma) if m.confidence > 0.5 else float("nan")
    except Exception:
        return float("nan")


def main():
    OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

    print("Loading calibration stack...")
    raw_frames, z_positions, _ = load_calibration_stack()
    cx, cy, radius = find_consensus_sphere(raw_frames, upper_only=True)
    print(f"  Consensus sphere: ({cx},{cy}) r={radius}")

    print("Loading model...")
    inf = load_inference_model()

    # First: measure ERF truth from default-mode crops (the existing
    # calibration measurement convention).
    print("\nComputing ERF truth (default-mode crops)...")
    erf_truth = []
    for raw in raw_frames:
        proc_def = process_default(raw, cx, cy, radius)
        erf_truth.append(measure_erf(proc_def))

    rho_truth, s0_truth, r2_truth, _ = linear_fit(z_positions, erf_truth)
    print(f"  ERF truth fit: rho={rho_truth:.3f}  s0={s0_truth:.3f}  "
          f"R²={r2_truth:.3f}")

    # Sweep
    print(f"\nSweeping {len(FEATHER_VALUES)} feather values × "
          f"{len(raw_frames)} frames × model inference...")
    rows = []
    for f_idx, feather_model_px in enumerate(FEATHER_VALUES):
        print(f"  [{f_idx+1}/{len(FEATHER_VALUES)}] "
              f"target_feather_model_px = {feather_model_px}")
        for raw, z in zip(raw_frames, z_positions):
            proc = unified_preprocess(
                raw, cx, cy, radius,
                target_feather_model_px=float(feather_model_px),
                model_size=MODEL_SIZE,
            )
            try:
                model_sigma = float(inf.estimate_blur_from_image(proc))
            except Exception:
                model_sigma = float("nan")
            rows.append({
                "z_mm": z,
                "feather_model_px": feather_model_px,
                "model_sigma": model_sigma,
            })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR_DEFAULT / "sweep_unified_feather.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6.5))
    z_arr = np.array(z_positions)

    # ERF truth as black reference
    ax.scatter(z_arr, erf_truth, s=40, color="black", alpha=0.85,
               label=f"ERF truth (default mode)\n"
                     f"rho={rho_truth:.3f} s0={s0_truth:.3f}",
               zorder=5)
    if np.isfinite(rho_truth):
        zfit = np.linspace(z_arr.min() - 0.5, z_arr.max() + 0.5, 100)
        ax.plot(zfit, rho_truth * np.abs(zfit) + s0_truth,
                "k-", linewidth=1.4, alpha=0.5)

    # Model output per feather
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(FEATHER_VALUES)))
    for i, feather_model_px in enumerate(FEATHER_VALUES):
        sub = df[df["feather_model_px"] == feather_model_px].sort_values("z_mm")
        sub_by_z = dict(zip(sub["z_mm"], sub["model_sigma"]))
        aligned = np.array([sub_by_z.get(z, np.nan) for z in z_arr])
        rho_m, s0_m, r2_m, _ = linear_fit(z_positions, aligned)
        slope_ratio = rho_m / rho_truth if np.isfinite(rho_m) else float("nan")
        ax.scatter(z_arr, aligned, s=20, color=cmap[i], alpha=0.65,
                   marker=["o", "s", "^", "v", "D", "P"][i % 6],
                   label=f"feather={feather_model_px}px  "
                         f"rho={rho_m:.3f} (×{slope_ratio:.2f}) "
                         f"s0={s0_m:.3f} R²={r2_m:.3f}")
        if np.isfinite(rho_m):
            ax.plot(zfit, rho_m * np.abs(zfit) + s0_m,
                    color=cmap[i], alpha=0.4, linewidth=1.0)

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Sigma (px)")
    ax.set_title("Unified-mode parameter sweep — model output vs target_feather_model_px\n"
                 "Black = ERF truth. Pick the feather where slope ratio is closest to 1.0.")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = OUTPUT_DIR_DEFAULT / "sweep_unified_feather.png"
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {plot_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY — ratio of model-rho to ERF-truth-rho per feather")
    print("(closer to 1.0 = better slope match)")
    print("=" * 80)
    for feather_model_px in FEATHER_VALUES:
        sub = df[df["feather_model_px"] == feather_model_px].sort_values("z_mm")
        rho_m, _, r2_m, _ = linear_fit(sub["z_mm"], sub["model_sigma"])
        ratio = rho_m / rho_truth if np.isfinite(rho_m) else float("nan")
        print(f"  feather={feather_model_px:>3}px  "
              f"rho_model={rho_m:>6.3f}  "
              f"slope_ratio={ratio:>6.3f}  R²={r2_m:>6.3f}")


if __name__ == "__main__":
    main()
