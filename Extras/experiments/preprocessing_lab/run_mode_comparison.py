"""Run all four preprocessing modes on the user's calibration data,
measure ERF + run trained model on each, produce comparison report.

Usage:
    python -m Extras.experiments.preprocessing_lab.run_mode_comparison

Outputs (under tools/preprocessing_lab/output/):
    mode_comparison.csv  — per-frame data
    mode_comparison.png  — ERF + model curves overlaid by mode
    sample_grid.png      — visual side-by-side at 3 z values

The four modes compared:
    default   — current calibration GUI behaviour
                (inner_margin=20, no exterior fill, MINMAX after)
    simple    — preprocessing's convention
                (3px feather, full binary, no MINMAX)
    inference — training_gui inference data prep
                (40px feather, inner_margin=1, full binary)
    unified   — candidate recipe from this lab
                (model-space-relative feather, full binary, resize last)

This is read-only with respect to the rest of the project — uses
existing functions, modifies nothing.
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

from sphere_processing import (
    crop_to_square, find_consensus_sphere, flatten_sphere_crop,
)
from blur_measurement import measure_blur_auto

from Extras.experiments.preprocessing_lab.unified_preprocess import unified_preprocess
from Extras.experiments.preprocessing_lab.visualize import (
    OUTPUT_DIR_DEFAULT, linear_fit, load_calibration_stack,
    load_inference_model, plot_mode_curves, plot_sample_grid,
)


# Existing-mode invocations (mirroring what _apply_sphere_pipeline
# does internally). Kept here so this script is self-contained and
# doesn't depend on the existing modes' source code being unchanged.
def process_default(raw, cx, cy, radius):
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = (crop.astype(np.float32) / 255.0
         if crop.max() > 1.5 else crop.astype(np.float32))
    flat, info = flatten_sphere_crop(
        f, feather=3, inner_margin=20, flatten_exterior=False)
    if info is None:
        flat = f
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_simple(raw, cx, cy, radius):
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = (crop.astype(np.float32) / 255.0
         if crop.max() > 1.5 else crop.astype(np.float32))
    flat, info = flatten_sphere_crop(
        f, feather=3, inner_margin=0, flatten_exterior=True)
    if info is None:
        flat = f
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8)


def process_inference(raw, cx, cy, radius):
    crop = crop_to_square(raw, cx, cy, radius, padding=1.2)
    f = (crop.astype(np.float32) / 255.0
         if crop.max() > 1.5 else crop.astype(np.float32))
    flat, info = flatten_sphere_crop(
        f, feather=40, inner_margin=1, flatten_exterior=True)
    if info is None:
        flat = f
    return (np.clip(flat, 0, 1) * 255).astype(np.uint8)


def process_unified(raw, cx, cy, radius,
                     target_feather_model_px=30.0, model_size=256):
    return unified_preprocess(
        raw, cx, cy, radius,
        target_feather_model_px=target_feather_model_px,
        model_size=model_size,
    )


MODES = {
    "default":   process_default,
    "simple":    process_simple,
    "inference": process_inference,
    "unified":   process_unified,
}

MODE_COLOURS = {
    "default":   "tab:red",
    "simple":    "tab:green",
    "inference": "tab:purple",
    "unified":   "tab:blue",
}


def measure_erf(processed_uint8):
    try:
        m = measure_blur_auto(
            processed_uint8, center=None, radius=None,
            method="erf", verbose=False)
        return float(m.sigma) if m.confidence > 0.5 else float("nan")
    except Exception:
        return float("nan")


def predict_model(processed_uint8, inf):
    try:
        return float(inf.estimate_blur_from_image(processed_uint8))
    except Exception:
        return float("nan")


def main():
    OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

    print("Loading calibration stack...")
    raw_frames, z_positions, _ = load_calibration_stack()
    print(f"  {len(raw_frames)} frames loaded")

    cx, cy, radius = find_consensus_sphere(raw_frames, upper_only=True)
    print(f"  Consensus sphere: center=({cx},{cy}) radius={radius}")

    print("Loading model...")
    inf = load_inference_model()

    print(f"\nRunning {len(MODES)} modes × {len(raw_frames)} frames "
          f"× (ERF + model)...")
    rows = []
    for i, (raw, z) in enumerate(zip(raw_frames, z_positions)):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(raw_frames)}")
        for mode_name, fn in MODES.items():
            proc = fn(raw, cx, cy, radius)
            erf = measure_erf(proc)
            mod = predict_model(proc, inf)
            rows.append({
                "z_mm": z, "mode": mode_name,
                "erf_sigma": erf, "model_sigma": mod,
            })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR_DEFAULT / "mode_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    # Build per-mode series for plotting
    z_arr = np.array(z_positions)
    series = {}
    for mode_name in MODES.keys():
        sub = df[df["mode"] == mode_name].sort_values("z_mm")
        # Re-align to z_arr order (sub is sorted by z; z_arr is the
        # original load order — we need them aligned)
        sub_by_z = dict(zip(sub["z_mm"], zip(sub["erf_sigma"],
                                              sub["model_sigma"])))
        erf_aligned = np.array(
            [sub_by_z.get(z, (np.nan, np.nan))[0] for z in z_arr])
        mod_aligned = np.array(
            [sub_by_z.get(z, (np.nan, np.nan))[1] for z in z_arr])
        series[mode_name] = {
            "erf":    erf_aligned,
            "model":  mod_aligned,
            "colour": MODE_COLOURS[mode_name],
        }

    plot_path = OUTPUT_DIR_DEFAULT / "mode_comparison.png"
    series = plot_mode_curves(z_arr, series, plot_path,
                               title="Preprocessing mode comparison "
                                     "— ERF and model on the same calibration stack")
    print(f"Wrote: {plot_path}")

    # Print fits
    print("\n" + "=" * 80)
    print("FITS — sigma = rho * |z| + sigma_0")
    print("=" * 80)
    print(f"{'mode':<10}  ERF: {'rho':>7} {'s0':>7} {'R2':>6}  | "
          f"MODEL: {'rho':>7} {'s0':>7} {'R2':>6}")
    print("-" * 80)
    for mode_name in MODES.keys():
        e = series[mode_name].get("erf_fit", (np.nan,) * 4)
        m = series[mode_name].get("model_fit", (np.nan,) * 4)
        print(f"{mode_name:<10}  ERF: {e[0]:>7.3f} {e[1]:>7.3f} {e[2]:>6.3f}  | "
              f"MODEL: {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>6.3f}")

    # Sample grid: pick 3 z values spread across the stack
    sample_z = []
    for target_z in [0.0, 3.0, 5.0]:
        idx = int(np.argmin(np.abs(z_arr - target_z)))
        if z_arr[idx] not in sample_z:
            sample_z.append(z_arr[idx])
    samples = {}
    for z_val in sample_z:
        idx = int(np.argmin(np.abs(z_arr - z_val)))
        raw = raw_frames[idx]
        samples[float(z_val)] = {}
        for mode_name, fn in MODES.items():
            samples[float(z_val)][mode_name] = fn(raw, cx, cy, radius)
    grid_path = OUTPUT_DIR_DEFAULT / "sample_grid.png"
    plot_sample_grid(samples, grid_path,
                      title=f"Sample frames at z = {sample_z} per mode")
    print(f"Wrote: {grid_path}")

    print("\nDone. Open the PNGs to compare modes visually.")


if __name__ == "__main__":
    main()
