"""Use the calibration sphere AT FOCUS as a sharp template for synthetic
blur, then test the model on it.

This isolates the question: is the issue that real defocus blur differs
from synthetic Gaussian blur, OR that calibration crops never look like
training data due to processing differences?

Procedure:
  1. Take calibration sphere at z=0 (in focus)
  2. Process it the SAME WAY synthetic generator processes its sharp templates:
     simple-mode flatten (interior=0, exterior=1, 3px feather)
  3. Apply Gaussian blur at sigmas 1, 2, 3, 4, 5, 6, 7 (in model space, scaled to source)
  4. Resize to 256, run model on each
  5. Plot model prediction vs true Gaussian sigma applied

If predictions track applied sigma (slope ~1, intercept ~0):
  -> Model handles sphere-shape templates fine, the issue is
     that calibration doesn't apply Gaussian blur explicitly.
  -> Fix: use sphere-as-template for synthetic, retrain.

If predictions are still flat or off:
  -> The model fundamentally can't read sphere-shape edges.
  -> Need different approach.

Also compare to:
  - Same procedure on a real DROPLET source (control, should work since this is what model trained on)
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

from sphere_processing import find_sphere_center, crop_to_square, flatten_sphere_crop
from cine_loader import CineLoader

MODEL_SIZE = 256
CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
BUNDLE_DIR = _REPO_ROOT / "Calibration" / "runs" / "20260426_045138_camera-g"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def to_model(img):
    if img.shape[0] != MODEL_SIZE or img.shape[1] != MODEL_SIZE:
        interp = cv2.INTER_AREA if max(img.shape) > MODEL_SIZE else cv2.INTER_CUBIC
        img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img, 0, 1).astype(np.float32) if img.max() <= 1.5 else img.astype(np.float32) / 255.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Find the in-focus calibration frame (z=0 = stage 7.0mm based on earlier test)
    measurements = pd.read_csv(BUNDLE_DIR / "measurements.csv")
    in_focus_idx = measurements['z_mm'].abs().idxmin()
    in_focus_row = measurements.loc[in_focus_idx]
    print(f"In-focus frame: {in_focus_row['filename']}, z={in_focus_row['z_mm']:.2f}, "
          f"true_sigma={in_focus_row['sigma_px']:.2f}px")

    cine_path = CINE_DIR / (Path(in_focus_row['filename']).stem + ".cine")
    loader = CineLoader(str(cine_path))
    raw = np.asarray(loader._load_frame(loader.frame_range[0]), dtype=np.float32)
    cx, cy, r_ = find_sphere_center(raw)

    # Process this in-focus sphere with simple-mode flatten -> sharp sphere template
    crop = crop_to_square(raw, cx, cy, r_, padding=1.2)
    f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
    sharp_sphere, _ = flatten_sphere_crop(f, feather=3, inner_margin=0, flatten_exterior=True)
    sharp_sphere = np.clip(sharp_sphere, 0, 1).astype(np.float32)
    print(f"Sharp sphere template: shape={sharp_sphere.shape}")

    # Also load a sharp DROPLET (control)
    prep_pngs = list(_REPO_ROOT.glob("Preprocessing/OUTPUT/**/g/crops/sphere*g_crop.png"))
    sharp_droplet = cv2.imread(str(prep_pngs[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    print(f"Sharp droplet template: shape={sharp_droplet.shape}")

    # Load model
    print("\nLoading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    # 2. Apply Gaussian blur at various sigmas (model-space) and run model
    target_model_sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    sphere_results = []
    droplet_results = []
    for target_model_sigma in target_model_sigmas:
        # For sphere: source size = max of crop dims, scale sigma to source space
        sphere_source_sigma = target_model_sigma * (max(sharp_sphere.shape) / MODEL_SIZE)
        sphere_blurred = cv2.GaussianBlur(sharp_sphere, (0, 0), sphere_source_sigma)
        sphere_256 = to_model(sphere_blurred)
        sphere_pred = float(inf.estimate_blur_from_image((sphere_256 * 255).astype(np.uint8)))
        sphere_results.append((target_model_sigma, sphere_pred, sphere_256))

        # For droplet: source size = 299
        droplet_source_sigma = target_model_sigma * (sharp_droplet.shape[0] / MODEL_SIZE)
        droplet_blurred = cv2.GaussianBlur(sharp_droplet, (0, 0), droplet_source_sigma)
        droplet_256 = to_model(droplet_blurred)
        droplet_pred = float(inf.estimate_blur_from_image((droplet_256 * 255).astype(np.uint8)))
        droplet_results.append((target_model_sigma, droplet_pred, droplet_256))

    # 3. Plot results
    fig, axs = plt.subplots(3, len(target_model_sigmas), figsize=(2.5 * len(target_model_sigmas), 9))

    print(f"\n{'target σ':>10} {'sphere pred':>12} {'sphere gap':>11} | "
          f"{'droplet pred':>13} {'droplet gap':>12}")
    print("-" * 70)
    for i, (target, (_, sphere_pred, sphere_img),
            (_, droplet_pred, droplet_img)) in enumerate(
            zip(target_model_sigmas, sphere_results, droplet_results)):
        print(f"{target:>10.1f} {sphere_pred:>12.2f} {sphere_pred-target:>+11.2f} | "
              f"{droplet_pred:>13.2f} {droplet_pred-target:>+12.2f}")

        axs[0, i].imshow(sphere_img, cmap='gray', vmin=0, vmax=1)
        axs[0, i].set_title(f"σ_target={target}\nSPHERE\nmodel={sphere_pred:.2f}",
                             fontsize=8)
        axs[0, i].axis('off')

        axs[1, i].imshow(droplet_img, cmap='gray', vmin=0, vmax=1)
        axs[1, i].set_title(f"DROPLET (control)\nmodel={droplet_pred:.2f}", fontsize=8)
        axs[1, i].axis('off')

    # Combined scatter
    big_ax = fig.add_subplot(3, 1, 3)
    sphere_targets = [s[0] for s in sphere_results]
    sphere_preds = [s[1] for s in sphere_results]
    droplet_preds = [d[1] for d in droplet_results]
    big_ax.plot(sphere_targets, sphere_preds, 'o-', label='Sphere template',
                color='tab:green', markersize=8)
    big_ax.plot(sphere_targets, droplet_preds, 's-', label='Droplet template (control)',
                color='tab:orange', markersize=8)
    big_ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, label='y=x (perfect)')
    big_ax.set_xlabel('Applied Gaussian σ (model-px)')
    big_ax.set_ylabel('Model predicted σ (model-px)')
    big_ax.set_title('Model prediction vs applied Gaussian sigma\n'
                     'Both pipelines use simple-mode flattened source + Gaussian blur')
    big_ax.legend(fontsize=10)
    big_ax.grid(alpha=0.3)
    big_ax.set_xlim(0, 10)
    big_ax.set_ylim(0, 12)

    fig.tight_layout()
    p = OUTPUT_DIR / "sphere_as_template.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")

    # Summary
    sphere_arr = np.array(sphere_preds)
    target_arr = np.array(sphere_targets)
    droplet_arr = np.array(droplet_preds)
    print("\nSummary:")
    print(f"  Sphere template: mean abs gap = {np.mean(np.abs(sphere_arr - target_arr)):.3f} px")
    print(f"  Droplet template (control): mean abs gap = {np.mean(np.abs(droplet_arr - target_arr)):.3f} px")
    if np.allclose(sphere_arr, target_arr, atol=0.5):
        print("  ✓ Sphere template gives correct predictions! Issue is just"
              " that calibration doesn't run through the synthetic-blur pipeline.")
    elif np.std(sphere_arr) < 0.5:
        print("  ✗ Sphere template gives flat predictions — model cannot"
              " read sphere shapes regardless of preprocessing.")
    else:
        print("  ? Sphere template gives mixed results — needs further investigation.")


if __name__ == '__main__':
    main()
