"""Test: if we change the synthetic generation order from
'flatten -> blur' to 'blur -> flatten', do synthetic crops actually
match real calibration sphere crops at the same true sigma?

If yes: the architecture proposal works (and is worth the retraining).
If no: even the new order doesn't bridge the gap.

Procedure:
  1. Pick calibration sphere at z=+3 (true sigma=3.27 from ERF)
  2. Apply simple-mode flatten (current pipeline) -> 'real_simple'
  3. Apply inference-mode flatten (40px) -> 'real_inference_mode'
  4. Take a SHARP preprocessed droplet (from existing data)
  5. Generate synthetic three ways:
     a. Current: input is already-flattened, apply Gaussian sigma=3.27 -> 'synth_current'
     b. New (3px feather, blur first): apply Gaussian, then re-flatten with 3px -> 'synth_new_3px'
     c. New (40px feather, blur first): apply Gaussian, then re-flatten with 40px -> 'synth_new_40px'
  6. Compare radial profiles + pixel histograms between real and each synth variant
  7. Run model on each synth variant — does prediction match true sigma?

Then on calibration:
  8. Run model on real_simple (currently broken) and real_inference_mode (proposed)
  9. Does inference_mode flatten preserve enough blur for model to predict close to true?
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

from sphere_processing import (
    find_sphere_center, crop_to_square, flatten_sphere_crop,
)
from cine_loader import CineLoader


MODEL_SIZE = 256
CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
BUNDLE_DIR = _REPO_ROOT / "Calibration" / "runs" / "20260426_045138_camera-g"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def to_model(img: np.ndarray) -> np.ndarray:
    """Resize to MODEL_SIZE, return float32 [0,1]."""
    if img.shape[0] != MODEL_SIZE or img.shape[1] != MODEL_SIZE:
        interp = cv2.INTER_AREA if max(img.shape) > MODEL_SIZE else cv2.INTER_CUBIC
        img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.max() > 1.5:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def radial_profile(img_01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_01.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = min(h, w) // 2
    profile = np.zeros(max_r + 1)
    for ri in range(max_r + 1):
        mask = r == ri
        if mask.any():
            profile[ri] = img_01[mask].mean()
    return np.arange(max_r + 1), profile


def flatten_simple(img_01, feather=3):
    flat, info = flatten_sphere_crop(img_01.astype(np.float32), feather=feather,
                                       inner_margin=0, flatten_exterior=True)
    return flat if info is not None else img_01


def flatten_inference(img_01, feather=40):
    flat, info = flatten_sphere_crop(img_01.astype(np.float32), feather=feather,
                                       inner_margin=1, flatten_exterior=True)
    return flat if info is not None else img_01


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get a calibration frame at z=+3
    measurements = pd.read_csv(BUNDLE_DIR / "measurements.csv")
    target_z = 3.0
    target_idx = (measurements['z_mm'] - target_z).abs().idxmin()
    chosen = measurements.loc[target_idx]
    target_sigma = float(chosen['sigma_px'])
    cine_name = Path(chosen['filename']).stem + ".cine"
    cine_path = CINE_DIR / cine_name
    print(f"Calibration: {cine_path.name}, z={chosen['z_mm']:+.2f}, "
          f"true_sigma={target_sigma:.2f}px (ERF on default-mode)")

    loader = CineLoader(str(cine_path))
    raw = np.asarray(loader._load_frame(loader.frame_range[0]), dtype=np.float32)
    cx, cy, r_ = find_sphere_center(raw)
    crop = crop_to_square(raw, cx, cy, r_, padding=1.2)
    crop_f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)

    real_simple = flatten_simple(crop_f, feather=3)
    real_inference = flatten_inference(crop_f, feather=40)

    # 2. Get a sharp preprocessed droplet
    prep_pngs = list(_REPO_ROOT.glob("Preprocessing/OUTPUT/**/g/crops/sphere*g_crop.png"))
    if not prep_pngs:
        print("ERROR: no preprocessing PNG found")
        return
    sharp_path = prep_pngs[0]
    sharp_uint8 = cv2.imread(str(sharp_path), cv2.IMREAD_GRAYSCALE)
    sharp_f = sharp_uint8.astype(np.float32) / 255.0
    print(f"Sharp source: {sharp_path.name}, shape={sharp_uint8.shape}")

    # 3. Generate synthetic three ways at the same source-pixel sigma.
    # Convert target_sigma (calibration-pixel) to source-pixel via the
    # source crop's resolution. Sphere source is 850px ish, sharp source
    # is 299px. The synthetic generator uses source-pixel sigma directly.
    # For a fair COMPARISON-IN-MODEL-SPACE, the sigma needs to be in the
    # SAME frame. Real sphere: target_sigma=3.27 px on a 850 source -> 0.98 px in 256 model space.
    # Synth at 299 source -> we want 0.98 px in model space -> 0.98 * 299/256 = 1.14 px in 299 source space.
    sphere_source_size = max(crop.shape[:2])
    sigma_in_model_space = target_sigma * (MODEL_SIZE / sphere_source_size)
    sigma_in_synth_source = sigma_in_model_space * (sharp_f.shape[0] / MODEL_SIZE)
    print(f"\nTarget sigma at MODEL scale = {sigma_in_model_space:.2f} px "
          f"(real sphere src={sphere_source_size}px, synth src={sharp_f.shape[0]}px, "
          f"synth-source sigma = {sigma_in_synth_source:.2f} px)")

    # a. Current synthetic order: flatten input is already-flat sharp, then blur
    synth_current = cv2.GaussianBlur(sharp_f, (0, 0), sigma_in_synth_source)

    # b. New order: blur first then flatten with 3px
    synth_new_3px = flatten_simple(
        cv2.GaussianBlur(sharp_f, (0, 0), sigma_in_synth_source), feather=3)

    # c. New order: blur first then flatten with 40px
    synth_new_40px = flatten_inference(
        cv2.GaussianBlur(sharp_f, (0, 0), sigma_in_synth_source), feather=40)

    # 4. Resize all to model size for fair comparison
    panels = {
        'REAL — simple-mode (current)': to_model(real_simple),
        'REAL — inference-mode (40px)': to_model(real_inference),
        'SYNTH current (flatten then blur)': to_model(synth_current),
        'SYNTH new 3px (blur then flatten)': to_model(synth_new_3px),
        'SYNTH new 40px (blur then flatten)': to_model(synth_new_40px),
    }

    # 5. Load model
    print("\nLoading model...")
    models = sorted(
        (_REPO_ROOT / "Training" / "training_output" / "models")
        .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    # 6. Predict on each + compute radial profile + plot
    fig, axs = plt.subplots(2, len(panels), figsize=(3 * len(panels), 7))
    print(f"\nModel predictions (true sigma in model space = {sigma_in_model_space:.2f} px):")
    for i, (name, img) in enumerate(panels.items()):
        ax = axs[0, i]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        pred = inf.estimate_blur_from_image((img * 255).astype(np.uint8))
        ax.set_title(f"{name}\nmodel pred = {pred:.2f} px",
                     fontsize=8)
        ax.axis('off')
        print(f"  {name:<42}  predicted = {pred:.3f} px  "
              f"gap = {pred - sigma_in_model_space:+.2f}")

        # Radial profile (zoomed to edge)
        ax = axs[1, i]
        r, p = radial_profile(img)
        ax.plot(r, p, linewidth=1.6)
        # Mark the contour transition zone
        # Find where intensity is between 0.1 and 0.9
        edge_band = (p > 0.1) & (p < 0.9)
        if edge_band.any():
            edge_r = r[edge_band]
            ax.axvspan(edge_r.min(), edge_r.max(), alpha=0.2, color='orange')
            edge_width = edge_r.max() - edge_r.min()
            ax.set_title(f"radial profile\nedge width 10-90% = {edge_width} px",
                         fontsize=8)
        ax.set_xlabel('radius (px)')
        ax.set_ylabel('intensity')
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"Real calibration sphere (z=+3, true sigma={target_sigma:.2f}) vs "
        f"three synthetic generation orders\n"
        f"True blur in model space = {sigma_in_model_space:.2f} px",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUTPUT_DIR / "synthetic_order_comparison.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
