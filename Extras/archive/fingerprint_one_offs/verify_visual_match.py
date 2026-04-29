"""Direct visual + edge-profile comparison: synthetic at matched
sigma+diameter vs calibration sphere processed with simple-mode flatten.

The user's question: do they actually look different, or are they
visually identical and the model is failing for some other reason?
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
SYNTH_DIR = (_REPO_ROOT / "Training" / "training_output" / "datasets"
             / "20260423_200211_newpreprocessingallcams")
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def to_model(img):
    if img.shape[0] != MODEL_SIZE or img.shape[1] != MODEL_SIZE:
        interp = cv2.INTER_AREA if max(img.shape) > MODEL_SIZE else cv2.INTER_CUBIC
        img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img, 0, 1).astype(np.float32) if img.max() <= 1.5 else img.astype(np.float32) / 255.0


def radial_profile(img):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2).astype(int)
    max_r = min(h, w) // 2
    profile = np.zeros(max_r + 1)
    for ri in range(max_r + 1):
        m = r == ri
        if m.any():
            profile[ri] = img[m].mean()
    return np.arange(max_r + 1), profile


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pick three z-values and matched synthetic samples
    measurements = pd.read_csv(BUNDLE_DIR / "measurements.csv")
    synth_meta = pd.read_csv(SYNTH_DIR / "metadata.csv")

    # We want z values where calibration's true sigma matches some synthetic sample's sigma
    # AND where synthetic has comparable diameter (~212 model-px to match cal sphere's 217)
    z_targets = [1.0, 3.0, 5.0]  # mid range, mid-high, high

    pairs = []
    for z in z_targets:
        cal_idx = (measurements['z_mm'] - z).abs().idxmin()
        cal_row = measurements.loc[cal_idx]
        true_sigma = float(cal_row['sigma_px'])

        # Find synth at matching sigma AND diameter ≈ 212 (calibration's diameter in model space)
        candidates = synth_meta[
            (synth_meta['sigma_applied_px'].between(true_sigma - 0.05, true_sigma + 0.05))
            & (synth_meta['diameter_model_px'] > 200)
        ]
        if candidates.empty:
            print(f"WARNING: no matching synth for z={z}, sigma={true_sigma}")
            continue
        synth_row = candidates.iloc[0]
        pairs.append({
            'z': z, 'true_sigma': true_sigma,
            'cal_filename': cal_row['filename'],
            'cal_cine': Path(cal_row['filename']).stem + ".cine",
            'synth_idx': int(synth_row['index']),
            'synth_sigma': float(synth_row['sigma_applied_px']),
            'synth_diameter': float(synth_row['diameter_model_px']),
        })

    # Load model
    print("Loading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    # For each pair, render and compare
    fig, axs = plt.subplots(len(pairs), 4, figsize=(16, 4 * len(pairs)))
    if len(pairs) == 1:
        axs = axs.reshape(1, -1)

    for i, p in enumerate(pairs):
        # Load + process calibration
        cine_path = CINE_DIR / p['cal_cine']
        loader = CineLoader(str(cine_path))
        raw = np.asarray(loader._load_frame(loader.frame_range[0]), dtype=np.float32)
        cx, cy, r_ = find_sphere_center(raw)
        crop = crop_to_square(raw, cx, cy, r_, padding=1.2)
        f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
        flat, _ = flatten_sphere_crop(f, feather=3, inner_margin=0, flatten_exterior=True)
        cal_img = to_model(flat)

        # Load synthetic
        synth_png = SYNTH_DIR / "blur" / f"{p['synth_idx']:06d}.png"
        synth_img = to_model(cv2.imread(str(synth_png), cv2.IMREAD_GRAYSCALE))

        # Difference image
        diff = synth_img - cal_img

        # Predictions
        cal_pred = float(inf.estimate_blur_from_image((cal_img * 255).astype(np.uint8)))
        synth_pred = float(inf.estimate_blur_from_image((synth_img * 255).astype(np.uint8)))

        # Plot
        axs[i, 0].imshow(cal_img, cmap='gray', vmin=0, vmax=1)
        axs[i, 0].set_title(
            f"CALIBRATION (simple flatten)\n"
            f"z={p['z']}mm true_sigma={p['true_sigma']:.2f}px (model: {cal_pred:.2f})",
            fontsize=9)
        axs[i, 0].axis('off')

        axs[i, 1].imshow(synth_img, cmap='gray', vmin=0, vmax=1)
        axs[i, 1].set_title(
            f"SYNTHETIC (matched)\n"
            f"sigma_applied={p['synth_sigma']:.2f}px diam={p['synth_diameter']:.0f} "
            f"(model: {synth_pred:.2f})",
            fontsize=9)
        axs[i, 1].axis('off')

        im = axs[i, 2].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axs[i, 2].set_title(
            f"SYNTH - CAL difference\n"
            f"max abs={np.abs(diff).max():.2f}  mean abs={np.abs(diff).mean():.3f}",
            fontsize=9)
        axs[i, 2].axis('off')
        plt.colorbar(im, ax=axs[i, 2], fraction=0.04)

        # Radial profiles overlaid
        ax = axs[i, 3]
        r_cal, p_cal = radial_profile(cal_img)
        r_syn, p_syn = radial_profile(synth_img)
        ax.plot(r_cal, p_cal, label=f'CALIBRATION (model→{cal_pred:.2f})',
                linewidth=2, color='tab:blue')
        ax.plot(r_syn, p_syn, label=f'SYNTHETIC (model→{synth_pred:.2f})',
                linewidth=2, color='tab:orange', linestyle='--')
        ax.set_xlabel('radius (px)')
        ax.set_ylabel('intensity')
        ax.set_title(f'Radial profile\n(true sigma in model-space = '
                     f'{p["true_sigma"] * MODEL_SIZE / max(crop.shape):.2f}px)',
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Side-by-side: simple-mode calibration sphere vs matched synthetic",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUTPUT_DIR / "visual_match_comparison.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
