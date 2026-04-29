"""Like-for-like calibration vs synthetic comparison.

Picks ONE calibration frame at known z and finds the synthetic samples that
match BOTH defocus AND object size. Then:
  1. Side-by-side image comparison (both at MODEL_SIZE)
  2. Radial intensity profile from object center outward (overlay)
  3. Pixel-histogram comparison
  4. (If model checkpoint available) inference on both — compare predicted sigma

This is the test the user is asking for: does the apparent domain gap
between calibration and synthetic disappear when we compare matched
samples instead of the heterogeneous synthetic average?
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


MODEL_SIZE = 256
BUNDLE_DIR = _REPO_ROOT / "Calibration" / "runs" / "20260426_045138_camera-g"
SYNTH_DIR = (
    _REPO_ROOT / "Training" / "training_output" / "datasets"
    / "20260423_200211_newpreprocessingallcams"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


def to_model(img: np.ndarray) -> np.ndarray:
    """Resize to MODEL_SIZE and return float32 [0,1]."""
    if img.shape[0] != MODEL_SIZE or img.shape[1] != MODEL_SIZE:
        interp = cv2.INTER_AREA if max(img.shape) > MODEL_SIZE else cv2.INTER_CUBIC
        img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.max() > 1.5:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def radial_profile(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean intensity at each integer radius from image centre.

    Returns (radii_px, mean_intensity_at_radius).
    """
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = min(h, w) // 2
    profile = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)
    for ri in range(max_r + 1):
        mask = r == ri
        if mask.any():
            profile[ri] = img[mask].mean()
            counts[ri] = mask.sum()
    radii = np.arange(max_r + 1)
    return radii, profile


def estimate_object_diameter_px(img_01: np.ndarray) -> float:
    """Rough diameter estimate by counting dark pixels and assuming a circle."""
    dark = (img_01 < 0.3).sum()
    return 2.0 * np.sqrt(dark / np.pi)


def match_synthetic(
    synth_meta: pd.DataFrame, target_sigma: float, target_diameter_px: float,
    n: int = 5,
):
    """Find n synthetic samples closest to BOTH target sigma AND target diameter.

    Uses a normalised L2 distance in (sigma, diameter) space.
    """
    sigma_range = synth_meta['sigma_applied_px'].std()
    diam_range = synth_meta['diameter_model_px'].std()
    sigma_z = (synth_meta['sigma_applied_px'] - target_sigma).abs() / sigma_range
    diam_z = (synth_meta['diameter_model_px'] - target_diameter_px).abs() / diam_range
    score = sigma_z + diam_z
    best_idx = score.nsmallest(n).index
    return synth_meta.loc[best_idx]


def try_load_model():
    """Try to load the user's model checkpoint. Returns inference fn or None."""
    try:
        import torch
        # Find latest model
        models_dir = _REPO_ROOT / "Training" / "training_output" / "models"
        if not models_dir.is_dir():
            return None, None
        candidates = sorted(
            [p for p in models_dir.glob("*/checkpoints/dme_best.pth")],
            reverse=True,
        )
        if not candidates:
            return None, None
        ckpt_path = candidates[0]
        print(f"\nFound model checkpoint: {ckpt_path}")
        sys.path.insert(0, str(_REPO_ROOT / "Training"))
        from inference_real_crops import RealCropInference  # type: ignore
        inf = RealCropInference(model_path=str(ckpt_path), device='cpu')
        return inf, ckpt_path
    except Exception as e:
        print(f"\nModel loading failed: {e}")
        return None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load calibration sample at z=+3 mm
    measurements = pd.read_csv(BUNDLE_DIR / "measurements.csv")
    target_z = 3.0
    target_idx = (measurements['z_mm'] - target_z).abs().idxmin()
    chosen = measurements.loc[target_idx]
    target_sigma = float(chosen['sigma_px'])
    print(f"Calibration anchor: {chosen['filename']}, "
          f"z={chosen['z_mm']:+.2f} mm, sigma={target_sigma:.2f} px")

    calib_png = BUNDLE_DIR / "processed_images" / chosen['filename']
    calib_raw = cv2.imread(str(calib_png), cv2.IMREAD_GRAYSCALE)
    calib_01 = to_model(calib_raw)
    calib_diameter_modelpx = estimate_object_diameter_px(calib_01)
    print(f"  Calibration crop in model space: shape={calib_01.shape}, "
          f"estimated diameter = {calib_diameter_modelpx:.0f} px "
          f"({calib_diameter_modelpx/MODEL_SIZE*100:.1f}% of frame)")

    # ── Find synthetic match — first defocus-only, then defocus+diameter
    synth_meta = pd.read_csv(SYNTH_DIR / "metadata.csv")
    print()
    print(f"Synthetic search: target sigma={target_sigma:.2f}, "
          f"target diameter={calib_diameter_modelpx:.0f} px")

    # Method A: defocus-only nearest (what the fingerprint tool does)
    method_a = synth_meta.iloc[
        (synth_meta['sigma_applied_px'] - target_sigma).abs().argsort()[:5]
    ]
    print()
    print("Method A — defocus-only matched (current fingerprint tool):")
    for _, r in method_a.iterrows():
        print(f"  idx={int(r['index']):>5}  sigma={r['sigma_applied_px']:.2f}  "
              f"diameter={r['diameter_model_px']:.0f} px ({r['diameter_model_px']/MODEL_SIZE*100:.0f}%)")

    # Method B: matched on defocus AND diameter (proposed fix)
    method_b = match_synthetic(synth_meta, target_sigma, calib_diameter_modelpx, n=5)
    print()
    print("Method B — matched on defocus AND diameter (proposed):")
    for _, r in method_b.iterrows():
        print(f"  idx={int(r['index']):>5}  sigma={r['sigma_applied_px']:.2f}  "
              f"diameter={r['diameter_model_px']:.0f} px ({r['diameter_model_px']/MODEL_SIZE*100:.0f}%)")

    # ── Load representative samples from each method
    def load_synth(idx: int) -> np.ndarray:
        p = SYNTH_DIR / "blur" / f"{idx:06d}.png"
        return to_model(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))

    synth_a = load_synth(int(method_a.iloc[0]['index']))
    synth_b = load_synth(int(method_b.iloc[0]['index']))

    # ── Side-by-side image grid + radial profiles
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    images = [
        ('CALIBRATION', calib_01,
         f"z={chosen['z_mm']:+.2f}mm σ={target_sigma:.2f}\n"
         f"diameter={calib_diameter_modelpx:.0f}px"),
        ('SYNTH (Method A)', synth_a,
         f"sigma_only match\nidx={int(method_a.iloc[0]['index'])}\n"
         f"σ={method_a.iloc[0]['sigma_applied_px']:.2f} d={method_a.iloc[0]['diameter_model_px']:.0f}px"),
        ('SYNTH (Method B)', synth_b,
         f"sigma+diameter match\nidx={int(method_b.iloc[0]['index'])}\n"
         f"σ={method_b.iloc[0]['sigma_applied_px']:.2f} d={method_b.iloc[0]['diameter_model_px']:.0f}px"),
        ('CALIB - SYNTH(B) diff', None, ""),
    ]
    for col, (name, img, caption) in enumerate(images):
        ax = axs[0, col]
        if img is not None:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            diff = calib_01 - synth_b
            im = ax.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
            plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(f"{name}\n{caption}", fontsize=9)
        ax.axis('off')

    # Radial profiles
    ax = axs[1, 0]
    r1, p1 = radial_profile(calib_01)
    r2, p2 = radial_profile(synth_a)
    r3, p3 = radial_profile(synth_b)
    ax.plot(r1, p1, label='Calibration', linewidth=1.8)
    ax.plot(r2, p2, label='Synth A (sigma-only)', alpha=0.7, linestyle='--')
    ax.plot(r3, p3, label='Synth B (sigma+diam)', alpha=0.7)
    ax.set_xlabel('Radius from centre (px)')
    ax.set_ylabel('Mean intensity at radius')
    ax.set_title('Radial intensity profile\n(centre → edge → background)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Pixel histograms
    ax = axs[1, 1]
    ax.hist(calib_01.ravel(), bins=50, alpha=0.5, label='Calibration', density=True)
    ax.hist(synth_b.ravel(), bins=50, alpha=0.5, label='Synth B', density=True)
    ax.set_xlabel('Pixel intensity [0,1]')
    ax.set_ylabel('Density')
    ax.set_title('Pixel intensity histogram\n(matched samples)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Edge transition zoom: pixels in [0.05, 0.95] (the ramp region)
    ax = axs[1, 2]
    edge_calib = calib_01[(calib_01 > 0.05) & (calib_01 < 0.95)]
    edge_synth = synth_b[(synth_b > 0.05) & (synth_b < 0.95)]
    bins = np.linspace(0.05, 0.95, 30)
    ax.hist(edge_calib, bins=bins, alpha=0.5, label=f'Calib (n={len(edge_calib)})')
    ax.hist(edge_synth, bins=bins, alpha=0.5, label=f'Synth B (n={len(edge_synth)})')
    ax.set_xlabel('Pixel intensity (edge-transition pixels only)')
    ax.set_ylabel('Count')
    ax.set_title('Edge-transition pixel distribution\n(only pixels at the gradient)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Model inference comparison if available
    ax = axs[1, 3]
    inf, ckpt = try_load_model()
    if inf is not None:
        try:
            print(f"\nRunning model inference on calibration + matched synthetic samples...")
            calib_pred = inf.estimate_blur_from_image((calib_01 * 255).astype(np.uint8))
            synth_a_pred = inf.estimate_blur_from_image((synth_a * 255).astype(np.uint8))
            synth_b_pred = inf.estimate_blur_from_image((synth_b * 255).astype(np.uint8))
            preds = [calib_pred, synth_a_pred, synth_b_pred]
            labels = ['Calibration', 'Synth A', 'Synth B']
            truths = [target_sigma, float(method_a.iloc[0]['sigma_applied_px']),
                      float(method_b.iloc[0]['sigma_applied_px'])]
            x = np.arange(3)
            ax.bar(x - 0.2, truths, width=0.4, label='True σ', alpha=0.7)
            ax.bar(x + 0.2, preds, width=0.4, label='Predicted σ', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel('Sigma (px)')
            ax.set_title(f'Model prediction vs truth\n'
                         f'gap on calib = {abs(calib_pred - target_sigma):.2f}px\n'
                         f'gap on synth B = {abs(synth_b_pred - method_b.iloc[0]["sigma_applied_px"]):.2f}px',
                         fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            print(f"  Calibration: true σ={target_sigma:.2f}, predicted σ={calib_pred:.2f}, "
                  f"gap={calib_pred - target_sigma:+.2f}")
            print(f"  Synth A:     true σ={method_a.iloc[0]['sigma_applied_px']:.2f}, predicted σ={synth_a_pred:.2f}, "
                  f"gap={synth_a_pred - method_a.iloc[0]['sigma_applied_px']:+.2f}")
            print(f"  Synth B:     true σ={method_b.iloc[0]['sigma_applied_px']:.2f}, predicted σ={synth_b_pred:.2f}, "
                  f"gap={synth_b_pred - method_b.iloc[0]['sigma_applied_px']:+.2f}")
        except Exception as e:
            ax.text(0.5, 0.5, f'inference failed:\n{e}', ha='center', va='center')
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'no model checkpoint found\n(skip inference comparison)',
                ha='center', va='center')
        ax.axis('off')

    fig.suptitle(f"Like-for-like comparison — calibration {chosen['filename']} "
                 f"(z={target_z}mm, σ={target_sigma:.2f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUTPUT_DIR / "like_for_like.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
