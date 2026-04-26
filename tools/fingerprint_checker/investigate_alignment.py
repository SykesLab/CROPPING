"""Phase 1 deep investigation — empirically explore what aligns calibration
to synthetic across multiple frames, padding values, and matched defoci.

What it produces:
  - tools/fingerprint_checker/output/flatten_alignment/
      multi_frame_grid.png       — same calibration frame, multi-defocus
      padding_sweep.png          — one frame, padding 1.2/1.5/1.94/2.5/3.0/4.0
      whitepad_compare.png       — white-padded calibration vs synthetic
      metric_polarity_check.txt  — Otsu mask vs known-polarity mask deltas

This is a pure investigation script — read-only, no production code touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Path setup
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
SYNTH_DIR = (
    _REPO_ROOT / "Training" / "training_output" / "datasets"
    / "20260423_200211_newpreprocessingallcams"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


# ── Loading helpers ───────────────────────────────────────────────────────


def load_cine_frame(cine_path: Path) -> np.ndarray:
    loader = CineLoader(str(cine_path))
    raw = loader._load_frame(loader.frame_range[0])
    return np.asarray(raw, dtype=np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.5:
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_model_size(img: np.ndarray) -> np.ndarray:
    interp = cv2.INTER_AREA if max(img.shape) > MODEL_SIZE else cv2.INTER_CUBIC
    return cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)


# ── Crop variants ─────────────────────────────────────────────────────────


def calibration_crop(
    raw_frame: np.ndarray, cx: int, cy: int, radius: int,
    padding: float, white_pad_to_size: int | None = None,
) -> np.ndarray:
    """Crop sphere from raw frame, optionally extending with white padding.

    If white_pad_to_size is provided AND larger than the natural crop, the
    crop is extended by adding white pixels around it. This simulates a
    wider field of view than the camera actually captured.
    """
    crop = crop_to_square(raw_frame, cx, cy, radius, padding=padding)
    h, w = crop.shape[:2]

    if white_pad_to_size and white_pad_to_size > max(h, w):
        # Determine source intensity scale (raw is float 0..255 typically)
        if crop.max() > 1.5:
            white_value = 255.0
        else:
            white_value = 1.0
        target = white_pad_to_size
        pad_top = (target - h) // 2
        pad_bottom = target - h - pad_top
        pad_left = (target - w) // 2
        pad_right = target - w - pad_left
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=white_value,
        )
    return crop


def flatten_after_crop(crop: np.ndarray, feather_src: int) -> np.ndarray:
    """Run simple-mode flatten (interior=0 / exterior=1) with given feather."""
    if crop.dtype == np.uint8:
        f = crop.astype(np.float32) / 255.0
    else:
        f = crop.astype(np.float32)
        if f.max() > 1.5:
            f = f / 255.0
    flat, info = flatten_sphere_crop(
        f, feather=feather_src, inner_margin=0, flatten_exterior=True,
    )
    if info is None:
        return f
    return flat


# ── Honest polarity-aware metrics (read directly from rendered binary) ───


def manual_bg_obj_means(image_01: np.ndarray, polarity: str = 'auto'):
    """Honest bg/obj means using a hard threshold on values, not Otsu.

    polarity='dark_object' assumes object pixels are < 0.5; bg pixels > 0.5.
    polarity='light_object' is the reverse.
    polarity='auto' picks based on mean intensity.
    """
    if polarity == 'auto':
        polarity = 'dark_object' if image_01.mean() > 0.5 else 'light_object'
    if polarity == 'dark_object':
        bg_mask = image_01 > 0.7
        obj_mask = image_01 < 0.3
    else:
        bg_mask = image_01 < 0.3
        obj_mask = image_01 > 0.7
    if bg_mask.sum() < 10 or obj_mask.sum() < 10:
        return float('nan'), float('nan'), float('nan'), float('nan'), polarity
    bg_mean = float(image_01[bg_mask].mean())
    bg_std = float(image_01[bg_mask].std())
    obj_mean = float(image_01[obj_mask].mean())
    contrast = (obj_mean - bg_mean) / max(bg_std, 1e-6)
    return bg_mean, bg_std, obj_mean, contrast, polarity


# ── TEST 1: padding sweep on one frame ────────────────────────────────────


def test_padding_sweep():
    """Show what each padding value produces visually, and quantify
    the white-padding option."""
    print("\n" + "=" * 70)
    print("TEST 1: Padding sweep (one frame, multiple padding values)")
    print("=" * 70)

    cine = CINE_DIR / "9mm_51.cine"  # z = +3 mm
    raw = load_cine_frame(cine)
    h_frame, w_frame = raw.shape[:2]
    cx, cy, r = find_sphere_center(raw)
    print(f"Frame {cine.name}: shape={raw.shape}, sphere r={r}, "
          f"max possible padding from frame = {min(w_frame, h_frame) / (2*r):.2f}")
    print(f"To match synthetic (~52% sphere/crop), need padding ~1.94")

    paddings = [1.2, 1.5, 1.94, 2.5, 3.0, 4.0]
    panels = {}

    # First: pure crop_to_square (image-clamped, no white extension)
    for pad in paddings:
        crop = calibration_crop(raw, cx, cy, r, padding=pad)
        ratio = 2 * r / max(crop.shape)
        flat = flatten_after_crop(crop, feather_src=8)
        flat256 = to_model_size(flat)
        panels[f"pad={pad} (clamped)\nratio={ratio:.0%}"] = flat256

    # Then: same but with white-padding to true target size
    for pad in [1.94, 2.5, 3.0]:
        target_size = int(round(2 * r * pad))
        crop = calibration_crop(raw, cx, cy, r, padding=pad,
                                 white_pad_to_size=target_size)
        ratio = 2 * r / max(crop.shape)
        flat = flatten_after_crop(crop, feather_src=8)
        flat256 = to_model_size(flat)
        panels[f"pad={pad} +whitepad\nratio={ratio:.0%}"] = flat256

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    axs = np.atleast_2d(axs)
    for i, (name, img) in enumerate(panels.items()):
        r_, c_ = i // cols, i % cols
        ax = axs[r_, c_]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        bg_m, bg_s, obj_m, con, pol = manual_bg_obj_means(img)
        ax.set_title(
            f"{name}\nbg={bg_m:.3f} obj={obj_m:.3f}\n"
            f"con={con:.1f} pol={pol[:5]}",
            fontsize=8,
        )
        ax.axis('off')
    for j in range(len(panels), rows * cols):
        axs[j // cols, j % cols].axis('off')
    fig.suptitle(
        "Padding sweep — first 6 use crop_to_square only (clamped to frame). "
        "Last 3 add white pixel padding to actually achieve the target.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUTPUT_DIR / "padding_sweep.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote: {p}")


# ── TEST 2: multi-frame compare across defocus ────────────────────────────


def test_multi_frame():
    """Process multiple frames at different z, compare all approaches against
    matching synthetic samples. The goal is to see whether the gap is
    consistent or varies with defocus."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-frame — same processing, different z values")
    print("=" * 70)

    measurements = pd.read_csv(BUNDLE_DIR / "measurements.csv")
    synth_meta = pd.read_csv(SYNTH_DIR / "metadata.csv")

    # Pick frames at z = 0, +2, +4 (focal, mid, high defocus on positive side)
    target_zs = [0.0, 2.0, 4.0]
    rows_data = []
    for z in target_zs:
        idx = (measurements['z_mm'] - z).abs().idxmin()
        rows_data.append(measurements.loc[idx])

    # For each frame, generate THREE versions:
    # - Current calibration ("default" mode, padding=1.2, image-clamped)
    # - Proposed: simple flatten + white-pad to padding=1.94
    # - Proposed: simple flatten + white-pad to padding=3.0
    # Plus matched synthetic for reference

    n_rows = len(rows_data)
    n_cols = 4  # current, whitepad194, whitepad300, synthetic
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.5))
    axs = np.atleast_2d(axs)

    print(f"\n{'frame':<14} {'z':>6} {'sigma':>6} | "
          f"{'variant':<22} {'bg':>7} {'obj':>7} {'con':>7} {'polarity':<14}")
    print("-" * 100)

    for row_i, row in enumerate(rows_data):
        cine_name = Path(row['filename']).stem + ".cine"
        cine = CINE_DIR / cine_name
        raw = load_cine_frame(cine)
        cx, cy, r_ = find_sphere_center(raw)
        z = float(row['z_mm']); sigma = float(row['sigma_px'])

        variants = {}

        # 1. Current calibration default mode (matches what's in the bundle)
        crop = calibration_crop(raw, cx, cy, r_, padding=1.2)
        if crop.dtype == np.uint8:
            f = crop.astype(np.float32) / 255.0
        else:
            f = crop.astype(np.float32) / 255.0 if crop.max() > 1.5 else crop.astype(np.float32)
        # Apply default mode: inner_margin=20, no exterior flatten
        flat, info = flatten_sphere_crop(
            f, feather=3, inner_margin=20, flatten_exterior=False,
        )
        # Then per-image normalize like calibration_gui currently does
        normed = cv2.normalize(flat, None, 0, 1.0, cv2.NORM_MINMAX) if info else f
        variants['current (default+MINMAX)'] = to_model_size(normed)

        # 2. White-padded to 1.94
        target = int(round(2 * r_ * 1.94))
        crop2 = calibration_crop(raw, cx, cy, r_, padding=1.94, white_pad_to_size=target)
        flat2 = flatten_after_crop(crop2, feather_src=8)
        variants['whitepad pad=1.94'] = to_model_size(flat2)

        # 3. White-padded to 3.0
        target = int(round(2 * r_ * 3.0))
        crop3 = calibration_crop(raw, cx, cy, r_, padding=3.0, white_pad_to_size=target)
        flat3 = flatten_after_crop(crop3, feather_src=12)
        variants['whitepad pad=3.0'] = to_model_size(flat3)

        # 4. Matching synthetic sample
        nearest_synth = synth_meta.iloc[
            (synth_meta['sigma_applied_px'] - sigma).abs().argsort()[:1]
        ].iloc[0]
        synth_path = SYNTH_DIR / "blur" / f"{int(nearest_synth['index']):06d}.png"
        synth_img = cv2.imread(str(synth_path), cv2.IMREAD_GRAYSCALE)
        variants['SYNTHETIC (target)'] = to_model_size(synth_img.astype(np.float32) / 255.0)

        # Plot row
        for col_i, (name, img) in enumerate(variants.items()):
            ax = axs[row_i, col_i]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            bg, std, obj, con, pol = manual_bg_obj_means(img)
            ax.set_title(
                f"z={z:+.1f} σ={sigma:.2f}\n{name}\n"
                f"bg={bg:.2f} obj={obj:.2f} con={con:.1f}",
                fontsize=8,
            )
            ax.axis('off')
            print(f"{cine_name:<14} {z:>6.1f} {sigma:>6.2f} | "
                  f"{name:<22} {bg:>7.3f} {obj:>7.3f} {con:>7.2f} {pol:<14}")

    fig.suptitle("Multi-frame at z=0, +2, +4 mm: current vs proposed vs synthetic target",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUTPUT_DIR / "multi_frame_grid.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


# ── TEST 3: Otsu-vs-known-polarity metric bias ────────────────────────────


def test_metric_polarity_bias():
    """For images where we KNOW the polarity (because we just rendered them
    binary), compare what manual-mask metrics give vs what the existing
    fingerprint metric (Otsu) gives. Quantify the bias."""
    print("\n" + "=" * 70)
    print("TEST 3: Metric polarity bias — Otsu mask vs known-polarity mask")
    print("=" * 70)

    from tools.fingerprint_checker.fingerprint_metrics import (
        _to_float01, detect_object_mask,
        metric_background_mean, metric_object_bg_contrast,
    )

    # Generate a series of synthetic test images: a black disk on white bg,
    # with varying disk-fill-fraction
    cases = []
    for fill_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        img = np.ones((256, 256), dtype=np.float32)
        radius = int(np.sqrt(fill_pct / 100 / np.pi) * 256)
        cv2.circle(img, (128, 128), radius, 0.0, -1)
        cases.append((fill_pct, img))

    print(f"\n{'fill%':>5} {'TRUE bg':>10} {'TRUE obj':>10} {'TRUE con':>10} | "
          f"{'OTSU bg':>10} {'OTSU con':>10} {'mismatch':>10}")
    print("-" * 90)
    rows = []
    for fill_pct, img in cases:
        true_bg, true_std, true_obj, true_con, _ = manual_bg_obj_means(img)
        # Otsu-based metric
        mask = detect_object_mask(img)
        otsu_bg = metric_background_mean(img, mask)
        otsu_con = metric_object_bg_contrast(img, mask)
        mismatch = ("FLIPPED" if abs(otsu_bg - true_obj) < abs(otsu_bg - true_bg)
                    else "ok")
        rows.append((fill_pct, true_bg, true_obj, true_con, otsu_bg, otsu_con, mismatch))
        print(f"{fill_pct:>5} {true_bg:>10.3f} {true_obj:>10.3f} {true_con:>10.2f} | "
              f"{otsu_bg:>10.3f} {otsu_con:>10.2f}  {mismatch}")

    # Save text version
    p = OUTPUT_DIR / "metric_polarity_check.txt"
    with open(p, 'w') as f:
        f.write("Fill%  TRUE_bg  TRUE_obj  TRUE_con  OTSU_bg  OTSU_con  Status\n")
        for r in rows:
            f.write(f"{r[0]:>5}  {r[1]:>7.3f}  {r[2]:>7.3f}  {r[3]:>7.2f}  "
                    f"{r[4]:>7.3f}  {r[5]:>7.2f}  {r[6]}\n")
    print(f"\nWrote: {p}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_padding_sweep()
    test_multi_frame()
    test_metric_polarity_bias()
    print("\n" + "=" * 70)
    print("All tests complete. Outputs in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
