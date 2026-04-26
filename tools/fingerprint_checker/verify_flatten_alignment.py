"""Phase 1 verification — render the same calibration frame through each
candidate flatten setting and compare against synthetic + preprocessing
references. One-shot diagnostic, throwaway script.

What it produces:
  - tools/fingerprint_checker/output/flatten_alignment/grid.png
      Visual side-by-side of all variants
  - tools/fingerprint_checker/output/flatten_alignment/metrics.txt
      Numeric table of background_mean, background_std,
      object_bg_contrast, edge_transition_width per variant

Usage:
  python -m tools.fingerprint_checker.verify_flatten_alignment

Decision criteria (from plan):
  background_mean delta vs synthetic       < 0.05
  background_std delta vs synthetic        < 0.02
  object_bg_contrast delta vs synthetic    abs < 5
  edge_transition_width delta vs synthetic abs < 1.0 px
"""

from __future__ import annotations

import sys
from pathlib import Path

# Path setup — same convention as the GUI
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
    _apply_sphere_pipeline, find_consensus_sphere, find_sphere_center,
    flatten_sphere_crop, crop_to_square,
)
from cine_loader import CineLoader

from tools.fingerprint_checker.fingerprint_metrics import (
    _to_float01, detect_object_mask,
    metric_background_mean, metric_background_std, metric_object_bg_contrast,
    metric_edge_transition_width,
)


MODEL_SIZE = 256


# ── Inputs ────────────────────────────────────────────────────────────────

CINE_DIR = _REPO_ROOT / "calibration spheres" / "9mm"
TARGET_Z_MM = 3.0
BUNDLE_DIR = _REPO_ROOT / "Calibration" / "runs" / "20260426_045138_camera-g"
SYNTH_DIR = (
    _REPO_ROOT / "Training" / "training_output" / "datasets"
    / "20260423_200211_newpreprocessingallcams"
)
PREPROCESS_PNG_PATTERN = "Preprocessing/OUTPUT/**/g/crops/sphere*g_crop.png"

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"


# ── Helpers ───────────────────────────────────────────────────────────────


def model_space_flatten(
    image: np.ndarray,
    target_feather_model_px: float,
    cx: int, cy: int, radius: int,
    target_inner_margin_model_px: float = 0.0,
    flatten_exterior: bool = True,
    padding: float = 1.94,
) -> np.ndarray:
    """Phase-1 reference impl of the proposed model-space flatten.

    Crops to square first (so source_size reflects the crop), then
    flattens with feather scaled from model space to source space.

    `padding` defaults to 1.94 to match preprocessing's effective
    object-diameter / crop-width ratio (~52%, computed empirically
    from existing synthetic dataset metadata).
    """
    proc = crop_to_square(image, cx, cy, radius, padding=padding)
    if proc.dtype == np.uint8:
        proc_f = proc.astype(np.float32) / 255.0
    else:
        proc_f = proc.astype(np.float32)
        if proc_f.max() > 1.0:
            proc_f /= proc_f.max()
    h, w = proc_f.shape[:2]
    source_size = max(h, w)
    scale = source_size / MODEL_SIZE
    feather_src = max(1, round(target_feather_model_px * scale))
    inner_src = max(0, round(target_inner_margin_model_px * scale))
    flat, info = flatten_sphere_crop(
        proc_f, feather=feather_src, inner_margin=inner_src,
        flatten_exterior=flatten_exterior,
    )
    if info is None:
        return proc_f
    return flat


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def to_model_size(img: np.ndarray) -> np.ndarray:
    """Resize to MODEL_SIZE × MODEL_SIZE for visual comparison."""
    if img.shape[0] == MODEL_SIZE and img.shape[1] == MODEL_SIZE:
        return img
    interp = cv2.INTER_AREA if max(img.shape) > MODEL_SIZE else cv2.INTER_CUBIC
    return cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)


def metrics_summary(image: np.ndarray) -> dict:
    """Compute the four headline metrics for the verification."""
    img01 = _to_float01(image)
    mask = detect_object_mask(img01)
    return {
        'background_mean': metric_background_mean(img01, mask),
        'background_std': metric_background_std(img01, mask),
        'object_bg_contrast': metric_object_bg_contrast(img01, mask),
        'edge_transition_width': metric_edge_transition_width(img01, mask),
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Find the cine that maps to z = +TARGET_Z_MM
    measurements = pd.read_csv(BUNDLE_DIR / "measurements.csv")
    nearest_idx = (measurements['z_mm'] - TARGET_Z_MM).abs().idxmin()
    chosen = measurements.loc[nearest_idx]
    print(f"Picked calibration frame: {chosen['filename']}  "
          f"(z={chosen['z_mm']:+.2f} mm, sigma={chosen['sigma_px']:.3f} px)")
    chosen_z = float(chosen['z_mm'])
    chosen_sigma = float(chosen['sigma_px'])
    cine_name = Path(chosen['filename']).stem + ".cine"
    cine_path = CINE_DIR / cine_name
    if not cine_path.is_file():
        print(f"ERROR: missing {cine_path}")
        sys.exit(1)

    # 2. Load the .cine frame
    print(f"Loading {cine_path.name}...")
    loader = CineLoader(str(cine_path))
    raw_frame = loader._load_frame(loader.frame_range[0])
    if raw_frame is None:
        print(f"ERROR: could not read {cine_path}")
        sys.exit(1)
    raw_frame = np.asarray(raw_frame, dtype=np.float32)
    print(f"  Raw frame: shape={raw_frame.shape}, dtype={raw_frame.dtype}, "
          f"range=[{raw_frame.min():.3f}, {raw_frame.max():.3f}]")

    # 3. Detect sphere on this frame
    detection = find_sphere_center(raw_frame)
    if detection is None:
        print("ERROR: sphere detection failed")
        sys.exit(1)
    cx, cy, radius = detection
    print(f"  Sphere: center=({cx},{cy}) radius={radius}")

    # 4. Process with all six variants
    variants = {}

    # Existing modes via _apply_sphere_pipeline
    for mode in ("simple", "inference", "default"):
        out = _apply_sphere_pipeline(
            raw_frame, cx, cy, radius,
            mirror=False, blacken=False, flatten=True, flatten_mode=mode,
        )
        variants[f"{mode} (current)"] = out

    # Proposed model-space flatten with three target feathers (padding=1.94)
    for target in (3, 6, 12):
        out = model_space_flatten(
            raw_frame, target_feather_model_px=float(target),
            cx=cx, cy=cy, radius=radius, padding=1.94,
        )
        variants[f"ms f={target}px pad=1.94"] = to_uint8(out)

    # Same three at the OLD padding (1.2) for direct A/B comparison
    for target in (3, 6, 12):
        out = model_space_flatten(
            raw_frame, target_feather_model_px=float(target),
            cx=cx, cy=cy, radius=radius, padding=1.2,
        )
        variants[f"ms f={target}px pad=1.2"] = to_uint8(out)

    # 5. Reference images for comparison: synthetic at matching sigma + preprocessing sample
    synth_meta = pd.read_csv(SYNTH_DIR / "metadata.csv")
    target_sigma = chosen_sigma
    nearest_synth = synth_meta.iloc[
        (synth_meta['sigma_applied_px'] - target_sigma).abs().argsort()[:1]
    ]
    s_row = nearest_synth.iloc[0]
    synth_blur_path = SYNTH_DIR / "blur" / f"{int(s_row['index']):06d}.png"
    if not synth_blur_path.is_file():
        print(f"ERROR: synthetic image missing: {synth_blur_path}")
        sys.exit(1)
    synth_img = cv2.imread(str(synth_blur_path), cv2.IMREAD_GRAYSCALE)
    print(f"\nSynthetic ref: {synth_blur_path.name}  "
          f"(sigma_applied={s_row['sigma_applied_px']:.3f}, "
          f"defocus={s_row['defocus_mm']:+.3f}, "
          f"source={s_row['source_image']}), shape={synth_img.shape}")

    # Find a preprocessing sample (any sharp crop from camera g)
    prep_pngs = list(_REPO_ROOT.glob(PREPROCESS_PNG_PATTERN))
    if prep_pngs:
        prep_path = prep_pngs[0]
        prep_img = cv2.imread(str(prep_path), cv2.IMREAD_GRAYSCALE)
        print(f"Preprocessing ref: {prep_path.relative_to(_REPO_ROOT)}, shape={prep_img.shape}")
    else:
        prep_img = None
        print("WARNING: no preprocessing PNG found — skipping reference")

    # 6. Resize everything to model size for visual & metric comparison
    panels = {}
    for name, img in variants.items():
        panels[name] = to_model_size(img)
    panels['SYNTHETIC ref'] = to_model_size(synth_img)
    if prep_img is not None:
        panels['PREPROCESSING ref'] = to_model_size(prep_img)

    # 7. Compute metrics on the model-sized versions
    metrics_per_panel = {name: metrics_summary(img) for name, img in panels.items()}

    # 8. Write metrics table
    print("\n" + "=" * 88)
    print("METRICS (computed on model-space resized images)")
    print("=" * 88)
    header = f"{'variant':<28}  {'bg_mean':>9}  {'bg_std':>8}  {'contrast':>9}  {'edge_w':>8}"
    print(header)
    print("-" * len(header))
    synth_m = metrics_per_panel['SYNTHETIC ref']
    rows = []
    for name, m in metrics_per_panel.items():
        row = (
            f"{name:<28}  {m['background_mean']:>9.4f}  "
            f"{m['background_std']:>8.4f}  {m['object_bg_contrast']:>9.4f}  "
            f"{m['edge_transition_width']:>8.4f}"
        )
        if name not in ('SYNTHETIC ref', 'PREPROCESSING ref'):
            d_bg = m['background_mean'] - synth_m['background_mean']
            d_bs = m['background_std'] - synth_m['background_std']
            d_co = m['object_bg_contrast'] - synth_m['object_bg_contrast']
            d_ew = m['edge_transition_width'] - synth_m['edge_transition_width']
            row += (f"   |dSynth: bg={d_bg:+.3f} std={d_bs:+.3f} "
                    f"con={d_co:+.2f} ew={d_ew:+.2f}|")
        print(row)
        rows.append((name, m))

    print("\n" + "=" * 88)
    print("DECISION GATE — pass thresholds vs SYNTHETIC ref")
    print("=" * 88)
    print(f"  background_mean        abs < 0.05")
    print(f"  background_std         abs < 0.02")
    print(f"  object_bg_contrast     abs < 5")
    print(f"  edge_transition_width  abs < 1.0 px")
    print()
    for name, m in rows:
        if name in ('SYNTHETIC ref', 'PREPROCESSING ref'):
            continue
        d_bg = abs(m['background_mean'] - synth_m['background_mean'])
        d_bs = abs(m['background_std'] - synth_m['background_std'])
        d_co = abs(m['object_bg_contrast'] - synth_m['object_bg_contrast'])
        d_ew = abs(m['edge_transition_width'] - synth_m['edge_transition_width'])
        passes = []
        passes.append('bg' if d_bg < 0.05 else 'bg!')
        passes.append('std' if d_bs < 0.02 else 'std!')
        passes.append('con' if d_co < 5 else 'con!')
        passes.append('ew' if d_ew < 1 else 'ew!')
        status = "PASS" if all('!' not in p for p in passes) else "FAIL"
        print(f"  {name:<28}  {status}  ({' '.join(passes)})")

    # 9. Save the grid PNG
    panel_names = list(panels.keys())
    n = len(panel_names)
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3.3))
    axs = np.atleast_2d(axs)
    for i, name in enumerate(panel_names):
        r, c = i // cols, i % cols
        ax = axs[r, c]
        ax.imshow(panels[name], cmap='gray', vmin=0, vmax=255)
        m = metrics_per_panel[name]
        title = (
            f"{name}\n"
            f"bg={m['background_mean']:.3f}  std={m['background_std']:.3f}\n"
            f"con={m['object_bg_contrast']:.1f}  ew={m['edge_transition_width']:.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    # Hide unused
    for j in range(len(panel_names), rows_n * cols):
        r, c = j // cols, j % cols
        axs[r, c].axis('off')
    fig.suptitle(
        f"Calibration frame {chosen['filename']} (z={chosen_z:+.2f}mm, "
        f"sigma={chosen_sigma:.2f}px) processed N ways, vs synthetic + preprocessing",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = OUTPUT_DIR / "grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {grid_path}")
    print(f"       (open it: start \"\" \"{grid_path}\")")


if __name__ == '__main__':
    main()
