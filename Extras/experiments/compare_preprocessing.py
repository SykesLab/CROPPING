"""
Compare preprocessing methods for inference on calibration z-stack.

Methods:
  A: Simple flatten (Canny, interior=0, exterior=1, 3px feather) — training-style
  B: Calibration flatten (Otsu, 20px margin, exterior untouched)
  C: No flatten (just crop)
  D: Otsu + full exterior flatten (hybrid: robust detection + training-style output)
  E: 299x299 centre crop then simple flatten (matches training resolution)
  F-I: Otsu + exterior flatten with 30/40/50/60 px exterior feather

Loads 7mm cine z-stack, applies each method, runs model inference, compares to ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'calibration'))

import numpy as np
import cv2
import pandas as pd
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cine_loader import CineLoader
from sphere_processing import (process_sphere_stack, find_consensus_sphere,
                                crop_to_square, flatten_sphere_crop,
                                _detect_sphere_contour_otsu)
from model import DefocusNet
from inference_real_crops import RealCropInference

# ── CONFIG ──────────────────────────────────────────────────────────────────
CINE_FOLDER = r"C:\Users\justi\Downloads\coursework\coursework\random\7mm-20260316T021631Z-3-001\7mm"
POSITIONS_CSV = os.path.join(CINE_FOLDER, "positions.csv")
MODEL_PATH = r"C:\Users\justi\Downloads\coursework\coursework\training\Training\training_output\checkpoints\dme_best.pth"
OUTPUT_DIR = r"C:\Users\justi\Downloads\coursework\coursework\training\Training\inference_results\preprocessing_comparison_v2"

FOCAL_PLANE_MM = 8.0
FRAME_IDX = 0
SPHERE_DIAMETER_MM = 7.0


def load_zstack():
    """Load all frames from cine folder and compute defocus positions."""
    positions_df = pd.read_csv(POSITIONS_CSV)

    images = []
    filenames = []
    stage_positions = []

    for _, row in positions_df.iterrows():
        cine_path = os.path.join(CINE_FOLDER, row['filename'])
        if not os.path.exists(cine_path):
            continue

        loader = CineLoader(cine_path)
        frame = loader.extract_frame(FRAME_IDX)
        if frame is None:
            continue

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        images.append(frame)
        filenames.append(row['filename'])
        stage_positions.append(float(row['stage_position_mm']))

    defocus = [pos - FOCAL_PLANE_MM for pos in stage_positions]

    print(f"Loaded {len(images)} frames, image size: {images[0].shape}")
    print(f"Defocus range: {min(defocus):.1f} to {max(defocus):.1f} mm")

    return images, filenames, defocus


def detect_sphere(images):
    """Detect consensus sphere across all frames."""
    sphere = find_consensus_sphere(images, upper_only=True)
    if sphere is None:
        raise RuntimeError("No sphere detected")
    cx, cy, radius = sphere
    scale = (radius * 2) / SPHERE_DIAMETER_MM
    print(f"Sphere: centre=({cx},{cy}), r={radius}, scale={scale:.1f} px/mm")
    return cx, cy, radius, scale


def apply_custom_flatten(img, cx, cy, radius, inner_margin=0, flatten_exterior=True,
                         feather=3, use_otsu=True):
    """Apply flattening with specific parameters, using Otsu detection."""
    # Crop to square first (same as process_sphere_stack pipeline)
    cropped = crop_to_square(img, cx, cy, radius, padding=1.2)

    if cropped.dtype == np.uint8:
        proc_f = cropped.astype(np.float32) / 255.0
    else:
        proc_f = cropped.astype(np.float32)
        if proc_f.max() > 1.0:
            proc_f /= proc_f.max()

    # Force Otsu detection by using inner_margin > 0 trick, or call directly
    if use_otsu:
        # Otsu detection needs inner_margin > 0 in the current code
        # Use at least 1 to trigger Otsu, then override with actual params
        flat, info = flatten_sphere_crop(proc_f, inner_margin=max(inner_margin, 1),
                                          flatten_exterior=flatten_exterior, feather=feather)
        # If we wanted inner_margin=0 but Otsu detection, we need to re-flatten
        # with the detected contour. But inner_margin=0 + Otsu isn't directly supported.
        # For inner_margin=0 cases, the 1px margin is negligible.
    else:
        flat, info = flatten_sphere_crop(proc_f, inner_margin=inner_margin,
                                          flatten_exterior=flatten_exterior, feather=feather)

    if info is not None:
        result = flat
    else:
        result = proc_f

    # Convert back to uint8
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def apply_299_crop(img, cx, cy, radius):
    """Crop a 299x299 region centred on the sphere, then simple-flatten."""
    h, w = img.shape[:2]
    half = 149  # 299 // 2

    # Centre crop on sphere centre
    y1 = max(0, cy - half)
    y2 = y1 + 299
    if y2 > h:
        y2 = h
        y1 = y2 - 299
    x1 = max(0, cx - half)
    x2 = x1 + 299
    if x2 > w:
        x2 = w
        x1 = x2 - 299

    cropped = img[y1:y2, x1:x2].copy()

    # Simple flatten on the 299x299 crop (Canny detection, full flatten)
    if cropped.dtype == np.uint8:
        proc_f = cropped.astype(np.float32) / 255.0
    else:
        proc_f = cropped.astype(np.float32)

    flat, info = flatten_sphere_crop(proc_f, inner_margin=0, flatten_exterior=True, feather=3)
    if info is not None:
        result = flat
    else:
        result = proc_f

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def preprocess_all_methods(images, cx, cy, radius):
    """Apply all preprocessing methods and return dict of {method_name: [processed_images]}."""
    methods = {}

    # A: Simple flatten (Canny, full flatten) — via process_sphere_stack
    print("  A: Simple flatten (Canny)...")
    proc_a, _ = process_sphere_stack(images, upper_only=True, blacken=False,
                                      flatten=True, flatten_mode="simple")
    methods['A_simple_canny'] = proc_a

    # B: Calibration flatten (Otsu, 20px margin, no exterior)
    print("  B: Calibration flatten (Otsu, 20px margin)...")
    proc_b, _ = process_sphere_stack(images, upper_only=True, blacken=False,
                                      flatten=True, flatten_mode="default")
    methods['B_calib_otsu_20px'] = proc_b

    # C: No flatten
    print("  C: No flatten...")
    proc_c, _ = process_sphere_stack(images, upper_only=True, blacken=False,
                                      flatten=False)
    methods['C_no_flatten'] = proc_c

    # D: Otsu + full exterior flatten (hybrid)
    print("  D: Otsu + full exterior flatten...")
    methods['D_otsu_full_flatten'] = [
        apply_custom_flatten(img, cx, cy, radius, inner_margin=0,
                             flatten_exterior=True, feather=3, use_otsu=True)
        for img in images
    ]

    # E: 299x299 centre crop + simple flatten
    print("  E: 299x299 crop + simple flatten...")
    methods['E_299crop_flatten'] = [
        apply_299_crop(img, cx, cy, radius)
        for img in images
    ]

    # F-I: Otsu + exterior flatten with varying exterior feather widths
    for ext_feather in [30, 40, 50, 60]:
        name = f'F_otsu_ext{ext_feather}px'
        print(f"  {name}...")
        methods[name] = [
            apply_custom_flatten(img, cx, cy, radius, inner_margin=0,
                                 flatten_exterior=True, feather=ext_feather, use_otsu=True)
            for img in images
        ]

    return methods


def run_inference_batch(processed_images, filenames, defocus_values, method_name, output_dir):
    """Run model inference on preprocessed images."""
    out_dir = Path(output_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Save crops
    for i, (img, fname, z) in enumerate(zip(processed_images, filenames, defocus_values)):
        stem = Path(fname).stem
        out_name = f"{stem}_z{z:+.2f}mm.png"
        img_save = img if img.dtype == np.uint8 else cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(crops_dir / out_name), img_save)

    # Load model (reuse across methods by passing it in would be better,
    # but for simplicity we load once per method)
    inference = RealCropInference(model_path=MODEL_PATH, device='cuda')

    results = []
    for img, fname, z_true in zip(processed_images, filenames, defocus_values):
        if img.dtype == np.uint8:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
            if gray.max() > 1.0:
                gray /= gray.max()

        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (inference.model_size, inference.model_size),
                              interpolation=cv2.INTER_AREA)
        img_norm = resized * 2.0 - 1.0
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float().to(inference.device)

        with torch.no_grad():
            pred_norm = inference.model(tensor)

        sigma_model = inference.denormalize_blur(pred_norm.squeeze())
        native_size = max(h, w)
        sigma_native = sigma_model * native_size / inference.model_size
        sigma_0 = inference.direct_offset if inference.direct_offset else 0.0
        z_pred = max(0.0, (sigma_native - sigma_0) / inference.direct_slope)

        results.append({
            'filename': fname,
            'z_true_mm': abs(z_true),
            'z_pred_mm': z_pred,
            'z_error_mm': z_pred - abs(z_true),
            'sigma_model_px': sigma_model,
            'sigma_native_px': sigma_native,
            'native_size': native_size,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'results.csv', index=False)

    # Compute stats
    df_v = df[df['z_true_mm'] > 0.01].copy()
    errors = df_v['z_error_mm'].abs()

    stats = {
        'method': method_name,
        'n': len(df_v),
        'mae_mm': errors.mean(),
        'rmse_mm': np.sqrt((errors**2).mean()),
        'median_ae_mm': errors.median(),
        'max_error_mm': errors.max(),
        'bias_mm': df_v['z_error_mm'].mean(),
    }

    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8)]:
        mask = (df_v['z_true_mm'] >= lo) & (df_v['z_true_mm'] < hi)
        if mask.sum() > 0:
            stats[f'mae_{lo}_{hi}mm'] = errors[mask].mean()
            stats[f'bias_{lo}_{hi}mm'] = df_v.loc[mask, 'z_error_mm'].mean()

    return df, stats


def plot_comparison(all_results, all_stats, output_dir):
    """Create comparison plots."""
    n_methods = len(all_results)
    cmap = plt.cm.tab10
    colors = {name: cmap(i / max(n_methods - 1, 1)) for i, name in enumerate(all_results.keys())}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predicted vs True
    ax = axes[0, 0]
    for method, df in all_results.items():
        df_v = df[df['z_true_mm'] > 0.01]
        ax.scatter(df_v['z_true_mm'], df_v['z_pred_mm'], s=10, alpha=0.5,
                   color=colors[method], label=method)
    ax.plot([0, 8], [0, 8], 'k--', alpha=0.5)
    ax.set_xlabel('True |z| (mm)')
    ax.set_ylabel('Predicted |z| (mm)')
    ax.set_title('Predicted vs True Defocus')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(alpha=0.2)

    # 2. Signed error vs True z
    ax = axes[0, 1]
    for method, df in all_results.items():
        df_v = df[df['z_true_mm'] > 0.01]
        ax.scatter(df_v['z_true_mm'], df_v['z_error_mm'], s=10, alpha=0.5,
                   color=colors[method], label=method)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('True |z| (mm)')
    ax.set_ylabel('Error (pred - true) (mm)')
    ax.set_title('Signed Error vs Defocus')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.2)

    # 3. Overall MAE bar chart
    ax = axes[1, 0]
    names = list(all_stats.keys())
    maes = [all_stats[n]['mae_mm'] for n in names]
    bar_colors = [colors[n] for n in names]
    bars = ax.bar(range(len(names)), maes, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE (mm)')
    ax.set_title('Overall MAE by Method')
    ax.grid(alpha=0.2, axis='y')
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mae:.2f}', ha='center', fontsize=7)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for name in names:
        s = all_stats[name]
        table_data.append([
            name,
            f"{s['mae_mm']:.3f}",
            f"{s['bias_mm']:+.3f}",
            f"{s.get('mae_0_2mm', 0):.3f}",
            f"{s.get('mae_2_4mm', 0):.3f}",
            f"{s.get('mae_4_6mm', 0):.3f}",
            f"{s.get('mae_6_8mm', 0):.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'MAE', 'Bias', '0-2mm', '2-4mm', '4-6mm', '6-8mm'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.1, 1.4)
    ax.set_title('Summary — MAE by range (mm)', fontweight='bold', pad=20)

    plt.suptitle('Preprocessing Comparison — 7mm Calibration Z-Stack (v2)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comparison_v2.png', dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/comparison_v2.png")


def main():
    print("=" * 70)
    print("PREPROCESSING COMPARISON v2 — Extended Methods")
    print("=" * 70)

    output_base = Path(OUTPUT_DIR)
    output_base.mkdir(parents=True, exist_ok=True)

    # Load z-stack
    print("\n1. Loading z-stack...")
    images, filenames, defocus = load_zstack()

    # Detect sphere (consensus across all frames)
    print("\n2. Detecting sphere...")
    cx, cy, radius, scale = detect_sphere(images)

    # Apply all preprocessing methods
    print("\n3. Applying preprocessing methods...")
    all_processed = preprocess_all_methods(images, cx, cy, radius)

    # Run inference for each method
    print("\n4. Running inference...")
    all_results = {}
    all_stats = {}

    for method_name, processed in all_processed.items():
        print(f"\n  --- {method_name} ---")
        method_output = output_base / method_name
        df, stats = run_inference_batch(processed, filenames, defocus, method_name, method_output)
        all_results[method_name] = df
        all_stats[method_name] = stats
        print(f"    MAE={stats['mae_mm']:.3f}mm  Bias={stats['bias_mm']:+.3f}mm  "
              f"Near={stats.get('mae_0_2mm',0):.3f}mm  Far={stats.get('mae_6_8mm',0):.3f}mm")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'MAE':>7} {'Bias':>7} {'0-2mm':>7} {'2-4mm':>7} {'4-6mm':>7} {'6-8mm':>7}")
    print("-" * 80)
    for name, s in all_stats.items():
        print(f"{name:<25} {s['mae_mm']:>7.3f} {s['bias_mm']:>+7.3f} "
              f"{s.get('mae_0_2mm',0):>7.3f} {s.get('mae_2_4mm',0):>7.3f} "
              f"{s.get('mae_4_6mm',0):>7.3f} {s.get('mae_6_8mm',0):>7.3f}")

    # Plot
    print("\n5. Creating plots...")
    plot_comparison(all_results, all_stats, output_base)

    # Save summary
    summary_df = pd.DataFrame(all_stats.values())
    summary_df.to_csv(output_base / 'summary_v2.csv', index=False)
    print(f"\nAll results saved to {output_base}")


if __name__ == "__main__":
    main()
