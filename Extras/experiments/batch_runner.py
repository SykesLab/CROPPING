"""
Batch Calibration Runner — Headless orchestration over the existing calibration pipeline.

Processes raw sphere calibration z-stacks (7mm, 9mm, 10mm) through the same
methodology as the calibration GUI, outputting structured results per sphere.

Usage:
    python batch_runner.py
"""

import matplotlib
matplotlib.use('Agg')

import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cine_loader import CineFolderLoader, PYPHANTOM_AVAILABLE
from blur_measurement import measure_blur_erf, detect_sphere, BlurMeasurement
from calibration_core import calibrate_approach_a, CalibrationResultA

from paths_config import BATCH_INPUT_DIR, BATCH_OUTPUT_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_DIR = BATCH_INPUT_DIR
OUTPUT_DIR = BATCH_OUTPUT_DIR

# Common crop size — all spheres cropped to this square, ensuring identical
# resize ratio when downscaled to 128×128 for training.
# 960 is the max square from 1280×960 raw frames.
COMMON_CROP_SIZE = 960

# Known sphere diameters (mm)
SPHERE_DIAMETERS = {"7mm": 7.0, "9mm": 9.0, "10mm": 10.0}


# ── Helper: focus detection ──────────────────────────────────────────────────

def find_focus_frame(images: List[np.ndarray]) -> Tuple[int, List[float]]:
    """
    Find sharpest frame via Laplacian variance — same logic as calibration GUI
    (calibration_gui.py lines 969-978).

    Returns:
        (focus_idx, sharpness_values)
    """
    sharpness_values = []
    for img in images:
        if img.dtype != np.uint8:
            img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_u8 = img
        lap = cv2.Laplacian(img_u8, cv2.CV_64F)
        mask = img_u8 > 5
        sharpness = lap[mask].var() if mask.any() else 0.0
        sharpness_values.append(sharpness)

    focus_idx = int(np.argmax(sharpness_values))
    return focus_idx, sharpness_values


# ── Helper: px/mm ────────────────────────────────────────────────────────────

def compute_px_per_mm(
    image: np.ndarray, sphere_diameter_mm: float
) -> Tuple[float, Optional[Tuple[int, int]], Optional[int]]:
    """
    Compute scale from the raw sharpest frame — same as GUI line 1640-1647.
    Uses detect_sphere (Otsu-based) from blur_measurement.py.

    Returns:
        (px_per_mm, center, radius) or (nan, None, None) on failure
    """
    center, radius = detect_sphere(image)
    if radius is None or radius <= 0:
        return float('nan'), None, None
    px_per_mm = (2 * radius) / sphere_diameter_mm
    return px_per_mm, center, radius


# ── Helper: fixed-size crop ─────────────────────────────────────────────────

def crop_fixed_square(
    img: np.ndarray, cx: int, cy: int, size: int = COMMON_CROP_SIZE
) -> np.ndarray:
    """Crop a fixed-size square centered on (cx, cy), clamped to frame bounds."""
    h, w = img.shape[:2]
    half = size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + size
    y2 = y1 + size

    # Shift window if it extends past frame edges
    if x1 < 0:
        x1, x2 = 0, size
    if y1 < 0:
        y1, y2 = 0, size
    if x2 > w:
        x1, x2 = w - size, w
    if y2 > h:
        y1, y2 = h - size, h

    crop = img[y1:y2, x1:x2]
    if crop.dtype != np.uint8:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return crop


# ── Helper: save calibration plot ────────────────────────────────────────────

def save_calibration_plot(
    path: Path,
    defocus_z: np.ndarray,
    sigmas: np.ndarray,
    cal_result: CalibrationResultA,
    sphere_label: str,
):
    """Save calibration curve: measured sigma vs defocus z with fit line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Measured data
    valid = sigmas > 0
    failed = sigmas <= 0
    ax.scatter(defocus_z[valid], sigmas[valid], c='tab:blue', s=20, label='Measured')
    if failed.any():
        ax.scatter(defocus_z[failed], sigmas[failed], c='gray', s=15,
                   marker='x', label='Failed', alpha=0.5)

    # Fit line
    z_line = np.linspace(defocus_z.min(), defocus_z.max(), 200)
    sigma_line = cal_result.rho_px_per_mm * np.abs(z_line) + cal_result.sigma_0
    ax.plot(z_line, sigma_line, 'r-', linewidth=1.5, label=(
        f'Fit: σ = {cal_result.rho_px_per_mm:.3f}|z| + {cal_result.sigma_0:.3f}'
    ))

    ax.set_xlabel('Defocus z (mm)')
    ax.set_ylabel('Blur σ (px)')
    ax.set_title(f'{sphere_label} — Direct Calibration  (R² = {cal_result.r_squared:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ── Helper: save per-frame CSV ───────────────────────────────────────────────

def save_per_frame_csv(
    path: Path,
    filenames: List[str],
    mechanical_positions: List[float],
    defocus_z: np.ndarray,
    focus_idx: int,
    sphere_diameter_mm: float,
    px_per_mm: float,
    original_resolution: Tuple[int, int],
    processed_resolution: Tuple[int, int],
    sigmas: np.ndarray,
    measurements: List[BlurMeasurement],
):
    """Write per-frame measurements CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'source_filename',
            'mechanical_position_mm',
            'defocus_z_mm',
            'is_focus_frame',
            'sphere_diameter_mm',
            'px_per_mm',
            'original_resolution',
            'processed_resolution',
            'sigma_px',
            'fit_confidence',
            'rays_accepted',
        ])
        for i in range(len(filenames)):
            m = measurements[i]
            rays = m.details.get('num_rays_used', 0) if m.details else 0
            writer.writerow([
                filenames[i],
                f'{mechanical_positions[i]:.2f}',
                f'{defocus_z[i]:.2f}',
                i == focus_idx,
                f'{sphere_diameter_mm:.1f}',
                f'{px_per_mm:.2f}',
                f'{original_resolution[1]}x{original_resolution[0]}',
                f'{processed_resolution[1]}x{processed_resolution[0]}',
                f'{sigmas[i]:.4f}' if sigmas[i] > 0 else '0',
                f'{m.confidence:.4f}' if m.confidence > 0 else '0',
                rays,
            ])


# ── Helper: save calibration YAML ────────────────────────────────────────────

def save_calibration_yaml(
    path: Path,
    sphere_diameter_mm: float,
    focus_idx: int,
    focus_filename: str,
    focus_position_mm: float,
    px_per_mm: float,
    cal_result: CalibrationResultA,
    num_frames: int,
    num_valid: int,
    z_range: Tuple[float, float],
    source_folder: str,
):
    """Write calibration summary YAML."""
    data = {
        'sphere_diameter_mm': sphere_diameter_mm,
        'focus_frame_index': focus_idx,
        'focus_frame_filename': focus_filename,
        'focus_mechanical_position_mm': float(round(focus_position_mm, 2)),
        'px_per_mm': float(round(px_per_mm, 2)),
        'rho_px_per_mm': float(round(cal_result.rho_px_per_mm, 4)),
        'sigma_0_px': float(round(cal_result.sigma_0, 4)),
        'r_squared': float(round(cal_result.r_squared, 4)),
        'num_frames': int(num_frames),
        'num_valid_frames': int(num_valid),
        'z_range_mm': [float(round(z_range[0], 2)), float(round(z_range[1], 2))],
        'exclude_near_focus_mm': 0.5,
        'preprocessing': {
            'crop_size': COMMON_CROP_SIZE,
            'blacken': False,
            'mirror': False,
        },
        'source_folder': source_folder,
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ── Main per-sphere pipeline ─────────────────────────────────────────────────

def process_single_sphere(
    folder_path: Path, sphere_diameter_mm: float, output_dir: Path
) -> Optional[Dict]:
    """
    Full calibration pipeline for one sphere z-stack.
    Mirrors the GUI workflow exactly.
    """
    label = folder_path.name
    print(f"\n{'='*60}")
    print(f"[{label}] Processing sphere (diameter = {sphere_diameter_mm} mm)")
    print(f"{'='*60}")

    # ── Step 1: Load .cine files + positions.csv ──
    csv_path = folder_path / 'positions.csv'
    if not csv_path.exists():
        print(f"[{label}] ERROR: positions.csv not found in {folder_path}")
        return None

    loader = CineFolderLoader(str(folder_path))
    print(f"[{label}] Loading {loader.num_files} .cine files from {folder_path}")

    images, positions, filenames = loader.load_with_positions_csv(
        csv_path=str(csv_path),
        stage_offset=0.0,
        frame_idx=0,
    )

    if not images:
        print(f"[{label}] ERROR: No images loaded")
        return None

    print(f"[{label}] Loaded {len(images)} frames, positions {min(positions):.1f} to {max(positions):.1f} mm")
    original_resolution = images[0].shape[:2]  # (h, w)

    # ── Step 2: Find focus frame (Laplacian variance on RAW images) ──
    focus_idx, sharpness_values = find_focus_frame(images)
    focus_position = positions[focus_idx]
    print(f"[{label}] Focus frame: #{focus_idx + 1} ({filenames[focus_idx]}) "
          f"at stage {focus_position:.2f} mm "
          f"(sharpness = {sharpness_values[focus_idx]:.1f})")

    # ── Step 3: Compute defocus positions ──
    defocus_z = np.array(positions) - focus_position
    print(f"[{label}] Defocus range: {defocus_z.min():.2f} to {defocus_z.max():.2f} mm")

    # ── Step 4: Compute px/mm from sharpest RAW frame ──
    px_per_mm, scale_center, scale_radius = compute_px_per_mm(
        images[focus_idx], sphere_diameter_mm
    )
    if np.isnan(px_per_mm):
        print(f"[{label}] WARNING: detect_sphere failed on sharpest frame")
        print(f"[{label}] ERROR: Cannot determine px/mm or crop center, skipping this sphere")
        return None

    print(f"[{label}] px/mm: {px_per_mm:.2f} (sphere radius: {scale_radius} px)")

    # ── Step 5: Crop all frames to common fixed size ──
    crop_cx, crop_cy = scale_center
    print(f"[{label}] Cropping all frames to {COMMON_CROP_SIZE}x{COMMON_CROP_SIZE} "
          f"centered on ({crop_cx},{crop_cy})...")
    processed_images = [crop_fixed_square(img, crop_cx, crop_cy) for img in images]

    processed_resolution = processed_images[0].shape[:2]  # (h, w)
    print(f"[{label}] Processed image size: {processed_resolution[1]}x{processed_resolution[0]}")

    # ── Step 6: Save processed images ──
    img_dir = output_dir / 'processed_images'
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, proc_img in enumerate(processed_images):
        stem = Path(filenames[i]).stem
        cv2.imwrite(str(img_dir / f'{stem}.png'), proc_img)

    # ── Step 7: Measure ERF blur on each processed image ──
    print(f"[{label}] Measuring blur (ERF)...")
    sigmas = np.zeros(len(processed_images))
    measurements: List[BlurMeasurement] = []

    for i, proc_img in enumerate(processed_images):
        z = defocus_z[i]
        m = measure_blur_erf(proc_img, center=None, radius=None, num_rays=36, verbose=False)
        sigmas[i] = m.sigma
        measurements.append(m)

        rays_used = m.details.get('num_rays_used', 0) if m.details else 0

        if m.sigma > 0:
            print(f"[{label}]   [{i+1:2d}/{len(processed_images)}] "
                  f"z={z:+6.2f} mm → σ={m.sigma:6.3f} px "
                  f"(conf={m.confidence:.3f}, {rays_used}/{36} rays)")
        else:
            print(f"[{label}]   [{i+1:2d}/{len(processed_images)}] "
                  f"z={z:+6.2f} mm → FAILED "
                  f"({m.details.get('error', 'unknown')})")

    # ── Step 8: Filter and fit calibration ──
    valid_mask = sigmas > 0
    num_valid = int(valid_mask.sum())
    print(f"[{label}] Valid measurements: {num_valid}/{len(sigmas)}")

    if num_valid < 5:
        print(f"[{label}] ERROR: Only {num_valid} valid measurements, need at least 5. Skipping fit.")
        # Still save CSV
        save_per_frame_csv(
            output_dir / 'per_frame_measurements.csv',
            filenames, list(positions), defocus_z, focus_idx,
            sphere_diameter_mm, px_per_mm, original_resolution,
            processed_resolution, sigmas, measurements,
        )
        return None

    filtered_z = defocus_z[valid_mask].tolist()
    filtered_sigmas = sigmas[valid_mask].tolist()

    cal_result = calibrate_approach_a(filtered_z, filtered_sigmas, exclude_near_focus=0.5)

    print(f"[{label}] Fit: ρ = {cal_result.rho_px_per_mm:.4f} px/mm, "
          f"σ₀ = {cal_result.sigma_0:.4f} px, "
          f"R² = {cal_result.r_squared:.4f} "
          f"({num_valid}/{len(sigmas)} valid)")

    # ── Step 9: Save outputs ──
    save_per_frame_csv(
        output_dir / 'per_frame_measurements.csv',
        filenames, list(positions), defocus_z, focus_idx,
        sphere_diameter_mm, px_per_mm, original_resolution,
        processed_resolution, sigmas, measurements,
    )
    print(f"[{label}] Saved per_frame_measurements.csv")

    save_calibration_yaml(
        output_dir / 'calibration_summary.yaml',
        sphere_diameter_mm, focus_idx, filenames[focus_idx],
        focus_position, px_per_mm, cal_result,
        len(sigmas), num_valid,
        (float(defocus_z.min()), float(defocus_z.max())),
        str(folder_path),
    )
    print(f"[{label}] Saved calibration_summary.yaml")

    save_calibration_plot(
        output_dir / 'calibration_curve.png',
        defocus_z, sigmas, cal_result, label,
    )
    print(f"[{label}] Saved calibration_curve.png")

    print(f"[{label}] Outputs saved to {output_dir}")

    return {
        'sphere_diameter_mm': sphere_diameter_mm,
        'folder': label,
        'focus_frame': filenames[focus_idx],
        'focus_position_mm': float(round(focus_position, 2)),
        'px_per_mm': float(round(px_per_mm, 2)),
        'rho_px_per_mm': float(round(cal_result.rho_px_per_mm, 4)),
        'sigma_0_px': float(round(cal_result.sigma_0, 4)),
        'r_squared': float(round(cal_result.r_squared, 4)),
        'num_frames': int(len(sigmas)),
        'num_valid': int(num_valid),
    }


# ── Batch orchestrator ───────────────────────────────────────────────────────

def run_batch():
    """Process all sphere folders."""
    print(f"Batch Calibration Runner")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if not PYPHANTOM_AVAILABLE:
        print("ERROR: pyphantom is not available. Cannot load .cine files.")
        print("Install pyphantom (Phantom SDK) and try again.")
        sys.exit(1)

    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover sphere folders
    sphere_folders = {}
    for subfolder in sorted(INPUT_DIR.iterdir()):
        if subfolder.is_dir():
            match = re.match(r'(\d+)\s*mm', subfolder.name, re.IGNORECASE)
            if match:
                diameter = float(match.group(1))
                sphere_folders[subfolder] = diameter
                print(f"  Found: {subfolder.name} → {diameter} mm sphere")

    if not sphere_folders:
        print("ERROR: No sphere folders found (expected folders like 7mm, 9mm, 10mm)")
        sys.exit(1)

    print(f"\nProcessing {len(sphere_folders)} sphere(s)...")
    t_start = time.time()

    results = {}
    for folder_path, diameter in sphere_folders.items():
        sphere_output = OUTPUT_DIR / folder_path.name
        sphere_output.mkdir(parents=True, exist_ok=True)
        result = process_single_sphere(folder_path, diameter, sphere_output)
        if result is not None:
            results[folder_path.name] = result

    # Save batch summary
    elapsed = time.time() - t_start
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(INPUT_DIR),
        'output_dir': str(OUTPUT_DIR),
        'elapsed_seconds': round(elapsed, 1),
        'spheres': results,
    }
    summary_path = OUTPUT_DIR / 'batch_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Batch complete — {len(results)}/{len(sphere_folders)} spheres succeeded")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Summary: {summary_path}")

    if results:
        print(f"\n{'Sphere':<8} {'ρ (px/mm)':>10} {'σ₀ (px)':>10} {'R²':>8} {'px/mm':>8}")
        print('-' * 50)
        for name, r in results.items():
            print(f"{name:<8} {r['rho_px_per_mm']:>10.4f} {r['sigma_0_px']:>10.4f} "
                  f"{r['r_squared']:>8.4f} {r['px_per_mm']:>8.2f}")

    print()


if __name__ == '__main__':
    run_batch()
