"""
Standalone script to flatten existing crops and reclassify focus using ERF sigma.

Reads all crops from OUTPUTNEW/, flattens each, measures ERF sigma, and builds
a new Focus/ directory with ERF-based focus classification.

Writes to a separate output directory (non-destructive by default).

Usage:
    python flatten_existing_crops.py
    python flatten_existing_crops.py --input OUTPUTNEW --output OUTPUTNEW_flattened
"""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Import from calibration module
try:
    from Calibration.sphere_processing import flatten_sphere_crop
except ImportError:
    _CALIB_DIR = str(Path(__file__).resolve().parent.parent / 'Calibration')
    if _CALIB_DIR not in sys.path:
        sys.path.insert(0, _CALIB_DIR)
    from sphere_processing import flatten_sphere_crop
from crop_blur_measurement import measure_erf_blur
from focus_classification import classify_by_erf_sigma

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def flatten_and_measure_crop(
    src_path: Path,
    dst_path: Path,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Flatten a single crop and measure its ERF sigma.

    Returns:
        (success, erf_sigma_or_None, error_message_or_None)
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, None, f"could not read: {src_path}"

    img_f = img.astype(np.float32) / 255.0
    flat, info = flatten_sphere_crop(img_f)

    if info is None:
        # Flatten failed — save original unchanged
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), img)
        # Still try to measure ERF on original
        sigma = measure_erf_blur(img)
        return False, sigma, f"sphere detection failed: {src_path.name}"

    # Save flattened crop
    out_img = (flat * 255).clip(0, 255).astype(np.uint8)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), out_img)

    # Measure ERF on flattened crop
    sigma = measure_erf_blur(out_img)

    return True, sigma, None


def main():
    parser = argparse.ArgumentParser(
        description="Flatten existing crops and reclassify focus using ERF sigma"
    )
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).parent / 'OUTPUTNEW',
                        help='Input directory containing preprocessing output')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'OUTPUTNEW_flattened',
                        help='Output directory (non-destructive)')
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output

    if not input_root.exists():
        print(f"ERROR: Input directory not found: {input_root}")
        return

    print("=" * 70)
    print("FLATTEN EXISTING CROPS + ERF RECLASSIFICATION")
    print("=" * 70)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print()

    output_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: Find all crops (excluding Focus/ directory)
    all_crops = []
    for crop_path in sorted(input_root.rglob('*_crop.png')):
        # Skip crops inside Focus/ directory
        rel = crop_path.relative_to(input_root)
        if rel.parts[0] == 'Focus':
            continue
        all_crops.append(crop_path)

    print(f"Found {len(all_crops)} crops to process (excluding Focus/)")
    print()

    # Phase 2: Flatten each crop and measure ERF sigma
    n_ok = 0
    n_flatten_fail = 0
    n_erf_fail = 0
    failures: List[str] = []
    crop_data: List[dict] = []

    t0 = time.time()
    for i, crop_path in enumerate(all_crops):
        rel = crop_path.relative_to(input_root)
        dst_path = output_root / rel

        success, sigma, error = flatten_and_measure_crop(crop_path, dst_path)

        if not success:
            n_flatten_fail += 1
            failures.append(f"FLATTEN_FAIL: {error}")

        if sigma is None:
            n_erf_fail += 1

        # Parse folder/camera from path: <material>/<cam>/crops/<filename>
        parts = rel.parts
        folder = parts[0] if len(parts) >= 1 else "unknown"
        camera = parts[1] if len(parts) >= 2 else "unknown"

        crop_data.append({
            'filename': crop_path.name,
            'folder': folder,
            'camera': camera,
            'crop_path': str(dst_path),
            'erf_sigma': sigma if sigma is not None else float('nan'),
            'flatten_ok': success,
        })

        if success:
            n_ok += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(all_crops):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(all_crops)}] {rate:.1f} crops/sec | "
                  f"OK: {n_ok} | Flatten fail: {n_flatten_fail} | ERF fail: {n_erf_fail}")

    elapsed = time.time() - t0
    print(f"\nPhase 2 complete: {elapsed:.1f}s ({len(all_crops)/elapsed:.1f} crops/sec)")
    print(f"  Flattened: {n_ok} | Flatten failed: {n_flatten_fail} | ERF failed: {n_erf_fail}")
    print()

    # Phase 2.5: Merge with existing metadata from summary CSVs
    # This preserves scale_px_per_mm, diameter_px, camera_model, etc.
    print("Merging with existing metadata from summary CSVs...")
    df = pd.DataFrame(crop_data)

    existing_meta = []
    for csv_path in sorted(input_root.rglob("*_summary.csv")):
        if "Focus" in csv_path.parts:
            continue
        try:
            meta_df = pd.read_csv(csv_path)
            if 'crop_path' in meta_df.columns:
                # Extract filename from crop_path for matching
                meta_df['_match_filename'] = meta_df['crop_path'].apply(
                    lambda p: Path(str(p)).name if pd.notna(p) else "")
                existing_meta.append(meta_df)
        except Exception as e:
            logger.warning(f"Failed to load metadata CSV: {e}")

    if existing_meta:
        meta_combined = pd.concat(existing_meta, ignore_index=True)
        # Merge on filename — carry over columns that the training GUI needs
        merge_cols = [c for c in meta_combined.columns
                      if c not in ('crop_path', 'focus_class', 'erf_sigma', '_match_filename',
                                   'folder', 'camera', 'filename')]
        merge_cols.append('_match_filename')
        meta_subset = meta_combined[merge_cols].drop_duplicates(subset='_match_filename')
        df = df.merge(meta_subset, left_on='filename', right_on='_match_filename', how='left')
        if '_match_filename' in df.columns:
            df.drop(columns=['_match_filename'], inplace=True)
        print(f"  Merged {len(merge_cols)-1} metadata columns from {len(existing_meta)} summary CSVs")

    # Phase 3: ERF-based focus classification
    print("Phase 3: ERF-based focus classification...")

    focus_dir = output_root / "Focus"
    focus_dir.mkdir(parents=True, exist_ok=True)

    folder_stats: List[dict] = []
    total_sharp_copied = 0

    for folder in sorted(df['folder'].unique()):
        folder_df = df[df['folder'] == folder]

        for cam in sorted(folder_df['camera'].unique()):
            cam_mask = (df['folder'] == folder) & (df['camera'] == cam)
            cam_df = df[cam_mask]
            cam_erf = cam_df['erf_sigma'].values

            valid_erf = cam_erf[~np.isnan(cam_erf)]
            if len(valid_erf) < 4:
                df.loc[cam_mask, 'focus_class'] = None
                continue

            classifications, sharp_thresh, blur_thresh = classify_by_erf_sigma(cam_erf)
            df.loc[cam_mask, 'focus_class'] = classifications

            cam_sharp = (df.loc[cam_mask, 'focus_class'] == 'sharp').sum()
            cam_medium = (df.loc[cam_mask, 'focus_class'] == 'medium').sum()
            cam_blurry = (df.loc[cam_mask, 'focus_class'] == 'blurry').sum()

            folder_stats.append({
                'folder': folder,
                'camera': cam,
                'n_total': len(cam_df),
                'n_measured': int(np.sum(~np.isnan(cam_erf))),
                'n_sharp': cam_sharp,
                'n_medium': cam_medium,
                'n_blurry': cam_blurry,
                'erf_sharp_thresh': sharp_thresh,
                'erf_blur_thresh': blur_thresh,
                'erf_sigma_mean': float(np.nanmean(cam_erf)),
            })

            # Copy sharp crops to Focus/<folder>/<cam>/
            sharp_rows = df[cam_mask & (df['focus_class'] == 'sharp')]
            if len(sharp_rows) > 0:
                cam_focus_dir = focus_dir / folder / cam
                cam_focus_dir.mkdir(parents=True, exist_ok=True)

                for _, row in sharp_rows.iterrows():
                    src = Path(row['crop_path'])
                    if src.exists():
                        shutil.copy2(src, cam_focus_dir / src.name)
                        total_sharp_copied += 1

            print(
                f"  {folder}/{cam}: {cam_sharp} sharp / {cam_medium} medium / {cam_blurry} blurry "
                f"(thresh: sharp<{sharp_thresh:.2f}, blur>{blur_thresh:.2f})")

    # Save classified CSV
    all_path = focus_dir / "focus_classified_all.csv"
    df.to_csv(all_path, index=False)

    # Save sharp-only CSV
    sharp_df = df[df['focus_class'] == 'sharp'].copy()
    sharp_df['native_blur_sigma'] = sharp_df['erf_sigma']
    # diameter_px from merged metadata, or compute from y_top/y_bottom if available
    if 'diameter_px' not in sharp_df.columns:
        if 'y_top' in sharp_df.columns and 'y_bottom' in sharp_df.columns:
            sharp_df['diameter_px'] = sharp_df['y_bottom'] - sharp_df['y_top']
        else:
            sharp_df['diameter_px'] = ''
    sharp_path = focus_dir / "sharp_crops.csv"
    sharp_df.to_csv(sharp_path, index=False)

    # Save folder stats
    stats_df = pd.DataFrame(folder_stats)
    stats_path = focus_dir / "focus_folder_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # Phase 4: Write provenance metadata
    config = {
        "flattening_method": "contour_distmap_simple",
        "margin": "none (edge boundary = contour)",
        "feather": "3px",
        "focus_ranking_metric": "erf_sigma",
        "focus_percentile_sharp": 25,
        "focus_percentile_blurry": 75,
        "input_directory": str(input_root),
        "output_directory": str(output_root),
        "total_crops": len(all_crops),
        "flattened_ok": n_ok,
        "flatten_failures": n_flatten_fail,
        "erf_failures": n_erf_fail,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = output_root / "flattening_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Phase 5: Write failure log
    if failures:
        log_path = output_root / "flatten_failures.log"
        with open(log_path, 'w') as f:
            f.write(f"Flatten failures: {n_flatten_fail}\n")
            f.write(f"ERF failures: {n_erf_fail}\n")
            f.write("=" * 50 + "\n")
            for entry in failures:
                f.write(f"{entry}\n")
        print(f"\nFailure log: {log_path}")

    # Summary
    total_sharp = (df['focus_class'] == 'sharp').sum()
    total_medium = (df['focus_class'] == 'medium').sum()
    total_blurry = (df['focus_class'] == 'blurry').sum()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total crops:      {len(all_crops)}")
    print(f"  Flattened OK:     {n_ok}")
    print(f"  Flatten failed:   {n_flatten_fail}")
    print(f"  ERF failed:       {n_erf_fail}")
    print(f"  Sharp:            {total_sharp}")
    print(f"  Medium:           {total_medium}")
    print(f"  Blurry:           {total_blurry}")
    print(f"  Sharp copied:     {total_sharp_copied}")
    print()
    print(f"  Output:           {output_root}")
    print(f"  Focus dir:        {focus_dir}")
    print(f"  Config:           {config_path}")
    print(f"  Sharp crops CSV:  {sharp_path}")
    print()
    print("Done.")


if __name__ == '__main__':
    main()
