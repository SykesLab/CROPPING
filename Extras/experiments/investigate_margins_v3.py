"""
Margin Investigation v3: ContourDistMap only, per-camera optimal margin.

Goal: Find the best margin for each camera type (v, g, m) independently.
Not pipeline default — what's actually optimal per camera.

Configs:
  Fixed: 40px, 50px (current), 60px
  Proportional: 14%, 20%, 25%

8 crops per camera for better statistics.

Run with: phantom_env\\Scripts\\python.exe investigate_margins_v3.py
"""

import sys
import csv
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

CALIB_DIR = str(Path(__file__).parent.parent.parent / 'calibration')
if CALIB_DIR not in sys.path:
    sys.path.insert(0, CALIB_DIR)

from blur_measurement import detect_sphere, measure_blur_erf
from sphere_processing import find_sphere_center

CROP_BASE = Path(r'C:\Users\justi\Downloads\coursework\coursework\preprocessing\Preprocessing\OUTPUTNEW')
SHARP_CSV = CROP_BASE / 'Focus' / 'sharp_crops.csv'
OUTPUT_DIR = Path(__file__).parent / 'margin_v3_output'
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_SIZE = 256
FEATHER = 10


# =============================================================================
# Detection
# =============================================================================

def detect_contour(img):
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return None
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    return {'cx': cx, 'cy': cy, 'radius': int(round(np.mean(dists))), 'contour': cnt}


# =============================================================================
# Flattening
# =============================================================================

def flatten_contour_distmap(image, contour, margin_inner, margin_outer, feather_width=FEATHER):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    signed_dist = np.where(mask > 0, dist_inside, -dist_outside)
    out = img_f.copy()
    fw = feather_width
    out[signed_dist > (margin_inner + fw)] = 0.0
    mask_fi = (signed_dist > margin_inner) & (signed_dist <= (margin_inner + fw))
    if np.any(mask_fi):
        t = np.clip((signed_dist[mask_fi] - margin_inner) / fw, 0, 1)
        out[mask_fi] = 0.5 * (1 + np.cos(np.pi * t)) * img_f[mask_fi]
    mask_fo = (signed_dist < -margin_outer) & (signed_dist >= -(margin_outer + fw))
    if np.any(mask_fo):
        t = np.clip((-signed_dist[mask_fo] - margin_outer) / fw, 0, 1)
        out[mask_fo] = img_f[mask_fo] + 0.5 * (1 - np.cos(np.pi * t)) * (1.0 - img_f[mask_fo])
    out[signed_dist < -(margin_outer + fw)] = 1.0
    return out


# =============================================================================
# Helpers
# =============================================================================

def make_kernel(sigma):
    if sigma <= 0:
        return np.array([[1.0]], dtype=np.float32)
    r = int(np.ceil(4.0 * sigma))
    s = 2 * r + 1
    ax = np.arange(s) - r
    X, Y = np.meshgrid(ax, ax)
    k = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    k /= k.sum()
    return k.astype(np.float32)

def apply_blur(image, sigma):
    if sigma <= 0.5:
        return image.copy()
    return cv2.filter2D(image, -1, make_kernel(sigma), borderType=cv2.BORDER_REPLICATE)

def measure_erf_sigma(image):
    centre, radius = detect_sphere(image)
    if centre is None:
        return float('nan'), float('nan')
    result = measure_blur_erf(image, centre, radius, num_rays=36)
    return result.sigma, result.confidence

def measure_interior(image, cx, cy, radius, fraction=0.5):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist < (radius * fraction)
    if not np.any(mask):
        return float('nan'), float('nan')
    region = image[mask]
    return float(np.var(region)), float(np.max(region))

def gray_to_bgr(img):
    if img.dtype != np.uint8:
        return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# =============================================================================
# Load crops — more per camera for better stats
# =============================================================================

def load_crops_by_camera(max_per_camera=8):
    import pandas as pd
    crops = {'g': [], 'v': [], 'm': []}

    if not SHARP_CSV.exists():
        for cam in ('g', 'v', 'm'):
            cam_crops = sorted(CROP_BASE.rglob(f'{cam}/crops/*_crop.png'))
            step = max(1, len(cam_crops) // max_per_camera)
            for path in cam_crops[::step][:max_per_camera]:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
                    crops[cam].append({'path': path, 'filename': path.name,
                                       'image': img.astype(np.float32) / 255.0})
        return crops

    df = pd.read_csv(SHARP_CSV)
    for cam in ('g', 'v', 'm'):
        cam_df = df[df['camera'] == cam]
        if len(cam_df) == 0:
            continue
        step = max(1, len(cam_df) // max_per_camera)
        sample = cam_df.iloc[::step].head(max_per_camera)
        for _, row in sample.iterrows():
            crop_path = None
            if 'crop_path' in row and pd.notna(row['crop_path']):
                crop_path = Path(row['crop_path'])
            if crop_path is None or not crop_path.exists():
                if 'folder' in row and 'filename' in row:
                    crop_path = CROP_BASE / row['folder'] / cam / 'crops' / row['filename']
            if crop_path is not None and crop_path.exists():
                img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
                    crops[cam].append({'path': crop_path, 'filename': crop_path.name,
                                       'image': img.astype(np.float32) / 255.0})
    return crops


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("MARGIN v3: ContourDistMap — Finding optimal margin per camera")
    print("=" * 90)
    print()

    crops = load_crops_by_camera(max_per_camera=8)
    for cam in ('g', 'v', 'm'):
        print(f"  Camera {cam}: {len(crops[cam])} crops")
    print()

    # Configs: just the ones we care about
    margin_configs = [
        ('Fixed 40px', False, 40, 0),
        ('Fixed 50px (current)', False, 50, 0),
        ('Fixed 60px', False, 60, 0),
        ('14% of radius', True, 0, 0.14),
        ('20% of radius', True, 0, 0.20),
        ('25% of radius', True, 0, 0.25),
    ]

    sigma_targets = [2, 4, 6, 8, 10, 12]
    results = []
    t_start = time.time()

    for cam in ('g', 'v', 'm'):
        print(f"\n{'=' * 90}")
        print(f"  CAMERA {cam.upper()}")
        print(f"{'=' * 90}")

        for ci, crop_info in enumerate(crops[cam]):
            img = crop_info['image']
            fn = crop_info['filename']

            det = detect_contour(img)
            if det is None:
                print(f"\n  [{ci+1}/{len(crops[cam])}] {fn}: DETECTION FAILED — skipping")
                continue

            R = det['radius']
            int_var_raw, int_max_raw = measure_interior(img, det['cx'], det['cy'], R)
            raw_native, _ = measure_erf_sigma(img)

            print(f"\n  [{ci+1}/{len(crops[cam])}] {fn}: R={R}px, "
                  f"raw native={'%.2f' % raw_native if not np.isnan(raw_native) else 'N/A'}, "
                  f"interior max={int_max_raw:.3f}")

            # Raw baseline
            for test_type in ['quadrature', 'fixed_kernel']:
                for st in sigma_targets:
                    if test_type == 'quadrature':
                        if np.isnan(raw_native) or st**2 <= raw_native**2:
                            continue
                        ks = np.sqrt(st**2 - raw_native**2)
                        expected = st
                    else:
                        ks = st
                        expected = np.sqrt(raw_native**2 + st**2) if not np.isnan(raw_native) else float('nan')
                    blurred = apply_blur(img, ks)
                    measured, _ = measure_erf_sigma(blurred)
                    err = 100 * (measured - expected) / expected if (not np.isnan(measured) and not np.isnan(expected) and expected > 0) else float('nan')
                    results.append({
                        'camera': cam, 'filename': fn, 'config': 'Raw (no flattening)',
                        'radius': R, 'm_in': 0, 'm_out': 0,
                        'margin_as_pct_of_R': 0,
                        'test_type': test_type, 'sigma_target': expected,
                        'kernel_applied': ks, 'native_sigma': raw_native,
                        'measured': measured, 'error_pct': err,
                        'int_var': int_var_raw, 'int_max': int_max_raw,
                    })

            # Print raw baseline
            raw_q = [r['error_pct'] for r in results if r['config'] == 'Raw (no flattening)' and r['filename'] == fn and r['test_type'] == 'quadrature' and not np.isnan(r['error_pct']) and r['error_pct'] > -50]
            raw_f = [r['error_pct'] for r in results if r['config'] == 'Raw (no flattening)' and r['filename'] == fn and r['test_type'] == 'fixed_kernel' and not np.isnan(r['error_pct']) and r['error_pct'] > -50]
            rq = f"{np.mean(raw_q):+.1f}%" if raw_q else "N/A"
            rf = f"{np.mean(raw_f):+.1f}%" if raw_f else "N/A"
            print(f"    {'Raw (no flattening)':>25s}  margin=  0px ( 0%R)  |  quad={rq:>8s}  fixed={rf:>8s}  |  max={int_max_raw:.3f}")

            # Each margin config
            for cfg_name, is_prop, fixed_px, frac in margin_configs:
                if is_prop:
                    m = max(int(R * frac), 1)
                else:
                    m = fixed_px

                if R - m <= 0:
                    print(f"    {cfg_name:>25s}  margin={m:>3d}px ({100*m/R:>3.0f}%R)  |  SKIPPED (margin >= radius)")
                    continue

                flat = flatten_contour_distmap(img, det['contour'], m, m)
                int_var, int_max = measure_interior(flat, det['cx'], det['cy'], R)
                native_sigma, _ = measure_erf_sigma(flat)

                for test_type in ['quadrature', 'fixed_kernel']:
                    for st in sigma_targets:
                        if test_type == 'quadrature':
                            if np.isnan(native_sigma) or st**2 <= native_sigma**2:
                                continue
                            ks = np.sqrt(st**2 - native_sigma**2)
                            expected = st
                        else:
                            ks = st
                            expected = np.sqrt(native_sigma**2 + ks**2) if not np.isnan(native_sigma) else float('nan')
                        blurred = apply_blur(flat, ks)
                        measured, _ = measure_erf_sigma(blurred)
                        err = 100 * (measured - expected) / expected if (not np.isnan(measured) and not np.isnan(expected) and expected > 0) else float('nan')
                        results.append({
                            'camera': cam, 'filename': fn, 'config': cfg_name,
                            'radius': R, 'm_in': m, 'm_out': m,
                            'margin_as_pct_of_R': 100 * m / R,
                            'test_type': test_type, 'sigma_target': expected,
                            'kernel_applied': ks, 'native_sigma': native_sigma,
                            'measured': measured, 'error_pct': err,
                            'int_var': int_var, 'int_max': int_max,
                        })

                # Summary line
                cfg_q = [r['error_pct'] for r in results if r['config'] == cfg_name and r['filename'] == fn and r['test_type'] == 'quadrature' and not np.isnan(r['error_pct']) and r['error_pct'] > -50]
                cfg_f = [r['error_pct'] for r in results if r['config'] == cfg_name and r['filename'] == fn and r['test_type'] == 'fixed_kernel' and not np.isnan(r['error_pct']) and r['error_pct'] > -50]
                qs = f"{np.mean(cfg_q):+.1f}%" if cfg_q else "N/A"
                fs = f"{np.mean(cfg_f):+.1f}%" if cfg_f else "N/A"
                ns = f"{native_sigma:.2f}" if not np.isnan(native_sigma) else "N/A"
                clean = "CLEAN" if int_max < 0.01 else ("partial" if int_max < 0.1 else "DIRTY")
                print(f"    {cfg_name:>25s}  margin={m:>3d}px ({100*m/R:>3.0f}%R)  |  quad={qs:>8s}  fixed={fs:>8s}  |  max={int_max:.3f} {clean}")

    # ==========================================================================
    # Save and summarise
    # ==========================================================================
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / 'margin_v3_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n\nCSV saved: {csv_path}")

    # Remove ERF fitting failures for summary
    clean_df = df[(df['error_pct'] > -50) | (df['error_pct'].isna())]
    quad = clean_df[clean_df['test_type'] == 'quadrature']
    fixed = clean_df[clean_df['test_type'] == 'fixed_kernel']

    print()
    print("=" * 100)
    print("RESULTS: What's the best margin for each camera?")
    print("  Quadrature test (pipeline-realistic), ERF failures removed")
    print("=" * 100)

    for cam in ('g', 'v', 'm'):
        cam_quad = quad[quad['camera'] == cam]
        if len(cam_quad) == 0:
            continue
        n_crops = cam_quad['filename'].nunique()
        radii = cam_quad.drop_duplicates('filename')['radius']
        print(f"\n  CAMERA {cam.upper()} — {n_crops} crops, R = {radii.min()}-{radii.max()}px")
        print(f"  {'Config':>25s}  {'Actual margin':>15s}  {'Quad err':>10s}  {'Fixed err':>10s}  {'Interior':>10s}  {'Verdict':>10s}")
        print(f"  {'-'*85}")

        for cfg in ['Raw (no flattening)'] + [c[0] for c in margin_configs]:
            cdf = cam_quad[cam_quad['config'] == cfg]
            cdf_f = fixed[(fixed['camera'] == cam) & (fixed['config'] == cfg)]
            if len(cdf) == 0 and len(cdf_f) == 0:
                continue

            q_err = cdf['error_pct'].dropna()
            f_err = cdf_f['error_pct'].dropna()
            im = cdf['int_max'].mean() if len(cdf) > 0 else (cdf_f['int_max'].mean() if len(cdf_f) > 0 else float('nan'))

            # Get actual margin range
            margins = cdf['m_in'].unique() if len(cdf) > 0 else cdf_f['m_in'].unique()
            pcts = cdf['margin_as_pct_of_R'].unique() if len(cdf) > 0 else [0]
            if len(margins) == 1:
                m_str = f"{margins[0]:>3.0f}px ({pcts[0]:>3.0f}%R)"
            else:
                m_str = f"{min(margins):.0f}-{max(margins):.0f}px"

            q_str = f"{q_err.mean():>+9.2f}%" if len(q_err) > 0 else f"{'N/A':>10s}"
            f_str = f"{f_err.mean():>+9.2f}%" if len(f_err) > 0 else f"{'N/A':>10s}"

            # Verdict
            q_val = q_err.mean() if len(q_err) > 0 else float('nan')
            if np.isnan(q_val) and np.isnan(im):
                verdict = "NO DATA"
            elif np.isnan(q_val):
                verdict = "ERF FAIL"
            elif im > 0.3:
                verdict = "DIRTY"
            elif im > 0.05:
                verdict = "PARTIAL"
            elif abs(q_val) < 3:
                verdict = "EXCELLENT"
            elif abs(q_val) < 8:
                verdict = "GOOD"
            elif abs(q_val) < 15:
                verdict = "OK"
            else:
                verdict = "POOR"

            print(f"  {cfg:>25s}  {m_str:>15s}  {q_str}  {f_str}  {im:>10.4f}  {verdict:>10s}")

    # Per-sigma per-camera for the key configs
    print()
    print("=" * 100)
    print("PER-SIGMA: How does each margin perform at different blur levels?")
    print("=" * 100)

    for cam in ('g', 'v', 'm'):
        cam_quad = quad[quad['camera'] == cam]
        if len(cam_quad) == 0:
            continue
        radii = cam_quad.drop_duplicates('filename')['radius']
        print(f"\n  CAMERA {cam.upper()} (R = {radii.min()}-{radii.max()}px):")

        key = ['Raw (no flattening)', '14% of radius', '20% of radius', '25% of radius', 'Fixed 50px (current)']
        for cfg in key:
            c = cam_quad[cam_quad['config'] == cfg]
            if len(c) == 0:
                continue
            sigmas_str = []
            for st in sigma_targets:
                st_df = c[(c['sigma_target'] > st-0.5) & (c['sigma_target'] < st+0.5)]
                valid = st_df['error_pct'].dropna()
                if len(valid) > 0:
                    sigmas_str.append(f"s={st:>2d}:{valid.mean():>+6.1f}%")
            if sigmas_str:
                print(f"    {cfg:>25s}:  {' | '.join(sigmas_str)}")

    total_time = time.time() - t_start
    print(f"\n\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("Done.")


if __name__ == '__main__':
    main()
