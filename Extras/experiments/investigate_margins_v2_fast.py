"""
Margin Investigation v2 FAST — Reduced scope based on initial findings.

Findings from full run:
  - EllipDet+EllipFlat is consistently worse (9-10% error). DROPPED.
  - Fixed pixel margins broken on small spheres. DROPPED.
  - Asymmetric barely differs from symmetric. Kept one for confirmation.
  - Detection method (circle vs ellipse vs contour) barely matters. Keeping all 3.

Approaches kept:
  CircDet+CircFlat    — current pipeline baseline
  EllipDet+CircFlat   — does better detection help?
  ContourDistMap      — shape-agnostic (best on g-camera)

Margin configs kept:
  10% of radius       — sweet spot lower bound
  14% of radius       — sweet spot mid
  20% of radius       — sweet spot upper bound
  14% min 10px        — best hybrid
  In8% Out14%         — one asymmetric for confirmation

Skips crops already processed in the full run (g-camera crops).

Run with: phantom_env\\Scripts\\python.exe investigate_margins_v2_fast.py
"""

import sys
import csv
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from paths_config import CROP_BASE, SHARP_CSV

CALIB_DIR = str(Path(__file__).parent.parent.parent / 'calibration')
if CALIB_DIR not in sys.path:
    sys.path.insert(0, CALIB_DIR)

from blur_measurement import detect_sphere, measure_blur_erf
from sphere_processing import find_sphere_center
OUTPUT_DIR = Path(__file__).parent / 'margin_v2_output'
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_SIZE = 256
FEATHER = 10


# =============================================================================
# Detection
# =============================================================================

def detect_circle(img):
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
    pts = cnt.reshape(-1, 2).astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return {'method': 'circle', 'cx': int(round(cx)), 'cy': int(round(cy)),
            'radius': int(round(r)), 'major': r, 'minor': r, 'angle': 0.0,
            'eccentricity': 1.0, 'contour': cnt}


def detect_ellipse(img):
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
    if len(cnt) < 5:
        return None
    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (w_axis, h_axis), angle = ellipse
    semi_a = max(w_axis, h_axis) / 2.0
    semi_b = min(w_axis, h_axis) / 2.0
    return {'method': 'ellipse', 'cx': int(round(cx)), 'cy': int(round(cy)),
            'radius': int(round((semi_a + semi_b) / 2)),
            'major': semi_a, 'minor': semi_b, 'angle': angle,
            'eccentricity': semi_b / semi_a if semi_a > 0 else 1.0, 'contour': cnt}


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
    mean_r = float(np.mean(dists))
    ellipse = cv2.fitEllipse(cnt) if len(cnt) >= 5 else None
    ecc = min(ellipse[1][0], ellipse[1][1]) / max(ellipse[1][0], ellipse[1][1]) if ellipse and max(ellipse[1]) > 0 else 1.0
    return {'method': 'contour', 'cx': cx, 'cy': cy, 'radius': int(round(mean_r)),
            'major': max(dists), 'minor': min(dists), 'angle': 0.0,
            'eccentricity': ecc, 'contour': cnt}


# =============================================================================
# Flattening
# =============================================================================

def flatten_circular(image, cx, cy, radius, margin_inner, margin_outer, feather_width=FEATHER):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
    out = img_f.copy()
    fw = feather_width
    r_inner, r_outer = radius - margin_inner, radius + margin_outer
    out[dist < (r_inner - fw)] = 0.0
    mask_fi = (dist >= (r_inner - fw)) & (dist < r_inner)
    if np.any(mask_fi):
        t = np.clip((dist[mask_fi] - (r_inner - fw)) / fw, 0, 1)
        out[mask_fi] = 0.5 * (1 - np.cos(np.pi * t)) * img_f[mask_fi]
    mask_fo = (dist > r_outer) & (dist <= (r_outer + fw))
    if np.any(mask_fo):
        t = np.clip((dist[mask_fo] - r_outer) / fw, 0, 1)
        out[mask_fo] = img_f[mask_fo] + 0.5 * (1 - np.cos(np.pi * t)) * (1.0 - img_f[mask_fo])
    out[dist > (r_outer + fw)] = 1.0
    return out


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

def make_kernel(sigma, rf=4.0):
    if sigma <= 0:
        return np.array([[1.0]], dtype=np.float32)
    r = int(np.ceil(rf * sigma))
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

def measure_contamination_profile(image, cx, cy, radius, ring_width=2):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
    max_scan = min(radius - 5, 60)
    profile = []
    for d in range(0, int(max_scan), ring_width):
        inner_r, outer_r = radius - d - ring_width, radius - d
        if inner_r <= 0:
            break
        ring_mask = (dist >= inner_r) & (dist < outer_r)
        if np.sum(ring_mask) < 10:
            continue
        profile.append({'direction': 'inner', 'distance_from_edge': d + ring_width // 2,
                        'variance': float(np.var(image[ring_mask]))})
    for d in range(0, 60, ring_width):
        inner_r, outer_r = radius + d, radius + d + ring_width
        ring_mask = (dist >= inner_r) & (dist < outer_r)
        if np.sum(ring_mask) < 10:
            continue
        profile.append({'direction': 'outer', 'distance_from_edge': d + ring_width // 2,
                        'variance': float(np.var(image[ring_mask]))})
    if not profile:
        return 0, 0
    inner_vars = [p['variance'] for p in profile if p['direction'] == 'inner' and p['distance_from_edge'] > 20]
    outer_vars = [p['variance'] for p in profile if p['direction'] == 'outer' and p['distance_from_edge'] > 20]
    inner_baseline = min(inner_vars) if inner_vars else 0.001
    outer_baseline = min(outer_vars) if outer_vars else 0.001
    inner_contam = max((p['distance_from_edge'] for p in profile if p['direction'] == 'inner' and p['variance'] > 2 * inner_baseline), default=0)
    outer_contam = max((p['distance_from_edge'] for p in profile if p['direction'] == 'outer' and p['variance'] > 2 * outer_baseline), default=0)
    return inner_contam, outer_contam

def gray_to_bgr(img):
    if img.dtype != np.uint8:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


# =============================================================================
# Visual output
# =============================================================================

def save_detection_comparison(cam, filename, img, det_c, det_e, det_d, output_dir):
    panels = []
    for det, label in [(det_c, 'Circle'), (det_e, 'Ellipse'), (det_d, 'Contour')]:
        vis = gray_to_bgr(img)
        if det:
            if det['method'] == 'contour':
                cv2.drawContours(vis, [det['contour']], 0, (0, 255, 0), 1)
                cv2.circle(vis, (det['cx'], det['cy']), 3, (0, 0, 255), -1)
            elif det['method'] == 'ellipse':
                cv2.ellipse(vis, (det['cx'], det['cy']),
                            (int(det['major']), int(det['minor'])),
                            det['angle'], 0, 360, (0, 255, 0), 1)
            else:
                cv2.circle(vis, (det['cx'], det['cy']), det['radius'], (0, 255, 0), 1)
            cv2.putText(vis, f"{label} R={det['radius']}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1)
        panels.append(vis)
    combined = np.hstack(panels)
    bar = np.zeros((22, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{cam}/{filename} - Detection Comparison", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    combined = np.vstack([bar, combined])
    crop_dir = output_dir / f'{cam}_{filename.replace(".png", "")}'
    crop_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_dir / 'detection_comparison.png'), combined)


def save_flatten_comparison(cam, filename, img, configs_results, output_dir):
    approach_names = {
        'CircDet+CircFlat': 'Circle Detect + Circle Flatten',
        'EllipDet+CircFlat': 'Ellipse Detect + Circle Flatten',
        'ContourDistMap': 'Contour Detect + Distance-Map Flatten',
    }
    groups = {}
    for label, flat_img, int_var, int_max in configs_results:
        for prefix in approach_names:
            if label.startswith(prefix):
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append((label, flat_img, int_var, int_max))
                break

    for prefix, items in groups.items():
        panels = [gray_to_bgr(img)]
        cv2.putText(panels[0], "RAW", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1)
        for label, flat_img, int_var, int_max in items:
            vis = gray_to_bgr(flat_img)
            short = label.replace(prefix + ' ', '')
            cv2.putText(vis, short, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
            clean = "CLEAN" if int_max < 0.01 else f"max={int_max:.2f}"
            cv2.putText(vis, clean, (3, vis.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
            panels.append(vis)
        combined = np.hstack(panels)
        bar = np.zeros((22, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, f"{cam}/{filename} | {approach_names[prefix]} | {len(items)} configs",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        combined = np.vstack([bar, combined])
        crop_dir = output_dir / f'{cam}_{filename.replace(".png", "")}'
        crop_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_dir / f'flatten_{prefix.replace("+", "_")}.png'), combined)


# =============================================================================
# Load crops
# =============================================================================

def load_crops_by_camera(max_per_camera=4, skip_cameras=None):
    import pandas as pd
    crops = {'g': [], 'v': [], 'm': []}
    skip_cameras = skip_cameras or []

    if not SHARP_CSV.exists():
        for cam in ('g', 'v', 'm'):
            if cam in skip_cameras:
                continue
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
        if cam in skip_cameras:
            continue
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
# Configs (reduced)
# =============================================================================

@dataclass
class MarginConfig:
    name: str
    det_source: str
    flat_geom: str
    is_prop: bool
    px_in: int
    px_out: int
    frac_in: float
    frac_out: float

    def compute_margins(self, radius: int) -> Tuple[int, int]:
        if self.is_prop:
            return max(int(radius * self.frac_in), self.px_in), max(int(radius * self.frac_out), self.px_out)
        return self.px_in, self.px_out


def get_configs():
    configs = []
    for approach, det, flat in [
        ('CircDet+CircFlat', 'circle', 'circle'),
        ('ContourDistMap', 'contour', 'distmap'),
    ]:
        # Proportional
        for frac in [0.10, 0.14, 0.20]:
            configs.append(MarginConfig(
                f'{approach} {int(frac*100)}% of radius',
                det, flat, True, 0, 0, frac, frac))
        # Best hybrid
        configs.append(MarginConfig(
            f'{approach} 14% min 10px',
            det, flat, True, 10, 10, 0.14, 0.14))
        # One asymmetric
        configs.append(MarginConfig(
            f'{approach} In8% Out14%',
            det, flat, True, 0, 0, 0.08, 0.14))
    return configs


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("MARGIN INVESTIGATION v2 FAST")
    print("  Reduced scope: 3 approaches x 5 margin configs + raw baseline")
    print("  Skipping g-camera (already tested in full run)")
    print("=" * 80)
    print()

    # Skip g-camera — already have full results
    crops = load_crops_by_camera(max_per_camera=4, skip_cameras=['g'])
    for cam in ('g', 'v', 'm'):
        print(f"  Camera {cam}: {len(crops[cam])} crops")
    print()

    configs = get_configs()
    sigma_targets = [2, 4, 6, 8, 10, 12]
    total_crops = sum(len(v) for v in crops.values())
    print(f"  {len(configs)} configs x {total_crops} crops x {len(sigma_targets)} sigmas x 2 tests")
    print()

    results = []
    detection_log = []
    t_start = time.time()

    for cam in ('g', 'v', 'm'):
        for ci, crop_info in enumerate(crops[cam]):
            img = crop_info['image']
            fn = crop_info['filename']
            t_crop = time.time()

            det_c = detect_circle(img)
            det_e = detect_ellipse(img)
            det_d = detect_contour(img)

            if det_c is None and det_e is None and det_d is None:
                print(f"  SKIP {cam}/{fn}: all detections failed")
                continue

            r_circ = det_c['radius'] if det_c else (det_e['radius'] if det_e else det_d['radius'])
            ecc = det_e['eccentricity'] if det_e else 1.0
            axis_diff = det_e['major'] - det_e['minor'] if det_e else 0
            r_contour = det_d['radius'] if det_d else 0

            det_for_contam = det_c if det_c else (det_e if det_e else det_d)
            inner_contam, outer_contam = measure_contamination_profile(
                img, det_for_contam['cx'], det_for_contam['cy'], det_for_contam['radius'])

            detection_log.append({
                'camera': cam, 'filename': fn,
                'circ_r': det_c['radius'] if det_c else None,
                'ellip_major': det_e['major'] if det_e else None,
                'ellip_minor': det_e['minor'] if det_e else None,
                'eccentricity': ecc, 'contour_r': r_contour,
                'inner_contamination_px': inner_contam,
                'outer_contamination_px': outer_contam,
            })

            ellip_str = f"({det_e['major']:.0f},{det_e['minor']:.0f})" if det_e else "N/A"
            print(f"\n  [{ci+1}/{len(crops[cam])}] {cam}/{fn}: R={r_circ}, ellip={ellip_str}, contour_R={r_contour}")
            print(f"      Contamination: {inner_contam}px inward, {outer_contam}px outward")

            # Save detection visual
            save_detection_comparison(cam, fn, img, det_c, det_e, det_d, OUTPUT_DIR)
            flatten_visuals = []

            # --- Raw baseline ---
            raw_native, _ = measure_erf_sigma(img)
            raw_int_var, raw_int_max = measure_interior(img, det_for_contam['cx'], det_for_contam['cy'], r_circ)

            for test_type in ['quadrature', 'fixed_kernel']:
                for sigma_target in sigma_targets:
                    if test_type == 'quadrature':
                        if np.isnan(raw_native) or sigma_target**2 <= raw_native**2:
                            continue
                        ks = np.sqrt(sigma_target**2 - raw_native**2)
                        expected = sigma_target
                    else:
                        ks = sigma_target
                        expected = np.sqrt(raw_native**2 + sigma_target**2) if not np.isnan(raw_native) else float('nan')
                    blurred = apply_blur(img, ks)
                    measured, _ = measure_erf_sigma(blurred)
                    err = 100 * (measured - expected) / expected if (not np.isnan(measured) and not np.isnan(expected) and expected > 0) else float('nan')
                    results.append({
                        'camera': cam, 'filename': fn, 'config': 'Raw (no flattening)',
                        'det_source': 'circle', 'flat_geom': 'none',
                        'radius': r_circ, 'eccentricity': ecc,
                        'm_in': 0, 'm_out': 0, 'test_type': test_type,
                        'sigma_target': expected, 'kernel_applied': ks,
                        'native_sigma': raw_native, 'measured': measured,
                        'error_pct': err, 'int_var': raw_int_var, 'int_max': raw_int_max,
                    })

            raw_q = [r['error_pct'] for r in results if r['config'] == 'Raw (no flattening)' and r['filename'] == fn and r['test_type'] == 'quadrature' and not np.isnan(r['error_pct'])]
            raw_f = [r['error_pct'] for r in results if r['config'] == 'Raw (no flattening)' and r['filename'] == fn and r['test_type'] == 'fixed_kernel' and not np.isnan(r['error_pct'])]
            rq = f"{np.mean(raw_q):+.2f}%" if raw_q else "N/A"
            rf = f"{np.mean(raw_f):+.2f}%" if raw_f else "N/A"
            rn = f"{raw_native:.2f}" if not np.isnan(raw_native) else "N/A"
            print(f"      {'Raw (no flattening)':>30s}  |  native={rn}  quad={rq}  fixed={rf}  |  DIRTY (max={raw_int_max:.3f})")

            # --- Test each config ---
            for cfg in configs:
                if cfg.det_source == 'ellipse' and det_e is None:
                    continue
                if cfg.det_source == 'circle' and det_c is None:
                    continue
                if cfg.det_source == 'contour' and det_d is None:
                    continue

                det = {'contour': det_d, 'ellipse': det_e, 'circle': det_c}[cfg.det_source]
                mean_r = int((det['major'] + det['minor']) / 2)
                m_in, m_out = cfg.compute_margins(mean_r)
                if mean_r - m_in <= 0:
                    continue

                if cfg.flat_geom == 'distmap':
                    flat = flatten_contour_distmap(img, det['contour'], m_in, m_out)
                else:
                    flat = flatten_circular(img, det['cx'], det['cy'], mean_r, m_in, m_out)

                int_var, int_max = measure_interior(flat, det['cx'], det['cy'], mean_r)
                native_sigma, _ = measure_erf_sigma(flat)
                flatten_visuals.append((cfg.name, flat, int_var, int_max))

                for test_type in ['quadrature', 'fixed_kernel']:
                    for sigma_target in sigma_targets:
                        if test_type == 'quadrature':
                            if np.isnan(native_sigma) or sigma_target**2 <= native_sigma**2:
                                continue
                            ks = np.sqrt(sigma_target**2 - native_sigma**2)
                            expected = sigma_target
                        else:
                            ks = sigma_target
                            expected = np.sqrt(native_sigma**2 + ks**2) if not np.isnan(native_sigma) else float('nan')
                        blurred = apply_blur(flat, ks)
                        measured, _ = measure_erf_sigma(blurred)
                        err = 100 * (measured - expected) / expected if (not np.isnan(measured) and not np.isnan(expected) and expected > 0) else float('nan')
                        results.append({
                            'camera': cam, 'filename': fn, 'config': cfg.name,
                            'det_source': cfg.det_source, 'flat_geom': cfg.flat_geom,
                            'radius': mean_r, 'eccentricity': ecc,
                            'm_in': m_in, 'm_out': m_out, 'test_type': test_type,
                            'sigma_target': expected, 'kernel_applied': ks,
                            'native_sigma': native_sigma, 'measured': measured,
                            'error_pct': err, 'int_var': int_var, 'int_max': int_max,
                        })

                # Summary line
                qe = [r['error_pct'] for r in results if r['config'] == cfg.name and r['filename'] == fn and r['test_type'] == 'quadrature' and not np.isnan(r['error_pct'])]
                fe = [r['error_pct'] for r in results if r['config'] == cfg.name and r['filename'] == fn and r['test_type'] == 'fixed_kernel' and not np.isnan(r['error_pct'])]
                qs = f"{np.mean(qe):+.2f}%" if qe else "N/A"
                fs = f"{np.mean(fe):+.2f}%" if fe else "N/A"
                ns = f"{native_sigma:.2f}" if not np.isnan(native_sigma) else "N/A"
                clean = "CLEAN" if int_max < 0.01 else ("mostly clean" if int_max < 0.1 else "DIRTY")
                print(f"      {cfg.name:>30s}  |  native={ns}  quad={qs}  fixed={fs}  |  {clean} (max={int_max:.3f})")

            # Save flatten visuals
            if flatten_visuals:
                save_flatten_comparison(cam, fn, img, flatten_visuals, OUTPUT_DIR)

            elapsed = time.time() - t_crop
            print(f"      (completed in {elapsed:.1f}s)")

    # Save results
    import pandas as pd
    if results:
        df = pd.DataFrame(results)
        # Append to existing CSV if present
        csv_path = OUTPUT_DIR / 'margin_v2_results.csv'
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"\nResults CSV: {csv_path} ({len(df)} rows)")

    det_df = pd.DataFrame(detection_log)
    det_csv = OUTPUT_DIR / 'detection_comparison.csv'
    if det_csv.exists():
        existing_det = pd.read_csv(det_csv)
        det_df = pd.concat([existing_det, det_df], ignore_index=True)
    det_df.to_csv(det_csv, index=False)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Summary
    if results:
        print()
        print("=" * 100)
        print("SUMMARY: Mean blur error % by config (quadrature test)")
        print("=" * 100)
        quad_df = df[df['test_type'] == 'quadrature']
        print(f"  {'Config':>35s}  ", end='')
        for cam in sorted(quad_df['camera'].unique()):
            print(f"  {'cam '+cam:>9s}", end='')
        print(f"  {'Overall':>9s}  {'IntMax':>7s}  {'Verdict':>10s}")
        print("  " + "-" * 90)

        for cfg_name in dict.fromkeys(r['config'] for r in results):
            cfg_df = quad_df[quad_df['config'] == cfg_name]
            if len(cfg_df) == 0:
                continue
            parts = []
            for cam in sorted(quad_df['camera'].unique()):
                cam_cfg = cfg_df[cfg_df['camera'] == cam]
                valid = cam_cfg['error_pct'].dropna()
                parts.append(f"{valid.mean():>+8.2f}%" if len(valid) > 0 else f"{'N/A':>9s}")
            all_valid = cfg_df['error_pct'].dropna()
            all_mean = all_valid.mean() if len(all_valid) > 0 else float('nan')
            int_max = cfg_df['int_max'].mean()
            if np.isnan(all_mean):
                verdict = "NO DATA"
            elif abs(all_mean) < 2 and int_max < 0.01:
                verdict = "EXCELLENT"
            elif abs(all_mean) < 5 and int_max < 0.05:
                verdict = "GOOD"
            elif abs(all_mean) < 10 and int_max < 0.1:
                verdict = "OK"
            elif int_max > 0.3:
                verdict = "DIRTY"
            else:
                verdict = "POOR"
            all_str = f"{all_mean:>+8.2f}%" if not np.isnan(all_mean) else f"{'N/A':>9s}"
            print(f"  {cfg_name:>35s}  {'  '.join(parts)}  {all_str}  {int_max:>7.4f}  {verdict:>10s}")

    print(f"\nVisual panels in: {OUTPUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
