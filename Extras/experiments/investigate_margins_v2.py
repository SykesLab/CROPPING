"""
Comprehensive Margin & Detection Investigation v2

Tests three axes of variation:
  A. Detection method: circle (minEnclosingCircle) vs ellipse (fitEllipse)
  B. Flattening geometry: circular zones vs elliptical zones
  C. Margin strategy: fixed px, proportional, hybrid, asymmetric

For each combination:
  1. Detect sphere (circle or ellipse)
  2. Flatten with that geometry + margin config
  3. Measure native ERF sigma on flattened crop
  4. Apply known synthetic blur (quadrature) at multiple target sigmas
  5. Measure ERF on blurred crop, compare to target -> error %
  6. Measure interior cleanliness (variance, max intensity)

All crops resized to 256x256 (model scale) before testing.

Run with: phantom_env\Scripts\python.exe investigate_margins_v2.py
"""

import sys
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

CALIB_DIR = str(Path(__file__).parent.parent.parent / 'calibration')
if CALIB_DIR not in sys.path:
    sys.path.insert(0, CALIB_DIR)

from blur_measurement import detect_sphere, measure_blur_erf
from sphere_processing import find_sphere_center

CROP_BASE = Path(r'C:\Users\justi\Downloads\coursework\coursework\preprocessing\Preprocessing\OUTPUTNEW')
SHARP_CSV = CROP_BASE / 'Focus' / 'sharp_crops.csv'
OUTPUT_DIR = Path(__file__).parent / 'margin_v2_output'
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_SIZE = 256
FEATHER = 10


# =============================================================================
# Detection methods
# =============================================================================

def detect_circle(img):
    """Current method: Canny + largest contour + minEnclosingCircle.
    Returns dict with detection info or None."""
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

    return {
        'method': 'circle',
        'cx': int(round(cx)), 'cy': int(round(cy)),
        'radius': int(round(r)),
        'major': r, 'minor': r, 'angle': 0.0,
        'eccentricity': 1.0,
        'contour': cnt,
    }


def detect_ellipse(img):
    """Ellipse fit: Canny + largest contour + fitEllipse.
    Returns dict with detection info or None."""
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
    # OpenCV fitEllipse returns full axis lengths, we want semi-axes
    semi_a = max(w_axis, h_axis) / 2.0  # major semi-axis
    semi_b = min(w_axis, h_axis) / 2.0  # minor semi-axis

    # Also get circle fit for comparison
    pts = cnt.reshape(-1, 2).astype(np.float32)
    _, r_circle = cv2.minEnclosingCircle(pts)

    return {
        'method': 'ellipse',
        'cx': int(round(cx)), 'cy': int(round(cy)),
        'radius': int(round((semi_a + semi_b) / 2)),  # mean radius for backwards compat
        'major': semi_a, 'minor': semi_b, 'angle': angle,
        'eccentricity': semi_b / semi_a if semi_a > 0 else 1.0,
        'circle_radius': r_circle,
        'contour': cnt,
    }


def detect_contour(img):
    """Contour-based detection: find the edge contour directly, compute centroid.
    No geometric fitting — uses the actual contour shape.
    Returns dict with detection info or None."""
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

    # Centroid from moments
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Compute mean distance from centroid to contour points as "effective radius"
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    mean_r = float(np.mean(dists))

    # Also fit ellipse for comparison stats
    ellipse = cv2.fitEllipse(cnt) if len(cnt) >= 5 else None
    if ellipse:
        (_, _), (w_axis, h_axis), _ = ellipse
        ecc = min(w_axis, h_axis) / max(w_axis, h_axis) if max(w_axis, h_axis) > 0 else 1.0
    else:
        ecc = 1.0

    return {
        'method': 'contour',
        'cx': cx, 'cy': cy,
        'radius': int(round(mean_r)),
        'major': max(dists), 'minor': min(dists),
        'angle': 0.0,
        'eccentricity': ecc,
        'contour': cnt,
    }


# =============================================================================
# Flattening: circular, elliptical, and contour distance-map
# =============================================================================

def flatten_circular(image, cx, cy, radius, margin_inner, margin_outer, feather_width=FEATHER):
    """Flatten using circular zones (current approach)."""
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()

    h, w = img_f.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)

    out = img_f.copy()
    fw = feather_width
    r_inner = radius - margin_inner
    r_outer = radius + margin_outer

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


def flatten_elliptical(image, cx, cy, semi_a, semi_b, angle_deg,
                       margin_inner, margin_outer, feather_width=FEATHER):
    """Flatten using elliptical zones — zones follow the ellipse shape."""
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()

    h, w = img_f.shape[:2]
    Y, X = np.mgrid[:h, :w]

    # Transform to ellipse-aligned coordinates
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx = (X - cx).astype(np.float32)
    dy = (Y - cy).astype(np.float32)

    # Rotate to ellipse frame
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    # Elliptical distance: 1.0 = on the ellipse boundary
    # Points inside have dist < 1.0, outside have dist > 1.0
    if semi_a <= 0 or semi_b <= 0:
        return img_f.copy()

    ellip_dist = np.sqrt((x_rot / semi_a)**2 + (y_rot / semi_b)**2).astype(np.float32)

    # Convert pixel margins to fractional margins relative to the ellipse
    # Use mean radius for margin scaling
    mean_r = (semi_a + semi_b) / 2.0
    frac_inner = margin_inner / mean_r if mean_r > 0 else 0
    frac_outer = margin_outer / mean_r if mean_r > 0 else 0
    frac_feather = feather_width / mean_r if mean_r > 0 else 0

    # Zone boundaries in elliptical distance units
    d_inner = 1.0 - frac_inner      # inner boundary
    d_outer = 1.0 + frac_outer      # outer boundary
    d_fi = d_inner - frac_feather   # inner feather start
    d_fo = d_outer + frac_feather   # outer feather end

    out = img_f.copy()

    # Zone 1: flat interior
    out[ellip_dist < d_fi] = 0.0

    # Zone 2: inner feather
    mask_fi = (ellip_dist >= d_fi) & (ellip_dist < d_inner)
    if np.any(mask_fi):
        t = np.clip((ellip_dist[mask_fi] - d_fi) / frac_feather, 0, 1)
        out[mask_fi] = 0.5 * (1 - np.cos(np.pi * t)) * img_f[mask_fi]

    # Zone 3: transition — untouched

    # Zone 4: outer feather
    mask_fo = (ellip_dist > d_outer) & (ellip_dist <= d_fo)
    if np.any(mask_fo):
        t = np.clip((ellip_dist[mask_fo] - d_outer) / frac_feather, 0, 1)
        out[mask_fo] = img_f[mask_fo] + 0.5 * (1 - np.cos(np.pi * t)) * (1.0 - img_f[mask_fo])

    # Zone 5: flat background
    out[ellip_dist > d_fo] = 1.0

    return out


def flatten_contour_distmap(image, contour, margin_inner, margin_outer, feather_width=FEATHER):
    """Flatten using signed distance from the actual contour.
    No geometric fitting — zones follow the true edge shape exactly.

    Uses cv2.pointPolygonTest for signed distance:
      positive = inside contour
      negative = outside contour
      0 = on the contour
    """
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]

    # Build signed distance map
    # pointPolygonTest is slow per-pixel, so use distanceTransform on a mask instead
    # Create filled contour mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)  # filled

    # Distance from outside to contour boundary
    dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    # Distance from inside to contour boundary
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)

    # Signed distance: positive = inside, negative = outside
    signed_dist = np.where(mask > 0, dist_inside, -dist_outside)

    # Zone definitions using signed distance:
    #   signed_dist > margin_inner + fw   : deep interior -> 0
    #   margin_inner < signed_dist <= margin_inner + fw : inner feather
    #   -margin_outer <= signed_dist <= margin_inner    : transition zone (preserve)
    #   -margin_outer - fw <= signed_dist < -margin_outer : outer feather
    #   signed_dist < -margin_outer - fw  : deep background -> 1

    out = img_f.copy()
    fw = feather_width

    # Zone 1: flat interior (far inside the contour)
    out[signed_dist > (margin_inner + fw)] = 0.0

    # Zone 2: inner feather
    mask_fi = (signed_dist > margin_inner) & (signed_dist <= (margin_inner + fw))
    if np.any(mask_fi):
        # t goes from 1 (at margin_inner) to 0 (at margin_inner + fw)
        t = np.clip((signed_dist[mask_fi] - margin_inner) / fw, 0, 1)
        weight = 0.5 * (1 + np.cos(np.pi * t))  # 1 at edge, 0 at interior
        out[mask_fi] = weight * img_f[mask_fi]

    # Zone 3: transition zone — untouched (between -margin_outer and +margin_inner)

    # Zone 4: outer feather
    mask_fo = (signed_dist < -margin_outer) & (signed_dist >= -(margin_outer + fw))
    if np.any(mask_fo):
        # t goes from 0 (at -margin_outer) to 1 (at -(margin_outer + fw))
        t = np.clip((-signed_dist[mask_fo] - margin_outer) / fw, 0, 1)
        weight = 0.5 * (1 - np.cos(np.pi * t))
        out[mask_fo] = img_f[mask_fo] + weight * (1.0 - img_f[mask_fo])

    # Zone 5: flat background (far outside the contour)
    out[signed_dist < -(margin_outer + fw)] = 1.0

    return out


# =============================================================================
# Measurement helpers
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
    k = make_kernel(sigma)
    return cv2.filter2D(image, -1, k, borderType=cv2.BORDER_REPLICATE)


def measure_erf_sigma(image):
    centre, radius = detect_sphere(image)
    if centre is None:
        return float('nan'), float('nan')
    result = measure_blur_erf(image, centre, radius, num_rays=36)
    return result.sigma, result.confidence


def measure_interior(image, cx, cy, radius, fraction=0.5):
    """Returns (variance, max_intensity) in deep interior."""
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist < (radius * fraction)
    if not np.any(mask):
        return float('nan'), float('nan')
    region = image[mask]
    return float(np.var(region)), float(np.max(region))


def measure_contamination_profile(image, cx, cy, radius, ring_width=2):
    """
    Measure how far interior texture and exterior gradient extend from the edge.

    Scans annular rings inward and outward from the detected edge.
    For each ring, computes intensity variance. High variance = contamination present.

    Returns:
        (inner_contamination_px, outer_contamination_px, profile_data)

    inner_contamination_px: how many pixels inside the edge the texture extends
    outer_contamination_px: how many pixels outside the edge the gradient extends

    Contamination is defined as variance > 2x the variance of the quietest ring.
    """
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)

    max_scan = min(radius - 5, 60)  # don't scan past center
    profile = []

    # Scan from edge inward (positive = inside sphere)
    for d in range(0, int(max_scan), ring_width):
        inner_r = radius - d - ring_width
        outer_r = radius - d
        if inner_r <= 0:
            break
        ring_mask = (dist >= inner_r) & (dist < outer_r)
        if np.sum(ring_mask) < 10:
            continue
        ring_var = float(np.var(image[ring_mask]))
        ring_mean = float(np.mean(image[ring_mask]))
        profile.append({
            'direction': 'inner',
            'distance_from_edge': d + ring_width // 2,
            'variance': ring_var,
            'mean_intensity': ring_mean,
        })

    # Scan from edge outward (positive = outside sphere)
    for d in range(0, 60, ring_width):
        inner_r = radius + d
        outer_r = radius + d + ring_width
        ring_mask = (dist >= inner_r) & (dist < outer_r)
        if np.sum(ring_mask) < 10:
            continue
        ring_var = float(np.var(image[ring_mask]))
        ring_mean = float(np.mean(image[ring_mask]))
        profile.append({
            'direction': 'outer',
            'distance_from_edge': d + ring_width // 2,
            'variance': ring_var,
            'mean_intensity': ring_mean,
        })

    if not profile:
        return 0, 0, []

    # Find baseline variance (quietest ring, far from edge)
    inner_vars = [p['variance'] for p in profile if p['direction'] == 'inner' and p['distance_from_edge'] > 20]
    outer_vars = [p['variance'] for p in profile if p['direction'] == 'outer' and p['distance_from_edge'] > 20]

    inner_baseline = min(inner_vars) if inner_vars else 0.001
    outer_baseline = min(outer_vars) if outer_vars else 0.001

    # Contamination = variance > 2x baseline
    inner_contam = 0
    for p in profile:
        if p['direction'] == 'inner' and p['variance'] > 2 * inner_baseline:
            inner_contam = max(inner_contam, p['distance_from_edge'])

    outer_contam = 0
    for p in profile:
        if p['direction'] == 'outer' and p['variance'] > 2 * outer_baseline:
            outer_contam = max(outer_contam, p['distance_from_edge'])

    return inner_contam, outer_contam, profile


# =============================================================================
# Visual output
# =============================================================================

def gray_to_bgr(img):
    if img.dtype != np.uint8:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def draw_detection(vis, det, color=(0, 255, 0)):
    """Draw circle or ellipse detection on visualization."""
    if det['method'] == 'circle':
        cv2.circle(vis, (det['cx'], det['cy']), det['radius'], color, 1)
    else:
        axes = (int(det['major']), int(det['minor']))
        cv2.ellipse(vis, (det['cx'], det['cy']), axes, det['angle'], 0, 360, color, 1)


def save_detection_comparison(cam, filename, img, det_circle, det_ellipse, det_contour, output_dir):
    """Save side-by-side detection comparison: circle vs ellipse vs contour."""
    vis_c = gray_to_bgr(img)
    vis_e = gray_to_bgr(img)
    vis_d = gray_to_bgr(img)

    if det_circle:
        cv2.circle(vis_c, (det_circle['cx'], det_circle['cy']),
                   det_circle['radius'], (0, 255, 0), 1)
        cv2.putText(vis_c, f"Circle R={det_circle['radius']}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    if det_ellipse:
        axes = (int(det_ellipse['major']), int(det_ellipse['minor']))
        cv2.ellipse(vis_e, (det_ellipse['cx'], det_ellipse['cy']),
                    axes, det_ellipse['angle'], 0, 360, (0, 255, 0), 1)
        cv2.putText(vis_e, f"Ellipse a={det_ellipse['major']:.1f} b={det_ellipse['minor']:.1f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(vis_e, f"ecc={det_ellipse['eccentricity']:.3f} ang={det_ellipse['angle']:.0f}",
                    (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    if det_contour:
        cv2.drawContours(vis_d, [det_contour['contour']], 0, (0, 255, 0), 1)
        cv2.circle(vis_d, (det_contour['cx'], det_contour['cy']), 3, (0, 0, 255), -1)
        cv2.putText(vis_d, f"Contour R_mean={det_contour['radius']}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(vis_d, f"centroid=({det_contour['cx']},{det_contour['cy']})", (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    combined = np.hstack([vis_c, vis_e, vis_d])
    bar = np.zeros((20, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{cam}/{filename}  |  Circle vs Ellipse vs Contour", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    combined = np.vstack([bar, combined])

    crop_dir = output_dir / f'{cam}_{filename.replace(".png", "")}'
    crop_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_dir / 'detection_comparison.png'), combined)


def save_flatten_comparison(cam, filename, img, detections, configs_results, output_dir):
    """Save panels comparing flattening approaches for one crop.
    Groups by approach type (CC, EC, EE, CD) so each gets its own row."""
    # Group configs by prefix
    groups = {}
    for label, flat_img, int_var, int_max in configs_results:
        # Extract approach from config name (everything before the margin description)
        if 'CircDet+CircFlat' in label:
            prefix = 'CircDet+CircFlat'
        elif 'EllipDet+CircFlat' in label:
            prefix = 'EllipDet+CircFlat'
        elif 'EllipDet+EllipFlat' in label:
            prefix = 'EllipDet+EllipFlat'
        elif 'ContourDistMap' in label:
            prefix = 'ContourDistMap'
        else:
            prefix = 'Other'
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((label, flat_img, int_var, int_max))

    approach_names = {
        'CC': 'Circle Detect + Circle Flatten',
        'EC': 'Ellipse Detect + Circle Flatten',
        'EE': 'Ellipse Detect + Ellipse Flatten',
        'CD': 'Contour Detect + DistMap Flatten',
    }

    for prefix, items in groups.items():
        # Show ALL configs in a grid layout (8 per row)
        max_per_row = 8
        img_size = img.shape[0]

        rows = []
        for row_start in range(0, len(items), max_per_row):
            row_items = items[row_start:row_start + max_per_row]
            panels = []

            # Raw image at start of first row only
            if row_start == 0:
                raw_panel = gray_to_bgr(img)
                cv2.putText(raw_panel, "RAW (no flatten)", (3, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1)
                panels.append(raw_panel)

            for label, flat_img, int_var, int_max in row_items:
                vis = gray_to_bgr(flat_img)
                # Get margin description (strip approach prefix)
                short_label = label
                for p in approach_names.keys():
                    short_label = short_label.replace(p + ' ', '')
                # Split long labels across lines
                words = short_label.split(' ')
                line1 = ' '.join(words[:3])
                line2 = ' '.join(words[3:]) if len(words) > 3 else ''
                cv2.putText(vis, line1, (3, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
                if line2:
                    cv2.putText(vis, line2, (3, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

                # Interior verdict at bottom
                clean = "CLEAN" if int_max < 0.01 else f"max={int_max:.2f}"
                cv2.putText(vis, clean, (3, img_size - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
                panels.append(vis)

            row_img = np.hstack(panels)
            rows.append(row_img)

        # Pad rows to same width
        max_width = max(r.shape[1] for r in rows)
        padded_rows = []
        for r in rows:
            if r.shape[1] < max_width:
                pad = np.zeros((r.shape[0], max_width - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded_rows.append(r)

        combined = np.vstack(padded_rows)

        # Title bar
        bar = np.zeros((28, combined.shape[1], 3), dtype=np.uint8)
        det_info = detections.get('ellipse', detections.get('circle', detections.get('contour', {})))
        r_val = det_info.get('radius', '?')
        title = f"{cam}/{filename}  |  {approach_names.get(prefix, prefix)}  |  R~{r_val}px  |  {len(items)} configs shown"
        cv2.putText(bar, title, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        combined = np.vstack([bar, combined])

        crop_dir = output_dir / f'{cam}_{filename.replace(".png", "")}'
        crop_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_dir / f'flatten_{prefix}.png'), combined)


# =============================================================================
# Load crops
# =============================================================================

def load_crops_by_camera(max_per_camera=4):
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
# Margin configs
# =============================================================================

@dataclass
class MarginConfig:
    name: str
    det_source: str  # 'circle' or 'ellipse' — which detection to use for geometry
    flat_geom: str   # 'circle' or 'ellipse' — shape of flattening zones
    is_prop: bool
    px_in: int       # fixed px, or floor for hybrid
    px_out: int
    frac_in: float   # fraction of radius (or mean radius for ellipse)
    frac_out: float

    def compute_margins(self, radius: int) -> Tuple[int, int]:
        if self.is_prop:
            m_in = max(int(radius * self.frac_in), self.px_in)
            m_out = max(int(radius * self.frac_out), self.px_out)
        else:
            m_in = self.px_in
            m_out = self.px_out
        return m_in, m_out


def get_configs() -> List[MarginConfig]:
    """
    Build all margin configurations to test.

    Naming convention:
      [Approach]_[Margin type][Value]

    Approaches:
      CircDet+CircFlat  = Detect sphere as circle, flatten with circular zones (CURRENT PIPELINE)
      EllipDet+CircFlat = Detect as ellipse, flatten with circular zones (mean radius)
      EllipDet+EllipFlat = Detect as ellipse, flatten with elliptical zones
      ContourDistMap    = Detect actual edge contour, flatten by distance to real edge

    Margin types:
      Fixed [N]px       = Fixed pixel margin (same for all sphere sizes)
      [N]% of radius    = Proportional to sphere size (scales with camera)
      [N]% min [M]px    = Proportional but with a minimum floor (hybrid)
      Inner[N]% Outer[M]% = Different margins for inside vs outside the edge
    """
    configs = []

    # === Circle detection + Circle flattening (current pipeline) ===
    # Fixed symmetric
    for px in [15, 25, 35, 50]:
        configs.append(MarginConfig(
            f'CircDet+CircFlat Fixed {px}px',
            'circle', 'circle', False, px, px, 0, 0))
    # Fixed asymmetric
    for px_in, px_out in [(10, 25), (10, 35), (15, 35), (15, 50), (20, 40), (25, 50)]:
        configs.append(MarginConfig(
            f'CircDet+CircFlat Fixed In{px_in}px Out{px_out}px',
            'circle', 'circle', False, px_in, px_out, 0, 0))
    # Proportional symmetric
    for frac in [0.05, 0.08, 0.10, 0.14, 0.20, 0.30]:
        configs.append(MarginConfig(
            f'CircDet+CircFlat {int(frac*100)}% of radius',
            'circle', 'circle', True, 0, 0, frac, frac))
    # Hybrid (proportional + minimum floor)
    for frac, floor in [(0.14, 10), (0.14, 15), (0.14, 20), (0.14, 25), (0.20, 10), (0.20, 15), (0.20, 20)]:
        configs.append(MarginConfig(
            f'CircDet+CircFlat {int(frac*100)}% min {floor}px',
            'circle', 'circle', True, floor, floor, frac, frac))
    # Proportional asymmetric (small inner = aggressive caustic removal)
    for inf, outf in [(0.05, 0.14), (0.05, 0.20), (0.08, 0.14), (0.08, 0.20), (0.10, 0.14), (0.10, 0.20)]:
        configs.append(MarginConfig(
            f'CircDet+CircFlat In{int(inf*100)}% Out{int(outf*100)}%',
            'circle', 'circle', True, 0, 0, inf, outf))

    # === Ellipse detection + Circle flattening ===
    for px in [15, 25, 35, 50]:
        configs.append(MarginConfig(
            f'EllipDet+CircFlat Fixed {px}px',
            'ellipse', 'circle', False, px, px, 0, 0))
    for px_in, px_out in [(10, 25), (10, 35), (15, 35), (15, 50), (20, 40), (25, 50)]:
        configs.append(MarginConfig(
            f'EllipDet+CircFlat Fixed In{px_in}px Out{px_out}px',
            'ellipse', 'circle', False, px_in, px_out, 0, 0))
    for frac in [0.05, 0.08, 0.10, 0.14, 0.20, 0.30]:
        configs.append(MarginConfig(
            f'EllipDet+CircFlat {int(frac*100)}% of radius',
            'ellipse', 'circle', True, 0, 0, frac, frac))
    for frac, floor in [(0.14, 10), (0.14, 15), (0.14, 20), (0.14, 25), (0.20, 10), (0.20, 15), (0.20, 20)]:
        configs.append(MarginConfig(
            f'EllipDet+CircFlat {int(frac*100)}% min {floor}px',
            'ellipse', 'circle', True, floor, floor, frac, frac))
    for inf, outf in [(0.05, 0.14), (0.05, 0.20), (0.08, 0.14), (0.08, 0.20), (0.10, 0.14), (0.10, 0.20)]:
        configs.append(MarginConfig(
            f'EllipDet+CircFlat In{int(inf*100)}% Out{int(outf*100)}%',
            'ellipse', 'circle', True, 0, 0, inf, outf))

    # === Ellipse detection + Ellipse flattening ===
    for px in [15, 25, 35, 50]:
        configs.append(MarginConfig(
            f'EllipDet+EllipFlat Fixed {px}px',
            'ellipse', 'ellipse', False, px, px, 0, 0))
    for px_in, px_out in [(10, 25), (10, 35), (15, 35), (15, 50), (20, 40), (25, 50)]:
        configs.append(MarginConfig(
            f'EllipDet+EllipFlat Fixed In{px_in}px Out{px_out}px',
            'ellipse', 'ellipse', False, px_in, px_out, 0, 0))
    for frac in [0.05, 0.08, 0.10, 0.14, 0.20, 0.30]:
        configs.append(MarginConfig(
            f'EllipDet+EllipFlat {int(frac*100)}% of radius',
            'ellipse', 'ellipse', True, 0, 0, frac, frac))
    for frac, floor in [(0.14, 10), (0.14, 15), (0.14, 20), (0.14, 25), (0.20, 10), (0.20, 15), (0.20, 20)]:
        configs.append(MarginConfig(
            f'EllipDet+EllipFlat {int(frac*100)}% min {floor}px',
            'ellipse', 'ellipse', True, floor, floor, frac, frac))
    for inf, outf in [(0.05, 0.14), (0.05, 0.20), (0.08, 0.14), (0.08, 0.20), (0.10, 0.14), (0.10, 0.20)]:
        configs.append(MarginConfig(
            f'EllipDet+EllipFlat In{int(inf*100)}% Out{int(outf*100)}%',
            'ellipse', 'ellipse', True, 0, 0, inf, outf))

    # === Contour detection + Distance-map flattening ===
    for px in [15, 25, 35, 50]:
        configs.append(MarginConfig(
            f'ContourDistMap Fixed {px}px',
            'contour', 'distmap', False, px, px, 0, 0))
    for px_in, px_out in [(10, 25), (10, 35), (15, 35), (15, 50), (20, 40), (25, 50)]:
        configs.append(MarginConfig(
            f'ContourDistMap Fixed In{px_in}px Out{px_out}px',
            'contour', 'distmap', False, px_in, px_out, 0, 0))
    for frac in [0.05, 0.08, 0.10, 0.14, 0.20, 0.30]:
        configs.append(MarginConfig(
            f'ContourDistMap {int(frac*100)}% of radius',
            'contour', 'distmap', True, 0, 0, frac, frac))
    for frac, floor in [(0.14, 10), (0.14, 15), (0.14, 20), (0.14, 25), (0.20, 10), (0.20, 15), (0.20, 20)]:
        configs.append(MarginConfig(
            f'ContourDistMap {int(frac*100)}% min {floor}px',
            'contour', 'distmap', True, floor, floor, frac, frac))
    for inf, outf in [(0.05, 0.14), (0.05, 0.20), (0.08, 0.14), (0.08, 0.20), (0.10, 0.14), (0.10, 0.20)]:
        configs.append(MarginConfig(
            f'ContourDistMap In{int(inf*100)}% Out{int(outf*100)}%',
            'contour', 'distmap', True, 0, 0, inf, outf))

    return configs


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE MARGIN & DETECTION INVESTIGATION v2")
    print("=" * 70)
    print(f"Model size: {MODEL_SIZE}x{MODEL_SIZE}")
    print()

    crops = load_crops_by_camera(max_per_camera=4)
    for cam in ('g', 'v', 'm'):
        print(f"  Camera {cam}: {len(crops[cam])} crops")
    print()

    configs = get_configs()
    sigma_targets = [2, 4, 6, 8, 10, 12]

    print(f"{len(configs)} configs x {sum(len(v) for v in crops.values())} crops x {len(sigma_targets)} sigmas")
    print()

    results = []
    detection_log = []

    for cam in ('g', 'v', 'm'):
        for crop_info in crops[cam]:
            img = crop_info['image']  # float32 [0,1], 256x256
            fn = crop_info['filename']

            # --- Run all three detection methods ---
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

            # Measure contamination profile on raw image
            det_for_contam = det_c if det_c else (det_e if det_e else det_d)
            inner_contam, outer_contam, contam_profile = measure_contamination_profile(
                img, det_for_contam['cx'], det_for_contam['cy'], det_for_contam['radius'])

            # Log detection comparison
            det_entry = {
                'camera': cam, 'filename': fn,
                'circ_cx': det_c['cx'] if det_c else None,
                'circ_cy': det_c['cy'] if det_c else None,
                'circ_r': det_c['radius'] if det_c else None,
                'ellip_cx': det_e['cx'] if det_e else None,
                'ellip_cy': det_e['cy'] if det_e else None,
                'ellip_major': det_e['major'] if det_e else None,
                'ellip_minor': det_e['minor'] if det_e else None,
                'ellip_angle': det_e['angle'] if det_e else None,
                'eccentricity': det_e['eccentricity'] if det_e else None,
                'axis_diff_px': (det_e['major'] - det_e['minor']) if det_e else None,
                'contour_cx': det_d['cx'] if det_d else None,
                'contour_cy': det_d['cy'] if det_d else None,
                'contour_r_mean': det_d['radius'] if det_d else None,
                'inner_contamination_px': inner_contam,
                'outer_contamination_px': outer_contam,
            }
            detection_log.append(det_entry)

            ellip_str = f"ellip=({det_e['major']:.1f},{det_e['minor']:.1f})" if det_e else "ellip=N/A"
            print(f"\n  {cam}/{fn}: circ_R={r_circ}, {ellip_str}, "
                  f"ecc={ecc:.3f}, diff={axis_diff:.1f}px, contour_R={r_contour}")
            print(f"    Contamination: interior extends {inner_contam}px inward from edge, "
                  f"background extends {outer_contam}px outward")

            # Save detection visual
            save_detection_comparison(cam, fn, img, det_c, det_e, det_d, OUTPUT_DIR)

            # --- RAW BASELINE: run both blur tests on the unflattened image ---
            raw_native, _ = measure_erf_sigma(img)
            mean_r_raw = r_circ  # use circle radius for interior measurement
            raw_int_var, raw_int_max = measure_interior(img, det_c['cx'] if det_c else det_e['cx'],
                                                         det_c['cy'] if det_c else det_e['cy'],
                                                         mean_r_raw)

            for test_type, sigma_list_fn in [
                ('quadrature', lambda st: (np.sqrt(st**2 - raw_native**2) if (not np.isnan(raw_native) and st**2 > raw_native**2) else None, st)),
                ('fixed_kernel', lambda st: (st, np.sqrt(raw_native**2 + st**2) if not np.isnan(raw_native) else float('nan'))),
            ]:
                for sigma_target in sigma_targets:
                    if test_type == 'quadrature':
                        if np.isnan(raw_native) or sigma_target**2 <= raw_native**2:
                            continue
                        kernel_sigma = np.sqrt(sigma_target**2 - raw_native**2)
                        expected = sigma_target
                    else:
                        kernel_sigma = sigma_target
                        expected = np.sqrt(raw_native**2 + sigma_target**2) if not np.isnan(raw_native) else float('nan')

                    blurred = apply_blur(img, kernel_sigma)
                    measured, _ = measure_erf_sigma(blurred)
                    error_pct = 100 * (measured - expected) / expected if (not np.isnan(measured) and not np.isnan(expected) and expected > 0) else float('nan')

                    results.append({
                        'camera': cam, 'filename': fn,
                        'config': 'Raw (no flattening)', 'det_source': 'circle', 'flat_geom': 'none',
                        'radius': mean_r_raw, 'eccentricity': ecc,
                        'm_in': 0, 'm_out': 0,
                        'test_type': test_type,
                        'sigma_target': expected,
                        'kernel_applied': kernel_sigma,
                        'native_sigma': raw_native,
                        'measured': measured,
                        'error_pct': error_pct,
                        'int_var': raw_int_var, 'int_max': raw_int_max,
                    })

            raw_quad = [r['error_pct'] for r in results if r['config'] == 'Raw (no flattening)' and r['filename'] == fn and r['test_type'] == 'quadrature' and not np.isnan(r['error_pct'])]
            raw_fixed = [r['error_pct'] for r in results if r['config'] == 'Raw (no flattening)' and r['filename'] == fn and r['test_type'] == 'fixed_kernel' and not np.isnan(r['error_pct'])]
            raw_q_str = f"{np.mean(raw_quad):+.2f}%" if raw_quad else "N/A"
            raw_f_str = f"{np.mean(raw_fixed):+.2f}%" if raw_fixed else "N/A"
            raw_n_str = f"{raw_native:.2f}" if not np.isnan(raw_native) else "N/A"
            print(f"    {'Raw (no flattening)':>25s}  (baseline)                      |  "
                  f"native={raw_n_str}  quad={raw_q_str}  fixed={raw_f_str}  |  "
                  f"interior: DIRTY (max={raw_int_max:.3f})")

            flatten_visuals = []

            for cfg in configs:
                # Determine which detection to use for geometry
                if cfg.det_source == 'ellipse' and det_e is None:
                    continue
                if cfg.det_source == 'circle' and det_c is None:
                    continue
                if cfg.det_source == 'contour' and det_d is None:
                    continue

                if cfg.det_source == 'contour':
                    det = det_d
                elif cfg.det_source == 'ellipse':
                    det = det_e
                else:
                    det = det_c

                mean_r = int((det['major'] + det['minor']) / 2)
                m_in, m_out = cfg.compute_margins(mean_r)

                # Skip if margin eats past center
                if mean_r - m_in <= 0:
                    continue

                # Flatten — geometry of zones depends on flat_geom
                if cfg.flat_geom == 'distmap':
                    # Contour distance-map flattening
                    flat = flatten_contour_distmap(
                        img, det['contour'], m_in, m_out)
                elif cfg.flat_geom == 'ellipse':
                    if det_e is None:
                        continue
                    flat = flatten_elliptical(
                        img, det['cx'], det['cy'],
                        det['major'], det['minor'], det['angle'],
                        m_in, m_out)
                else:
                    # Circular flattening
                    flat = flatten_circular(
                        img, det['cx'], det['cy'], mean_r,
                        m_in, m_out)

                # Measure interior cleanliness
                int_var, int_max = measure_interior(flat, det['cx'], det['cy'], mean_r)

                # Measure native ERF sigma on flattened crop
                native_sigma, _ = measure_erf_sigma(flat)

                # Collect for visual comparison
                flatten_visuals.append((cfg.name, flat, int_var, int_max))

                # ---- TEST A: Quadrature blur (realistic pipeline test) ----
                # Applies blur so total sigma = target, accounting for native blur
                for sigma_target in sigma_targets:
                    if np.isnan(native_sigma):
                        results.append({
                            'camera': cam, 'filename': fn,
                            'config': cfg.name, 'det_source': cfg.det_source, 'flat_geom': cfg.flat_geom,
                            'radius': mean_r, 'eccentricity': ecc,
                            'm_in': m_in, 'm_out': m_out,
                            'test_type': 'quadrature',
                            'sigma_target': sigma_target,
                            'kernel_applied': float('nan'),
                            'native_sigma': float('nan'),
                            'measured': float('nan'),
                            'error_pct': float('nan'),
                            'int_var': int_var, 'int_max': int_max,
                        })
                        continue

                    kernel_sq = sigma_target**2 - native_sigma**2
                    if kernel_sq <= 0:
                        continue

                    kernel_sigma = np.sqrt(kernel_sq)
                    blurred = apply_blur(flat, kernel_sigma)
                    measured, _ = measure_erf_sigma(blurred)
                    error_pct = 100 * (measured - sigma_target) / sigma_target if not np.isnan(measured) else float('nan')

                    results.append({
                        'camera': cam, 'filename': fn,
                        'config': cfg.name, 'det_source': cfg.det_source, 'flat_geom': cfg.flat_geom,
                        'radius': mean_r, 'eccentricity': ecc,
                        'm_in': m_in, 'm_out': m_out,
                        'test_type': 'quadrature',
                        'sigma_target': sigma_target,
                        'kernel_applied': kernel_sigma,
                        'native_sigma': native_sigma,
                        'measured': measured,
                        'error_pct': error_pct,
                        'int_var': int_var, 'int_max': int_max,
                    })

                # ---- TEST B: Fixed kernel (cross-comparable between configs) ----
                # Applies the exact same Gaussian kernel to every config
                # so the only variable is the flattening approach
                for kernel_sigma in sigma_targets:
                    blurred = apply_blur(flat, kernel_sigma)
                    measured, _ = measure_erf_sigma(blurred)
                    # Expected total sigma = sqrt(native^2 + kernel^2) if native is known
                    if not np.isnan(native_sigma):
                        expected = np.sqrt(native_sigma**2 + kernel_sigma**2)
                    else:
                        expected = float('nan')
                    error_pct = 100 * (measured - expected) / expected if (not np.isnan(measured) and not np.isnan(expected) and expected > 0) else float('nan')

                    results.append({
                        'camera': cam, 'filename': fn,
                        'config': cfg.name, 'det_source': cfg.det_source, 'flat_geom': cfg.flat_geom,
                        'radius': mean_r, 'eccentricity': ecc,
                        'm_in': m_in, 'm_out': m_out,
                        'test_type': 'fixed_kernel',
                        'sigma_target': expected,  # expected total sigma
                        'kernel_applied': kernel_sigma,
                        'native_sigma': native_sigma,
                        'measured': measured,
                        'error_pct': error_pct,
                        'int_var': int_var, 'int_max': int_max,
                    })

                # Print per-config summary
                quad_res = [r for r in results if r['config'] == cfg.name and r['filename'] == fn and r['test_type'] == 'quadrature']
                fixed_res = [r for r in results if r['config'] == cfg.name and r['filename'] == fn and r['test_type'] == 'fixed_kernel']
                quad_errs = [r['error_pct'] for r in quad_res if not np.isnan(r['error_pct'])]
                fixed_errs = [r['error_pct'] for r in fixed_res if not np.isnan(r['error_pct'])]

                # Interior cleanliness verdict
                if int_max < 0.01:
                    clean_verdict = "CLEAN"
                elif int_max < 0.1:
                    clean_verdict = "mostly clean"
                elif int_max < 0.5:
                    clean_verdict = "some texture"
                else:
                    clean_verdict = "DIRTY"

                quad_str = f"{np.mean(quad_errs):+.2f}%" if quad_errs else "N/A"
                fixed_str = f"{np.mean(fixed_errs):+.2f}%" if fixed_errs else "N/A"
                native_str = f"{native_sigma:.2f}" if not np.isnan(native_sigma) else "N/A"

                print(f"    {cfg.name:>25s}  margin=({m_in:>2d}px in, {m_out:>2d}px out)  |  "
                      f"native={native_str}  quad={quad_str}  fixed={fixed_str}  |  "
                      f"interior: {clean_verdict} (max={int_max:.3f})")

            # Save flatten comparison visuals (one per approach type)
            if flatten_visuals:
                det_dict = {}
                if det_c: det_dict['circle'] = det_c
                if det_e: det_dict['ellipse'] = det_e
                if det_d: det_dict['contour'] = det_d
                save_flatten_comparison(cam, fn, img, det_dict, flatten_visuals, OUTPUT_DIR)

    # ==========================================================================
    # Save results
    # ==========================================================================

    import pandas as pd

    # Detection log
    det_df = pd.DataFrame(detection_log)
    det_df.to_csv(OUTPUT_DIR / 'detection_comparison.csv', index=False)
    print(f"\nDetection CSV: {OUTPUT_DIR / 'detection_comparison.csv'}")

    if not results:
        print("No results to analyse.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'margin_v2_results.csv', index=False)
    print(f"Results CSV: {OUTPUT_DIR / 'margin_v2_results.csv'}")

    # ==========================================================================
    # Summary analysis
    # ==========================================================================

    print()
    print("=" * 90)
    print("SPHERE DETECTION: How does each method see the sphere?")
    print("=" * 90)
    for cam in sorted(det_df['camera'].unique()):
        cd_df = det_df[det_df['camera'] == cam]
        print(f"\n  Camera {cam}:")
        print(f"    Circle (minEnclosingCircle):  R = {cd_df['circ_r'].min()}-{cd_df['circ_r'].max()} px")
        if cd_df['ellip_major'].notna().any():
            print(f"    Ellipse (fitEllipse):         major = {cd_df['ellip_major'].min():.1f}-{cd_df['ellip_major'].max():.1f} px, "
                  f"minor = {cd_df['ellip_minor'].min():.1f}-{cd_df['ellip_minor'].max():.1f} px")
            print(f"                                  axis difference = {cd_df['axis_diff_px'].min():.1f}-{cd_df['axis_diff_px'].max():.1f} px "
                  f"(how non-circular the sphere is)")
            print(f"                                  eccentricity = {cd_df['eccentricity'].min():.3f}-{cd_df['eccentricity'].max():.3f} "
                  f"(1.0 = perfect circle)")
        if cd_df['contour_r_mean'].notna().any():
            print(f"    Contour (centroid+mean dist): R_mean = {cd_df['contour_r_mean'].min()}-{cd_df['contour_r_mean'].max()} px")

    det_labels = {'circle': 'Circle', 'ellipse': 'Ellipse', 'contour': 'Contour'}
    flat_labels = {'circle': 'Circle', 'ellipse': 'Ellipse', 'distmap': 'DistMap'}

    for test_type, test_desc in [('quadrature', 'QUADRATURE TEST (realistic pipeline)'),
                                  ('fixed_kernel', 'FIXED KERNEL TEST (cross-comparable)')]:
        test_df = df[df['test_type'] == test_type]
        if len(test_df) == 0:
            continue

        print()
        print("=" * 110)
        print(f"BLUR MEASUREMENT ACCURACY -- {test_desc}")
        if test_type == 'quadrature':
            print("  Each config gets its own native blur subtracted, so total blur should equal target.")
            print("  This tests the real pipeline scenario: 'if I train with this flattening, is my blur label correct?'")
        else:
            print("  The exact same Gaussian kernel is applied to every config.")
            print("  This isolates the flattening effect: 'given identical blur, does this config distort the reading?'")
        print("  Closer to 0% = more accurate. Positive = overestimates blur.")
        print("=" * 110)
        print()
        print(f"  {'Config':>27s}  {'Detect':>7s} {'Flatten':>7s}  ", end='')
        for cam in ('g', 'v', 'm'):
            print(f"  {'cam '+cam:>9s}", end='')
        print(f"  {'Overall':>9s}  {'Interior':>8s}  {'Verdict':>10s}")
        print("  " + "-" * 105)

        for cfg_name in dict.fromkeys(r['config'] for r in results):
            cfg_df = test_df[test_df['config'] == cfg_name]
            if len(cfg_df) == 0:
                continue
            det_src = cfg_df['det_source'].iloc[0]
            flat_gm = cfg_df['flat_geom'].iloc[0]

            parts = []
            for cam in ('g', 'v', 'm'):
                cam_cfg = cfg_df[cfg_df['camera'] == cam]
                valid = cam_cfg['error_pct'].dropna()
                parts.append(f"{valid.mean():>+8.2f}%" if len(valid) > 0 else f"{'N/A':>9s}")

            all_valid = cfg_df['error_pct'].dropna()
            all_mean_val = all_valid.mean() if len(all_valid) > 0 else float('nan')
            all_mean_str = f"{all_mean_val:>+8.2f}%" if not np.isnan(all_mean_val) else f"{'N/A':>9s}"
            int_max = cfg_df['int_max'].mean()

            # Verdict
            if np.isnan(all_mean_val):
                verdict = "NO DATA"
            elif abs(all_mean_val) < 2 and int_max < 0.01:
                verdict = "EXCELLENT"
            elif abs(all_mean_val) < 5 and int_max < 0.05:
                verdict = "GOOD"
            elif abs(all_mean_val) < 10 and int_max < 0.1:
                verdict = "OK"
            elif int_max > 0.3:
                verdict = "DIRTY"
            else:
                verdict = "POOR"

            print(f"  {cfg_name:>27s}  {det_labels[det_src]:>7s} {flat_labels[flat_gm]:>7s}  "
                  f"{'  '.join(parts)}  {all_mean_str}  {int_max:>8.4f}  {verdict:>10s}")

    # Head-to-head for each test type
    for test_type, test_label in [('quadrature', 'Quadrature (pipeline-realistic)'),
                                   ('fixed_kernel', 'Fixed Kernel (cross-comparable)')]:
        test_df = df[df['test_type'] == test_type]
        if len(test_df) == 0:
            continue

        print()
        print("=" * 115)
        print(f"HEAD-TO-HEAD -- {test_label}")
        print()
        print("  CircCirc   = Circle detect + Circle flatten (current pipeline)")
        print("  EllipCirc  = Ellipse detect + Circle flatten")
        print("  EllipEllip = Ellipse detect + Ellipse flatten")
        print("  ContDist   = Contour detect + Distance-map flatten (shape-agnostic)")
        print()
        print("  blur_err = how much flattening distorts the blur reading (0% = perfect)")
        print("  int = max brightness inside sphere (0 = interior fully cleaned)")
        print("=" * 115)
        print(f"  {'Margin strategy':>25s}  {'CircCirc':>11s} {'int':>5s}  "
              f"{'EllipCirc':>11s} {'int':>5s}  "
              f"{'EllipEllip':>11s} {'int':>5s}  "
              f"{'ContDist':>11s} {'int':>5s}")
        print("  " + "-" * 115)

        prefix_map = {}
        approach_prefixes = ['CircDet+CircFlat', 'EllipDet+CircFlat', 'EllipDet+EllipFlat', 'ContourDistMap']
        short_labels = {'CircDet+CircFlat': 'CircCirc', 'EllipDet+CircFlat': 'EllipCirc',
                        'EllipDet+EllipFlat': 'EllipEllip', 'ContourDistMap': 'ContDist'}
        for prefix in approach_prefixes:
            prefix_map[prefix] = {c.replace(f'{prefix} ', ''): c
                                  for c in test_df['config'].unique() if c.startswith(prefix)}

        all_suffixes = sorted(set(
            s for d in prefix_map.values() for s in d.keys()
        ))

        for suffix in all_suffixes:
            parts = []
            for prefix in approach_prefixes:
                cfg_name = prefix_map[prefix].get(suffix)
                if cfg_name and cfg_name in test_df['config'].values:
                    cfg_df = test_df[test_df['config'] == cfg_name]
                    valid = cfg_df['error_pct'].dropna()
                    im = cfg_df['int_max'].mean()
                    err_str = f"{valid.mean():>+10.2f}%" if len(valid) > 0 else f"{'N/A':>11s}"
                    im_str = f"{im:>5.3f}"
                else:
                    err_str = f"{'---':>11s}"
                    im_str = f"{'---':>5s}"
                parts.append(f"{err_str} {im_str}")
            print(f"  {suffix:>20s}  {'  '.join(parts)}")

    # Per-sigma breakdown (quadrature only — this is the pipeline-relevant one)
    quad_df = df[df['test_type'] == 'quadrature']
    if len(quad_df) > 0:
        print()
        print("=" * 90)
        print("PER-SIGMA BREAKDOWN (Quadrature Test)")
        print("  Shows blur accuracy at each defocus level, broken down by camera.")
        print("  This tells you: at what blur levels does each config start to struggle?")
        print("=" * 90)
        top_configs = [
            'CircDet+CircFlat Fixed 50px',
            'CircDet+CircFlat 14% min 20px',
            'CircDet+CircFlat 20% min 15px',
            'EllipDet+CircFlat 14% min 20px',
            'EllipDet+CircFlat 20% min 15px',
            'EllipDet+EllipFlat 14% of radius',
            'EllipDet+EllipFlat 14% min 20px',
            'EllipDet+EllipFlat 20% min 15px',
            'ContourDistMap 14% of radius',
            'ContourDistMap 14% min 20px',
            'ContourDistMap 20% min 15px',
        ]
        for cfg_name in top_configs:
            cfg_df = quad_df[quad_df['config'] == cfg_name]
            if len(cfg_df) == 0:
                continue
            det_src = cfg_df['det_source'].iloc[0]
            flat_gm = cfg_df['flat_geom'].iloc[0]
            print(f"\n  {cfg_name} ({det_labels[det_src]} detect, {flat_labels[flat_gm]} flatten):")
            for st in sigma_targets:
                st_df = cfg_df[cfg_df['sigma_target'] == st]
                valid = st_df['error_pct'].dropna()
                if len(valid) > 0:
                    per_cam = []
                    for cam in ('g', 'v', 'm'):
                        cv = st_df[st_df['camera'] == cam]['error_pct'].dropna()
                        if len(cv) > 0:
                            per_cam.append(f"{cam}={cv.mean():+.1f}%")
                    print(f"    target blur={st:>2d}px:  error={valid.mean():+.2f}%  ({', '.join(per_cam)})")

    # Fixed kernel cross-comparison
    fixed_df = df[df['test_type'] == 'fixed_kernel']
    if len(fixed_df) > 0:
        print()
        print("=" * 90)
        print("FIXED KERNEL CROSS-COMPARISON")
        print("  Same kernel applied to all configs. Differences are purely from flattening.")
        print("  If two configs show different errors here, flattening is the cause.")
        print("=" * 90)
        # Show kernel=6 as representative
        for kernel in [4, 8]:
            k_df = fixed_df[fixed_df['kernel_applied'] == kernel]
            if len(k_df) == 0:
                continue
            print(f"\n  Kernel = {kernel}px:")
            for cfg_name in top_configs:
                cfg_df = k_df[k_df['config'] == cfg_name]
                if len(cfg_df) == 0:
                    continue
                valid = cfg_df['error_pct'].dropna()
                measured = cfg_df['measured'].dropna()
                if len(valid) > 0:
                    per_cam = []
                    for cam in ('g', 'v', 'm'):
                        cv = cfg_df[cfg_df['camera'] == cam]['measured'].dropna()
                        if len(cv) > 0:
                            per_cam.append(f"{cam} measured={cv.mean():.2f}px")
                    print(f"    {cfg_name:>27s}:  error={valid.mean():+.2f}%  ({', '.join(per_cam)})")

    print(f"\nAll results and visual panels saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
