"""
Sphere Crop Flattening Investigation Script

Three investigations:
1. Margin determination -- Otsu detection error vs blur level
2. Visual verification -- before/after flattening panels
3. ERF improvement -- systematic bias with and without flattening

Run with: phantom_env\Scripts\python.exe investigate_flattening.py
"""

import sys
import os
import csv
import math
import numpy as np
import cv2
from pathlib import Path
from io import StringIO

# Add calibration module to path
CALIB_DIR = str(Path(__file__).parent.parent.parent / 'calibration')
if CALIB_DIR not in sys.path:
    sys.path.insert(0, CALIB_DIR)

from blur_measurement import detect_sphere, measure_blur_erf, erf_edge
from sphere_processing import flatten_sphere_crop
from scipy.optimize import curve_fit
from scipy.special import erf as scipy_erf

from paths_config import CROP_BASE, CALIB_IMG_DIR, CALIB_CSV, SHARP_CSV
OUTPUT_DIR = Path(__file__).parent / 'investigation_output'
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_SIZE = 256
CROP_SIZE = 299
MARGIN = 50
FEATHER = 10


# -- Logger (tee to console + file) ----------------------------------------

class TeeLogger:
    """Writes to both console and a string buffer for saving to file."""
    def __init__(self):
        self.buf = StringIO()
        self.lines = []

    def log(self, msg=""):
        print(msg)
        self.buf.write(msg + "\n")
        self.lines.append(msg)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.buf.getvalue())

LOG = TeeLogger()


# -- Shared Utilities ------------------------------------------------------

def make_kernel(sigma, rf=4.0):
    """Create 2D Gaussian kernel."""
    if sigma <= 0:
        return np.array([[1.0]], dtype=np.float32)
    r = int(np.ceil(rf * sigma))
    s = 2 * r + 1
    ax = np.arange(s) - r
    X, Y = np.meshgrid(ax, ax)
    k = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    k /= k.sum()
    return k.astype(np.float32)


def apply_blur(image, sigma, rf=4.0):
    """Apply Gaussian blur to image."""
    if sigma <= 0.5:
        return image.copy()
    k = make_kernel(sigma, rf)
    return cv2.filter2D(image, -1, k, borderType=cv2.BORDER_REPLICATE)


def load_calib_data():
    """Load calibration images and measurements CSV."""
    measurements = {}
    with open(CALIB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row['source_filename'].replace('.cine', '.png')
            measurements[fn] = {
                'z_mm': float(row['defocus_z_mm']),
                'sigma_px': float(row['sigma_px']),
                'is_focus': row['is_focus_frame'].strip().lower() == 'true',
            }
    images = {}
    for png in sorted(CALIB_IMG_DIR.glob('*.png')):
        if png.name in measurements:
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[png.name] = img.astype(np.float32) / 255.0
    return images, measurements


def load_v_crops_with_metadata():
    """Load V-camera crop metadata from sharp_crops.csv."""
    crops = []
    with open(SHARP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['camera'] != 'v':
                continue
            folder = row['folder']
            fn = row['filename']
            path = CROP_BASE / folder / 'v' / 'crops' / fn
            if path.exists() and row.get('scale_px_per_mm', '').strip() and row.get('native_blur_sigma', '').strip():
                crops.append({
                    'path': path,
                    'filename': fn,
                    'diameter_px': int(float(row['diameter_px'])),
                    'diameter_model': round(int(float(row['diameter_px'])) * MODEL_SIZE / CROP_SIZE),
                    'scale_px_per_mm': float(row['scale_px_per_mm']),
                    'native_blur': float(row['native_blur_sigma']),
                })
    return crops


def measure_erf_sigma(image, num_rays=36):
    """Measure blur sigma via ERF fitting. Returns (sigma, r2) or (nan, nan)."""
    centre, radius = detect_sphere(image)
    if centre is None:
        return float('nan'), float('nan')
    result = measure_blur_erf(image, centre, radius, num_rays=num_rays)
    return result.sigma, result.confidence


def extract_radial_profile(image, centre, radius, num_rays=36):
    """Extract averaged radial intensity profile using bilinear interpolation.

    Returns (r_values, mean_intensities) where r_values is distance from centre.
    Samples from 0 to min(radius + 80, image boundary) in 0.5px steps.
    """
    cx, cy = centre
    h, w = image.shape[:2]
    max_r = min(radius + 80, cx, cy, w - 1 - cx, h - 1 - cy)
    r_values = np.arange(0, max_r, 0.5)
    all_intensities = []

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Check how far this ray can go before hitting image edge
        ray_max = max_r
        if cos_a > 0:
            ray_max = min(ray_max, (w - 1 - cx) / cos_a)
        elif cos_a < 0:
            ray_max = min(ray_max, -cx / cos_a)
        if sin_a > 0:
            ray_max = min(ray_max, (h - 1 - cy) / sin_a)
        elif sin_a < 0:
            ray_max = min(ray_max, -cy / sin_a)

        ray_r = r_values[r_values < ray_max]
        xs = cx + ray_r * cos_a
        ys = cy + ray_r * sin_a

        # Bilinear interpolation
        x0 = np.floor(xs).astype(int)
        y0 = np.floor(ys).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)

        wx = xs - x0
        wy = ys - y0

        vals = (image[y0, x0] * (1 - wx) * (1 - wy) +
                image[y0, x1] * wx * (1 - wy) +
                image[y1, x0] * (1 - wx) * wy +
                image[y1, x1] * wx * wy)

        # Pad shorter rays with nan
        padded = np.full(len(r_values), np.nan)
        padded[:len(vals)] = vals
        all_intensities.append(padded)

    all_intensities = np.array(all_intensities)
    mean_profile = np.nanmean(all_intensities, axis=0)
    return r_values, mean_profile


def extract_single_ray_profile(image, centre, radius, angle_deg):
    """Extract intensity profile along a single ray at given angle.

    Returns (r_values, intensities).
    """
    cx, cy = centre
    h, w = image.shape[:2]
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    max_r = min(radius + 80, w - 1, h - 1)
    # Constrain by image boundaries
    if cos_a > 0:
        max_r = min(max_r, (w - 1 - cx) / cos_a)
    elif cos_a < 0:
        max_r = min(max_r, -cx / cos_a)
    if sin_a > 0:
        max_r = min(max_r, (h - 1 - cy) / sin_a)
    elif sin_a < 0:
        max_r = min(max_r, -cy / sin_a)

    r_values = np.arange(max(0, radius - 80), max_r, 0.5)
    xs = cx + r_values * cos_a
    ys = cy + r_values * sin_a

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = xs - x0
    wy = ys - y0

    vals = (image[y0, x0] * (1 - wx) * (1 - wy) +
            image[y0, x1] * wx * (1 - wy) +
            image[y1, x0] * (1 - wx) * wy +
            image[y1, x1] * wx * wy)

    return r_values, vals


def fit_erf_to_profile(r_values, intensities):
    """Fit erf_edge model to a radial profile. Returns (popt, r_squared) or None."""
    valid = ~np.isnan(intensities)
    r = r_values[valid]
    I = intensities[valid]
    if len(r) < 10:
        return None

    # Initial guesses
    I_bg_guess = np.median(I[-10:]) if len(I) >= 10 else I[-1]
    I_sphere_guess = np.median(I[:10]) if len(I) >= 10 else I[0]
    r_edge_guess = r[len(r) // 2]
    sigma_guess = 2.0

    try:
        popt, _ = curve_fit(
            erf_edge, r, I,
            p0=[I_bg_guess, I_sphere_guess, r_edge_guess, sigma_guess],
            bounds=([0, 0, r[0], 0.01], [1.5, 1.5, r[-1], 50]),
            maxfev=5000
        )
        fitted = erf_edge(r, *popt)
        ss_res = np.sum((I - fitted) ** 2)
        ss_tot = np.sum((I - np.mean(I)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt, r2
    except Exception:
        return None


# -- Drawing helpers --------------------------------------------------------

def put_title(img, text, font_scale=None, thickness=None, color=(255, 255, 255),
              bg_color=(0, 0, 0), position='top'):
    """Add a title bar at top or bottom of an image."""
    h, w = img.shape[:2]
    if font_scale is None:
        font_scale = max(0.4, w / 600)
    if thickness is None:
        thickness = max(1, int(font_scale + 0.5))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    bar_h = th + baseline + 12
    bar = np.full((bar_h, w, 3), bg_color, dtype=np.uint8)
    tx = max(4, (w - tw) // 2)
    ty = th + 6
    cv2.putText(bar, text, (tx, ty), font, font_scale, color, thickness)
    if position == 'top':
        return np.vstack([bar, img])
    else:
        return np.vstack([img, bar])


def put_label(img, text, pos=(5, 20), font_scale=0.5, color=(0, 255, 0),
              thickness=1, bg=True):
    """Put text with optional dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = img.copy()
    if bg:
        (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = pos
        cv2.rectangle(out, (x - 2, y - th - 4), (x + tw + 2, y + bl + 2),
                      (0, 0, 0), -1)
    cv2.putText(out, text, pos, font, font_scale, color, thickness)
    return out


def draw_zone_overlay(vis, cx, cy, r, margin, feather, margin_outer=None):
    """Draw zone boundaries with legend-friendly colours."""
    r_inner = r - margin
    r_outer = r + (margin_outer if margin_outer is not None else margin)
    # Detected edge = green solid
    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
    # Inner zone boundary = cyan dashed (approximated as solid thin)
    cv2.circle(vis, (cx, cy), max(0, r_inner), (255, 255, 0), 1)
    # Inner feather boundary = cyan dotted
    cv2.circle(vis, (cx, cy), max(0, r_inner - feather), (255, 200, 0), 1)
    # Outer zone boundary = red
    cv2.circle(vis, (cx, cy), r_outer, (0, 0, 255), 1)
    # Outer feather boundary
    cv2.circle(vis, (cx, cy), r_outer + feather, (0, 0, 200), 1)
    return vis


def make_legend(width, entries, font_scale=0.4):
    """Create a legend strip. entries = [(color_bgr, label), ...]."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    row_h = 18
    h = row_h * len(entries) + 8
    legend = np.zeros((h, width, 3), dtype=np.uint8)
    for i, (color, label) in enumerate(entries):
        y = 4 + row_h * i + 12
        cv2.line(legend, (8, y - 4), (28, y - 4), color, 2)
        cv2.putText(legend, label, (34, y), font, font_scale,
                    (255, 255, 255), thickness)
    return legend


def gray_to_bgr(img_f32):
    """Convert float32 grayscale [0,1] to uint8 BGR."""
    return cv2.cvtColor((np.clip(img_f32, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)


# ======================================================================
# INVESTIGATION 1: Margin Determination
# ======================================================================

def investigation_1():
    LOG.log("=" * 70)
    LOG.log("INVESTIGATION 1: Detection Error vs Blur Level")
    LOG.log("=" * 70)
    LOG.log()
    LOG.log("Purpose: Determine whether a 50px margin around the detected sphere")
    LOG.log("edge is sufficient by measuring how much Otsu detection drifts with blur.")
    LOG.log(f"Required margin = 3*sigma_max + max_detection_error + safety_buffer")

    # -- Part A: Calibration z-stack --
    LOG.log()
    LOG.log("-- Part A: Calibration Z-Stack (960x960) --")
    LOG.log("  Each image has real optical defocus. We detect the sphere on each")
    LOG.log("  and compare radius/centre to the in-focus reference frame.")
    LOG.log()
    images, measurements = load_calib_data()

    # Find focus frame reference
    focus_fn = None
    for fn, m in measurements.items():
        if m['is_focus']:
            focus_fn = fn
            break
    if focus_fn is None:
        focus_fn = min(measurements, key=lambda f: abs(measurements[f]['z_mm']))

    ref_img = images[focus_fn]
    ref_centre, ref_radius = detect_sphere(ref_img)
    LOG.log(f"Reference frame: {focus_fn}")
    LOG.log(f"  z = {measurements[focus_fn]['z_mm']:.1f} mm, "
            f"centre = {ref_centre}, radius = {ref_radius} px")
    LOG.log()

    header = f"{'Filename':>20s}  {'z_mm':>6s}  {'sigma':>6s}  {'radius':>6s}  {'dr':>5s}  {'d_centre':>8s}"
    LOG.log(header)
    LOG.log("-" * len(header))

    calib_results = []
    for fn in sorted(images.keys(), key=lambda f: measurements[f]['z_mm']):
        m = measurements[fn]
        c, r = detect_sphere(images[fn])
        if c is None:
            LOG.log(f"{fn:>20s}  {m['z_mm']:>6.1f}  {m['sigma_px']:>6.2f}  {'FAIL':>6s}  {'--':>5s}  {'--':>8s}")
            continue
        dr = r - ref_radius
        dc = math.sqrt((c[0] - ref_centre[0])**2 + (c[1] - ref_centre[1])**2)
        calib_results.append({'fn': fn, 'z': m['z_mm'], 'sigma': m['sigma_px'],
                              'r': r, 'dr': dr, 'dc': dc})
        LOG.log(f"{fn:>20s}  {m['z_mm']:>6.1f}  {m['sigma_px']:>6.2f}  {r:>6d}  {dr:>+5d}  {dc:>8.1f}")

    max_dr_c = max(abs(r['dr']) for r in calib_results)
    max_dc_c = max(r['dc'] for r in calib_results)
    margin_calib = math.ceil(3 * 11.5 + max_dr_c + 5)
    LOG.log()
    LOG.log(f"Result: max |delta_r| = {max_dr_c} px, max delta_centre = {max_dc_c:.1f} px")
    LOG.log(f"Recommended margin (calib): ceil(3 x 11.5 + {max_dr_c} + 5) = {margin_calib} px")

    # -- Plot: detection error vs |z| for calibration --
    plot_h, plot_w = 400, 700
    plot = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255

    z_vals = [abs(r['z']) for r in calib_results]
    dr_vals = [abs(r['dr']) for r in calib_results]
    dc_vals = [r['dc'] for r in calib_results]

    z_max = max(z_vals) * 1.1
    err_max = max(max(dr_vals), max(dc_vals)) * 1.2

    ox, oy = 60, 40  # origin offset
    pw, ph = plot_w - ox - 30, plot_h - oy - 50  # plot area

    def to_px(z, err):
        x = int(ox + (z / z_max) * pw)
        y = int(oy + ph - (err / err_max) * ph)
        return (x, y)

    # Axes
    cv2.line(plot, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
    cv2.line(plot, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Axis labels
    cv2.putText(plot, '|z| (mm)', (ox + pw // 2 - 30, plot_h - 5), font, 0.4, (0, 0, 0), 1)
    # Y-axis ticks
    for val in range(0, int(err_max) + 2, 2):
        pt = to_px(0, val)
        cv2.putText(plot, str(val), (5, pt[1] + 4), font, 0.35, (0, 0, 0), 1)
        cv2.line(plot, (ox - 3, pt[1]), (ox, pt[1]), (0, 0, 0), 1)
    # X-axis ticks
    for val in range(0, int(z_max) + 1, 2):
        pt = to_px(val, 0)
        cv2.putText(plot, str(val), (pt[0] - 5, oy + ph + 18), font, 0.35, (0, 0, 0), 1)
        cv2.line(plot, (pt[0], oy + ph), (pt[0], oy + ph + 3), (0, 0, 0), 1)

    # Plot points
    for r in calib_results:
        p1 = to_px(abs(r['z']), abs(r['dr']))
        p2 = to_px(abs(r['z']), r['dc'])
        cv2.circle(plot, p1, 3, (255, 0, 0), -1)  # blue = |dr|
        cv2.circle(plot, p2, 3, (0, 0, 255), -1)  # red = d_centre

    # Legend
    cv2.circle(plot, (ox + pw - 120, oy + 15), 3, (255, 0, 0), -1)
    cv2.putText(plot, '|delta_r| (px)', (ox + pw - 110, oy + 19), font, 0.35, (0, 0, 0), 1)
    cv2.circle(plot, (ox + pw - 120, oy + 32), 3, (0, 0, 255), -1)
    cv2.putText(plot, 'delta_centre (px)', (ox + pw - 110, oy + 36), font, 0.35, (0, 0, 0), 1)

    plot = put_title(plot, 'Inv1A: Calibration Detection Error vs |z|  (960x960)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv1a_calib_detection_error.png'), plot)

    # -- Part B: Training crops at model scale --
    LOG.log()
    LOG.log("-- Part B: Training Crops at 256x256 --")
    LOG.log("  We blur sharp crops with known sigma and check if Otsu detection shifts.")
    LOG.log()
    all_crops = load_v_crops_with_metadata()
    all_crops.sort(key=lambda c: c['diameter_px'])
    step = max(1, len(all_crops) // 20)
    selected = all_crops[::step][:20]

    sigmas_test = [0.5, 1, 2, 4, 6, 8, 10, 12]
    max_dr_model, max_dc_model = 0, 0.0

    header = f"{'Crop':>25s}  {'d_model':>7s}  {'sigma':>5s}  {'r_ref':>5s}  {'r_blur':>6s}  {'dr':>4s}  {'d_c':>5s}"
    LOG.log(header)
    LOG.log("-" * len(header))

    model_results = []  # (crop_name, d_model, sigma, dr, dc)
    for crop_info in selected:
        img_raw = cv2.imread(str(crop_info['path']), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        img = cv2.resize(img_raw, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        c0, r0 = detect_sphere(img)
        if c0 is None:
            continue
        for sigma in sigmas_test:
            blurred = apply_blur(img, sigma)
            cb, rb = detect_sphere(blurred)
            if cb is None:
                LOG.log(f"{crop_info['filename']:>25s}  {crop_info['diameter_model']:>7d}  "
                        f"{sigma:>5.1f}  {r0:>5d}  {'FAIL':>6s}  {'--':>4s}  {'--':>5s}")
                continue
            dr = rb - r0
            dc = math.sqrt((cb[0] - c0[0])**2 + (cb[1] - c0[1])**2)
            max_dr_model = max(max_dr_model, abs(dr))
            max_dc_model = max(max_dc_model, dc)
            model_results.append((sigma, abs(dr), dc))
            LOG.log(f"{crop_info['filename']:>25s}  {crop_info['diameter_model']:>7d}  "
                    f"{sigma:>5.1f}  {r0:>5d}  {rb:>6d}  {dr:>+4d}  {dc:>5.1f}")

    margin_model = math.ceil(3 * 12.4 + max_dr_model + 5)
    LOG.log()
    LOG.log(f"Result: max |delta_r| = {max_dr_model} px, max delta_centre = {max_dc_model:.1f} px")
    LOG.log(f"Recommended margin (model): ceil(3 x 12.4 + {max_dr_model} + 5) = {margin_model} px")
    LOG.log()
    LOG.log(f"50 px sufficient for model scale?  {'YES' if margin_model <= 50 else 'NO -- need ' + str(margin_model)}")
    LOG.log(f"50 px sufficient for calib scale?  {'YES' if margin_calib <= 50 else 'NO -- need ' + str(margin_calib)}")

    # -- Plot: detection error vs sigma for training crops --
    plot2 = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
    if model_results:
        s_vals = [r[0] for r in model_results]
        dr2 = [r[1] for r in model_results]
        dc2 = [r[2] for r in model_results]
        s_max = 13
        e_max2 = max(max(dr2 + [1]), max(dc2 + [1])) * 1.3

        def to_px2(s, err):
            x = int(ox + (s / s_max) * pw)
            y = int(oy + ph - (err / e_max2) * ph)
            return (x, y)

        cv2.line(plot2, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
        cv2.line(plot2, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
        cv2.putText(plot2, 'Applied sigma (px)', (ox + pw // 2 - 50, plot_h - 5), font, 0.4, (0, 0, 0), 1)
        for val in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            if val <= e_max2:
                pt = to_px2(0, val)
                cv2.putText(plot2, f'{val:.1f}', (5, pt[1] + 4), font, 0.3, (0, 0, 0), 1)
                cv2.line(plot2, (ox - 3, pt[1]), (ox, pt[1]), (150, 150, 150), 1)
        for val in sigmas_test:
            pt = to_px2(val, 0)
            cv2.putText(plot2, str(int(val)), (pt[0] - 3, oy + ph + 18), font, 0.35, (0, 0, 0), 1)

        for s, d, dc in model_results:
            cv2.circle(plot2, to_px2(s, d), 2, (255, 0, 0), -1)
            cv2.circle(plot2, to_px2(s, dc), 2, (0, 0, 255), -1)

        cv2.circle(plot2, (ox + pw - 120, oy + 15), 3, (255, 0, 0), -1)
        cv2.putText(plot2, '|delta_r| (px)', (ox + pw - 110, oy + 19), font, 0.35, (0, 0, 0), 1)
        cv2.circle(plot2, (ox + pw - 120, oy + 32), 3, (0, 0, 255), -1)
        cv2.putText(plot2, 'delta_centre (px)', (ox + pw - 110, oy + 36), font, 0.35, (0, 0, 0), 1)

    plot2 = put_title(plot2, 'Inv1B: Training Crop Detection Error vs Applied Sigma  (256x256)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv1b_model_detection_error.png'), plot2)

    LOG.log()
    LOG.log(f"Plots saved: inv1a_calib_detection_error.png, inv1b_model_detection_error.png")


# ======================================================================
# INVESTIGATION 2: Visual Verification
# ======================================================================

def investigation_2(margin=MARGIN):
    LOG.log()
    LOG.log("=" * 70)
    LOG.log("INVESTIGATION 2: Visual Verification")
    LOG.log("=" * 70)
    LOG.log()
    LOG.log("Purpose: Visually confirm that flattening removes interior texture and")
    LOG.log("background gradients while preserving the edge transition zone.")
    LOG.log(f"Margin = {margin} px, Feather = {FEATHER} px")
    LOG.log()
    LOG.log("Zone colour key for overlays:")
    LOG.log("  Cyan-yellow circle = inner boundary (R - margin): inside this = flat 0")
    LOG.log("  Green circle       = detected sphere edge (Otsu)")
    LOG.log("  Red circle         = outer boundary (R + margin): outside this = flat 1")

    all_crops = load_v_crops_with_metadata()
    all_crops.sort(key=lambda c: c['diameter_model'])

    # Pick 6 crops: 2 small, 2 medium, 2 large
    n = len(all_crops)
    small = all_crops[:n // 3]
    medium = all_crops[n // 3:2 * n // 3]
    large = all_crops[2 * n // 3:]

    selected = []
    for group, label in [(small, 'small'), (medium, 'medium'), (large, 'large')]:
        selected.append((group[0], label))
        selected.append((group[len(group) // 2], label))

    LOG.log()

    # -- Part A: Training crops --
    LOG.log("-- Part A: Training Crops (256x256) --")
    blur_demo_indices = [0, 2, 4]  # One per size bin

    zone_legend = make_legend(MODEL_SIZE, [
        ((255, 200, 0), 'Inner feather edge'),
        ((255, 255, 0), 'Inner boundary (R-margin)'),
        ((0, 255, 0), 'Detected edge (Otsu R)'),
        ((0, 0, 255), 'Outer boundary (R+margin)'),
        ((0, 0, 200), 'Outer feather edge'),
    ])

    for i, (crop_info, size_label) in enumerate(selected):
        img_raw = cv2.imread(str(crop_info['path']), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        img = cv2.resize(img_raw, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        flat, info = flatten_sphere_crop(img, margin_inner=margin, margin_outer=margin)
        if info is None:
            LOG.log(f"  {crop_info['filename']}: detect_sphere FAILED, skipping")
            continue

        cx, cy = info['center']
        r = info['radius']
        d_model = crop_info['diameter_model']
        LOG.log(f"  {crop_info['filename']}: d_model={d_model}, R={r}, "
                f"centre=({cx},{cy}), size={size_label}")

        # Build 2-panel: raw | flattened
        raw_vis = gray_to_bgr(img)
        flat_vis = gray_to_bgr(flat)

        draw_zone_overlay(raw_vis, cx, cy, r, margin, FEATHER)
        draw_zone_overlay(flat_vis, cx, cy, r, margin, FEATHER)

        # Add per-panel labels
        raw_vis = put_label(raw_vis, 'RAW', (5, 20))
        flat_vis = put_label(flat_vis, 'FLATTENED', (5, 20))

        panel = np.hstack([raw_vis, flat_vis])
        # Stretch legend to panel width
        legend_row = cv2.resize(zone_legend, (panel.shape[1], zone_legend.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        panel = np.vstack([panel, legend_row])

        title = (f"Inv2: {crop_info['filename']}  |  {size_label} sphere  |  "
                 f"d_model={d_model}px  R={r}px  margin={margin}px")
        panel = put_title(panel, title)

        out_name = f"inv2_crop_{i}_{size_label}_{crop_info['filename'].replace('.png', '')}.png"
        cv2.imwrite(str(OUTPUT_DIR / out_name), panel)

        # 4-panel with blur for selected crops
        if i in blur_demo_indices:
            sub_panels = []
            configs = [
                ('Raw (original)', img, False),
                ('Flattened', flat, False),
                ('Flattened + sigma=6', None, 6),
                ('Flattened + sigma=12', None, 12),
            ]
            for lbl, src, sigma in configs:
                if sigma:
                    src = apply_blur(flat, sigma)
                vis = gray_to_bgr(src)
                vis = put_label(vis, lbl, (5, 20), font_scale=0.4)
                sub_panels.append(vis)

            panel_4 = np.hstack(sub_panels)
            title4 = (f"Inv2: Blur progression on {crop_info['filename']}  |  "
                      f"{size_label}  d={d_model}px  R={r}px")
            panel_4 = put_title(panel_4, title4)

            out_name = f"inv2_blur_{i}_{size_label}_{crop_info['filename'].replace('.png', '')}.png"
            cv2.imwrite(str(OUTPUT_DIR / out_name), panel_4)

    # -- Part B: Calibration z-stack --
    LOG.log()
    LOG.log("-- Part B: Calibration Z-Stack (960x960) --")
    LOG.log("  Real optical defocus at three z-positions: near-focus, moderate, heavy.")
    images, measurements = load_calib_data()

    target_z = [-0.4, -4.0, -8.0]
    defocus_labels = ['Near-focus', 'Moderate defocus', 'Heavy defocus']
    for tz, dlabel in zip(target_z, defocus_labels):
        best_fn = min(images.keys(),
                      key=lambda f: abs(measurements[f]['z_mm'] - tz))
        m = measurements[best_fn]
        img = images[best_fn]

        flat, info = flatten_sphere_crop(img, margin_inner=margin, margin_outer=margin)
        if info is None:
            LOG.log(f"  {best_fn}: detect_sphere FAILED")
            continue

        cx, cy = info['center']
        r = info['radius']
        LOG.log(f"  {best_fn}: z={m['z_mm']:.1f}mm, sigma={m['sigma_px']:.2f}px, "
                f"R={r}px  ({dlabel})")

        raw_vis = gray_to_bgr(img)
        flat_vis = gray_to_bgr(flat)

        draw_zone_overlay(raw_vis, cx, cy, r, margin, FEATHER)
        draw_zone_overlay(flat_vis, cx, cy, r, margin, FEATHER)

        raw_vis = put_label(raw_vis, 'RAW', (10, 30), font_scale=0.7)
        flat_vis = put_label(flat_vis, 'FLATTENED', (10, 30), font_scale=0.7)

        # Scale down to fit side by side reasonably
        scale = 0.5
        raw_small = cv2.resize(raw_vis, None, fx=scale, fy=scale)
        flat_small = cv2.resize(flat_vis, None, fx=scale, fy=scale)
        panel = np.hstack([raw_small, flat_small])

        title = (f"Inv2: {best_fn}  |  {dlabel}  |  "
                 f"z={m['z_mm']:.1f}mm  sigma={m['sigma_px']:.2f}px  R={r}px  margin={margin}px")
        panel = put_title(panel, title)

        z_str = f"{m['z_mm']:.1f}".replace('-', 'neg')
        out_name = f"inv2_calib_z{z_str}_{best_fn}"
        cv2.imwrite(str(OUTPUT_DIR / out_name), panel)

    LOG.log()
    LOG.log(f"All visual panels saved to: {OUTPUT_DIR}")


# ======================================================================
# INVESTIGATION 3: ERF Improvement Quantification
# ======================================================================

def investigation_3(margin=MARGIN):
    LOG.log()
    LOG.log("=" * 70)
    LOG.log("INVESTIGATION 3: ERF Improvement Quantification")
    LOG.log("=" * 70)
    LOG.log()
    LOG.log("Purpose: Measure systematic ERF bias before and after flattening.")
    LOG.log("Part A uses known applied sigma (ground truth) on training crops.")
    LOG.log("Part B uses the calibration z-stack and re-fits the linear model.")

    # -- Part A: Synthetic ground truth on real crops --
    LOG.log()
    LOG.log("-- Part A: Synthetic Ground Truth on Real Crops (256x256) --")
    LOG.log("  For each crop, we apply known Gaussian blur and measure via ERF.")
    LOG.log("  Error = (measured - target) / target x 100%")
    LOG.log("  Negative = ERF underestimates blur. Positive = overestimates.")
    LOG.log()

    all_crops = load_v_crops_with_metadata()
    all_crops.sort(key=lambda c: c['diameter_px'])
    step = max(1, len(all_crops) // 10)
    selected = all_crops[::step][:10]

    sigma_targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    header = (f"{'Crop':>25s}  {'sig_tgt':>7s}  {'sig_raw':>7s}  {'err_raw%':>8s}  "
              f"{'sig_flat':>8s}  {'err_flat%':>9s}")
    LOG.log(header)
    LOG.log("-" * len(header))

    raw_errors_by_sigma = {s: [] for s in sigma_targets}
    flat_errors_by_sigma = {s: [] for s in sigma_targets}
    all_rows_3a = []

    for crop_info in selected:
        img_raw = cv2.imread(str(crop_info['path']), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        img = cv2.resize(img_raw, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        native_raw, _ = measure_erf_sigma(img)
        flat_img, finfo = flatten_sphere_crop(img, margin_inner=margin, margin_outer=margin)
        if finfo is None:
            continue
        native_flat, _ = measure_erf_sigma(flat_img)

        for sigma_target in sigma_targets:
            # Raw path
            if not np.isnan(native_raw):
                ksq = sigma_target**2 - native_raw**2
                if ksq > 0:
                    blurred_raw = apply_blur(img, np.sqrt(ksq))
                    meas_raw, _ = measure_erf_sigma(blurred_raw)
                else:
                    meas_raw = float('nan')
            else:
                meas_raw = float('nan')

            # Flat path
            if not np.isnan(native_flat):
                ksq_f = sigma_target**2 - native_flat**2
                if ksq_f > 0:
                    blurred_flat = apply_blur(flat_img, np.sqrt(ksq_f))
                    meas_flat, _ = measure_erf_sigma(blurred_flat)
                else:
                    meas_flat = float('nan')
            else:
                meas_flat = float('nan')

            err_raw = 100 * (meas_raw - sigma_target) / sigma_target if not np.isnan(meas_raw) else float('nan')
            err_flat = 100 * (meas_flat - sigma_target) / sigma_target if not np.isnan(meas_flat) else float('nan')

            if not np.isnan(err_raw):
                raw_errors_by_sigma[sigma_target].append(err_raw)
            if not np.isnan(err_flat):
                flat_errors_by_sigma[sigma_target].append(err_flat)

            all_rows_3a.append((sigma_target, err_raw, err_flat))

            LOG.log(f"{crop_info['filename']:>25s}  {sigma_target:>7.1f}  "
                    f"{meas_raw:>7.3f}  {err_raw:>+8.2f}  "
                    f"{meas_flat:>8.3f}  {err_flat:>+9.2f}")

    # Summary table
    LOG.log()
    LOG.log(f"{'--- Summary by sigma ---':>40s}")
    summary_header = f"{'sigma':>7s}  {'mean_raw%':>9s}  {'mean_flat%':>10s}  {'max|raw|%':>9s}  {'max|flat|%':>10s}"
    LOG.log(summary_header)
    LOG.log("-" * len(summary_header))

    summary_data = []  # (sigma, mean_raw, mean_flat, max_raw, max_flat)
    for s in sigma_targets:
        re = raw_errors_by_sigma[s]
        fe = flat_errors_by_sigma[s]
        mr = np.mean(re) if re else float('nan')
        mf = np.mean(fe) if fe else float('nan')
        xr = max(abs(e) for e in re) if re else float('nan')
        xf = max(abs(e) for e in fe) if fe else float('nan')
        summary_data.append((s, mr, mf, xr, xf))
        LOG.log(f"{s:>7.0f}  {mr:>+9.2f}  {mf:>+10.2f}  {xr:>9.2f}  {xf:>10.2f}")

    # -- Plot: Mean ERF error vs sigma (raw vs flat) --
    plot_h, plot_w = 450, 700
    plot3a = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    ox, oy = 70, 50
    pw, ph = plot_w - ox - 40, plot_h - oy - 60

    # Y range: -20% to +5%
    y_min, y_max = -22, 5

    def to_px3(sigma, err_pct):
        x = int(ox + ((sigma - 1) / 11.0) * pw)
        y = int(oy + ph - ((err_pct - y_min) / (y_max - y_min)) * ph)
        return (x, y)

    # Grid
    cv2.line(plot3a, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
    cv2.line(plot3a, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
    # Zero line
    zero_y = to_px3(1, 0)[1]
    cv2.line(plot3a, (ox, zero_y), (ox + pw, zero_y), (180, 180, 180), 1)

    # Axis labels
    cv2.putText(plot3a, 'Target sigma (px)', (ox + pw // 2 - 60, plot_h - 8), font, 0.4, (0, 0, 0), 1)
    # Rotated Y label approximation
    cv2.putText(plot3a, 'Error %', (5, oy + ph // 2), font, 0.4, (0, 0, 0), 1)

    # Y ticks
    for val in range(-20, 6, 5):
        pt = to_px3(1, val)
        cv2.putText(plot3a, f'{val:+d}%', (15, pt[1] + 4), font, 0.35, (0, 0, 0), 1)
        cv2.line(plot3a, (ox - 3, pt[1]), (ox + pw, pt[1]), (230, 230, 230), 1)

    # X ticks
    for s in sigma_targets:
        pt = to_px3(s, y_min)
        cv2.putText(plot3a, str(s), (pt[0] - 3, oy + ph + 18), font, 0.35, (0, 0, 0), 1)

    # Plot mean error lines
    for j in range(len(summary_data) - 1):
        s1, mr1, mf1, _, _ = summary_data[j]
        s2, mr2, mf2, _, _ = summary_data[j + 1]
        if not np.isnan(mr1) and not np.isnan(mr2):
            cv2.line(plot3a, to_px3(s1, mr1), to_px3(s2, mr2), (0, 0, 200), 2)
        if not np.isnan(mf1) and not np.isnan(mf2):
            cv2.line(plot3a, to_px3(s1, mf1), to_px3(s2, mf2), (0, 160, 0), 2)

    # Plot individual points as scatter
    for sigma_t, er, ef in all_rows_3a:
        if not np.isnan(er):
            cv2.circle(plot3a, to_px3(sigma_t, er), 2, (0, 0, 200), -1)
        if not np.isnan(ef):
            cv2.circle(plot3a, to_px3(sigma_t, ef), 2, (0, 160, 0), -1)

    # Legend
    lx, ly = ox + 10, oy + 10
    cv2.line(plot3a, (lx, ly), (lx + 20, ly), (0, 0, 200), 2)
    cv2.putText(plot3a, 'Raw (mean)', (lx + 25, ly + 4), font, 0.4, (0, 0, 0), 1)
    cv2.line(plot3a, (lx, ly + 18), (lx + 20, ly + 18), (0, 160, 0), 2)
    cv2.putText(plot3a, 'Flattened (mean)', (lx + 25, ly + 22), font, 0.4, (0, 0, 0), 1)
    cv2.putText(plot3a, 'dots = individual crops', (lx, ly + 40), font, 0.35, (120, 120, 120), 1)

    plot3a = put_title(plot3a, 'Inv3A: ERF Measurement Error vs Target Sigma  (raw vs flattened, 256x256)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv3a_erf_error_vs_sigma.png'), plot3a)

    # -- Part B: Calibration z-stack --
    LOG.log()
    LOG.log("-- Part B: Calibration Z-Stack (960x960) --")
    LOG.log("  Measure ERF on raw and flattened calibration images, re-fit linear model.")
    LOG.log("  sigma = rho * |z| + sigma_0")
    LOG.log()

    images, measurements = load_calib_data()

    results = []
    for fn in sorted(images.keys(), key=lambda f: measurements[f]['z_mm']):
        m = measurements[fn]
        img = images[fn]
        sigma_raw, r2_raw = measure_erf_sigma(img)
        flat_img, finfo = flatten_sphere_crop(img, margin_inner=margin, margin_outer=margin)
        if finfo is not None:
            sigma_flat, r2_flat = measure_erf_sigma(flat_img)
        else:
            sigma_flat, r2_flat = float('nan'), float('nan')
        results.append({
            'fn': fn, 'z': m['z_mm'],
            'sigma_csv': m['sigma_px'],
            'sigma_raw': sigma_raw, 'r2_raw': r2_raw,
            'sigma_flat': sigma_flat, 'r2_flat': r2_flat,
        })

    header3b = f"{'z_mm':>6s}  {'sig_csv':>7s}  {'sig_raw':>7s}  {'sig_flat':>8s}  {'diff':>6s}  {'r2_raw':>6s}  {'r2_flat':>7s}"
    LOG.log(header3b)
    LOG.log("-" * len(header3b))
    for r in results:
        diff = r['sigma_flat'] - r['sigma_raw'] if not (np.isnan(r['sigma_flat']) or np.isnan(r['sigma_raw'])) else float('nan')
        LOG.log(f"{r['z']:>6.1f}  {r['sigma_csv']:>7.2f}  {r['sigma_raw']:>7.3f}  "
                f"{r['sigma_flat']:>8.3f}  {diff:>+6.3f}  {r['r2_raw']:>6.4f}  {r['r2_flat']:>7.4f}")

    # Fit linear models
    def linear_model(z, rho, sigma_0):
        return rho * np.abs(z) + sigma_0

    valid_raw = [(r['z'], r['sigma_raw']) for r in results
                 if not np.isnan(r['sigma_raw']) and abs(r['z']) > 0.5]
    valid_flat = [(r['z'], r['sigma_flat']) for r in results
                  if not np.isnan(r['sigma_flat']) and abs(r['z']) > 0.5]

    popt_r = popt_f = None
    r2_raw_fit = r2_flat_fit = float('nan')

    if valid_raw:
        z_arr = np.array([v[0] for v in valid_raw])
        s_arr = np.array([v[1] for v in valid_raw])
        popt_r, _ = curve_fit(linear_model, z_arr, s_arr, p0=[1.4, 0.2])
        ss_res = np.sum((s_arr - linear_model(z_arr, *popt_r))**2)
        ss_tot = np.sum((s_arr - np.mean(s_arr))**2)
        r2_raw_fit = 1 - ss_res / ss_tot
        LOG.log(f"\nRaw linear fit:  rho = {popt_r[0]:.4f} px/mm, sigma_0 = {popt_r[1]:.4f} px, R2 = {r2_raw_fit:.6f}")

    if valid_flat:
        z_arr = np.array([v[0] for v in valid_flat])
        s_arr = np.array([v[1] for v in valid_flat])
        popt_f, _ = curve_fit(linear_model, z_arr, s_arr, p0=[1.4, 0.2])
        ss_res = np.sum((s_arr - linear_model(z_arr, *popt_f))**2)
        ss_tot = np.sum((s_arr - np.mean(s_arr))**2)
        r2_flat_fit = 1 - ss_res / ss_tot
        LOG.log(f"Flat linear fit: rho = {popt_f[0]:.4f} px/mm, sigma_0 = {popt_f[1]:.4f} px, R2 = {r2_flat_fit:.6f}")

    if popt_r is not None and popt_f is not None:
        LOG.log()
        LOG.log(f"Change: delta_rho = {popt_f[0]-popt_r[0]:+.4f} px/mm ({100*(popt_f[0]-popt_r[0])/popt_r[0]:+.1f}%)")
        LOG.log(f"        delta_sigma_0 = {popt_f[1]-popt_r[1]:+.4f} px")
        LOG.log(f"        delta_R2 = {r2_flat_fit-r2_raw_fit:+.6f}")
        LOG.log()
        LOG.log("Interpretation: Flattening reveals the true (larger) blur that was")
        LOG.log("being masked by texture contamination. rho increases because the raw")
        LOG.log("measurements systematically underestimated sigma at high defocus.")
        LOG.log("If flattening is adopted, recalibrated values would be:")
        LOG.log(f"  rho = {popt_f[0]:.4f} px/mm,  sigma_0 = {popt_f[1]:.4f} px")

    # -- Plot: sigma vs |z| with raw, flat, and linear fits --
    plot3b = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
    z_all = [abs(r['z']) for r in results]
    s_raw_all = [r['sigma_raw'] for r in results]
    s_flat_all = [r['sigma_flat'] for r in results]

    z_plot_max = max(z_all) * 1.1
    s_plot_max = max(max(s for s in s_flat_all if not np.isnan(s)),
                     max(s for s in s_raw_all if not np.isnan(s))) * 1.15

    def to_px3b(z, s):
        x = int(ox + (z / z_plot_max) * pw)
        y = int(oy + ph - (s / s_plot_max) * ph)
        return (x, y)

    # Axes
    cv2.line(plot3b, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
    cv2.line(plot3b, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
    cv2.putText(plot3b, '|z| (mm)', (ox + pw // 2 - 30, plot_h - 8), font, 0.4, (0, 0, 0), 1)
    cv2.putText(plot3b, 'sigma', (5, oy + ph // 2), font, 0.4, (0, 0, 0), 1)
    cv2.putText(plot3b, '(px)', (5, oy + ph // 2 + 18), font, 0.4, (0, 0, 0), 1)

    # Y ticks
    for val in range(0, int(s_plot_max) + 2, 2):
        pt = to_px3b(0, val)
        cv2.putText(plot3b, str(val), (25, pt[1] + 4), font, 0.35, (0, 0, 0), 1)
        cv2.line(plot3b, (ox, pt[1]), (ox + pw, pt[1]), (240, 240, 240), 1)
    # X ticks
    for val in range(0, int(z_plot_max) + 1, 1):
        pt = to_px3b(val, 0)
        cv2.putText(plot3b, str(val), (pt[0] - 3, oy + ph + 18), font, 0.35, (0, 0, 0), 1)

    # Data points
    for r in results:
        z = abs(r['z'])
        if not np.isnan(r['sigma_raw']):
            cv2.circle(plot3b, to_px3b(z, r['sigma_raw']), 3, (0, 0, 200), -1)
        if not np.isnan(r['sigma_flat']):
            cv2.circle(plot3b, to_px3b(z, r['sigma_flat']), 3, (0, 160, 0), -1)

    # Fit lines
    if popt_r is not None:
        for z in np.linspace(0, z_plot_max, 100):
            s = popt_r[0] * z + popt_r[1]
            if 0 <= s <= s_plot_max:
                cv2.circle(plot3b, to_px3b(z, s), 1, (0, 0, 200), -1)
    if popt_f is not None:
        for z in np.linspace(0, z_plot_max, 100):
            s = popt_f[0] * z + popt_f[1]
            if 0 <= s <= s_plot_max:
                cv2.circle(plot3b, to_px3b(z, s), 1, (0, 160, 0), -1)

    # Legend
    lx, ly = ox + pw - 200, oy + 15
    cv2.circle(plot3b, (lx, ly), 3, (0, 0, 200), -1)
    cv2.putText(plot3b, f'Raw (rho={popt_r[0]:.3f})' if popt_r is not None else 'Raw',
                (lx + 8, ly + 4), font, 0.35, (0, 0, 0), 1)
    cv2.circle(plot3b, (lx, ly + 18), 3, (0, 160, 0), -1)
    cv2.putText(plot3b, f'Flat (rho={popt_f[0]:.3f})' if popt_f is not None else 'Flat',
                (lx + 8, ly + 22), font, 0.35, (0, 0, 0), 1)

    plot3b = put_title(plot3b, 'Inv3B: Calibration sigma vs |z|  --  Raw vs Flattened  (960x960)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv3b_calib_sigma_vs_z.png'), plot3b)

    # -- Plot: difference (flat - raw) vs |z| --
    plot3c = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
    diffs = [(abs(r['z']), r['sigma_flat'] - r['sigma_raw'])
             for r in results
             if not np.isnan(r['sigma_flat']) and not np.isnan(r['sigma_raw'])]

    d_max = max(d for _, d in diffs) * 1.2

    def to_px3c(z, d):
        x = int(ox + (z / z_plot_max) * pw)
        y = int(oy + ph - (d / d_max) * ph)
        return (x, y)

    cv2.line(plot3c, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
    cv2.line(plot3c, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
    cv2.putText(plot3c, '|z| (mm)', (ox + pw // 2 - 30, plot_h - 8), font, 0.4, (0, 0, 0), 1)
    cv2.putText(plot3c, 'sig_flat', (2, oy + ph // 2 - 8), font, 0.35, (0, 0, 0), 1)
    cv2.putText(plot3c, '- sig_raw', (2, oy + ph // 2 + 8), font, 0.35, (0, 0, 0), 1)
    cv2.putText(plot3c, '(px)', (12, oy + ph // 2 + 24), font, 0.35, (0, 0, 0), 1)

    # Y ticks
    for val_10 in range(0, int(d_max * 10) + 5, 5):
        val = val_10 / 10.0
        if val <= d_max:
            pt = to_px3c(0, val)
            cv2.putText(plot3c, f'{val:.1f}', (25, pt[1] + 4), font, 0.3, (0, 0, 0), 1)
            cv2.line(plot3c, (ox, pt[1]), (ox + pw, pt[1]), (240, 240, 240), 1)
    # X ticks
    for val in range(0, int(z_plot_max) + 1, 1):
        pt = to_px3c(val, 0)
        cv2.putText(plot3c, str(val), (pt[0] - 3, oy + ph + 18), font, 0.35, (0, 0, 0), 1)

    for z, d in diffs:
        cv2.circle(plot3c, to_px3c(z, d), 3, (200, 100, 0), -1)

    plot3c = put_title(plot3c, 'Inv3C: Flattening Effect (sigma_flat - sigma_raw) vs |z|  (960x960)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv3c_flattening_delta_vs_z.png'), plot3c)

    LOG.log()
    LOG.log(f"Plots saved: inv3a_erf_error_vs_sigma.png, inv3b_calib_sigma_vs_z.png, inv3c_flattening_delta_vs_z.png")


# ======================================================================
# INVESTIGATION 4: Radial Intensity Profiles
# ======================================================================

def investigation_4(margin=MARGIN):
    LOG.log()
    LOG.log("=" * 70)
    LOG.log("INVESTIGATION 4: Radial Intensity Profiles")
    LOG.log("=" * 70)
    LOG.log()
    LOG.log("Purpose: Show the 1D signal the ERF fitting sees -- intensity vs radial")
    LOG.log("distance. Reveals texture contamination in raw profiles and clean step")
    LOG.log("edges after flattening.")
    LOG.log()

    all_crops = load_v_crops_with_metadata()
    all_crops.sort(key=lambda c: c['diameter_model'])
    n = len(all_crops)
    # Pick 3: small, medium, large
    selected = [
        (all_crops[0], 'small'),
        (all_crops[n // 2], 'medium'),
        (all_crops[-1], 'large'),
    ]

    plot_h, plot_w = 500, 800
    font = cv2.FONT_HERSHEY_SIMPLEX

    for crop_info, size_label in selected:
        img_raw = cv2.imread(str(crop_info['path']), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        img = cv2.resize(img_raw, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        centre, radius = detect_sphere(img)
        if centre is None:
            continue

        flat, _ = flatten_sphere_crop(img, center=centre, radius=radius,
                                      margin_inner=margin, margin_outer=margin)
        flat_b6 = apply_blur(flat, 6)
        flat_b12 = apply_blur(flat, 12)

        # Extract profiles
        configs = [
            ('Raw (sharp)', img, (0, 0, 200)),        # red
            ('Flattened (sharp)', flat, (0, 160, 0)),  # green
            ('Flattened + sigma=6', flat_b6, (200, 100, 0)),  # blue
            ('Flattened + sigma=12', flat_b12, (180, 0, 180)),  # purple
        ]

        profiles = []
        for label, src, color in configs:
            r_vals, intensities = extract_radial_profile(src, centre, radius)
            profiles.append((label, r_vals, intensities, color))

        LOG.log(f"  {crop_info['filename']}: {size_label}, d_model={crop_info['diameter_model']}, R={radius}")

        # Plot
        plot = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
        ox, oy = 60, 40
        pw, ph = plot_w - ox - 30, plot_h - oy - 60

        r_max = max(len(p[1]) for p in profiles) * 0.5  # max radial distance
        r_max = min(r_max, radius + 70)

        def to_px(r, intensity):
            x = int(ox + (r / r_max) * pw)
            y = int(oy + ph - intensity * ph)
            return (x, y)

        # Axes
        cv2.line(plot, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
        cv2.line(plot, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
        cv2.putText(plot, 'Radial distance from centre (px)', (ox + pw // 3 - 30, plot_h - 8),
                    font, 0.4, (0, 0, 0), 1)
        cv2.putText(plot, 'Intensity', (2, oy + ph // 2), font, 0.35, (0, 0, 0), 1)

        # Y ticks
        for val_10 in range(0, 11, 2):
            val = val_10 / 10.0
            pt = to_px(0, val)
            cv2.putText(plot, f'{val:.1f}', (15, pt[1] + 4), font, 0.3, (0, 0, 0), 1)
            cv2.line(plot, (ox, pt[1]), (ox + pw, pt[1]), (240, 240, 240), 1)

        # X ticks
        for val in range(0, int(r_max) + 1, 20):
            pt = to_px(val, 0)
            cv2.putText(plot, str(val), (pt[0] - 5, oy + ph + 18), font, 0.3, (0, 0, 0), 1)

        # Zone boundary lines
        r_inner = radius - margin
        r_outer = radius + margin
        for bnd, bnd_color, bnd_label in [
            (r_inner, (255, 255, 0), f'R-{margin}'),
            (radius, (0, 255, 0), 'R (edge)'),
            (r_outer, (0, 0, 255), f'R+{margin}'),
        ]:
            if 0 < bnd < r_max:
                x = int(ox + (bnd / r_max) * pw)
                cv2.line(plot, (x, oy), (x, oy + ph), bnd_color, 1)
                cv2.putText(plot, bnd_label, (x + 3, oy + 12), font, 0.3, bnd_color, 1)

        # Plot profiles
        for label, r_vals, intensities, color in profiles:
            for j in range(1, len(r_vals)):
                if np.isnan(intensities[j]) or np.isnan(intensities[j - 1]):
                    continue
                if r_vals[j] > r_max:
                    break
                p1 = to_px(r_vals[j - 1], intensities[j - 1])
                p2 = to_px(r_vals[j], intensities[j])
                cv2.line(plot, p1, p2, color, 1)

        # Legend
        lx, ly = ox + 10, oy + 10
        for k, (label, _, _, color) in enumerate(profiles):
            y = ly + k * 16
            cv2.line(plot, (lx, y), (lx + 15, y), color, 2)
            cv2.putText(plot, label, (lx + 20, y + 4), font, 0.35, (0, 0, 0), 1)

        title = (f"Inv4: Radial Profile -- {crop_info['filename']}  |  "
                 f"{size_label}  d={crop_info['diameter_model']}px  R={radius}px")
        plot = put_title(plot, title)
        out_name = f"inv4_radial_profile_{size_label}_{crop_info['filename'].replace('.png', '')}.png"
        cv2.imwrite(str(OUTPUT_DIR / out_name), plot)

    LOG.log()
    LOG.log(f"Plots saved: inv4_radial_profile_*.png")


# ======================================================================
# INVESTIGATION 5: Asymmetric Margin Test
# ======================================================================

def _run_3a_for_margin(selected_crops, sigma_targets, mi, mo):
    """Run Investigation 3A logic for given inner/outer margins. Returns errors_by_sigma."""
    errors_by_sigma = {s: [] for s in sigma_targets}
    for crop_info in selected_crops:
        img_raw = cv2.imread(str(crop_info['path']), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        img = cv2.resize(img_raw, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        flat_img, finfo = flatten_sphere_crop(img, margin_inner=mi, margin_outer=mo)
        if finfo is None:
            continue
        native_flat, _ = measure_erf_sigma(flat_img)

        for sigma_target in sigma_targets:
            if not np.isnan(native_flat):
                ksq_f = sigma_target**2 - native_flat**2
                if ksq_f > 0:
                    blurred_flat = apply_blur(flat_img, np.sqrt(ksq_f))
                    meas_flat, _ = measure_erf_sigma(blurred_flat)
                else:
                    meas_flat = float('nan')
            else:
                meas_flat = float('nan')

            err_flat = 100 * (meas_flat - sigma_target) / sigma_target if not np.isnan(meas_flat) else float('nan')
            if not np.isnan(err_flat):
                errors_by_sigma[sigma_target].append(err_flat)

    return errors_by_sigma


def _run_3b_for_margin(images, measurements, mi, mo):
    """Run Investigation 3B logic for given inner/outer margins. Returns (results, popt, r2)."""
    results = []
    for fn in sorted(images.keys(), key=lambda f: measurements[f]['z_mm']):
        m = measurements[fn]
        img = images[fn]
        flat_img, finfo = flatten_sphere_crop(img, margin_inner=mi, margin_outer=mo)
        if finfo is not None:
            sigma_flat, r2_flat = measure_erf_sigma(flat_img)
        else:
            sigma_flat, r2_flat = float('nan'), float('nan')
        results.append({'z': m['z_mm'], 'sigma_flat': sigma_flat})

    def linear_model(z, rho, sigma_0):
        return rho * np.abs(z) + sigma_0

    valid = [(r['z'], r['sigma_flat']) for r in results
             if not np.isnan(r['sigma_flat']) and abs(r['z']) > 0.5]
    if valid:
        z_arr = np.array([v[0] for v in valid])
        s_arr = np.array([v[1] for v in valid])
        popt, _ = curve_fit(linear_model, z_arr, s_arr, p0=[1.4, 0.2])
        ss_res = np.sum((s_arr - linear_model(z_arr, *popt))**2)
        ss_tot = np.sum((s_arr - np.mean(s_arr))**2)
        r2 = 1 - ss_res / ss_tot
        return results, popt, r2
    return results, None, float('nan')


def investigation_5():
    LOG.log()
    LOG.log("=" * 70)
    LOG.log("INVESTIGATION 5: Asymmetric Margin Test")
    LOG.log("=" * 70)
    LOG.log()
    LOG.log("Purpose: Test whether inner margin can be smaller than outer without")
    LOG.log("degrading ERF accuracy. Compares symmetric 50/50 vs asymmetric 30/50 and 20/50.")

    all_crops = load_v_crops_with_metadata()
    all_crops.sort(key=lambda c: c['diameter_px'])
    step = max(1, len(all_crops) // 10)
    selected = all_crops[::step][:10]
    sigma_targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    configs = [
        ('50/50 (symmetric)', 50, 50),
        ('30/50 (inner=30)', 30, 50),
        ('20/50 (inner=20)', 20, 50),
    ]

    # -- Part A: Synthetic ground truth --
    LOG.log()
    LOG.log("-- Part A: Synthetic Ground Truth (256x256, 10 crops x 12 sigmas) --")
    LOG.log()

    all_config_errors = {}
    for config_label, mi, mo in configs:
        LOG.log(f"  Running {config_label}...")
        errors = _run_3a_for_margin(selected, sigma_targets, mi, mo)
        all_config_errors[config_label] = errors

    # Summary table
    header5a = f"{'sigma':>7s}"
    for label, _, _ in configs:
        header5a += f"  {'mean%':>7s}  {'max|%|':>7s}"
    LOG.log()
    LOG.log(header5a)
    LOG.log("-" * len(header5a))

    # Collect for plotting
    plot_data_5a = {}  # config_label -> [(sigma, mean_err)]
    for label, _, _ in configs:
        plot_data_5a[label] = []

    for s in sigma_targets:
        row = f"{s:>7.0f}"
        for label, _, _ in configs:
            errs = all_config_errors[label][s]
            m = np.mean(errs) if errs else float('nan')
            x = max(abs(e) for e in errs) if errs else float('nan')
            row += f"  {m:>+7.2f}  {x:>7.2f}"
            plot_data_5a[label].append((s, m))
        LOG.log(row)

    # -- Part B: Calibration z-stack --
    LOG.log()
    LOG.log("-- Part B: Calibration Z-Stack (960x960) --")
    LOG.log()

    images, measurements = load_calib_data()
    calib_results_5 = {}

    for config_label, mi, mo in configs:
        LOG.log(f"  Running {config_label}...")
        results, popt, r2 = _run_3b_for_margin(images, measurements, mi, mo)
        calib_results_5[config_label] = (results, popt, r2)
        if popt is not None:
            LOG.log(f"    rho = {popt[0]:.4f} px/mm, sigma_0 = {popt[1]:.4f} px, R2 = {r2:.6f}")

    # Comparison table
    LOG.log()
    LOG.log(f"{'Config':>20s}  {'rho':>8s}  {'sigma_0':>8s}  {'R2':>10s}")
    LOG.log("-" * 52)
    for label, _, _ in configs:
        _, popt, r2 = calib_results_5[label]
        if popt is not None:
            LOG.log(f"{label:>20s}  {popt[0]:>8.4f}  {popt[1]:>8.4f}  {r2:>10.6f}")

    # -- Plot 5A: Mean error vs sigma for each config --
    plot_h, plot_w = 450, 700
    font = cv2.FONT_HERSHEY_SIMPLEX
    ox, oy = 70, 50
    pw, ph = plot_w - ox - 40, plot_h - oy - 60

    plot5a = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
    y_min, y_max = -5, 5
    colors_5 = [(0, 160, 0), (200, 100, 0), (0, 0, 200)]

    def to_px5(sigma, err_pct):
        x = int(ox + ((sigma - 1) / 11.0) * pw)
        y = int(oy + ph - ((err_pct - y_min) / (y_max - y_min)) * ph)
        return (x, y)

    cv2.line(plot5a, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
    cv2.line(plot5a, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
    zero_y = to_px5(1, 0)[1]
    cv2.line(plot5a, (ox, zero_y), (ox + pw, zero_y), (180, 180, 180), 1)
    cv2.putText(plot5a, 'Target sigma (px)', (ox + pw // 2 - 60, plot_h - 8), font, 0.4, (0, 0, 0), 1)
    cv2.putText(plot5a, 'Mean error %', (2, oy + ph // 2), font, 0.35, (0, 0, 0), 1)

    for val in range(y_min, y_max + 1, 1):
        pt = to_px5(1, val)
        cv2.putText(plot5a, f'{val:+d}%', (20, pt[1] + 4), font, 0.3, (0, 0, 0), 1)
        cv2.line(plot5a, (ox, pt[1]), (ox + pw, pt[1]), (240, 240, 240), 1)

    for s in sigma_targets:
        pt = to_px5(s, y_min)
        cv2.putText(plot5a, str(s), (pt[0] - 3, oy + ph + 18), font, 0.35, (0, 0, 0), 1)

    # Plot lines
    lx, ly = ox + 10, oy + 10
    for k, (label, _, _) in enumerate(configs):
        color = colors_5[k]
        data = plot_data_5a[label]
        for j in range(1, len(data)):
            s1, m1 = data[j - 1]
            s2, m2 = data[j]
            if not np.isnan(m1) and not np.isnan(m2):
                cv2.line(plot5a, to_px5(s1, m1), to_px5(s2, m2), color, 2)
        for s, m in data:
            if not np.isnan(m):
                cv2.circle(plot5a, to_px5(s, m), 3, color, -1)
        # Legend
        y = ly + k * 16
        cv2.line(plot5a, (lx, y), (lx + 15, y), color, 2)
        cv2.putText(plot5a, label, (lx + 20, y + 4), font, 0.35, (0, 0, 0), 1)

    plot5a = put_title(plot5a, 'Inv5A: Asymmetric Margin -- Mean ERF Error vs Sigma (256x256)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv5a_asymmetric_margin_error.png'), plot5a)

    # -- Plot 5B: sigma vs |z| for each config --
    plot5b = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
    z_plot_max = 9.0
    s_plot_max = 14.0

    def to_px5b(z, s):
        x = int(ox + (z / z_plot_max) * pw)
        y = int(oy + ph - (s / s_plot_max) * ph)
        return (x, y)

    cv2.line(plot5b, (ox, oy), (ox, oy + ph), (0, 0, 0), 1)
    cv2.line(plot5b, (ox, oy + ph), (ox + pw, oy + ph), (0, 0, 0), 1)
    cv2.putText(plot5b, '|z| (mm)', (ox + pw // 2 - 30, plot_h - 8), font, 0.4, (0, 0, 0), 1)
    cv2.putText(plot5b, 'sigma (px)', (2, oy + ph // 2), font, 0.35, (0, 0, 0), 1)

    for val in range(0, int(s_plot_max) + 1, 2):
        pt = to_px5b(0, val)
        cv2.putText(plot5b, str(val), (25, pt[1] + 4), font, 0.3, (0, 0, 0), 1)
        cv2.line(plot5b, (ox, pt[1]), (ox + pw, pt[1]), (240, 240, 240), 1)
    for val in range(0, int(z_plot_max) + 1):
        pt = to_px5b(val, 0)
        cv2.putText(plot5b, str(val), (pt[0] - 3, oy + ph + 18), font, 0.35, (0, 0, 0), 1)

    lx, ly = ox + pw - 200, oy + 15
    for k, (label, _, _) in enumerate(configs):
        color = colors_5[k]
        res, popt, r2 = calib_results_5[label]
        # Data points
        for r in res:
            z = abs(r['z'])
            if not np.isnan(r['sigma_flat']):
                cv2.circle(plot5b, to_px5b(z, r['sigma_flat']), 2, color, -1)
        # Fit line
        if popt is not None:
            for z in np.linspace(0, z_plot_max, 100):
                s = popt[0] * z + popt[1]
                if 0 <= s <= s_plot_max:
                    cv2.circle(plot5b, to_px5b(z, s), 1, color, -1)
        # Legend
        y = ly + k * 16
        cv2.circle(plot5b, (lx, y), 3, color, -1)
        rho_str = f' rho={popt[0]:.3f}' if popt is not None else ''
        cv2.putText(plot5b, f'{label}{rho_str}', (lx + 8, y + 4), font, 0.3, (0, 0, 0), 1)

    plot5b = put_title(plot5b, 'Inv5B: Asymmetric Margin -- Calibration sigma vs |z| (960x960)')
    cv2.imwrite(str(OUTPUT_DIR / 'inv5b_asymmetric_margin_calib.png'), plot5b)

    LOG.log()
    LOG.log(f"Plots saved: inv5a_asymmetric_margin_error.png, inv5b_asymmetric_margin_calib.png")


# ======================================================================
# INVESTIGATION 6: Per-Ray Residual Plots
# ======================================================================

def investigation_6(margin=MARGIN):
    LOG.log()
    LOG.log("=" * 70)
    LOG.log("INVESTIGATION 6: Per-Ray Residual Plots")
    LOG.log("=" * 70)
    LOG.log()
    LOG.log("Purpose: Show that flattening makes the edge profile conform to the")
    LOG.log("Gaussian ERF model. Raw images should show structured residuals from")
    LOG.log("texture; flattened images should show clean random residuals.")
    LOG.log()

    all_crops = load_v_crops_with_metadata()
    all_crops.sort(key=lambda c: c['diameter_px'])
    # Pick 2 crops: one small, one large
    test_crops = [all_crops[0], all_crops[-1]]
    test_sigmas = [10, 12]
    ray_angles = [0, 60, 120, 180, 240, 300]  # 6 evenly spaced rays

    font = cv2.FONT_HERSHEY_SIMPLEX

    for crop_info, sigma_target in zip(test_crops, test_sigmas):
        img_raw = cv2.imread(str(crop_info['path']), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        img = cv2.resize(img_raw, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        centre, radius = detect_sphere(img)
        if centre is None:
            continue

        # Prepare raw blurred and flat blurred
        native_raw, _ = measure_erf_sigma(img)
        flat_img, finfo = flatten_sphere_crop(img, center=centre, radius=radius,
                                              margin_inner=margin, margin_outer=margin)
        if finfo is None:
            continue
        native_flat, _ = measure_erf_sigma(flat_img)

        # Apply blur via quadrature
        ksq_raw = sigma_target**2 - native_raw**2 if not np.isnan(native_raw) else -1
        ksq_flat = sigma_target**2 - native_flat**2 if not np.isnan(native_flat) else -1

        raw_blurred = apply_blur(img, np.sqrt(ksq_raw)) if ksq_raw > 0 else img
        flat_blurred = apply_blur(flat_img, np.sqrt(ksq_flat)) if ksq_flat > 0 else flat_img

        LOG.log(f"  {crop_info['filename']}: sigma_target={sigma_target}, d_model={crop_info['diameter_model']}")

        # Build figure: 6 rays x 2 columns (raw, flat) x 2 rows (profile, residual)
        # Layout: each ray gets a small subplot. Arrange as 6 rows x 2 columns.
        sub_w, sub_h = 350, 140  # each subplot
        n_rays = len(ray_angles)
        fig_w = sub_w * 2 + 40  # 2 columns + gap
        fig_h = sub_h * n_rays + 80  # rows + title

        fig = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

        # Column headers
        cv2.putText(fig, 'RAW (blurred)', (sub_w // 2 - 40, 25), font, 0.5, (0, 0, 200), 1)
        cv2.putText(fig, 'FLATTENED (blurred)', (sub_w + 40 + sub_w // 2 - 60, 25), font, 0.5, (0, 160, 0), 1)

        for ray_idx, angle in enumerate(ray_angles):
            for col_idx, (src_img, col_label, col_color) in enumerate([
                (raw_blurred, 'raw', (0, 0, 200)),
                (flat_blurred, 'flat', (0, 160, 0)),
            ]):
                # Extract ray profile
                r_vals, intensities = extract_single_ray_profile(
                    src_img, centre, radius, angle)

                if len(r_vals) < 10:
                    continue

                # Fit ERF
                fit_result = fit_erf_to_profile(r_vals, intensities)
                if fit_result is None:
                    continue
                popt, r2 = fit_result
                fitted = erf_edge(r_vals, *popt)
                residuals = intensities - fitted

                # Draw subplot
                x_off = col_idx * (sub_w + 40)
                y_off = 35 + ray_idx * sub_h

                # Profile plot (top 70% of sub_h)
                prof_h = int(sub_h * 0.65)
                res_h = sub_h - prof_h

                r_min, r_max_plot = r_vals[0], r_vals[-1]
                i_min, i_max = 0.0, 1.0

                def to_sub(r, val, is_residual=False):
                    x = int(x_off + 35 + ((r - r_min) / (r_max_plot - r_min)) * (sub_w - 45))
                    if is_residual:
                        res_range = 0.1
                        y = int(y_off + prof_h + res_h // 2 - (val / res_range) * (res_h // 2 - 5))
                        y = max(y_off + prof_h + 2, min(y, y_off + sub_h - 2))
                    else:
                        y = int(y_off + prof_h - 5 - ((val - i_min) / (i_max - i_min)) * (prof_h - 15))
                    return (x, y)

                # Profile: data as dots, fit as line
                for j in range(len(r_vals)):
                    pt = to_sub(r_vals[j], intensities[j])
                    cv2.circle(fig, pt, 1, (150, 150, 150), -1)

                for j in range(1, len(r_vals)):
                    p1 = to_sub(r_vals[j - 1], fitted[j - 1])
                    p2 = to_sub(r_vals[j], fitted[j])
                    cv2.line(fig, p1, p2, col_color, 1)

                # Residuals: dots
                # Zero line for residuals
                p_z1 = to_sub(r_vals[0], 0, True)
                p_z2 = to_sub(r_vals[-1], 0, True)
                cv2.line(fig, p_z1, p_z2, (200, 200, 200), 1)

                for j in range(len(r_vals)):
                    pt = to_sub(r_vals[j], residuals[j], True)
                    cv2.circle(fig, pt, 1, col_color, -1)

                # Labels
                cv2.putText(fig, f'{angle}deg sig={popt[3]:.2f} R2={r2:.4f}',
                            (x_off + 5, y_off + 12), font, 0.28, (0, 0, 0), 1)
                cv2.putText(fig, 'resid', (x_off + 5, y_off + prof_h + 12),
                            font, 0.25, (100, 100, 100), 1)

        title = (f"Inv6: Per-Ray ERF Residuals -- {crop_info['filename']}  |  "
                 f"sigma_target={sigma_target}  d={crop_info['diameter_model']}px")
        fig = put_title(fig, title)

        out_name = f"inv6_residuals_sig{sigma_target}_{crop_info['filename'].replace('.png', '')}.png"
        cv2.imwrite(str(OUTPUT_DIR / out_name), fig)

    LOG.log()
    LOG.log(f"Plots saved: inv6_residuals_*.png")


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    LOG.log("Sphere Crop Flattening Investigation")
    LOG.log(f"Margin = {MARGIN} px, Feather = {FEATHER} px, Model size = {MODEL_SIZE}")
    LOG.log(f"Output dir: {OUTPUT_DIR}")
    LOG.log()

    investigation_1()
    investigation_2(margin=MARGIN)
    investigation_3(margin=MARGIN)
    investigation_4(margin=MARGIN)
    investigation_5()
    investigation_6(margin=MARGIN)

    # Save all results to file
    results_file = OUTPUT_DIR / 'results.txt'
    LOG.save(str(results_file))
    print(f"\nAll results saved to: {results_file}")
