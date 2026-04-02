"""
Margin Investigation Script: Should margins be radius-proportional? Should inner != outer?

Loads crops from all three cameras (v, g, m) at different scales, then tests
a matrix of margin configurations and measures the effect on:
  1. ERF sigma accuracy (does changing margin corrupt the blur measurement?)
  2. Interior cleanliness (does the caustic/texture get removed?)
  3. Background cleanliness (does the illumination gradient get removed?)
  4. Transition zone preservation (is the edge information intact?)

Configurations tested:
  - Fixed pixel margins: 20, 35, 50 px (current default)
  - Radius-proportional: 5%, 10%, 14%, 20%, 30% of radius
  - Asymmetric: inner < outer (aggressive interior cleanup)
  - Asymmetric: inner > outer (conservative interior)

Run with: phantom_env\Scripts\python.exe investigate_margins.py
"""

import sys
import csv
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Add calibration module to path
CALIB_DIR = str(Path(__file__).parent.parent.parent / 'calibration')
if CALIB_DIR not in sys.path:
    sys.path.insert(0, CALIB_DIR)

from blur_measurement import detect_sphere, measure_blur_erf
from sphere_processing import find_sphere_center

# -- Paths ---------------------------------------------------------------------

CROP_BASE = Path(r'C:\Users\justi\Downloads\coursework\coursework\preprocessing\Preprocessing\OUTPUTNEW')
SHARP_CSV = CROP_BASE / 'Focus' / 'sharp_crops.csv'
OUTPUT_DIR = Path(__file__).parent / 'margin_investigation_output'
OUTPUT_DIR.mkdir(exist_ok=True)

FEATHER = 10


# -- Flatten with configurable margins ----------------------------------------

def flatten_sphere_crop(image, center=None, radius=None,
                        margin_inner=50, margin_outer=50,
                        feather_width=FEATHER):
    """Flatten with explicit inner/outer margins."""
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)

    if center is None or radius is None:
        result = find_sphere_center(img_f)
        if result is None:
            # Fallback to detect_sphere
            center, radius = detect_sphere(img_f)
            if center is None:
                return img_f.copy(), None
        else:
            cx, cy, radius = result
            center = (int(cx), int(cy))
            radius = int(radius)

    cx, cy = center
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

    return out, {'center': center, 'radius': radius,
                 'margin_inner': margin_inner, 'margin_outer': margin_outer}


# -- Measurement helpers -------------------------------------------------------

def measure_erf_sigma(image):
    """ERF sigma measurement. Returns (sigma, confidence) or (nan, nan)."""
    centre, radius = detect_sphere(image)
    if centre is None:
        return float('nan'), float('nan')
    result = measure_blur_erf(image, centre, radius, num_rays=36)
    return result.sigma, result.confidence


def measure_interior_variance(image, center, radius, fraction=0.5):
    """Measure intensity variance in the deep interior (within fraction * radius).
    Lower = cleaner flattening (caustic removed)."""
    cx, cy = center
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    interior_mask = dist < (radius * fraction)
    if not np.any(interior_mask):
        return float('nan')
    return float(np.var(image[interior_mask]))


def measure_background_variance(image, center, radius, margin=1.5):
    """Measure intensity variance in the background (beyond margin * radius).
    Lower = cleaner flattening (gradient removed)."""
    cx, cy = center
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    bg_mask = dist > (radius * margin)
    if not np.any(bg_mask):
        return float('nan')
    return float(np.var(image[bg_mask]))


def measure_interior_max(image, center, radius, fraction=0.5):
    """Max intensity in deep interior. Should be ~0 after good flattening."""
    cx, cy = center
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    interior_mask = dist < (radius * fraction)
    if not np.any(interior_mask):
        return float('nan')
    return float(np.max(image[interior_mask]))


# -- Visual helpers ------------------------------------------------------------

def gray_to_bgr(img):
    if img.dtype != np.uint8:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def put_text_block(vis, lines, start_y=15, font_scale=0.35, color=(0, 255, 0)):
    """Put multiple lines of text on an image."""
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (5, start_y + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    return vis


def draw_zones(vis, cx, cy, r, margin_inner, margin_outer, feather):
    """Draw zone boundaries."""
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 1)  # detected edge
    cv2.circle(vis, (int(cx), int(cy)), max(0, int(r - margin_inner)), (255, 255, 0), 1)  # inner
    cv2.circle(vis, (int(cx), int(cy)), int(r + margin_outer), (0, 0, 255), 1)  # outer
    return vis


# -- Load crops ----------------------------------------------------------------

def load_crops_by_camera(max_per_camera=5):
    """Load a sample of crops from each camera, with metadata."""
    crops = {'g': [], 'v': [], 'm': []}

    if not SHARP_CSV.exists():
        print(f"  sharp_crops.csv not found at {SHARP_CSV}")
        print("  Falling back to scanning OUTPUTNEW directly...")

        for cam in ('g', 'v', 'm'):
            cam_crops = sorted(CROP_BASE.rglob(f'{cam}/crops/*_crop.png'))
            step = max(1, len(cam_crops) // max_per_camera)
            for path in cam_crops[::step][:max_per_camera]:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    crops[cam].append({'path': path, 'image': img, 'filename': path.name})
        return crops

    import pandas as pd
    df = pd.read_csv(SHARP_CSV)

    for cam in ('g', 'v', 'm'):
        cam_df = df[df['camera'] == cam]
        if len(cam_df) == 0:
            continue

        # Sample evenly across the range
        step = max(1, len(cam_df) // max_per_camera)
        sample = cam_df.iloc[::step].head(max_per_camera)

        for _, row in sample.iterrows():
            # Try crop_path first, then construct from folder/filename
            crop_path = None
            if 'crop_path' in row and pd.notna(row['crop_path']):
                crop_path = Path(row['crop_path'])

            if crop_path is None or not crop_path.exists():
                if 'folder' in row and 'filename' in row:
                    crop_path = CROP_BASE / row['folder'] / cam / 'crops' / row['filename']

            if crop_path is not None and crop_path.exists():
                img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    crops[cam].append({
                        'path': crop_path,
                        'image': img,
                        'filename': crop_path.name,
                        'scale': float(row.get('scale_px_per_mm', 0)) if 'scale_px_per_mm' in row else 0,
                    })

    return crops


# -- Margin configurations to test --------------------------------------------

@dataclass
class MarginConfig:
    name: str
    inner: int  # computed per-crop if proportional
    outer: int
    is_proportional: bool = False
    inner_frac: float = 0.0
    outer_frac: float = 0.0

    def compute(self, radius: int) -> Tuple[int, int]:
        if self.is_proportional:
            return int(radius * self.inner_frac), int(radius * self.outer_frac)
        return self.inner, self.outer


def get_margin_configs() -> List[MarginConfig]:
    """Define the full matrix of margin configurations to test."""
    configs = []

    # Fixed pixel margins
    for px in [15, 25, 35, 50, 70]:
        configs.append(MarginConfig(f"fixed_{px}px", px, px))

    # Radius-proportional (symmetric)
    for frac in [0.05, 0.08, 0.10, 0.14, 0.20, 0.30]:
        pct = int(frac * 100)
        configs.append(MarginConfig(
            f"prop_{pct}pct", 0, 0,
            is_proportional=True, inner_frac=frac, outer_frac=frac))

    # Asymmetric: aggressive interior (small inner), conservative exterior
    for inner_f, outer_f in [(0.05, 0.14), (0.08, 0.14), (0.05, 0.20), (0.10, 0.20)]:
        ip, op = int(inner_f*100), int(outer_f*100)
        configs.append(MarginConfig(
            f"asym_in{ip}_out{op}", 0, 0,
            is_proportional=True, inner_frac=inner_f, outer_frac=outer_f))

    # Asymmetric: conservative interior, aggressive exterior
    for inner_f, outer_f in [(0.14, 0.05), (0.20, 0.08), (0.14, 0.08)]:
        ip, op = int(inner_f*100), int(outer_f*100)
        configs.append(MarginConfig(
            f"asym_in{ip}_out{op}", 0, 0,
            is_proportional=True, inner_frac=inner_f, outer_frac=outer_f))

    return configs


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MARGIN INVESTIGATION: Fixed vs Proportional, Symmetric vs Asymmetric")
    print("=" * 70)
    print()

    # Load sample crops from each camera
    print("Loading sample crops from each camera...")
    crops = load_crops_by_camera(max_per_camera=6)
    for cam in ('g', 'v', 'm'):
        print(f"  Camera {cam}: {len(crops[cam])} crops loaded")
    print()

    configs = get_margin_configs()
    print(f"Testing {len(configs)} margin configurations:")
    for c in configs:
        if c.is_proportional:
            print(f"  {c.name}: inner={c.inner_frac*100:.0f}%R, outer={c.outer_frac*100:.0f}%R")
        else:
            print(f"  {c.name}: inner={c.inner}px, outer={c.outer}px")
    print()

    # Run tests
    results = []
    all_visual_data = []  # For generating comparison panels

    for cam in ('g', 'v', 'm'):
        for crop_info in crops[cam]:
            img = crop_info['image']
            img_f = img.astype(np.float32) / 255.0

            # Detect sphere
            result = find_sphere_center(img_f)
            if result is None:
                c_r = detect_sphere(img_f)
                if c_r[0] is None:
                    print(f"  SKIP {crop_info['filename']}: detection failed")
                    continue
                center, radius = c_r
            else:
                cx, cy, radius = result
                center = (int(cx), int(cy))
                radius = int(radius)

            # Baseline measurements (raw, no flattening)
            raw_erf, raw_conf = measure_erf_sigma(img_f)
            raw_int_var = measure_interior_variance(img_f, center, radius)
            raw_bg_var = measure_background_variance(img_f, center, radius)
            raw_int_max = measure_interior_max(img_f, center, radius)

            print(f"\n  {cam}/{crop_info['filename']}: R={radius}px, "
                  f"raw_erf={raw_erf:.3f}, int_var={raw_int_var:.6f}, "
                  f"bg_var={raw_bg_var:.6f}, int_max={raw_int_max:.4f}")

            crop_visuals = []

            for config in configs:
                m_inner, m_outer = config.compute(radius)

                # Skip if margins would make r_inner negative
                if radius - m_inner <= 0:
                    continue

                flat, info = flatten_sphere_crop(
                    img_f, center=center, radius=radius,
                    margin_inner=m_inner, margin_outer=m_outer)

                if info is None:
                    continue

                # Measure on flattened image
                flat_erf, flat_conf = measure_erf_sigma(flat)
                flat_int_var = measure_interior_variance(flat, center, radius)
                flat_bg_var = measure_background_variance(flat, center, radius)
                flat_int_max = measure_interior_max(flat, center, radius)

                # ERF change from raw
                erf_delta = flat_erf - raw_erf if not (np.isnan(flat_erf) or np.isnan(raw_erf)) else float('nan')
                erf_pct = 100 * erf_delta / raw_erf if not np.isnan(erf_delta) and raw_erf > 0 else float('nan')

                row = {
                    'camera': cam,
                    'filename': crop_info['filename'],
                    'radius': radius,
                    'config': config.name,
                    'margin_inner_px': m_inner,
                    'margin_outer_px': m_outer,
                    'inner_frac': m_inner / radius,
                    'outer_frac': m_outer / radius,
                    # Raw baseline
                    'raw_erf': raw_erf,
                    'raw_int_var': raw_int_var,
                    'raw_bg_var': raw_bg_var,
                    'raw_int_max': raw_int_max,
                    # Flattened
                    'flat_erf': flat_erf,
                    'flat_int_var': flat_int_var,
                    'flat_bg_var': flat_bg_var,
                    'flat_int_max': flat_int_max,
                    # Deltas
                    'erf_delta': erf_delta,
                    'erf_pct': erf_pct,
                    'int_var_reduction': 1 - flat_int_var / raw_int_var if raw_int_var > 0 else float('nan'),
                    'bg_var_reduction': 1 - flat_bg_var / raw_bg_var if raw_bg_var > 0 else float('nan'),
                }
                results.append(row)

                fmt_erf = f"{flat_erf:.3f}" if not np.isnan(flat_erf) else "FAIL"
                fmt_delta = f"{erf_pct:+.2f}%" if not np.isnan(erf_pct) else "--"
                print(f"    {config.name:>20s}: m_in={m_inner:>3d} m_out={m_outer:>3d} | "
                      f"erf={fmt_erf} ({fmt_delta}) | "
                      f"int_var={flat_int_var:.6f} int_max={flat_int_max:.4f} | "
                      f"bg_var={flat_bg_var:.6f}")

                crop_visuals.append((config.name, flat, m_inner, m_outer, flat_erf, flat_int_var, flat_int_max))

            # Save visual comparison for this crop (select representative configs)
            if crop_visuals:
                _save_crop_visual(cam, crop_info, img_f, center, radius,
                                  raw_erf, raw_int_var, raw_int_max, crop_visuals)

    # Save results CSV
    if results:
        csv_path = OUTPUT_DIR / 'margin_investigation_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved: {csv_path}")

    # Summary analysis
    _print_summary(results)


def _save_crop_visual(cam, crop_info, img_f, center, radius,
                      raw_erf, raw_int_var, raw_int_max, visuals):
    """Save a comparison panel for one crop showing selected configurations."""
    # Pick: raw, fixed_50, prop_10pct, prop_14pct, asym_in5_out14, asym_in8_out14
    target_names = ['fixed_50px', 'fixed_15px', 'prop_5pct', 'prop_10pct',
                    'prop_14pct', 'prop_20pct', 'asym_in5_out14', 'asym_in8_out14']
    selected = []
    for name in target_names:
        for v in visuals:
            if v[0] == name:
                selected.append(v)
                break

    if not selected:
        selected = visuals[:6]

    # Build panel: raw + selected configs
    size = img_f.shape[0]
    panels = []

    # Raw
    raw_vis = gray_to_bgr(img_f)
    draw_zones(raw_vis, center[0], center[1], radius, 0, 0, 0)
    put_text_block(raw_vis, [
        f"RAW (no flatten)",
        f"R={radius} erf={raw_erf:.3f}",
        f"int_var={raw_int_var:.6f}",
        f"int_max={raw_int_max:.4f}",
    ])
    panels.append(raw_vis)

    for name, flat, m_inner, m_outer, erf_val, int_var, int_max in selected:
        vis = gray_to_bgr(flat)
        draw_zones(vis, center[0], center[1], radius, m_inner, m_outer, FEATHER)
        erf_str = f"{erf_val:.3f}" if not np.isnan(erf_val) else "FAIL"
        put_text_block(vis, [
            f"{name}",
            f"in={m_inner} out={m_outer}",
            f"erf={erf_str}",
            f"int_var={int_var:.6f}",
            f"int_max={int_max:.4f}",
        ])
        panels.append(vis)

    combined = np.hstack(panels)

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_h = 25
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    title = f"{cam}/{crop_info['filename']}  R={radius}px  ({size}x{size})"
    cv2.putText(bar, title, (5, 18), font, 0.5, (255, 255, 255), 1)
    combined = np.vstack([bar, combined])

    out_name = f"margin_{cam}_{crop_info['filename'].replace('.png', '')}.png"
    cv2.imwrite(str(OUTPUT_DIR / out_name), combined)


def _print_summary(results):
    """Print aggregate analysis to help make the decision."""
    import pandas as pd

    if not results:
        print("\nNo results to summarise.")
        return

    df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    # Per-camera radius stats
    print("\n--- Sphere radii by camera ---")
    for cam in sorted(df['camera'].unique()):
        cam_df = df[df['camera'] == cam]
        radii = cam_df.drop_duplicates('filename')['radius']
        print(f"  Camera {cam}: R = {radii.min()}-{radii.max()} px "
              f"(mean={radii.mean():.0f}, n={len(radii)} crops)")

    # Fixed margin: what fraction of R does 50px represent per camera?
    print("\n--- What does fixed 50px margin mean per camera? ---")
    for cam in sorted(df['camera'].unique()):
        cam_df = df[df['camera'] == cam]
        radii = cam_df.drop_duplicates('filename')['radius'].values
        fracs = 50.0 / radii
        print(f"  Camera {cam}: 50px = {fracs.min()*100:.1f}%-{fracs.max()*100:.1f}% of R "
              f"(mean={fracs.mean()*100:.1f}%)")

    # Per-config aggregate: mean ERF change, mean interior cleanup
    print("\n--- Config effectiveness (averaged across all crops) ---")
    print(f"{'Config':>22s}  {'ERF d%':>8s}  {'IntVar Red':>10s}  {'BgVar Red':>9s}  "
          f"{'IntMax':>8s}  {'ERF fail':>8s}")
    print("-" * 80)

    for config_name in df['config'].unique():
        cfg = df[df['config'] == config_name]
        erf_pct = cfg['erf_pct'].dropna()
        int_red = cfg['int_var_reduction'].dropna()
        bg_red = cfg['bg_var_reduction'].dropna()
        int_max = cfg['flat_int_max'].dropna()
        erf_fail = cfg['flat_erf'].isna().sum()

        print(f"{config_name:>22s}  "
              f"{erf_pct.mean():>+7.2f}%  "
              f"{int_red.mean()*100:>9.1f}%  "
              f"{bg_red.mean()*100:>8.1f}%  "
              f"{int_max.mean():>8.5f}  "
              f"{erf_fail:>8d}")

    # Per-camera breakdown for key configs
    print("\n--- Per-camera: interior max intensity (lower = caustic removed) ---")
    key_configs = ['fixed_50px', 'fixed_15px', 'prop_10pct', 'prop_14pct',
                   'asym_in5_out14', 'asym_in8_out14']
    for config_name in key_configs:
        cfg = df[df['config'] == config_name]
        if len(cfg) == 0:
            continue
        parts = []
        for cam in sorted(cfg['camera'].unique()):
            cam_cfg = cfg[cfg['camera'] == cam]
            mean_max = cam_cfg['flat_int_max'].mean()
            parts.append(f"{cam}={mean_max:.4f}")
        print(f"  {config_name:>22s}: {', '.join(parts)}")

    # The key question: does proportional margin fix m-camera without hurting v-camera?
    print("\n--- KEY QUESTION: Does proportional margin fix m-camera? ---")
    for config_name in ['fixed_50px', 'prop_10pct', 'prop_14pct', 'asym_in5_out14']:
        cfg = df[df['config'] == config_name]
        if len(cfg) == 0:
            continue
        print(f"\n  {config_name}:")
        for cam in sorted(cfg['camera'].unique()):
            cam_cfg = cfg[cfg['camera'] == cam]
            mean_erf_pct = cam_cfg['erf_pct'].mean()
            mean_int_max = cam_cfg['flat_int_max'].mean()
            mean_int_red = cam_cfg['int_var_reduction'].mean() * 100
            m_in = cam_cfg['margin_inner_px'].iloc[0]
            m_out = cam_cfg['margin_outer_px'].iloc[0]
            r = cam_cfg['radius'].iloc[0]
            print(f"    {cam}: R={r}, margin_in={m_in}px ({m_in/r*100:.1f}%R), "
                  f"margin_out={m_out}px ({m_out/r*100:.1f}%R)")
            print(f"       ERF change: {mean_erf_pct:+.2f}%, "
                  f"interior max: {mean_int_max:.4f}, "
                  f"interior var reduction: {mean_int_red:.1f}%")

    # Asymmetric analysis
    print("\n--- Should inner != outer? ---")
    print("  Rationale: Interior has caustics (need aggressive removal = small inner margin).")
    print("  Background has illumination gradient (also needs removal but edge blur extends outward).")
    print()
    asym_configs = [c for c in df['config'].unique() if 'asym' in c]
    sym_configs = [c for c in df['config'].unique() if 'prop' in c and 'asym' not in c]

    if asym_configs and sym_configs:
        # Compare best symmetric vs best asymmetric
        for ac in asym_configs:
            ac_df = df[df['config'] == ac]
            ac_int = ac_df['flat_int_max'].mean()
            ac_erf = ac_df['erf_pct'].dropna().mean()

            # Find symmetric config with same outer fraction
            ac_outer_frac = ac_df['outer_frac'].iloc[0]
            ac_inner_frac = ac_df['inner_frac'].iloc[0]

            print(f"  {ac}: inner={ac_inner_frac*100:.0f}%R, outer={ac_outer_frac*100:.0f}%R")
            print(f"    Mean int_max={ac_int:.5f}, ERF change={ac_erf:+.2f}%")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Review the visual panels per camera to see the actual effect.")
    print("Done.")


if __name__ == '__main__':
    main()
