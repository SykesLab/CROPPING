"""
Test adaptive margin formula using per-crop scale_px_per_mm.

Formula: margin_fraction = 0.0000334 * scale^2 + 0.094
         margin_px = margin_fraction * R

Uses actual per-crop scale from sharp_crops.csv.
"""

import sys, numpy as np, cv2, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'calibration'))
from blur_measurement import detect_sphere, measure_blur_erf

CROP_BASE = Path(r'C:\Users\justi\Downloads\coursework\coursework\preprocessing\Preprocessing\OUTPUTNEW')
SHARP_CSV = CROP_BASE / 'Focus' / 'sharp_crops.csv'
MODEL_SIZE = 256
FEATHER = 10

def detect_contour(img):
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20: return None
    M = cv2.moments(cnt)
    if M['m00'] == 0: return None
    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    return {'cx': cx, 'cy': cy, 'radius': int(round(np.mean(dists))), 'contour': cnt}

def flatten_contour_distmap(image, contour, margin_inner, margin_outer, fw=FEATHER):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    dist_out = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    sd = np.where(mask > 0, dist_in, -dist_out)
    out = img_f.copy()
    out[sd > (margin_inner + fw)] = 0.0
    mfi = (sd > margin_inner) & (sd <= (margin_inner + fw))
    if np.any(mfi):
        t = np.clip((sd[mfi] - margin_inner) / fw, 0, 1)
        out[mfi] = 0.5 * (1 + np.cos(np.pi * t)) * img_f[mfi]
    mfo = (sd < -margin_outer) & (sd >= -(margin_outer + fw))
    if np.any(mfo):
        t = np.clip((-sd[mfo] - margin_outer) / fw, 0, 1)
        out[mfo] = img_f[mfo] + 0.5*(1-np.cos(np.pi*t))*(1.0-img_f[mfo])
    out[sd < -(margin_outer + fw)] = 1.0
    return out

def make_kernel(sigma):
    r = int(np.ceil(4.0*sigma)); s = 2*r+1; ax = np.arange(s)-r
    X, Y = np.meshgrid(ax, ax)
    k = np.exp(-(X**2+Y**2)/(2*sigma**2)); k /= k.sum()
    return k.astype(np.float32)

def apply_blur(image, sigma):
    if sigma <= 0.5: return image.copy()
    return cv2.filter2D(image, -1, make_kernel(sigma), borderType=cv2.BORDER_REPLICATE)

def measure_erf(image):
    c, r = detect_sphere(image)
    if c is None: return float('nan')
    return measure_blur_erf(image, c, r, num_rays=36).sigma

def measure_interior_max(image, cx, cy, radius):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-cx)**2+(Y-cy)**2)
    mask = dist < (radius * 0.5)
    if not np.any(mask): return float('nan')
    return float(np.max(image[mask]))

def adaptive_margin(scale_px_per_mm, R):
    """Compute margin from camera scale and detected radius."""
    frac = 0.0000334 * scale_px_per_mm**2 + 0.094
    return max(int(round(frac * R)), 2)

def run_blur_test(image, sigma_targets):
    native = measure_erf(image)
    errs = []
    for st in sigma_targets:
        if np.isnan(native) or st**2 <= native**2:
            continue
        ks = np.sqrt(st**2 - native**2)
        blurred = apply_blur(image, ks)
        measured = measure_erf(blurred)
        if not np.isnan(measured):
            err = 100 * (measured - st) / st
            if err > -50:
                errs.append(err)
    return errs, native


def main():
    sdf = pd.read_csv(SHARP_CSV)

    print("=" * 100)
    print("ADAPTIVE MARGIN v2: Using per-crop scale_px_per_mm")
    print()
    print("  Formula: margin = (0.0000334 * scale^2 + 0.094) * R")
    print("  scale = camera resolution in px/mm (from sharp_crops.csv)")
    print("  R = detected sphere radius in current image")
    print("=" * 100)
    print()

    sigma_targets = [4, 6, 8, 10, 12]
    results = []
    t_start = time.time()

    for cam in ('g', 'v', 'm'):
        cam_df = sdf[sdf['camera'] == cam]
        if len(cam_df) == 0:
            continue

        # Sample 8 crops with their actual scale values
        step = max(1, len(cam_df) // 8)
        sample = cam_df.iloc[::step].head(8)

        print(f"\n  CAMERA {cam.upper()} ({len(sample)} crops, "
              f"scale range: {sample['scale_px_per_mm'].min():.1f}-{sample['scale_px_per_mm'].max():.1f} px/mm)")
        print(f"  {'-'*95}")

        for ci, (_, row) in enumerate(sample.iterrows()):
            fn = row['filename']
            scale = float(row['scale_px_per_mm']) if pd.notna(row.get('scale_px_per_mm')) else None
            if scale is None:
                print(f"  [{ci+1}] {fn}: no scale data, skipping")
                continue

            # Load crop
            crop_path = None
            if 'crop_path' in row and pd.notna(row['crop_path']):
                crop_path = Path(row['crop_path'])
            if crop_path is None or not crop_path.exists():
                crop_path = CROP_BASE / row['folder'] / cam / 'crops' / fn
            if not crop_path.exists():
                continue

            img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0

            det = detect_contour(img)
            if det is None:
                print(f"  [{ci+1}] {fn}: detection failed")
                continue

            R = det['radius']

            # Compute adaptive margin using this crop's actual scale
            adp_m = adaptive_margin(scale, R)
            adp_frac = adp_m / R * 100

            # Also compute fixed 14% for comparison
            m14 = max(int(R * 0.14), 2)

            # Raw baseline
            raw_errs, raw_native = run_blur_test(img, sigma_targets)
            raw_im = measure_interior_max(img, det['cx'], det['cy'], R)

            # Fixed 14%
            flat14 = flatten_contour_distmap(img, det['contour'], m14, m14)
            f14_errs, f14_native = run_blur_test(flat14, sigma_targets)
            f14_im = measure_interior_max(flat14, det['cx'], det['cy'], R)

            # Adaptive
            flat_adp = flatten_contour_distmap(img, det['contour'], adp_m, adp_m)
            adp_errs, adp_native = run_blur_test(flat_adp, sigma_targets)
            adp_im = measure_interior_max(flat_adp, det['cx'], det['cy'], R)

            # Print
            def fmt(errs):
                return f"{np.mean(errs):+.1f}%" if errs else "N/A"
            def clean(im):
                return "CLEAN" if im < 0.01 else ("partial" if im < 0.05 else "DIRTY")

            print(f"\n  [{ci+1}/{len(sample)}] {fn}  R={R}  scale={scale:.1f} px/mm")
            print(f"    {'Raw':>12s}: quad={fmt(raw_errs):>8s}  interior={clean(raw_im)} ({raw_im:.3f})")
            print(f"    {'Fixed 14%':>12s}: quad={fmt(f14_errs):>8s}  margin={m14}px ({100*m14/R:.0f}%R)  interior={clean(f14_im)} ({f14_im:.3f})")
            print(f"    {'ADAPTIVE':>12s}: quad={fmt(adp_errs):>8s}  margin={adp_m}px ({adp_frac:.0f}%R)  interior={clean(adp_im)} ({adp_im:.3f})")

            results.append({
                'cam': cam, 'fn': fn, 'R': R, 'scale': scale,
                'adp_m': adp_m, 'adp_frac': adp_frac,
                'raw_err': np.mean(raw_errs) if raw_errs else float('nan'),
                'f14_err': np.mean(f14_errs) if f14_errs else float('nan'),
                'adp_err': np.mean(adp_errs) if adp_errs else float('nan'),
                'raw_im': raw_im, 'f14_im': f14_im, 'adp_im': adp_im,
            })

    elapsed = time.time() - t_start

    # Summary
    print(f"\n\n{'='*100}")
    print(f"SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*100}")
    print(f"\n  {'Camera':>6s}  {'N':>3s}  {'Scale range':>15s}  "
          f"{'Raw err':>10s}  "
          f"{'14% err':>10s} {'14% int':>8s}  "
          f"{'Adapt err':>10s} {'Adp int':>8s}  {'Margin range':>20s}")
    print(f"  {'-'*105}")

    for cam in ('g', 'v', 'm'):
        cr = [r for r in results if r['cam'] == cam]
        if not cr: continue
        n = len(cr)
        scales = [r['scale'] for r in cr]
        raw_e = [r['raw_err'] for r in cr if not np.isnan(r['raw_err'])]
        f14_e = [r['f14_err'] for r in cr if not np.isnan(r['f14_err'])]
        adp_e = [r['adp_err'] for r in cr if not np.isnan(r['adp_err'])]
        f14_im = np.mean([r['f14_im'] for r in cr])
        adp_im = np.mean([r['adp_im'] for r in cr])
        margins = [r['adp_m'] for r in cr]
        pcts = [r['adp_frac'] for r in cr]

        re = f"{np.mean(raw_e):+.1f}%" if raw_e else "N/A"
        fe = f"{np.mean(f14_e):+.1f}%" if f14_e else "N/A"
        ae = f"{np.mean(adp_e):+.1f}%" if adp_e else "N/A"

        print(f"  {cam:>6s}  {n:>3d}  {min(scales):.0f}-{max(scales):.0f} px/mm      "
              f"{re:>10s}  "
              f"{fe:>10s} {f14_im:>8.3f}  "
              f"{ae:>10s} {adp_im:>8.3f}  "
              f"{min(margins)}-{max(margins)}px ({min(pcts):.0f}-{max(pcts):.0f}%R)")

    # Winner per crop
    print(f"\n  Per-crop winner (lowest |error| with acceptable interior):")
    print(f"  {'Crop':>25s}  {'Scale':>6s}  {'R':>3s}  {'Raw':>8s}  {'14%':>8s}  {'Adapt':>8s}  {'Winner':>10s}  {'Margin':>12s}")
    print(f"  {'-'*90}")
    wins = {'Raw': 0, '14%': 0, 'ADAPTIVE': 0}
    for r in results:
        raw = abs(r['raw_err']) if not np.isnan(r['raw_err']) else 999
        f14 = abs(r['f14_err']) if not np.isnan(r['f14_err']) else 999
        adp = abs(r['adp_err']) if not np.isnan(r['adp_err']) else 999

        # Penalise dirty
        if r['raw_im'] > 0.3: raw = 999
        if r['f14_im'] > 0.1: f14 = 999
        if r['adp_im'] > 0.1: adp = 999

        if raw <= f14 and raw <= adp:
            winner = 'Raw'
        elif f14 <= adp:
            winner = '14%'
        else:
            winner = 'ADAPTIVE'
        wins[winner] += 1

        re = f"{r['raw_err']:+.1f}%" if not np.isnan(r['raw_err']) else "N/A"
        fe = f"{r['f14_err']:+.1f}%" if not np.isnan(r['f14_err']) else "N/A"
        ae = f"{r['adp_err']:+.1f}%" if not np.isnan(r['adp_err']) else "N/A"
        print(f"  {r['cam']+'/'+r['fn']:>25s}  {r['scale']:>5.0f}  {r['R']:>3d}  "
              f"{re:>8s}  {fe:>8s}  {ae:>8s}  {winner:>10s}  "
              f"{r['adp_m']}px ({r['adp_frac']:.0f}%R)")

    print(f"\n  Wins: Raw={wins['Raw']}, 14%={wins['14%']}, Adaptive={wins['ADAPTIVE']}")
    print("\nDone.")


if __name__ == '__main__':
    main()
