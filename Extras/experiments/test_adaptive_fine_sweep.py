"""
Fine-grained margin sweep to find per-crop optimal margin.

Uses narrow sweep range based on scale_px_per_mm:
  Low scale (<60): sweep 2-15px in 1px steps
  High scale (>80): sweep 30-80px in 2px steps
  Mid scale: sweep 10-40px in 2px steps

Tests at sigma=8 and sigma=12 only (high blur — where accuracy matters most).
ContourDistMap flattening only.

Output: per-crop (scale, R, optimal_margin) for refitting the adaptive formula.
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


def flatten_contour_distmap(image, contour, margin, fw=FEATHER):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    dist_out = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    sd = np.where(mask > 0, dist_in, -dist_out)
    out = img_f.copy()
    out[sd > (margin + fw)] = 0.0
    mfi = (sd > margin) & (sd <= (margin + fw))
    if np.any(mfi):
        t = np.clip((sd[mfi] - margin) / fw, 0, 1)
        out[mfi] = 0.5 * (1 + np.cos(np.pi * t)) * img_f[mfi]
    mfo = (sd < -margin) & (sd >= -(margin + fw))
    if np.any(mfo):
        t = np.clip((-sd[mfo] - margin) / fw, 0, 1)
        out[mfo] = img_f[mfo] + 0.5*(1-np.cos(np.pi*t))*(1.0-img_f[mfo])
    out[sd < -(margin + fw)] = 1.0
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


def get_sweep_range(scale, R):
    """Determine sweep range based on camera scale."""
    if scale < 60:
        # G/M camera territory — small margins
        return range(2, min(int(R * 0.4), 20), 1)
    elif scale > 80:
        # V camera territory — large margins
        return range(max(10, int(R * 0.15)), min(int(R * 0.75), 85), 2)
    else:
        # Mid range — interpolate
        return range(5, min(int(R * 0.55), 45), 2)


def main():
    sdf = pd.read_csv(SHARP_CSV)
    sigma_targets = [8, 12]

    print("=" * 100)
    print("FINE-GRAINED MARGIN SWEEP: Finding true optimal per-crop")
    print("  Testing at sigma=8 and sigma=12 (high blur, where accuracy matters)")
    print("  ContourDistMap flattening, narrow sweep per camera scale")
    print("=" * 100)
    print()

    all_results = []  # (cam, fn, scale, R, optimal_margin, optimal_error, interior_max)
    t_start = time.time()

    for cam in ('g', 'v', 'm'):
        cam_df = sdf[sdf['camera'] == cam]
        if len(cam_df) == 0:
            continue

        step = max(1, len(cam_df) // 8)
        sample = cam_df.iloc[::step].head(8)

        print(f"\n  CAMERA {cam.upper()} ({len(sample)} crops)")
        print(f"  {'-'*90}")

        for ci, (_, row) in enumerate(sample.iterrows()):
            fn = row['filename']
            scale = float(row['scale_px_per_mm']) if pd.notna(row.get('scale_px_per_mm')) else None
            if scale is None:
                continue

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
            sweep = get_sweep_range(scale, R)
            sweep_list = list(sweep)

            # Pre-compute distance map once (expensive part)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [det['contour']], 0, 255, -1)
            dist_out = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5).astype(np.float32)
            dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            sd = np.where(mask > 0, dist_in, -dist_out)

            best_margin = 0
            best_err = 999
            best_im = 999
            sweep_data = []

            for margin in sweep_list:
                if R - margin <= 0:
                    continue

                # Flatten using pre-computed distance map
                fw = FEATHER
                out = img.copy()
                out[sd > (margin + fw)] = 0.0
                mfi = (sd > margin) & (sd <= (margin + fw))
                if np.any(mfi):
                    t = np.clip((sd[mfi] - margin) / fw, 0, 1)
                    out[mfi] = 0.5 * (1 + np.cos(np.pi * t)) * img[mfi]
                mfo = (sd < -margin) & (sd >= -(margin + fw))
                if np.any(mfo):
                    t = np.clip((-sd[mfo] - margin) / fw, 0, 1)
                    out[mfo] = img[mfo] + 0.5*(1-np.cos(np.pi*t))*(1.0-img[mfo])
                out[sd < -(margin + fw)] = 1.0

                im = measure_interior_max(out, det['cx'], det['cy'], R)
                native = measure_erf(out)

                # Test at sigma targets
                errs = []
                for st in sigma_targets:
                    if np.isnan(native) or st**2 <= native**2:
                        continue
                    ks = np.sqrt(st**2 - native**2)
                    blurred = apply_blur(out, ks)
                    measured = measure_erf(blurred)
                    if not np.isnan(measured):
                        err = 100 * (measured - st) / st
                        if err > -50:
                            errs.append(err)

                mean_err = np.mean(errs) if errs else float('nan')
                abs_err = abs(mean_err) if not np.isnan(mean_err) else 999

                sweep_data.append((margin, mean_err, im, native))

                # Track best (lowest |error| with acceptable interior)
                if abs_err < best_err and im < 0.15:
                    best_err = abs_err
                    best_margin = margin
                    best_im = im

            # Report
            pct = 100 * best_margin / R if R > 0 else 0
            print(f"\n  [{ci+1}/{len(sample)}] {fn}  R={R}  scale={scale:.1f} px/mm")
            print(f"    Sweep: {sweep_list[0]}-{sweep_list[-1]}px ({len(sweep_list)} values)")
            print(f"    Optimal: margin={best_margin}px ({pct:.0f}%R)  |error|={best_err:.2f}%  interior_max={best_im:.3f}")

            # Show the error curve
            print(f"    {'Margin':>8s}  {'%R':>5s}  {'Error':>8s}  {'IntMax':>7s}  {'Native':>7s}")
            for margin, err, im, native in sweep_data:
                marker = " <-- BEST" if margin == best_margin else ""
                e_str = f"{err:+.2f}%" if not np.isnan(err) else "N/A"
                n_str = f"{native:.2f}" if not np.isnan(native) else "N/A"
                clean = "CLEAN" if im < 0.01 else ("partial" if im < 0.05 else "")
                print(f"    {margin:>7d}px  {100*margin/R:>4.0f}%  {e_str:>8s}  {im:>7.3f}  {n_str:>7s}  {clean}{marker}")

            all_results.append({
                'cam': cam, 'fn': fn, 'scale': scale, 'R': R,
                'optimal_margin': best_margin, 'optimal_pct': pct,
                'optimal_error': best_err, 'optimal_im': best_im,
            })

    elapsed = time.time() - t_start

    # Refit the formula using all per-crop data points
    print(f"\n\n{'='*100}")
    print(f"REFITTING ADAPTIVE FORMULA ({elapsed:.0f}s)")
    print(f"{'='*100}")

    if not all_results:
        print("No results.")
        return

    rdf = pd.DataFrame(all_results)
    rdf.to_csv(Path(__file__).parent / 'margin_v3_output' / 'fine_sweep_results.csv', index=False)

    print(f"\n  {len(rdf)} data points (was 3 camera averages, now per-crop)")
    print()

    # margin_fraction = optimal_margin / R
    rdf['margin_frac'] = rdf['optimal_margin'] / rdf['R']

    print(f"  {'Cam':>3s}  {'Crop':>25s}  {'Scale':>6s}  {'R':>4s}  {'OptM':>5s}  {'%R':>5s}  {'Err':>7s}")
    print(f"  {'-'*65}")
    for _, r in rdf.iterrows():
        print(f"  {r['cam']:>3s}  {r['fn']:>25s}  {r['scale']:>5.0f}  {r['R']:>4.0f}  "
              f"{r['optimal_margin']:>4.0f}px  {r['optimal_pct']:>4.0f}%  {r['optimal_error']:>+6.2f}%")

    # Fit: margin_fraction = a * scale^2 + b
    scales = rdf['scale'].values
    fracs = rdf['margin_frac'].values

    A = np.vstack([scales**2, np.ones(len(scales))]).T
    result = np.linalg.lstsq(A, fracs, rcond=None)
    a, b = result[0]

    predicted = a * scales**2 + b
    residuals = fracs - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fracs - np.mean(fracs))**2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n  ORIGINAL formula (3 points):  margin_frac = 0.0000334 * scale^2 + 0.094")
    print(f"  REFINED formula ({len(rdf)} points):  margin_frac = {a:.7f} * scale^2 + {b:.4f}")
    print(f"  R-squared: {r2:.4f}")

    # Compare predictions
    print(f"\n  {'Cam':>3s}  {'Scale':>6s}  {'R':>4s}  {'Actual':>7s}  {'Old pred':>9s}  {'New pred':>9s}")
    print(f"  {'-'*50}")
    for _, r in rdf.iterrows():
        old_pred = (0.0000334 * r['scale']**2 + 0.094) * r['R']
        new_pred = (a * r['scale']**2 + b) * r['R']
        print(f"  {r['cam']:>3s}  {r['scale']:>5.0f}  {r['R']:>4.0f}  "
              f"{r['optimal_margin']:>6.0f}px  {old_pred:>8.1f}px  {new_pred:>8.1f}px")

    # Also try linear in scale
    A2 = np.vstack([scales, np.ones(len(scales))]).T
    result2 = np.linalg.lstsq(A2, fracs, rcond=None)
    a2, b2 = result2[0]
    pred2 = a2 * scales + b2
    r2_2 = 1 - np.sum((fracs - pred2)**2) / ss_tot

    print(f"\n  Linear fit: margin_frac = {a2:.6f} * scale + {b2:.4f}  (R2={r2_2:.4f})")

    # Power law
    log_s = np.log(scales)
    log_f = np.log(fracs)
    A3 = np.vstack([log_s, np.ones(len(log_s))]).T
    result3 = np.linalg.lstsq(A3, log_f, rcond=None)
    p, log_a3 = result3[0]
    a3 = np.exp(log_a3)
    pred3 = a3 * scales**p
    r2_3 = 1 - np.sum((fracs - pred3)**2) / ss_tot

    print(f"  Power law: margin_frac = {a3:.7f} * scale^{p:.3f}  (R2={r2_3:.4f})")

    print(f"\n  Best fit model: ", end="")
    if r2 >= r2_2 and r2 >= r2_3:
        print(f"Quadratic (R2={r2:.4f})")
        print(f"  margin = ({a:.7f} * scale^2 + {b:.4f}) * R")
    elif r2_2 >= r2_3:
        print(f"Linear (R2={r2_2:.4f})")
        print(f"  margin = ({a2:.6f} * scale + {b2:.4f}) * R")
    else:
        print(f"Power law (R2={r2_3:.4f})")
        print(f"  margin = {a3:.7f} * scale^{p:.3f} * R")

    print("\nDone.")


if __name__ == '__main__':
    main()
