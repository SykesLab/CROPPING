"""
Test adaptive margin formula: margin = 0.00374 * R^2 / native_sigma + 0.1353 * R

Compares:
  Raw (no flattening)
  Fixed 14% of radius (current best proportional)
  ADAPTIVE (data-driven from native sigma)

On all 24 crops (8 per camera).
"""

import sys, numpy as np, cv2, time
from pathlib import Path

from paths_config import CROP_BASE, SHARP_CSV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'calibration'))
from blur_measurement import detect_sphere, measure_blur_erf
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
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return None
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    return {'cx': cx, 'cy': cy, 'radius': int(round(np.mean(dists))), 'contour': cnt}


def flatten_contour_distmap(image, contour, margin_inner, margin_outer, fw=FEATHER):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    dist_out = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
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
        out[mfo] = img_f[mfo] + 0.5 * (1 - np.cos(np.pi * t)) * (1.0 - img_f[mfo])
    out[sd < -(margin_outer + fw)] = 1.0
    return out


def make_kernel(sigma):
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


def measure_erf(image):
    c, r = detect_sphere(image)
    if c is None:
        return float('nan')
    return measure_blur_erf(image, c, r, num_rays=36).sigma


def measure_interior_max(image, cx, cy, radius):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist < (radius * 0.5)
    if not np.any(mask):
        return float('nan')
    return float(np.max(image[mask]))


def compute_adaptive_margin(R, native_sigma):
    """Adaptive margin from the hyperbolic fit."""
    if native_sigma <= 0 or np.isnan(native_sigma):
        return int(R * 0.14)  # fallback to 14%
    margin = 0.00374 * R**2 / native_sigma + 0.1353 * R
    return max(int(round(margin)), 2)


def run_blur_test(image, sigma_targets):
    """Run quadrature blur test, return list of errors."""
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
            if err > -50:  # filter ERF failures
                errs.append(err)
    return errs, native


def main():
    import pandas as pd

    print("=" * 100)
    print("ADAPTIVE MARGIN TEST")
    print()
    print("  Formula: margin = 0.00374 * R^2 / native_sigma + 0.1353 * R")
    print("  The margin adapts to both sphere size AND edge sharpness.")
    print("  Sharp edges (low native sigma) get wide margins.")
    print("  Blurry edges (high native sigma) get narrow margins.")
    print("=" * 100)
    print()

    # Load 8 crops per camera
    crops = []
    if SHARP_CSV.exists():
        sdf = pd.read_csv(SHARP_CSV)
        for cam in ('g', 'v', 'm'):
            cam_df = sdf[sdf['camera'] == cam]
            step = max(1, len(cam_df) // 8)
            for _, row in cam_df.iloc[::step].head(8).iterrows():
                cp = None
                if 'crop_path' in row and pd.notna(row['crop_path']):
                    cp = Path(row['crop_path'])
                if cp is None or not cp.exists():
                    if 'folder' in row and 'filename' in row:
                        cp = CROP_BASE / row['folder'] / cam / 'crops' / row['filename']
                if cp and cp.exists():
                    img = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
                        crops.append({'cam': cam, 'fn': cp.name,
                                      'image': img.astype(np.float32) / 255.0})

    sigma_targets = [2, 4, 6, 8, 10, 12]
    results = []
    t_start = time.time()

    for cam in ('g', 'v', 'm'):
        cam_crops = [c for c in crops if c['cam'] == cam]
        if not cam_crops:
            continue

        print(f"\n  CAMERA {cam.upper()} ({len(cam_crops)} crops)")
        print(f"  {'-' * 95}")

        for ci, crop in enumerate(cam_crops):
            img = crop['image']
            fn = crop['fn']

            det = detect_contour(img)
            if det is None:
                print(f"  [{ci+1}/{len(cam_crops)}] {fn}: DETECTION FAILED")
                continue

            R = det['radius']

            # Measure native on RAW (needed for adaptive formula)
            raw_native = measure_erf(img)
            raw_im = measure_interior_max(img, det['cx'], det['cy'], R)

            # Compute adaptive margin
            adaptive_m = compute_adaptive_margin(R, raw_native)
            adaptive_pct = 100 * adaptive_m / R

            # Three approaches
            approaches = {}

            # 1. Raw (no flattening)
            raw_errs, _ = run_blur_test(img, sigma_targets)
            approaches['Raw'] = {
                'errs': raw_errs, 'margin': 0, 'pct': 0,
                'im': raw_im, 'native': raw_native,
            }

            # 2. Fixed 14%
            m14 = max(int(R * 0.14), 2)
            flat14 = flatten_contour_distmap(img, det['contour'], m14, m14)
            f14_errs, f14_native = run_blur_test(flat14, sigma_targets)
            f14_im = measure_interior_max(flat14, det['cx'], det['cy'], R)
            approaches['Fixed 14%'] = {
                'errs': f14_errs, 'margin': m14, 'pct': 100 * m14 / R,
                'im': f14_im, 'native': f14_native,
            }

            # 3. Adaptive
            flat_adp = flatten_contour_distmap(img, det['contour'], adaptive_m, adaptive_m)
            adp_errs, adp_native = run_blur_test(flat_adp, sigma_targets)
            adp_im = measure_interior_max(flat_adp, det['cx'], det['cy'], R)
            approaches['Adaptive'] = {
                'errs': adp_errs, 'margin': adaptive_m, 'pct': adaptive_pct,
                'im': adp_im, 'native': adp_native,
            }

            # Print
            rn_str = f"{raw_native:.2f}" if not np.isnan(raw_native) else "N/A"
            print(f"\n  [{ci+1}/{len(cam_crops)}] {fn}  R={R}  raw_native={rn_str}  "
                  f"-> adaptive margin = {adaptive_m}px ({adaptive_pct:.0f}%R)")

            for name, data in approaches.items():
                errs = data['errs']
                m = data['margin']
                pct = data['pct']
                im = data['im']
                native = data['native']
                clean = "CLEAN" if im < 0.01 else ("partial" if im < 0.05 else "DIRTY")
                e_str = f"{np.mean(errs):+.1f}%" if errs else "N/A"
                n_str = f"{native:.2f}" if not np.isnan(native) else "N/A"
                print(f"    {name:>12s}:  margin={m:>3d}px ({pct:>4.0f}%R)  "
                      f"quad_err={e_str:>8s}  native={n_str:>5s}  "
                      f"interior={clean} (max={im:.3f})")

            results.append({
                'cam': cam, 'fn': fn, 'R': R, 'raw_native': raw_native,
                'adaptive_m': adaptive_m, 'adaptive_pct': adaptive_pct,
                'raw_err': np.mean(approaches['Raw']['errs']) if approaches['Raw']['errs'] else float('nan'),
                'f14_err': np.mean(approaches['Fixed 14%']['errs']) if approaches['Fixed 14%']['errs'] else float('nan'),
                'adp_err': np.mean(approaches['Adaptive']['errs']) if approaches['Adaptive']['errs'] else float('nan'),
                'raw_im': approaches['Raw']['im'],
                'f14_im': approaches['Fixed 14%']['im'],
                'adp_im': approaches['Adaptive']['im'],
            })

    # Summary
    elapsed = time.time() - t_start
    print(f"\n\n{'=' * 100}")
    print(f"SUMMARY ({elapsed:.0f}s)")
    print(f"{'=' * 100}")
    print()
    print(f"  {'Camera':>8s}  {'Crops':>5s}  "
          f"{'Raw err':>10s}  "
          f"{'14% err':>10s}  {'14% int':>8s}  "
          f"{'Adapt err':>10s}  {'Adp int':>8s}  {'Adp margin':>15s}")
    print(f"  {'-' * 90}")

    for cam in ('g', 'v', 'm'):
        cam_r = [r for r in results if r['cam'] == cam]
        if not cam_r:
            continue

        n = len(cam_r)
        raw_e = [r['raw_err'] for r in cam_r if not np.isnan(r['raw_err'])]
        f14_e = [r['f14_err'] for r in cam_r if not np.isnan(r['f14_err'])]
        adp_e = [r['adp_err'] for r in cam_r if not np.isnan(r['adp_err'])]
        f14_im = np.mean([r['f14_im'] for r in cam_r])
        adp_im = np.mean([r['adp_im'] for r in cam_r])
        margins = [r['adaptive_m'] for r in cam_r]
        pcts = [r['adaptive_pct'] for r in cam_r]

        re = f"{np.mean(raw_e):+.1f}%" if raw_e else "N/A"
        fe = f"{np.mean(f14_e):+.1f}%" if f14_e else "N/A"
        ae = f"{np.mean(adp_e):+.1f}%" if adp_e else "N/A"
        ms = f"{min(margins)}-{max(margins)}px ({min(pcts):.0f}-{max(pcts):.0f}%R)"

        print(f"  {cam:>8s}  {n:>5d}  "
              f"{re:>10s}  "
              f"{fe:>10s}  {f14_im:>8.3f}  "
              f"{ae:>10s}  {adp_im:>8.3f}  {ms:>15s}")

    # Best per crop
    print()
    print(f"  {'PER-CROP WINNER':>20s}:")
    print(f"  {'Crop':>25s}  {'Raw':>8s}  {'14%':>8s}  {'Adapt':>8s}  {'Winner':>10s}  {'Adapt margin':>15s}")
    print(f"  {'-' * 85}")
    for r in results:
        raw = abs(r['raw_err']) if not np.isnan(r['raw_err']) else 999
        f14 = abs(r['f14_err']) if not np.isnan(r['f14_err']) else 999
        adp = abs(r['adp_err']) if not np.isnan(r['adp_err']) else 999

        winner = 'Raw' if raw < f14 and raw < adp else ('14%' if f14 < adp else 'ADAPTIVE')
        # Penalise dirty interior
        if r['raw_im'] > 0.3:
            winner = '14%' if f14 < adp else 'ADAPTIVE'
        if r['f14_im'] > 0.1 and winner == '14%':
            winner = 'ADAPTIVE'

        re = f"{r['raw_err']:+.1f}%" if not np.isnan(r['raw_err']) else "N/A"
        fe = f"{r['f14_err']:+.1f}%" if not np.isnan(r['f14_err']) else "N/A"
        ae = f"{r['adp_err']:+.1f}%" if not np.isnan(r['adp_err']) else "N/A"

        print(f"  {r['cam']+'/'+r['fn']:>25s}  {re:>8s}  {fe:>8s}  {ae:>8s}  {winner:>10s}  "
              f"{r['adaptive_m']:>3d}px ({r['adaptive_pct']:.0f}%R)")

    print("\nDone.")


if __name__ == '__main__':
    main()
