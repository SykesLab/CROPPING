"""
Feather Width Test: Should feather scale with margin, radius, or stay fixed?

Uses the adaptive margin formula throughout. Only varies the feather width.

Configs:
  Fixed: 3, 5, 8, 10 (current), 15, 20 px
  Proportional to margin: 10%, 20%, 30%, 50% of margin
  Proportional to radius: 2%, 4%, 6%, 8% of R

8 crops per camera, quadrature test at sigma=8 and sigma=12.
"""

import sys, numpy as np, cv2, time
from pathlib import Path
import pandas as pd

from paths_config import CROP_BASE, SHARP_CSV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'calibration'))
from blur_measurement import detect_sphere, measure_blur_erf
OUTPUT_DIR = Path(__file__).parent / 'feather_test_output'
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_SIZE = 256


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


def adaptive_margin(scale, R):
    frac = 3.23e-5 * scale**2 + 0.162
    return max(int(round(frac * R)), 2)


def flatten_contour(image, contour, margin, feather):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image.copy()
    h, w = img_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    dist_out = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    sd = np.where(mask > 0, dist_in, -dist_out)
    out = img_f.copy()
    fw = max(feather, 1)

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


def main():
    sdf = pd.read_csv(SHARP_CSV)
    sigma_targets = [8, 12]

    print("=" * 100)
    print("FEATHER WIDTH TEST")
    print("  Adaptive margin throughout. Only feather varies.")
    print("  Testing: does feather need to scale with margin, radius, or stay fixed?")
    print("=" * 100)
    print()

    # Feather configs: (name, compute_fn)
    # compute_fn takes (margin, R) and returns feather in px
    feather_configs = [
        # Fixed pixel
        ("Fixed 3px",           lambda m, R: 3),
        ("Fixed 5px",           lambda m, R: 5),
        ("Fixed 8px",           lambda m, R: 8),
        ("Fixed 10px (current)", lambda m, R: 10),
        ("Fixed 15px",          lambda m, R: 15),
        ("Fixed 20px",          lambda m, R: 20),
        # Proportional to margin
        ("10% of margin",       lambda m, R: max(int(round(m * 0.10)), 1)),
        ("20% of margin",       lambda m, R: max(int(round(m * 0.20)), 1)),
        ("30% of margin",       lambda m, R: max(int(round(m * 0.30)), 1)),
        ("50% of margin",       lambda m, R: max(int(round(m * 0.50)), 1)),
        # Proportional to radius
        ("2% of R",             lambda m, R: max(int(round(R * 0.02)), 1)),
        ("4% of R",             lambda m, R: max(int(round(R * 0.04)), 1)),
        ("6% of R",             lambda m, R: max(int(round(R * 0.06)), 1)),
        ("8% of R",             lambda m, R: max(int(round(R * 0.08)), 1)),
        # No feather (hard step — worst case reference)
        ("No feather (hard)",   lambda m, R: 0),
    ]

    results = []
    t_start = time.time()

    for cam in ('g', 'v', 'm'):
        cam_df = sdf[sdf['camera'] == cam]
        if len(cam_df) == 0:
            continue

        step = max(1, len(cam_df) // 8)
        sample = cam_df.iloc[::step].head(8)

        print(f"\n  CAMERA {cam.upper()} ({len(sample)} crops)")
        print(f"  {'-'*95}")

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
            margin = adaptive_margin(scale, R)

            # Pre-compute distance map
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [det['contour']], 0, 255, -1)
            dist_out = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5).astype(np.float32)
            dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            sd = np.where(mask > 0, dist_in, -dist_out)

            print(f"\n  [{ci+1}/{len(sample)}] {fn}  R={R}  scale={scale:.0f}  margin={margin}px ({100*margin/R:.0f}%R)")

            for cfg_name, compute_fw in feather_configs:
                fw = compute_fw(margin, R)

                # Flatten with pre-computed distance map
                out = img.copy()
                fw_safe = max(fw, 0.001)  # avoid division by zero

                out[sd > (margin + fw_safe)] = 0.0
                mfi = (sd > margin) & (sd <= (margin + fw_safe))
                if np.any(mfi):
                    t = np.clip((sd[mfi] - margin) / fw_safe, 0, 1)
                    out[mfi] = 0.5 * (1 + np.cos(np.pi * t)) * img[mfi]
                mfo = (sd < -margin) & (sd >= -(margin + fw_safe))
                if np.any(mfo):
                    t = np.clip((-sd[mfo] - margin) / fw_safe, 0, 1)
                    out[mfo] = img[mfo] + 0.5*(1-np.cos(np.pi*t))*(1.0-img[mfo])
                out[sd < -(margin + fw_safe)] = 1.0

                im = measure_interior_max(out, det['cx'], det['cy'], R)
                native = measure_erf(out)

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
                e_str = f"{mean_err:+.1f}%" if not np.isnan(mean_err) else "N/A"
                n_str = f"{native:.2f}" if not np.isnan(native) else "N/A"
                clean = "CLEAN" if im < 0.01 else ("partial" if im < 0.05 else "DIRTY")

                # Feather as % of margin and % of R for context
                fw_pct_m = 100*fw/margin if margin > 0 else 0
                fw_pct_r = 100*fw/R if R > 0 else 0

                print(f"    {cfg_name:>22s}  feather={fw:>3d}px ({fw_pct_m:>3.0f}%M, {fw_pct_r:>3.0f}%R)  "
                      f"quad={e_str:>8s}  native={n_str:>5s}  interior={clean} ({im:.3f})")

                results.append({
                    'cam': cam, 'fn': fn, 'scale': scale, 'R': R,
                    'margin': margin, 'config': cfg_name,
                    'feather_px': fw, 'feather_pct_margin': fw_pct_m,
                    'feather_pct_R': fw_pct_r,
                    'quad_err': mean_err, 'native': native, 'int_max': im,
                })

    elapsed = time.time() - t_start

    # Save CSV
    rdf = pd.DataFrame(results)
    rdf.to_csv(OUTPUT_DIR / 'feather_test_results.csv', index=False)
    print(f"\n\nCSV saved: {OUTPUT_DIR / 'feather_test_results.csv'}")

    # Summary
    clean_df = rdf[(rdf['quad_err'] > -50) | (rdf['quad_err'].isna())]

    print(f"\n{'='*100}")
    print(f"SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*100}")

    print(f"\n  {'Feather config':>22s}  ", end='')
    for cam in ('g', 'v', 'm'):
        print(f"{'cam '+cam+' err':>12s} {'int':>6s}  ", end='')
    print(f"{'Overall err':>12s}")
    print(f"  {'-'*85}")

    for cfg_name, _ in feather_configs:
        parts = []
        for cam in ('g', 'v', 'm'):
            c = clean_df[(clean_df['config']==cfg_name) & (clean_df['cam']==cam)]
            e = c['quad_err'].dropna()
            im = c['int_max'].mean()
            e_str = f"{e.mean():>+10.2f}%" if len(e) > 0 else f"{'N/A':>11s}"
            im_str = f"{im:>6.3f}" if not np.isnan(im) else f"{'N/A':>6s}"
            parts.append(f"{e_str} {im_str}")

        all_e = clean_df[clean_df['config']==cfg_name]['quad_err'].dropna()
        all_str = f"{all_e.mean():>+10.2f}%" if len(all_e) > 0 else f"{'N/A':>11s}"
        print(f"  {cfg_name:>22s}  {'  '.join(parts)}  {all_str}")

    # Per-camera: which feather type works best?
    print(f"\n  Per-camera best feather (lowest |error|):")
    for cam in ('g', 'v', 'm'):
        cam_df = clean_df[clean_df['cam']==cam]
        if len(cam_df) == 0:
            continue
        best_err = 999
        best_cfg = ""
        for cfg_name, _ in feather_configs:
            c = cam_df[cam_df['config']==cfg_name]
            e = c['quad_err'].dropna()
            if len(e) == 0:
                continue
            ae = abs(e.mean())
            if ae < best_err:
                best_err = ae
                best_cfg = cfg_name
        fw_vals = cam_df[cam_df['config']==best_cfg]['feather_px']
        print(f"    {cam}: {best_cfg} (feather={fw_vals.min():.0f}-{fw_vals.max():.0f}px, err={best_err:.2f}%)")

    # Does feather as % of margin give consistent results across cameras?
    print(f"\n  Feather as % of margin — consistency check:")
    for cfg_name, _ in feather_configs:
        if "margin" not in cfg_name:
            continue
        cam_errs = {}
        for cam in ('g', 'v', 'm'):
            c = clean_df[(clean_df['config']==cfg_name) & (clean_df['cam']==cam)]
            e = c['quad_err'].dropna()
            if len(e) > 0:
                cam_errs[cam] = abs(e.mean())
        if cam_errs:
            spread = max(cam_errs.values()) - min(cam_errs.values())
            vals = " ".join(f"{k}={v:.1f}%" for k, v in cam_errs.items())
            print(f"    {cfg_name:>22s}: {vals}  (spread={spread:.1f}%)")

    print("\nDone.")


if __name__ == '__main__':
    main()
