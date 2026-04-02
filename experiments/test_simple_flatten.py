"""
Test simple flattening: detect boundary, interior=0, exterior=1, small feather.
No adaptive margin, no formula. Just Otsu detection and hard flatten.

2 crops per material/camera folder.
Measures: raw, raw+blur, flattened native, flattened+blur.
"""

import sys, cv2, numpy as np, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'calibration'))
from blur_measurement import detect_sphere, measure_blur_erf

CROP_BASE = Path(r'C:\Users\justi\Downloads\coursework\coursework\preprocessing\Preprocessing\OUTPUTNEW')
MODEL_SIZE = 256
FEATHER_PX = 3  # tiny feather just for smoothness


def detect_and_flatten(image, feather=FEATHER_PX):
    """
    Simple flatten: Otsu threshold, set interior=0, exterior=1, small feather.
    No margin — the Otsu boundary IS the edge.
    """
    img_f = image.astype(np.float32) if image.dtype == np.float32 else image.astype(np.float32) / 255.0

    # Canny edge detection to find sphere boundary
    img_u8 = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return img_f.copy(), None

    # The sphere is at the center of the crop. Find whichever contour
    # contains the center point — that's the sphere boundary.
    h, w = img_f.shape[:2]
    center_pt = (w // 2, h // 2)

    cnt = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.pointPolygonTest(c, center_pt, False) >= 0:
            cnt = c
            break

    if cnt is None:
        # Fallback: largest contour
        cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return img_f.copy(), None

    # Centroid and radius
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return img_f.copy(), None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    radius = int(round(np.mean(dists)))

    # Build filled mask using fillPoly (guaranteed to fill, unlike drawContours on thin edges)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [cnt.reshape(-1, 2)], 255)
    dist_out = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    sd = np.where(mask > 0, dist_in, -dist_out)

    out = img_f.copy()
    fw = max(feather, 1)

    # Interior: everything inside contour beyond feather -> 0
    out[sd > fw] = 0.0
    # Inner feather (0 to fw from boundary, inside)
    mfi = (sd > 0) & (sd <= fw)
    if np.any(mfi):
        t = np.clip(sd[mfi] / fw, 0, 1)
        out[mfi] = (1 - t) * img_f[mfi]  # blend from original to 0

    # Exterior: everything outside contour beyond feather -> 1
    out[sd < -fw] = 1.0
    # Outer feather (0 to fw from boundary, outside)
    mfo = (sd < 0) & (sd >= -fw)
    if np.any(mfo):
        t = np.clip(-sd[mfo] / fw, 0, 1)
        out[mfo] = img_f[mfo] + t * (1.0 - img_f[mfo])  # blend from original to 1

    return out, {'cx': cx, 'cy': cy, 'radius': radius, 'feather': fw}


def measure_erf(image):
    c, r = detect_sphere(image)
    if c is None:
        return float('nan')
    return measure_blur_erf(image, c, r, num_rays=36).sigma


def make_kernel(sigma):
    r = int(np.ceil(4.0 * sigma)); s = 2*r+1; ax = np.arange(s)-r
    X, Y = np.meshgrid(ax, ax)
    k = np.exp(-(X**2+Y**2)/(2*sigma**2)); k /= k.sum()
    return k.astype(np.float32)

def apply_blur(image, sigma):
    if sigma <= 0.5: return image.copy()
    return cv2.filter2D(image, -1, make_kernel(sigma), borderType=cv2.BORDER_REPLICATE)

def interior_max(image, cx, cy, radius):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    mask = dist < (radius * 0.5)
    if not np.any(mask): return float('nan')
    return float(np.max(image[mask]))


def main():
    print("=" * 100)
    print("SIMPLE FLATTEN TEST: Otsu boundary, interior=0, exterior=1, feather=3px")
    print("  No adaptive margin. No formula. Just detect and flatten.")
    print("  2 crops per material/camera. Quadrature blur test at sigma=4,8,12.")
    print("=" * 100)
    print()

    sigma_targets = [4, 8, 12]

    # Collect 2 crops per material/camera
    test_crops = []
    for material_dir in sorted(CROP_BASE.iterdir()):
        if not material_dir.is_dir() or material_dir.name == 'Focus':
            continue
        for cam_dir in sorted(material_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            crops_dir = cam_dir / 'crops'
            if not crops_dir.exists():
                continue
            all_crops = sorted(crops_dir.glob('*_crop.png'))
            if len(all_crops) >= 2:
                step = max(1, len(all_crops) // 2)
                test_crops.extend([all_crops[0], all_crops[step]])
            elif all_crops:
                test_crops.append(all_crops[0])

    print(f"Testing {len(test_crops)} crops")
    print()

    results = []
    t_start = time.time()

    for ci, cp in enumerate(test_crops):
        rel = cp.relative_to(CROP_BASE)
        cam = rel.parts[1] if len(rel.parts) >= 2 else '?'

        img = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
        img_f = img.astype(np.float32) / 255.0

        # Raw measurements
        raw_native = measure_erf(img_f)

        raw_errs = []
        for st in sigma_targets:
            if np.isnan(raw_native) or st**2 <= raw_native**2:
                continue
            ks = np.sqrt(st**2 - raw_native**2)
            blurred = apply_blur(img_f, ks)
            measured = measure_erf(blurred)
            if not np.isnan(measured):
                err = 100 * (measured - st) / st
                if err > -50:
                    raw_errs.append(err)

        # Flatten
        flat, info = detect_and_flatten(img_f)

        if info is None:
            print(f"  [{ci+1}/{len(test_crops)}] {str(rel):>55s}  FLATTEN FAILED")
            continue

        im = interior_max(flat, info['cx'], info['cy'], info['radius'])
        flat_native = measure_erf(flat)

        flat_errs = []
        for st in sigma_targets:
            if np.isnan(flat_native) or st**2 <= flat_native**2:
                continue
            ks = np.sqrt(st**2 - flat_native**2)
            blurred = apply_blur(flat, ks)
            measured = measure_erf(blurred)
            if not np.isnan(measured):
                err = 100 * (measured - st) / st
                if err > -50:
                    flat_errs.append(err)

        raw_q = f"{np.mean(raw_errs):+.1f}%" if raw_errs else "N/A"
        flat_q = f"{np.mean(flat_errs):+.1f}%" if flat_errs else "N/A"
        rn = f"{raw_native:.2f}" if not np.isnan(raw_native) else "N/A"
        fn = f"{flat_native:.2f}" if not np.isnan(flat_native) else "N/A"
        clean = "CLEAN" if im < 0.01 else ("partial" if im < 0.05 else "DIRTY")

        improved = ""
        if raw_errs and flat_errs:
            if abs(np.mean(flat_errs)) < abs(np.mean(raw_errs)):
                improved = "BETTER"
            else:
                improved = "worse"

        print(f"  [{ci+1:>2d}/{len(test_crops)}] {str(rel):>55s}  cam={cam}  R={info['radius']:>3d}")
        print(f"           Raw:   native={rn:>5s}  quad={raw_q:>8s}")
        print(f"           Flat:  native={fn:>5s}  quad={flat_q:>8s}  interior={clean} ({im:.3f})  {improved}")
        print()

        results.append({
            'file': str(rel), 'cam': cam, 'R': info['radius'],
            'raw_native': raw_native, 'flat_native': flat_native,
            'raw_err': np.mean(raw_errs) if raw_errs else float('nan'),
            'flat_err': np.mean(flat_errs) if flat_errs else float('nan'),
            'int_max': im,
        })

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*100}")
    print(f"SUMMARY ({elapsed:.0f}s, {len(results)} crops)")
    print(f"{'='*100}")

    for cam in sorted(set(r['cam'] for r in results)):
        cr = [r for r in results if r['cam'] == cam]
        raw_e = [r['raw_err'] for r in cr if not np.isnan(r['raw_err'])]
        flat_e = [r['flat_err'] for r in cr if not np.isnan(r['flat_err'])]
        ims = [r['int_max'] for r in cr]
        n_clean = sum(1 for im in ims if im < 0.01)
        n_partial = sum(1 for im in ims if 0.01 <= im < 0.05)
        n_dirty = sum(1 for im in ims if im >= 0.05)

        re = f"{np.mean(raw_e):+.1f}%" if raw_e else "N/A"
        fe = f"{np.mean(flat_e):+.1f}%" if flat_e else "N/A"
        print(f"\n  Camera {cam} ({len(cr)} crops):")
        print(f"    Raw quad error:  {re}")
        print(f"    Flat quad error: {fe}")
        print(f"    Interior: {n_clean} CLEAN, {n_partial} partial, {n_dirty} DIRTY")

    # Overall
    all_raw = [r['raw_err'] for r in results if not np.isnan(r['raw_err'])]
    all_flat = [r['flat_err'] for r in results if not np.isnan(r['flat_err'])]
    n_better = sum(1 for r in results if not np.isnan(r['raw_err']) and not np.isnan(r['flat_err']) and abs(r['flat_err']) < abs(r['raw_err']))
    n_worse = sum(1 for r in results if not np.isnan(r['raw_err']) and not np.isnan(r['flat_err']) and abs(r['flat_err']) >= abs(r['raw_err']))
    print(f"\n  Overall:")
    print(f"    Raw:  {np.mean(all_raw):+.1f}%" if all_raw else "    Raw: N/A")
    print(f"    Flat: {np.mean(all_flat):+.1f}%" if all_flat else "    Flat: N/A")
    print(f"    Flattening improved {n_better}/{n_better+n_worse} crops")

    print("\nDone.")


if __name__ == '__main__':
    main()
