"""
Concern 1 Test Script: Compare Three Geometry Approaches for Flattening After crop_to_square

Loads the calibration z-stack from raw .cine files (same as calibration GUI),
finds focus via laplacian variance, runs find_consensus_sphere, then for each
frame simulates _apply_sphere_pipeline and compares three flattening methods:

  Method A (original plan): center = (cw//2, ch//2),
           radius = int(min(ch,cw) / (2 * 1.2))

  Method B (our fix): center = (shape//2, shape//2),
           radius = consensus radius (unchanged by crop)

  Method C (ChatGPT): auto-detect per frame on cropped image,
           no geometry passed

Output:
  concern1_output/
    concern1_results.csv
    concern1_summary.txt
    concern1_visual_*.png

Run with: python test_concern1_geometry.py
"""

import sys
import csv
import math
import numpy as np
import cv2
from pathlib import Path

# Add calibration module to path
CALIB_DIR = str(Path(__file__).parent.parent.parent / 'calibration')
if CALIB_DIR not in sys.path:
    sys.path.insert(0, CALIB_DIR)

from blur_measurement import detect_sphere, measure_blur_erf
from sphere_processing import find_sphere_center, find_consensus_sphere, crop_to_square
from cine_loader import CineFolderLoader

from paths_config import CINE_FOLDER

# -- Config --------------------------------------------------------------------

POSITIONS_CSV = CINE_FOLDER / 'positions.csv'
OUTPUT_DIR = Path(__file__).parent / 'concern1_output'
OUTPUT_DIR.mkdir(exist_ok=True)

MARGIN = 50
FEATHER = 10
PADDING = 1.2


# -- Flatten function (local copy — what will go into sphere_processing.py) ----

def flatten_sphere_crop(image, center=None, radius=None,
                        margin_inner=MARGIN, margin_outer=MARGIN,
                        feather_width=FEATHER):
    """
    Flatten sphere interior to 0, background to 1,
    preserve transition zone around edge with cosine feathering.
    Returns: (flattened_image, info_dict_or_None)
    """
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)

    if center is None or radius is None:
        result = find_sphere_center(img_f)
        if result is None:
            return img_f.copy(), None
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


# -- Helpers -------------------------------------------------------------------

def measure_erf_sigma(image):
    """Measure blur sigma via ERF fitting. Returns (sigma, confidence) or (nan, nan)."""
    centre, radius = detect_sphere(image)
    if centre is None:
        return float('nan'), float('nan')
    result = measure_blur_erf(image, centre, radius, num_rays=36)
    return result.sigma, result.confidence


def find_focus_frame(images):
    """Find sharpest frame by laplacian variance (same as calibration GUI)."""
    best_idx = 0
    best_var = -1
    for i, img in enumerate(images):
        if img.dtype != np.uint8:
            img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_u8 = img
        lap_var = cv2.Laplacian(img_u8, cv2.CV_64F).var()
        if lap_var > best_var:
            best_var = lap_var
            best_idx = i
    return best_idx, best_var


def gray_to_bgr(img):
    if img.dtype != np.uint8:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = img
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def put_title(img, text, font_scale=0.5, thickness=1):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    bar_h = th + baseline + 12
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    tx = max(4, (w - tw) // 2)
    ty = th + 6
    cv2.putText(bar, text, (tx, ty), font, font_scale, (255, 255, 255), thickness)
    return np.vstack([bar, img])


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CONCERN 1 TEST: Three Geometry Approaches for Post-Crop Flattening")
    print("=" * 70)
    print()

    # Step 1: Load raw .cine z-stack using CineFolderLoader
    print(f"Loading .cine files from: {CINE_FOLDER}")
    print(f"Positions CSV: {POSITIONS_CSV}")
    loader = CineFolderLoader(str(CINE_FOLDER))
    print(f"  Found {loader.num_files} .cine files")

    # Load with positions CSV, stage_offset=0 initially (we'll compute focus ourselves)
    images, stage_positions, filenames = loader.load_with_positions_csv(
        str(POSITIONS_CSV), stage_offset=0.0, frame_idx=0
    )
    print(f"  Loaded {len(images)} frames")
    print(f"  Stage range: {min(stage_positions):.1f} to {max(stage_positions):.1f} mm")
    print(f"  Image shape: {images[0].shape}")
    print()

    # Step 2: Find focus frame (sharpest by laplacian variance)
    print("Finding focus frame by laplacian variance...")
    focus_idx, focus_var = find_focus_frame(images)
    focus_stage = stage_positions[focus_idx]
    print(f"  Focus frame: index {focus_idx}, file {filenames[focus_idx]}")
    print(f"  Stage position: {focus_stage:.1f} mm, laplacian variance: {focus_var:.1f}")

    # Compute defocus positions (z = stage - focus)
    z_positions = [sp - focus_stage for sp in stage_positions]
    print(f"  Defocus range: {min(z_positions):.1f} to {max(z_positions):.1f} mm")
    print()

    # Step 3: Find consensus sphere (same as production pipeline)
    print("Finding consensus sphere across all frames...")
    consensus = find_consensus_sphere(images, upper_only=True)
    if consensus is None:
        print("ERROR: find_consensus_sphere returned None!")
        return
    con_cx, con_cy, con_radius = consensus
    print(f"  Consensus: cx={con_cx}, cy={con_cy}, radius={con_radius}")
    print()

    # Step 4: For each frame, simulate _apply_sphere_pipeline then test 3 methods
    results = []

    # Pick representative frames for visual output
    z_targets = [0.0, -2.0, -4.0, 2.0, 4.0]
    visual_indices = set()
    for zt in z_targets:
        best_idx = min(range(len(z_positions)),
                       key=lambda i: abs(z_positions[i] - zt))
        visual_indices.add(best_idx)

    print(f"Processing {len(images)} frames...")
    print(f"{'Idx':>4s}  {'z_mm':>6s}  {'file':>16s}  "
          f"{'A_ok':>4s}  {'A_r':>4s}  {'A_sig':>7s}  "
          f"{'B_ok':>4s}  {'B_r':>4s}  {'B_sig':>7s}  "
          f"{'C_ok':>4s}  {'C_r':>4s}  {'C_sig':>7s}  "
          f"{'AB':>7s}  {'AC':>7s}  {'BC':>7s}")
    print("-" * 130)

    for idx in range(len(images)):
        img = images[idx]
        z = z_positions[idx]
        fn = filenames[idx]

        # Simulate _apply_sphere_pipeline: mirror=False, blacken=False, then crop
        cropped = crop_to_square(img, con_cx, con_cy, con_radius, padding=PADDING)

        # Convert to float for flattening
        if cropped.dtype == np.uint8:
            crop_f = cropped.astype(np.float32) / 255.0
        else:
            crop_f = cropped.astype(np.float32)
            if crop_f.max() > 1.0:
                crop_f = crop_f / crop_f.max()

        ch, cw = crop_f.shape[:2]

        # --- Method A: Original plan (reconstruct radius from padding) ---
        a_cx = cw // 2
        a_cy = ch // 2
        a_r = int(min(ch, cw) / (2 * PADDING))
        flat_a, info_a = flatten_sphere_crop(crop_f, center=(a_cx, a_cy), radius=a_r)
        a_ok = info_a is not None
        sig_a, conf_a = measure_erf_sigma(flat_a) if a_ok else (float('nan'), float('nan'))

        # --- Method B: Our fix (use consensus radius directly) ---
        b_cx = cw // 2
        b_cy = ch // 2
        b_r = con_radius
        flat_b, info_b = flatten_sphere_crop(crop_f, center=(b_cx, b_cy), radius=b_r)
        b_ok = info_b is not None
        sig_b, conf_b = measure_erf_sigma(flat_b) if b_ok else (float('nan'), float('nan'))

        # --- Method C: ChatGPT (auto-detect on cropped image) ---
        flat_c, info_c = flatten_sphere_crop(crop_f)
        c_ok = info_c is not None
        if c_ok:
            c_cx_used, c_cy_used = info_c['center']
            c_r_used = info_c['radius']
            sig_c, conf_c = measure_erf_sigma(flat_c)
        else:
            c_cx_used, c_cy_used, c_r_used = -1, -1, -1
            sig_c, conf_c = float('nan'), float('nan')

        # Pairwise sigma differences
        ab = abs(sig_a - sig_b) if not (np.isnan(sig_a) or np.isnan(sig_b)) else float('nan')
        ac = abs(sig_a - sig_c) if not (np.isnan(sig_a) or np.isnan(sig_c)) else float('nan')
        bc = abs(sig_b - sig_c) if not (np.isnan(sig_b) or np.isnan(sig_c)) else float('nan')

        row = {
            'frame_idx': idx, 'filename': fn, 'z_mm': z,
            'stage_mm': stage_positions[idx],
            'crop_h': ch, 'crop_w': cw,
            'consensus_cx': con_cx, 'consensus_cy': con_cy, 'consensus_r': con_radius,
            'a_ok': a_ok, 'a_cx': a_cx, 'a_cy': a_cy, 'a_r': a_r,
            'a_sigma': sig_a, 'a_conf': conf_a,
            'b_ok': b_ok, 'b_cx': b_cx, 'b_cy': b_cy, 'b_r': b_r,
            'b_sigma': sig_b, 'b_conf': conf_b,
            'c_ok': c_ok, 'c_cx': c_cx_used, 'c_cy': c_cy_used, 'c_r': c_r_used,
            'c_sigma': sig_c, 'c_conf': conf_c,
            'ab_diff': ab, 'ac_diff': ac, 'bc_diff': bc,
        }
        results.append(row)

        def fs(s):
            return f"{s:>7.3f}" if not np.isnan(s) else "   FAIL"
        def fd(d):
            return f"{d:>7.4f}" if not np.isnan(d) else "     --"

        print(f"{idx:>4d}  {z:>6.1f}  {fn:>16s}  "
              f"{'  OK' if a_ok else 'FAIL'}  {a_r:>4d}  {fs(sig_a)}  "
              f"{'  OK' if b_ok else 'FAIL'}  {b_r:>4d}  {fs(sig_b)}  "
              f"{'  OK' if c_ok else 'FAIL'}  {c_r_used:>4d}  {fs(sig_c)}  "
              f"{fd(ab)}  {fd(ac)}  {fd(bc)}")

        # Visual panels for representative frames
        if idx in visual_indices:
            panels = []
            # Raw cropped
            raw_vis = gray_to_bgr(crop_f)
            cv2.putText(raw_vis, "Raw cropped", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            panels.append(raw_vis)

            for label, flat_img, ok, cx_u, cy_u, r_u, sig in [
                ('A: Reconstruct', flat_a, a_ok, a_cx, a_cy, a_r, sig_a),
                ('B: Consensus R', flat_b, b_ok, b_cx, b_cy, b_r, sig_b),
                ('C: Auto-detect', flat_c, c_ok, c_cx_used, c_cy_used, c_r_used, sig_c),
            ]:
                vis = gray_to_bgr(flat_img)
                if ok and r_u > 0:
                    cv2.circle(vis, (int(cx_u), int(cy_u)), int(r_u), (0, 255, 0), 1)
                    cv2.circle(vis, (int(cx_u), int(cy_u)), max(0, int(r_u) - MARGIN), (255, 255, 0), 1)
                    cv2.circle(vis, (int(cx_u), int(cy_u)), int(r_u) + MARGIN, (0, 0, 255), 1)

                sig_str = f"sig={sig:.3f}" if not np.isnan(sig) else "FAIL"
                cv2.putText(vis, label, (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(vis, f"c=({cx_u},{cy_u}) r={r_u} {sig_str}", (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                panels.append(vis)

            combined = np.hstack(panels)
            z_str = f"{z:.1f}".replace('-', 'neg').replace('.', 'p')
            title = (f"z={z:.1f}mm  |  A: r={a_r} sig={fs(sig_a).strip()}  |  "
                     f"B: r={b_r} sig={fs(sig_b).strip()}  |  "
                     f"C: r={c_r_used} sig={fs(sig_c).strip()}")
            combined = put_title(combined, title, font_scale=0.4)
            cv2.imwrite(str(OUTPUT_DIR / f'concern1_visual_z{z_str}.png'), combined)

    # -- Write CSV --
    csv_path = OUTPUT_DIR / 'concern1_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # -- Summary --
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Geometry comparison
    print(f"\nCrop dimensions: {results[0]['crop_h']}x{results[0]['crop_w']}")
    print(f"Consensus radius: {con_radius}")
    a_r_val = results[0]['a_r']
    print(f"Method A radius (reconstructed): {a_r_val}")
    print(f"Method B radius (consensus):     {con_radius}")
    print(f"Radius difference A vs B:        {con_radius - a_r_val}")
    print()

    for method, prefix in [('A (reconstruct)', 'a'), ('B (consensus R)', 'b'), ('C (auto-detect)', 'c')]:
        ok_count = sum(1 for r in results if r[f'{prefix}_ok'])
        fail_count = len(results) - ok_count
        sigmas = [r[f'{prefix}_sigma'] for r in results if not np.isnan(r[f'{prefix}_sigma'])]
        print(f"Method {method}:")
        print(f"  Detection: {ok_count} OK, {fail_count} FAIL")
        if sigmas:
            print(f"  ERF sigma: min={min(sigmas):.3f}, max={max(sigmas):.3f}, mean={np.mean(sigmas):.3f}")
        print()

    # C-only failures
    c_only_fails = [r for r in results if not r['c_ok'] and r['a_ok'] and r['b_ok']]
    if c_only_fails:
        print(f"WARNING: Method C failed on {len(c_only_fails)} frames where A/B succeeded:")
        for r in c_only_fails:
            print(f"  Frame {r['frame_idx']} z={r['z_mm']:.1f}mm ({r['filename']})")
    else:
        print("Method C: no unique failures vs A/B")

    # Sigma discrepancies
    threshold = 0.1
    disc = [(r['frame_idx'], r['z_mm'], pair, r[key])
            for r in results
            for pair, key in [('A-B', 'ab_diff'), ('A-C', 'ac_diff'), ('B-C', 'bc_diff')]
            if not np.isnan(r[key]) and r[key] > threshold]

    if disc:
        print(f"\nDiscrepancies > {threshold} px:")
        for fi, z, pair, diff in disc:
            print(f"  Frame {fi} z={z:.1f}mm: {pair} diff = {diff:.4f} px")
    else:
        print(f"\nNo discrepancies > {threshold} px between any methods.")

    # Recommendation
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    a_fails = sum(1 for r in results if not r['a_ok'])
    b_fails = sum(1 for r in results if not r['b_ok'])
    c_fails = sum(1 for r in results if not r['c_ok'])

    if a_fails == b_fails == c_fails == 0 and not disc:
        print("All three methods produce equivalent results with zero failures.")
        print("=> Use Method B (simplest, uses known consensus geometry directly).")
    elif c_fails > a_fails or c_fails > b_fails:
        print(f"Method C has more failures ({c_fails}) than A ({a_fails}) / B ({b_fails}).")
        print("=> Auto-detect unreliable on blurry frames. Use Method B.")
    elif disc:
        print("Methods produce different ERF sigma values.")
        print("=> Review visual panels and CSV. Default: Method B.")
    else:
        print("=> Use Method B (simplest + uses known geometry).")

    # Save summary
    summary_path = OUTPUT_DIR / 'concern1_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Concern 1 Test Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Cine folder: {CINE_FOLDER}\n")
        f.write(f"Focus frame: {filenames[focus_idx]} (stage={focus_stage:.1f}mm, lap_var={focus_var:.1f})\n")
        f.write(f"Consensus sphere: cx={con_cx}, cy={con_cy}, radius={con_radius}\n")
        f.write(f"Crop dimensions: {results[0]['crop_h']}x{results[0]['crop_w']}\n")
        f.write(f"Total frames: {len(results)}\n\n")
        f.write(f"Method A radius (reconstructed): {a_r_val}\n")
        f.write(f"Method B radius (consensus):     {con_radius}\n")
        f.write(f"Difference: {con_radius - a_r_val}\n\n")
        for method, prefix in [('A', 'a'), ('B', 'b'), ('C', 'c')]:
            ok = sum(1 for r in results if r[f'{prefix}_ok'])
            fail = len(results) - ok
            sigmas = [r[f'{prefix}_sigma'] for r in results if not np.isnan(r[f'{prefix}_sigma'])]
            f.write(f"Method {method}: {ok} OK, {fail} FAIL")
            if sigmas:
                f.write(f", sigma [{min(sigmas):.3f}, {max(sigmas):.3f}], mean={np.mean(sigmas):.3f}")
            f.write("\n")
        f.write(f"\nDiscrepancies > {threshold} px: {len(disc)}\n")
        for fi, z, pair, diff in disc:
            f.write(f"  Frame {fi} z={z:.1f}mm: {pair} diff = {diff:.4f} px\n")

    print(f"\nSummary saved: {summary_path}")
    print(f"Visual panels: {OUTPUT_DIR}/concern1_visual_*.png")
    print("Done.")


if __name__ == '__main__':
    main()
