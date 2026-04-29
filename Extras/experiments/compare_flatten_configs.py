"""
compare_flatten_configs.py
==========================

Visual + quantitative comparison of the four active flatten configurations
on REAL data from the project. Answers the question:

    "Does it actually matter which flatten recipe my inference uses?"

Configurations compared (per the audit in physics.py / Phase 11 work):

  CONFIG 1 — calibration mode      flatten_sphere_crop(inner_margin=20,
                                       flatten_exterior=False)
                                    (used by Calibration GUI default →
                                     produces tertiary training samples)

  CONFIG 2 — inference (wide feather) flatten_sphere_crop(inner_margin=1,
                                       flatten_exterior=True, feather=40)
                                    (alternative mode; preserves blur via
                                     wide feather)

  CONFIG 3 — simple mode           flatten_sphere_crop(inner_margin=0,
                                       flatten_exterior=True)  [defaults]
                                    (used by Preprocessing pipeline →
                                     produces primary training samples)

  CONFIG 4 — boundary_normalise    Otsu + cosine feather + caustic fill
                                    (used by Inference engine for real
                                     droplet deployment, Path C)

Inputs scanned automatically:
    - A few defocused calibration sphere PNGs (where simple mode would
      destructively chop blur)
    - A few sharp preprocessing crops (where simple mode is in-spec)

Outputs (under ``Extras/flatten_compare_output/``):
    - For each input: one PNG showing all 4 configs side-by-side + the
      original, plus radial-edge intensity profiles overlaid.
    - ``flatten_compare_summary.csv`` — per-input + per-config:
        ERF-measured sigma (when fittable), pixel-mean of interior,
        pixel-mean of exterior, max difference vs config 1.

Run from the CROPPING root:

    python -m Extras.compare_flatten_configs
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Path setup — CROPPING root + Calibration + Inference modules
# ──────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _module in ("Calibration", "Inference", "Preprocessing", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402

from sphere_processing import flatten_sphere_crop  # noqa: E402
from inference_engine import boundary_normalise  # noqa: E402
from blur_measurement import measure_blur_erf  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "flatten_compare_output"


# ──────────────────────────────────────────────────────────────────────────
# Config wrappers — single-arg API for each of the 4 recipes
# ──────────────────────────────────────────────────────────────────────────

def cfg1_calibration_mode(img_uint8: np.ndarray) -> Optional[np.ndarray]:
    """Calibration mode: inner_margin=20, exterior preserved."""
    img_f = img_uint8.astype(np.float32) / 255.0
    flat, info = flatten_sphere_crop(img_f, inner_margin=20,
                                       flatten_exterior=False)
    if info is None:
        return None
    return (flat * 255).clip(0, 255).astype(np.uint8)


def cfg2_inference_wide_feather(img_uint8: np.ndarray) -> Optional[np.ndarray]:
    """Inference mode: inner_margin=1, full flatten, feather=40."""
    img_f = img_uint8.astype(np.float32) / 255.0
    flat, info = flatten_sphere_crop(img_f, inner_margin=1,
                                       flatten_exterior=True, feather=40)
    if info is None:
        return None
    return (flat * 255).clip(0, 255).astype(np.uint8)


def cfg3_simple_mode(img_uint8: np.ndarray) -> Optional[np.ndarray]:
    """Simple mode: inner_margin=0, full flatten, default 3px feather."""
    img_f = img_uint8.astype(np.float32) / 255.0
    flat, info = flatten_sphere_crop(img_f)  # all defaults
    if info is None:
        return None
    return (flat * 255).clip(0, 255).astype(np.uint8)


def cfg4_boundary_normalise(img_uint8: np.ndarray) -> Optional[np.ndarray]:
    """Inference engine's boundary_normalise: Otsu + cosine feather."""
    out = boundary_normalise(img_uint8)
    return (out * 255).clip(0, 255).astype(np.uint8)


CONFIGS = [
    ("config1_calibration", cfg1_calibration_mode),
    ("config2_inference_wide", cfg2_inference_wide_feather),
    ("config3_simple", cfg3_simple_mode),
    ("config4_boundary_normalise", cfg4_boundary_normalise),
]


# ──────────────────────────────────────────────────────────────────────────
# Auto-discover sample inputs
# ──────────────────────────────────────────────────────────────────────────

def find_calibration_raw_cines() -> list[tuple]:
    """Locate raw .cine files for the calibration sphere stack.

    Returns a list of (cine_path, label, true_z_mm) tuples. The z value
    comes from a measurements.csv lookup (so we know what frame is what).
    """
    spheres_dir = _REPO_ROOT / "calibration spheres" / "9mm"
    runs_dir = _REPO_ROOT / "Calibration" / "runs"
    if not spheres_dir.is_dir() or not runs_dir.is_dir():
        return []
    # Find latest run with a measurements.csv to map .cine -> z
    latest = None
    for r in sorted(runs_dir.glob("*"), key=lambda p: p.stat().st_mtime,
                     reverse=True):
        if (r / "measurements.csv").is_file():
            latest = r
            break
    if latest is None:
        return []
    import pandas as pd
    m = pd.read_csv(latest / "measurements.csv")
    # filename column maps to (renamed) PNG names like "9mm_z-7.0mm.png"
    # The original .cine files are named like "9mm_1.cine".
    # Sort measurements by z to pick samples spanning the range.
    m = m.sort_values("z_mm")
    targets_mm = [-7.0, -5.0, -3.0, 0.0, 3.0]
    picks: list = []
    used_z = set()
    for tgt in targets_mm:
        # closest available z
        idx = (m["z_mm"] - tgt).abs().idxmin()
        z = float(m.at[idx, "z_mm"])
        if z in used_z:
            continue
        used_z.add(z)
        # The .cine files are 9mm_1.cine ... 9mm_61.cine in z-sort order
        # (after sphere stack load + sort by z). Their order in
        # measurements.csv matches the loaded order. We need the
        # corresponding cine file. Use the row's order in the original
        # (unsorted) DataFrame.
        # Find row index in m_orig (unsorted) → cine number = index + 1
        cine_num = list(pd.read_csv(latest / "measurements.csv")
                          ["filename"]).index(m.at[idx, "filename"]) + 1
        cine_path = spheres_dir / f"9mm_{cine_num}.cine"
        if cine_path.is_file():
            picks.append((cine_path, f"z{z:+.1f}mm", z))
    return picks


def find_droplet_raw_cines() -> list[tuple]:
    """Locate a few raw droplet .cine files (any size). Returns
    (cine_path, label, None) — z unknown for droplets."""
    base = _REPO_ROOT / "Preprocessing" / "OUTPUT" / "Focus"
    # OUTPUT/Focus is processed; we want raw .cines.
    # Look for raw cines — they typically live above OUTPUT.
    # Scan a couple of material folders directly for any .cine.
    candidates = []
    for parent in [
        _REPO_ROOT / "Preprocessing",
        _REPO_ROOT,
    ]:
        for c in parent.rglob("*.cine"):
            # Skip anything under "calibration spheres"
            if "calibration spheres" in c.parts or "OUTPUT" in c.parts:
                continue
            candidates.append(c)
            if len(candidates) >= 3:
                break
        if candidates:
            break
    return [(p, p.stem, None) for p in candidates[:3]]


def load_raw_calibration_crop(cine_path: Path) -> Optional[np.ndarray]:
    """Extract first frame of cine, detect sphere, spatially crop to
    a sphere-centred square WITHOUT flattening."""
    try:
        from cine_loader import CineLoader
        from sphere_processing import find_sphere_center, crop_to_square
    except Exception as e:
        print(f"  could not import cine_loader: {e}")
        return None
    try:
        loader = CineLoader(str(cine_path))  # constructor auto-loads
        if loader.cine_obj is None:
            print(f"    failed to load .cine: {cine_path.name}")
            return None
        # Use the first frame in the cine — for calibration single-frame
        # cines this IS the z-position frame
        frame = loader.extract_frame(loader.frame_range[0])
        if frame is None:
            return None
        # extract_frame returns float32 grayscale (already 2D)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # Detect sphere on the raw frame
        det = find_sphere_center(frame, upper_only=True)
        if det is None:
            print(f"    sphere detect failed on {cine_path.name}")
            return None
        cx, cy, radius = det
        crop = crop_to_square(frame, cx, cy, radius, padding=1.2)
        # Normalise to uint8 (extract_frame is float32)
        if crop.dtype != np.uint8:
            crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return crop
    except Exception as e:
        print(f"    error loading {cine_path.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ──────────────────────────────────────────────────────────────────────────

def measure_erf_sigma(img_uint8: np.ndarray) -> Optional[float]:
    """Return ERF-fitted sigma in pixels, or None on failure."""
    if img_uint8.ndim == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = measure_blur_erf(img_uint8.astype(np.float32) / 255.0,
                                       num_rays=36, verbose=False)
        if result.confidence < 0.3:
            return None
        return float(result.sigma)
    except Exception:
        return None


def radial_intensity_profile(img: np.ndarray, n_bins: int = 60) -> tuple:
    """Compute mean intensity per concentric ring around image centre."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.indices((h, w))
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_max = min(cx, cy) * 0.95
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.any():
            means[i] = float(img[mask].mean())
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centres, means


# ──────────────────────────────────────────────────────────────────────────
# Comparison plot per input
# ──────────────────────────────────────────────────────────────────────────

def make_comparison_plot(input_path: Path, source_label: str,
                           rows_csv: list,
                           pre_loaded_image: Optional[np.ndarray] = None) -> None:
    """Apply all 4 configs to one input image, plot side-by-side with diff
    maps + radial-profile overlay + annotation guide, and append a row
    per config to ``rows_csv``.

    If ``pre_loaded_image`` is provided it's used as the input directly
    (raw frame from a .cine, etc.). Otherwise the image is loaded from
    ``input_path``. The latter path is for already-processed PNGs and
    is now disfavoured (the comparison's "original" must be unflattened).
    """
    if pre_loaded_image is not None:
        img = pre_loaded_image
    else:
        img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  skip {input_path.name}: could not read")
        return

    # Resize to a manageable size if huge (raw frames can be large).
    if max(img.shape) > 600:
        target = 512
        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)

    # Apply each config
    results = {"original": img}
    for name, fn in CONFIGS:
        out = fn(img)
        if out is None:
            print(f"  {input_path.name}: {name} -> detection failed (skipped)")
            results[name] = None
        else:
            results[name] = out

    sigma_vals = {n: measure_erf_sigma(r) if r is not None else None
                  for n, r in results.items()}

    # Reference: cfg1 (calibration mode) is baseline for diff maps
    ref_img = results.get("config1_calibration")
    if ref_img is None:
        # Fall back to original if cfg1 failed
        ref_img = results["original"]

    # ── Layout: 4 rows × 5 cols ──────────────────────────────────────
    # Row 0: 5 images (original + 4 configs)
    # Row 1: 4 diff heatmaps (each config minus cfg1) — col 0 is annotation text
    # Row 2: single big radial-profile plot (all 4 configs overlaid) — spans 5 cols
    # Row 3: single big "what to look for" guide — spans 5 cols
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, height_ratios=[3, 2, 3, 1.5],
                            hspace=0.35, wspace=0.15)

    panels = list(results.items())  # [(original, ...), (cfg1, ...), ...]

    # Row 0 — images
    for col, (name, r_img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        if r_img is None:
            ax.text(0.5, 0.5, "(failed)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            continue
        ax.imshow(r_img, cmap='gray', vmin=0, vmax=255)
        sigma = sigma_vals[name]
        sigma_str = f"\nERF σ = {sigma:.3f} px" if sigma is not None else "\n(σ fit failed)"
        title_color = 'black'
        # Highlight the baseline
        if name == 'config1_calibration':
            title_color = '#3060a0'
        ax.set_title(f"{name}{sigma_str}", fontsize=11, color=title_color,
                     fontweight='bold')
        ax.axis('off')

    # Row 1 — diff heatmaps (each config minus cfg1, only the 4 configs)
    # Col 0 = textual key
    ax_key = fig.add_subplot(gs[1, 0])
    sigma_orig = sigma_vals.get("original")
    sigma_cfg1 = sigma_vals.get("config1_calibration")
    key_text = (
        "DIFF MAPS\n(each config\n  minus cfg1)\n\n"
        "Red = config\nincreased\nbrightness\nrelative to cfg1\n\n"
        "Blue = config\ndecreased\nbrightness\nrelative to cfg1"
    )
    ax_key.text(0.5, 0.5, key_text, ha='center', va='center',
                transform=ax_key.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                           alpha=0.8))
    ax_key.axis('off')

    diff_panels = panels[1:]  # skip "original"
    for col, (name, r_img) in enumerate(diff_panels, start=1):
        ax = fig.add_subplot(gs[1, col])
        if r_img is None or ref_img is None:
            ax.axis('off')
            continue
        diff = r_img.astype(np.int16) - ref_img.astype(np.int16)
        vlim = max(20, int(np.percentile(np.abs(diff), 99)))
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        ax.set_title(f"{name}\n− cfg1 (baseline)", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

    # Row 2 — single radial-profile plot, all 4 configs + original overlaid
    ax_prof = fig.add_subplot(gs[2, :])
    r_orig, p_orig = radial_intensity_profile(results["original"])
    ax_prof.plot(r_orig, p_orig, '-', color='black', alpha=0.6, lw=2,
                  label="ORIGINAL (input)")
    colours = {'config1_calibration': '#3060a0',
               'config2_inference_wide': '#30a070',
               'config3_simple': '#c04040',
               'config4_boundary_normalise': '#a060c0'}
    for name, r_img in diff_panels:
        if r_img is None:
            continue
        r, p = radial_intensity_profile(r_img)
        sigma = sigma_vals[name]
        sigma_str = f"  σ={sigma:.3f}" if sigma is not None else ""
        ax_prof.plot(r, p, '-', color=colours.get(name, 'gray'), lw=1.6,
                      label=f"{name}{sigma_str}")
    ax_prof.set_xlabel("Radius from image centre (px)")
    ax_prof.set_ylabel("Mean intensity (0–255)")
    ax_prof.set_title("Radial intensity profile — divergence at the sphere edge "
                       "is what matters", fontsize=11, fontweight='bold')
    ax_prof.legend(loc='best', fontsize=9)
    ax_prof.grid(alpha=0.3)

    # Row 3 — annotation: what to look for
    ax_guide = fig.add_subplot(gs[3, :])
    ax_guide.axis('off')

    # Build a context-aware guide based on whether this is a defocused or sharp input
    is_defocused = ('z-' in input_path.name or 'z+' in input_path.name) and \
                   not ('z+0.0mm' in input_path.name or 'z-0.0mm' in input_path.name)
    if source_label == 'calibration_sphere':
        if is_defocused:
            sigma_orig_str = (f"{sigma_orig:.3f}"
                              if sigma_orig is not None else "n/a")
            guide = (
                "WHAT TO CHECK (defocused calibration sphere — the worst case for flatten differences):\n"
                f"  • cfg1 (BASELINE): this is what your tertiary training samples look like.  Original σ ≈ {sigma_orig_str} px (model trained on this)\n"
                "  • cfg3 (simple): if you see a BRIGHT inner ring in the diff (red blob inside the sphere edge) and a DROP in σ → simple-mode chops blur. Confirmed destructive.\n"
                "  • cfg4 (boundary_normalise): if you see a BLUE halo OUTSIDE the sphere edge (decreased brightness) and σ INCREASES → over-feathering adds apparent blur. THIS IS WHAT INFERENCE CURRENTLY USES.\n"
                "  • cfg2 (wide feather): similar effect to cfg4 but milder.\n"
                "  → If cfg4's σ is much higher than cfg1's, your real-deployment inference will OVER-PREDICT defocus on droplets at this z."
            )
        else:
            guide = (
                "WHAT TO CHECK (focal-plane calibration sphere — minimal blur, all configs should agree closely):\n"
                "  • All four configs should give σ values close to each other.\n"
                "  • Diff maps should be mostly white (small differences).\n"
                "  • If they DON'T agree near focus, the configs are doing something fundamentally different even on sharp data."
            )
    else:  # droplet_sharp
        guide = (
            "WHAT TO CHECK (sharp droplet crop from Preprocessing — primary training input):\n"
            "  • cfg1 BASELINE for comparison.\n"
            "  • cfg3 (simple): this is what your PRIMARY training samples were flattened with. σ should be slightly LOWER than cfg1.\n"
            "  • cfg4 (boundary_normalise): how INFERENCE currently processes droplet crops. Should agree with cfg1 closely on sharp inputs.\n"
            "  • If cfg4 disagrees with cfg3 on a sharp droplet → there's a sim-to-real gap in inference even at low defocus.\n"
            "  → For sharp droplets, expect ~0.18 px σ drop with cfg3, near-zero diff for cfg2/cfg4 (vs cfg1)."
        )
    ax_guide.text(0.01, 0.95, guide, ha='left', va='top',
                   transform=ax_guide.transAxes, fontsize=10,
                   family='monospace',
                   bbox=dict(boxstyle="round", facecolor='lightyellow',
                             edgecolor='gray', alpha=0.9))

    fig.suptitle(f"FLATTEN COMPARISON — {source_label}: {input_path.name}",
                 fontsize=14, fontweight='bold', y=0.995)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / f"{source_label}_{input_path.stem}.png"
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_png.name}")

    # CSV rows
    sigma_orig = measure_erf_sigma(results["original"])
    for name, r_img in results.items():
        if name == "original":
            continue
        if r_img is None:
            rows_csv.append({
                'source': source_label,
                'input': input_path.name,
                'config': name,
                'erf_sigma_px': '',
                'erf_sigma_orig_px': sigma_orig if sigma_orig is not None else '',
                'sigma_diff_vs_orig_px': '',
                'pixel_max_diff_vs_cfg1': '',
                'detection': 'failed',
            })
            continue
        sigma_cfg = measure_erf_sigma(r_img)
        diff_vs_orig = (sigma_cfg - sigma_orig
                        if (sigma_cfg is not None and sigma_orig is not None)
                        else None)
        # Pixel max diff vs config 1 (calibration mode = baseline)
        cfg1_img = results.get("config1_calibration")
        if cfg1_img is not None:
            d = np.abs(r_img.astype(np.int32) - cfg1_img.astype(np.int32))
            max_diff = int(d.max())
        else:
            max_diff = ''
        rows_csv.append({
            'source': source_label,
            'input': input_path.name,
            'config': name,
            'erf_sigma_px': f"{sigma_cfg:.4f}" if sigma_cfg is not None else '',
            'erf_sigma_orig_px': f"{sigma_orig:.4f}" if sigma_orig is not None else '',
            'sigma_diff_vs_orig_px': (f"{diff_vs_orig:+.4f}"
                                       if diff_vs_orig is not None else ''),
            'pixel_max_diff_vs_cfg1': max_diff,
            'detection': 'ok',
        })


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FLATTEN CONFIGURATION COMPARISON")
    print("=" * 70)
    print("Output dir:", OUTPUT_DIR)

    rows_csv: list = []

    print("\n[1] Defocused calibration sphere — RAW frames from .cine")
    print("    (loading raw frames so 'original' is genuinely unflattened)")
    cal_picks = find_calibration_raw_cines()
    if not cal_picks:
        print("    No calibration .cine files found — skip")
    else:
        for cine_path, label, true_z in cal_picks:
            print(f"  processing: {cine_path.name} ({label})")
            raw = load_raw_calibration_crop(cine_path)
            if raw is None:
                continue
            # Build a synthetic input_path that carries the z label so
            # the annotation block picks the defocused/focused branch.
            fake_path = cine_path.with_name(f"{cine_path.stem}_{label}.png")
            make_comparison_plot(fake_path, "calibration_sphere",
                                  rows_csv, pre_loaded_image=raw)

    print("\n[2] Raw droplet .cine frames (if any present in repo)")
    print("    (note: if no raw droplet cines available, this section is empty)")
    drop_picks = find_droplet_raw_cines()
    if not drop_picks:
        print("    No raw droplet .cines found — skip")
    else:
        for cine_path, label, _ in drop_picks:
            print(f"  processing: {cine_path.name}")
            raw = load_raw_calibration_crop(cine_path)  # re-uses sphere detect+crop
            if raw is None:
                continue
            fake_path = cine_path.with_name(f"{cine_path.stem}.png")
            make_comparison_plot(fake_path, "droplet_raw",
                                  rows_csv, pre_loaded_image=raw)

    # Write CSV
    if rows_csv:
        csv_path = OUTPUT_DIR / "flatten_compare_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_csv)
        print(f"\nWrote summary: {csv_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Inspect the PNGs in", OUTPUT_DIR)
    print("Each PNG shows the same input through all 4 configs,")
    print("with ERF sigma measured on each + radial profiles for shape diff.")
    print()
    print("Read the summary CSV: lower 'sigma_diff_vs_orig_px' = closer to")
    print("the input's blur (good); larger = the config altered the blur.")
    print("Note: simple mode (cfg3) on defocused inputs should show large")
    print("sigma drop (it chops interior blur).")


if __name__ == "__main__":
    main()
