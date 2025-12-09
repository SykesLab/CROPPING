# pipeline_global_every10_parallel.py
from pathlib import Path
import csv
import cv2

from config_modular import OUTPUT_ROOT, CINE_ROOT, CINE_STEP, CROP_SAFETY_PIXELS
from cine_io_modular import group_cines_by_droplet, safe_load_cine, iter_subfolders
from darkness_analysis_modular import analyze_cine_darkness, choose_best_frame_with_geo
from crop_calibration_modular import maybe_add_calibration_sample, compute_crop_size
from cropping_modular import crop_droplet_with_sphere_guard
from plotting_modular import save_darkness_plot, save_geometric_overlay
from parallel_utils_modular import run_parallel


# -----------------------------
# WORKER A: Analyze a single folder
# -----------------------------
def _analyze_folder_for_calibration(sub):
    """
    Analyze droplets in one folder and return:
      - folder_analyses dict for this folder
      - lists of diameters & gaps for global crop calibration

    Returns:
        (sub_name, folder_analyses, diams, gaps)
        where folder_analyses[(droplet_id, cam)] = {
            "path", "first", "last", "curve", "best", "geo"
        }
    """
    sub = Path(sub)
    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    if n_groups == 0:
        return (sub.name, {}, [], [])

    # Only use droplets that will actually be processed (every CINE_STEP)
    selected_indices = list(range(0, n_groups, CINE_STEP))

    folder_analyses = {}
    diams = []
    gaps = []

    for g_index in selected_indices:
        droplet_id, cams = groups[g_index]

        for cam in ("g", "v"):
            path = cams.get(cam)
            if path is None:
                continue

            c = safe_load_cine(path)
            if c is None:
                continue

            # Darkness curve for full cine
            dark = analyze_cine_darkness(c)
            curve = dark["darkness_curve"]
            first = dark["first_frame"]
            last = dark["last_frame"]

            # Best frame + geometry (pre-collision, centred, darkness tie-break)
            best_idx, geo = choose_best_frame_with_geo(c, curve)

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": curve,
                "best": best_idx,
                "geo": geo,
            }

            # Use best-frame geometry as a calibration sample (if valid)
            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps)


# -----------------------------
# WORKER B: Produce outputs for one folder
# -----------------------------
def _process_folder_outputs(args):
    """
    Worker that writes crops, overlays, and CSV for a single folder.

    args = (subfolder_path, analyses_dict, CNN_SIZE)
      - subfolder_path: str or Path to the cine folder
      - analyses_dict[(droplet_id, cam)] = analysis dict from worker A
      - CNN_SIZE: global crop size (px)
    """
    sub_path, analyses_dict, CNN_SIZE = args
    sub = Path(sub_path)

    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    selected_indices = list(range(0, n_groups, CINE_STEP))

    out_sub = OUTPUT_ROOT / sub.name
    out_sub.mkdir(parents=True, exist_ok=True)

    csv_path = out_sub / f"{sub.name}_summary_global.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "droplet_id", "camera", "cine_file",
            "first_frame", "last_frame",
            "best_frame", "dark_fraction",
            "y_top", "y_bottom", "y_sphere",
            "crop_size_px", "crop_path",
        ])

        for g_index in selected_indices:
            droplet_id, cams = groups[g_index]

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    continue

                info = analyses_dict.get((droplet_id, cam))
                if info is None:
                    # Could happen if cine failed to load or had no valid frames
                    continue

                curve = info["curve"]
                first = info["first"]
                last = info["last"]
                best_idx = info["best"]
                geo = info["geo"]

                dark_val = float(curve[best_idx - first])

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]
                cx = geo["cx"]

                # ---- Save crop (if geometry valid) ----
                crop_path = ""
                if y_top is not None and y_bottom is not None:
                    crop = crop_droplet_with_sphere_guard(
                        geo["frame"],
                        y_top,
                        y_bottom,
                        cx,
                        target_w=CNN_SIZE,
                        target_h=CNN_SIZE,
                        y_sphere=y_sphere,
                        safety=CROP_SAFETY_PIXELS,
                    )

                    out_crop = out_sub / f"{path.stem}_crop.png"
                    cv2.imwrite(str(out_crop), crop)
                    crop_path = str(out_crop)

                # ---- Save plots ----
                save_darkness_plot(
                    out_sub / f"{path.stem}_darkness.png",
                    curve, first, last, best_idx, path.name
                )
                save_geometric_overlay(
                    out_sub / f"{path.stem}_overlay.png", geo, best_idx
                )

                writer.writerow([
                    droplet_id, cam, path.name,
                    first, last,
                    best_idx, dark_val,
                    y_top, y_bottom, y_sphere,
                    CNN_SIZE, crop_path,
                ])

    return f"[DONE] {sub.name}"


# -----------------------------
# MAIN PARALLEL GLOBAL PIPELINE
# -----------------------------
def process_global_every_10_parallel():
    print("\n[GLOBAL PARALLEL] Phase 1: analysing folders in parallel.\n")

    subfolders = iter_subfolders(CINE_ROOT)

    # Phase 1: Analyze each folder in parallel
    results = run_parallel(_analyze_folder_for_calibration, subfolders)

    # Aggregate calibration samples + per-folder analyses
    all_diams = []
    all_gaps = []
    global_analyses = {}

    for (subname, folder_analyses, diams, gaps) in results:
        global_analyses[subname] = folder_analyses
        all_diams.extend(diams)
        all_gaps.extend(gaps)

    # Phase 2: Compute a single GLOBAL crop size for all folders
    CNN_SIZE = compute_crop_size(all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS)
    print(f"\n[GLOBAL PARALLEL] → GLOBAL CROP SIZE = {CNN_SIZE} × {CNN_SIZE}\n")

    # Phase 3: Produce outputs in parallel (per folder)
    print("[GLOBAL PARALLEL] Phase 3: generating folder outputs in parallel.\n")

    # Reuse the same subfolder list to build args
    args_list = []
    for sub in subfolders:
        analyses_dict = global_analyses.get(sub.name, {})
        args_list.append((str(sub), analyses_dict, CNN_SIZE))

    output_results = run_parallel(_process_folder_outputs, args_list)

    print("\n".join(output_results))
    print("\n=== GLOBAL PARALLEL PROCESSING COMPLETE ===")
