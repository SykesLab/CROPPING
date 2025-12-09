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
from timing_utils_modular import Timer


# ============================================================
# WORKER A — Analyse folder
# ============================================================
def _analyze_folder_for_calibration(sub):
    sub = Path(sub)
    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    if n_groups == 0:
        return (sub.name, {}, [], [])

    selected = list(range(0, n_groups, CINE_STEP))
    folder_analyses = {}
    diams, gaps = [], []

    for g_index in selected:
        droplet_id, cams = groups[g_index]

        for cam in ("g", "v"):
            path = cams.get(cam)
            if path is None:
                continue

            c = safe_load_cine(path)
            if c is None:
                continue

            dark = analyze_cine_darkness(c)
            curve = dark["darkness_curve"]
            first = dark["first_frame"]
            last = dark["last_frame"]

            best_idx, geo = choose_best_frame_with_geo(c, curve)

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": curve,
                "best": best_idx,
                "geo": geo,
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps)


# ============================================================
# WORKER B — Output folder results
# ============================================================
def _process_folder_outputs(args):
    sub_path, analyses_dict, CNN_SIZE = args
    sub = Path(sub_path)

    groups = group_cines_by_droplet(sub)
    selected = list(range(0, len(groups), CINE_STEP))

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
            "crop_size_px", "crop_path"
        ])

        for idx in selected:
            droplet_id, cams = groups[idx]

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    continue

                info = analyses_dict.get((droplet_id, cam))
                if info is None:
                    continue

                curve = info["curve"]
                first = info["first"]
                last = info["last"]
                best_idx = info["best"]
                geo = info["geo"]

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]
                cx = geo["cx"]

                dark_val = float(curve[best_idx - first])

                # ---- Save crop ----
                crop_path = ""
                if y_top is not None and y_bottom is not None:
                    crop = crop_droplet_with_sphere_guard(
                        geo["frame"], y_top, y_bottom, cx,
                        target_w=CNN_SIZE, target_h=CNN_SIZE,
                        y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
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
                    out_sub / f"{path.stem}_overlay.png",
                    geo, best_idx, CNN_SIZE=CNN_SIZE
                )

                writer.writerow([
                    droplet_id, cam, path.name,
                    first, last, best_idx, dark_val,
                    y_top, y_bottom, y_sphere,
                    CNN_SIZE, crop_path
                ])

    return f"[DONE] {sub.name}"


# ============================================================
# MAIN: GLOBAL PIPELINE
# ============================================================
def process_global_every_10_parallel():
    print("\n[GLOBAL] Phase 1: analysing folders...\n")

    subfolders = iter_subfolders(CINE_ROOT)
    timer = Timer()

    results = run_parallel(
        _analyze_folder_for_calibration,
        subfolders,
        desc="Global analysis"
    )
    print(f"[GLOBAL] Phase 1 complete. Elapsed: {timer.elapsed}")

    global_analyses = {}
    all_diams, all_gaps = [], []

    for (subname, folder_analyses, diams, gaps) in results:
        global_analyses[subname] = folder_analyses
        all_diams.extend(diams)
        all_gaps.extend(gaps)

    # Compute global crop size
    CNN_SIZE = compute_crop_size(all_diams, all_gaps, CROP_SAFETY_PIXELS)
    print(f"\n[GLOBAL] Crop size = {CNN_SIZE} × {CNN_SIZE}\n")

    # Phase 3: Output
    args_list = [
        (str(sub), global_analyses.get(sub.name, {}), CNN_SIZE)
        for sub in subfolders
    ]

    print("[GLOBAL] Phase 3: generating outputs...\n")
    results_out = run_parallel(
        _process_folder_outputs,
        args_list,
        desc="Global outputs"
    )

    print("\n".join(results_out))
    print("\n=== GLOBAL PROCESSING COMPLETE ===")
