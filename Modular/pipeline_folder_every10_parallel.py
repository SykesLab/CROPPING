# pipeline_folder_every10_parallel.py
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
# WORKER A — Analyse a single droplet
# ============================================================
def _analyze_droplet_worker(args):
    droplet_id, cams = args
    folder_results = {}
    diams, gaps = [], []

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

        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": curve,
            "best": best_idx,
            "geo": geo,
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps)


# ============================================================
# WORKER B — Output crops + plots for a droplet
# ============================================================
def _process_droplet_output_worker(args):
    droplet_id, cam_data, CNN_SIZE, out_sub_path = args
    out_sub_path = Path(out_sub_path)

    for cam, info in cam_data.items():
        path = info["path"]
        curve = info["curve"]
        first = info["first"]
        last = info["last"]
        best_idx = info["best"]
        geo = info["geo"]

        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]
        cx = geo["cx"]

        # ---- Save crop ----
        if y_top is not None and y_bottom is not None:
            crop = crop_droplet_with_sphere_guard(
                geo["frame"], y_top, y_bottom, cx,
                target_w=CNN_SIZE, target_h=CNN_SIZE,
                y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
            )
            cv2.imwrite(str(out_sub_path / f"{path.stem}_crop.png"), crop)

        # ---- Save plots ----
        save_darkness_plot(
            out_sub_path / f"{path.stem}_darkness.png",
            curve, first, last, best_idx, path.name
        )
        save_geometric_overlay(
            out_sub_path / f"{path.stem}_overlay.png",
            geo, best_idx, CNN_SIZE=CNN_SIZE
        )

    return f"[DONE] droplet {droplet_id}"


# ============================================================
# MAIN PARALLEL PER-FOLDER PIPELINE
# ============================================================
def process_per_folder_every_10_parallel():
    subfolders = iter_subfolders(CINE_ROOT)
    total_folders = len(subfolders)

    # Global droplet counting
    total_droplets = sum(
        len(group_cines_by_droplet(sub))
        for sub in subfolders
    )
    global_done = 0
    global_timer = Timer()

    print(f"\n[PER-FOLDER PARALLEL] Found {total_folders} subfolders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print(f"\n==============================")
        print(f"[FOLDER {f_idx}/{total_folders}] {sub.name}")
        print(f"==============================")

        groups = group_cines_by_droplet(sub)
        n_groups = len(groups)
        if n_groups == 0:
            print("  [INFO] No droplets in this folder.")
            continue

        selected_indices = list(range(0, n_groups, CINE_STEP))
        droplets_to_process = [
            (groups[idx][0], groups[idx][1])
            for idx in selected_indices
        ]

        # -------------------------
        # Phase 1 — Analysis
        # -------------------------
        print(f"[{sub.name}] Phase 1: Analysing droplets...")
        timer = Timer()

        results = run_parallel(
            _analyze_droplet_worker,
            droplets_to_process,
            desc=f"{sub.name}: analysing droplets"
        )
        print(f"[{sub.name}] Phase 1 done. Elapsed: {timer.elapsed}")

        # Collect calibration samples
        folder_analyses = {}
        all_diams, all_gaps = [], []

        for (droplet_id, cam_dict, diams, gaps) in results:
            folder_analyses[droplet_id] = cam_dict
            all_diams.extend(diams)
            all_gaps.extend(gaps)

        # -------------------------
        # Phase 2 — Crop size calibration
        # -------------------------
        if not all_gaps:
            CNN_SIZE = 128
            print(f"[CAL:{sub.name}] No valid geometry → fallback 128x128")
        else:
            CNN_SIZE = compute_crop_size(all_diams, all_gaps, CROP_SAFETY_PIXELS)
            print(f"[CAL RESULTS:{sub.name}] Folder crop size = {CNN_SIZE} × {CNN_SIZE}")

        # -------------------------
        # Phase 3 — Outputs (parallel)
        # -------------------------
        print(f"[{sub.name}] Phase 3: Generating droplet outputs...")

        out_sub = OUTPUT_ROOT / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        output_args = [
            (droplet_id, folder_analyses[droplet_id], CNN_SIZE, str(out_sub))
            for droplet_id in folder_analyses
        ]

        results_out = run_parallel(
            _process_droplet_output_worker,
            output_args,
            desc=f"{sub.name}: generating outputs"
        )
        print(f"[{sub.name}] Phase 3 done. Elapsed: {timer.elapsed}")

        # -------------------------
        # Global droplet progress update
        # -------------------------
        global_done += len(output_args)
        pct = global_done / total_droplets * 100
        print(f"[GLOBAL PROGRESS] {global_done}/{total_droplets} droplets "
              f"({pct:.1f}%) — elapsed {global_timer.elapsed}")

    print("\n=== PER-FOLDER PROCESSING COMPLETE ===")
