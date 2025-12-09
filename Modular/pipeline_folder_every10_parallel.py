# pipeline_folder_every10_parallel.py
from pathlib import Path
import csv
import cv2

from config_modular import (
    OUTPUT_ROOT, CINE_ROOT, CINE_STEP, CROP_SAFETY_PIXELS
)
from cine_io_modular import group_cines_by_droplet, safe_load_cine, iter_subfolders
from darkness_analysis_modular import analyze_cine_darkness, choose_best_frame_with_geo
from crop_calibration_modular import maybe_add_calibration_sample, compute_crop_size
from cropping_modular import crop_droplet_with_sphere_guard
from plotting_modular import save_darkness_plot, save_geometric_overlay
from parallel_utils_modular import run_parallel


# ============================================================
# WORKER A — Analyse *one droplet* within a folder
# ============================================================
def _analyze_droplet_worker(args):
    """
    args = (droplet_id, cam_paths_dict)
    cam_paths_dict = {"g": path_or_None, "v": path_or_None}

    Returns:
        (droplet_id, {cam: analysis_dict}, diams, gaps)
    """
    droplet_id, cams = args
    folder_results = {}
    diams = []
    gaps = []

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
# WORKER B — Generate crops + plots for one droplet
# ============================================================
def _process_droplet_output_worker(args):
    """
    args = (droplet_id, cam_results_dict, CNN_SIZE, out_sub_path)

    Writes crops + plots for both cameras.
    Returns status string.
    """

    droplet_id, cam_data, CNN_SIZE, out_sub_path = args

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

        dark_val = float(curve[best_idx - first])

        # ---- Save crop ----
        crop_path = ""
        if y_top is not None and y_bottom is not None:
            crop = crop_droplet_with_sphere_guard(
                geo["frame"], y_top, y_bottom, cx,
                target_w=CNN_SIZE, target_h=CNN_SIZE,
                y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
            )
            out_crop = Path(out_sub_path) / f"{path.stem}_crop.png"
            cv2.imwrite(str(out_crop), crop)
            crop_path = str(out_crop)

        # ---- Save plots ----
        save_darkness_plot(
            Path(out_sub_path) / f"{path.stem}_darkness.png",
            curve, first, last, best_idx, path.name
        )
        save_geometric_overlay(
            Path(out_sub_path) / f"{path.stem}_overlay.png",
            geo, best_idx
        )

    return f"[DONE] droplet {droplet_id}"


# ============================================================
# MAIN: PARALLEL PER-FOLDER PIPELINE
# ============================================================
def process_per_folder_every_10_parallel():
    subfolders = iter_subfolders(CINE_ROOT)
    n_folders = len(subfolders)

    print(f"\n[PER-FOLDER PARALLEL] Found {n_folders} subfolders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print("\n==============================")
        print(f"[FOLDER {f_idx}/{n_folders}] {sub.name}")
        print("==============================")

        groups = group_cines_by_droplet(sub)
        n_groups = len(groups)
        if n_groups == 0:
            print("  [INFO] No droplets in this folder.")
            continue

        # Select droplets (every Nth)
        selected_indices = list(range(0, n_groups, CINE_STEP))
        droplets_to_process = [
            (groups[idx][0], groups[idx][1])   # (droplet_id, {"g": path, "v": path})
            for idx in selected_indices
        ]

        # ======================================================
        # PHASE 1: Parallel droplet analysis for this folder
        # ======================================================
        print(f"[{sub.name}] Phase 1: Analysing {len(droplets_to_process)} droplets in parallel...")

        results = run_parallel(_analyze_droplet_worker, droplets_to_process)

        # Accumulate calibration samples + analyses into dictionary
        folder_analyses = {}       # key: droplet_id -> cam dict
        all_diams = []
        all_gaps = []

        for (droplet_id, cam_dict, diams, gaps) in results:
            folder_analyses[droplet_id] = cam_dict
            all_diams.extend(diams)
            all_gaps.extend(gaps)

        # ======================================================
        # PHASE 2: Compute crop size for THIS folder
        # ======================================================
        if not all_gaps:
            CNN_SIZE = 128
            print(f"[CAL:{sub.name}] No valid geometry; fallback {CNN_SIZE}×{CNN_SIZE}.")
        else:
            CNN_SIZE = compute_crop_size(all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS)
            print(f"[CAL RESULTS:{sub.name}]  → Folder crop size = {CNN_SIZE} × {CNN_SIZE}")

        # ======================================================
        # PHASE 3: Produce outputs (parallel per droplet)
        # ======================================================
        print(f"[{sub.name}] Phase 3: Generating outputs in parallel...")

        out_sub = OUTPUT_ROOT / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        # Create args for output workers
        output_args = []
        for droplet_id in folder_analyses.keys():
            output_args.append(
                (droplet_id, folder_analyses[droplet_id], CNN_SIZE, str(out_sub))
            )

        output_results = run_parallel(_process_droplet_output_worker, output_args)
        print("\n".join(output_results))

        # ======================================================
        # Write CSV summary
        # ======================================================
        print(f"[{sub.name}] Writing CSV summary...")

        csv_path = out_sub / f"{sub.name}_summary_perfolder.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "droplet_id", "camera", "cine_file",
                "best_frame", "dark_fraction",
                "y_top", "y_bottom", "y_sphere",
                "crop_size_px", "crop_path"
            ])

            for droplet_id, cam_dict in folder_analyses.items():
                for cam, info in cam_dict.items():
                    path = info["path"]
                    curve = info["curve"]
                    best_idx = info["best"]
                    first = info["first"]

                    geo = info["geo"]
                    y_top = geo["y_top"]
                    y_bottom = geo["y_bottom"]
                    y_sphere = geo["y_bottom_sphere"]

                    dark_val = float(curve[best_idx - first])

                    crop_path = str(out_sub / f"{path.stem}_crop.png")

                    writer.writerow([
                        droplet_id, cam, path.name,
                        best_idx, dark_val,
                        y_top, y_bottom, y_sphere,
                        CNN_SIZE, crop_path
                    ])

        print(f"[FOLDER COMPLETE] CSV saved: {csv_path}")
