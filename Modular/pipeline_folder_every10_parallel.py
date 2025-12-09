# pipeline_folder_every10_parallel.py
#
# Per-folder pipeline with DETAILED TIMING BREAKDOWN.
# Shows time spent in: cine loading, darkness curve, best-frame selection,
# cropping, darkness plot, geometry overlay.

from pathlib import Path
import csv
import cv2
import json
import time

from config_modular import (
    OUTPUT_ROOT, CINE_ROOT, CINE_STEP, CROP_SAFETY_PIXELS
)
from cine_io_modular import group_cines_by_droplet, safe_load_cine, iter_subfolders
from darkness_analysis_modular import (
    analyze_cine_darkness, 
    choose_best_frame_with_geo,
    choose_best_frame_geometry_only
)
from crop_calibration_modular import maybe_add_calibration_sample, compute_crop_size
from cropping_modular import crop_droplet_with_sphere_guard
from plotting_modular import save_darkness_plot, save_geometric_overlay
from parallel_utils_modular import run_parallel
from timing_utils_modular import Timer


# ============================================================
# WORKER A — Analyse droplet SAFE MODE (with darkness curve)
# ============================================================
def _analyze_droplet_worker_safe(args):
    """
    SAFE MODE: Analyse a single droplet (both cameras).
    Computes full darkness curve for diagnostics.
    Returns timing breakdown for: load, darkness_curve, best_frame_selection.
    """
    droplet_id, cams = args
    folder_results = {}
    diams = []
    gaps = []
    
    # Timing accumulators (in seconds)
    timing = {
        "load_cine": 0.0,
        "darkness_curve": 0.0,
        "best_frame": 0.0,
        "n_frames": 0,
    }

    for cam in ("g", "v"):
        path = cams.get(cam)
        if path is None:
            continue

        # --- Time: load cine ---
        t0 = time.perf_counter()
        c = safe_load_cine(path)
        timing["load_cine"] += time.perf_counter() - t0
        
        if c is None:
            continue

        first, last = c.range
        timing["n_frames"] += (last - first + 1)

        # --- Time: darkness curve ---
        t0 = time.perf_counter()
        dark = analyze_cine_darkness(c)
        timing["darkness_curve"] += time.perf_counter() - t0
        
        curve = dark["darkness_curve"]

        # --- Time: best frame selection ---
        t0 = time.perf_counter()
        best_idx, geo = choose_best_frame_with_geo(c, curve)
        timing["best_frame"] += time.perf_counter() - t0

        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": curve,
            "best": best_idx,
            "geo": geo,
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps, timing)


# ============================================================
# WORKER A — Analyse droplet FAST MODE (geometry only)
# ============================================================
def _analyze_droplet_worker_fast(args):
    """
    FAST MODE: Analyse a single droplet (both cameras).
    Skips darkness curve - does single-pass geometry scan.
    """
    droplet_id, cams = args
    folder_results = {}
    diams = []
    gaps = []
    
    # Timing accumulators (in seconds)
    timing = {
        "load_cine": 0.0,
        "geometry_scan": 0.0,
        "n_frames": 0,
    }

    for cam in ("g", "v"):
        path = cams.get(cam)
        if path is None:
            continue

        # --- Time: load cine ---
        t0 = time.perf_counter()
        c = safe_load_cine(path)
        timing["load_cine"] += time.perf_counter() - t0
        
        if c is None:
            continue

        first, last = c.range
        timing["n_frames"] += (last - first + 1)

        # --- Time: geometry scan (no darkness curve!) ---
        t0 = time.perf_counter()
        best_idx, geo = choose_best_frame_geometry_only(c)
        timing["geometry_scan"] += time.perf_counter() - t0

        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": None,  # No curve in fast mode
            "best": best_idx,
            "geo": geo,
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps, timing)


# ============================================================
# WORKER B — Output droplet (with detailed timing)
# ============================================================
def _process_droplet_output_worker(args):
    """
    Generate outputs for a single droplet (both cameras).
    In FAST mode (curve=None): only crops, no plots.
    In SAFE mode: crops + darkness plot + overlay plot.
    """
    droplet_id, cam_data, CNN_SIZE, out_sub_path = args
    out_sub_path = Path(out_sub_path)
    
    # Timing accumulators
    timing = {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }

    for cam, info in cam_data.items():
        path = info["path"]
        curve = info["curve"]  # None in fast mode
        first = info["first"]
        last = info["last"]
        best_idx = info["best"]
        geo = info["geo"]

        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]
        cx = geo["cx"]

        # --- Time: cropping ---
        if y_top is not None and y_bottom is not None:
            t0 = time.perf_counter()
            crop = crop_droplet_with_sphere_guard(
                geo["frame"], y_top, y_bottom, cx,
                target_w=CNN_SIZE, target_h=CNN_SIZE,
                y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
            )
            timing["crop"] += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            cv2.imwrite(str(out_sub_path / f"{path.stem}_crop.png"), crop)
            timing["imwrite"] += time.perf_counter() - t0

        # --- SAFE MODE ONLY: generate plots ---
        if curve is not None:
            # --- Time: darkness plot ---
            t0 = time.perf_counter()
            save_darkness_plot(
                out_sub_path / f"{path.stem}_darkness.png",
                curve, first, last, best_idx, path.name
            )
            timing["darkness_plot"] += time.perf_counter() - t0

            # --- Time: overlay plot ---
            t0 = time.perf_counter()
            save_geometric_overlay(
                out_sub_path / f"{path.stem}_overlay.png",
                geo, best_idx, CNN_SIZE=CNN_SIZE
            )
            timing["overlay_plot"] += time.perf_counter() - t0

    return (f"[DONE] {droplet_id}", timing)


# ============================================================
# TIMING AGGREGATOR
# ============================================================
def _aggregate_timings(timing_list, label=""):
    """
    Aggregate timing dicts from multiple workers and print summary.
    """
    if not timing_list:
        return {}
    
    # Sum all keys
    totals = {}
    for t in timing_list:
        for k, v in t.items():
            totals[k] = totals.get(k, 0.0) + v
    
    # Print breakdown
    print(f"\n  [TIMING {label}]")
    for k, v in sorted(totals.items()):
        if k == "n_frames":
            print(f"    {k}: {int(v)} frames")
        else:
            print(f"    {k}: {v:.2f}s")
    
    return totals


# ============================================================
# QUICK TEST IMPLEMENTATION (per-folder)
# ============================================================
def _quick_test_per_folder(safe_mode: bool, profile: bool):
    """
    Quick test per-folder with detailed timing.
    """
    subfolders = iter_subfolders(CINE_ROOT)
    total_folders = len(subfolders)
    global_timer = Timer()

    folders_profile = []

    print("\n[PER-FOLDER MODE] QUICK TEST")
    print(f"Found {total_folders} subfolders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print("\n==============================")
        print(f"[FOLDER {f_idx}/{total_folders}] {sub.name}")
        print("==============================")

        t_folder_start = time.time()
        
        # Per-folder timing
        folder_timing = {
            "load_cine": 0.0,
            "darkness_curve": 0.0,
            "best_frame": 0.0,
            "darkness_plot": 0.0,
            "overlay_plot": 0.0,
            "n_frames": 0,
        }

        groups = group_cines_by_droplet(sub)
        if not groups:
            print("  [INFO] No droplets in this folder.")
            continue

        # First droplet only
        droplet_id, cams = groups[0]

        out_sub = OUTPUT_ROOT / (sub.name + "_quicktest")
        out_sub.mkdir(parents=True, exist_ok=True)

        csv_path = out_sub / f"{sub.name}_quicktest_summary_perfolder.csv"
        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                "droplet_id",
                "camera",
                "cine_file",
                "first_frame",
                "last_frame",
                "best_frame",
                "dark_fraction",
                "y_top",
                "y_bottom",
                "y_sphere",
                "crop_size_px",
                "crop_path",
            ])

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    continue

                print(f"  [QUICKTEST] droplet {droplet_id}, cam {cam}, file {path.name}")

                # --- Load ---
                t0 = time.perf_counter()
                c = safe_load_cine(path)
                folder_timing["load_cine"] += time.perf_counter() - t0
                
                if c is None:
                    print("    [WARN] Failed to load cine.")
                    continue

                # --- Darkness curve ---
                t0 = time.perf_counter()
                dark = analyze_cine_darkness(c)
                folder_timing["darkness_curve"] += time.perf_counter() - t0
                
                curve = dark["darkness_curve"]
                first = dark["first_frame"]
                last = dark["last_frame"]
                folder_timing["n_frames"] += len(curve)

                # --- Best frame ---
                t0 = time.perf_counter()
                best_idx, geo = choose_best_frame_with_geo(c, curve)
                folder_timing["best_frame"] += time.perf_counter() - t0
                
                dark_val = float(curve[best_idx - first])

                # --- Darkness plot ---
                t0 = time.perf_counter()
                save_darkness_plot(
                    out_sub / f"{path.stem}_darkness.png",
                    curve, first, last, best_idx, path.name
                )
                folder_timing["darkness_plot"] += time.perf_counter() - t0

                # --- Overlay plot ---
                t0 = time.perf_counter()
                save_geometric_overlay(
                    out_sub / f"{path.stem}_overlay.png",
                    geo, best_idx, CNN_SIZE=None
                )
                folder_timing["overlay_plot"] += time.perf_counter() - t0

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]

                writer.writerow([
                    droplet_id,
                    cam,
                    path.name,
                    first,
                    last,
                    best_idx,
                    dark_val,
                    y_top,
                    y_bottom,
                    y_sphere,
                    "",
                    "",
                ])

        folder_sec = time.time() - t_folder_start
        
        # Print timing breakdown
        print(f"\n  [TIMING BREAKDOWN]")
        print(f"    load_cine:      {folder_timing['load_cine']:.2f}s")
        print(f"    darkness_curve: {folder_timing['darkness_curve']:.2f}s")
        print(f"    best_frame:     {folder_timing['best_frame']:.2f}s")
        print(f"    darkness_plot:  {folder_timing['darkness_plot']:.2f}s")
        print(f"    overlay_plot:   {folder_timing['overlay_plot']:.2f}s")
        print(f"    total_frames:   {folder_timing['n_frames']}")
        print(f"    ---")
        print(f"    TOTAL:          {folder_sec:.2f}s")

        if profile:
            folders_profile.append({
                "folder": sub.name,
                "n_droplets_used": 1,
                "total_sec": folder_sec,
                "timing": folder_timing,
            })

    total_sec = global_timer.seconds
    print(f"\n=== PER-FOLDER QUICK TEST COMPLETE — {total_sec:.1f}s ===")

    if profile:
        prof_path = OUTPUT_ROOT / "profiling_perfolder_quicktest.json"
        with open(prof_path, "w") as f:
            json.dump({
                "mode": "per-folder_quicktest",
                "safe_mode": safe_mode,
                "total_folders": total_folders,
                "total_seconds": total_sec,
                "folders": folders_profile,
            }, f, indent=2)
        print(f"[PROFILE] Per-folder quicktest profiling → {prof_path}")


# ============================================================
# MAIN PER-FOLDER PIPELINE
# ============================================================
def process_per_folder_every_10_parallel(
        safe_mode: bool = False,
        profile: bool = False,
        quick_test: bool = False):
    """
    Per-folder pipeline with detailed timing breakdown.
    """
    if quick_test:
        _quick_test_per_folder(safe_mode=safe_mode, profile=profile)
        return

    # ------------------------------
    # NORMAL PER-FOLDER PIPELINE
    # ------------------------------
    subfolders = iter_subfolders(CINE_ROOT)
    total_folders = len(subfolders)

    total_droplets = 0
    for sub in subfolders:
        groups = group_cines_by_droplet(sub)
        ng = len(groups)
        total_droplets += len(range(0, ng, CINE_STEP))

    global_done = 0
    global_timer = Timer()
    folder_profiles = []
    
    # Global timing aggregates - keys differ between SAFE and FAST modes
    # SAFE: load_cine, darkness_curve, best_frame
    # FAST: load_cine, geometry_scan
    global_analysis_timing = {}
    global_output_timing = {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }

    mode_str = "SAFE" if safe_mode else "FAST"
    if profile:
        mode_str += " + PROFILING"
    print(f"\n[PER-FOLDER MODE] {mode_str}")
    print(f"Found {total_folders} subfolders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print("\n==============================")
        print(f"[FOLDER {f_idx}/{total_folders}] {sub.name}")
        print("==============================")

        folder_start = time.time()

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

        # Phase 1: Analysis
        print(f"[{sub.name}] Phase 1: Analyse droplets...")
        p1_timer = Timer()

        # Choose worker based on mode
        worker_func = _analyze_droplet_worker_safe if safe_mode else _analyze_droplet_worker_fast
        
        results = run_parallel(
            worker_func,
            droplets_to_process,
            desc=f"{sub.name}: analyse",
            safe_mode=safe_mode,
        )
        p1_sec = p1_timer.seconds
        print(f"[{sub.name}] Phase 1 done — {p1_timer.elapsed}")

        folder_analyses = {}
        all_diams = []
        all_gaps = []
        phase1_timings = []

        for droplet_id, cam_dict, diams, gaps, timing in results:
            folder_analyses[droplet_id] = cam_dict
            all_diams.extend(diams)
            all_gaps.extend(gaps)
            phase1_timings.append(timing)
        
        # Aggregate and print Phase 1 timing
        p1_timing_totals = _aggregate_timings(phase1_timings, "Phase 1 - Analysis")
        # Copy all keys from aggregated timings
        for k, v in p1_timing_totals.items():
            global_analysis_timing[k] = global_analysis_timing.get(k, 0.0) + v

        # Phase 2: Calibration
        p2_start = time.time()

        if not all_gaps:
            CNN_SIZE = 128
            print(f"[CAL:{sub.name}] No valid geometry → fallback 128×128")
        else:
            CNN_SIZE = compute_crop_size(
                all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS
            )
            print(f"[CAL RESULTS:{sub.name}] crop size = {CNN_SIZE} × {CNN_SIZE}")

        p2_sec = time.time() - p2_start

        # Phase 3: Outputs
        p3_timer = Timer()
        print(f"[{sub.name}] Phase 3: Outputs...")

        out_sub = OUTPUT_ROOT / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        output_args = [
            (droplet_id, folder_analyses[droplet_id], CNN_SIZE, str(out_sub))
            for droplet_id in folder_analyses
        ]

        results_out = run_parallel(
            _process_droplet_output_worker,
            output_args,
            desc=f"{sub.name}: outputs",
            safe_mode=safe_mode,
        )
        p3_sec = p3_timer.seconds

        # Separate messages and timings
        phase3_timings = []
        for msg, timing in results_out:
            print(msg)
            phase3_timings.append(timing)
        
        # Aggregate and print Phase 3 timing
        p3_timing_totals = _aggregate_timings(phase3_timings, "Phase 3 - Outputs")
        for k, v in p3_timing_totals.items():
            if k in global_output_timing:
                global_output_timing[k] += v

        print(f"[{sub.name}] Phase 3 done — {p3_timer.elapsed}")

        # Global progress
        global_done += len(output_args)
        pct = (global_done / total_droplets) * 100 if total_droplets > 0 else 100.0
        print(f"[GLOBAL PROGRESS] {global_done}/{total_droplets} — {pct:.1f}% "
              f"(elapsed {global_timer.elapsed})")

        # CSV summary
        csv_path = out_sub / f"{sub.name}_summary_perfolder.csv"
        print(f"[{sub.name}] Writing CSV → {csv_path}")

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
                    curve = info["curve"]  # None in fast mode
                    best_idx = info["best"]
                    first = info["first"]

                    geo = info["geo"]
                    y_top = geo["y_top"]
                    y_bottom = geo["y_bottom"]
                    y_sphere = geo["y_bottom_sphere"]

                    # dark_val only available if we have a curve
                    dark_val = ""
                    if curve is not None:
                        dark_val = float(curve[best_idx - first])
                    
                    crop_path = str(out_sub / f"{path.stem}_crop.png")

                    writer.writerow([
                        droplet_id, cam, path.name,
                        best_idx, dark_val,
                        y_top, y_bottom, y_sphere,
                        CNN_SIZE, crop_path,
                    ])

        print(f"[FOLDER COMPLETE] CSV saved.")

        folder_total_sec = time.time() - folder_start

        if profile:
            folder_profiles.append({
                "folder": sub.name,
                "phase1_sec": p1_sec,
                "phase2_sec": p2_sec,
                "phase3_sec": p3_sec,
                "total_sec": folder_total_sec,
                "n_droplets_used": len(output_args),
                "phase1_breakdown": p1_timing_totals,
                "phase3_breakdown": p3_timing_totals,
            })

    total_sec = global_timer.seconds
    
    # Print global timing summary
    print("\n" + "=" * 50)
    print("GLOBAL TIMING SUMMARY")
    print("=" * 50)
    print("\n[ANALYSIS PHASE - All Folders]")
    for k, v in sorted(global_analysis_timing.items()):
        if k == "n_frames":
            print(f"  {k}: {int(v)} frames total")
        else:
            print(f"  {k}: {v:.2f}s")
    
    print("\n[OUTPUT PHASE - All Folders]")
    for k, v in sorted(global_output_timing.items()):
        print(f"  {k}: {v:.2f}s")
    
    print(f"\n=== PER-FOLDER COMPLETE — {total_sec:.1f}s ===")

    if profile:
        prof_path = OUTPUT_ROOT / "profiling_perfolder.json"
        with open(prof_path, "w") as f:
            json.dump({
                "mode": "per-folder",
                "safe_mode": safe_mode,
                "quick_test": False,
                "total_seconds": total_sec,
                "global_analysis_timing": global_analysis_timing,
                "global_output_timing": global_output_timing,
                "folders": folder_profiles,
            }, f, indent=2)
        print(f"[PROFILE] Per-folder profiling → {prof_path}")