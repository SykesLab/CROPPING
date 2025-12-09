# pipeline_global_every10_parallel.py
#
# Global pipeline with DETAILED TIMING BREAKDOWN.
# Shows time spent in: cine loading, darkness curve, best-frame selection,
# cropping, darkness plot, geometry overlay.

from pathlib import Path
import csv
import cv2
import json
import time

from config_modular import OUTPUT_ROOT, CINE_ROOT, CINE_STEP, CROP_SAFETY_PIXELS
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
# WORKER A — Analyse folder SAFE MODE (with darkness curve)
# ============================================================
def _analyze_folder_for_calibration_safe(sub):
    """
    SAFE MODE: Analyse all selected droplets in a folder.
    Computes full darkness curve for diagnostics.
    """
    sub = Path(sub)
    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    
    # Timing accumulators
    timing = {
        "load_cine": 0.0,
        "darkness_curve": 0.0,
        "best_frame": 0.0,
        "n_frames": 0,
        "n_cines": 0,
    }
    
    if n_groups == 0:
        return (sub.name, {}, [], [], timing)

    selected = list(range(0, n_groups, CINE_STEP))

    folder_analyses = {}
    diams, gaps = [], []

    for g_index in selected:
        droplet_id, cams = groups[g_index]

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
            
            timing["n_cines"] += 1
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

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": curve,
                "best": best_idx,
                "geo": geo,
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps, timing)


# ============================================================
# WORKER A — Analyse folder FAST MODE (geometry only)
# ============================================================
def _analyze_folder_for_calibration_fast(sub):
    """
    FAST MODE: Analyse all selected droplets in a folder.
    Skips darkness curve - does single-pass geometry scan.
    """
    sub = Path(sub)
    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    
    # Timing accumulators
    timing = {
        "load_cine": 0.0,
        "geometry_scan": 0.0,
        "n_frames": 0,
        "n_cines": 0,
    }
    
    if n_groups == 0:
        return (sub.name, {}, [], [], timing)

    selected = list(range(0, n_groups, CINE_STEP))

    folder_analyses = {}
    diams, gaps = [], []

    for g_index in selected:
        droplet_id, cams = groups[g_index]

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
            
            timing["n_cines"] += 1
            first, last = c.range
            timing["n_frames"] += (last - first + 1)

            # --- Time: geometry scan (no darkness curve!) ---
            t0 = time.perf_counter()
            best_idx, geo = choose_best_frame_geometry_only(c)
            timing["geometry_scan"] += time.perf_counter() - t0

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": None,  # No curve in fast mode
                "best": best_idx,
                "geo": geo,
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps, timing)


# ============================================================
# WORKER B — Output folder (with detailed timing)
# ============================================================
def _process_folder_outputs(args):
    """
    Generate outputs for all droplets in a folder.
    Returns timing breakdown for: crop, darkness_plot, overlay_plot.
    """
    sub_path, analyses_dict, CNN_SIZE = args
    sub = Path(sub_path)

    # Timing accumulators
    timing = {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }

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
            "crop_size_px", "crop_path",
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

                timing["n_outputs"] += 1

                curve = info["curve"]  # None in fast mode
                first = info["first"]
                last = info["last"]
                best_idx = info["best"]
                geo = info["geo"]

                # dark_val only available if we have a curve
                dark_val = ""
                if curve is not None:
                    dark_val = float(curve[best_idx - first])

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]
                cx = geo["cx"]

                crop_path = ""
                if y_top is not None and y_bottom is not None:
                    # --- Time: cropping ---
                    t0 = time.perf_counter()
                    crop = crop_droplet_with_sphere_guard(
                        geo["frame"], y_top, y_bottom, cx,
                        target_w=CNN_SIZE, target_h=CNN_SIZE,
                        y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
                    )
                    timing["crop"] += time.perf_counter() - t0
                    
                    # --- Time: imwrite ---
                    t0 = time.perf_counter()
                    out_crop = out_sub / f"{path.stem}_crop.png"
                    cv2.imwrite(str(out_crop), crop)
                    timing["imwrite"] += time.perf_counter() - t0
                    crop_path = str(out_crop)

                # --- SAFE MODE ONLY: generate plots ---
                if curve is not None:
                    # --- Time: darkness plot ---
                    t0 = time.perf_counter()
                    save_darkness_plot(
                        out_sub / f"{path.stem}_darkness.png",
                        curve, first, last, best_idx, path.name
                    )
                    timing["darkness_plot"] += time.perf_counter() - t0

                    # --- Time: overlay plot ---
                    t0 = time.perf_counter()
                    save_geometric_overlay(
                        out_sub / f"{path.stem}_overlay.png",
                        geo, best_idx, CNN_SIZE=CNN_SIZE
                    )
                    timing["overlay_plot"] += time.perf_counter() - t0

                writer.writerow([
                    droplet_id, cam, path.name,
                    first, last,
                    best_idx, dark_val,
                    y_top, y_bottom, y_sphere,
                    CNN_SIZE, crop_path,
                ])

    return (f"[DONE] {sub.name}", timing)


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
        if k in ("n_frames", "n_cines", "n_outputs"):
            print(f"    {k}: {int(v)}")
        else:
            print(f"    {k}: {v:.2f}s")
    
    return totals


# ============================================================
# QUICK TEST IMPLEMENTATION (global mode)
# ============================================================
def _quick_test_global(safe_mode: bool, profile: bool):
    """
    Global QUICK TEST with detailed timing.
    """
    subfolders = iter_subfolders(CINE_ROOT)
    total_folders = len(subfolders)
    global_timer = Timer()

    folder_profiles = []
    
    # Global timing aggregates
    global_timing = {
        "load_cine": 0.0,
        "darkness_curve": 0.0,
        "best_frame": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "n_frames": 0,
    }

    print("\n[GLOBAL] Mode: QUICK TEST")
    print(f"Found {total_folders} folders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print("\n==============================")
        print(f"[FOLDER {f_idx}/{total_folders}] {sub.name}")
        print("==============================")

        t_folder = time.time()
        
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

        droplet_id, cams = groups[0]

        out_sub = OUTPUT_ROOT / (sub.name + "_quicktest")
        out_sub.mkdir(parents=True, exist_ok=True)

        csv_path = out_sub / f"{sub.name}_quicktest_summary_global.csv"
        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                "droplet_id", "camera", "cine_file",
                "first_frame", "last_frame",
                "best_frame", "dark_fraction",
                "y_top", "y_bottom", "y_sphere",
                "crop_size_px", "crop_path",
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

        folder_sec = time.time() - t_folder
        
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
        
        # Accumulate to global
        for k in global_timing:
            global_timing[k] += folder_timing[k]

        if profile:
            folder_profiles.append({
                "folder": sub.name,
                "n_droplets_used": 1,
                "total_sec": folder_sec,
                "timing": folder_timing,
            })

    total_sec = global_timer.seconds
    
    # Print global summary
    print("\n" + "=" * 50)
    print("GLOBAL TIMING SUMMARY (QUICK TEST)")
    print("=" * 50)
    for k, v in sorted(global_timing.items()):
        if k == "n_frames":
            print(f"  {k}: {int(v)} frames total")
        else:
            print(f"  {k}: {v:.2f}s")
    
    print(f"\n=== GLOBAL QUICK TEST COMPLETE — {total_sec:.1f}s ===")

    if profile:
        prof_path = OUTPUT_ROOT / "profiling_global_quicktest.json"
        with open(prof_path, "w") as f:
            json.dump({
                "mode": "global_quicktest",
                "safe_mode": safe_mode,
                "total_folders": total_folders,
                "total_seconds": total_sec,
                "global_timing": global_timing,
                "folders": folder_profiles,
            }, f, indent=2)
        print(f"[PROFILE] Global quicktest profiling → {prof_path}")


# ============================================================
# MAIN GLOBAL PIPELINE
# ============================================================
def process_global_every_10_parallel(
        safe_mode: bool = False,
        profile: bool = False,
        quick_test: bool = False):
    """
    Global pipeline with detailed timing breakdown.
    """
    if quick_test:
        _quick_test_global(safe_mode=safe_mode, profile=profile)
        return

    mode_str = "SAFE" if safe_mode else "FAST"
    if profile:
        mode_str += " + PROFILING"
    print(f"\n[GLOBAL] Mode: {mode_str}\n")

    subfolders = iter_subfolders(CINE_ROOT)
    n_folders = len(subfolders)

    global_timer = Timer()
    
    # Global timing aggregates - keys differ between SAFE and FAST modes
    # SAFE: load_cine, darkness_curve, best_frame
    # FAST: load_cine, geometry_scan
    global_analysis_timing = {}
    global_output_timing = {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }

    # Phase 1 — analyse folders
    print("[GLOBAL] Phase 1: Analysing all folders...")
    phase1_timer = Timer()
    
    # Choose worker based on mode
    worker_func = _analyze_folder_for_calibration_safe if safe_mode else _analyze_folder_for_calibration_fast
    
    results = run_parallel(
        worker_func,
        subfolders,
        desc="Global analysis",
        safe_mode=safe_mode,
    )
    phase1_sec = phase1_timer.seconds
    print(f"[GLOBAL] Phase 1 complete — {phase1_timer.elapsed}")

    global_analyses = {}
    all_diams, all_gaps = [], []
    phase1_timings = []

    for (subname, folder_analyses, diams, gaps, timing) in results:
        global_analyses[subname] = folder_analyses
        all_diams.extend(diams)
        all_gaps.extend(gaps)
        phase1_timings.append(timing)
    
    # Aggregate and print Phase 1 timing
    p1_timing_totals = _aggregate_timings(phase1_timings, "Phase 1 - Analysis")
    # Copy all keys from aggregated timings
    for k, v in p1_timing_totals.items():
        global_analysis_timing[k] = global_analysis_timing.get(k, 0.0) + v

    # Phase 2 — crop calibration
    phase2_start = time.time()
    CNN_SIZE = compute_crop_size(all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS)
    phase2_sec = time.time() - phase2_start
    print(f"\n[GLOBAL] Crop size = {CNN_SIZE} × {CNN_SIZE}")
    print(f"[GLOBAL] Phase 2 (calibration) — {phase2_sec:.3f}s\n")

    # Phase 3 — outputs
    print("[GLOBAL] Phase 3: Generating outputs...")
    phase3_timer = Timer()

    args_list = [
        (str(sub), global_analyses.get(sub.name, {}), CNN_SIZE)
        for sub in subfolders
    ]

    results_out = run_parallel(
        _process_folder_outputs,
        args_list,
        desc="Global outputs",
        safe_mode=safe_mode,
    )
    phase3_sec = phase3_timer.seconds

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

    print(f"\n[GLOBAL] Phase 3 complete — {phase3_timer.elapsed}")

    total_sec = global_timer.seconds
    
    # Print global timing summary
    print("\n" + "=" * 50)
    print("GLOBAL TIMING SUMMARY")
    print("=" * 50)
    print("\n[ANALYSIS PHASE]")
    for k, v in sorted(global_analysis_timing.items()):
        if k in ("n_frames", "n_cines"):
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.2f}s")
    
    print("\n[OUTPUT PHASE]")
    for k, v in sorted(global_output_timing.items()):
        if k == "n_outputs":
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.2f}s")
    
    print(f"\n[PHASE TOTALS]")
    print(f"  Phase 1 (analysis):    {phase1_sec:.2f}s")
    print(f"  Phase 2 (calibration): {phase2_sec:.3f}s")
    print(f"  Phase 3 (outputs):     {phase3_sec:.2f}s")
    
    print(f"\n=== GLOBAL PROCESSING COMPLETE — {total_sec:.1f}s ===")

    if profile:
        profile_data = {
            "mode": "global",
            "safe_mode": safe_mode,
            "profile": profile,
            "quick_test": False,
            "total_folders": n_folders,
            "phase1_sec_analysis": phase1_sec,
            "phase2_sec_calibration": phase2_sec,
            "phase3_sec_outputs": phase3_sec,
            "total_seconds": total_sec,
            "global_analysis_timing": global_analysis_timing,
            "global_output_timing": global_output_timing,
        }

        prof_path = OUTPUT_ROOT / "profiling_global.json"
        with open(prof_path, "w") as f:
            json.dump(profile_data, f, indent=2)
        print(f"[PROFILE] Global profiling → {prof_path}")