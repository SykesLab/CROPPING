"""Global pipeline orchestration.

Calibrates crop size across ALL folders before processing.
"""

import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import config_modular
from cine_io_modular import group_cines_by_droplet, iter_subfolders, safe_load_cine
from config_modular import CINE_ROOT, CROP_SAFETY_PIXELS, OUTPUT_ROOT
from crop_calibration_modular import compute_crop_size
from darkness_analysis_modular import (
    analyze_cine_darkness,
    choose_best_frame_geometry_only,
    choose_best_frame_with_geo,
)
from image_utils_modular import otsu_mask
from output_writer_modular import generate_folder_outputs
from parallel_utils_modular import run_parallel
from plotting_modular import save_darkness_plot, save_geometric_overlay
from profiling_modular import (
    aggregate_timings,
    print_global_summary,
    save_profile_json,
)
from timing_utils_modular import Timer
from workers_modular import analyze_folder_crops_only, analyze_folder_full


def process_global(
    safe_mode: bool = False,
    profile: bool = False,
    quick_test: bool = False,
    full_output: bool = True,
) -> None:
    """Execute global pipeline.

    Calibrates crop size across ALL folders, then processes.

    Args:
        safe_mode: If True, run single-process for debugging.
        profile: If True, save profiling JSON.
        quick_test: If True, process only first droplet per folder.
        full_output: If True, generate all plots.
    """
    if quick_test:
        _quick_test_global(
            safe_mode=safe_mode, profile=profile, full_output=full_output
        )
        return

    # Get step from config (set by main_runner)
    step = config_modular.CINE_STEP

    mode_str = "SAFE" if safe_mode else "FAST"
    output_str = "full output" if full_output else "crops only"
    if profile:
        mode_str += " + PROFILING"
    print(f"\n[GLOBAL MODE] {mode_str}, {output_str}, step={step}\n")

    subfolders = iter_subfolders(CINE_ROOT)
    n_folders = len(subfolders)

    global_timer = Timer()
    global_analysis_timing: Dict[str, float] = {}
    global_output_timing = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }

    # Select worker based on output mode
    worker_func = analyze_folder_full if full_output else analyze_folder_crops_only

    # Phase 1: Analyse all folders (pass step to workers)
    print("[GLOBAL] Phase 1: Analysing all folders...")
    phase1_timer = Timer()

    # Pack args as (subfolder, step) tuples
    analysis_args = [(sub, step) for sub in subfolders]

    results = run_parallel(
        worker_func,
        analysis_args,
        desc="Global analysis",
        safe_mode=safe_mode,
    )
    phase1_sec = phase1_timer.seconds
    print(f"[GLOBAL] Phase 1 complete — {phase1_timer.elapsed}")

    global_analyses: Dict[str, Dict] = {}
    all_diams: List[float] = []
    all_gaps: List[float] = []
    phase1_timings: List[Dict[str, float]] = []

    for subname, folder_analyses, diams, gaps, timing in results:
        global_analyses[subname] = folder_analyses
        all_diams.extend(diams)
        all_gaps.extend(gaps)
        phase1_timings.append(timing)

    p1_timing_totals = aggregate_timings(phase1_timings, "Phase 1 - Analysis")
    for k, v in p1_timing_totals.items():
        global_analysis_timing[k] = global_analysis_timing.get(k, 0.0) + v

    # Phase 2: Crop calibration
    phase2_start = time.time()
    cnn_size = compute_crop_size(
        all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS
    )
    phase2_sec = time.time() - phase2_start
    print(f"\n[GLOBAL] Crop size = {cnn_size}×{cnn_size}")
    print(f"[GLOBAL] Phase 2 (calibration) — {phase2_sec:.3f}s\n")

    # Phase 3: Generate outputs
    print("[GLOBAL] Phase 3: Generating outputs...")
    phase3_timer = Timer()

    args_list = [
        (str(sub), global_analyses.get(sub.name, {}), cnn_size, step)
        for sub in subfolders
    ]

    results_out = run_parallel(
        generate_folder_outputs,
        args_list,
        desc="Global outputs",
        safe_mode=safe_mode,
    )
    phase3_sec = phase3_timer.seconds

    phase3_timings: List[Dict[str, float]] = []
    for msg, timing in results_out:
        phase3_timings.append(timing)

    p3_timing_totals = aggregate_timings(phase3_timings, "Phase 3 - Outputs")
    for k, v in p3_timing_totals.items():
        if k in global_output_timing:
            global_output_timing[k] += v

    print(f"\n[GLOBAL] Phase 3 complete — {phase3_timer.elapsed}")

    total_sec = global_timer.seconds

    print_global_summary(
        global_analysis_timing,
        global_output_timing,
        phase_times={
            "phase1_sec": phase1_sec,
            "phase2_sec": phase2_sec,
            "phase3_sec": phase3_sec,
        },
    )

    print(f"\n=== GLOBAL PROCESSING COMPLETE — {total_sec:.1f}s ===")

    if profile:
        save_profile_json(
            OUTPUT_ROOT,
            "profiling_global.json",
            {
                "mode": "global",
                "safe_mode": safe_mode,
                "full_output": full_output,
                "step": step,
                "total_folders": n_folders,
                "phase1_sec": phase1_sec,
                "phase2_sec": phase2_sec,
                "phase3_sec": phase3_sec,
                "total_seconds": total_sec,
                "global_analysis_timing": global_analysis_timing,
                "global_output_timing": global_output_timing,
            },
        )


def _quick_test_global(
    safe_mode: bool = False,
    profile: bool = False,
    full_output: bool = True,
) -> None:
    """Quick test: first droplet per folder only."""
    subfolders = iter_subfolders(CINE_ROOT)
    total_folders = len(subfolders)

    global_timer = Timer()
    global_timing = {
        "load_cine": 0.0,
        "darkness_curve": 0.0,
        "best_frame": 0.0,
        "geometry_scan": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "n_frames": 0,
    }
    folder_profiles: List[Dict[str, Any]] = []

    output_str = "full output" if full_output else "crops only"
    print(f"\n[QUICK TEST - GLOBAL] {'SAFE' if safe_mode else 'FAST'} mode, {output_str}")
    print(f"Processing first droplet from each of {total_folders} folders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print(f"\n[{f_idx}/{total_folders}] {sub.name}")
        t_folder = time.time()

        folder_timing = {
            "load_cine": 0.0,
            "darkness_curve": 0.0,
            "best_frame": 0.0,
            "geometry_scan": 0.0,
            "darkness_plot": 0.0,
            "overlay_plot": 0.0,
            "n_frames": 0,
        }

        groups = group_cines_by_droplet(sub)
        if not groups:
            print("  No droplets found.")
            continue

        droplet_id, cams = groups[0]
        out_sub = OUTPUT_ROOT / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        csv_path = out_sub / f"{sub.name}_quicktest.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
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
            ])

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    continue

                t0 = time.perf_counter()
                cine_obj = safe_load_cine(path)
                folder_timing["load_cine"] += time.perf_counter() - t0

                if cine_obj is None:
                    continue

                first, last = cine_obj.range
                folder_timing["n_frames"] += last - first + 1

                if full_output:
                    t0 = time.perf_counter()
                    dark = analyze_cine_darkness(cine_obj)
                    folder_timing["darkness_curve"] += time.perf_counter() - t0
                    curve = dark["darkness_curve"]

                    t0 = time.perf_counter()
                    best_idx, geo = choose_best_frame_with_geo(cine_obj, curve)
                    folder_timing["best_frame"] += time.perf_counter() - t0

                    dark_val = float(curve[best_idx - first])

                    t0 = time.perf_counter()
                    save_darkness_plot(
                        out_sub / f"{path.stem}_darkness.png",
                        curve,
                        first,
                        last,
                        best_idx,
                        path.name,
                    )
                    folder_timing["darkness_plot"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    frame = geo["frame"]
                    _, mask = otsu_mask(frame)
                    geo_for_plot = {
                        "frame": frame,
                        "mask": mask,
                        "y_top": geo["y_top"],
                        "y_bottom": geo["y_bottom"],
                        "y_bottom_sphere": geo["y_bottom_sphere"],
                        "cx": geo["cx"],
                    }
                    save_geometric_overlay(
                        out_sub / f"{path.stem}_overlay.png",
                        geo_for_plot,
                        best_idx,
                        cnn_size=None,
                    )
                    folder_timing["overlay_plot"] += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    best_idx, geo = choose_best_frame_geometry_only(cine_obj)
                    folder_timing["geometry_scan"] += time.perf_counter() - t0
                    dark_val = ""

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
                ])

        folder_sec = time.time() - t_folder

        analysis_time = (
            folder_timing["darkness_curve"]
            + folder_timing["best_frame"]
            + folder_timing["geometry_scan"]
        )
        print(
            f"  load: {folder_timing['load_cine']:.2f}s | "
            f"analysis: {analysis_time:.2f}s | "
            f"frames: {folder_timing['n_frames']}"
        )

        for k in global_timing:
            global_timing[k] += folder_timing[k]

        if profile:
            folder_profiles.append({
                "folder": sub.name,
                "total_sec": folder_sec,
                "timing": folder_timing,
            })

    total_sec = global_timer.seconds

    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    for k, v in sorted(global_timing.items()):
        if k == "n_frames":
            print(f"  {k}: {int(v)}")
        elif v > 0:
            print(f"  {k}: {v:.2f}s")

    print(f"\n=== QUICK TEST COMPLETE — {total_sec:.1f}s ===")

    if profile:
        save_profile_json(
            OUTPUT_ROOT,
            "profiling_global_quicktest.json",
            {
                "mode": "global-quicktest",
                "safe_mode": safe_mode,
                "full_output": full_output,
                "total_seconds": total_sec,
                "global_timing": global_timing,
                "folders": folder_profiles,
            },
        )
