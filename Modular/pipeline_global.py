"""Global pipeline orchestration.

Calibrates crop size across ALL folders before processing.

REFACTORED: Now parallelizes at DROPLET level (not folder level) for:
    - Fine-grained progress tracking
    - Better CPU utilization across folders
    - Continuous progress bar updates
"""

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import config_modular
from cine_io_modular import group_cines_by_droplet, iter_subfolders, safe_load_cine
from config_modular import CINE_ROOT, CROP_SAFETY_PIXELS, OUTPUT_ROOT
from crop_calibration_modular import compute_crop_size, maybe_add_calibration_sample
from darkness_analysis_modular import (
    analyze_cine_darkness,
    choose_best_frame_geometry_only,
    choose_best_frame_with_geo,
)
from geom_analysis_modular import extract_geometry_info
from image_utils_modular import otsu_mask
from parallel_utils_modular import run_parallel
from plotting_modular import save_darkness_plot, save_geometric_overlay
from profiling_modular import (
    aggregate_timings,
    print_global_summary,
    save_profile_json,
)
from timing_utils_modular import Timer, format_time


# ============================================================
# DROPLET-LEVEL WORKERS FOR GLOBAL PIPELINE
# ============================================================

def _analyze_droplet_global_full(
    args: Tuple[Path, str, Dict[str, Path]],
) -> Tuple[str, str, Dict[str, Dict[str, Any]], List[float], List[float], Dict[str, float]]:
    """Analyse single droplet with full darkness curve (global mode).

    Args:
        args: Tuple of (folder_path, droplet_id, cams_dict).

    Returns:
        Tuple of (folder_name, droplet_id, cam_results, diams, gaps, timing).
    """
    folder_path, droplet_id, cams = args
    folder_name = folder_path.name
    
    cam_results: Dict[str, Dict[str, Any]] = {}
    diams: List[float] = []
    gaps: List[float] = []

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

        t0 = time.perf_counter()
        cine_obj = safe_load_cine(path)
        timing["load_cine"] += time.perf_counter() - t0

        if cine_obj is None:
            continue

        first, last = cine_obj.range
        timing["n_frames"] += last - first + 1

        t0 = time.perf_counter()
        dark = analyze_cine_darkness(cine_obj)
        timing["darkness_curve"] += time.perf_counter() - t0

        curve = dark["darkness_curve"]

        t0 = time.perf_counter()
        best_idx, geo = choose_best_frame_with_geo(cine_obj, curve)
        timing["best_frame"] += time.perf_counter() - t0

        cam_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": curve,
            "best": best_idx,
            "geo": extract_geometry_info(geo),
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (folder_name, droplet_id, cam_results, diams, gaps, timing)


def _analyze_droplet_global_crops_only(
    args: Tuple[Path, str, Dict[str, Path]],
) -> Tuple[str, str, Dict[str, Dict[str, Any]], List[float], List[float], Dict[str, float]]:
    """Analyse single droplet with geometry-only scan (global mode).

    Args:
        args: Tuple of (folder_path, droplet_id, cams_dict).

    Returns:
        Tuple of (folder_name, droplet_id, cam_results, diams, gaps, timing).
    """
    folder_path, droplet_id, cams = args
    folder_name = folder_path.name
    
    cam_results: Dict[str, Dict[str, Any]] = {}
    diams: List[float] = []
    gaps: List[float] = []

    timing = {
        "load_cine": 0.0,
        "geometry_scan": 0.0,
        "n_frames": 0,
    }

    for cam in ("g", "v"):
        path = cams.get(cam)
        if path is None:
            continue

        t0 = time.perf_counter()
        cine_obj = safe_load_cine(path)
        timing["load_cine"] += time.perf_counter() - t0

        if cine_obj is None:
            continue

        first, last = cine_obj.range
        timing["n_frames"] += last - first + 1

        t0 = time.perf_counter()
        best_idx, geo = choose_best_frame_geometry_only(cine_obj)
        timing["geometry_scan"] += time.perf_counter() - t0

        cam_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": None,
            "best": best_idx,
            "geo": extract_geometry_info(geo),
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (folder_name, droplet_id, cam_results, diams, gaps, timing)


def _generate_droplet_output_global(
    args: Tuple[str, str, Dict[str, Dict[str, Any]], int],
) -> Tuple[str, Dict[str, float]]:
    """Generate output for single droplet (global mode).

    Args:
        args: Tuple of (folder_name, droplet_id, cam_data, cnn_size).

    Returns:
        Tuple of (message, timing_dict).
    """
    import cv2
    from config_modular import FOCUS_METRICS_ENABLED
    from cropping_modular import crop_droplet_with_sphere_guard
    from focus_metrics_modular import compute_all_focus_metrics
    from image_utils_modular import load_frame_gray, otsu_mask
    
    folder_name, droplet_id, cam_data, cnn_size = args
    
    out_sub = OUTPUT_ROOT / folder_name
    out_sub.mkdir(parents=True, exist_ok=True)

    timing = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "focus_metrics": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }

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

        if y_top is None or y_bottom is None:
            continue

        # Reload frame
        t0 = time.perf_counter()
        cine_obj = safe_load_cine(path)
        if cine_obj is None:
            continue
        frame = load_frame_gray(cine_obj, best_idx)
        _, mask = otsu_mask(frame)
        timing["reload_frame"] += time.perf_counter() - t0

        # Generate crop
        t0 = time.perf_counter()
        crop = crop_droplet_with_sphere_guard(
            frame,
            y_top,
            y_bottom,
            cx,
            target_w=cnn_size,
            target_h=cnn_size,
            y_sphere=y_sphere,
            safety=CROP_SAFETY_PIXELS,
        )
        timing["crop"] += time.perf_counter() - t0

        # Compute focus metrics
        if FOCUS_METRICS_ENABLED:
            t0 = time.perf_counter()
            _ = compute_all_focus_metrics(crop)
            timing["focus_metrics"] += time.perf_counter() - t0

        # Save crop
        t0 = time.perf_counter()
        crop_path = out_sub / f"{path.stem}_crop.png"
        cv2.imwrite(str(crop_path), crop)
        timing["imwrite"] += time.perf_counter() - t0

        # Full output mode: generate plots
        if curve is not None:
            t0 = time.perf_counter()
            save_darkness_plot(
                out_sub / f"{path.stem}_darkness.png",
                curve,
                first,
                last,
                best_idx,
                path.name,
            )
            timing["darkness_plot"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            geo_for_plot = {
                "frame": frame,
                "mask": mask,
                "y_top": y_top,
                "y_bottom": y_bottom,
                "y_bottom_sphere": y_sphere,
                "cx": cx,
            }
            save_geometric_overlay(
                out_sub / f"{path.stem}_overlay.png",
                geo_for_plot,
                best_idx,
                cnn_size=cnn_size,
            )
            timing["overlay_plot"] += time.perf_counter() - t0

    return (f"{folder_name}/{droplet_id}", timing)


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_global(
    safe_mode: bool = False,
    profile: bool = False,
    quick_test: bool = False,
    full_output: bool = True,
    gui_mode: bool = False,
) -> None:
    """Execute global pipeline with droplet-level parallelization.

    Calibrates crop size across ALL folders, then processes.
    Progress tracks individual droplets, not folders.

    Args:
        safe_mode: If True, run single-process for debugging.
        profile: If True, save profiling JSON.
        quick_test: If True, process only first droplet per folder.
        full_output: If True, generate all plots.
        gui_mode: If True, print progress instead of tqdm bars.
    """
    if quick_test:
        _quick_test_global(
            safe_mode=safe_mode, profile=profile, full_output=full_output,
            gui_mode=gui_mode
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

    # ============================================================
    # COLLECT ALL DROPLETS FROM ALL FOLDERS
    # ============================================================
    print("[GLOBAL] Collecting droplets from all folders...")
    all_droplets: List[Tuple[Path, str, Dict[str, Path]]] = []
    folder_droplet_counts: Dict[str, int] = {}

    for sub in subfolders:
        groups = group_cines_by_droplet(sub)
        selected_indices = list(range(0, len(groups), step))
        folder_droplet_counts[sub.name] = len(selected_indices)
        
        for idx in selected_indices:
            droplet_id, cams = groups[idx]
            all_droplets.append((sub, droplet_id, cams))

    total_droplets = len(all_droplets)
    print(f"[GLOBAL] Found {total_droplets} droplets across {n_folders} folders")
    
    for folder_name, count in sorted(folder_droplet_counts.items()):
        print(f"  {folder_name}: {count} droplets")

    global_timer = Timer()
    global_analysis_timing: Dict[str, float] = {}
    global_output_timing = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "focus_metrics": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }

    # Select worker based on output mode
    worker_func = _analyze_droplet_global_full if full_output else _analyze_droplet_global_crops_only

    # ============================================================
    # PHASE 1: Analyse all droplets (parallelized at droplet level)
    # ============================================================
    print(f"\n[GLOBAL] Phase 1: Analysing {total_droplets} droplets...")
    phase1_timer = Timer()

    results = run_parallel(
        worker_func,
        all_droplets,
        desc="Analysing droplets",
        safe_mode=safe_mode,
        gui_mode=gui_mode,
    )
    phase1_sec = phase1_timer.seconds
    print(f"[GLOBAL] Phase 1 complete — {phase1_timer.elapsed}")

    # Organize results by folder
    # Structure: {folder_name: {droplet_id: {cam: info}}}
    folder_analyses: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    all_diams: List[float] = []
    all_gaps: List[float] = []
    phase1_timings: List[Dict[str, float]] = []

    for folder_name, droplet_id, cam_results, diams, gaps, timing in results:
        if folder_name not in folder_analyses:
            folder_analyses[folder_name] = {}
        folder_analyses[folder_name][droplet_id] = cam_results
        all_diams.extend(diams)
        all_gaps.extend(gaps)
        phase1_timings.append(timing)

    p1_timing_totals = aggregate_timings(phase1_timings, "Phase 1 - Analysis")
    for k, v in p1_timing_totals.items():
        global_analysis_timing[k] = global_analysis_timing.get(k, 0.0) + v

    # ============================================================
    # PHASE 2: Crop calibration
    # ============================================================
    phase2_start = time.time()
    cnn_size = compute_crop_size(
        all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS
    )
    phase2_sec = time.time() - phase2_start
    print(f"\n[GLOBAL] Crop size = {cnn_size}×{cnn_size}")
    print(f"[GLOBAL] Phase 2 (calibration) — {format_time(phase2_sec)}\n")

    # ============================================================
    # PHASE 3: Generate outputs (parallelized at droplet level)
    # ============================================================
    print(f"[GLOBAL] Phase 3: Generating {total_droplets} outputs...")
    phase3_timer = Timer()

    # Build output args: (folder_name, droplet_id, cam_data, cnn_size)
    output_args: List[Tuple[str, str, Dict[str, Dict[str, Any]], int]] = []
    for folder_name, droplets in folder_analyses.items():
        for droplet_id, cam_data in droplets.items():
            output_args.append((folder_name, droplet_id, cam_data, cnn_size))

    results_out = run_parallel(
        _generate_droplet_output_global,
        output_args,
        desc="Generating outputs",
        safe_mode=safe_mode,
        gui_mode=gui_mode,
    )
    phase3_sec = phase3_timer.seconds

    phase3_timings: List[Dict[str, float]] = []
    for msg, timing in results_out:
        phase3_timings.append(timing)
        global_output_timing["n_outputs"] += 1

    p3_timing_totals = aggregate_timings(phase3_timings, "Phase 3 - Outputs")
    for k, v in p3_timing_totals.items():
        if k in global_output_timing:
            global_output_timing[k] += v

    print(f"\n[GLOBAL] Phase 3 complete — {phase3_timer.elapsed}")

    # ============================================================
    # WRITE SUMMARY CSVs (one per folder)
    # ============================================================
    print("\n[GLOBAL] Writing summary CSVs...")
    _write_global_csvs(folder_analyses, cnn_size)

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

    print(f"\n=== GLOBAL PROCESSING COMPLETE — {format_time(total_sec)} ===")

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
                "total_droplets": total_droplets,
                "phase1_sec": phase1_sec,
                "phase2_sec": phase2_sec,
                "phase3_sec": phase3_sec,
                "total_seconds": total_sec,
                "global_analysis_timing": global_analysis_timing,
                "global_output_timing": global_output_timing,
            },
        )


def _write_global_csvs(
    folder_analyses: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    cnn_size: int,
) -> None:
    """Write summary CSV for each folder.

    Args:
        folder_analyses: Nested dict of {folder: {droplet: {cam: info}}}.
        cnn_size: Crop size used.
    """
    import cv2
    from config_modular import FOCUS_METRICS_ENABLED
    from cropping_modular import crop_droplet_with_sphere_guard
    from focus_metrics_modular import compute_all_focus_metrics
    from image_utils_modular import load_frame_gray

    for folder_name, droplets in folder_analyses.items():
        out_sub = OUTPUT_ROOT / folder_name
        out_sub.mkdir(parents=True, exist_ok=True)
        csv_path = out_sub / f"{folder_name}_summary.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            header = [
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
            ]
            if FOCUS_METRICS_ENABLED:
                header.extend([
                    "laplacian_var",
                    "tenengrad",
                    "tenengrad_var",
                    "brenner",
                    "norm_laplacian",
                    "energy_gradient",
                ])
            writer.writerow(header)

            for droplet_id, cam_data in droplets.items():
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

                    dark_val = ""
                    if curve is not None:
                        dark_val = float(curve[best_idx - first])

                    crop_path = str(out_sub / f"{path.stem}_crop.png")

                    # Compute focus metrics from saved crop
                    focus_metrics: Dict[str, float] = {}
                    if FOCUS_METRICS_ENABLED and Path(crop_path).exists():
                        crop_img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
                        if crop_img is not None:
                            focus_metrics = compute_all_focus_metrics(crop_img)

                    row = [
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
                        cnn_size,
                        crop_path,
                    ]
                    if FOCUS_METRICS_ENABLED:
                        row.extend([
                            focus_metrics.get("laplacian_var", ""),
                            focus_metrics.get("tenengrad", ""),
                            focus_metrics.get("tenengrad_var", ""),
                            focus_metrics.get("brenner", ""),
                            focus_metrics.get("norm_laplacian", ""),
                            focus_metrics.get("energy_gradient", ""),
                        ])
                    writer.writerow(row)

        print(f"  Saved: {csv_path.name}")


def _quick_test_global(
    safe_mode: bool = False,
    profile: bool = False,
    full_output: bool = True,
    gui_mode: bool = False,
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
                    from image_utils_modular import load_frame_gray
                    frame = load_frame_gray(cine_obj, best_idx)
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
            f"  load: {format_time(folder_timing['load_cine'])} | "
            f"analysis: {format_time(analysis_time)} | "
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
            print(f"  {k}: {format_time(v)}")

    print(f"\n=== QUICK TEST COMPLETE — {format_time(total_sec)} ===")

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
