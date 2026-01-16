"""Per-folder pipeline orchestration.

Calibrates crop size separately for each folder, processing each
subfolder independently before moving to the next.

This mode is useful when:
- Different folders have different optical setups
- You want to process folders incrementally
- You need folder-specific calibration

Functions:
    process_per_folder: Main entry point for per-folder pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cine_io_modular import group_cines_by_droplet, get_cine_folders, iter_subfolders, safe_load_cine
import config_modular
from config_modular import CINE_ROOT, CROP_SAFETY_PIXELS, OUTPUT_ROOT
from crop_calibration_modular import compute_crop_size
from darkness_analysis_modular import (
    analyze_cine_darkness,
    choose_best_frame_geometry_only,
    choose_best_frame_with_geo,
)
from focus_metrics_modular import classify_folder_focus
from image_utils_modular import otsu_mask
from output_writer_modular import generate_droplet_outputs, write_folder_csv
from parallel_utils_modular import run_parallel
from plotting_modular import save_darkness_plot, save_geometric_overlay
from profiling_modular import (
    aggregate_timings,
    print_global_summary,
    save_profile_json,
)
from timing_utils_modular import Timer, format_time
from workers_modular import analyze_droplet_crops_only, analyze_droplet_full

logger = logging.getLogger(__name__)


def process_per_folder(
    safe_mode: bool = False,
    profile: bool = False,
    quick_test: bool = False,
    full_output: bool = True,
    gui_mode: bool = False,
    focus_classification: bool = True,
) -> None:
    """Execute per-folder pipeline.

    Calibrates crop size separately for each folder, processing
    folders sequentially with parallelization within each folder.

    Args:
        safe_mode: If True, run single-process for debugging.
        profile: If True, save profiling JSON with timing breakdown.
        quick_test: If True, process only first droplet per folder.
        full_output: If True, generate darkness plots and overlays.
                    If False, generate crops only (faster).
        gui_mode: If True, emit progress markers for GUI instead of tqdm.
        focus_classification: If True, run per-folder focus classification
                            after all folders are processed.

    Example:
        >>> process_per_folder(safe_mode=True, full_output=False)
        # Processes all folders in safe mode, crops only
    """
    if quick_test:
        _quick_test_per_folder(
            safe_mode=safe_mode, profile=profile, full_output=full_output,
            gui_mode=gui_mode
        )
        return

    subfolders = get_cine_folders(CINE_ROOT)
    total_folders = len(subfolders)

    if total_folders == 0:
        logger.warning("No folders with .cine files found!")
        print("[PER-FOLDER] No folders with .cine files found!")
        return

    # Count total droplets for progress tracking
    total_droplets = _count_total_droplets(subfolders)

    global_done = 0
    global_timer = Timer()
    folder_profiles: List[Dict[str, Any]] = []

    global_analysis_timing: Dict[str, float] = {}
    global_output_timing: Dict[str, float] = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }

    mode_str = "SAFE" if safe_mode else "FAST"
    output_str = "full output" if full_output else "crops only"
    if profile:
        mode_str += " + PROFILING"
    step = config_modular.CINE_STEP

    logger.info(f"Per-folder mode: {mode_str}, {output_str}, step={step}")
    print(f"\n[PER-FOLDER MODE] {mode_str}, {output_str}, step={step}")
    print(f"Found {total_folders} subfolders.\n")

    # Select worker based on output mode
    worker_func = analyze_droplet_full if full_output else analyze_droplet_crops_only

    for f_idx, sub in enumerate(subfolders, start=1):
        folder_result = _process_single_folder(
            sub=sub,
            f_idx=f_idx,
            total_folders=total_folders,
            worker_func=worker_func,
            safe_mode=safe_mode,
            gui_mode=gui_mode,
            profile=profile,
        )

        if folder_result is None:
            continue

        # Unpack results
        (
            p1_sec, p2_sec, p3_sec, folder_total_sec,
            p1_timing_totals, p3_timing_totals,
            n_outputs
        ) = folder_result

        # Accumulate global timings
        for k, v in p1_timing_totals.items():
            global_analysis_timing[k] = global_analysis_timing.get(k, 0.0) + v

        for k, v in p3_timing_totals.items():
            if k in global_output_timing:
                global_output_timing[k] += v

        # Global progress
        global_done += n_outputs
        pct = (global_done / total_droplets) * 100 if total_droplets > 0 else 100.0
        print(
            f"[GLOBAL] {global_done}/{total_droplets} — {pct:.1f}% "
            f"(elapsed {global_timer.elapsed})"
        )

        if profile:
            folder_profiles.append({
                "folder": sub.name,
                "phase1_sec": p1_sec,
                "phase2_sec": p2_sec,
                "phase3_sec": p3_sec,
                "total_sec": folder_total_sec,
                "n_droplets": n_outputs,
                "phase1_breakdown": p1_timing_totals,
                "phase3_breakdown": p3_timing_totals,
            })

    total_sec = global_timer.seconds

    print_global_summary(global_analysis_timing, global_output_timing)

    # Focus classification (per-folder)
    if focus_classification:
        logger.info("Running focus classification...")
        print("\n[PER-FOLDER] Running focus classification...")
        _run_focus_classification_perfolder()

    logger.info(f"Per-folder pipeline complete in {format_time(total_sec)}")
    print(f"\n=== PER-FOLDER COMPLETE — {format_time(total_sec)} ===")

    if profile:
        save_profile_json(
            OUTPUT_ROOT,
            "profiling_perfolder.json",
            {
                "mode": "per-folder",
                "safe_mode": safe_mode,
                "full_output": full_output,
                "step": config_modular.CINE_STEP,
                "total_seconds": total_sec,
                "global_analysis_timing": global_analysis_timing,
                "global_output_timing": global_output_timing,
                "folders": folder_profiles,
            },
        )


def _count_total_droplets(subfolders: List[Path]) -> int:
    """Count total droplets across all folders considering step size.

    Args:
        subfolders: List of folder paths containing .cine files.

    Returns:
        Total number of droplets that will be processed.
    """
    total = 0
    for sub in subfolders:
        groups = group_cines_by_droplet(sub)
        total += len(range(0, len(groups), config_modular.CINE_STEP))
    return total


def _process_single_folder(
    sub: Path,
    f_idx: int,
    total_folders: int,
    worker_func: Any,
    safe_mode: bool,
    gui_mode: bool,
    profile: bool,
) -> Optional[Tuple[float, float, float, float, Dict[str, float], Dict[str, float], int]]:
    """Process a single folder through all pipeline phases.

    Args:
        sub: Path to the folder to process.
        f_idx: Current folder index (1-based).
        total_folders: Total number of folders.
        worker_func: Worker function for analysis (full or crops-only).
        safe_mode: If True, run single-process.
        gui_mode: If True, emit progress markers.
        profile: If True, collect timing information.

    Returns:
        Tuple of (p1_sec, p2_sec, p3_sec, total_sec, p1_timing, p3_timing, n_outputs)
        or None if folder has no droplets.
    """
    print("\n" + "=" * 30)
    print(f"[FOLDER {f_idx}/{total_folders}] {sub.name}")
    print("=" * 30)

    folder_start = time.time()

    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)

    if n_groups == 0:
        logger.info(f"No droplets in folder: {sub.name}")
        print("  [INFO] No droplets in this folder.")
        return None

    selected_indices = list(range(0, n_groups, config_modular.CINE_STEP))
    droplets_to_process = [
        (groups[idx][0], groups[idx][1]) for idx in selected_indices
    ]

    # Phase 1: Analysis
    logger.debug(f"[{sub.name}] Starting Phase 1: Analysis")
    print(f"[{sub.name}] Phase 1: Analyse droplets...")
    p1_timer = Timer()

    results = run_parallel(
        worker_func,
        droplets_to_process,
        desc=f"{sub.name}: analyse",
        safe_mode=safe_mode,
        gui_mode=gui_mode,
    )
    p1_sec = p1_timer.seconds
    print(f"[{sub.name}] Phase 1 done — {p1_timer.elapsed}")

    folder_analyses: Dict[str, Dict] = {}
    all_diams: List[float] = []
    all_gaps: List[float] = []
    phase1_timings: List[Dict[str, float]] = []

    for droplet_id, cam_dict, diams, gaps, timing in results:
        folder_analyses[droplet_id] = cam_dict
        all_diams.extend(diams)
        all_gaps.extend(gaps)
        phase1_timings.append(timing)

    p1_timing_totals = aggregate_timings(phase1_timings, "Phase 1 - Analysis")

    # Phase 2: Calibration
    p2_start = time.time()

    if not all_gaps:
        cnn_size = 128
        logger.warning(f"[{sub.name}] No valid geometry, using fallback 128x128")
        print(f"[CAL:{sub.name}] No valid geometry → fallback 128×128")
    else:
        cnn_size = compute_crop_size(
            all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS
        )
        logger.info(f"[{sub.name}] Calibrated crop size: {cnn_size}x{cnn_size}")
        print(f"[CAL:{sub.name}] crop size = {cnn_size}×{cnn_size}")

    p2_sec = time.time() - p2_start

    # Phase 3: Outputs
    p3_timer = Timer()
    logger.debug(f"[{sub.name}] Starting Phase 3: Outputs")
    print(f"[{sub.name}] Phase 3: Outputs...")

    out_sub = OUTPUT_ROOT / sub.name
    out_sub.mkdir(parents=True, exist_ok=True)

    output_args = [
        (droplet_id, folder_analyses[droplet_id], cnn_size, str(out_sub))
        for droplet_id in folder_analyses
    ]

    results_out = run_parallel(
        generate_droplet_outputs,
        output_args,
        desc=f"{sub.name}: outputs",
        safe_mode=safe_mode,
        gui_mode=gui_mode,
    )
    p3_sec = p3_timer.seconds

    phase3_timings: List[Dict[str, float]] = []
    for msg, timing in results_out:
        phase3_timings.append(timing)

    p3_timing_totals = aggregate_timings(phase3_timings, "Phase 3 - Outputs")

    print(f"[{sub.name}] Phase 3 done — {p3_timer.elapsed}")

    # Write CSV
    csv_path = out_sub / f"{sub.name}_summary.csv"
    write_folder_csv(csv_path, folder_analyses, out_sub, cnn_size)
    logger.info(f"[{sub.name}] Summary CSV saved: {csv_path}")
    print(f"[{sub.name}] CSV saved.")

    folder_total_sec = time.time() - folder_start

    return (
        p1_sec, p2_sec, p3_sec, folder_total_sec,
        p1_timing_totals, p3_timing_totals,
        len(output_args)
    )


def _quick_test_per_folder(
    safe_mode: bool = False,
    profile: bool = False,
    full_output: bool = True,
    gui_mode: bool = False,
) -> None:
    """Quick test: process first droplet per folder only.

    Useful for verifying pipeline setup and detecting issues
    before running a full processing job.

    Args:
        safe_mode: If True, run single-process for debugging.
        profile: If True, save profiling JSON.
        full_output: If True, generate all plots.
        gui_mode: If True, emit progress markers for GUI.
    """
    subfolders = get_cine_folders(CINE_ROOT)
    total_folders = len(subfolders)

    if total_folders == 0:
        logger.warning("No folders with .cine files found!")
        print("[QUICK TEST] No folders with .cine files found!")
        return

    global_timer = Timer()
    global_timing: Dict[str, float] = {
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
    logger.info(f"Quick test: {total_folders} folders, {output_str}")
    print(
        f"\n[QUICK TEST - PER-FOLDER] {'SAFE' if safe_mode else 'FAST'} mode, "
        f"{output_str}"
    )
    print(f"Processing first droplet from each of {total_folders} folders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        folder_timing = _quick_test_single_folder(
            sub=sub,
            f_idx=f_idx,
            total_folders=total_folders,
            full_output=full_output,
        )

        if folder_timing is None:
            continue

        # Accumulate global timing
        for k in global_timing:
            global_timing[k] += folder_timing.get(k, 0.0)

        if profile:
            folder_profiles.append({
                "folder": sub.name,
                "total_sec": sum(v for k, v in folder_timing.items() if k != "n_frames"),
                "timing": folder_timing,
            })

    total_sec = global_timer.seconds

    _print_quick_test_summary(global_timing, total_sec)

    if profile:
        save_profile_json(
            OUTPUT_ROOT,
            "profiling_perfolder_quicktest.json",
            {
                "mode": "per-folder-quicktest",
                "safe_mode": safe_mode,
                "full_output": full_output,
                "total_seconds": total_sec,
                "global_timing": global_timing,
                "folders": folder_profiles,
            },
        )


def _quick_test_single_folder(
    sub: Path,
    f_idx: int,
    total_folders: int,
    full_output: bool,
) -> Optional[Dict[str, float]]:
    """Process first droplet from a single folder for quick testing.

    Args:
        sub: Path to the folder.
        f_idx: Current folder index (1-based).
        total_folders: Total number of folders.
        full_output: If True, generate all plots.

    Returns:
        Dictionary of timing measurements, or None if no droplets found.
    """
    print(f"\n[{f_idx}/{total_folders}] {sub.name}")
    t_folder = time.time()

    folder_timing: Dict[str, float] = {
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
        logger.info(f"No droplets found in {sub.name}")
        print("  No droplets found.")
        return None

    droplet_id, cams = groups[0]
    out_sub = OUTPUT_ROOT / sub.name
    out_sub.mkdir(parents=True, exist_ok=True)

    for cam in ("g", "v"):
        path = cams.get(cam)
        if path is None:
            continue

        t0 = time.perf_counter()
        cine_obj = safe_load_cine(path)
        folder_timing["load_cine"] += time.perf_counter() - t0

        if cine_obj is None:
            logger.warning(f"Failed to load cine: {path}")
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

    # Print folder summary
    analysis_time = (
        folder_timing["darkness_curve"]
        + folder_timing["best_frame"]
        + folder_timing["geometry_scan"]
    )
    print(
        f"  load: {format_time(folder_timing['load_cine'])} | "
        f"analysis: {format_time(analysis_time)} | "
        f"frames: {int(folder_timing['n_frames'])}"
    )

    return folder_timing


def _print_quick_test_summary(global_timing: Dict[str, float], total_sec: float) -> None:
    """Print summary of quick test results.

    Args:
        global_timing: Accumulated timing across all folders.
        total_sec: Total elapsed time in seconds.
    """
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    for k, v in sorted(global_timing.items()):
        if k == "n_frames":
            print(f"  {k}: {int(v)}")
        elif v > 0:
            print(f"  {k}: {format_time(v)}")

    logger.info(f"Quick test complete in {format_time(total_sec)}")
    print(f"\n=== QUICK TEST COMPLETE — {format_time(total_sec)} ===")


def _run_focus_classification_perfolder() -> None:
    """Run per-folder focus classification.

    Delegates to the shared implementation in pipeline_global to avoid
    code duplication.
    """
    # Import from pipeline_global to reuse the same function
    from pipeline_global import _run_focus_classification
    _run_focus_classification()
