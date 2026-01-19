"""
Global pipeline orchestration.

Calibrates crop size across ALL folders before processing, ensuring
uniform crop dimensions across the entire dataset. Parallelises at the
droplet level for finer progress tracking and better CPU utilisation.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)
from cine_io import group_cines_by_droplet, get_cine_folders, iter_subfolders, safe_load_cine
from config import CINE_ROOT, CROP_SAFETY_PIXELS, OUTPUT_ROOT
from crop_calibration import compute_crop_size, maybe_add_calibration_sample
from darkness_analysis import (
    analyze_cine_darkness,
    choose_best_frame_geometry_only,
    choose_best_frame_with_geo,
)
from geom_analysis import extract_geometry_info
from focus_metrics import classify_folder_focus, suggest_thresholds
from image_utils import otsu_mask
from parallel_utils import run_parallel
from plotting import save_darkness_plot, save_geometric_overlay
from profiling import (
    aggregate_timings,
    print_global_summary,
    save_profile_json,
)
from profiling import Timer, format_time


# --- Droplet-level workers for global pipeline ---

def _analyze_droplet_global_full(
    args: Tuple[Path, str, Dict[str, Path]],
) -> Tuple[str, str, Dict[str, Dict[str, Any]], List[float], List[float], Dict[str, float]]:
    """Analyse single droplet with full darkness curve (global mode)."""
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
    """Analyse single droplet with geometry-only scan (global mode)."""
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
    """Generate output for single droplet (global mode)."""
    import cv2
    from config import FOCUS_METRICS_ENABLED
    from cropping import crop_droplet_with_sphere_guard
    from focus_metrics import compute_all_focus_metrics
    from image_utils import load_frame_gray, otsu_mask
    
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


# --- Main pipeline ---

def process_global(
    safe_mode: bool = False,
    profile: bool = False,
    quick_test: bool = False,
    full_output: bool = True,
    gui_mode: bool = False,
    focus_classification: bool = True,
) -> None:
    """
    Execute global pipeline with droplet-level parallelisation.

    Calibrates crop size across ALL folders, then processes all droplets
    with a uniform crop size. Progress tracks individual droplets.
    """
    if quick_test:
        _quick_test_global(
            safe_mode=safe_mode, profile=profile, full_output=full_output,
            gui_mode=gui_mode
        )
        return

    # Get step from config (set by main_runner)
    step = config.CINE_STEP

    mode_str = "SAFE" if safe_mode else "FAST"
    output_str = "full output" if full_output else "crops only"
    if profile:
        mode_str += " + PROFILING"
    if focus_classification:
        mode_str += " + FOCUS CLASSIFICATION"
    print(f"\n[GLOBAL MODE] {mode_str}, {output_str}, step={step}\n")

    subfolders = get_cine_folders(CINE_ROOT)
    n_folders = len(subfolders)

    if n_folders == 0:
        logger.warning("No folders with .cine files found!")
        print("[GLOBAL] No folders with .cine files found!")
        return

    # Collect all droplets from all folders
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
    logger.info(f"Found {total_droplets} droplets across {n_folders} folders")
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

    # Total work = Phase 1 (analysis) + Phase 3 (outputs)
    # Phase 2 (calibration) is instant, doesn't count
    total_work = total_droplets * 2

    # Phase 1: Analyse all droplets
    print(f"\n[GLOBAL] Phase 1: Analysing {total_droplets} droplets...")
    phase1_timer = Timer()

    results = run_parallel(
        worker_func,
        all_droplets,
        desc="Analysing droplets",
        safe_mode=safe_mode,
        gui_mode=gui_mode,
        progress_offset=0,
        progress_total=total_work,
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

    # Phase 2: Crop calibration
    phase2_start = time.time()
    cnn_size = compute_crop_size(
        all_diams, all_gaps, safety_pixels=CROP_SAFETY_PIXELS
    )
    phase2_sec = time.time() - phase2_start
    logger.info(f"Calibrated crop size: {cnn_size}x{cnn_size}")
    print(f"\n[GLOBAL] Crop size = {cnn_size}×{cnn_size}")
    print(f"[GLOBAL] Phase 2 (calibration) — {format_time(phase2_sec)}\n")

    # Phase 3: Generate outputs
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
        progress_offset=total_droplets,  # Continue from Phase 1
        progress_total=total_work,
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

    # Write summary CSVs (one per folder)
    print("\n[GLOBAL] Writing summary CSVs...")
    _write_global_csvs(folder_analyses, cnn_size)

    # Focus classification (per-folder)
    if focus_classification:
        logger.info("Running per-folder focus classification...")
        print("\n[GLOBAL] Running per-folder focus classification...")
        _run_focus_classification()

    total_sec = global_timer.seconds
    logger.info(f"Global pipeline complete in {format_time(total_sec)}")

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


def _run_focus_classification() -> None:
    """
    Run per-folder focus classification on all summary CSVs.

    Uses per-folder percentile thresholds (25th/75th) to classify crops
    as sharp, medium, or blurry based on Laplacian variance. This ensures
    balanced classification even when folders have different optical setups.

    Outputs saved to OUTPUT/Focus/:
        - focus_classified_all.csv: All crops with classifications
        - sharp_crops.csv: Only sharp crops (for CNN training)
        - focus_folder_stats.csv: Per-folder threshold statistics
        - focus_classification_summary.png: Visualisation
    """
    logger.info("Starting focus classification...")
    import shutil
    import matplotlib.pyplot as plt
    
    all_data = []
    folder_stats = []
    total_sharp_copied = 0
    
    # Create Focus output directory
    focus_dir = OUTPUT_ROOT / "Focus"
    focus_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each folder's CSV
    for csv_path in OUTPUT_ROOT.rglob("*_summary.csv"):
        # Skip CSVs inside the Focus directory
        if "Focus" in csv_path.parts:
            continue
            
        try:
            df = pd.read_csv(csv_path)
            
            if 'laplacian_var' not in df.columns or df['laplacian_var'].isna().all():
                print(f"  Skipping {csv_path.name} (no focus metrics)")
                continue
            
            folder_name = csv_path.parent.name
            scores = df['laplacian_var'].dropna().values
            
            if len(scores) < 4:
                print(f"  Skipping {csv_path.name} (too few samples: {len(scores)})")
                continue
            
            # Per-folder classification
            classifications, sharp_thresh, blur_thresh = classify_folder_focus(scores)
            
            # Add classification to dataframe
            df['focus_class'] = None
            valid_idx = df['laplacian_var'].notna()
            df.loc[valid_idx, 'focus_class'] = classifications
            
            # Add folder column
            df['folder'] = folder_name
            
            # Save updated CSV
            df.to_csv(csv_path, index=False)
            
            # Copy sharp images to Focus folder
            sharp_df = df[df['focus_class'] == 'sharp']
            if len(sharp_df) > 0:
                folder_focus_dir = focus_dir / folder_name
                folder_focus_dir.mkdir(parents=True, exist_ok=True)
                
                for _, row in sharp_df.iterrows():
                    crop_path = row.get('crop_path', '')
                    if crop_path and Path(crop_path).exists():
                        dest_path = folder_focus_dir / Path(crop_path).name
                        shutil.copy2(crop_path, dest_path)
                        total_sharp_copied += 1
            
            # Collect stats
            n_sharp = (df['focus_class'] == 'sharp').sum()
            n_medium = (df['focus_class'] == 'medium').sum()
            n_blurry = (df['focus_class'] == 'blurry').sum()
            
            folder_stats.append({
                'folder': folder_name,
                'n_total': len(df),
                'n_sharp': n_sharp,
                'n_medium': n_medium,
                'n_blurry': n_blurry,
                'sharp_thresh': sharp_thresh,
                'blur_thresh': blur_thresh,
                'mean_laplacian': df['laplacian_var'].mean(),
            })
            
            all_data.append(df)
            
            print(f"  {folder_name}: {n_sharp} sharp / {n_medium} medium / {n_blurry} blurry "
                  f"(thresholds: {blur_thresh:.0f}-{sharp_thresh:.0f})")
            
        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")
    
    if not all_data:
        print("  No valid CSVs found for focus classification")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save combined CSV to Focus directory
    combined_path = focus_dir / "focus_classified_all.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n  Saved combined dataset: Focus/{combined_path.name} ({len(combined_df)} crops)")
    
    # Save sharp-only CSV to Focus directory
    sharp_only_df = combined_df[combined_df['focus_class'] == 'sharp'].copy()
    # Add diameter column
    sharp_only_df["diameter_px"] = sharp_only_df["y_bottom"] - sharp_only_df["y_top"]
    sharp_path = focus_dir / "sharp_crops.csv"
    sharp_only_df.to_csv(sharp_path, index=False)
    print(f"  Saved sharp crops list: Focus/{sharp_path.name} ({len(sharp_only_df)} crops)")
    
    # Save folder statistics to Focus directory
    stats_df = pd.DataFrame(folder_stats)
    stats_path = focus_dir / "focus_folder_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved folder statistics: Focus/{stats_path.name}")
    
    # Print summary
    total_sharp = combined_df['focus_class'].eq('sharp').sum()
    total_medium = combined_df['focus_class'].eq('medium').sum()
    total_blurry = combined_df['focus_class'].eq('blurry').sum()
    
    print(f"\n  FOCUS CLASSIFICATION SUMMARY (per-folder thresholds)")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Total crops:  {len(combined_df)}")
    print(f"  Sharp:        {total_sharp} ({100*total_sharp/len(combined_df):.1f}%)")
    print(f"  Medium:       {total_medium} ({100*total_medium/len(combined_df):.1f}%)")
    print(f"  Blurry:       {total_blurry} ({100*total_blurry/len(combined_df):.1f}%)")
    print(f"  Sharp images copied to Focus/: {total_sharp_copied}")
    
    # Generate summary plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: classification breakdown
        ax1 = axes[0]
        counts = [total_sharp, total_medium, total_blurry]
        labels = [f'Sharp\n({total_sharp})', f'Medium\n({total_medium})', f'Blurry\n({total_blurry})']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax1.pie(counts, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        ax1.set_title('Focus Classification (Per-Folder Thresholds)')
        
        # Right: per-folder sharp count
        ax2 = axes[1]
        stats_df_sorted = stats_df.sort_values('n_sharp', ascending=True)
        colors = ['#2ecc71' if s > m + b else '#f39c12' if m > s + b else '#e74c3c' 
                  for s, m, b in zip(stats_df_sorted['n_sharp'], 
                                     stats_df_sorted['n_medium'], 
                                     stats_df_sorted['n_blurry'])]
        ax2.barh(range(len(stats_df_sorted)), stats_df_sorted['n_sharp'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(stats_df_sorted)))
        ax2.set_yticklabels(stats_df_sorted['folder'], fontsize=8)
        ax2.set_xlabel('Number of Sharp Crops')
        ax2.set_title('Sharp Crops per Folder')
        
        plt.tight_layout()
        plot_path = focus_dir / "focus_classification_summary.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved summary plot: Focus/{plot_path.name}")
        
    except Exception as e:
        print(f"  Could not generate plot: {e}")


def _write_global_csvs(
    folder_analyses: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    cnn_size: int,
) -> None:
    """Write summary CSV for each folder with droplet metadata and focus metrics."""
    logger.debug("Writing summary CSVs for all folders...")
    import cv2
    from config import FOCUS_METRICS_ENABLED
    from cropping import crop_droplet_with_sphere_guard
    from focus_metrics import compute_all_focus_metrics
    from image_utils import load_frame_gray

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
    subfolders = get_cine_folders(CINE_ROOT)
    total_folders = len(subfolders)
    
    if total_folders == 0:
        print("[QUICK TEST] No folders with .cine files found!")
        return

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
                    from image_utils import load_frame_gray
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
