# workers_modular.py
#
# All worker functions for droplet/folder analysis.
# Used by both per-folder and global pipelines.
#
# MEMORY OPTIMIZATION: Workers do NOT store frames in results.
# Frames are reloaded during output phase when needed.

import time
from pathlib import Path

from config_modular import CINE_STEP
from cine_io_modular import group_cines_by_droplet, safe_load_cine
from darkness_analysis_modular import (
    analyze_cine_darkness,
    choose_best_frame_with_geo,
    choose_best_frame_geometry_only,
)
from crop_calibration_modular import maybe_add_calibration_sample


def _extract_geometry_info(geo):
    """
    Extract only the essential geometry info (no frame/mask).
    This is what gets stored in results for memory efficiency.
    """
    return {
        "y_top": geo["y_top"],
        "y_bottom": geo["y_bottom"],
        "y_bottom_sphere": geo["y_bottom_sphere"],
        "cx": geo["cx"],
    }


# ============================================================
# PER-DROPLET WORKERS (used by per-folder pipeline)
# ============================================================

def analyze_droplet_full(args):
    """
    FULL OUTPUT: Analyse a single droplet (both cameras).
    Computes darkness curve for plotting.
    
    Args:
        args: tuple of (droplet_id, cams_dict)
        
    Returns:
        (droplet_id, folder_results, diams, gaps, timing)
    """
    droplet_id, cams = args
    folder_results = {}
    diams = []
    gaps = []
    
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
        c = safe_load_cine(path)
        timing["load_cine"] += time.perf_counter() - t0
        
        if c is None:
            continue

        first, last = c.range
        timing["n_frames"] += (last - first + 1)

        t0 = time.perf_counter()
        dark = analyze_cine_darkness(c)
        timing["darkness_curve"] += time.perf_counter() - t0
        
        curve = dark["darkness_curve"]

        t0 = time.perf_counter()
        best_idx, geo = choose_best_frame_with_geo(c, curve)
        timing["best_frame"] += time.perf_counter() - t0

        # Store results WITHOUT frame/mask (memory optimization)
        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": curve,  # Small array, OK to store
            "best": best_idx,
            "geo": _extract_geometry_info(geo),
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps, timing)


def analyze_droplet_crops_only(args):
    """
    CROPS ONLY: Analyse a single droplet (both cameras).
    Skips darkness curve - does single-pass geometry scan.
    
    Args:
        args: tuple of (droplet_id, cams_dict)
        
    Returns:
        (droplet_id, folder_results, diams, gaps, timing)
    """
    droplet_id, cams = args
    folder_results = {}
    diams = []
    gaps = []
    
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
        c = safe_load_cine(path)
        timing["load_cine"] += time.perf_counter() - t0
        
        if c is None:
            continue

        first, last = c.range
        timing["n_frames"] += (last - first + 1)

        t0 = time.perf_counter()
        best_idx, geo = choose_best_frame_geometry_only(c)
        timing["geometry_scan"] += time.perf_counter() - t0

        # Store results WITHOUT frame/mask (memory optimization)
        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": None,  # No curve in crops-only mode
            "best": best_idx,
            "geo": _extract_geometry_info(geo),
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps, timing)


# ============================================================
# PER-FOLDER WORKERS (used by global pipeline)
# ============================================================

def analyze_folder_full(sub):
    """
    FULL OUTPUT: Analyse all selected droplets in a folder.
    Computes darkness curve for plotting.
    
    Args:
        sub: Path to folder (or string)
        
    Returns:
        (folder_name, folder_analyses, diams, gaps, timing)
    """
    sub = Path(sub)
    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    
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

            t0 = time.perf_counter()
            c = safe_load_cine(path)
            timing["load_cine"] += time.perf_counter() - t0
            
            if c is None:
                continue
            
            timing["n_cines"] += 1
            first, last = c.range
            timing["n_frames"] += (last - first + 1)

            t0 = time.perf_counter()
            dark = analyze_cine_darkness(c)
            timing["darkness_curve"] += time.perf_counter() - t0
            
            curve = dark["darkness_curve"]

            t0 = time.perf_counter()
            best_idx, geo = choose_best_frame_with_geo(c, curve)
            timing["best_frame"] += time.perf_counter() - t0

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": curve,
                "best": best_idx,
                "geo": _extract_geometry_info(geo),
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps, timing)


def analyze_folder_crops_only(sub):
    """
    CROPS ONLY: Analyse all selected droplets in a folder.
    Skips darkness curve - does single-pass geometry scan.
    
    Args:
        sub: Path to folder (or string)
        
    Returns:
        (folder_name, folder_analyses, diams, gaps, timing)
    """
    sub = Path(sub)
    groups = group_cines_by_droplet(sub)
    n_groups = len(groups)
    
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

            t0 = time.perf_counter()
            c = safe_load_cine(path)
            timing["load_cine"] += time.perf_counter() - t0
            
            if c is None:
                continue
            
            timing["n_cines"] += 1
            first, last = c.range
            timing["n_frames"] += (last - first + 1)

            t0 = time.perf_counter()
            best_idx, geo = choose_best_frame_geometry_only(c)
            timing["geometry_scan"] += time.perf_counter() - t0

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": None,
                "best": best_idx,
                "geo": _extract_geometry_info(geo),
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps, timing)
