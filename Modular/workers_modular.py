"""Worker functions for droplet and folder analysis.

Provides worker functions for both per-folder and global pipelines.
Memory-optimised: workers do NOT store frames in results.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cine_io_modular import group_cines_by_droplet, safe_load_cine
from crop_calibration_modular import maybe_add_calibration_sample
from darkness_analysis_modular import (
    analyze_cine_darkness,
    choose_best_frame_geometry_only,
    choose_best_frame_with_geo,
)
from geom_analysis_modular import extract_geometry_info


# Type aliases for clarity
DropletResult = Tuple[
    str,                        # droplet_id
    Dict[str, Dict[str, Any]],  # folder_results
    List[float],                # diams
    List[float],                # gaps
    Dict[str, float],           # timing
]

FolderResult = Tuple[
    str,                        # folder_name
    Dict[Tuple[str, str], Dict[str, Any]],  # folder_analyses
    List[float],                # diams
    List[float],                # gaps
    Dict[str, float],           # timing
]


# ============================================================
# PER-DROPLET WORKERS (used by per-folder pipeline)
# ============================================================

def analyze_droplet_full(args: Tuple[str, Dict[str, Any]]) -> DropletResult:
    """Analyse single droplet with full darkness curve.

    Args:
        args: Tuple of (droplet_id, cams_dict).

    Returns:
        Tuple of (droplet_id, results, diams, gaps, timing).
    """
    droplet_id, cams = args
    folder_results: Dict[str, Dict[str, Any]] = {}
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

        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": curve,
            "best": best_idx,
            "geo": extract_geometry_info(geo),
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps, timing)


def analyze_droplet_crops_only(args: Tuple[str, Dict[str, Any]]) -> DropletResult:
    """Analyse single droplet with geometry-only scan.

    Args:
        args: Tuple of (droplet_id, cams_dict).

    Returns:
        Tuple of (droplet_id, results, diams, gaps, timing).
    """
    droplet_id, cams = args
    folder_results: Dict[str, Dict[str, Any]] = {}
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

        folder_results[cam] = {
            "path": path,
            "first": first,
            "last": last,
            "curve": None,
            "best": best_idx,
            "geo": extract_geometry_info(geo),
        }

        maybe_add_calibration_sample(diams, gaps, geo)

    return (droplet_id, folder_results, diams, gaps, timing)


# ============================================================
# PER-FOLDER WORKERS (used by global pipeline)
# ============================================================

def analyze_folder_full(args: Tuple[Path, int]) -> FolderResult:
    """Analyse all selected droplets in folder with full darkness curve.

    Args:
        args: Tuple of (path_to_folder, step).

    Returns:
        Tuple of (folder_name, analyses, diams, gaps, timing).
    """
    sub, step = args
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

    selected = list(range(0, n_groups, step))
    folder_analyses: Dict[Tuple[str, str], Dict[str, Any]] = {}
    diams: List[float] = []
    gaps: List[float] = []

    for g_index in selected:
        droplet_id, cams = groups[g_index]

        for cam in ("g", "v"):
            path = cams.get(cam)
            if path is None:
                continue

            t0 = time.perf_counter()
            cine_obj = safe_load_cine(path)
            timing["load_cine"] += time.perf_counter() - t0

            if cine_obj is None:
                continue

            timing["n_cines"] += 1
            first, last = cine_obj.range
            timing["n_frames"] += last - first + 1

            t0 = time.perf_counter()
            dark = analyze_cine_darkness(cine_obj)
            timing["darkness_curve"] += time.perf_counter() - t0

            curve = dark["darkness_curve"]

            t0 = time.perf_counter()
            best_idx, geo = choose_best_frame_with_geo(cine_obj, curve)
            timing["best_frame"] += time.perf_counter() - t0

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": curve,
                "best": best_idx,
                "geo": extract_geometry_info(geo),
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps, timing)


def analyze_folder_crops_only(args: Tuple[Path, int]) -> FolderResult:
    """Analyse all selected droplets in folder with geometry-only scan.

    Args:
        args: Tuple of (path_to_folder, step).

    Returns:
        Tuple of (folder_name, analyses, diams, gaps, timing).
    """
    sub, step = args
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

    selected = list(range(0, n_groups, step))
    folder_analyses: Dict[Tuple[str, str], Dict[str, Any]] = {}
    diams: List[float] = []
    gaps: List[float] = []

    for g_index in selected:
        droplet_id, cams = groups[g_index]

        for cam in ("g", "v"):
            path = cams.get(cam)
            if path is None:
                continue

            t0 = time.perf_counter()
            cine_obj = safe_load_cine(path)
            timing["load_cine"] += time.perf_counter() - t0

            if cine_obj is None:
                continue

            timing["n_cines"] += 1
            first, last = cine_obj.range
            timing["n_frames"] += last - first + 1

            t0 = time.perf_counter()
            best_idx, geo = choose_best_frame_geometry_only(cine_obj)
            timing["geometry_scan"] += time.perf_counter() - t0

            folder_analyses[(droplet_id, cam)] = {
                "path": path,
                "first": first,
                "last": last,
                "curve": None,
                "best": best_idx,
                "geo": extract_geometry_info(geo),
            }

            maybe_add_calibration_sample(diams, gaps, geo)

    return (sub.name, folder_analyses, diams, gaps, timing)
