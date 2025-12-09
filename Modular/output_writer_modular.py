# output_writer_modular.py
#
# Output generation: crops, plots, and CSV writing.
# Used by both per-folder and global pipelines.
#
# MEMORY OPTIMIZATION: Frames are reloaded here when needed,
# rather than being stored in memory during analysis phase.

import csv
import time
from pathlib import Path

import cv2

from config_modular import OUTPUT_ROOT, CINE_STEP, CROP_SAFETY_PIXELS
from cine_io_modular import group_cines_by_droplet, safe_load_cine
from cropping_modular import crop_droplet_with_sphere_guard
from plotting_modular import save_darkness_plot, save_geometric_overlay
from image_utils_modular import load_frame_gray, otsu_mask


def _reload_frame_and_mask(path, best_idx):
    """
    Reload a single frame from cine file and compute mask.
    Used during output phase for cropping and plotting.
    
    Returns:
        (frame, mask) or (None, None) on failure
    """
    c = safe_load_cine(path)
    if c is None:
        return None, None
    
    frame = load_frame_gray(c, best_idx)
    _, mask = otsu_mask(frame)
    
    return frame, mask


# ============================================================
# OUTPUT WORKER FOR PER-DROPLET (used by per-folder pipeline)
# ============================================================

def generate_droplet_outputs(args):
    """
    Generate outputs for a single droplet (both cameras).
    Reloads frames as needed (memory efficient).
    
    If curve is None: crops only (fast mode)
    If curve is not None: crops + all plots (full output mode)
    
    Args:
        args: tuple of (droplet_id, cam_data, CNN_SIZE, out_sub_path)
        
    Returns:
        (message, timing_dict)
    """
    droplet_id, cam_data, CNN_SIZE, out_sub_path = args
    out_sub_path = Path(out_sub_path)
    
    timing = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }

    for cam, info in cam_data.items():
        path = info["path"]
        curve = info["curve"]  # None in crops-only mode
        first = info["first"]
        last = info["last"]
        best_idx = info["best"]
        geo = info["geo"]  # Contains y_top, y_bottom, y_bottom_sphere, cx

        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]
        cx = geo["cx"]

        # Reload frame for cropping
        if y_top is not None and y_bottom is not None:
            t0 = time.perf_counter()
            frame, mask = _reload_frame_and_mask(path, best_idx)
            timing["reload_frame"] += time.perf_counter() - t0
            
            if frame is not None:
                t0 = time.perf_counter()
                crop = crop_droplet_with_sphere_guard(
                    frame, y_top, y_bottom, cx,
                    target_w=CNN_SIZE, target_h=CNN_SIZE,
                    y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
                )
                timing["crop"] += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                cv2.imwrite(str(out_sub_path / f"{path.stem}_crop.png"), crop)
                timing["imwrite"] += time.perf_counter() - t0

                # Full output mode: generate plots
                if curve is not None:
                    # Darkness plot
                    t0 = time.perf_counter()
                    save_darkness_plot(
                        out_sub_path / f"{path.stem}_darkness.png",
                        curve, first, last, best_idx, path.name
                    )
                    timing["darkness_plot"] += time.perf_counter() - t0

                    # Overlay plot (needs frame and mask)
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
                        out_sub_path / f"{path.stem}_overlay.png",
                        geo_for_plot, best_idx, CNN_SIZE=CNN_SIZE
                    )
                    timing["overlay_plot"] += time.perf_counter() - t0

    return (f"[DONE] {droplet_id}", timing)


# ============================================================
# OUTPUT WORKER FOR PER-FOLDER (used by global pipeline)
# ============================================================

def generate_folder_outputs(args):
    """
    Generate outputs for all droplets in a folder.
    Writes CSV and generates crops/plots.
    Reloads frames as needed (memory efficient).
    
    Args:
        args: tuple of (sub_path, analyses_dict, CNN_SIZE)
        
    Returns:
        (message, timing_dict)
    """
    sub_path, analyses_dict, CNN_SIZE = args
    sub = Path(sub_path)

    timing = {
        "reload_frame": 0.0,
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

    csv_path = out_sub / f"{sub.name}_summary.csv"

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

                curve = info["curve"]
                first = info["first"]
                last = info["last"]
                best_idx = info["best"]
                geo = info["geo"]

                dark_val = ""
                if curve is not None:
                    dark_val = float(curve[best_idx - first])

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]
                cx = geo["cx"]

                crop_path = ""
                if y_top is not None and y_bottom is not None:
                    # Reload frame
                    t0 = time.perf_counter()
                    frame, mask = _reload_frame_and_mask(path, best_idx)
                    timing["reload_frame"] += time.perf_counter() - t0
                    
                    if frame is not None:
                        t0 = time.perf_counter()
                        crop = crop_droplet_with_sphere_guard(
                            frame, y_top, y_bottom, cx,
                            target_w=CNN_SIZE, target_h=CNN_SIZE,
                            y_sphere=y_sphere, safety=CROP_SAFETY_PIXELS,
                        )
                        timing["crop"] += time.perf_counter() - t0
                        
                        t0 = time.perf_counter()
                        out_crop = out_sub / f"{path.stem}_crop.png"
                        cv2.imwrite(str(out_crop), crop)
                        timing["imwrite"] += time.perf_counter() - t0
                        crop_path = str(out_crop)

                        # Full output mode: generate plots
                        if curve is not None:
                            t0 = time.perf_counter()
                            save_darkness_plot(
                                out_sub / f"{path.stem}_darkness.png",
                                curve, first, last, best_idx, path.name
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
                                geo_for_plot, best_idx, CNN_SIZE=CNN_SIZE
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
# CSV WRITING UTILITIES
# ============================================================

def write_folder_csv(csv_path, folder_analyses, out_sub, CNN_SIZE):
    """
    Write CSV summary for a folder (per-folder pipeline).
    
    Args:
        csv_path: Path to output CSV
        folder_analyses: dict of {droplet_id: {cam: info}}
        out_sub: Output subfolder path
        CNN_SIZE: Crop size used
    """
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
