"""Output generation: crops, plots, and CSV writing.

Memory-optimised: frames are reloaded when needed rather than stored.
"""

import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2

from cine_io_modular import group_cines_by_droplet, safe_load_cine
from config_modular import CROP_SAFETY_PIXELS, OUTPUT_ROOT
from cropping_modular import crop_droplet_with_sphere_guard
from image_utils_modular import load_frame_gray, otsu_mask
from plotting_modular import save_darkness_plot, save_geometric_overlay


# Optional callback for GUI notification
_on_image_saved: Optional[Callable[[Path], None]] = None


def set_image_callback(callback: Optional[Callable[[Path], None]]) -> None:
    """Set callback to be called when an image is saved.
    
    Args:
        callback: Function that takes a Path, or None to disable.
    """
    global _on_image_saved
    _on_image_saved = callback


def _notify_image_saved(path: Path) -> None:
    """Notify callback if set."""
    if _on_image_saved is not None:
        try:
            _on_image_saved(path)
        except Exception:
            pass  # Don't let GUI errors stop processing


def _reload_frame_and_mask(
    path: Path,
    best_idx: int,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Reload frame and compute mask for output generation.

    Args:
        path: Path to cine file.
        best_idx: Frame index to load.

    Returns:
        Tuple of (frame, mask) or (None, None) on failure.
    """
    cine_obj = safe_load_cine(path)
    if cine_obj is None:
        return None, None

    frame = load_frame_gray(cine_obj, best_idx)
    _, mask = otsu_mask(frame)

    return frame, mask


def generate_droplet_outputs(
    args: Tuple[str, Dict[str, Dict[str, Any]], int, str],
) -> Tuple[str, Dict[str, float]]:
    """Generate outputs for single droplet (both cameras).

    Args:
        args: Tuple of (droplet_id, cam_data, cnn_size, out_sub_path).

    Returns:
        Tuple of (message, timing_dict).
    """
    droplet_id, cam_data, cnn_size, out_sub_path = args
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
        frame, mask = _reload_frame_and_mask(path, best_idx)
        timing["reload_frame"] += time.perf_counter() - t0

        if frame is None:
            continue

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

        t0 = time.perf_counter()
        crop_path = out_sub_path / f"{path.stem}_crop.png"
        cv2.imwrite(str(crop_path), crop)
        timing["imwrite"] += time.perf_counter() - t0
        _notify_image_saved(crop_path)

        # Full output mode: generate plots
        if curve is not None:
            t0 = time.perf_counter()
            save_darkness_plot(
                out_sub_path / f"{path.stem}_darkness.png",
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
                out_sub_path / f"{path.stem}_overlay.png",
                geo_for_plot,
                best_idx,
                cnn_size=cnn_size,
            )
            timing["overlay_plot"] += time.perf_counter() - t0

    return (f"[DONE] {droplet_id}", timing)


def generate_folder_outputs(
    args: Tuple[str, Dict[Tuple[str, str], Dict[str, Any]], int, int],
) -> Tuple[str, Dict[str, float]]:
    """Generate outputs for all droplets in a folder.

    Args:
        args: Tuple of (sub_path, analyses_dict, cnn_size, step).

    Returns:
        Tuple of (message, timing_dict).
    """
    sub_path, analyses_dict, cnn_size, step = args
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
    selected = list(range(0, len(groups), step))

    out_sub = OUTPUT_ROOT / sub.name
    out_sub.mkdir(parents=True, exist_ok=True)

    csv_path = out_sub / f"{sub.name}_summary.csv"

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
            "crop_size_px",
            "crop_path",
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

                dark_val: Union[str, float] = ""
                if curve is not None:
                    dark_val = float(curve[best_idx - first])

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]
                cx = geo["cx"]

                crop_path = ""
                if y_top is not None and y_bottom is not None:
                    t0 = time.perf_counter()
                    frame, mask = _reload_frame_and_mask(path, best_idx)
                    timing["reload_frame"] += time.perf_counter() - t0

                    if frame is not None:
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

                        t0 = time.perf_counter()
                        out_crop = out_sub / f"{path.stem}_crop.png"
                        cv2.imwrite(str(out_crop), crop)
                        timing["imwrite"] += time.perf_counter() - t0
                        crop_path = str(out_crop)
                        _notify_image_saved(out_crop)

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
                    cnn_size,
                    crop_path,
                ])

    return (f"[DONE] {sub.name}", timing)


def write_folder_csv(
    csv_path: Union[str, Path],
    folder_analyses: Dict[str, Dict[str, Dict[str, Any]]],
    out_sub: Path,
    cnn_size: int,
) -> None:
    """Write CSV summary for a folder.

    Args:
        csv_path: Output CSV path.
        folder_analyses: Dict of {droplet_id: {cam: info}}.
        out_sub: Output subfolder path.
        cnn_size: Crop size used.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "droplet_id",
            "camera",
            "cine_file",
            "best_frame",
            "dark_fraction",
            "y_top",
            "y_bottom",
            "y_sphere",
            "crop_size_px",
            "crop_path",
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

                dark_val: Union[str, float] = ""
                if curve is not None:
                    dark_val = float(curve[best_idx - first])

                crop_path = str(out_sub / f"{path.stem}_crop.png")

                writer.writerow([
                    droplet_id,
                    cam,
                    path.name,
                    best_idx,
                    dark_val,
                    y_top,
                    y_bottom,
                    y_sphere,
                    cnn_size,
                    crop_path,
                ])
