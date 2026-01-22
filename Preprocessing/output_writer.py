"""
Output generation: crops, plots, and CSV writing.

Memory-optimised - frames are reloaded when needed rather than stored.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2

from cine_io import group_cines_by_droplet, safe_load_cine
from config import CROP_SAFETY_PIXELS, OUTPUT_ROOT, FOCUS_METRICS_ENABLED
from cropping import crop_droplet_with_sphere_guard
from focus_metrics import compute_all_focus_metrics
from image_utils import load_frame_gray, otsu_mask
from plotting import save_darkness_plot, save_geometric_overlay

logger = logging.getLogger(__name__)


# Optional callback for GUI notification
_on_image_saved: Optional[Callable[[Path], None]] = None


def set_image_callback(callback: Optional[Callable[[Path], None]]) -> None:
    """Set callback to be called when an image is saved (for GUI notifications)."""
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
    """Reload frame and compute Otsu mask for output generation."""
    cine_obj = safe_load_cine(path)
    if cine_obj is None:
        return None, None

    frame = load_frame_gray(cine_obj, best_idx)
    _, mask = otsu_mask(frame)

    return frame, mask


def generate_droplet_outputs(
    args: Tuple[str, Dict[str, Dict[str, Any]], int, str],
) -> Tuple[str, Dict[str, float]]:
    """Generate outputs for single droplet (all cameras)."""
    droplet_id, cam_data, cnn_size, out_sub_path = args
    out_sub_path = Path(out_sub_path)

    timing = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }

    # Create camera subfolders with crops/ and visualizations/ inside each
    cam_dirs: Dict[str, Dict[str, Path]] = {}
    for cam in ("g", "v", "m"):
        cam_base = out_sub_path / cam
        cam_dirs[cam] = {
            "crops": cam_base / "crops",
            "visualizations": cam_base / "visualizations",
        }
        cam_dirs[cam]["crops"].mkdir(parents=True, exist_ok=True)
        cam_dirs[cam]["visualizations"].mkdir(parents=True, exist_ok=True)

    for cam, info in cam_data.items():
        path = info.get("path")
        if path is None:
            logger.warning(f"Missing path for {droplet_id} camera {cam}")
            continue

        curve = info.get("curve")
        first = info.get("first", 0)
        last = info.get("last", 0)
        best_idx = info.get("best", 0)
        geo = info.get("geo", {})

        y_top = geo.get("y_top")
        y_bottom = geo.get("y_bottom")
        y_sphere = geo.get("y_bottom_sphere")
        cx = geo.get("cx")

        if y_top is None or y_bottom is None:
            logger.debug(f"Skipping {droplet_id}/{cam}: missing geometry")
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

        # Save crop to camera subfolder
        t0 = time.perf_counter()
        crop_path = cam_dirs[cam]["crops"] / f"{path.stem}_crop.png"
        try:
            cv2.imwrite(str(crop_path), crop)
            _notify_image_saved(crop_path)
        except Exception as e:
            logger.error(f"Failed to write crop {crop_path}: {e}")
        timing["imwrite"] += time.perf_counter() - t0

        # Full output mode: generate plots to camera subfolder
        if curve is not None:
            viz_dir = cam_dirs[cam]["visualizations"]

            t0 = time.perf_counter()
            save_darkness_plot(
                viz_dir / f"{path.stem}_darkness.png",
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
                viz_dir / f"{path.stem}_overlay.png",
                geo_for_plot,
                best_idx,
                cnn_size=cnn_size,
            )
            timing["overlay_plot"] += time.perf_counter() - t0

    return (f"[DONE] {droplet_id}", timing)


def generate_folder_outputs(
    args: Tuple[str, Dict[Tuple[str, str], Dict[str, Any]], int, int],
) -> Tuple[str, Dict[str, float]]:
    """Generate outputs for all droplets in a folder."""
    sub_path, analyses_dict, cnn_size, step = args
    sub = Path(sub_path)

    timing = {
        "reload_frame": 0.0,
        "crop": 0.0,
        "focus_metrics": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }

    groups = group_cines_by_droplet(sub)
    selected = list(range(0, len(groups), step))

    out_sub = OUTPUT_ROOT / sub.name
    out_sub.mkdir(parents=True, exist_ok=True)

    # Create camera subfolders with crops/ and visualizations/ inside each
    cam_dirs: Dict[str, Dict[str, Path]] = {}
    for cam in ("g", "v", "m"):
        cam_base = out_sub / cam
        cam_dirs[cam] = {
            "crops": cam_base / "crops",
            "visualizations": cam_base / "visualizations",
        }
        cam_dirs[cam]["crops"].mkdir(parents=True, exist_ok=True)
        cam_dirs[cam]["visualizations"].mkdir(parents=True, exist_ok=True)

    csv_path = out_sub / f"{sub.name}_summary.csv"

    try:
        f = open(csv_path, "w", newline="")
    except IOError as e:
        logger.error(f"Failed to open CSV for writing: {csv_path}: {e}")
        return (f"[ERROR] {sub.name}: could not write CSV", timing)

    try:
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
            # Focus metrics
            "laplacian_var",
            "tenengrad",
            "tenengrad_var",
            "brenner",
            "norm_laplacian",
            "energy_gradient",
        ])

        for idx in selected:
            droplet_id, cams = groups[idx]

            for cam in ("g", "v", "m"):
                path = cams.get(cam)
                if path is None:
                    continue

                info = analyses_dict.get((droplet_id, cam))
                if info is None:
                    continue

                timing["n_outputs"] += 1

                curve = info.get("curve")
                first = info.get("first", 0)
                last = info.get("last", 0)
                best_idx = info.get("best", 0)
                geo = info.get("geo", {})

                dark_val: Union[str, float] = ""
                if curve is not None and best_idx >= first:
                    try:
                        dark_val = float(curve[best_idx - first])
                    except (IndexError, TypeError):
                        logger.warning(f"Invalid curve index for {droplet_id}/{cam}")

                y_top = geo.get("y_top")
                y_bottom = geo.get("y_bottom")
                y_sphere = geo.get("y_bottom_sphere")
                cx = geo.get("cx")

                crop_path = ""
                focus_metrics: Dict[str, float] = {}
                
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

                        # Compute focus metrics
                        if FOCUS_METRICS_ENABLED:
                            t0 = time.perf_counter()
                            focus_metrics = compute_all_focus_metrics(crop)
                            timing["focus_metrics"] += time.perf_counter() - t0

                        # Save crop to camera subfolder
                        t0 = time.perf_counter()
                        out_crop = cam_dirs[cam]["crops"] / f"{path.stem}_crop.png"
                        try:
                            cv2.imwrite(str(out_crop), crop)
                            crop_path = str(out_crop)
                            _notify_image_saved(out_crop)
                        except Exception as e:
                            logger.error(f"Failed to write crop {out_crop}: {e}")
                        timing["imwrite"] += time.perf_counter() - t0

                        # Save visualizations to camera subfolder
                        if curve is not None:
                            viz_dir = cam_dirs[cam]["visualizations"]

                            t0 = time.perf_counter()
                            save_darkness_plot(
                                viz_dir / f"{path.stem}_darkness.png",
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
                                viz_dir / f"{path.stem}_overlay.png",
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
                    # Focus metrics (empty string if not computed)
                    focus_metrics.get("laplacian_var", ""),
                    focus_metrics.get("tenengrad", ""),
                    focus_metrics.get("tenengrad_var", ""),
                    focus_metrics.get("brenner", ""),
                    focus_metrics.get("norm_laplacian", ""),
                    focus_metrics.get("energy_gradient", ""),
                ])

        logger.info(f"Wrote summary CSV: {csv_path}")
    except IOError as e:
        logger.error(f"Error writing CSV {csv_path}: {e}")
        return (f"[ERROR] {sub.name}: CSV write failed", timing)
    finally:
        f.close()

    return (f"[DONE] {sub.name}", timing)


def write_folder_csv(
    csv_path: Union[str, Path],
    folder_analyses: Dict[str, Dict[str, Dict[str, Any]]],
    out_sub: Path,
    cnn_size: int,
) -> bool:
    """Write CSV summary for a folder."""
    try:
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
                # Focus metrics
                "laplacian_var",
                "tenengrad",
                "tenengrad_var",
                "brenner",
                "norm_laplacian",
                "energy_gradient",
            ])

            for droplet_id, cam_dict in folder_analyses.items():
                for cam, info in cam_dict.items():
                    path = info.get("path")
                    if path is None:
                        continue

                    curve = info.get("curve")
                    best_idx = info.get("best", 0)
                    first = info.get("first", 0)
                    geo = info.get("geo", {})

                    y_top = geo.get("y_top")
                    y_bottom = geo.get("y_bottom")
                    y_sphere = geo.get("y_bottom_sphere")

                    dark_val: Union[str, float] = ""
                    if curve is not None and best_idx >= first:
                        try:
                            dark_val = float(curve[best_idx - first])
                        except (IndexError, TypeError):
                            logger.warning(f"Invalid curve index for {droplet_id}/{cam}")

                    crop_path_str = str(out_sub / f"{path.stem}_crop.png")

                    # Compute focus metrics from saved crop
                    focus_metrics: Dict[str, float] = {}
                    if FOCUS_METRICS_ENABLED:
                        try:
                            crop_img = cv2.imread(crop_path_str, cv2.IMREAD_GRAYSCALE)
                            if crop_img is not None:
                                focus_metrics = compute_all_focus_metrics(crop_img)
                        except Exception:
                            pass  # Leave metrics empty on error

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
                        crop_path_str,
                        # Focus metrics (empty string if not computed)
                        focus_metrics.get("laplacian_var", ""),
                        focus_metrics.get("tenengrad", ""),
                        focus_metrics.get("tenengrad_var", ""),
                        focus_metrics.get("brenner", ""),
                        focus_metrics.get("norm_laplacian", ""),
                        focus_metrics.get("energy_gradient", ""),
                    ])

        logger.info(f"Wrote folder CSV: {csv_path}")
        return True

    except IOError as e:
        logger.error(f"Failed to write folder CSV {csv_path}: {e}")
        return False
