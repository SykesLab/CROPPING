"""geometry_overlay — draw V1's geometry annotations on a cine frame.

Pure function lifted verbatim from V1's
``Inference/inference_gui.py:_draw_overlay``. Takes a grayscale frame
plus the engine's geometry dict and returns a BGR colour image with:

  - horizontal guide lines at droplet top / bottom / sphere
  - vertical pixel-distance annotations on the right edge
  - a cyan crop box (only on the best frame)
  - frame number label top-right
  - colour legend top-left
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def draw_geometry_overlay(
    frame_gray: np.ndarray,
    geo: Optional[Dict[str, Any]],
    crop_size: int,
    safety: int = 3,
    show_crop_box: bool = True,
    frame_idx: Optional[int] = None,
    best_idx: Optional[int] = None,
) -> np.ndarray:
    """Annotate a grayscale frame with V1-style geometry overlay.

    Parameters
    ----------
    frame_gray : np.ndarray
        Grayscale uint8 image (H, W).
    geo : dict or None
        Geometry dict from ``InferenceEngine.select_best_frame``. Expected
        keys: ``y_top``, ``y_bottom``, ``cx``, ``y_bottom_sphere``. None
        means "no droplet detected" — function stamps that on the frame.
    crop_size : int
        Pixel size of the crop box (only drawn on the best frame).
    safety : int
        Pixel gap to keep between the crop bottom and the sphere.
    show_crop_box : bool
        Whether to draw the cyan crop box at all.
    frame_idx, best_idx : int or None
        Used to (a) draw the frame number label and (b) decide whether
        this is THE best frame (crop box only drawn then).
    """
    rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    h, w = frame_gray.shape

    if geo is None:
        cv2.putText(rgb, "No droplet detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return rgb

    y_top = geo.get("y_top")
    y_bottom = geo.get("y_bottom")
    cx = geo.get("cx")
    y_sphere = geo.get("y_bottom_sphere")

    # Horizontal guide lines
    if y_top is not None:
        cv2.line(rgb, (0, int(y_top)), (w, int(y_top)), (0, 255, 0), 1)
    if y_bottom is not None:
        cv2.line(rgb, (0, int(y_bottom)), (w, int(y_bottom)),
                 (0, 165, 255), 1)
    if y_sphere is not None:
        cv2.line(rgb, (0, int(y_sphere)), (w, int(y_sphere)), (0, 0, 255), 2)

    # Top margin annotation
    if y_top is not None:
        top_margin_px = int(y_top)
        x_ann0 = w - 120
        cv2.line(rgb, (x_ann0, 0), (x_ann0, int(y_top)), (0, 255, 0), 1)
        cv2.putText(rgb, f"{top_margin_px} px",
                    (x_ann0 - 55, int(y_top // 2) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Droplet height annotation
    if y_top is not None and y_bottom is not None:
        dist_px = int(abs(y_bottom - y_top))
        mid_y = int(0.5 * (y_top + y_bottom))
        x_ann = w - 80
        cv2.line(rgb, (x_ann, int(y_top)), (x_ann, int(y_bottom)),
                 (255, 255, 255), 1)
        cv2.putText(rgb, f"{dist_px} px", (x_ann - 55, mid_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Gap-to-sphere annotation
    if y_bottom is not None and y_sphere is not None:
        gap_px = int(abs(y_sphere - y_bottom))
        mid_gap = int(0.5 * (y_bottom + y_sphere))
        x_ann2 = w - 40
        cv2.line(rgb, (x_ann2, int(y_bottom)), (x_ann2, int(y_sphere)),
                 (100, 100, 255), 1)
        cv2.putText(rgb, f"{gap_px} px", (x_ann2 - 55, mid_gap + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    # Crop box (only on the best frame)
    is_best = (frame_idx is not None and best_idx is not None
               and frame_idx == best_idx)
    if (show_crop_box and is_best and y_top is not None
            and y_bottom is not None and cx is not None):
        cy = int(0.5 * (y_top + y_bottom))
        half = crop_size // 2
        x0 = max(0, int(cx) - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + crop_size)
        y1 = min(h, y0 + crop_size)
        if y_sphere is not None and y1 > (y_sphere - safety):
            shift = y1 - int(y_sphere - safety)
            y0 = max(0, y0 - shift)
            y1 = y0 + crop_size
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(rgb, f"BEST - Crop {crop_size}x{crop_size}",
                    (x0, max(y0 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Frame number label (top-right)
    if frame_idx is not None:
        label = f"Frame {frame_idx}"
        if is_best:
            label += " (BEST)"
        cv2.putText(rgb, label, (w - 180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Colour legend (top-left)
    legend_y = 20
    items = [((0, 255, 0), "Droplet top"),
             ((0, 165, 255), "Droplet bottom"),
             ((0, 0, 255), "Sphere")]
    if is_best:
        items.append(((255, 255, 0), "Crop region"))
    for colour, label in items:
        cv2.line(rgb, (10, legend_y), (30, legend_y), colour, 2)
        cv2.putText(rgb, label, (35, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        legend_y += 18

    return rgb
