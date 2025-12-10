"""Geometric analysis for sphere and droplet detection.

Analyses individual frames to detect sphere and droplet positions
using connected component analysis.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config_modular import (
    GEOM_MIN_AREA,
    SPHERE_CENTER_TOLERANCE,
    SPHERE_WIDTH_RATIO,
)
from image_utils_modular import load_frame_gray, otsu_mask


def analyze_frame_geometric(
    cine_obj: Any,
    frame_idx: int,
    min_area: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform geometric analysis for sphere and droplet detection.

    Detection rules:
        - Droplet must NOT touch left/right image borders.
        - Sphere CAN touch borders.
        - Sphere is identified as large, wide, central component near bottom.
        - Droplet is identified as highest component above sphere.

    Args:
        cine_obj: Loaded cine object.
        frame_idx: Absolute frame index to analyse.
        min_area: Minimum component area in pixels (default from config).

    Returns:
        Dictionary containing:
            - frame: Grayscale frame (uint8)
            - mask: Boolean dark mask
            - y_top: Droplet top row (int or None)
            - y_bottom: Droplet bottom row (int or None)
            - y_bottom_sphere: Sphere top row (int or None)
            - cx: Droplet centre x (float, defaults to W/2)
    """
    if min_area is None:
        min_area = GEOM_MIN_AREA

    gray = load_frame_gray(cine_obj, frame_idx)
    height, width = gray.shape
    gray, dark_mask = otsu_mask(gray)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8), connectivity=8
    )

    # Build component list (skip background label 0)
    components: List[Dict[str, Any]] = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area:
            continue

        components.append({
            "label": label_id,
            "x": x,
            "w": w,
            "y_top": y,
            "y_bottom": y + h - 1,
            "cx": centroids[label_id][0],
            "area": area,
        })

    # No components found
    if not components:
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": None,
            "cx": width / 2.0,
        }

    # Find sphere: large, wide, roughly central
    sphere_candidates = [
        comp for comp in components
        if (comp["area"] > min_area * 5
            and comp["w"] / width > SPHERE_WIDTH_RATIO
            and abs(comp["cx"] - width / 2.0) < width * SPHERE_CENTER_TOLERANCE)
    ]

    # Fallback: take lowest component if no candidates
    if sphere_candidates:
        sphere = max(sphere_candidates, key=lambda c: c["y_top"])
    else:
        sphere = max(components, key=lambda c: c["y_top"])

    y_sphere = int(sphere["y_top"])

    # Find droplet: above sphere, not touching left/right borders
    droplet_candidates = [
        comp for comp in components
        if (comp["y_bottom"] < y_sphere
            and comp["x"] != 0
            and comp["x"] + comp["w"] != width)
    ]

    if not droplet_candidates:
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": y_sphere,
            "cx": width / 2.0,
        }

    # Droplet is highest component above sphere
    droplet = min(droplet_candidates, key=lambda c: c["y_top"])

    return {
        "frame": gray,
        "mask": dark_mask,
        "y_top": int(droplet["y_top"]),
        "y_bottom": int(droplet["y_bottom"]),
        "y_bottom_sphere": y_sphere,
        "cx": float(droplet["cx"]),
    }


def extract_geometry_info(geo: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential geometry info without frame/mask.

    Used for memory-efficient storage between analysis and output phases.

    Args:
        geo: Full geometry dict from analyze_frame_geometric.

    Returns:
        Dict with only y_top, y_bottom, y_bottom_sphere, cx.
    """
    return {
        "y_top": geo["y_top"],
        "y_bottom": geo["y_bottom"],
        "y_bottom_sphere": geo["y_bottom_sphere"],
        "cx": geo["cx"],
    }
