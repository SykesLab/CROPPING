"""
Geometric analysis for sphere and droplet detection using connected components.

The sphere sits at the bottom of the frame (injection needle tip) and the
droplet hangs above it. We use size, position, and border-touching rules
to distinguish between them.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import GEOM_MIN_AREA, SPHERE_CENTER_TOLERANCE, SPHERE_WIDTH_RATIO
from image_utils import load_frame_gray, otsu_mask


def analyze_frame_geometric(
    cine_obj: Any,
    frame_idx: int,
    min_area: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect sphere and droplet positions in a single frame.

    The sphere is identified as a large, wide component near the image centre.
    The droplet is the highest component above the sphere that doesn't touch
    the left/right borders (to exclude partial frames or artefacts).

    Returns dict with: frame, mask, y_top, y_bottom, y_bottom_sphere, cx
    """
    if min_area is None:
        min_area = GEOM_MIN_AREA

    gray = load_frame_gray(cine_obj, frame_idx)
    height, width = gray.shape
    gray, dark_mask = otsu_mask(gray)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8), connectivity=8
    )

    # Collect components (skip background label 0)
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

    if not components:
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": None,
            "cx": width / 2.0,
        }

    # Sphere: large, spans significant width, roughly centred
    sphere_candidates = [
        c for c in components
        if (c["area"] > min_area * 5
            and c["w"] / width > SPHERE_WIDTH_RATIO
            and abs(c["cx"] - width / 2.0) < width * SPHERE_CENTER_TOLERANCE)
    ]

    # Fall back to lowest component if no clear sphere candidate
    if sphere_candidates:
        sphere = max(sphere_candidates, key=lambda c: c["y_top"])
    else:
        sphere = max(components, key=lambda c: c["y_top"])

    y_sphere = int(sphere["y_top"])

    # Droplet: above sphere, not touching left/right edges
    droplet_candidates = [
        c for c in components
        if (c["y_bottom"] < y_sphere
            and c["x"] != 0
            and c["x"] + c["w"] != width)
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
    """Extract just the coordinate info (without frame/mask) for storage."""
    return {
        "y_top": geo["y_top"],
        "y_bottom": geo["y_bottom"],
        "y_bottom_sphere": geo["y_bottom_sphere"],
        "cx": geo["cx"],
    }
