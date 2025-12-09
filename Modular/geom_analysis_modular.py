# geom_analysis_modular.py
#
# Geometric analysis for sphere + droplet detection in a single frame.
# Variable naming cleaned up to avoid any potential shadowing issues.

import numpy as np
import cv2
from image_utils_modular import load_frame_gray, otsu_mask


def analyze_frame_geometric(cine_obj, frame_idx, min_area=50):
    """
    Geometric analysis for sphere + droplet in a single frame.

    Rules:
      - Droplet must NOT touch left/right borders.
      - Sphere CAN touch borders.

    Parameters
    ----------
    cine_obj : Cine
        The loaded cine object.
    frame_idx : int
        Absolute frame index to analyse.
    min_area : int
        Minimum connected component area (pixels) to consider.

    Returns
    -------
    dict with keys:
        "frame": gray_frame (uint8)
        "mask": dark_mask_bool
        "y_top": droplet top row (int) or None
        "y_bottom": droplet bottom row (int) or None
        "y_bottom_sphere": sphere top row (int) or None
        "cx": droplet centre x (float) or W/2
    """
    gray = load_frame_gray(cine_obj, frame_idx)
    H, W = gray.shape

    gray, dark_mask = otsu_mask(gray)

    # Connected components on dark mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8), connectivity=8
    )

    # Build list of valid components
    components = []
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
    if len(components) == 0:
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": None,
            "cx": W / 2.0,
        }

    # --- Find sphere (large, wide, near bottom, roughly central) ---
    sphere_candidates = []
    for comp in components:
        area = comp["area"]
        cx = comp["cx"]
        width_ratio = comp["w"] / W

        # Sphere is typically wide + large
        if area > min_area * 5 and width_ratio > 0.30:
            if abs(cx - W / 2.0) < W * 0.35:
                sphere_candidates.append(comp)

    if len(sphere_candidates) == 0:
        # Fallback: take the lowest-top component (closest to bottom)
        sphere = max(components, key=lambda comp: comp["y_top"])
    else:
        sphere = max(sphere_candidates, key=lambda comp: comp["y_top"])

    y_sphere = int(sphere["y_top"])

    # --- Droplet candidates: ABOVE sphere, not touching left/right ---
    droplet_candidates = []
    for comp in components:
        if comp["y_bottom"] < y_sphere:
            touches_left = (comp["x"] == 0)
            touches_right = (comp["x"] + comp["w"] == W)
            if not touches_left and not touches_right:
                droplet_candidates.append(comp)

    if len(droplet_candidates) == 0:
        # Droplet merged with sphere or missing
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": y_sphere,
            "cx": W / 2.0,
        }

    # Droplet = highest component above sphere (smallest y_top)
    droplet = min(droplet_candidates, key=lambda comp: comp["y_top"])

    return {
        "frame": gray,
        "mask": dark_mask,
        "y_top": int(droplet["y_top"]),
        "y_bottom": int(droplet["y_bottom"]),
        "y_bottom_sphere": y_sphere,
        "cx": float(droplet["cx"]),
    }