# geom_analysis_modular.py
import numpy as np
import cv2
from image_utils_modular import load_frame_gray, otsu_mask


def analyze_frame_geometric(c, frame_idx, min_area=50):
    """
    Geometric analysis for sphere + droplet in a single frame.

    Rules:
      - Droplet must NOT touch left/right borders.
      - Sphere CAN touch borders.

    Returns dict:
      {
        "frame": gray_frame,
        "mask": dark_mask_bool,
        "y_top": droplet_top_or_None,
        "y_bottom": droplet_bottom_or_None,
        "y_bottom_sphere": sphere_top_or_None,
        "cx": droplet_centre_x_or_W/2,
      }
    """

    gray = load_frame_gray(c, frame_idx)
    H, W = gray.shape

    gray, dark_mask = otsu_mask(gray)

    # Connected components on dark mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8), connectivity=8
    )

    comps = []
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < min_area:
            continue

        comps.append({
            "label": lab,
            "x": x,
            "w": w,
            "y_top": y,
            "y_bottom": y + h - 1,
            "cx": centroids[lab][0],
            "area": area,
        })

    if len(comps) == 0:
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": None,
            "cx": W / 2.0,
        }

    # --- Find sphere (large, wide, near bottom, roughly central) ---
    sphere_cands = []
    for c_comp in comps:
        area = c_comp["area"]
        cx = c_comp["cx"]
        width_ratio = c_comp["w"] / W

        # Sphere is typically wide + large
        if area > min_area * 5 and width_ratio > 0.30:
            if abs(cx - W / 2.0) < W * 0.35:
                sphere_cands.append(c_comp)

    if len(sphere_cands) == 0:
        # Fallback: take the lowest-top component (closest to bottom)
        sphere = max(comps, key=lambda c_comp: c_comp["y_top"])
    else:
        sphere = max(sphere_cands, key=lambda c_comp: c_comp["y_top"])

    y_sphere = int(sphere["y_top"])

    # --- Droplet candidates: ABOVE sphere, not touching left/right ---
    droplet_cands = []
    for c_comp in comps:
        if c_comp["y_bottom"] < y_sphere:
            touches_left = (c_comp["x"] == 0)
            touches_right = (c_comp["x"] + c_comp["w"] == W)
            if not touches_left and not touches_right:
                droplet_cands.append(c_comp)

    if len(droplet_cands) == 0:
        # Droplet merged with sphere or missing
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": y_sphere,
            "cx": W / 2.0,
        }

    # Droplet = highest component above sphere
    droplet = min(droplet_cands, key=lambda c_comp: c_comp["y_top"])

    return {
        "frame": gray,
        "mask": dark_mask,
        "y_top": int(droplet["y_top"]),
        "y_bottom": int(droplet["y_bottom"]),
        "y_bottom_sphere": y_sphere,
        "cx": float(droplet["cx"]),
    }
