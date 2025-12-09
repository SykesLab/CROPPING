# frame_cropping_full.py
#
# Full geometric analysis for sphere + droplet detection.
#
# Rules:
#   - Droplet must NOT touch left/right borders.
#   - Sphere CAN touch borders.
#
# This geometry is used by:
#   - frame_selector_full.py   (for best-frame selection)
#   - cine_iterator_full...    (for cropping / overlays)

import numpy as np
import cv2
from pyphantom import utils


# ------------------------------------------------------------
# Load cine frame (grayscale)
# ------------------------------------------------------------
def _load_frame_gray(c, idx):
    fr = utils.FrameRange(idx, idx)
    frame = c.get_images(fr, Option=1)
    arr = np.squeeze(frame).astype(np.float32)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)


# ------------------------------------------------------------
# Otsu mask
# ------------------------------------------------------------
def _otsu_mask(gray):
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, (mask == 0)


# ------------------------------------------------------------
# Geometric analysis: sphere + droplet
# ------------------------------------------------------------
def analyze_frame_geometric(c, frame_idx, min_area=50):
    """
    Droplet must NOT touch left/right borders.
    Sphere CAN touch borders.
    """

    gray = _load_frame_gray(c, frame_idx)
    H, W = gray.shape

    gray, dark_mask = _otsu_mask(gray)

    # ================================
    # 1. Connected components on FULL mask
    # ================================
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
            "area": area
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

    # ================================
    # 2. Find sphere (large, near bottom, central-ish)
    # ================================
    sphere_cands = []

    for c_comp in comps:
        area = c_comp["area"]
        cx   = c_comp["cx"]
        width_ratio = c_comp["w"] / W

        # Sphere is wide and large
        if area > min_area * 5 and width_ratio > 0.30:
            # roughly central
            if abs(cx - W / 2.0) < W * 0.35:
                sphere_cands.append(c_comp)

    if len(sphere_cands) == 0:
        # Fallback: just take the lowest-top component
        sphere = max(comps, key=lambda c_comp: c_comp["y_top"])
    else:
        sphere = max(sphere_cands, key=lambda c_comp: c_comp["y_top"])

    y_sphere = int(sphere["y_top"])

    # ================================
    # 3. Droplet candidates:
    #    - above sphere
    #    - NOT touching left/right borders
    # ================================
    droplet_cands = []
    for c_comp in comps:
        if c_comp["y_bottom"] < y_sphere:
            touches_left  = (c_comp["x"] == 0)
            touches_right = (c_comp["x"] + c_comp["w"] == W)

            if not touches_left and not touches_right:
                droplet_cands.append(c_comp)

    if len(droplet_cands) == 0:
        # touching / merged with sphere, or no droplet
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": y_sphere,
            "cx": W / 2.0,
        }

    # Droplet = highest comp above sphere
    droplet = min(droplet_cands, key=lambda c_comp: c_comp["y_top"])

    return {
        "frame": gray,
        "mask": dark_mask,
        "y_top": int(droplet["y_top"]),
        "y_bottom": int(droplet["y_bottom"]),
        "y_bottom_sphere": y_sphere,
        "cx": float(droplet["cx"]),
    }


# ------------------------------------------------------------
# Cropping based on droplet centre
# ------------------------------------------------------------
def crop_autocenter_simple(frame, y_top, y_bottom, cx,
                           target_w=320, target_h=320,
                           y_sphere=None, safety=3):
    """
    Centres the crop on the droplet, but guarantees:
        - sphere never appears in crop
        - fixed, folder-wide crop size
        - no warping, pure crop
        - droplet remains exactly centered unless sphere constraint forces shift up
    """

    H, W = frame.shape

    # 1) droplet vertical centre
    cy = 0.5 * (y_top + y_bottom)
    half_h = target_h // 2
    half_w = target_w // 2

    # Raw centred crop
    x0 = int(cx - half_w)
    x1 = x0 + target_w
    y0 = int(cy - half_h)
    y1 = y0 + target_h

    # ---- NEW LOGIC: ensure sphere never appears ----
    if y_sphere is not None:
        # the crop bottom must be strictly above sphere minus safety margin
        max_y1 = y_sphere - safety

        if y1 > max_y1:
            # shift the entire crop up
            shift = y1 - max_y1
            y0 -= shift
            y1 -= shift

    # Clamp to image boundaries
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > H:
        y0 -= (y1 - H)
        y1 = H

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > W:
        x0 -= (x1 - W)
        x1 = W

    return frame[y0:y1, x0:x1]
