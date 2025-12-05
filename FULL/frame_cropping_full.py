# frame_cropping_full.py
#
# Full geometric analysis for sphere + droplet detection.
# Critical fix: CC done on width-masked dark mask so that
# selection and overlay use EXACT SAME GEOMETRY.

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
# Geometric analysis using width-masked connected components
# ------------------------------------------------------------
def analyze_frame_geometric(c, frame_idx,
                            min_area=50,
                            center_frac=0.20):
    """
    Critical change:
    We apply width-mask BEFORE connected components, meaning
    the side wall noise can NEVER influence the blob geometry.

    Sphere = lowest connected component
    Droplet = highest connected component above sphere
    """

    gray = _load_frame_gray(c, frame_idx)
    H, W = gray.shape

    gray, dark_mask = _otsu_mask(gray)

    # ================================
    # 1. Width mask (remove all side noise)
    # ================================
    cx = W // 2
    half_w = int(W * center_frac / 2)

    x0 = max(0, cx - half_w)
    x1 = min(W, cx + half_w)

    width_mask = np.zeros_like(dark_mask, dtype=bool)
    width_mask[:, x0:x1] = True

    filt_mask = dark_mask & width_mask

    # ================================
    # 2. Connected components (on filtered mask only!)
    # ================================
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        filt_mask.astype(np.uint8), connectivity=8
    )

    comps = []
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < min_area:
            continue
        comps.append({
            "label": lab,
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
            "cx": W/2
        }

    # ================================
    # 3. Sphere = lowest-top component
    # ================================
    sphere = max(comps, key=lambda c: c["y_top"])
    y_sphere = int(sphere["y_top"])

    # ================================
    # 4. Droplet = highest comp above sphere
    # ================================
    droplet_cands = [c for c in comps if c["y_bottom"] < y_sphere]

    if len(droplet_cands) == 0:
        # Already touching sphere → not valid frame
        return {
            "frame": gray,
            "mask": dark_mask,
            "y_top": None,
            "y_bottom": None,
            "y_bottom_sphere": y_sphere,
            "cx": W/2
        }

    droplet = min(droplet_cands, key=lambda c: c["y_top"])

    return {
        "frame": gray,
        "mask": dark_mask,
        "y_top": int(droplet["y_top"]),
        "y_bottom": int(droplet["y_bottom"]),
        "y_bottom_sphere": y_sphere,
        "cx": float(droplet["cx"])
    }


# ------------------------------------------------------------
# Cropping based on droplet centre
# ------------------------------------------------------------
def crop_autocenter_simple(frame, y_top, y_bottom, cx,
                           target_w=320, target_h=240):

    H, W = frame.shape
    cy = (y_top + y_bottom) / 2

    x0 = int(cx - target_w/2)
    x1 = x0 + target_w

    y0 = int(cy - target_h/2)
    y1 = y0 + target_h

    # Clamp
    if x0 < 0:
        x0 = 0
        x1 = target_w
    if x1 > W:
        x1 = W
        x0 = W - target_w

    if y0 < 0:
        y0 = 0
        y1 = target_h
    if y1 > H:
        y1 = H
        y0 = H - target_h

    return frame[y0:y1, x0:x1]
