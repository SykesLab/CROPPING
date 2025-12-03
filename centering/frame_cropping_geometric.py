# frame_cropping_geometric.py
#
# Provides:
#   - analyze_frame_geometric(c, frame_idx)
#   - crop_autocenter_simple(frame, y_top, y_bottom, cx, target_w, target_h)
#
# This file extracts the droplet geometry + autonomous cropping.

import numpy as np
import cv2
from pyphantom import utils


# ------------------------------------------------------------
# Load cine frame as grayscale uint8
# ------------------------------------------------------------
def _load_frame_gray(c, idx):
    fr = utils.FrameRange(idx, idx)
    frame = c.get_images(fr, Option=1)
    arr = np.squeeze(frame).astype(np.float32)

    if arr.ndim == 3:
        arr_gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        arr_gray = arr

    norm = cv2.normalize(arr_gray, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


# ------------------------------------------------------------
# Otsu mask helper
# ------------------------------------------------------------
def _otsu_mask(arr_gray):
    _, mask = cv2.threshold(
        arr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dark_mask = (mask == 0)
    return arr_gray, dark_mask


# ------------------------------------------------------------
# Geometric analysis: find top, bottom, sphere top, horizontal center
# ------------------------------------------------------------
def analyze_frame_geometric(c, frame_idx,
                            center_frac=0.5,
                            sphere_search_start_frac=0.55,
                            cut_above_sphere_margin=3):

    norm = _load_frame_gray(c, frame_idx)
    H, W = norm.shape

    norm, mask_dark = _otsu_mask(norm)

    # ------ approximate sphere top ------
    mid = int(H * sphere_search_start_frac)
    mid = max(0, min(H - 1, mid))
    sphere_rows = np.where(mask_dark[mid:].any(axis=1))[0]
    y_sphere = int(mid + sphere_rows[0]) if sphere_rows.size > 0 else None

    # ------ restrict detection to center strip ------
    droplet_mask = mask_dark.copy()
    cx_strip = W // 2
    half_strip = int(W * center_frac / 2)
    x0_strip = max(0, cx_strip - half_strip)
    x1_strip = min(W, cx_strip + half_strip)

    central_strip = np.zeros_like(droplet_mask, dtype=bool)
    central_strip[:, x0_strip:x1_strip] = True
    droplet_mask &= central_strip

    # remove rows below sphere
    if y_sphere is not None:
        cutoff = max(0, y_sphere - cut_above_sphere_margin)
        droplet_mask[cutoff:, :] = False

    dark_rows = np.where(droplet_mask.any(axis=1))[0]

    if dark_rows.size == 0:
        fallback_rows = np.where(mask_dark.any(axis=1))[0]
        if fallback_rows.size == 0:
            return {
                "frame": norm,
                "mask": mask_dark,
                "y_top": None,
                "y_bottom": None,
                "y_bottom_sphere": y_sphere,
                "cx": W / 2,
            }
        y_top = int(fallback_rows.min())
        y_bottom = int(fallback_rows.max())
    else:
        y_top = int(dark_rows.min())
        y_bottom = int(dark_rows.max())

    # ------ TRUE HORIZONTAL CENTER DETECTION ------
    cols = np.where(droplet_mask.any(axis=0))[0]
    if cols.size > 0:
        cx = (cols.min() + cols.max()) / 2.0
    else:
        cx = W / 2.0  # fallback

    return {
        "frame": norm,
        "mask": mask_dark,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "y_bottom_sphere": y_sphere,
        "cx": cx,  # horizontal droplet center
    }


# ------------------------------------------------------------
# Autonomous cropping using real droplet center
# ------------------------------------------------------------
def crop_autocenter_simple(frame, y_top, y_bottom, cx,
                           target_w=320, target_h=240):
    """
    Droplet-centered, distortion-free cropping.

    - Droplet vertical center = center of crop vertically
    - Droplet horizontal center = measured cx
    - NO resizing, NO padding, NO warping
    - Crop is fixed-size: target_h × target_w
    """

    H, W = frame.shape

    cy = (y_top + y_bottom) / 2.0

    x0 = int(cx - target_w / 2)
    x1 = x0 + target_w

    y0 = int(cy - target_h / 2)
    y1 = y0 + target_h

    # clamp horizontally
    if x0 < 0:
        x0 = 0
        x1 = target_w
    if x1 > W:
        x1 = W
        x0 = W - target_w

    # clamp vertically
    if y0 < 0:
        y0 = 0
        y1 = target_h
    if y1 > H:
        y1 = H
        y0 = H - target_h

    return frame[y0:y1, x0:x1]
