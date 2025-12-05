# frame_selector_full.py
#
# Best-frame selection using strict pre-collision rule:
#   Sphere = lowest CC
#   Droplet = highest CC above sphere
#   Frame must be pre-collision
#   Best frame = minimises |top_margin - bottom_gap|
#
# 100% consistent with frame_cropping_full.py

import numpy as np
import cv2
from pyphantom import utils

from frame_cropping_full import analyze_frame_geometric


# -----------------------------------------------------------
# Load frame grayscale
# -----------------------------------------------------------
def _load_frame_gray(c, idx):
    fr = utils.FrameRange(idx, idx)
    frame = c.get_images(fr, Option=1)
    arr = np.squeeze(frame).astype(np.float32)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)


# -----------------------------------------------------------
# Darkness metric
# -----------------------------------------------------------
def _get_dark_fraction(c, idx):
    gray = _load_frame_gray(c, idx)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((mask == 0).mean())


# -----------------------------------------------------------
# Darkness curve
# -----------------------------------------------------------
def analyze_cine_full(c):
    first, last = c.range
    curve = [_get_dark_fraction(c, i) for i in range(first, last + 1)]
    return {
        "first_frame": first,
        "last_frame": last,
        "darkness_curve": np.array(curve, dtype=np.float32),
        "total_frames": len(curve),
    }


# -----------------------------------------------------------
# Best-frame = most equidistant droplet position (pre-collision)
# -----------------------------------------------------------
def choose_best_frame_full(c, curve, first):
    first_frame, last_frame = c.range

    dmin = float(curve.min())
    dmax = float(curve.max())
    dspan = max(1e-6, dmax - dmin)

    best_frame = None
    best_score = None

    for idx in range(first_frame, last_frame + 1):
        geo = analyze_frame_geometric(c, idx)
        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]

        # Must have full geometry
        if y_top is None or y_bottom is None or y_sphere is None:
            continue

        # Pre-collision: droplet entirely above sphere
        if y_bottom >= y_sphere:
            continue

        # Distances
        top_margin = float(y_top)
        bottom_gap = float(y_sphere - y_bottom)

        cent_err = abs(top_margin - bottom_gap)

        dark_norm = (float(curve[idx - first_frame]) - dmin) / dspan

        # MINIMISE cent_err, darkness is tie-breaker only
        score = -cent_err + 0.05 * dark_norm

        if (best_frame is None) or (score > best_score):
            best_frame = idx
            best_score = score

    if best_frame is None:
        best_frame = first + int(curve.argmax())

    return best_frame
