# frame_selector_geometric_4mm.py
#
# Provides:
#   - analyze_cine_geometric(c)
#   - choose_best_frame_geometric(c, curve, first)
#
# Hybrid scoring: geometry + darkness

import numpy as np
import cv2
from pyphantom import utils

from frame_cropping_geometric import analyze_frame_geometric


# -----------------------------------------------------------
# Utility: darkness metric for each frame
# -----------------------------------------------------------
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


def _get_dark_fraction(c, idx):
    gray = _load_frame_gray(c, idx)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dark_mask = (mask == 0)
    return float(dark_mask.mean())


# -----------------------------------------------------------
# Darkness curve
# -----------------------------------------------------------
def analyze_cine_geometric(c):
    first, last = c.range
    curve = [ _get_dark_fraction(c, idx) for idx in range(first, last + 1) ]
    return {
        "first_frame": first,
        "last_frame": last,
        "darkness_curve": np.array(curve, dtype=np.float32),
        "total_frames": (last - first + 1),
    }


# -----------------------------------------------------------
# Hybrid best-frame scoring
# -----------------------------------------------------------
def choose_best_frame_geometric(c, curve, first):
    first_frame, last_frame = c.range

    dmin = float(curve.min())
    dmax = float(curve.max())
    dspan = max(1e-6, dmax - dmin)

    best_frame = None
    best_score = -1.0

    for idx in range(first_frame, last_frame + 1):

        geo = analyze_frame_geometric(c, idx)
        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]

        if y_top is None or y_bottom is None or y_sphere is None:
            geom_score = 0.0
        else:
            top_margin = float(y_top)
            bottom_gap = float(y_sphere - y_bottom)
            centering_error = abs(top_margin - bottom_gap)
            geom_score = 1.0 / (1.0 + centering_error)

        dark_score = (float(curve[idx - first_frame]) - dmin) / dspan

        score = 0.7 * geom_score + 0.3 * dark_score

        if score > best_score:
            best_score = score
            best_frame = idx

    if best_frame is None:
        best_frame = first + int(curve.argmax())

    return best_frame
