# darkness_analysis_modular.py
import numpy as np
import cv2
from image_utils_modular import load_frame_gray
from geom_analysis_modular import analyze_frame_geometric


def get_dark_fraction(c, idx):
    """
    Fraction of dark pixels in a frame (via Otsu).
    """
    gray = load_frame_gray(c, idx)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((mask == 0).mean())


def analyze_cine_darkness(c):
    """
    Compute darkness curve for entire cine.
    Returns dict with 'first_frame', 'last_frame', 'darkness_curve', 'total_frames'.
    """
    first, last = c.range
    curve = [get_dark_fraction(c, i) for i in range(first, last + 1)]
    return {
        "first_frame": first,
        "last_frame": last,
        "darkness_curve": np.array(curve, dtype=np.float32),
        "total_frames": len(curve),
    }


def choose_best_frame_with_geo(c, curve):
    """
    Best-frame selection using pre-collision geometry + darkness tie-breaker.

    Returns:
      best_frame_idx, best_geometry_dict
    """

    first_frame, last_frame = c.range

    dmin = float(curve.min())
    dmax = float(curve.max())
    dspan = max(1e-6, dmax - dmin)

    best_frame = None
    best_score = None
    best_geo = None

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

        top_margin = float(y_top)
        bottom_gap = float(y_sphere - y_bottom)

        cent_err = abs(top_margin - bottom_gap)

        dark_norm = (float(curve[idx - first_frame]) - dmin) / dspan

        score = -cent_err + 0.05 * dark_norm

        if (best_frame is None) or (score > best_score):
            best_frame = idx
            best_score = score
            best_geo = geo

    # Fallback: use darkest frame if no valid geo/pre-collision
    if best_frame is None:
        best_frame = first_frame + int(curve.argmax())
        best_geo = analyze_frame_geometric(c, best_frame)

    return best_frame, best_geo
