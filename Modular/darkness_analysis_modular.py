# darkness_analysis_modular.py
#
# Darkness curve computation and best-frame selection.
# 
# TWO MODES:
#   FULL OUTPUT: Darkness curve + candidate-based geometry (for plots)
#   CROPS ONLY: Single pass geometry scan (faster, no darkness curve)

import numpy as np
import cv2
from image_utils_modular import load_frame_gray
from geom_analysis_modular import analyze_frame_geometric
from config_modular import N_CANDIDATES, DARKNESS_THRESHOLD_PERCENTILE, DARKNESS_WEIGHT


def get_dark_fraction(cine_obj, idx):
    """
    Fraction of dark pixels in a frame (via Otsu).
    """
    gray = load_frame_gray(cine_obj, idx)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((mask == 0).mean())


def analyze_cine_darkness(cine_obj):
    """
    Compute darkness curve for entire cine.
    Used in full output mode for diagnostics/plotting.
    
    Returns dict with 'first_frame', 'last_frame', 'darkness_curve', 'total_frames'.
    """
    first, last = cine_obj.range
    curve = [get_dark_fraction(cine_obj, i) for i in range(first, last + 1)]
    return {
        "first_frame": first,
        "last_frame": last,
        "darkness_curve": np.array(curve, dtype=np.float32),
        "total_frames": len(curve),
    }


# ============================================================
# CROPS ONLY MODE: Direct geometry scan (no darkness curve)
# ============================================================
def choose_best_frame_geometry_only(cine_obj):
    """
    CROPS ONLY MODE: Find best frame by scanning geometry on all frames directly.
    
    No darkness curve computed - just one pass through all frames doing
    full geometry analysis and picking the best pre-collision frame.
    
    Returns
    -------
    best_frame_idx : int
    best_geo : dict
    """
    first_frame, last_frame = cine_obj.range
    
    best_frame = None
    best_score = None
    best_geo = None
    
    for idx in range(first_frame, last_frame + 1):
        geo = analyze_frame_geometric(cine_obj, idx)
        
        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]
        
        # Must have full geometry
        if y_top is None or y_bottom is None or y_sphere is None:
            continue
        
        # Pre-collision: droplet entirely above sphere
        if y_bottom >= y_sphere:
            continue
        
        # Score: prefer well-centred droplets (equal top margin and bottom gap)
        top_margin = float(y_top)
        bottom_gap = float(y_sphere - y_bottom)
        cent_err = abs(top_margin - bottom_gap)
        
        score = -cent_err  # Lower error = higher score
        
        if (best_frame is None) or (score > best_score):
            best_frame = idx
            best_score = score
            best_geo = geo
    
    # Fallback: if no valid pre-collision frame, take middle frame
    if best_frame is None:
        mid_frame = (first_frame + last_frame) // 2
        best_frame = mid_frame
        best_geo = analyze_frame_geometric(cine_obj, mid_frame)
    
    return best_frame, best_geo


# ============================================================
# FULL OUTPUT MODE: Darkness curve + candidate-based geometry
# ============================================================
def _find_candidate_frames(curve, first_frame, n_candidates=None, threshold_percentile=None):
    """
    Find candidate frame indices where darkness is high (likely droplet visible).
    """
    if n_candidates is None:
        n_candidates = N_CANDIDATES
    if threshold_percentile is None:
        threshold_percentile = DARKNESS_THRESHOLD_PERCENTILE
    
    threshold = np.percentile(curve, threshold_percentile)
    above_thresh = np.where(curve >= threshold)[0]
    
    if len(above_thresh) == 0:
        peak_idx = int(np.argmax(curve))
        return [first_frame + peak_idx]
    
    sorted_by_darkness = sorted(above_thresh, key=lambda i: -curve[i])
    selected = sorted_by_darkness[:n_candidates]
    
    return [first_frame + int(i) for i in selected]


def choose_best_frame_with_geo(cine_obj, curve, n_candidates=None, threshold_percentile=None):
    """
    FULL OUTPUT MODE: Best-frame selection using darkness curve + geometry.
    
    Uses darkness curve to find candidate frames, then runs geometry
    analysis only on those candidates.
    
    Parameters
    ----------
    cine_obj : Cine
        The loaded cine object.
    curve : np.ndarray
        Darkness curve from analyze_cine_darkness().
    n_candidates : int, optional
        Maximum number of candidate frames to analyse geometrically.
    threshold_percentile : float, optional
        Only consider frames with darkness above this percentile.

    Returns
    -------
    best_frame_idx : int
    best_geo : dict
    """
    if n_candidates is None:
        n_candidates = N_CANDIDATES
    if threshold_percentile is None:
        threshold_percentile = DARKNESS_THRESHOLD_PERCENTILE
    
    first_frame, last_frame = cine_obj.range

    dmin = float(curve.min())
    dmax = float(curve.max())
    dspan = max(1e-6, dmax - dmin)

    candidates = _find_candidate_frames(
        curve, first_frame,
        n_candidates=n_candidates,
        threshold_percentile=threshold_percentile
    )

    best_frame = None
    best_score = None
    best_geo = None

    for idx in candidates:
        geo = analyze_frame_geometric(cine_obj, idx)
        
        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]

        if y_top is None or y_bottom is None or y_sphere is None:
            continue

        if y_bottom >= y_sphere:
            continue

        top_margin = float(y_top)
        bottom_gap = float(y_sphere - y_bottom)
        cent_err = abs(top_margin - bottom_gap)
        dark_norm = (float(curve[idx - first_frame]) - dmin) / dspan

        score = -cent_err + DARKNESS_WEIGHT * dark_norm

        if (best_frame is None) or (score > best_score):
            best_frame = idx
            best_score = score
            best_geo = geo

    # Fallback: expand search if no valid frame found
    if best_frame is None:
        threshold = np.percentile(curve, max(50, threshold_percentile - 20))
        all_above = np.where(curve >= threshold)[0]
        
        for rel_idx in all_above:
            idx = first_frame + int(rel_idx)
            if idx in candidates:
                continue
                
            geo = analyze_frame_geometric(cine_obj, idx)
            
            y_top = geo["y_top"]
            y_bottom = geo["y_bottom"]
            y_sphere = geo["y_bottom_sphere"]

            if y_top is None or y_bottom is None or y_sphere is None:
                continue
            if y_bottom >= y_sphere:
                continue

            top_margin = float(y_top)
            bottom_gap = float(y_sphere - y_bottom)
            cent_err = abs(top_margin - bottom_gap)
            dark_norm = (float(curve[idx - first_frame]) - dmin) / dspan
            score = -cent_err + DARKNESS_WEIGHT * dark_norm

            if (best_frame is None) or (score > best_score):
                best_frame = idx
                best_score = score
                best_geo = geo

    # Final fallback
    if best_frame is None:
        best_frame = first_frame + int(curve.argmax())
        best_geo = analyze_frame_geometric(cine_obj, best_frame)

    return best_frame, best_geo
