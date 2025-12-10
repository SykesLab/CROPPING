"""Darkness curve computation and best-frame selection.

Provides two modes:
    - FULL OUTPUT: Darkness curve + candidate-based geometry (for plots)
    - CROPS ONLY: Single-pass geometry scan (faster, no darkness curve)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config_modular import (
    DARKNESS_THRESHOLD_PERCENTILE,
    DARKNESS_WEIGHT,
    N_CANDIDATES,
)
from geom_analysis_modular import analyze_frame_geometric
from image_utils_modular import load_frame_gray


def get_dark_fraction(cine_obj: Any, idx: int) -> float:
    """Calculate fraction of dark pixels in a frame using Otsu thresholding.

    Args:
        cine_obj: Loaded cine object.
        idx: Frame index.

    Returns:
        Fraction of pixels classified as dark (0.0 to 1.0).
    """
    gray = load_frame_gray(cine_obj, idx)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float((mask == 0).mean())


def analyze_cine_darkness(cine_obj: Any) -> Dict[str, Any]:
    """Compute darkness curve for entire cine.

    Args:
        cine_obj: Loaded cine object.

    Returns:
        Dictionary containing:
            - first_frame: First frame index
            - last_frame: Last frame index
            - darkness_curve: numpy array of dark fractions
            - total_frames: Number of frames
    """
    first, last = cine_obj.range
    curve = [get_dark_fraction(cine_obj, i) for i in range(first, last + 1)]

    return {
        "first_frame": first,
        "last_frame": last,
        "darkness_curve": np.array(curve, dtype=np.float32),
        "total_frames": len(curve),
    }


def choose_best_frame_geometry_only(
    cine_obj: Any,
) -> Tuple[int, Dict[str, Any]]:
    """Find best frame using geometry-only scan (crops-only mode).

    Scans all frames and selects the best pre-collision frame based on
    droplet centring within the frame.

    Args:
        cine_obj: Loaded cine object.

    Returns:
        Tuple of (best_frame_index, geometry_dict).
    """
    first_frame, last_frame = cine_obj.range

    best_frame: Optional[int] = None
    best_score: Optional[float] = None
    best_geo: Optional[Dict[str, Any]] = None

    for idx in range(first_frame, last_frame + 1):
        geo = analyze_frame_geometric(cine_obj, idx)

        y_top = geo["y_top"]
        y_bottom = geo["y_bottom"]
        y_sphere = geo["y_bottom_sphere"]

        # Require complete geometry
        if y_top is None or y_bottom is None or y_sphere is None:
            continue

        # Pre-collision constraint: droplet entirely above sphere
        if y_bottom >= y_sphere:
            continue

        # Score: minimise centring error
        top_margin = float(y_top)
        bottom_gap = float(y_sphere - y_bottom)
        centring_error = abs(top_margin - bottom_gap)
        score = -centring_error

        if best_frame is None or score > best_score:
            best_frame = idx
            best_score = score
            best_geo = geo

    # Fallback to middle frame if no valid frame found
    if best_frame is None:
        mid_frame = (first_frame + last_frame) // 2
        best_frame = mid_frame
        best_geo = analyze_frame_geometric(cine_obj, mid_frame)

    return best_frame, best_geo


def _find_candidate_frames(
    curve: np.ndarray,
    first_frame: int,
    n_candidates: Optional[int] = None,
    threshold_percentile: Optional[float] = None,
) -> List[int]:
    """Find candidate frames with high darkness values.

    Args:
        curve: Darkness curve array.
        first_frame: First frame index (for offset calculation).
        n_candidates: Max candidates to return.
        threshold_percentile: Only consider frames above this percentile.

    Returns:
        List of absolute frame indices.
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

    # Sort by darkness (descending) and take top candidates
    sorted_by_darkness = sorted(above_thresh, key=lambda i: -curve[i])
    selected = sorted_by_darkness[:n_candidates]

    return [first_frame + int(i) for i in selected]


def choose_best_frame_with_geo(
    cine_obj: Any,
    curve: np.ndarray,
    n_candidates: Optional[int] = None,
    threshold_percentile: Optional[float] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Find best frame using darkness curve and geometry (full output mode).

    Uses darkness curve to identify candidate frames, then performs
    geometric analysis only on those candidates for efficiency.

    Args:
        cine_obj: Loaded cine object.
        curve: Darkness curve from analyze_cine_darkness().
        n_candidates: Max candidate frames to analyse.
        threshold_percentile: Darkness threshold percentile.

    Returns:
        Tuple of (best_frame_index, geometry_dict).
    """
    if n_candidates is None:
        n_candidates = N_CANDIDATES
    if threshold_percentile is None:
        threshold_percentile = DARKNESS_THRESHOLD_PERCENTILE

    first_frame, last_frame = cine_obj.range

    # Normalisation for darkness scoring
    dmin = float(curve.min())
    dmax = float(curve.max())
    dspan = max(1e-6, dmax - dmin)

    candidates = _find_candidate_frames(
        curve, first_frame,
        n_candidates=n_candidates,
        threshold_percentile=threshold_percentile,
    )
    candidates_set = set(candidates)  # For O(1) lookup in fallback

    best_frame: Optional[int] = None
    best_score: Optional[float] = None
    best_geo: Optional[Dict[str, Any]] = None

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
        centring_error = abs(top_margin - bottom_gap)
        dark_norm = (float(curve[idx - first_frame]) - dmin) / dspan

        score = -centring_error + DARKNESS_WEIGHT * dark_norm

        if best_frame is None or score > best_score:
            best_frame = idx
            best_score = score
            best_geo = geo

    # Fallback: expand search if no valid frame found
    if best_frame is None:
        threshold = np.percentile(
            curve, max(50, threshold_percentile - 20)
        )
        all_above = np.where(curve >= threshold)[0]

        for rel_idx in all_above:
            idx = first_frame + int(rel_idx)
            if idx in candidates_set:
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
            centring_error = abs(top_margin - bottom_gap)
            dark_norm = (float(curve[idx - first_frame]) - dmin) / dspan
            score = -centring_error + DARKNESS_WEIGHT * dark_norm

            if best_frame is None or score > best_score:
                best_frame = idx
                best_score = score
                best_geo = geo

    # Final fallback
    if best_frame is None:
        best_frame = first_frame + int(curve.argmax())
        best_geo = analyze_frame_geometric(cine_obj, best_frame)

    return best_frame, best_geo
