"""
Crop size calibration based on observed droplet geometry.

Computes the optimal crop size from measured diameters and sphere gaps,
using percentile-based estimation for outlier robustness.
"""

from typing import Any, Dict, List

import numpy as np

from config_modular import CALIBRATION_PERCENTILE, MAX_CNN_SIZE, MIN_CNN_SIZE


def maybe_add_calibration_sample(
    diams: List[float],
    gaps: List[float],
    geo: Dict[str, Any],
) -> None:
    """Add a geometry sample to the calibration lists if valid (droplet above sphere)."""
    y_top = geo.get("y_top")
    y_bottom = geo.get("y_bottom")
    y_sphere = geo.get("y_bottom_sphere")

    if (
        y_top is not None
        and y_bottom is not None
        and y_sphere is not None
        and y_bottom < y_sphere
    ):
        diameter = float(y_bottom - y_top)
        gap = float(y_sphere - y_bottom)
        diams.append(diameter)
        gaps.append(gap)


def compute_crop_size(
    diams: List[float],
    gaps: List[float],
    safety_pixels: float,
    fallback: int = 128,
) -> int:
    """
    Compute optimal crop size from calibration samples.

    For each sample, the maximum safe crop height is:
        allowed_h = diameter + 2 * max(0, gap - safety_pixels)

    Uses a low percentile (from config) to be conservative and avoid
    crops that would include the sphere.
    """
    if not diams or not gaps or len(diams) != len(gaps):
        return fallback

    allowed_heights: List[float] = []
    for d, g in zip(diams, gaps):
        if g <= 0:
            continue
        allowed = float(d) + 2.0 * max(0.0, float(g) - float(safety_pixels))
        if allowed > 0:
            allowed_heights.append(allowed)

    if not allowed_heights:
        return fallback

    crop_h = int(np.percentile(allowed_heights, CALIBRATION_PERCENTILE))
    crop_h = min(crop_h, MAX_CNN_SIZE)
    return max(crop_h, MIN_CNN_SIZE) if crop_h >= MIN_CNN_SIZE else crop_h
