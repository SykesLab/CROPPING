"""Crop size calibration based on droplet geometry.

Computes optimal crop size using observed droplet diameters and
sphere gaps, with outlier-robust percentile-based estimation.
"""

from typing import Any, Dict, List

import numpy as np

from config_modular import (
    CALIBRATION_PERCENTILE,
    MAX_CNN_SIZE,
    MIN_CNN_SIZE,
)


def maybe_add_calibration_sample(
    diams: List[float],
    gaps: List[float],
    geo: Dict[str, Any],
) -> None:
    """Add geometry sample to calibration lists if valid.

    A sample is valid if the droplet is entirely above the sphere.

    Args:
        diams: List to append diameter to (modified in place).
        gaps: List to append gap to (modified in place).
        geo: Geometry dict from analyze_frame_geometric.
    """
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
    """Compute optimal crop size from calibration samples.

    For each (diameter, gap) pair, the maximum safe crop height is:
        allowed_h = diameter + 2 * max(0, gap - safety_pixels)

    Uses percentile-based estimation for outlier robustness.

    Args:
        diams: List of droplet diameters.
        gaps: List of sphere gaps.
        safety_pixels: Minimum gap to maintain between crop and sphere.
        fallback: Default size if calibration fails.

    Returns:
        Computed crop size in pixels, bounded by MIN/MAX_CNN_SIZE.
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

    # Use percentile for outlier robustness
    crop_h = int(np.percentile(allowed_heights, CALIBRATION_PERCENTILE))

    # Apply bounds
    crop_h = min(crop_h, MAX_CNN_SIZE)

    # Prefer MIN_CNN_SIZE but don't violate geometry constraints
    return max(crop_h, MIN_CNN_SIZE) if crop_h >= MIN_CNN_SIZE else crop_h
