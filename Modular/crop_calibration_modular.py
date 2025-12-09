# crop_calibration_modular.py
#
# Crop size calibration based on observed droplet geometry.
# Uses percentile-based approach for outlier robustness.

import numpy as np
from typing import List
from config_modular import MIN_CNN_SIZE, MAX_CNN_SIZE, CALIBRATION_PERCENTILE


def maybe_add_calibration_sample(diams: List[float], gaps: List[float], geo: dict):
    """
    If geometry is valid (droplet above sphere), append diameter + gap to lists.

    NOTE: diams[i] and gaps[i] are treated as a *pair* belonging to the same frame.
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
    """
    Compute a CNN crop size based on observed droplet diameters and sphere gaps.

    For each (diameter_i, gap_i) pair, the *largest* crop height that still keeps the
    sphere out while centering the droplet is:

        allowed_h_i = diameter_i + 2 * max(0, gap_i - safety_pixels)

    To use a single crop size for all droplets, we use a percentile-based approach
    (rather than strict min) to be robust to outliers:

        crop_h = percentile(allowed_heights, CALIBRATION_PERCENTILE)

    This guarantees:
        - droplet can be centred vertically for most samples
        - sphere stays below the crop by at least `safety_pixels`
        - robust to occasional outlier measurements
    """
    if not diams or not gaps or len(diams) != len(gaps):
        return fallback

    allowed_heights = []
    for d, g in zip(diams, gaps):
        # If gap is non-positive, this sample isn't useful for calibration
        if g <= 0:
            continue
        # Max crop height that still excludes sphere for this droplet
        allowed = float(d) + 2.0 * max(0.0, float(g) - float(safety_pixels))
        if allowed > 0:
            allowed_heights.append(allowed)

    if not allowed_heights:
        return fallback

    # Use percentile for outlier robustness (default 5th percentile)
    crop_h = int(np.percentile(allowed_heights, CALIBRATION_PERCENTILE))

    # Upper-bound by MAX_CNN_SIZE (we never want to exceed this)
    crop_h = min(crop_h, int(MAX_CNN_SIZE))

    # We *prefer* at least MIN_CNN_SIZE, but we won't violate geometry to enforce it.
    # Only raise to MIN_CNN_SIZE if that does not exceed the computed value.
    if crop_h >= MIN_CNN_SIZE:
        return crop_h
    else:
        # Geometry says we can't safely reach MIN_CNN_SIZE; honour geometry.
        return crop_h
