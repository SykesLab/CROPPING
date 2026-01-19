"""
Droplet cropping with sphere exclusion.

Extracts fixed-size crops centred on the droplet while keeping the
sphere (injection needle tip) out of frame.
"""

import numpy as np


def crop_droplet_with_sphere_guard(
    frame: np.ndarray,
    y_top: int,
    y_bottom: int,
    cx: float,
    target_w: int,
    target_h: int,
    y_sphere: int = None,
    safety: int = 3,
) -> np.ndarray:
    """
    Extract a fixed-size crop centred on the droplet.

    The crop is shifted upward if it would otherwise include the sphere.
    After all adjustments, the output is always target_w × target_h.
    """
    height, width = frame.shape

    cy = 0.5 * (y_top + y_bottom)
    half_h = target_h // 2
    half_w = target_w // 2

    x0 = int(cx - half_w)
    x1 = x0 + target_w
    y0 = int(cy - half_h)
    y1 = y0 + target_h

    # Shift up if crop would include the sphere
    if y_sphere is not None:
        max_y1 = int(y_sphere - safety)
        if y1 > max_y1:
            shift = y1 - max_y1
            y0 -= shift
            y1 -= shift

    # Clamp to image bounds while preserving size
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > height:
        y0 -= (y1 - height)
        y1 = height

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > width:
        x0 -= (x1 - width)
        x1 = width

    # Re-check sphere guard after clamping
    if y_sphere is not None:
        max_y1 = int(y_sphere - safety)
        if y1 > max_y1:
            shift = y1 - max_y1
            y0 -= shift
            y1 -= shift
            if y0 < 0:
                y1 -= y0
                y0 = 0

    return frame[int(y0):int(y1), int(x0):int(x1)]
