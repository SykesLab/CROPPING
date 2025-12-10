"""Droplet cropping with sphere exclusion.

Provides functions for extracting fixed-size crops centred on droplets
while ensuring the sphere remains outside the crop region.
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
    """Extract a fixed-size crop centred on the droplet.

    Guarantees:
        - Fixed target_w × target_h output size
        - No warping (pure crop)
        - Sphere excluded from crop if y_sphere is provided

    Logic:
        1. Start centred on droplet (cx, cy).
        2. If bottom would reveal sphere, shift crop up.
        3. Clamp to image boundaries while preserving size.
        4. Re-apply sphere guard after clamping.

    Args:
        frame: Grayscale image as 2D numpy array.
        y_top: Droplet top row.
        y_bottom: Droplet bottom row.
        cx: Droplet centre x coordinate.
        target_w: Desired crop width in pixels.
        target_h: Desired crop height in pixels.
        y_sphere: Sphere top row (optional).
        safety: Minimum pixels between crop bottom and sphere.

    Returns:
        Cropped image of size (target_h, target_w).
    """
    height, width = frame.shape

    # Calculate crop centre
    cy = 0.5 * (y_top + y_bottom)
    half_h = target_h // 2
    half_w = target_w // 2

    # Initial centred crop
    x0 = int(cx - half_w)
    x1 = x0 + target_w
    y0 = int(cy - half_h)
    y1 = y0 + target_h

    # Sphere guard (first pass)
    if y_sphere is not None:
        max_y1 = int(y_sphere - safety)
        if y1 > max_y1:
            shift = y1 - max_y1
            y0 -= shift
            y1 -= shift

    # Clamp vertically, preserving height
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > height:
        y0 -= (y1 - height)
        y1 = height

    # Clamp horizontally, preserving width
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > width:
        x0 -= (x1 - width)
        x1 = width

    # Sphere guard (second pass after clamping)
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
