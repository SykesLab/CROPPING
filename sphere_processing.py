"""
Sphere detection and processing utilities for calibration images.

Functions for detecting spheres, mirroring to remove stage artifacts,
blackening interiors to remove hot spots, and cropping to square.
"""

import cv2
import numpy as np


def find_sphere_center(
    img: np.ndarray,
    upper_only: bool = True,
    upper_fraction: float = 0.60,
) -> tuple | None:
    """
    Detect sphere edge and fit circle.

    Args:
        img: Grayscale image (2D numpy array)
        upper_only: If True, use only the upper portion of the contour
                    for circle fitting (avoids stage reflection)
        upper_fraction: Fraction of contour height to use when upper_only=True

    Returns:
        (cx, cy, radius) or None if detection fails
    """
    if img.ndim != 2:
        return None

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return None

    pts = cnt.reshape(-1, 2)

    if upper_only:
        y_min = pts[:, 1].min()
        y_max = pts[:, 1].max()
        y_cut = y_min + upper_fraction * (y_max - y_min)
        upper_pts = pts[pts[:, 1] <= y_cut]
        if len(upper_pts) >= 20:
            pts = upper_pts

    (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))

    h, w = img.shape[:2]
    if r_i <= 0 or r_i > max(h, w):
        return None
    if not (0 <= cx_i < w and 0 <= cy_i < h):
        return None

    return cx_i, cy_i, r_i


def mirror_from_center(img: np.ndarray, center_y: int) -> np.ndarray:
    """Mirror about y=center_y using top half -> flipped bottom half."""
    h, _ = img.shape[:2]
    center_y = int(np.clip(center_y, 0, h - 1))
    top_half = img[:center_y + 1, :]
    bottom_half = cv2.flip(top_half, 0)
    return np.vstack([top_half, bottom_half])


def blacken_sphere_interior(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    radius_fraction: float = 0.98,
    edge_margin_px: int = 2,
) -> np.ndarray:
    """
    Black out sphere interior, stopping slightly short of the fitted edge.

    Args:
        img: Input image
        center_x, center_y: Sphere center coordinates
        radius: Sphere radius in pixels
        radius_fraction: Fraction of radius to black out (default 0.98)
        edge_margin_px: Additional margin in pixels from edge (default 2)
    """
    if radius <= 0:
        return img

    h, w = img.shape[:2]
    r_eff = int(min(radius * radius_fraction, radius - edge_margin_px))
    if r_eff <= 0:
        return img

    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    img[dist_sq < (r_eff ** 2)] = 0
    return img


def crop_to_square(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    padding: float = 1.2,
) -> np.ndarray:
    """
    Crop to a square centered on (center_x, center_y).

    Args:
        img: Input image
        center_x, center_y: Center coordinates
        radius: Sphere radius
        padding: Multiplier for half-size (1.2 = 20% padding beyond radius)
    """
    h, w = img.shape[:2]
    half_size = int(radius * padding)

    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)

    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    size = min(ch, cw)
    if size <= 0:
        return crop

    cx_local = min(cw - 1, max(0, center_x - x1))
    cy_local = min(ch - 1, max(0, center_y - y1))

    sx1 = cx_local - size // 2
    sy1 = cy_local - size // 2
    sx2 = sx1 + size
    sy2 = sy1 + size

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > cw:
        sx1 -= (sx2 - cw)
        sx2 = cw
    if sy2 > ch:
        sy1 -= (sy2 - ch)
        sy2 = ch

    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(cw, sx2), min(ch, sy2)

    return crop[sy1:sy2, sx1:sx2]


def process_sphere_image(
    img: np.ndarray,
    upper_only: bool = True,
    output_size: int | None = None,
) -> tuple[np.ndarray | None, tuple | None]:
    """
    Full sphere processing pipeline: detect → mirror → blacken → crop → resize.

    Args:
        img: Grayscale input image
        upper_only: Use upper contour only for circle fitting
        output_size: If set, resize the cropped image to this square size

    Returns:
        (processed_image, (cx, cy, radius)) or (None, None) if detection fails
    """
    result = find_sphere_center(img, upper_only=upper_only)
    if result is None:
        return None, None

    cx, cy, radius = result

    processed = mirror_from_center(img.copy(), cy)
    processed = blacken_sphere_interior(processed, cx, cy, radius)
    processed = crop_to_square(processed, cx, cy, radius, padding=1.2)

    if output_size is not None and output_size > 0:
        h, w = processed.shape[:2]
        if h != output_size or w != output_size:
            processed = cv2.resize(
                processed, (output_size, output_size),
                interpolation=cv2.INTER_AREA
            )

    return processed, (cx, cy, radius)
