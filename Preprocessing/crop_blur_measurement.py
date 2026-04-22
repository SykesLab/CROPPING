"""
Erf-based Gaussian blur measurement for sharp crop images.

Directly mirrors the logic in calibration/blur_measurement.py:
  - Normalises image to [0, 1]
  - Detects sphere centre / radius via Otsu bounding-box (closest-to-centre
    connected component so small camera-m droplets are found correctly)
  - Fits erf per radial ray independently, accepts per-ray with R² > 0.5 and
    contrast_ratio > 0.2
  - Returns median sigma across accepted rays

This module is self-contained (no import from the calibration directory) so
it can run inside the preprocessing pipeline environment.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

# ERF fitting constants
SIGMA_INIT_GUESSES = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
MIN_CONTRAST = 0.05
MIN_EDGE_MARGIN_PX = 20


# ---------------------------------------------------------------------------
# erf edge model (identical to calibration/blur_measurement.py)
# ---------------------------------------------------------------------------

def _erf_edge(r: np.ndarray, I_bg: float, I_sphere: float, r_edge: float, sigma: float) -> np.ndarray:
    sigma_safe = max(sigma, 0.001)
    return (I_bg + I_sphere) / 2 + (I_bg - I_sphere) / 2 * erf((r - r_edge) / (sigma_safe * np.sqrt(2)))


# ---------------------------------------------------------------------------
# Sphere detection
# ---------------------------------------------------------------------------

def _detect_circle(img_u8: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Detect sphere centre and radius from a uint8 image using HoughCircles.

    HoughCircles works on local edges (Canny internally) so it is unaffected
    by large-scale illumination gradients.  If multiple circles are found, the
    one whose centre is closest to the image centre is returned.

    Returns (cx, cy, radius) as ints, or None.
    """
    h, w = img_u8.shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0

    blurred = cv2.GaussianBlur(img_u8, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 2,   # expect only one sphere per crop
        param1=50,                 # Canny high threshold
        param2=20,                 # accumulator threshold — low to catch faint edges
        minRadius=10,
        maxRadius=min(h, w) // 2,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Pick circle whose centre is nearest the image centre
    best = min(circles, key=lambda c: (c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2)
    cx, cy, radius = int(best[0]), int(best[1]), int(best[2])

    if radius < 5:
        return None

    return cx, cy, radius


# ---------------------------------------------------------------------------
# Per-ray erf fit (mirrors calibration/blur_measurement.py _fit_erf_multi_start)
# ---------------------------------------------------------------------------

def _fit_ray(r_valid: np.ndarray, intensities: np.ndarray,
             radius: float, I_bg_init: float, I_sphere_init: float,
             edge_margin: float) -> Tuple[Optional[np.ndarray], float]:
    """Fit erf to one radial ray. Returns (popt, r_squared) or (None, 0)."""
    r_min, r_max = r_valid.min(), r_valid.max()
    sigma_inits = SIGMA_INIT_GUESSES
    best_popt, best_r2, best_res = None, -np.inf, np.inf

    for s0 in sigma_inits:
        try:
            popt, _ = curve_fit(
                _erf_edge, r_valid, intensities,
                p0=[I_bg_init, I_sphere_init, radius, s0],
                bounds=([0, 0, r_min, 0.01], [1, 1, r_max, 500]),
                maxfev=2000,
            )
            fitted = _erf_edge(r_valid, *popt)
            res = np.sum((intensities - fitted) ** 2)
            ss_tot = np.sum((intensities - intensities.mean()) ** 2)
            r2 = 1.0 - res / ss_tot if ss_tot > 0 else 0.0
            if res < best_res:
                best_popt, best_r2, best_res = popt, r2, res
        except (RuntimeError, ValueError):
            continue

    return best_popt, best_r2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_erf_blur(
    img: np.ndarray,
    num_rays: int = 36,
    center_xy: Optional[tuple] = None,
    radius_px: Optional[float] = None,
) -> Optional[float]:
    """
    Measure Gaussian blur sigma of a crop image using per-ray erf fitting.

    Mirrors calibration/blur_measurement.py measure_blur_erf():
      - normalises image to [0, 1]
      - fits erf independently on each radial ray
      - returns median sigma of accepted rays (R² > 0.5, contrast ratio > 0.2)

    Args:
        img: Grayscale image (uint8 / uint16 / float) or BGR.
        num_rays: Number of radial directions to sample.
        center_xy: Optional (cx, cy) to skip auto-detection.
        radius_px: Optional radius in pixels to skip auto-detection.

    Returns:
        Gaussian sigma in pixels, or None if measurement fails.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalise to [0, 1]  (same as calibration code)
    img_f = img.astype(np.float32)
    if img_f.max() > 1:
        img_f = img_f / 255.0

    img_u8 = (img_f * 255).astype(np.uint8)
    h, w = img_f.shape[:2]

    # Detect or use provided centre / radius
    if center_xy is not None and radius_px is not None:
        cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
        radius = int(round(radius_px))
    else:
        result = _detect_circle(img_u8)
        if result is None:
            return None
        cx, cy, radius = result

    # Global contrast estimate (same as calibration)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    inner_mask = dist < radius * 0.7
    outer_mask = (dist > radius * 1.3) & (dist < radius * 1.8)

    if inner_mask.any() and outer_mask.any():
        I_sphere_est = float(np.median(img_f[inner_mask]))
        I_bg_est = float(np.median(img_f[outer_mask]))
        contrast = abs(I_bg_est - I_sphere_est)
    else:
        I_sphere_est, I_bg_est, contrast = 0.0, 1.0, 1.0

    if contrast < MIN_CONTRAST:
        return None

    # Sharp crops have small sigma — MIN_EDGE_MARGIN_PX is plenty and keeps
    # runtime fast. (Calibration uses 80 px for heavily-defocused z-stacks.)
    edge_margin = max(MIN_EDGE_MARGIN_PX, int(radius * 0.3))

    sigmas = []

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Image-boundary-limited max radius for this ray
        max_r_x = (w - 2 - cx) / cos_a if cos_a > 0.01 else ((1 - cx) / cos_a
                                                             if cos_a < -0.01 else np.inf)
        max_r_y = (h - 2 - cy) / sin_a if sin_a > 0.01 else ((1 - cy) / sin_a
                                                             if sin_a < -0.01 else np.inf)
        max_r = min(max_r_x, max_r_y)

        start_r = max(0, radius - edge_margin)
        end_r = min(radius + edge_margin, max_r)
        if end_r <= start_r:
            continue

        r_values = np.arange(start_r, end_r, 0.5)   # sub-pixel
        if len(r_values) < 20:
            continue

        x_coords = cx + r_values * cos_a
        y_coords = cy + r_values * sin_a
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1, y1 = x0 + 1, y0 + 1

        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        if valid.sum() < 20:
            continue

        r_valid = r_values[valid]
        xf = x_coords[valid] - x0[valid]
        yf = y_coords[valid] - y0[valid]
        x0v, y0v, x1v, y1v = x0[valid], y0[valid], x1[valid], y1[valid]

        # Bilinear interpolation
        intensities = (
            img_f[y0v, x0v] * (1 - xf) * (1 - yf) +
            img_f[y0v, x1v] * xf * (1 - yf) +
            img_f[y1v, x0v] * (1 - xf) * yf +
            img_f[y1v, x1v] * xf * yf
        )

        popt, r2 = _fit_ray(r_valid, intensities, radius, I_bg_est, I_sphere_est, edge_margin)
        if popt is None:
            continue

        I_bg_fit, I_sphere_fit, r_edge, sigma = popt
        fit_contrast = abs(I_bg_fit - I_sphere_fit)
        contrast_ratio = fit_contrast / contrast if contrast else 1.0
        edge_offset = abs(r_edge - radius)

        if (r2 > 0.5 and sigma > 0.01 and contrast_ratio > 0.2 and edge_offset < edge_margin):
            sigmas.append(sigma)   # sigma is in pixel units (r_values are pixels)

    if not sigmas:
        return None

    return float(np.median(sigmas))
