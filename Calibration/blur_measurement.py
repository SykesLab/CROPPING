"""
Blur Measurement Methods for Calibration

This module provides different methods to measure blur (sigma) from calibration images.
Each method quantifies how blurry a sphere or edge appears in an image.

Methods:
1. Erf edge fitting (recommended) - fits error function to edge profile (physically exact for Gaussian blur)
2. Gradient-based (Sobel) - measures average gradient magnitude at edges
3. Laplacian variance - simple sharpness metric (inverse relationship to blur)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import sobel
from scipy.optimize import curve_fit
from scipy.special import erf


@dataclass
class BlurMeasurement:
    """Result of a blur measurement."""
    sigma: float  # Blur sigma in pixels
    method: str  # Which method was used
    confidence: float  # 0-1 confidence score
    details: Dict  # Method-specific details


def erf_edge(r: np.ndarray, I_bg: float, I_sphere: float, r_edge: float, sigma: float) -> np.ndarray:
    """
    Error function edge profile — physical model for Gaussian-blurred step edge.

    I(r) = (I_bg + I_sphere)/2 + (I_bg - I_sphere)/2 × erf((r - r_edge) / (σ√2))

    A sharp step edge convolved with a Gaussian of std σ produces exactly this
    profile, so the fitted σ is the true Gaussian blur sigma.

    Args:
        r: Radial positions (pixels)
        I_bg: Background intensity
        I_sphere: Sphere intensity
        r_edge: Edge position
        sigma: Gaussian blur sigma (what we want to measure)

    Returns:
        Intensity values
    """
    sigma_safe = max(sigma, 0.001)
    return (I_bg + I_sphere) / 2 + (I_bg - I_sphere) / 2 * erf((r - r_edge) / (sigma_safe * np.sqrt(2)))


def _fit_erf_multi_start(
    r_valid: np.ndarray,
    intensities: np.ndarray,
    radius: float,
    I_bg_init: float,
    I_sphere_init: float,
    edge_margin: float = 30.0
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Fit erf edge profile with multiple starting points to avoid local minima.

    Args:
        r_valid: Radial positions of samples
        intensities: Intensity values at those positions
        radius: Expected edge radius
        I_bg_init: Initial guess for background intensity
        I_sphere_init: Initial guess for sphere intensity
        edge_margin: How far from radius the edge can be

    Returns:
        (best_popt, best_r_squared, best_residual) or (None, 0, inf) if all fail
    """
    # Try multiple initial sigma values to avoid local minima
    # Include small values for sharp edges and large values for heavily defocused images
    sigma_inits = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    best_popt = None
    best_r_squared = -np.inf
    best_residual = np.inf

    # Bounds: edge should be within the sampled region
    r_min = r_valid.min()
    r_max = r_valid.max()

    for sigma_init in sigma_inits:
        try:
            popt, _ = curve_fit(
                erf_edge, r_valid, intensities,
                p0=[I_bg_init, I_sphere_init, radius, sigma_init],
                bounds=(
                    [0, 0, r_min, 0.01],  # Edge must be within sampled region
                    [1, 1, r_max, 500]
                ),
                maxfev=2000
            )

            # Calculate fit quality
            fitted = erf_edge(r_valid, *popt)
            residual = np.sum((intensities - fitted) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r_squared = 1 - (residual / ss_tot) if ss_tot > 0 else 0

            # Keep best fit (lowest residual)
            if residual < best_residual:
                best_popt = popt
                best_r_squared = r_squared
                best_residual = residual

        except (RuntimeError, ValueError):
            continue

    return best_popt, best_r_squared, best_residual


def measure_blur_erf(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
    num_rays: int = 36,
    verbose: bool = False
) -> BlurMeasurement:
    """
    Measure blur by fitting erf (error function) to edge profiles.

    This method extracts radial intensity profiles from the sphere center
    outward, fits an erf edge model to each profile, and takes the median
    sigma. The erf is the physically exact result of convolving a step edge
    with a Gaussian, so the fitted sigma is the true Gaussian blur sigma.

    Args:
        image: Grayscale image (0-255 or 0-1)
        center: (x, y) center of sphere. If None, auto-detects.
        radius: Approximate radius of sphere. If None, auto-detects.
        num_rays: Number of radial profiles to extract
        verbose: If True, print diagnostic information

    Returns:
        BlurMeasurement with sigma and details
    """
    # Normalize image to 0-1
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0

    # Auto-detect center and radius if not provided
    if center is None or radius is None:
        center, radius = detect_sphere(image)
        if center is None:
            return BlurMeasurement(
                sigma=0.0, method='erf', confidence=0.0,
                details={'error': 'Could not detect sphere'}
            )

    cx, cy = center
    h, w = image.shape[:2]

    if verbose:
        print(f"  Image size: {w} x {h}")
        print(f"  Detected: center=({cx}, {cy}), radius={radius}")

    # Check for sufficient contrast using regions well inside/outside the sphere
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    inner_mask = dist < radius * 0.7
    outer_mask = (dist > radius * 1.3) & (dist < radius * 1.8)

    if np.any(inner_mask) and np.any(outer_mask):
        I_sphere_est = np.median(image[inner_mask])
        I_bg_est = np.median(image[outer_mask])
        contrast = abs(I_bg_est - I_sphere_est)
        if verbose:
            print(f"  Contrast: {contrast:.3f} (I_sphere={I_sphere_est:.3f}, I_bg={I_bg_est:.3f})")

        if contrast < 0.05:
            return BlurMeasurement(
                sigma=0.0, method='erf', confidence=0.0,
                details={'error': f'Insufficient contrast: {contrast:.3f}'}
            )
    else:
        I_sphere_est = 0.0
        I_bg_est = 1.0
        contrast = 1.0

    # Extract radial profiles - FOCUS on the edge region
    # For sharp images, sampling 0.5r to 1.5r gives mostly flat data
    # Instead, sample tightly around the expected edge with sub-pixel resolution
    sigmas = []
    r_squareds = []
    fit_details = []

    # Determine edge sampling window based on expected blur
    # Use 30% of radius to capture wide blur transitions (large defocus → large σ)
    edge_margin = max(80, int(radius * 0.3))  # Sample ± this many pixels from edge

    if verbose:
        print(f"  Edge margin: {edge_margin} px, sampling r=[{radius-edge_margin}, {radius+edge_margin}]")

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Calculate max radius for THIS ray direction based on image bounds
        # For each direction, find how far we can go before hitting image edge
        if cos_a > 0.01:
            max_r_x = (w - 2 - cx) / cos_a  # How far right before hitting w
        elif cos_a < -0.01:
            max_r_x = (1 - cx) / cos_a  # How far left before hitting 0
        else:
            max_r_x = float('inf')

        if sin_a > 0.01:
            max_r_y = (h - 2 - cy) / sin_a  # How far down before hitting h
        elif sin_a < -0.01:
            max_r_y = (1 - cy) / sin_a  # How far up before hitting 0
        else:
            max_r_y = float('inf')

        max_r_for_ray = min(max_r_x, max_r_y)

        # Sample with sub-pixel resolution around the expected edge
        start_r = max(0, radius - edge_margin)
        end_r = min(radius + edge_margin, max_r_for_ray)

        if end_r <= start_r:
            continue

        r_values = np.arange(start_r, end_r, 0.5)  # Sub-pixel sampling
        if len(r_values) < 20:
            continue

        # Get sub-pixel coordinates
        x_coords = cx + r_values * cos_a
        y_coords = cy + r_values * sin_a

        # Bilinear interpolation for sub-pixel sampling
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # Ensure within bounds (should mostly pass now due to max_r calculation)
        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        if np.sum(valid) < 20:
            continue

        r_valid = r_values[valid]
        x0_v, y0_v = x0[valid], y0[valid]
        x1_v, y1_v = x1[valid], y1[valid]
        xf = (x_coords[valid] - x0_v)  # Fractional part
        yf = (y_coords[valid] - y0_v)

        # Bilinear interpolation
        intensities = (
            image[y0_v, x0_v] * (1 - xf) * (1 - yf) +
            image[y0_v, x1_v] * xf * (1 - yf) +
            image[y1_v, x0_v] * (1 - xf) * yf +
            image[y1_v, x1_v] * xf * yf
        )

        # Use known contrast values for initial guesses
        I_bg_init = I_bg_est
        I_sphere_init = I_sphere_est

        # Use multi-start optimization
        popt, r_squared, residual = _fit_erf_multi_start(
            r_valid, intensities, radius, I_bg_init, I_sphere_init, edge_margin
        )

        if popt is None:
            continue

        I_bg, I_sphere, r_edge, sigma = popt

        # Check if fit makes physical sense
        fit_contrast = abs(I_bg - I_sphere)

        # Calculate acceptance metrics
        contrast_ratio = fit_contrast / contrast if contrast else 1.0
        edge_offset = abs(r_edge - radius)

        # Accept fit if:
        # 1. R² is reasonable
        # 2. sigma is positive
        # 3. Fitted contrast is close to actual contrast (within 50%)
        # 4. Edge position is reasonable (within edge_margin of expected radius)
        min_r2 = 0.5  # Lowered - sharp edges may have lower R² due to pixelation
        fit_accepted = (
            r_squared > min_r2 and
            sigma > 0.01 and
            contrast_ratio > 0.2 and  # Fit contrast at least 20% of actual
            edge_offset < edge_margin  # Edge found within expected region
        )

        if fit_accepted:
            sigmas.append(sigma)
            r_squareds.append(r_squared)
            fit_details.append({
                'angle': np.degrees(angle),
                'sigma': sigma,
                'r_squared': r_squared,
                'contrast': fit_contrast,
                'r_edge': r_edge
            })

    if len(sigmas) == 0:
        return BlurMeasurement(
            sigma=0.0, method='erf', confidence=0.0,
            details={'error': f'Could not fit erf to any of {num_rays} rays'}
        )

    # Use median to be robust to outliers
    sigma_final = np.median(sigmas)
    confidence = np.mean(r_squareds)

    if verbose:
        print(f"  Rays: {len(sigmas)}/{num_rays} accepted, σ={sigma_final:.3f} px")

    return BlurMeasurement(
        sigma=sigma_final,
        method='erf',
        confidence=confidence,
        details={
            'num_rays_used': len(sigmas),
            'sigma_std': np.std(sigmas),
            'mean_r_squared': np.mean(r_squareds),
            'all_sigmas': sigmas,
            'center': (cx, cy),
            'radius': radius,
            'contrast': contrast
        }
    )


def measure_blur_gradient(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None
) -> BlurMeasurement:
    """
    Measure blur using gradient-based method.

    Sharper edges have higher gradients. This method computes the average
    gradient magnitude in the edge region and converts it to an equivalent
    sigma value.

    Args:
        image: Grayscale image
        center: (x, y) center of sphere. If None, auto-detects.
        radius: Approximate radius of sphere. If None, auto-detects.

    Returns:
        BlurMeasurement with sigma and details
    """
    # Normalize image
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0

    # Auto-detect if needed
    if center is None or radius is None:
        center, radius = detect_sphere(image)
        if center is None:
            return BlurMeasurement(
                sigma=0.0, method='gradient', confidence=0.0,
                details={'error': 'Could not detect sphere'}
            )

    cx, cy = center
    h, w = image.shape[:2]

    # Compute gradient magnitude
    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)
    gradient_mag = np.sqrt(gx**2 + gy**2)

    # Create annular mask around sphere edge
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

    edge_width = max(radius * 0.2, 5)
    edge_mask = (dist_from_center >= radius - edge_width) & (dist_from_center <= radius + edge_width)

    if not np.any(edge_mask):
        return BlurMeasurement(
            sigma=0.0, method='gradient', confidence=0.0,
            details={'error': 'Edge mask is empty'}
        )

    # Average gradient in edge region
    mean_gradient = np.mean(gradient_mag[edge_mask])
    max_gradient = np.max(gradient_mag[edge_mask])

    # Convert gradient to sigma estimate
    # For a step edge blurred with sigma, the max gradient is approximately:
    # G_max ≈ (I_high - I_low) / (sigma * sqrt(2*pi))
    # So sigma ≈ contrast / (G_max * sqrt(2*pi))

    # Estimate contrast from image
    inner_mask = dist_from_center < radius * 0.7
    outer_mask = (dist_from_center > radius * 1.3) & (dist_from_center < radius * 2)

    if np.any(inner_mask) and np.any(outer_mask):
        I_sphere = np.median(image[inner_mask])
        I_bg = np.median(image[outer_mask])
        contrast = abs(I_bg - I_sphere)
    else:
        contrast = 0.5  # Default assumption

    if max_gradient > 0.01:
        sigma = contrast / (max_gradient * np.sqrt(2 * np.pi))
        sigma = np.clip(sigma, 0.5, 50)
        confidence = min(mean_gradient / 0.1, 1.0)  # Higher gradient = more confident
    else:
        sigma = 50  # Very blurry
        confidence = 0.3

    return BlurMeasurement(
        sigma=sigma,
        method='gradient',
        confidence=confidence,
        details={
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'estimated_contrast': contrast
        }
    )


def measure_blur_laplacian(image: np.ndarray) -> BlurMeasurement:
    """
    Measure blur using Laplacian variance.

    This is the simplest method - just computes the variance of the Laplacian.
    Higher variance = sharper image. This gives a relative sharpness metric
    rather than a direct sigma measurement.

    Args:
        image: Grayscale image

    Returns:
        BlurMeasurement with equivalent sigma
    """
    # Normalize image
    if image.max() > 1:
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Compute Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()

    # Convert variance to approximate sigma
    # This is an empirical relationship - sharper images have higher variance
    # sigma ≈ k / sqrt(variance) for some constant k
    # Calibrate k based on typical values
    k = 100  # Empirical constant

    if variance > 0:
        sigma = k / np.sqrt(variance)
        sigma = np.clip(sigma, 0.5, 50)
        confidence = min(variance / 500, 1.0)
    else:
        sigma = 50
        confidence = 0.1

    return BlurMeasurement(
        sigma=sigma,
        method='laplacian',
        confidence=confidence,
        details={
            'laplacian_variance': variance
        }
    )


def detect_sphere(image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    Auto-detect sphere center and radius in image.

    Uses get_sphere_mask() for thresholding, then finds center as the
    midpoint between opposite extremes:
    - cx = (leftmost + rightmost) / 2
    - cy = (topmost + bottommost) / 2

    Args:
        image: Grayscale image

    Returns:
        (center, radius) or (None, None) if not found
    """
    mask = get_sphere_mask(image)
    if mask is None:
        return None, None

    # Find all sphere pixels
    white_pixels = np.where(mask == 255)
    if len(white_pixels[0]) == 0:
        return None, None

    y_coords = white_pixels[0]
    x_coords = white_pixels[1]

    # Minimum area threshold
    h, w = image.shape[:2]
    min_area = (min(h, w) * 0.05) ** 2 * np.pi
    if len(y_coords) < min_area:
        return None, None

    # Find extreme points
    top_y = np.min(y_coords)
    bottom_y = np.max(y_coords)
    left_x = np.min(x_coords)
    right_x = np.max(x_coords)

    # Center is midpoint of extremes
    cx = int((left_x + right_x) / 2)
    cy = int((top_y + bottom_y) / 2)

    # Radius: average of horizontal and vertical half-spans
    radius_x = (right_x - left_x) / 2
    radius_y = (bottom_y - top_y) / 2
    radius = int((radius_x + radius_y) / 2)

    # Sanity check: reject if not roughly circular (aspect ratio > 1.3)
    if min(radius_x, radius_y) > 0 and max(radius_x, radius_y) / min(radius_x, radius_y) > 1.3:
        return None, None

    return (cx, cy), radius


def get_sphere_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Get the binary mask of the sphere using Otsu thresholding.

    Args:
        image: Grayscale image

    Returns:
        Binary mask (255 = sphere, 0 = background) or None if failed
    """
    if image.max() <= 1:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 1)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep only the largest connected component (eliminates corner/edge noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, np.uint8(255), np.uint8(0))

    return mask


def measure_blur_auto(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
    method: str = 'erf',
    verbose: bool = False
) -> BlurMeasurement:
    """
    Measure blur using the specified method.

    Args:
        image: Grayscale image
        center: (x, y) center of sphere. If None, auto-detects.
        radius: Approximate radius. If None, auto-detects.
        method: 'erf', 'gradient', or 'laplacian'
        verbose: If True, print diagnostic information

    Returns:
        BlurMeasurement result
    """
    if method == 'erf':
        return measure_blur_erf(image, center, radius, verbose=verbose)
    elif method == 'gradient':
        return measure_blur_gradient(image, center, radius)
    elif method == 'laplacian':
        return measure_blur_laplacian(image)
    else:
        raise ValueError(f"Unknown method: {method}")


def measure_blur_batch(
    images: List[np.ndarray],
    positions: List[float],
    method: str = 'erf',
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None
) -> Tuple[List[float], List[float], List[BlurMeasurement]]:
    """
    Measure blur for a batch of images at known z-positions.

    Args:
        images: List of grayscale images
        positions: List of z-positions (mm)
        method: Measurement method
        center: Optional fixed center for all images
        radius: Optional fixed radius for all images

    Returns:
        (z_positions, sigmas, measurements)
    """
    sigmas = []
    measurements = []
    valid_positions = []

    # Auto-detect from sharpest image if not provided
    if center is None or radius is None:
        # Find sharpest image (likely at focus)
        sharpness = [cv2.Laplacian(img, cv2.CV_64F).var() for img in images]
        best_idx = np.argmax(sharpness)
        center, radius = detect_sphere(images[best_idx])

    for img, z in zip(images, positions):
        result = measure_blur_auto(img, center, radius, method)

        if result.confidence > 0.5:
            sigmas.append(result.sigma)
            measurements.append(result)
            valid_positions.append(z)
        else:
            sigmas.append(np.nan)
            measurements.append(result)
            valid_positions.append(z)

    return valid_positions, sigmas, measurements
