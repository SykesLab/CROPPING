"""
Blur Measurement Methods for Calibration

This module provides different methods to measure blur (sigma) from calibration images.
Each method quantifies how blurry a sphere or edge appears in an image.

Methods:
1. Sigmoid edge fitting (recommended) - fits sigmoid to edge profile
2. Gradient-based (Sobel) - measures average gradient magnitude at edges
3. Laplacian variance - simple sharpness metric (inverse relationship to blur)
"""

import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.ndimage import sobel, gaussian_filter
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class BlurMeasurement:
    """Result of a blur measurement."""
    sigma: float  # Blur sigma in pixels
    method: str  # Which method was used
    confidence: float  # 0-1 confidence score
    details: Dict  # Method-specific details


def sigmoid(r: np.ndarray, I_bg: float, I_sphere: float, r_edge: float, sigma: float) -> np.ndarray:
    """
    Sigmoid function for edge profile fitting.

    I(r) = I_bg - (I_bg - I_sphere) / (1 + exp((r - r_edge) / sigma))

    Args:
        r: Radial positions (pixels)
        I_bg: Background intensity
        I_sphere: Sphere intensity
        r_edge: Edge position
        sigma: Blur width (what we want to measure)

    Returns:
        Intensity values
    """
    # Avoid overflow in exp
    z = np.clip((r - r_edge) / max(sigma, 0.1), -50, 50)
    return I_bg - (I_bg - I_sphere) / (1 + np.exp(z))


def measure_blur_sigmoid(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
    num_rays: int = 36
) -> BlurMeasurement:
    """
    Measure blur by fitting sigmoid to edge profiles.

    This method extracts radial intensity profiles from the sphere center
    outward, fits a sigmoid to each profile, and averages the sigma values.

    Args:
        image: Grayscale image (0-255 or 0-1)
        center: (x, y) center of sphere. If None, auto-detects.
        radius: Approximate radius of sphere. If None, auto-detects.
        num_rays: Number of radial profiles to extract

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
                sigma=0.0, method='sigmoid', confidence=0.0,
                details={'error': 'Could not detect sphere'}
            )

    cx, cy = center
    h, w = image.shape[:2]

    # Extract radial profiles
    sigmas = []
    r_squareds = []

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays

        # Sample points along ray - start from 50% of radius to skip center artifacts
        # (e.g., bright spots, reflections)
        start_r = int(radius * 0.5)
        max_r = int(min(radius * 1.5, min(cx, cy, w - cx, h - cy)))
        r_values = np.arange(start_r, max_r)

        x_coords = (cx + r_values * np.cos(angle)).astype(int)
        y_coords = (cy + r_values * np.sin(angle)).astype(int)

        # Ensure within bounds
        valid = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        r_valid = r_values[valid]
        x_valid = x_coords[valid]
        y_valid = y_coords[valid]

        if len(r_valid) < 10:
            continue

        # Get intensity profile
        intensities = image[y_valid, x_valid]

        # Fit sigmoid
        try:
            # Initial guesses - use start of profile (inside sphere) and end (background)
            I_bg_init = np.median(intensities[-10:])
            I_sphere_init = np.median(intensities[:10])
            r_edge_init = radius
            sigma_init = 2.0

            popt, pcov = curve_fit(
                sigmoid, r_valid, intensities,
                p0=[I_bg_init, I_sphere_init, r_edge_init, sigma_init],
                bounds=(
                    [0, 0, radius * 0.5, 0.1],
                    [1, 1, radius * 1.5, 50]
                ),
                maxfev=1000
            )

            I_bg, I_sphere, r_edge, sigma = popt

            # Calculate R-squared
            fitted = sigmoid(r_valid, *popt)
            ss_res = np.sum((intensities - fitted) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r_squared > 0.8 and sigma > 0.1:
                sigmas.append(sigma)
                r_squareds.append(r_squared)

        except (RuntimeError, ValueError):
            continue

    if len(sigmas) == 0:
        return BlurMeasurement(
            sigma=0.0, method='sigmoid', confidence=0.0,
            details={'error': 'Could not fit sigmoid to any rays'}
        )

    # Use median to be robust to outliers
    sigma_final = np.median(sigmas)
    confidence = np.mean(r_squareds)

    return BlurMeasurement(
        sigma=sigma_final,
        method='sigmoid',
        confidence=confidence,
        details={
            'num_rays_used': len(sigmas),
            'sigma_std': np.std(sigmas),
            'mean_r_squared': np.mean(r_squareds),
            'all_sigmas': sigmas
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

    Uses Otsu thresholding to get binary mask, then finds center as the
    midpoint between opposite extremes:
    - cx = (leftmost + rightmost) / 2
    - cy = (topmost + bottommost) / 2

    This gives a geometric center that works for irregular/elliptical shapes.

    Args:
        image: Grayscale image

    Returns:
        (center, radius) or (None, None) if not found
    """
    # Ensure uint8 for OpenCV
    if image.max() <= 1:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 1)

    # Otsu thresholding - automatically finds optimal threshold
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all white (sphere) pixels
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

    return mask


def measure_blur_auto(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
    method: str = 'sigmoid'
) -> BlurMeasurement:
    """
    Measure blur using the specified method.

    Args:
        image: Grayscale image
        center: (x, y) center of sphere. If None, auto-detects.
        radius: Approximate radius. If None, auto-detects.
        method: 'sigmoid', 'gradient', or 'laplacian'

    Returns:
        BlurMeasurement result
    """
    if method == 'sigmoid':
        return measure_blur_sigmoid(image, center, radius)
    elif method == 'gradient':
        return measure_blur_gradient(image, center, radius)
    elif method == 'laplacian':
        return measure_blur_laplacian(image)
    else:
        raise ValueError(f"Unknown method: {method}")


def measure_blur_batch(
    images: List[np.ndarray],
    positions: List[float],
    method: str = 'sigmoid',
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
