"""
Validation Module for Calibration

This module validates calibration results against historical data (e.g., 2021 dataset).
It helps determine which aperture setting produces physically plausible depths.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import cv2

from blur_measurement import measure_blur_auto, BlurMeasurement
from calibration_core import (
    CalibrationResultA, CalibrationResultB, CalibrationResultHybrid,
    sigma_to_depth_approach_a, sigma_to_depth_approach_b, OpticalParams
)


@dataclass
class ValidationResult:
    """Result of validating calibration against test images."""
    num_images: int
    depths: List[float]
    sigmas: List[float]

    # Statistics
    depth_min: float = 0.0
    depth_max: float = 0.0
    depth_mean: float = 0.0
    depth_std: float = 0.0
    depth_median: float = 0.0

    # Plausibility assessment
    is_plausible: bool = False
    plausibility_score: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

        if len(self.depths) > 0:
            self.depth_min = np.min(self.depths)
            self.depth_max = np.max(self.depths)
            self.depth_mean = np.mean(self.depths)
            self.depth_std = np.std(self.depths)
            self.depth_median = np.median(self.depths)


def load_validation_images(
    folder: Path,
    extensions: List[str] = None,
    max_images: int = 100
) -> List[Tuple[Path, np.ndarray]]:
    """
    Load validation images from a folder.

    Args:
        folder: Folder containing images
        extensions: List of valid extensions
        max_images: Maximum number of images to load

    Returns:
        List of (path, image) tuples
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    folder = Path(folder)
    images = []

    for ext in extensions:
        for img_path in folder.glob(f'*{ext}'):
            if len(images) >= max_images:
                break

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((img_path, img))

    return images


def measure_blur_from_images(
    images: List[Tuple[Path, np.ndarray]],
    method: str = 'sigmoid'
) -> List[Tuple[Path, float, BlurMeasurement]]:
    """
    Measure blur from a list of images.

    Args:
        images: List of (path, image) tuples
        method: Blur measurement method

    Returns:
        List of (path, sigma, measurement) tuples
    """
    results = []

    for path, img in images:
        measurement = measure_blur_auto(img, method=method)
        results.append((path, measurement.sigma, measurement))

    return results


def validate_calibration_approach_a(
    blur_measurements: List[Tuple[Path, float, BlurMeasurement]],
    calibration: CalibrationResultA,
    expected_depth_range: Tuple[float, float] = (-15, 15)
) -> ValidationResult:
    """
    Validate Approach A calibration against measured blur values.

    Args:
        blur_measurements: List of (path, sigma, measurement) from validation images
        calibration: Approach A calibration result
        expected_depth_range: Expected range of depths in mm

    Returns:
        ValidationResult with depth distribution
    """
    sigmas = []
    depths = []

    for path, sigma, measurement in blur_measurements:
        if measurement.confidence > 0.5 and sigma > 0:
            sigmas.append(sigma)
            depth = sigma_to_depth_approach_a(
                sigma,
                calibration.rho_px_per_mm,
                calibration.sigma_0
            )
            depths.append(depth)

    result = ValidationResult(
        num_images=len(depths),
        depths=depths,
        sigmas=sigmas
    )

    # Assess plausibility - all thresholds relative to expected range
    result.warnings = []
    result.plausibility_score = 0.0

    range_size = expected_depth_range[1] - expected_depth_range[0]

    if len(depths) == 0:
        result.warnings.append("No valid measurements")
        result.is_plausible = False
        return result

    # Check 1: Most depths within expected range
    in_range = sum(1 for d in depths if expected_depth_range[0] <= d <= expected_depth_range[1])
    in_range_fraction = in_range / len(depths)

    if in_range_fraction < 0.8:
        result.warnings.append(
            f"Only {in_range_fraction*100:.0f}% of depths in expected range "
            f"[{expected_depth_range[0]}, {expected_depth_range[1]}] mm"
        )
    else:
        result.plausibility_score += 0.4

    # Check 2: Reasonable spread - relative to range size
    std_too_low = 0.1 * range_size   # e.g., 0.6mm for ±3mm range
    std_too_high = 0.5 * range_size  # e.g., 3mm for ±3mm range

    if result.depth_std < std_too_low:
        result.warnings.append(f"Very low depth variance (σ={result.depth_std:.2f} mm, expected >{std_too_low:.1f} mm)")
    elif result.depth_std > std_too_high:
        result.warnings.append(f"Very high depth variance (σ={result.depth_std:.2f} mm, expected <{std_too_high:.1f} mm)")
    else:
        result.plausibility_score += 0.3

    # Check 3: Mean depth near zero - relative to range size
    mean_threshold = 0.3 * range_size  # e.g., 1.8mm for ±3mm range

    if abs(result.depth_mean) > mean_threshold:
        result.warnings.append(f"Mean depth far from zero ({result.depth_mean:.1f} mm, expected within ±{mean_threshold:.1f} mm)")
    else:
        result.plausibility_score += 0.3

    result.is_plausible = result.plausibility_score >= 0.7

    return result


def validate_calibration_approach_b(
    blur_measurements: List[Tuple[Path, float, BlurMeasurement]],
    calibration: CalibrationResultB,
    expected_depth_range: Tuple[float, float] = (-15, 15)
) -> ValidationResult:
    """
    Validate Approach B calibration against measured blur values.

    Args:
        blur_measurements: List of (path, sigma, measurement) from validation images
        calibration: Approach B calibration result
        expected_depth_range: Expected range of depths in mm

    Returns:
        ValidationResult with depth distribution
    """
    sigmas = []
    depths = []

    for path, sigma, measurement in blur_measurements:
        if measurement.confidence > 0.5 and sigma > 0:
            sigmas.append(sigma)
            depth = sigma_to_depth_approach_b(
                sigma,
                calibration.rho,
                calibration.optical_params
            )
            depths.append(depth)

    result = ValidationResult(
        num_images=len(depths),
        depths=depths,
        sigmas=sigmas
    )

    # Plausibility assessment - all thresholds relative to expected range
    result.warnings = []
    result.plausibility_score = 0.0

    range_size = expected_depth_range[1] - expected_depth_range[0]

    if len(depths) == 0:
        result.warnings.append("No valid measurements")
        result.is_plausible = False
        return result

    in_range = sum(1 for d in depths if expected_depth_range[0] <= d <= expected_depth_range[1])
    in_range_fraction = in_range / len(depths)

    if in_range_fraction < 0.8:
        result.warnings.append(
            f"Only {in_range_fraction*100:.0f}% of depths in expected range "
            f"[{expected_depth_range[0]}, {expected_depth_range[1]}] mm"
        )
    else:
        result.plausibility_score += 0.4

    # Check 2: Reasonable spread - relative to range size
    std_too_low = 0.1 * range_size
    std_too_high = 0.5 * range_size

    if result.depth_std < std_too_low:
        result.warnings.append(f"Very low depth variance (σ={result.depth_std:.2f} mm, expected >{std_too_low:.1f} mm)")
    elif result.depth_std > std_too_high:
        result.warnings.append(f"Very high depth variance (σ={result.depth_std:.2f} mm, expected <{std_too_high:.1f} mm)")
    else:
        result.plausibility_score += 0.3

    # Check 3: Mean depth near zero - relative to range size
    mean_threshold = 0.3 * range_size

    if abs(result.depth_mean) > mean_threshold:
        result.warnings.append(f"Mean depth far from zero ({result.depth_mean:.1f} mm, expected within ±{mean_threshold:.1f} mm)")
    else:
        result.plausibility_score += 0.3

    result.is_plausible = result.plausibility_score >= 0.7

    return result


def compare_aperture_settings(
    validation_results: Dict[str, ValidationResult]
) -> Tuple[str, Dict[str, float]]:
    """
    Compare validation results across different aperture settings.

    Args:
        validation_results: Dict mapping aperture_setting -> ValidationResult

    Returns:
        (best_aperture, scores_dict)
    """
    scores = {}

    for aperture, result in validation_results.items():
        scores[aperture] = result.plausibility_score

    if not scores:
        return None, {}

    best_aperture = max(scores, key=scores.get)

    return best_aperture, scores


def generate_validation_report(
    validation_results: Dict[str, ValidationResult]
) -> str:
    """
    Generate a text report comparing validation results.

    Args:
        validation_results: Dict mapping aperture_setting -> ValidationResult

    Returns:
        Formatted report string
    """
    lines = ["=" * 60]
    lines.append("VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    for aperture, result in validation_results.items():
        lines.append(f"Aperture Setting: {aperture}")
        lines.append("-" * 40)
        lines.append(f"  Images analyzed: {result.num_images}")
        lines.append(f"  Depth range: [{result.depth_min:.1f}, {result.depth_max:.1f}] mm")
        lines.append(f"  Depth mean: {result.depth_mean:.1f} mm")
        lines.append(f"  Depth std: {result.depth_std:.1f} mm")
        lines.append(f"  Plausibility score: {result.plausibility_score:.2f}")
        lines.append(f"  Is plausible: {'YES' if result.is_plausible else 'NO'}")

        if result.warnings:
            lines.append("  Warnings:")
            for warning in result.warnings:
                lines.append(f"    - {warning}")

        lines.append("")

    # Summary
    best, scores = compare_aperture_settings(validation_results)
    if best:
        lines.append("=" * 60)
        lines.append("RECOMMENDATION")
        lines.append("=" * 60)
        lines.append(f"Best aperture setting: {best} (score: {scores[best]:.2f})")

    return "\n".join(lines)


@dataclass
class MultiCameraValidation:
    """Validation for multi-camera sign resolution."""
    camera_results: Dict[str, ValidationResult]
    focal_plane_offsets: Dict[str, float]  # camera -> offset from reference


def validate_sign_resolution(
    sigma_cam1: float,
    sigma_cam2: float,
    rho_cam1: float,
    rho_cam2: float,
    focal_offset: float
) -> Tuple[float, int]:
    """
    Validate sign resolution between two cameras.

    Args:
        sigma_cam1: Blur from camera 1
        sigma_cam2: Blur from camera 2
        rho_cam1: Calibration constant for camera 1
        rho_cam2: Calibration constant for camera 2
        focal_offset: Camera 2 focal plane offset from camera 1 (mm)

    Returns:
        (depth_magnitude, sign) where sign is +1 or -1
    """
    d1 = sigma_cam1 / rho_cam1
    d2 = sigma_cam2 / rho_cam2

    # Average magnitude
    depth_magnitude = (d1 + d2) / 2

    # Determine sign
    if focal_offset > 0:
        # Camera 2's focal plane is behind camera 1's
        if sigma_cam2 < sigma_cam1:
            sign = +1  # Droplet is behind camera 1's focal plane
        else:
            sign = -1  # Droplet is in front
    else:
        # Reverse logic
        if sigma_cam2 < sigma_cam1:
            sign = -1
        else:
            sign = +1

    return depth_magnitude, sign
