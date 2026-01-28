"""
Calibration Core Module

This module handles the core calibration logic for determining rho (ρ).
Supports three approaches:
- Approach A: Direct empirical (σ = ρ × |d|)
- Approach B: Optical formula + ρ (σ = ρ × CoC)
- Hybrid: Approach A calibration converted to Approach B format
"""

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# Add training folder to path for importing synthetic_blur
training_path = Path(__file__).parent.parent / "training" / "Training"
if str(training_path) not in sys.path:
    sys.path.insert(0, str(training_path))


@dataclass
class OpticalParams:
    """Optical parameters for Approach B calibration."""
    focal_length_mm: float = 50.0
    f_number: float = 4.0
    focus_distance_mm: float = 300.0
    pixel_size_mm: float = 0.01

    def calculate_coc(self, defocus_mm: float) -> float:
        """
        Calculate theoretical Circle of Confusion in pixels.

        CoC_mm = |f² × d| / (N × D × (D + d))
        CoC_px = CoC_mm / pixel_size

        Args:
            defocus_mm: Distance from focal plane (mm)

        Returns:
            CoC in pixels
        """
        f = self.focal_length_mm
        N = self.f_number
        D = self.focus_distance_mm
        d = D + defocus_mm

        if d <= 0:
            return float('inf')

        coc_mm = abs((f * f * defocus_mm) / (N * D * d))
        coc_px = coc_mm / self.pixel_size_mm

        return coc_px


@dataclass
class CalibrationResultA:
    """Result from Approach A (Direct Empirical) calibration."""
    rho_px_per_mm: float  # Direct conversion: pixels per mm
    sigma_0: float  # Residual blur at focal plane
    r_squared: float  # Fit quality
    num_points: int
    z_values: List[float] = field(default_factory=list)
    sigma_values: List[float] = field(default_factory=list)
    sigma_fitted: List[float] = field(default_factory=list)


@dataclass
class CalibrationResultB:
    """Result from Approach B (Optical Formula) calibration."""
    rho: float  # Dimensionless correction factor
    optical_params: OpticalParams = None
    r_squared: float = 0.0
    num_points: int = 0
    z_values: List[float] = field(default_factory=list)
    sigma_values: List[float] = field(default_factory=list)
    coc_values: List[float] = field(default_factory=list)
    rho_per_point: List[float] = field(default_factory=list)


@dataclass
class CalibrationResultHybrid:
    """Result from Hybrid calibration (A → B conversion)."""
    direct_result: CalibrationResultA = None
    formula_result: CalibrationResultB = None
    conversion_reference_d: float = 5.0  # Reference defocus for conversion


def linear_model(z: np.ndarray, rho: float, sigma_0: float) -> np.ndarray:
    """Linear blur model: σ = ρ × |z| + σ_0"""
    return rho * np.abs(z) + sigma_0


def calibrate_approach_a(
    z_positions: List[float],
    sigma_values: List[float],
    exclude_near_focus: float = 0.5
) -> CalibrationResultA:
    """
    Calibrate using Approach A (Direct Empirical).

    Fits σ = ρ × |z| + σ_0 to the measured data.

    Args:
        z_positions: Defocus positions in mm
        sigma_values: Measured blur sigma in pixels
        exclude_near_focus: Exclude points closer than this to focus (mm)

    Returns:
        CalibrationResultA with fitted parameters
    """
    z = np.array(z_positions)
    sigma = np.array(sigma_values)

    # Remove NaN values
    valid = ~np.isnan(sigma)
    z = z[valid]
    sigma = sigma[valid]

    # Optionally exclude points very close to focus (where σ_0 dominates)
    far_from_focus = np.abs(z) >= exclude_near_focus
    z_fit = z[far_from_focus]
    sigma_fit = sigma[far_from_focus]

    if len(z_fit) < 3:
        # Use all points if not enough
        z_fit = z
        sigma_fit = sigma

    # Fit linear model
    try:
        popt, pcov = curve_fit(
            linear_model, z_fit, sigma_fit,
            p0=[1.0, 1.0],
            bounds=([0.01, 0], [100, 50])
        )
        rho, sigma_0 = popt

        # Calculate R-squared using all data
        sigma_pred = linear_model(z, rho, sigma_0)
        ss_res = np.sum((sigma - sigma_pred) ** 2)
        ss_tot = np.sum((sigma - np.mean(sigma)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    except (RuntimeError, ValueError):
        # Fallback: simple linear regression
        abs_z = np.abs(z)
        rho = np.sum(abs_z * sigma) / np.sum(abs_z ** 2) if np.sum(abs_z ** 2) > 0 else 1.0
        sigma_0 = 0.0
        r_squared = 0.0

    return CalibrationResultA(
        rho_px_per_mm=rho,
        sigma_0=sigma_0,
        r_squared=r_squared,
        num_points=len(z),
        z_values=list(z),
        sigma_values=list(sigma),
        sigma_fitted=list(linear_model(z, rho, sigma_0))
    )


def calibrate_approach_b(
    z_positions: List[float],
    sigma_values: List[float],
    optical_params: OpticalParams,
    exclude_near_focus: float = 0.5
) -> CalibrationResultB:
    """
    Calibrate using Approach B (Optical Formula + ρ).

    For each point, computes theoretical CoC and finds ρ = σ / CoC.

    Args:
        z_positions: Defocus positions in mm
        sigma_values: Measured blur sigma in pixels
        optical_params: Optical parameters for CoC calculation
        exclude_near_focus: Exclude points closer than this to focus (mm)

    Returns:
        CalibrationResultB with fitted ρ
    """
    z = np.array(z_positions)
    sigma = np.array(sigma_values)

    # Remove NaN values
    valid = ~np.isnan(sigma)
    z = z[valid]
    sigma = sigma[valid]

    # Calculate CoC for each point
    coc_values = []
    rho_values = []
    valid_z = []
    valid_sigma = []

    for z_i, sigma_i in zip(z, sigma):
        if abs(z_i) < exclude_near_focus:
            continue

        coc = optical_params.calculate_coc(z_i)

        if coc > 0.1 and coc < 1000:  # Reasonable range
            coc_values.append(coc)
            rho_i = sigma_i / coc
            rho_values.append(rho_i)
            valid_z.append(z_i)
            valid_sigma.append(sigma_i)

    if len(rho_values) == 0:
        return CalibrationResultB(
            rho=1.0,
            optical_params=optical_params,
            r_squared=0.0,
            num_points=0
        )

    # Use median for robustness
    rho_final = np.median(rho_values)

    # Calculate R-squared
    sigma_pred = np.array(coc_values) * rho_final
    ss_res = np.sum((np.array(valid_sigma) - sigma_pred) ** 2)
    ss_tot = np.sum((np.array(valid_sigma) - np.mean(valid_sigma)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return CalibrationResultB(
        rho=rho_final,
        optical_params=optical_params,
        r_squared=r_squared,
        num_points=len(valid_z),
        z_values=valid_z,
        sigma_values=valid_sigma,
        coc_values=coc_values,
        rho_per_point=rho_values
    )


def calibrate_hybrid(
    z_positions: List[float],
    sigma_values: List[float],
    optical_params: OpticalParams,
    reference_defocus: float = 5.0
) -> CalibrationResultHybrid:
    """
    Calibrate using Hybrid approach.

    1. Fit Approach A (direct empirical)
    2. Convert to Approach B ρ for given optical params

    The conversion ensures that at the reference defocus, both approaches
    produce the same blur:
        ρ_formula = (ρ_direct × d_ref) / CoC(d_ref)

    Args:
        z_positions: Defocus positions in mm
        sigma_values: Measured blur sigma in pixels
        optical_params: Optical parameters for Approach B
        reference_defocus: Reference defocus distance for conversion (mm)

    Returns:
        CalibrationResultHybrid with both results
    """
    # Step 1: Approach A calibration
    result_a = calibrate_approach_a(z_positions, sigma_values)

    # Step 2: Convert to Approach B ρ
    coc_at_ref = optical_params.calculate_coc(reference_defocus)
    sigma_at_ref = result_a.rho_px_per_mm * reference_defocus + result_a.sigma_0

    if coc_at_ref > 0:
        rho_formula = sigma_at_ref / coc_at_ref
    else:
        rho_formula = 1.0

    # Create Approach B result with converted ρ
    result_b = CalibrationResultB(
        rho=rho_formula,
        optical_params=optical_params,
        r_squared=result_a.r_squared,  # Same underlying data
        num_points=result_a.num_points
    )

    return CalibrationResultHybrid(
        direct_result=result_a,
        formula_result=result_b,
        conversion_reference_d=reference_defocus
    )


def sigma_to_depth_approach_a(sigma: float, rho_px_per_mm: float, sigma_0: float = 0) -> float:
    """
    Convert blur sigma to depth using Approach A.

    depth = (σ - σ_0) / ρ

    Args:
        sigma: Measured blur sigma (pixels)
        rho_px_per_mm: Calibrated ρ value
        sigma_0: Residual blur at focus

    Returns:
        Absolute depth in mm
    """
    if rho_px_per_mm <= 0:
        return 0.0
    return max(0, (sigma - sigma_0) / rho_px_per_mm)


def sigma_to_depth_approach_b(
    sigma: float,
    rho: float,
    optical_params: OpticalParams,
    max_iterations: int = 100
) -> float:
    """
    Convert blur sigma to depth using Approach B.

    This requires numerical inversion since the CoC formula is nonlinear.

    Args:
        sigma: Measured blur sigma (pixels)
        rho: Calibrated ρ value
        optical_params: Optical parameters

    Returns:
        Absolute depth in mm
    """
    if rho <= 0:
        return 0.0

    target_coc = sigma / rho

    def objective(d):
        coc = optical_params.calculate_coc(d)
        return (coc - target_coc) ** 2

    # Search for the defocus that gives the target CoC
    result = minimize_scalar(
        objective,
        bounds=(0.1, 50),
        method='bounded',
        options={'maxiter': max_iterations}
    )

    return result.x if result.success else sigma / rho  # Fallback to linear


def find_focal_plane(sigmas: List[float], positions: List[float]) -> Tuple[int, float]:
    """
    Find the focal plane (minimum blur position).

    Args:
        sigmas: List of measured blur values
        positions: List of z-positions

    Returns:
        (index, z_position) of focal plane
    """
    sigmas = np.array(sigmas)
    positions = np.array(positions)

    # Remove NaN
    valid = ~np.isnan(sigmas)
    sigmas = sigmas[valid]
    positions = positions[valid]

    if len(sigmas) == 0:
        return 0, 0.0

    min_idx = np.argmin(sigmas)
    return min_idx, positions[min_idx]


def validate_calibration(
    result: CalibrationResultA,
    min_r_squared: float = 0.9,
    max_sigma_0: float = 5.0
) -> Tuple[bool, List[str]]:
    """
    Validate calibration result quality.

    Args:
        result: Calibration result to validate
        min_r_squared: Minimum acceptable R-squared
        max_sigma_0: Maximum acceptable residual blur

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []

    if result.r_squared < min_r_squared:
        warnings.append(f"Low R² ({result.r_squared:.3f} < {min_r_squared})")

    if result.sigma_0 > max_sigma_0:
        warnings.append(f"High residual blur σ_0 ({result.sigma_0:.1f} > {max_sigma_0})")

    if result.num_points < 10:
        warnings.append(f"Few calibration points ({result.num_points} < 10)")

    if result.rho_px_per_mm < 0.5 or result.rho_px_per_mm > 20:
        warnings.append(f"Unusual ρ value ({result.rho_px_per_mm:.2f} px/mm)")

    is_valid = len(warnings) == 0

    return is_valid, warnings


def export_calibration_yaml(
    result_hybrid: CalibrationResultHybrid,
    camera: str = "unknown",
    aperture_setting: str = "unknown",
    focal_plane_offset_mm: float = 0.0
) -> Dict:
    """
    Export calibration results to YAML-compatible dict.

    Args:
        result_hybrid: Hybrid calibration result
        camera: Camera identifier
        aperture_setting: Aperture setting description
        focal_plane_offset_mm: Offset from reference camera

    Returns:
        Dictionary ready for YAML export
    """
    a = result_hybrid.direct_result
    b = result_hybrid.formula_result

    return {
        'camera': camera,
        'aperture_setting': aperture_setting,
        'approach': 'hybrid',

        'direct': {
            'rho_px_per_mm': float(a.rho_px_per_mm),
            'sigma_0': float(a.sigma_0),
            'r_squared': float(a.r_squared),
            'num_points': a.num_points
        },

        'optical_params': {
            'focal_length_mm': b.optical_params.focal_length_mm,
            'f_number': b.optical_params.f_number,
            'focus_distance_mm': b.optical_params.focus_distance_mm,
            'pixel_size_mm': b.optical_params.pixel_size_mm
        },
        'formula_rho': float(b.rho),

        'focal_plane_offset_mm': focal_plane_offset_mm,

        'conversion': {
            'reference_defocus_mm': result_hybrid.conversion_reference_d,
            'method': 'rho_formula = (rho_direct * d_ref) / CoC(d_ref)'
        }
    }
