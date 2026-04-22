"""
Calibration Core Module

This module handles the core calibration logic for determining rho (ρ).
Supports two user-facing approaches:
- Estimated Optics (Hybrid): Fits linear model from data, converts to dimensionless ρ
- Known Optics (Approach B): Uses optical formula directly (σ = ρ × CoC)

Internal functions (calibrate_approach_a) are used by the Hybrid approach.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


@dataclass
class OpticalParams:
    """Optical parameters for Approach B calibration."""
    focal_length_mm: float = 50.0
    f_number: float = 4.0
    focus_distance_mm: float = 300.0
    pixel_size_mm: float = 0.01

    @property
    def aperture_diameter_mm(self) -> float:
        """Calculate aperture diameter from focal length and f-number."""
        return self.focal_length_mm / self.f_number

    @property
    def imaging_distance_mm(self) -> float:
        """
        Calculate imaging distance using thin lens equation.

        From: 1/f = 1/D + 1/u0
        Therefore: u0 = f × D / (D - f)

        Where:
            D = focus_distance (object distance when in focus)
            u0 = imaging_distance (image distance)
            f = focal_length
        """
        f = self.focal_length_mm
        D = self.focus_distance_mm

        if D <= f:
            # Object inside focal length - invalid for real imaging
            return D  # Fallback

        return f * D / (D - f)

    def to_training_format(self, rho: float = 1.0) -> Dict:
        """
        Convert to training module's expected format.

        Args:
            rho: The calibrated rho value (dimensionless)

        Returns:
            Dict compatible with training optical_config.yaml
        """
        return {
            'optics': {
                'focal_length_mm': self.focal_length_mm,
                'aperture_diameter_mm': self.aperture_diameter_mm,
                'focus_distance_mm': self.focus_distance_mm,
                'imaging_distance_mm': self.imaging_distance_mm,
                'pixel_size_mm': self.pixel_size_mm
            },
            'blur': {
                'rho': rho
            }
        }

    def calculate_coc(self, defocus_mm: float) -> float:
        """
        Calculate theoretical Circle of Confusion in pixels.

        Uses Wang et al. formula (Physics of Fluids, 2022):
            CoC = D_lens × u₀ × |1/F - 1/u₀ - 1/(d + d₀)|

        Where:
            D_lens = aperture diameter (mm)
            u₀ = imaging distance (mm)
            F = focal length (mm)
            d₀ = focus distance (mm)
            d = defocus distance (mm)

        Args:
            defocus_mm: Distance from focal plane (mm)

        Returns:
            CoC in pixels
        """
        F = self.focal_length_mm
        d0 = self.focus_distance_mm
        u0 = self.imaging_distance_mm
        D_lens = self.aperture_diameter_mm

        # Object distance
        object_dist = defocus_mm + d0

        if object_dist <= 0:
            return float('inf')

        # Wang et al. formula
        term1 = 1.0 / F - 1.0 / u0
        term2 = 1.0 / object_dist
        coc_mm = D_lens * u0 * abs(term1 - term2)
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


@dataclass
class LOOCVResult:
    """Result from leave-one-out cross-validation on calibration."""
    rho_mean: float
    rho_std: float
    sigma_0_mean: float
    sigma_0_std: float
    loo_residuals: List[float] = field(default_factory=list)
    loo_mae: float = 0.0


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

    Fits σ = ρ × |z| + σ_0 to the measured data, with automatic plateau
    detection. At extreme defocus, blur can saturate (ERF cannot measure
    blur larger than the crop allows), creating a plateau where measured σ
    falls below the true linear relationship. These points are iteratively
    excluded to produce an accurate linear fit.

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

    # Exclude points very close to focus (where σ_0 dominates)
    far_from_focus = np.abs(z) >= exclude_near_focus
    z_fit = z[far_from_focus]
    sigma_fit = sigma[far_from_focus]

    if len(z_fit) < 3:
        z_fit = z
        sigma_fit = sigma

    # Delta-based plateau detection: at extreme defocus, ERF saturates and
    # consecutive points have nearly identical sigma. Detect this by looking
    # at consecutive deltas from each end of the z-range.
    # Sort by z for consecutive delta analysis
    sort_idx = np.argsort(z_fit)
    z_sorted = z_fit[sort_idx]
    s_sorted = sigma_fit[sort_idx]

    deltas = np.abs(np.diff(s_sorted))
    median_delta = np.median(deltas)
    threshold = median_delta * 0.3  # plateau deltas are much smaller than linear

    # Scan from negative end (index 0): remove while delta < threshold
    # Need at least 3 consecutive plateau points to confirm
    neg_plateau_end = 0
    consecutive = 0
    for i in range(len(deltas)):
        if deltas[i] < threshold:
            consecutive += 1
        else:
            if consecutive >= 3:
                neg_plateau_end = i  # first linear point index
            break
    else:
        if consecutive >= 3:
            neg_plateau_end = len(deltas)

    # Scan from positive end (last index): remove while delta < threshold
    pos_plateau_start = len(z_sorted)
    consecutive = 0
    for i in range(len(deltas) - 1, -1, -1):
        if deltas[i] < threshold:
            consecutive += 1
        else:
            if consecutive >= 3:
                pos_plateau_start = i + 1  # last linear point index + 1
            break
    else:
        if consecutive >= 3:
            pos_plateau_start = 0

    # Keep only the linear region
    keep_mask = np.ones(len(z_sorted), dtype=bool)
    keep_mask[:neg_plateau_end] = False
    keep_mask[pos_plateau_start:] = False

    n_plateau = np.sum(~keep_mask)
    z_fit = z_sorted[keep_mask]
    sigma_fit = s_sorted[keep_mask]

    # Final fit on cleaned data
    try:
        popt, pcov = curve_fit(
            linear_model, z_fit, sigma_fit,
            p0=[1.0, 1.0],
            bounds=([0.01, 0], [100, 50])
        )
        rho, sigma_0 = popt

        # R-squared on the fitted points only (not including excluded plateau)
        sigma_pred_fit = linear_model(z_fit, rho, sigma_0)
        ss_res = np.sum((sigma_fit - sigma_pred_fit) ** 2)
        ss_tot = np.sum((sigma_fit - np.mean(sigma_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if n_plateau > 0:
            print(f"  Plateau detection: excluded {n_plateau} saturated points, "
                  f"fitting on {len(z_fit)}/{len(z_fit) + n_plateau} points "
                  f"(threshold={threshold:.3f})")

    except (RuntimeError, ValueError):
        abs_z = np.abs(z)
        rho = np.sum(abs_z * sigma) / np.sum(abs_z ** 2) if np.sum(abs_z ** 2) > 0 else 1.0
        sigma_0 = 0.0
        r_squared = 0.0

    return CalibrationResultA(
        rho_px_per_mm=rho,
        sigma_0=sigma_0,
        r_squared=r_squared,
        num_points=len(z_fit),
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
    focal_plane_offset_mm: float = 0.0,
    defocus_range_mm: Optional[Tuple[float, float]] = None,
    reference_resolution: Optional[int] = None,
    calibration_mode: str = "optical",
    scale_calib_px_per_mm: Optional[float] = None,
) -> Dict:
    """
    Export calibration results to YAML-compatible dict.

    Args:
        result_hybrid: Hybrid calibration result
        camera: Camera identifier
        aperture_setting: Aperture setting description
        focal_plane_offset_mm: Offset from reference camera
        defocus_range_mm: Optional (min, max) defocus range from z-stack
        reference_resolution: Image size (px) where sigma was measured (for scaling)
        calibration_mode: Calibration mode (optical/direct)

    Returns:
        Dictionary ready for YAML export
    """
    a = result_hybrid.direct_result
    b = result_hybrid.formula_result

    result = {
        'camera': camera,
        'aperture_setting': aperture_setting,
        'approach': 'hybrid',
        'calibration_mode': calibration_mode,

        'direct': {
            'rho_px_per_mm': float(a.rho_px_per_mm),
            'sigma_0': float(a.sigma_0),
            'r_squared': float(a.r_squared),
            'num_points': a.num_points,
            **(
                {'scale_calib_px_per_mm': float(scale_calib_px_per_mm)}
                if scale_calib_px_per_mm is not None else {}
            ),
        },

        'optical_params': {
            'focal_length_mm': b.optical_params.focal_length_mm,
            'f_number': b.optical_params.f_number,
            'focus_distance_mm': b.optical_params.focus_distance_mm,
            'pixel_size_mm': b.optical_params.pixel_size_mm
        },
        'formula_rho': float(b.rho),

        'focal_plane_offset_mm': focal_plane_offset_mm,

        # Reference resolution where sigma was measured (for cross-resolution scaling)
        'reference_resolution': reference_resolution,

        'conversion': {
            'reference_defocus_mm': result_hybrid.conversion_reference_d,
            'method': 'rho_formula = (rho_direct * d_ref) / CoC(d_ref)'
        },

        # Training-compatible format (can be copied directly to optical_config.yaml)
        'training_config': {
            'optics': {
                'focal_length_mm': b.optical_params.focal_length_mm,
                'aperture_diameter_mm': b.optical_params.aperture_diameter_mm,
                'focus_distance_mm': b.optical_params.focus_distance_mm,
                'imaging_distance_mm': b.optical_params.imaging_distance_mm,
                'pixel_size_mm': b.optical_params.pixel_size_mm
            },
            'blur': {
                'rho': float(b.rho)
            },
            # Calibration camera info (for cross-camera/resolution scaling)
            'calibration': {
                'pixel_size_mm': b.optical_params.pixel_size_mm,
                'reference_resolution': reference_resolution,
                **(
                    {'scale_calib_px_per_mm': float(scale_calib_px_per_mm)}
                    if scale_calib_px_per_mm is not None else {}
                ),
            }
        }
    }

    # Add defocus range if provided
    if defocus_range_mm is not None:
        result['defocus_range_mm'] = [float(defocus_range_mm[0]), float(defocus_range_mm[1])]
        result['training_config']['data'] = {
            'defocus_range_mm': [float(defocus_range_mm[0]), float(defocus_range_mm[1])]
        }

    return result


def export_calibration_yaml_direct(
    result: CalibrationResultA,
    camera: str = "unknown",
    aperture_setting: str = "unknown",
    pixel_size_mm: Optional[float] = None,
    defocus_range_mm: Optional[Tuple[float, float]] = None,
    reference_resolution: Optional[int] = None,
    scale_calib_px_per_mm: Optional[float] = None,
) -> Dict:
    """
    Export direct-only calibration results to YAML-compatible dict.

    No optical parameters needed — just the linear fit results.

    Args:
        result: Direct calibration result (Approach A)
        camera: Camera identifier
        aperture_setting: Aperture setting description
        pixel_size_mm: Pixel size in mm (for cross-resolution scaling)
        defocus_range_mm: Optional (min, max) defocus range from z-stack
        reference_resolution: Image size (px) where sigma was measured

    Returns:
        Dictionary ready for YAML export
    """
    direct_section = {
        'rho_px_per_mm': float(result.rho_px_per_mm),
        'sigma_0': float(result.sigma_0),
        'r_squared': float(result.r_squared),
        'num_points': result.num_points,
    }
    if scale_calib_px_per_mm is not None:
        direct_section['scale_calib_px_per_mm'] = float(scale_calib_px_per_mm)

    output = {
        'camera': camera,
        'aperture_setting': aperture_setting,
        'calibration_mode': 'direct',

        'direct': direct_section,

        'reference_resolution': reference_resolution,

        # Ready-to-paste block for optical_config.yaml training section
        'training_config': {
            'rho_direct': float(result.rho_px_per_mm),
            'sigma_0': float(result.sigma_0),
            **(
                {'scale_calib_px_per_mm': float(scale_calib_px_per_mm)}
                if scale_calib_px_per_mm is not None else {}
            ),
        },
    }

    if pixel_size_mm is not None:
        output['pixel_size_mm'] = float(pixel_size_mm)

    if defocus_range_mm is not None:
        output['defocus_range_mm'] = [float(defocus_range_mm[0]), float(defocus_range_mm[1])]

    return output


def loo_cv(
    z_positions: List[float],
    sigma_values: List[float],
    exclude_near_focus: float = 0.5,
) -> LOOCVResult:
    """
    Leave-one-out cross-validation on calibration fit.

    For each point, removes it, refits the linear model on the
    remaining n-1 points, and predicts the held-out sigma. Returns
    error bars on rho and sigma_0.

    Args:
        z_positions: Defocus positions in mm
        sigma_values: Measured blur sigma in pixels
        exclude_near_focus: Exclude points closer than this to focus (mm)

    Returns:
        LOOCVResult with parameter uncertainties and prediction errors
    """
    z = np.array(z_positions)
    sigma = np.array(sigma_values)

    # Remove NaN and near-focus points (same filtering as calibrate_approach_a)
    valid = ~np.isnan(sigma) & (np.abs(z) >= exclude_near_focus)
    z = z[valid]
    sigma = sigma[valid]
    n = len(z)

    if n < 4:
        return LOOCVResult(
            rho_mean=0.0, rho_std=0.0,
            sigma_0_mean=0.0, sigma_0_std=0.0,
            loo_residuals=[], loo_mae=0.0,
        )

    rho_values = []
    sigma_0_values = []
    residuals = []

    for i in range(n):
        z_train = np.delete(z, i)
        sigma_train = np.delete(sigma, i)

        try:
            popt, _ = curve_fit(
                linear_model, z_train, sigma_train,
                p0=[1.0, 1.0],
                bounds=([0.01, 0], [100, 50]),
            )
            rho_i, sigma_0_i = popt
            rho_values.append(rho_i)
            sigma_0_values.append(sigma_0_i)

            # Predict held-out point
            pred = linear_model(np.array([z[i]]), rho_i, sigma_0_i)[0]
            residuals.append(pred - sigma[i])
        except (RuntimeError, ValueError):
            continue

    if not rho_values:
        return LOOCVResult(
            rho_mean=0.0, rho_std=0.0,
            sigma_0_mean=0.0, sigma_0_std=0.0,
            loo_residuals=[], loo_mae=0.0,
        )

    return LOOCVResult(
        rho_mean=float(np.mean(rho_values)),
        rho_std=float(np.std(rho_values)),
        sigma_0_mean=float(np.mean(sigma_0_values)),
        sigma_0_std=float(np.std(sigma_0_values)),
        loo_residuals=[float(r) for r in residuals],
        loo_mae=float(np.mean(np.abs(residuals))),
    )


def generate_quality_report(
    result: CalibrationResultA,
    output_path: Path,
    loo_result: Optional[LOOCVResult] = None,
    camera: str = "unknown",
) -> None:
    """
    Generate a standalone calibration quality report as a PNG image.

    Contains: summary statistics, calibration curve with fit,
    residual plot, and optionally LOO-CV results.

    Args:
        result: Calibration result from Approach A
        output_path: Path to save the report image
        loo_result: Optional LOO-CV results for uncertainty display
        camera: Camera identifier for the title
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Calibration Quality Report — Camera {camera}", fontsize=13)

    z = np.array(result.z_values)
    sigma = np.array(result.sigma_values)
    sigma_fit = np.array(result.sigma_fitted)

    # Left panel: calibration curve + fit
    ax1 = axes[0]
    ax1.scatter(z, sigma, s=20, c="steelblue", zorder=3, label="Measured")
    sort_idx = np.argsort(z)
    ax1.plot(z[sort_idx], sigma_fit[sort_idx], "r-", linewidth=1.5,
             label=f"Fit: \u03c1={result.rho_px_per_mm:.4f} px/mm")
    ax1.set_xlabel("Defocus z (mm)")
    ax1.set_ylabel("Blur \u03c3 (px)")
    ax1.set_title("Calibration Curve")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right panel: residuals
    ax2 = axes[1]
    residuals = sigma - sigma_fit
    ax2.scatter(z, residuals, s=20, c="steelblue", zorder=3)
    ax2.axhline(0, color="red", linewidth=1, linestyle="--")
    ax2.set_xlabel("Defocus z (mm)")
    ax2.set_ylabel("Residual (px)")
    ax2.set_title("Fit Residuals")
    ax2.grid(True, alpha=0.3)

    # Summary text box
    quality = "Good" if result.r_squared > 0.95 else ("Fair" if result.r_squared > 0.9 else "Poor")
    lines = [
        f"R\u00b2 = {result.r_squared:.4f}  ({quality})",
        f"\u03c1 = {result.rho_px_per_mm:.4f} px/mm",
        f"\u03c3\u2080 = {result.sigma_0:.3f} px",
        f"n = {result.num_points} points",
    ]
    if loo_result and loo_result.rho_std > 0:
        lines.append(f"\u03c1 (LOO-CV) = {loo_result.rho_mean:.4f} \u00b1 {loo_result.rho_std:.4f}")
        lines.append(f"\u03c3\u2080 (LOO-CV) = {loo_result.sigma_0_mean:.3f} \u00b1 {loo_result.sigma_0_std:.3f}")
        lines.append(f"LOO MAE = {loo_result.loo_mae:.3f} px")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    text = "\n".join(lines)
    fig.text(0.5, -0.02, text, ha="center", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
