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
    """Linear blur model: σ = ρ × |z| + σ_0.

    Used by ``calibrate_approach_a`` and other linear-fit pathways.
    For method-aware forward σ↔z conversion supporting all three
    methods, use ``physics.CalibrationModel.forward(abs_z_mm)``.
    """
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

    try:
        popt, pcov = curve_fit(
            linear_model, z_fit, sigma_fit,
            p0=[1.0, 1.0],
            bounds=([0.01, 0], [100, 50])
        )
        rho, sigma_0 = popt

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

    valid = ~np.isnan(sigma)
    z = z[valid]
    sigma = sigma[valid]

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
    result_a = calibrate_approach_a(z_positions, sigma_values)

    coc_at_ref = optical_params.calculate_coc(reference_defocus)
    sigma_at_ref = result_a.rho_px_per_mm * reference_defocus + result_a.sigma_0

    if coc_at_ref > 0:
        rho_formula = sigma_at_ref / coc_at_ref
    else:
        rho_formula = 1.0

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
    Convert blur sigma to depth using Approach A (LINEAR ONLY).

    depth = (σ - σ_0) / ρ

    .. note::
        Linear-formula only. For quadrature/hybrid calibrations, prefer
        ``physics.CalibrationModel.inverse(sigma_calib_px)`` which handles
        all three methods correctly. This function is preserved for the
        legacy Approach A pathway and back-compat callers.

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


def loo_cv_for_method(
    method: str,
    z_positions: List[float],
    sigma_values: List[float],
    exclude_near_focus: float = 0.5,
) -> dict:
    """Method-aware leave-one-out cross-validation.

    Like ``loo_cv`` but refits the requested method (linear / quadrature /
    hybrid) each fold. The returned uncertainty bars apply to THAT
    method, unlike ``loo_cv`` which is linear-only.

    Returns a dict with the same shape across methods so callers can
    consume it generically:

    - ``rho_mean`` / ``rho_std`` — slope uncertainty (px/mm)
    - ``aux_param_name`` — 'sigma_0' for linear, 'sigma_floor' for
      quadrature/hybrid
    - ``aux_param_mean`` / ``aux_param_std`` — uncertainty on the
      method's second parameter
    - ``loo_mae`` — mean absolute prediction error on held-out points
    - ``loo_residuals`` — per-fold prediction errors
    - ``num_folds`` — actual number of successful folds
    - ``method`` — echoed back for clarity

    For hybrid, the residual LUT is refitted each fold but only its
    parametric part (rho, sigma_floor) is tracked for uncertainty —
    the LUT itself doesn't have a meaningful "std" since each entry
    is a deterministic per-point residual.
    """
    z = np.asarray(z_positions, dtype=float)
    sigma = np.asarray(sigma_values, dtype=float)
    valid = ~np.isnan(sigma) & (np.abs(z) >= exclude_near_focus)
    z = z[valid]
    sigma = sigma[valid]
    n = len(z)

    aux_name = 'sigma_0' if method == 'linear' else 'sigma_floor'

    if n < 4:
        return {
            'method': method,
            'rho_mean': 0.0, 'rho_std': 0.0,
            'aux_param_name': aux_name,
            'aux_param_mean': 0.0, 'aux_param_std': 0.0,
            'loo_residuals': [], 'loo_mae': 0.0, 'num_folds': 0,
        }

    rho_values = []
    aux_values = []
    residuals = []

    for i in range(n):
        z_train = list(np.delete(z, i))
        sigma_train = list(np.delete(sigma, i))
        try:
            model = calibrate_to_model(
                method, z_train, sigma_train,
                exclude_near_focus=0.0,  # already filtered above
            )
            rho_values.append(model.rho_px_per_mm)
            aux_values.append(model.sigma_0_calib_px if method == 'linear'
                                 else model.sigma_floor_calib_px)
            # Predict held-out point
            pred = float(model.forward(abs(float(z[i]))))
            residuals.append(pred - float(sigma[i]))
        except Exception:
            continue

    if not rho_values:
        return {
            'method': method,
            'rho_mean': 0.0, 'rho_std': 0.0,
            'aux_param_name': aux_name,
            'aux_param_mean': 0.0, 'aux_param_std': 0.0,
            'loo_residuals': [], 'loo_mae': 0.0, 'num_folds': 0,
        }

    return {
        'method': method,
        'rho_mean': float(np.mean(rho_values)),
        'rho_std': float(np.std(rho_values)),
        'aux_param_name': aux_name,
        'aux_param_mean': float(np.mean(aux_values)),
        'aux_param_std': float(np.std(aux_values)),
        'loo_residuals': [float(r) for r in residuals],
        'loo_mae': float(np.mean(np.abs(residuals))),
        'num_folds': len(rho_values),
    }


def generate_quality_report(
    result: CalibrationResultA,
    output_path: Path,
    loo_result: Optional[LOOCVResult] = None,
    camera: str = "unknown",
    calibration_model=None,  # type: Optional[Any]
) -> None:
    """
    Generate a standalone calibration quality report as a PNG image.

    Contains: summary statistics, calibration curve with fit,
    residual plot, and optionally LOO-CV results.

    Phase 11: when ``calibration_model`` (the unified
    ``physics.CalibrationModel``) is provided, the report is method-aware:
    the fit curve, residuals, and LOO numbers all reflect the chosen
    method (linear / quadrature / hybrid). Without it, falls back to
    the linear fit embedded in ``result``.

    Args:
        result: Calibration result from Approach A (linear-only legacy)
        output_path: Path to save the report image
        loo_result: Optional LOO-CV results (linear) for uncertainty display
        camera: Camera identifier for the title
        calibration_model: Optional CalibrationModel for method-aware
            display. Carries its own loo_cv field with per-method LOO.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cm = calibration_model
    method_label = cm.method.upper() if cm is not None else "LINEAR"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Calibration Quality Report — Camera {camera} "
        f"(method: {method_label})",
        fontsize=13)

    z = np.array(result.z_values)
    sigma = np.array(result.sigma_values)
    if cm is not None:
        # Method-aware fit curve via the unified CalibrationModel
        sigma_fit = np.array([float(cm.forward(abs(float(zi)))) for zi in z])
    else:
        sigma_fit = np.array(result.sigma_fitted)

    # Left panel: calibration curve + fit
    ax1 = axes[0]
    ax1.scatter(z, sigma, s=20, c="steelblue", zorder=3, label="Measured")
    sort_idx = np.argsort(z)
    # Method-aware fit-line label
    if cm is not None:
        if cm.method == 'linear':
            _fit_lbl = (f"Fit (linear): \u03c1={cm.rho_px_per_mm:.4f}, "
                        f"\u03c3\u2080={cm.sigma_0_calib_px:.3f}")
        elif cm.method == 'quadrature':
            _fit_lbl = (f"Fit (quadrature): \u03c1={cm.rho_px_per_mm:.4f}, "
                        f"\u03c3_floor={cm.sigma_floor_calib_px:.3f}")
        else:
            _lut_n = (len(cm.residual_lut_mm_px)
                      if cm.residual_lut_mm_px else 0)
            _fit_lbl = (f"Fit (hybrid): \u03c1={cm.rho_px_per_mm:.4f}, "
                        f"\u03c3_floor={cm.sigma_floor_calib_px:.3f}, "
                        f"LUT n={_lut_n}")
    else:
        _fit_lbl = f"Fit: \u03c1={result.rho_px_per_mm:.4f} px/mm"
    ax1.plot(z[sort_idx], sigma_fit[sort_idx], "r-", linewidth=1.5,
             label=_fit_lbl)
    # Show linear fit as faint reference when chosen method is non-linear
    if cm is not None and cm.method != 'linear':
        sigma_lin = np.array(result.sigma_fitted)
        ax1.plot(z[sort_idx], sigma_lin[sort_idx], "--", color="gray",
                 alpha=0.5, linewidth=1,
                 label=f"(linear ref: \u03c1={result.rho_px_per_mm:.4f})")
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
    # Method-aware LOO when present on the CalibrationModel; else linear fallback
    method_loo = (cm.loo_cv if (cm is not None and cm.loo_cv) else None)
    if method_loo and method_loo.get('rho_std', 0) > 0:
        aux_name = method_loo.get('aux_param_name', 'aux')
        lines.append(
            f"\u03c1 (LOO-CV, {method_label.lower()}) = "
            f"{cm.rho_px_per_mm:.4f} \u00b1 {method_loo['rho_std']:.4f}")
        lines.append(
            f"{aux_name} std (LOO-CV) = "
            f"\u00b1 {method_loo['aux_param_std']:.4f}")
        lines.append(f"LOO MAE = {method_loo['loo_mae']:.3f} px")
    elif loo_result and loo_result.rho_std > 0:
        lines.append(f"\u03c1 (LOO-CV, linear) = {loo_result.rho_mean:.4f} \u00b1 {loo_result.rho_std:.4f}")
        lines.append(
            f"\u03c3\u2080 (LOO-CV, linear) = {loo_result.sigma_0_mean:.3f} \u00b1 {loo_result.sigma_0_std:.3f}")
        lines.append(f"LOO MAE = {loo_result.loo_mae:.3f} px")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    text = "\n".join(lines)
    fig.text(0.5, -0.02, text, ha="center", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 — Quadrature and hybrid calibration fits
# (Linear fit `calibrate_approach_a` above is unchanged for backward compat;
# these new fits return a `CalibrationModel` directly so downstream pipeline
# stages can use the unified abstraction.)
# ══════════════════════════════════════════════════════════════════════════


def _ensure_repo_in_syspath():
    """physics.py lives at the CROPPING root. When this module is imported
    from a context that didn't already add the root (e.g. a test or
    standalone script), make sure it's available."""
    import sys as _sys
    from pathlib import Path as _P
    _root = _P(__file__).resolve().parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))


def _filter_for_fit(z_positions, sigma_values, exclude_near_focus=0.5):
    """Apply the same near-focus + plateau filtering that
    ``calibrate_approach_a`` uses, returning the kept (z, sigma) arrays
    plus the count of excluded points by reason.

    Returns
    -------
    z_kept : np.ndarray
    sigma_kept : np.ndarray
    n_near_focus_excluded : int
    n_plateau_excluded : int
    """
    z = np.asarray(z_positions, dtype=float)
    sigma = np.asarray(sigma_values, dtype=float)

    valid = ~np.isnan(sigma)
    z = z[valid]
    sigma = sigma[valid]

    # Near-focus exclusion
    far_from_focus = np.abs(z) >= exclude_near_focus
    n_near_focus = int(np.sum(~far_from_focus))
    z_far = z[far_from_focus]
    sigma_far = sigma[far_from_focus]

    if len(z_far) < 3:
        return z, sigma, n_near_focus, 0  # too few; skip plateau detection

    # Plateau detection (mirrors calibrate_approach_a logic)
    sort_idx = np.argsort(z_far)
    z_sorted = z_far[sort_idx]
    s_sorted = sigma_far[sort_idx]
    deltas = np.abs(np.diff(s_sorted))
    median_delta = np.median(deltas) if len(deltas) > 0 else 0.0
    threshold = median_delta * 0.3

    neg_plateau_end = 0
    consecutive = 0
    for i in range(len(deltas)):
        if deltas[i] < threshold:
            consecutive += 1
        else:
            if consecutive >= 3:
                neg_plateau_end = i
            break
    else:
        if consecutive >= 3:
            neg_plateau_end = len(deltas)

    pos_plateau_start = len(z_sorted)
    consecutive = 0
    for i in range(len(deltas) - 1, -1, -1):
        if deltas[i] < threshold:
            consecutive += 1
        else:
            if consecutive >= 3:
                pos_plateau_start = i + 1
            break
    else:
        if consecutive >= 3:
            pos_plateau_start = 0

    keep_mask = np.ones(len(z_sorted), dtype=bool)
    keep_mask[:neg_plateau_end] = False
    keep_mask[pos_plateau_start:] = False
    n_plateau = int(np.sum(~keep_mask))

    return z_sorted[keep_mask], s_sorted[keep_mask], n_near_focus, n_plateau


def _per_side_slopes(z_kept, sigma_kept):
    """Diagnostic only: fit ρ separately for +z and −z subsets.

    Returns ``{'neg': ρ_neg, 'pos': ρ_pos}`` (each None if insufficient
    data on that side). Used as ``fit_metadata.rho_per_side`` to
    quantify lens asymmetry.
    """
    out = {"neg": None, "pos": None}
    z = np.asarray(z_kept, dtype=float)
    s = np.asarray(sigma_kept, dtype=float)
    for side, mask in [("neg", z < 0), ("pos", z > 0)]:
        if int(np.sum(mask)) < 3:
            continue
        z_side = np.abs(z[mask])
        s_side = s[mask]
        try:
            popt, _ = curve_fit(linear_model, z_side, s_side,
                                  p0=[1.0, np.min(s_side)],
                                  bounds=([0.01, 0], [100, 50]))
            out[side] = float(popt[0])
        except (RuntimeError, ValueError):
            pass
    return out


def _bounds_from_fit(z_kept, sigma_kept, model):
    """Compute trust bounds (σ_min/max_trusted, z_min/max_trusted, plus
    asymmetric per-side σ and z caps) from the kept-after-filter points
    and the fitted CalibrationModel.

    sigma_max_trusted is the *cross-side conservative* min — the
    smaller of the per-side σ ceilings. This is what a sign-blind model
    can trust regardless of which side a prediction came from. The
    per-side values are also recorded for diagnostics and for downstream
    consumers that know the sign.
    """
    z_kept_arr = np.asarray(z_kept, dtype=float)
    sigma_kept_arr = np.asarray(sigma_kept, dtype=float)

    sigma_min = float(np.min(sigma_kept_arr))

    # Per-side σ ceilings — the fit's σ domain on each side independently.
    neg_mask = z_kept_arr < 0
    pos_mask = z_kept_arr > 0
    sigma_max_neg = float(sigma_kept_arr[neg_mask].max()) if np.any(neg_mask) else None
    sigma_max_pos = float(sigma_kept_arr[pos_mask].max()) if np.any(pos_mask) else None
    candidates = [s for s in (sigma_max_neg, sigma_max_pos) if s is not None]
    sigma_max = min(candidates) if candidates else float(sigma_kept_arr.max())

    abs_z_min = float(np.min(np.abs(z_kept_arr)))
    abs_z_max = float(np.max(np.abs(z_kept_arr)))
    z_max_neg = float(np.max(np.abs(z_kept_arr[neg_mask]))) if np.any(neg_mask) else None
    z_max_pos = float(np.max(z_kept_arr[pos_mask])) if np.any(pos_mask) else None
    return {
        "sigma_min_trusted_calib_px": sigma_min,
        "sigma_max_trusted_calib_px": sigma_max,
        "sigma_max_trusted_neg_calib_px": sigma_max_neg,
        "sigma_max_trusted_pos_calib_px": sigma_max_pos,
        "z_min_trusted_mm": abs_z_min,
        "z_max_trusted_mm": abs_z_max,
        "z_max_trusted_neg_mm": z_max_neg,
        "z_max_trusted_pos_mm": z_max_pos,
    }


def _quadrature_model_func(z, rho, sigma_floor):
    """σ = √((ρ|z|)² + σ_floor²) — least-squares target for quadrature fit."""
    return np.sqrt((rho * np.abs(z)) ** 2 + sigma_floor ** 2)


def calibrate_to_model(
    method: str,
    z_positions: List[float],
    sigma_values: List[float],
    s_calib_px_per_mm: Optional[float] = None,
    exclude_near_focus: float = 0.5,
    source_csv: Optional[str] = None,
):
    """Unified entry point — fit a calibration of the requested method
    and return a ``physics.CalibrationModel`` ready for the pipeline.

    Method dispatch:
    - ``linear``     — σ = ρ|z| + σ₀  (delegates to calibrate_approach_a)
    - ``quadrature`` — σ = √((ρ|z|)² + σ_floor²)
    - ``hybrid``     — quadrature + per-|z| residual LUT

    Trust bounds derived from the post-plateau-filter kept points (53 of
    61 in the user's calibration). Per-side slopes recorded in
    ``fit_metadata.rho_per_side`` for asymmetry diagnostics.
    """
    _ensure_repo_in_syspath()
    from physics import CalibrationModel

    if method not in ("linear", "quadrature", "hybrid"):
        raise ValueError(
            f"Unknown calibration method '{method}'. "
            f"Must be one of: linear, quadrature, hybrid")

    z_kept, sigma_kept, n_near_focus, n_plateau = _filter_for_fit(
        z_positions, sigma_values, exclude_near_focus)
    n_total = int(len(z_positions))
    n_kept = int(len(z_kept))

    if n_kept < 3:
        raise ValueError(
            f"Insufficient calibration points after filtering: {n_kept} kept "
            f"from {n_total} (filtered {n_near_focus} near focus, {n_plateau} "
            f"plateau). Need at least 3 to fit.")

    # Diagnostic: per-side slopes (always recorded)
    rho_per_side = _per_side_slopes(z_kept, sigma_kept)

    fit_metadata_base = {
        "num_points_total": n_total,
        "num_points_kept": n_kept,
        "num_near_focus_excluded": n_near_focus,
        "num_plateau_excluded": n_plateau,
        "rho_per_side": rho_per_side,
        "fit_timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_csv": source_csv,
        # Phase 11: arrays so plot code can color-code kept vs excluded.
        # Lists of floats; not vectorised here for yaml-friendliness.
        "z_kept_mm": [float(zi) for zi in z_kept],
        "sigma_kept_calib_px": [float(si) for si in sigma_kept],
        "exclude_near_focus_mm": float(exclude_near_focus),
    }

    # Method-specific fits
    abs_z_kept = np.abs(z_kept)

    if method == "linear":
        try:
            popt, _ = curve_fit(
                linear_model, z_kept, sigma_kept,
                p0=[1.0, max(np.min(sigma_kept) - 0.5, 0.0)],
                bounds=([0.01, 0], [100, 50]),
            )
            rho, sigma_0 = float(popt[0]), float(popt[1])
            sigma_pred = linear_model(z_kept, rho, sigma_0)
            ss_res = float(np.sum((sigma_kept - sigma_pred) ** 2))
            ss_tot = float(np.sum((sigma_kept - np.mean(sigma_kept)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Linear fit failed: {e}")

        bounds = _bounds_from_fit(z_kept, sigma_kept, None)
        fit_metadata = {**fit_metadata_base, "r_squared": r_squared}
        return CalibrationModel(
            method="linear",
            rho_px_per_mm=rho,
            sigma_0_calib_px=sigma_0,
            s_calib_px_per_mm=s_calib_px_per_mm,
            fit_metadata=fit_metadata,
            **bounds,
        )

    if method == "quadrature":
        # Fit σ² = (ρ|z|)² + σ_floor² as σ = √(...) directly
        try:
            popt, _ = curve_fit(
                _quadrature_model_func, z_kept, sigma_kept,
                p0=[1.0, max(np.min(sigma_kept), 0.1)],
                bounds=([0.01, 0.0], [100, 50]),
            )
            rho, sigma_floor_raw = float(popt[0]), float(popt[1])
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Quadrature fit failed: {e}")

        sigma_floor = sigma_floor_raw
        if sigma_floor < 0:
            print(f"  WARNING: quadrature fit returned negative sigma_floor "
                  f"({sigma_floor_raw}); clamping to 0")
            sigma_floor = 0.0

        sigma_pred = _quadrature_model_func(z_kept, rho, sigma_floor)
        ss_res = float(np.sum((sigma_kept - sigma_pred) ** 2))
        ss_tot = float(np.sum((sigma_kept - np.mean(sigma_kept)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        bounds = _bounds_from_fit(z_kept, sigma_kept, None)
        fit_metadata = {
            **fit_metadata_base,
            "r_squared": r_squared,
            "sigma_floor_raw_px": sigma_floor_raw,
        }
        return CalibrationModel(
            method="quadrature",
            rho_px_per_mm=rho,
            sigma_floor_calib_px=sigma_floor,
            s_calib_px_per_mm=s_calib_px_per_mm,
            fit_metadata=fit_metadata,
            **bounds,
        )

    if method == "hybrid":
        # Step 1: fit quadrature first
        try:
            popt, _ = curve_fit(
                _quadrature_model_func, z_kept, sigma_kept,
                p0=[1.0, max(np.min(sigma_kept), 0.1)],
                bounds=([0.01, 0.0], [100, 50]),
            )
            rho, sigma_floor_raw = float(popt[0]), float(popt[1])
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Hybrid (quadrature step) fit failed: {e}")

        sigma_floor = max(sigma_floor_raw, 0.0)
        sigma_pred = _quadrature_model_func(z_kept, rho, sigma_floor)
        residuals = sigma_kept - sigma_pred  # Δσ per point at calib scale

        # Step 2: build per-|z| residual LUT, averaging ±z at same magnitude.
        # Round abs_z to 4 decimals so e.g. -0.6 and +0.6 collide cleanly.
        rounded_abs_z = np.round(abs_z_kept, 4)
        unique_abs = np.unique(rounded_abs_z)
        lut: List[Tuple[float, float]] = []
        for az in unique_abs:
            mask = rounded_abs_z == az
            d_mean = float(np.mean(residuals[mask]))
            lut.append((float(az), d_mean))
        lut.sort(key=lambda p: p[0])

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((sigma_kept - np.mean(sigma_kept)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        bounds = _bounds_from_fit(z_kept, sigma_kept, None)
        max_abs_residual = float(np.max(np.abs(residuals)))
        fit_metadata = {
            **fit_metadata_base,
            "r_squared": r_squared,
            "sigma_floor_raw_px": sigma_floor_raw,
            "max_abs_residual_px": max_abs_residual,
            "n_lut_points": len(lut),
        }
        return CalibrationModel(
            method="hybrid",
            rho_px_per_mm=rho,
            sigma_floor_calib_px=sigma_floor,
            residual_lut_mm_px=lut,
            s_calib_px_per_mm=s_calib_px_per_mm,
            fit_metadata=fit_metadata,
            **bounds,
        )

    raise RuntimeError(f"Unhandled method branch: {method}")  # defensive
