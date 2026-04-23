"""
Unified Physics Module for Defocus-Blur Scaling

Single source of truth for all forward/inverse conversions between
defocus distance, blur sigma, model-space values, and normalised labels.

Every other module (training, inference, calibration, synthetic generation)
should import from here rather than reimplementing these equations.

Equations (direct mode):
    Forward:  sigma_calib = rho * |z| + sigma_0
    Inverse:  |z| = (sigma_native - sigma_0_eff) / rho_eff

    Cross-camera:  rho_eff   = rho   * (s_inference / s_calib)
                   sigma_0_eff = sigma_0 * (s_inference / s_calib)

    Model scaling:  sigma_model  = sigma_native * (model_size / native_size)
                    sigma_native = sigma_model  * (native_size / model_size)

    Normalisation:  label = clamp(sigma_model / max_blur, 0, 1)
                    sigma_model = label * max_blur
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


SATURATION_THRESHOLD = 0.01  # predictions within this of 0 or 1 are flagged

# Conservative fallback for the defocus window when a checkpoint doesn't carry
# enough info to derive its own valid range (see ``calib_valid_defocus_mm``).
# Tuned for Camera G but intentionally wide; prefer the derived range.
CALIB_VALID_DEFOCUS_MM_FALLBACK = (0.5, 8.0)


def calib_valid_defocus_mm(config: Dict) -> Tuple[float, float]:
    """Defocus-magnitude window over which a pred-vs-truth fit is reliable.

    Derived from the model's actual training blur range via the direct-mode
    calibration:
        z_min = max(0, (blur_min - σ₀) / ρ)
        z_max = (blur_max - σ₀) / ρ

    ``config`` is the dict stored in ``checkpoint['config']`` (same shape as
    ``generation_config.yaml`` / ``training_config.yaml``). If the keys needed
    for the derivation are missing, falls back to
    ``CALIB_VALID_DEFOCUS_MM_FALLBACK``.
    """
    data = config.get('data', {}) if config else {}
    training = config.get('training', {}) if config else {}
    blur_range = data.get('blur_range_px')
    rho = training.get('rho_direct')
    if not blur_range or rho is None or rho <= 0:
        return CALIB_VALID_DEFOCUS_MM_FALLBACK
    sigma_0 = training.get('sigma_0') or 0.0
    blur_min, blur_max = float(blur_range[0]), float(blur_range[1])
    z_min = max(0.0, (blur_min - float(sigma_0)) / float(rho))
    z_max = (blur_max - float(sigma_0)) / float(rho)
    return (z_min, z_max)


@dataclass
class ScalingParams:
    """All parameters needed for the forward/inverse blur chain."""
    rho: float                          # calibration slope (px/mm)
    sigma_0: float = 0.0               # residual blur at focus (px)
    s_calib: float = 1.0               # calibration camera scale (px/mm)
    s_inference: float = 1.0           # inference camera scale (px/mm)
    max_blur: float = 20.0             # max blur for label normalisation (px)
    model_size: int = 256              # network input resolution (px)

    @property
    def scale_ratio(self) -> float:
        """Cross-camera scale factor: s_inference / s_calib."""
        if self.s_calib > 0:
            return self.s_inference / self.s_calib
        return 1.0

    @property
    def rho_eff(self) -> float:
        """Effective rho at inference camera scale."""
        return self.rho * self.scale_ratio

    @property
    def sigma_0_eff(self) -> float:
        """Effective sigma_0 at inference camera scale."""
        return self.sigma_0 * self.scale_ratio


# ── Forward: defocus → blur ──────────────────────────────────────────────

def defocus_to_sigma_calib(z_mm: float, rho: float, sigma_0: float = 0.0) -> float:
    """Convert defocus to blur at calibration camera scale.

    sigma_calib = rho * |z| + sigma_0
    """
    return rho * abs(z_mm) + sigma_0


def sigma_calib_to_native(
    sigma_calib: float,
    s_inference: float,
    s_calib: float,
) -> float:
    """Scale sigma from calibration camera to inference camera.

    sigma_native = sigma_calib * (s_inference / s_calib)
    """
    if s_calib > 0:
        return sigma_calib * (s_inference / s_calib)
    return sigma_calib


def sigma_native_to_model(
    sigma_native: float,
    model_size: int,
    native_size: int,
) -> float:
    """Scale sigma from native crop resolution to model input resolution.

    sigma_model = sigma_native * (model_size / native_size)
    """
    if native_size > 0:
        return sigma_native * (model_size / native_size)
    return sigma_native


def normalise_label(sigma_model: float, max_blur: float) -> float:
    """Normalise sigma_model to [0, 1] for training labels.

    label = clamp(sigma_model / max_blur, 0, 1)
    """
    if max_blur <= 0:
        return 0.0
    return max(0.0, min(sigma_model / max_blur, 1.0))


def defocus_to_label(
    z_mm: float,
    params: ScalingParams,
    native_size: int,
) -> float:
    """Full forward chain: defocus (mm) → normalised label [0, 1].

    1. sigma_calib = rho * |z| + sigma_0
    2. sigma_native = sigma_calib * scale_ratio
    3. sigma_model = sigma_native * (model_size / native_size)
    4. label = clamp(sigma_model / max_blur, 0, 1)
    """
    sigma_calib = defocus_to_sigma_calib(z_mm, params.rho, params.sigma_0)
    sigma_native = sigma_calib_to_native(sigma_calib, params.s_inference, params.s_calib)
    sigma_model = sigma_native_to_model(sigma_native, params.model_size, native_size)
    return normalise_label(sigma_model, params.max_blur)


# ── Inverse: model output → defocus ──────────────────────────────────────

def denormalise_label(label: float, max_blur: float) -> float:
    """Convert normalised prediction [0, 1] back to sigma at model scale.

    sigma_model = label * max_blur
    """
    return label * max_blur


def sigma_model_to_native(
    sigma_model: float,
    model_size: int,
    native_size: int,
) -> float:
    """Scale sigma from model resolution to native crop resolution.

    sigma_native = sigma_model * (native_size / model_size)
    """
    if model_size > 0:
        return sigma_model * (native_size / model_size)
    return sigma_model


def sigma_native_to_defocus(
    sigma_native: float,
    rho_eff: float,
    sigma_0_eff: float = 0.0,
) -> Tuple[float, bool]:
    """Invert blur to defocus distance.

    |z| = (sigma_native - sigma_0_eff) / rho_eff

    Returns (defocus_mm, was_clamped).
    Clamps negative values to 0 and reports via the flag.
    """
    if rho_eff <= 0:
        return 0.0, False
    raw = (sigma_native - sigma_0_eff) / rho_eff
    clamped = raw < 0.0
    return max(0.0, raw), clamped


def label_to_defocus(
    label: float,
    params: ScalingParams,
    native_size: int,
) -> Tuple[float, bool, bool]:
    """Full inverse chain: normalised prediction → defocus (mm).

    1. sigma_model = label * max_blur
    2. sigma_native = sigma_model * (native_size / model_size)
    3. |z| = (sigma_native - sigma_0_eff) / rho_eff

    Returns (defocus_mm, was_clamped, is_saturated).
    """
    sigma_model = denormalise_label(label, params.max_blur)
    sigma_native = sigma_model_to_native(sigma_model, params.model_size, native_size)
    defocus_mm, clamped = sigma_native_to_defocus(
        sigma_native, params.rho_eff, params.sigma_0_eff
    )
    saturated = label < SATURATION_THRESHOLD or label > (1.0 - SATURATION_THRESHOLD)
    return defocus_mm, clamped, saturated


# ── Convenience: full chain results ──────────────────────────────────────

@dataclass
class InversionResult:
    """Complete result from the inverse chain."""
    pred_norm: float
    sigma_model: float
    sigma_native: float
    defocus_mm: float
    saturated: bool
    clamped: bool


def invert_prediction(
    pred_norm: float,
    params: ScalingParams,
    native_size: int,
) -> InversionResult:
    """Run the full inverse chain and return all intermediates.

    This is the canonical function that inference code should call.
    """
    sigma_model = denormalise_label(pred_norm, params.max_blur)
    sigma_native = sigma_model_to_native(sigma_model, params.model_size, native_size)
    defocus_mm, clamped = sigma_native_to_defocus(
        sigma_native, params.rho_eff, params.sigma_0_eff
    )
    saturated = pred_norm < 0.01 or pred_norm > 0.99

    return InversionResult(
        pred_norm=pred_norm,
        sigma_model=sigma_model,
        sigma_native=sigma_native,
        defocus_mm=defocus_mm,
        saturated=saturated,
        clamped=clamped,
    )


# ══════════════════════════════════════════════════════════════════════════
# Config Validation
# ══════════════════════════════════════════════════════════════════════════

class ConfigError(Exception):
    """Raised when training/inference config is missing or incoherent."""
    pass


def validate_training_config(config: Dict, training_mode: str = "direct") -> List[str]:
    """Validate a training config dict before training starts.

    Raises ConfigError if critical keys are missing.
    Returns list of warnings for non-critical issues.
    """
    errors = []
    warnings = []

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    # Required for all modes
    if not data_cfg.get("blur_range_px"):
        errors.append("data.blur_range_px is missing — needed for label normalisation")

    image_size = data_cfg.get("image_size_px")
    if not image_size or image_size < 32:
        errors.append(f"data.image_size_px is missing or too small ({image_size})")

    # Direct mode requirements
    if training_mode == "direct":
        rho = training_cfg.get("rho_direct")
        if rho is None:
            errors.append("training.rho_direct is missing — required for direct mode")
        elif rho <= 0:
            errors.append(f"training.rho_direct must be positive, got {rho}")

        scale_calib = training_cfg.get("scale_calib_px_per_mm")
        if scale_calib is None:
            warnings.append(
                "training.scale_calib_px_per_mm missing — cross-camera correction disabled")
        elif scale_calib <= 0:
            errors.append(f"training.scale_calib_px_per_mm must be positive, got {scale_calib}")

    # Optical mode requirements
    if training_mode == "optical":
        optics = config.get("optics", {})
        for key in ("focal_length_mm", "aperture_diameter_mm", "focus_distance_mm"):
            if key not in optics:
                errors.append(f"optics.{key} is missing — required for optical mode")

    if errors:
        raise ConfigError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )

    return warnings


def validate_inference_config(
    rho: float,
    sigma_0: float,
    s_calib: float,
    s_c: float,
    max_blur: float,
    model_size: int,
    crop_size: int,
) -> List[str]:
    """Validate inference parameters before processing.

    Returns list of issues (empty = all good).
    """
    issues = []

    if rho <= 0:
        issues.append(f"rho must be positive (got {rho})")
    if sigma_0 < 0:
        issues.append(f"sigma_0 cannot be negative (got {sigma_0})")
    if s_calib <= 0:
        issues.append(f"s_calib must be positive (got {s_calib})")
    if s_c <= 0:
        issues.append(f"s_c must be positive (got {s_c})")
    if max_blur <= 0:
        issues.append(f"max_blur must be positive (got {max_blur})")
    if model_size < 32:
        issues.append(f"model_size too small (got {model_size})")
    if crop_size < 32:
        issues.append(f"crop_size too small (got {crop_size})")
    if crop_size != model_size:
        issues.append(
            f"crop_size ({crop_size}) != model training size ({model_size}) "
            f"— rescaling will occur"
        )

    # Plausibility checks
    if rho > 50:
        issues.append(f"rho={rho} px/mm is unusually large — check calibration")
    if sigma_0 > 10:
        issues.append(f"sigma_0={sigma_0} px is unusually large — check calibration")

    scale_ratio = s_c / s_calib if s_calib > 0 else 1.0
    if scale_ratio > 5 or scale_ratio < 0.2:
        issues.append(
            f"Camera scale ratio {scale_ratio:.2f} (s_c/s_calib = {s_c}/{s_calib}) "
            f"is extreme — check values"
        )

    return issues


# ══════════════════════════════════════════════════════════════════════════
# Uncertainty Propagation
# ══════════════════════════════════════════════════════════════════════════

def defocus_uncertainty(
    sigma_native: float,
    rho: float,
    sigma_0: float,
    rho_std: float,
    sigma_0_std: float,
) -> float:
    """Propagate calibration uncertainty to defocus estimate.

    Given z = (sigma_native - sigma_0) / rho, the uncertainty is:

        delta_z = sqrt(
            (sigma_0_std / rho)^2 +
            ((sigma_native - sigma_0) * rho_std / rho^2)^2
        )

    Args:
        sigma_native: blur at native scale (px)
        rho: calibration slope (px/mm)
        sigma_0: residual blur (px)
        rho_std: uncertainty in rho from LOO-CV
        sigma_0_std: uncertainty in sigma_0 from LOO-CV

    Returns:
        Uncertainty in defocus (mm), one standard deviation.
    """
    if rho <= 0:
        return 0.0

    # Partial derivatives of z = (sigma - sigma_0) / rho
    # dz/d(sigma_0) = -1/rho
    # dz/d(rho) = -(sigma - sigma_0) / rho^2
    z = (sigma_native - sigma_0) / rho

    term_sigma_0 = (sigma_0_std / rho) ** 2
    term_rho = (z * rho_std / rho) ** 2

    return math.sqrt(term_sigma_0 + term_rho)


@dataclass
class InversionResultWithUncertainty(InversionResult):
    """Inversion result extended with confidence interval."""
    defocus_uncertainty_mm: float = 0.0


def invert_prediction_with_uncertainty(
    pred_norm: float,
    params: ScalingParams,
    native_size: int,
    rho_std: float = 0.0,
    sigma_0_std: float = 0.0,
) -> InversionResultWithUncertainty:
    """Run inverse chain with uncertainty propagation.

    Like invert_prediction(), but also computes defocus confidence interval
    from LOO-CV uncertainties in rho and sigma_0.
    """
    base = invert_prediction(pred_norm, params, native_size)

    # Scale uncertainties by cross-camera ratio (same as rho/sigma_0 themselves)
    rho_std_eff = rho_std * params.scale_ratio
    sigma_0_std_eff = sigma_0_std * params.scale_ratio

    unc = defocus_uncertainty(
        base.sigma_native,
        params.rho_eff,
        params.sigma_0_eff,
        rho_std_eff,
        sigma_0_std_eff,
    )

    return InversionResultWithUncertainty(
        pred_norm=base.pred_norm,
        sigma_model=base.sigma_model,
        sigma_native=base.sigma_native,
        defocus_mm=base.defocus_mm,
        saturated=base.saturated,
        clamped=base.clamped,
        defocus_uncertainty_mm=unc,
    )
