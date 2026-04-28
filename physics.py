"""
Unified Physics Module for Defocus-Blur Scaling

Single source of truth for all forward/inverse conversions between
defocus distance, blur sigma, model-space values, and normalised labels.

Every other module (training, inference, calibration, synthetic generation)
should import from here rather than reimplementing these equations.

PRIMARY API: ``CalibrationModel`` class (defined at the bottom of this
module). Supports three methods for the σ↔z relationship:

    linear:      sigma = rho * |z| + sigma_0
    quadrature:  sigma = sqrt((rho * |z|)^2 + sigma_floor^2)
    hybrid:      quadrature + per-|z| residual LUT

The chosen method is recorded at calibration time and propagates through
synthetic generation -> training checkpoint -> inference. Use
``CalibrationModel.forward()`` / ``inverse()`` for calibration-pixel-space
math and ``forward_at()`` / ``inverse_at()`` for cross-camera +
model-resolution scaling. ``defocus_uncertainty()`` propagates LOO-CV
parameter std through the right Jacobian per method.

LEGACY API (kept for back-compat): the standalone functions below
(``invert_prediction``, ``defocus_to_label``, ``ScalingParams``,
``defocus_uncertainty``) are linear-formula shims that delegate to a
``CalibrationModel(method='linear')`` internally. New code should use
the class API directly.

Cross-camera and resolution scaling (applied by forward_at/inverse_at):

    Cross-camera:  sigma_inf_native = sigma_calib * (s_inference / s_calib)
    Model scaling: sigma_model      = sigma_inf_native * (model_size / native_size)
    Normalisation: label            = clamp(sigma_model / max_blur, 0, 1)
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SATURATION_THRESHOLD = 0.01  # predictions within this of 0 or 1 are flagged

# Conservative fallback for the defocus window when a checkpoint doesn't carry
# enough info to derive its own valid range (see ``calib_valid_defocus_mm``).
# Tuned for Camera G but intentionally wide; prefer the derived range.
CALIB_VALID_DEFOCUS_MM_FALLBACK = (0.5, 8.0)


def calib_valid_defocus_mm(config: Dict) -> Tuple[float, float]:
    """Defocus-magnitude window over which a pred-vs-truth fit is reliable.

    Method-aware: when ``config['training']['calibration_model']`` is
    present, derives the window via ``CalibrationModel.inverse()`` for
    the chosen method. Otherwise falls back to the legacy linear math:

        z_min = max(0, (blur_min - σ₀) / ρ)
        z_max = (blur_max - σ₀) / ρ

    Linear and quadrature give similar windows in the linear regime;
    they diverge near focus (quadrature handles the focus floor more
    correctly).

    ``config`` is the dict stored in ``checkpoint['config']`` (same shape as
    ``generation_config.yaml`` / ``training_config.yaml``). If the keys needed
    for the derivation are missing, falls back to
    ``CALIB_VALID_DEFOCUS_MM_FALLBACK``.
    """
    data = config.get('data', {}) if config else {}
    training = config.get('training', {}) if config else {}
    blur_range = data.get('blur_range_px')

    # Method-aware path: build CalibrationModel and use its inverse
    cm_dict = training.get('calibration_model')
    if isinstance(cm_dict, dict) and cm_dict and blur_range:
        try:
            cm = CalibrationModel.from_dict(cm_dict)
            blur_min, blur_max = float(blur_range[0]), float(blur_range[1])
            # blur_range is in MODEL pixels — convert to calib pixels
            # via the dataset's image_size (assume native==model when
            # only blur_range_px is available — same convention as
            # the legacy linear path below)
            z_min, _ = cm.inverse(blur_min)
            z_max, _ = cm.inverse(blur_max)
            # NaN can come back if SATURATED (blur_max above trust ceiling);
            # in that case fall through to legacy bound
            if z_min == z_min and z_max == z_max:
                return (max(0.0, float(z_min)), float(z_max))
        except Exception:
            pass  # fall through to legacy path

    # Legacy path: linear formula via rho_direct / sigma_0
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


def _legacy_linear_model(params: ScalingParams) -> "CalibrationModel":
    """Build an internal `CalibrationModel(method='linear', ...)` from a
    legacy `ScalingParams`. Trust bounds disabled (inf / 0) so callers
    get the original "no flagging, just compute" behaviour.

    Used by the Phase 2 shims so all legacy callers route through the
    canonical CalibrationModel without behaviour change.
    """
    return CalibrationModel(
        method="linear",
        rho_px_per_mm=params.rho,
        sigma_0_calib_px=params.sigma_0,
        s_calib_px_per_mm=params.s_calib,
        sigma_min_trusted_calib_px=0.0,
        sigma_max_trusted_calib_px=float("inf"),
    )


def defocus_to_label(
    z_mm: float,
    params: ScalingParams,
    native_size: int,
) -> float:
    """Full forward chain: defocus (mm) → normalised label [0, 1].

    Delegates to ``CalibrationModel.forward_at`` for the σ chain;
    legacy ``ScalingParams.max_blur`` still drives normalisation.

    1. sigma_calib = rho * |z| + sigma_0           (linear forward)
    2. sigma_native = sigma_calib * scale_ratio
    3. sigma_model = sigma_native * (model_size / native_size)
    4. label = clamp(sigma_model / max_blur, 0, 1)
    """
    model = _legacy_linear_model(params)
    sigma_model = model.forward_at(
        abs_z_mm=z_mm,
        s_inf_px_per_mm=params.s_inference,
        model_size=params.model_size,
        native_size=native_size,
    )
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

    Delegates to ``CalibrationModel.inverse_at``. The ``clamped`` flag
    (raw |z| would have been negative) and ``saturated`` flag (label
    near sigmoid extremes) preserve the legacy semantics — separate
    from the new ``BoundsFlag`` which is method-aware.

    Returns (defocus_mm, was_clamped, is_saturated).
    """
    model = _legacy_linear_model(params)
    sigma_model = denormalise_label(label, params.max_blur)
    # Track clamped: would the raw (sigma - sigma_0)/rho have been negative?
    sigma_native = sigma_model_to_native(sigma_model, params.model_size, native_size)
    raw_z = (sigma_native - params.sigma_0_eff) / params.rho_eff if params.rho_eff > 0 else 0.0
    clamped = raw_z < 0.0
    # Defocus via the canonical inverse (which clamps to 0 internally)
    defocus_mm, _flag = model.inverse_at(
        sigma_model_px=sigma_model,
        s_inf_px_per_mm=params.s_inference,
        model_size=params.model_size,
        native_size=native_size,
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

    Delegates to ``CalibrationModel.inverse_at``; the legacy
    ``saturated``/``clamped`` flags preserve their original semantics
    (sigmoid edge / negative raw |z|), independent of the new
    ``BoundsFlag``.

    This is the canonical function that legacy inference code calls.
    """
    model = _legacy_linear_model(params)
    sigma_model = denormalise_label(pred_norm, params.max_blur)
    sigma_native = sigma_model_to_native(sigma_model, params.model_size, native_size)
    # Track clamped per legacy semantics (raw (sigma - sigma_0_eff)/rho_eff < 0)
    raw_z = (sigma_native - params.sigma_0_eff) / params.rho_eff if params.rho_eff > 0 else 0.0
    clamped = raw_z < 0.0
    # Defocus via canonical inverse (clamps to 0 internally)
    defocus_mm, _flag = model.inverse_at(
        sigma_model_px=sigma_model,
        s_inf_px_per_mm=params.s_inference,
        model_size=params.model_size,
        native_size=native_size,
    )
    saturated = (pred_norm < SATURATION_THRESHOLD
                 or pred_norm > (1.0 - SATURATION_THRESHOLD))

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


# ══════════════════════════════════════════════════════════════════════════
# CalibrationModel — single source of truth for σ ↔ z calibration
# (Phase 1 of the calibration-method overhaul. Co-exists with the legacy
# ScalingParams/invert_prediction; Phase 2 makes those delegate here.)
# ══════════════════════════════════════════════════════════════════════════


class BoundsFlag(str, Enum):
    """Inversion-time trust classification of a prediction.

    IN_RANGE     — σ is within calibrated bounds; |z| is trustworthy.
    BELOW_FLOOR  — σ ≤ σ_min_trusted (or ≤ σ_floor for quadrature/hybrid);
                    |z| returned as 0 (at or below resolution limit).
    SATURATED    — σ ≥ min(σ_max_trusted_calib, σ_max_model_observed);
                    |z| returned as nan (no honest answer possible).
    """
    IN_RANGE = "IN_RANGE"
    BELOW_FLOOR = "BELOW_FLOOR"
    SATURATED = "SATURATED"


_KNOWN_METHODS = ("linear", "quadrature", "hybrid")


@dataclass
class CalibrationModel:
    """Source-of-truth calibration model for σ ↔ z conversions.

    All σ values and σ-derived parameters live in CALIBRATION-NATIVE pixel
    space. Cross-camera (s_inf/s_calib) and model-resolution
    (model_size/native_size) conversions are handled by the
    ``forward_at`` / ``inverse_at`` boundary wrappers; never by callers
    directly.

    Three methods supported:

    - ``linear``     σ = ρ·|z| + σ₀
    - ``quadrature`` σ = √((ρ·|z|)² + σ_floor²)
    - ``hybrid``     σ = quadrature(z) + Δσ(|z|)  with residual LUT

    Bounds:

    - ``sigma_min_trusted_calib_px`` / ``sigma_max_trusted_calib_px`` —
      the calibration's filtered σ range. Predictions outside flag
      BELOW_FLOOR / SATURATED.
    - ``sigma_max_model_observed_px`` — empirical max σ the trained
      model produces on the calibration stack. Used for the SATURATED
      threshold as ``min(sigma_max_trusted, sigma_max_model_observed)``;
      catches the case where the model's plateau is below the
      calibration's trust ceiling.
    """

    method: str
    rho_px_per_mm: float

    # Method-specific parameters (only the relevant one is used per method)
    sigma_0_calib_px: Optional[float] = None      # linear only
    sigma_floor_calib_px: Optional[float] = None  # quadrature / hybrid

    # Hybrid-only: per-|z| residuals, sorted by |z|
    residual_lut_mm_px: Optional[List[Tuple[float, float]]] = None

    # Trust bounds
    sigma_min_trusted_calib_px: float = 0.0
    sigma_max_trusted_calib_px: float = float("inf")
    sigma_max_model_observed_px: Optional[float] = None
    z_min_trusted_mm: float = 0.0
    z_max_trusted_mm: float = float("inf")
    z_max_trusted_neg_mm: Optional[float] = None  # asymmetric (diagnostic)
    z_max_trusted_pos_mm: Optional[float] = None

    # Optical scale (used by forward_at / inverse_at for cross-camera math)
    s_calib_px_per_mm: Optional[float] = None

    # Phase 10: method-aware LOO-CV uncertainty (rho_std + aux_param_std).
    # Populated by calibration's loo_cv_for_method(). Used by
    # `defocus_uncertainty()` to propagate per-prediction error bars.
    # Schema:
    #   {'rho_std': float, 'aux_param_name': 'sigma_0' | 'sigma_floor',
    #    'aux_param_std': float, 'loo_mae': float, 'num_folds': int}
    loo_cv: Optional[Dict[str, Any]] = None

    # Provenance / diagnostic info, not used by math
    fit_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Validation ────────────────────────────────────────────────────────
    def __post_init__(self):
        if self.method not in _KNOWN_METHODS:
            raise ValueError(
                f"Unknown calibration method '{self.method}'. "
                f"Must be one of {_KNOWN_METHODS}.")
        if self.rho_px_per_mm <= 0:
            raise ValueError(
                f"rho_px_per_mm must be positive, got {self.rho_px_per_mm}")
        if self.method == "linear":
            if self.sigma_0_calib_px is None:
                raise ValueError("linear method requires sigma_0_calib_px")
        if self.method in ("quadrature", "hybrid"):
            if self.sigma_floor_calib_px is None:
                raise ValueError(
                    f"{self.method} method requires sigma_floor_calib_px")
            if self.sigma_floor_calib_px < 0:
                raise ValueError(
                    f"sigma_floor_calib_px must be non-negative, "
                    f"got {self.sigma_floor_calib_px}")
        if self.method == "hybrid":
            if not self.residual_lut_mm_px:
                raise ValueError(
                    "hybrid method requires non-empty residual_lut_mm_px")
        # Sort LUT by |z| for interpolation (stable input order)
        if self.residual_lut_mm_px:
            self.residual_lut_mm_px = sorted(
                [(float(z), float(d)) for z, d in self.residual_lut_mm_px],
                key=lambda p: p[0])

    # ── Internal: residual interpolation ─────────────────────────────────
    def _residual_at(self, abs_z_mm: float) -> float:
        """Linear-interpolate residual at |z|. Clamps to LUT endpoints
        for out-of-range queries."""
        if not self.residual_lut_mm_px:
            return 0.0
        z = abs(abs_z_mm)
        zs = [p[0] for p in self.residual_lut_mm_px]
        ds = [p[1] for p in self.residual_lut_mm_px]
        if z <= zs[0]:
            return ds[0]
        if z >= zs[-1]:
            return ds[-1]
        for i in range(len(zs) - 1):
            if zs[i] <= z <= zs[i + 1]:
                span = zs[i + 1] - zs[i]
                if span <= 0:
                    return ds[i]
                t = (z - zs[i]) / span
                return ds[i] + t * (ds[i + 1] - ds[i])
        return 0.0  # unreachable

    # ── Forward: |z|_mm → σ_calib_px ──────────────────────────────────────
    def forward(self, abs_z_mm: float) -> float:
        """Compute σ at calibration native pixel scale for a given |z| in mm.

        Always operates on |z| (input is treated as |abs_z_mm|).
        """
        z = abs(float(abs_z_mm))
        if self.method == "linear":
            return self.rho_px_per_mm * z + float(self.sigma_0_calib_px)
        if self.method == "quadrature":
            floor = float(self.sigma_floor_calib_px)
            return math.sqrt((self.rho_px_per_mm * z) ** 2 + floor ** 2)
        if self.method == "hybrid":
            floor = float(self.sigma_floor_calib_px)
            base = math.sqrt((self.rho_px_per_mm * z) ** 2 + floor ** 2)
            return base + self._residual_at(z)
        raise RuntimeError(f"Unhandled method: {self.method}")  # defensive

    # ── Inverse: σ_calib_px → (|z|_mm, BoundsFlag) ───────────────────────
    def _saturation_threshold(self) -> float:
        """min(sigma_max_trusted_calib_px, sigma_max_model_observed_px).

        SATURATED fires when σ_pred is at or above this value.
        """
        thr = self.sigma_max_trusted_calib_px
        if self.sigma_max_model_observed_px is not None:
            thr = min(thr, self.sigma_max_model_observed_px)
        return thr

    def inverse(self, sigma_calib_px: float) -> Tuple[float, BoundsFlag]:
        """Invert σ at calib scale → (|z|_mm, BoundsFlag).

        SATURATED  → returns (nan, SATURATED)
        BELOW_FLOOR → returns (0.0, BELOW_FLOOR)
        IN_RANGE    → returns (|z|, IN_RANGE)
        """
        sigma = float(sigma_calib_px)

        # Saturation check first — overrides everything else
        if sigma >= self._saturation_threshold():
            return float("nan"), BoundsFlag.SATURATED

        # Method-specific inversion
        if self.method == "linear":
            z = (sigma - float(self.sigma_0_calib_px)) / self.rho_px_per_mm
            z = max(z, 0.0)
        elif self.method == "quadrature":
            floor = float(self.sigma_floor_calib_px)
            if sigma <= floor:
                return 0.0, BoundsFlag.BELOW_FLOOR
            inner = sigma * sigma - floor * floor
            z = math.sqrt(max(inner, 0.0)) / self.rho_px_per_mm
        elif self.method == "hybrid":
            floor = float(self.sigma_floor_calib_px)
            # Initial guess via quadrature inverse
            if sigma <= floor:
                z_init = 0.0
            else:
                inner = sigma * sigma - floor * floor
                z_init = math.sqrt(max(inner, 0.0)) / self.rho_px_per_mm
            z = z_init
            converged = False
            for _ in range(30):
                f = self.forward(z) - sigma
                if abs(f) < 1e-6:
                    converged = True
                    break
                # Numerical derivative — handles residual LUT smoothly
                eps = max(1e-4, abs(z) * 1e-4)
                df = (self.forward(z + eps) - self.forward(max(z - eps, 0.0))) / (2.0 * eps)
                if abs(df) < 1e-12:
                    break
                step = f / df
                z_new = z - step
                # Divergence guard
                if not math.isfinite(z_new) or abs(z_new - z) > self.z_max_trusted_mm + 5.0:
                    break
                z = max(z_new, 0.0)
            if not converged:
                # Rate-limit: warn once per process, then count silently.
                # Newton struggles at the LUT endpoint (derivative
                # discontinuity); fallback is mathematically valid.
                if not getattr(CalibrationModel, '_newton_warn_emitted', False):
                    logger.warning(
                        "CalibrationModel.inverse (hybrid): Newton did not "
                        "converge for σ=%.4f; falling back to quadrature root. "
                        "(Subsequent occurrences silenced — common at LUT "
                        "endpoint, math fallback is still valid.)", sigma)
                    CalibrationModel._newton_warn_emitted = True
                # Track count for diagnostics
                CalibrationModel._newton_fallback_count = getattr(
                    CalibrationModel, '_newton_fallback_count', 0) + 1
                z = z_init
        else:
            raise RuntimeError(f"Unhandled method: {self.method}")

        # Below-floor flag — applies after we have a |z| estimate
        if sigma < self.sigma_min_trusted_calib_px:
            return 0.0, BoundsFlag.BELOW_FLOOR
        return z, BoundsFlag.IN_RANGE

    # ── Boundary wrappers: cross-camera + model-resolution scaling ───────
    def _scale_factor_inf_to_calib(self, s_inf_px_per_mm: Optional[float]) -> float:
        """sigma_native = sigma_calib * (s_inf / s_calib).
        Returns the scale factor; defaults to 1.0 when scales aren't
        configured (assumes inference camera == calibration camera).
        """
        if (self.s_calib_px_per_mm is None or self.s_calib_px_per_mm <= 0
                or s_inf_px_per_mm is None or s_inf_px_per_mm <= 0):
            return 1.0
        return float(s_inf_px_per_mm) / float(self.s_calib_px_per_mm)

    def forward_at(
        self,
        abs_z_mm: float,
        s_inf_px_per_mm: Optional[float],
        model_size: int,
        native_size: int,
    ) -> float:
        """|z|_mm → σ_model_px. Applies cross-camera + resolution scaling.

        ``model_size``: training/inference model resolution (px).
        ``native_size``: source image native resolution (px).
        ``s_inf_px_per_mm``: inference camera scale; if None or
        ``s_calib_px_per_mm`` is unset, no cross-camera scaling.
        """
        sigma_calib = self.forward(abs_z_mm)
        sigma_inf_native = sigma_calib * self._scale_factor_inf_to_calib(s_inf_px_per_mm)
        if native_size <= 0:
            raise ValueError(f"native_size must be positive, got {native_size}")
        return sigma_inf_native * (float(model_size) / float(native_size))

    def inverse_at(
        self,
        sigma_model_px: float,
        s_inf_px_per_mm: Optional[float],
        model_size: int,
        native_size: int,
    ) -> Tuple[float, BoundsFlag]:
        """σ_model_px → (|z|_mm, BoundsFlag). Reverses forward_at."""
        if model_size <= 0:
            raise ValueError(f"model_size must be positive, got {model_size}")
        sigma_inf_native = float(sigma_model_px) * (float(native_size) / float(model_size))
        cc = self._scale_factor_inf_to_calib(s_inf_px_per_mm)
        # σ_calib = σ_inf_native / cc  (since σ_inf_native = σ_calib · cc)
        sigma_calib = sigma_inf_native / cc if cc != 0 else sigma_inf_native
        return self.inverse(sigma_calib)

    # ── Per-method uncertainty propagation ───────────────────────────────
    def defocus_uncertainty(
        self,
        sigma_calib_px: float,
        rho_std_override: Optional[float] = None,
        aux_param_std_override: Optional[float] = None,
    ) -> float:
        """Propagate parameter uncertainty (from LOO-CV) to defocus mm.

        Method-aware Jacobians:

        - linear:     dz/dρ = -|z|/ρ,   dz/dσ_0     = -1/ρ
        - quadrature: dz/dρ = -|z|/ρ,   dz/dσ_floor = -σ_floor/(ρ·f)
                      where f = √(σ² - σ_floor²) and |z| = f/ρ
        - hybrid:     numerical via central finite differences on inverse()

        Returns 0.0 when LOO data isn't attached or the prediction is
        outside trust (BELOW_FLOOR / SATURATED — uncertainty isn't
        meaningful there).
        """
        # Determine std inputs (allow override for testing / cross-camera)
        loo = self.loo_cv or {}
        rho_std = (rho_std_override if rho_std_override is not None
                    else float(loo.get("rho_std", 0.0)))
        aux_std = (aux_param_std_override if aux_param_std_override is not None
                    else float(loo.get("aux_param_std", 0.0)))
        if rho_std <= 0.0 and aux_std <= 0.0:
            return 0.0
        # SATURATED → no honest uncertainty
        sat_threshold = self._saturation_threshold()
        if sigma_calib_px >= sat_threshold:
            return 0.0
        rho = self.rho_px_per_mm
        if rho <= 0.0:
            return 0.0

        if self.method == "linear":
            sigma_0 = float(self.sigma_0_calib_px or 0.0)
            z = max((sigma_calib_px - sigma_0) / rho, 0.0)
            term_aux = (aux_std / rho) ** 2
            term_rho = (z * rho_std / rho) ** 2
            return math.sqrt(term_aux + term_rho)

        if self.method == "quadrature":
            floor = float(self.sigma_floor_calib_px or 0.0)
            if sigma_calib_px <= floor:
                return 0.0
            f = math.sqrt(max(sigma_calib_px ** 2 - floor ** 2, 0.0))
            if f <= 0.0:
                return 0.0
            z = f / rho
            # ∂z/∂σ_floor = -σ_floor / (ρ · f)
            d_aux = floor / (rho * f) if f > 0 else 0.0
            term_aux = (aux_std * d_aux) ** 2
            term_rho = (z * rho_std / rho) ** 2
            return math.sqrt(term_aux + term_rho)

        if self.method == "hybrid":
            # Numerical Jacobian via central finite differences on inverse().
            # We perturb the params on a temporary copy, re-invert, take
            # the difference. Cheap (~3 inverses).
            z_center, _ = self.inverse(sigma_calib_px)
            if z_center != z_center:  # nan = saturated
                return 0.0
            from copy import copy as _copy
            eps_rho = max(1e-6, abs(rho) * 1e-3) if rho_std > 0 else 0.0
            eps_aux = max(1e-6, abs(self.sigma_floor_calib_px or 0.0) * 1e-3) if aux_std > 0 else 0.0
            d_z_d_rho = 0.0
            d_z_d_aux = 0.0
            if eps_rho > 0:
                m_lo = _copy(self)
                m_hi = _copy(self)
                m_lo.rho_px_per_mm = rho - eps_rho
                m_hi.rho_px_per_mm = rho + eps_rho
                z_lo, _ = m_lo.inverse(sigma_calib_px)
                z_hi, _ = m_hi.inverse(sigma_calib_px)
                if (z_lo == z_lo) and (z_hi == z_hi):
                    d_z_d_rho = (z_hi - z_lo) / (2.0 * eps_rho)
            if eps_aux > 0:
                m_lo = _copy(self)
                m_hi = _copy(self)
                floor = float(self.sigma_floor_calib_px or 0.0)
                m_lo.sigma_floor_calib_px = floor - eps_aux
                m_hi.sigma_floor_calib_px = floor + eps_aux
                z_lo, _ = m_lo.inverse(sigma_calib_px)
                z_hi, _ = m_hi.inverse(sigma_calib_px)
                if (z_lo == z_lo) and (z_hi == z_hi):
                    d_z_d_aux = (z_hi - z_lo) / (2.0 * eps_aux)
            term_rho = (rho_std * d_z_d_rho) ** 2
            term_aux = (aux_std * d_z_d_aux) ** 2
            return math.sqrt(term_rho + term_aux)

        return 0.0  # unknown method — no uncertainty

    def defocus_uncertainty_at(
        self,
        sigma_model_px: float,
        s_inf_px_per_mm: Optional[float],
        model_size: int,
        native_size: int,
    ) -> float:
        """Boundary wrapper: compute uncertainty given σ in model pixels.

        Mirrors ``inverse_at``: converts σ_model → σ_calib, then calls
        ``defocus_uncertainty``.
        """
        sigma_inf_native = float(sigma_model_px) * (float(native_size) / float(model_size))
        cc = self._scale_factor_inf_to_calib(s_inf_px_per_mm)
        sigma_calib = sigma_inf_native / cc if cc != 0 else sigma_inf_native
        return self.defocus_uncertainty(sigma_calib)

    # ── Serialisation ────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain-dict (yaml/json safe)."""
        d: Dict[str, Any] = {
            "method": self.method,
            "rho_px_per_mm": self.rho_px_per_mm,
            "sigma_0_calib_px": self.sigma_0_calib_px,
            "sigma_floor_calib_px": self.sigma_floor_calib_px,
            "residual_lut_mm_px": (
                [[float(z), float(d)] for z, d in self.residual_lut_mm_px]
                if self.residual_lut_mm_px else None),
            "sigma_min_trusted_calib_px": self.sigma_min_trusted_calib_px,
            "sigma_max_trusted_calib_px": self.sigma_max_trusted_calib_px,
            "sigma_max_model_observed_px": self.sigma_max_model_observed_px,
            "z_min_trusted_mm": self.z_min_trusted_mm,
            "z_max_trusted_mm": self.z_max_trusted_mm,
            "z_max_trusted_neg_mm": self.z_max_trusted_neg_mm,
            "z_max_trusted_pos_mm": self.z_max_trusted_pos_mm,
            "s_calib_px_per_mm": self.s_calib_px_per_mm,
            "loo_cv": (dict(self.loo_cv) if self.loo_cv else None),
            "fit_metadata": dict(self.fit_metadata or {}),
        }
        return d

    _LEGACY_FIELD_MAP = {
        # old → new
        "rho_direct": "rho_px_per_mm",
        "rho": "rho_px_per_mm",
        "sigma_0": "sigma_0_calib_px",
        "scale_calib_px_per_mm": "s_calib_px_per_mm",
    }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationModel":
        """Build a CalibrationModel from its dict form. Accepts both new
        field names and known legacy aliases. When both new and legacy
        fields are provided in the same dict, the NEW name wins
        (legacy is only used as a fallback when the new field is absent
        or None).
        """
        if not isinstance(d, dict):
            raise TypeError(f"from_dict expects a dict, got {type(d).__name__}")

        # Pass 1: copy all NEW (canonical) field names verbatim
        norm: Dict[str, Any] = {}
        for k, v in d.items():
            if k not in cls._LEGACY_FIELD_MAP:
                norm[k] = v
        # Pass 2: fill in legacy aliases for fields still absent / None
        for k, v in d.items():
            new_key = cls._LEGACY_FIELD_MAP.get(k)
            if new_key is None:
                continue
            if norm.get(new_key) is None:
                norm[new_key] = v

        # Default to linear if method missing (backward compat)
        if "method" not in norm or norm["method"] is None:
            norm["method"] = "linear"

        # Convert LUT entries from lists to tuples (yaml round-trip yields lists)
        if norm.get("residual_lut_mm_px"):
            norm["residual_lut_mm_px"] = [
                (float(p[0]), float(p[1])) for p in norm["residual_lut_mm_px"]
            ]

        # Filter to known constructor fields
        known = {
            "method", "rho_px_per_mm",
            "sigma_0_calib_px", "sigma_floor_calib_px",
            "residual_lut_mm_px",
            "sigma_min_trusted_calib_px", "sigma_max_trusted_calib_px",
            "sigma_max_model_observed_px",
            "z_min_trusted_mm", "z_max_trusted_mm",
            "z_max_trusted_neg_mm", "z_max_trusted_pos_mm",
            "s_calib_px_per_mm", "loo_cv", "fit_metadata",
        }
        kwargs = {k: v for k, v in norm.items() if k in known}
        return cls(**kwargs)

    def sha256(self) -> str:
        """Stable canonical hash — useful for sha256 chains across yaml /
        dataset config / checkpoint."""
        canon = json.dumps(
            self.to_dict(), sort_keys=True, default=str, ensure_ascii=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    # ── Convenience: build a linear model from existing ScalingParams ────
    @classmethod
    def from_legacy_scaling(
        cls,
        rho: float,
        sigma_0: float,
        s_calib_px_per_mm: Optional[float] = None,
        sigma_min_trusted_calib_px: float = 0.0,
        sigma_max_trusted_calib_px: float = float("inf"),
    ) -> "CalibrationModel":
        """Adapter for callers that have legacy (rho, sigma_0) only.

        Used by phase-2 shims and by inference paths where the checkpoint
        config lacks an explicit `inversion_method`.
        """
        return cls(
            method="linear",
            rho_px_per_mm=float(rho),
            sigma_0_calib_px=float(sigma_0),
            s_calib_px_per_mm=(float(s_calib_px_per_mm)
                                if s_calib_px_per_mm is not None else None),
            sigma_min_trusted_calib_px=sigma_min_trusted_calib_px,
            sigma_max_trusted_calib_px=sigma_max_trusted_calib_px,
        )
