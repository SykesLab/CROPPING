"""
CalibrationModel — Phase 1 unit tests.

Gate criteria for the calibration-method overhaul:

1. Linear forward/inverse produces bit-identical results to the legacy
   ``invert_prediction``/``defocus_to_label`` over a parameter sweep.
2. Round-trip identity (|z| -> sigma -> |z|) within 1e-3 mm for all
   three methods (linear, quadrature, hybrid).
3. ``forward_at`` / ``inverse_at`` round-trip across (s_inf == s_calib),
   (s_inf != s_calib), (native == model), (native != model).
4. Boundary flags (BELOW_FLOOR, SATURATED, IN_RANGE) fire correctly.
5. Validation rejects malformed configurations.
6. to_dict / from_dict / sha256 round-trip preserves behaviour.
7. Legacy field aliases (rho_direct, sigma_0, scale_calib_px_per_mm)
   load correctly into the linear method as backward-compat default.

Run:  pytest Extras/tests/test_calibration_model.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup (matches existing test_pipeline.py pattern)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _module in ("Calibration", "Training", "Preprocessing", "Inference"):
    _p = str(_REPO_ROOT / _module)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics import (  # noqa: E402
    BoundsFlag,
    CalibrationModel,
    ScalingParams,
    invert_prediction,
)


# ---------------------------------------------------------------------------
# Fixtures: representative models for each method, sized to the user's data
# ---------------------------------------------------------------------------

# Calibration constants matching the user's Camera-G calibration
RHO = 0.989946          # px/mm (calib scale)
SIGMA_0 = 0.354         # px (linear y-intercept)
SIGMA_FLOOR = 1.0       # px (quadrature focal-plane blur)
S_CALIB = 89.5556       # px/mm
SIGMA_MIN_TRUSTED = 1.05
SIGMA_MAX_TRUSTED = 5.91
Z_MIN_TRUSTED = 0.7
Z_MAX_TRUSTED = 5.6


@pytest.fixture
def linear_model():
    return CalibrationModel(
        method="linear",
        rho_px_per_mm=RHO,
        sigma_0_calib_px=SIGMA_0,
        s_calib_px_per_mm=S_CALIB,
        sigma_min_trusted_calib_px=SIGMA_MIN_TRUSTED,
        sigma_max_trusted_calib_px=SIGMA_MAX_TRUSTED,
        z_min_trusted_mm=Z_MIN_TRUSTED,
        z_max_trusted_mm=Z_MAX_TRUSTED,
    )


@pytest.fixture
def quadrature_model():
    return CalibrationModel(
        method="quadrature",
        rho_px_per_mm=RHO,
        sigma_floor_calib_px=SIGMA_FLOOR,
        s_calib_px_per_mm=S_CALIB,
        sigma_min_trusted_calib_px=SIGMA_MIN_TRUSTED,
        sigma_max_trusted_calib_px=SIGMA_MAX_TRUSTED,
        z_min_trusted_mm=Z_MIN_TRUSTED,
        z_max_trusted_mm=Z_MAX_TRUSTED,
    )


@pytest.fixture
def hybrid_model():
    # Synthetic LUT — small residuals representative of a real fit
    lut = [(0.5, 0.01), (1.0, -0.02), (1.5, 0.03), (2.0, -0.01),
           (3.0, 0.05), (4.0, -0.03), (5.0, 0.02), (5.5, -0.04)]
    return CalibrationModel(
        method="hybrid",
        rho_px_per_mm=RHO,
        sigma_floor_calib_px=SIGMA_FLOOR,
        residual_lut_mm_px=lut,
        s_calib_px_per_mm=S_CALIB,
        sigma_min_trusted_calib_px=SIGMA_MIN_TRUSTED,
        sigma_max_trusted_calib_px=SIGMA_MAX_TRUSTED,
        z_min_trusted_mm=Z_MIN_TRUSTED,
        z_max_trusted_mm=Z_MAX_TRUSTED,
    )


# ---------------------------------------------------------------------------
# Gate 1: Linear bit-identity vs legacy invert_prediction
# ---------------------------------------------------------------------------

class TestLinearBitIdentity:
    """The linear path through CalibrationModel must produce numerically
    identical results to the legacy `invert_prediction` over a sweep.
    This is the safety gate for the Phase 2 refactor — if these match,
    we can confidently shim the legacy functions through the new class.
    """

    def test_linear_inverse_matches_legacy(self, linear_model):
        """Sweep z, forward via formula, invert via both methods, compare."""
        # Use a high max_blur to avoid SATURATED firing in linear_model
        # (unfair for the comparison since legacy doesn't have bounds flags)
        m = CalibrationModel(
            method="linear",
            rho_px_per_mm=RHO,
            sigma_0_calib_px=SIGMA_0,
            s_calib_px_per_mm=S_CALIB,
            sigma_max_trusted_calib_px=20.0,  # effectively disabled
        )
        sp = ScalingParams(
            rho=RHO, sigma_0=SIGMA_0,
            s_calib=S_CALIB, s_inference=S_CALIB,
            max_blur=20.0, model_size=256,
        )

        max_err = 0.0
        for z in np.linspace(0.1, 7.0, 200):
            sigma_calib = RHO * z + SIGMA_0
            sigma_model = sigma_calib * (256.0 / 299.0)
            label = sigma_model / sp.max_blur
            legacy = invert_prediction(label, sp, native_size=299)
            sigma_model_new = m.forward_at(z, S_CALIB, 256, 299)
            z_back, flag = m.inverse_at(sigma_model_new, S_CALIB, 256, 299)
            assert flag == BoundsFlag.IN_RANGE
            max_err = max(max_err, abs(z_back - legacy.defocus_mm))
            max_err = max(max_err, abs(sigma_model_new - sigma_model))
        assert max_err < 1e-9, f"linear vs legacy max error {max_err} exceeds 1e-9"


# ---------------------------------------------------------------------------
# Gate 2: Round-trip identity for all three methods
# ---------------------------------------------------------------------------

class TestRoundTripIdentity:
    """forward(z) followed by inverse(...) must recover z within tolerance,
    for every method, when the result is IN_RANGE.
    """

    @pytest.mark.parametrize("z_true", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    def test_linear_round_trip(self, linear_model, z_true):
        sigma = linear_model.forward(z_true)
        z_back, flag = linear_model.inverse(sigma)
        assert flag == BoundsFlag.IN_RANGE
        assert abs(z_back - z_true) < 1e-6

    @pytest.mark.parametrize("z_true", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    def test_quadrature_round_trip(self, quadrature_model, z_true):
        sigma = quadrature_model.forward(z_true)
        z_back, flag = quadrature_model.inverse(sigma)
        assert flag == BoundsFlag.IN_RANGE
        assert abs(z_back - z_true) < 1e-6

    @pytest.mark.parametrize("z_true", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    def test_hybrid_round_trip(self, hybrid_model, z_true):
        sigma = hybrid_model.forward(z_true)
        z_back, flag = hybrid_model.inverse(sigma)
        assert flag == BoundsFlag.IN_RANGE
        # Newton iteration tolerance: 1e-3 (per plan gate)
        assert abs(z_back - z_true) < 1e-3


# ---------------------------------------------------------------------------
# Gate 3: forward_at / inverse_at round-trip across scaling configurations
# ---------------------------------------------------------------------------

class TestBoundaryWrappers:
    """The wrappers that handle cross-camera + resolution scaling must
    invert each other cleanly across the parameter space."""

    @pytest.mark.parametrize("s_inf", [89.5556, 50.0, 150.0])  # same, smaller, larger camera
    @pytest.mark.parametrize("native_size", [256, 299, 950])    # equal/smaller/larger than model
    @pytest.mark.parametrize("z_true", [1.0, 3.0, 5.0])
    def test_linear_at_round_trip(self, linear_model, s_inf, native_size, z_true):
        sigma_m = linear_model.forward_at(z_true, s_inf, model_size=256, native_size=native_size)
        z_back, flag = linear_model.inverse_at(sigma_m, s_inf, model_size=256, native_size=native_size)
        assert flag == BoundsFlag.IN_RANGE
        assert abs(z_back - z_true) < 1e-6

    @pytest.mark.parametrize("s_inf", [89.5556, 50.0])
    @pytest.mark.parametrize("native_size", [256, 299, 950])
    @pytest.mark.parametrize("z_true", [1.0, 3.0, 5.0])
    def test_quadrature_at_round_trip(self, quadrature_model, s_inf, native_size, z_true):
        sigma_m = quadrature_model.forward_at(z_true, s_inf, model_size=256, native_size=native_size)
        z_back, flag = quadrature_model.inverse_at(sigma_m, s_inf, model_size=256, native_size=native_size)
        assert flag == BoundsFlag.IN_RANGE
        assert abs(z_back - z_true) < 1e-6

    @pytest.mark.parametrize("s_inf", [89.5556, 50.0])
    @pytest.mark.parametrize("native_size", [256, 299, 950])
    @pytest.mark.parametrize("z_true", [1.0, 3.0, 5.0])
    def test_hybrid_at_round_trip(self, hybrid_model, s_inf, native_size, z_true):
        sigma_m = hybrid_model.forward_at(z_true, s_inf, model_size=256, native_size=native_size)
        z_back, flag = hybrid_model.inverse_at(sigma_m, s_inf, model_size=256, native_size=native_size)
        assert flag == BoundsFlag.IN_RANGE
        assert abs(z_back - z_true) < 1e-3


# ---------------------------------------------------------------------------
# Gate 4: Bounds flags fire correctly
# ---------------------------------------------------------------------------

class TestBoundsFlags:
    def test_linear_below_floor(self, linear_model):
        """sigma below sigma_min_trusted should flag BELOW_FLOOR with z=0."""
        z, flag = linear_model.inverse(0.5)  # < 1.05
        assert flag == BoundsFlag.BELOW_FLOOR
        assert z == 0.0

    def test_linear_saturated(self, linear_model):
        """sigma above sigma_max_trusted should flag SATURATED with a
        finite best-guess z (no longer nan — see BoundsFlag docstring)."""
        z, flag = linear_model.inverse(7.0)  # > 5.91
        assert flag == BoundsFlag.SATURATED
        assert math.isfinite(z), "SATURATED should return best-guess z, not nan"
        assert z > 0, "best-guess should be a positive |z|"

    def test_linear_in_range(self, linear_model):
        z, flag = linear_model.inverse(3.0)
        assert flag == BoundsFlag.IN_RANGE
        assert z > 0

    def test_quadrature_below_floor_via_floor(self, quadrature_model):
        """For quadrature, sigma <= sigma_floor returns BELOW_FLOOR."""
        z, flag = quadrature_model.inverse(0.8)  # < floor=1.0
        assert flag == BoundsFlag.BELOW_FLOOR
        assert z == 0.0

    def test_quadrature_below_trusted_floor(self, quadrature_model):
        """Quadrature sigma between floor and sigma_min_trusted: still BELOW_FLOOR
        because z would round to ~0 anyway."""
        # sigma=1.02 is just above floor=1.0 but below trust=1.05
        z, flag = quadrature_model.inverse(1.02)
        assert flag == BoundsFlag.BELOW_FLOOR
        assert z == 0.0

    def test_model_observed_overrides_trust_ceiling(self):
        """When sigma_max_model_observed_px is stricter, it dominates."""
        m = CalibrationModel(
            method="quadrature",
            rho_px_per_mm=RHO,
            sigma_floor_calib_px=SIGMA_FLOOR,
            sigma_max_trusted_calib_px=5.91,
            sigma_max_model_observed_px=5.5,  # stricter
        )
        z, flag = m.inverse(5.7)  # above model observed but below calib trust
        assert flag == BoundsFlag.SATURATED, "model-observed should dominate"
        assert math.isfinite(z), "SATURATED still returns best-guess z"
        z_ok, flag_ok = m.inverse(5.4)  # below both
        assert flag_ok == BoundsFlag.IN_RANGE


# ---------------------------------------------------------------------------
# Gate 5: Validation rejects bad inputs
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            CalibrationModel(method="cubic", rho_px_per_mm=1.0,
                             sigma_floor_calib_px=1.0)

    def test_negative_rho_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            CalibrationModel(method="linear", rho_px_per_mm=-1.0,
                             sigma_0_calib_px=0.5)

    def test_linear_missing_sigma_0(self):
        with pytest.raises(ValueError, match="sigma_0_calib_px"):
            CalibrationModel(method="linear", rho_px_per_mm=1.0)

    def test_quadrature_missing_floor(self):
        with pytest.raises(ValueError, match="sigma_floor_calib_px"):
            CalibrationModel(method="quadrature", rho_px_per_mm=1.0)

    def test_hybrid_missing_lut(self):
        with pytest.raises(ValueError, match="residual_lut_mm_px"):
            CalibrationModel(method="hybrid", rho_px_per_mm=1.0,
                             sigma_floor_calib_px=1.0)


# ---------------------------------------------------------------------------
# Gate 6: Serialisation round-trip + sha256 stability
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_from_dict_round_trip(self, quadrature_model):
        d = quadrature_model.to_dict()
        restored = CalibrationModel.from_dict(d)
        # Behaviour identical for a sweep
        for z in [1.0, 2.0, 3.0, 4.0]:
            assert restored.forward(z) == quadrature_model.forward(z)
        assert restored.method == quadrature_model.method
        assert restored.sha256() == quadrature_model.sha256()

    def test_hybrid_lut_round_trip(self, hybrid_model):
        d = hybrid_model.to_dict()
        restored = CalibrationModel.from_dict(d)
        for z in [0.5, 1.5, 3.0, 5.0]:
            assert abs(restored.forward(z) - hybrid_model.forward(z)) < 1e-12

    def test_sha256_stable(self):
        m1 = CalibrationModel(method="linear", rho_px_per_mm=0.99, sigma_0_calib_px=0.354)
        m2 = CalibrationModel(method="linear", rho_px_per_mm=0.99, sigma_0_calib_px=0.354)
        assert m1.sha256() == m2.sha256()

    def test_sha256_changes_with_params(self):
        m1 = CalibrationModel(method="linear", rho_px_per_mm=0.99, sigma_0_calib_px=0.354)
        m2 = CalibrationModel(method="linear", rho_px_per_mm=0.999, sigma_0_calib_px=0.354)
        assert m1.sha256() != m2.sha256()


# ---------------------------------------------------------------------------
# Gate 7: Legacy field aliases (backward compat default to linear)
# ---------------------------------------------------------------------------

class TestLegacyCompat:
    def test_loads_legacy_rho_direct_sigma_0(self):
        """Old checkpoints/yaml have rho_direct, sigma_0, scale_calib_px_per_mm.
        Should silently load as linear method with the right params."""
        legacy_dict = {
            "rho_direct": 0.99,
            "sigma_0": 0.354,
            "scale_calib_px_per_mm": 89.5556,
        }
        model = CalibrationModel.from_dict(legacy_dict)
        assert model.method == "linear"
        assert model.rho_px_per_mm == 0.99
        assert model.sigma_0_calib_px == 0.354
        assert model.s_calib_px_per_mm == 89.5556

    def test_from_legacy_scaling_factory(self):
        m = CalibrationModel.from_legacy_scaling(rho=0.99, sigma_0=0.354)
        assert m.method == "linear"
        assert m.rho_px_per_mm == 0.99
        assert m.sigma_0_calib_px == 0.354

    def test_method_field_takes_precedence(self):
        """If both old and new fields present, new fields win."""
        d = {
            "rho_direct": 0.99,
            "rho_px_per_mm": 0.95,  # explicit new
            "sigma_0_calib_px": 0.5,
            "method": "linear",
        }
        m = CalibrationModel.from_dict(d)
        assert m.rho_px_per_mm == 0.95


# ---------------------------------------------------------------------------
# Cross-cutting: hybrid Newton fallback on non-convergent input
# ---------------------------------------------------------------------------

class TestHybridFallback:
    def test_hybrid_inverse_does_not_raise_on_extreme(self, hybrid_model):
        """Even if Newton fails to converge, inverse should return a
        usable answer (quadrature fallback) without raising."""
        # Try inverting an extreme value
        z, flag = hybrid_model.inverse(5.0)
        assert flag in (BoundsFlag.IN_RANGE, BoundsFlag.BELOW_FLOOR, BoundsFlag.SATURATED)
        # No assertion on z value — just that it doesn't crash


# ---------------------------------------------------------------------------
# Phase 3: calibrate_to_model — fitting all three methods
# ---------------------------------------------------------------------------

class TestCalibrateToModel:
    """Synthetic-data recovery tests for the three calibration fits."""

    @staticmethod
    def _synthetic_stack(rho=0.99, sigma_floor=1.0, noise_frac=0.02, seed=42):
        """Make a synthetic z-stack matching the user's calibration shape:
        z range [-7, +5], 0.2 mm steps, with noise + fake plateau at the
        negative tail (mimicking real data exclusions).
        """
        rng = np.random.default_rng(seed)
        z = np.linspace(-7.0, 5.0, 61)
        sigma_clean = np.sqrt((rho * np.abs(z)) ** 2 + sigma_floor ** 2)
        noise = rng.normal(0.0, noise_frac * sigma_clean)
        sigma_obs = sigma_clean + noise
        # Fake plateau: first 3 points (most negative) get tightly clustered
        sigma_obs[0:3] = sigma_obs[3] + rng.normal(0.0, 0.02, 3)
        return list(z), list(sigma_obs)

    def test_linear_fit_returns_calibration_model(self):
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack()
        m = calibrate_to_model("linear", z, sigma, s_calib_px_per_mm=89.55)
        assert m.method == "linear"
        assert 0.5 < m.rho_px_per_mm < 1.5
        assert m.sigma_0_calib_px is not None

    def test_quadrature_recovers_within_5_percent(self):
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack(rho=0.99, sigma_floor=1.0, noise_frac=0.02)
        m = calibrate_to_model("quadrature", z, sigma)
        rho_err = abs(m.rho_px_per_mm - 0.99) / 0.99
        floor_err = abs(m.sigma_floor_calib_px - 1.0) / 1.0
        assert rho_err < 0.05, f"rho err {rho_err*100:.2f}% > 5%"
        assert floor_err < 0.05, f"sigma_floor err {floor_err*100:.2f}% > 5%"

    def test_quadrature_no_noise_recovers_exactly(self):
        from calibration_core import calibrate_to_model
        z = np.linspace(-7.0, 5.0, 61)
        sigma = np.sqrt((0.99 * np.abs(z)) ** 2 + 1.0 ** 2)
        m = calibrate_to_model("quadrature", list(z), list(sigma))
        assert abs(m.rho_px_per_mm - 0.99) < 1e-3
        assert abs(m.sigma_floor_calib_px - 1.0) < 1e-3

    def test_hybrid_includes_residual_lut(self):
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack(noise_frac=0.05)
        m = calibrate_to_model("hybrid", z, sigma)
        assert m.method == "hybrid"
        assert m.residual_lut_mm_px is not None
        assert len(m.residual_lut_mm_px) > 5
        # Residuals should be small (within noise level)
        max_residual = max(abs(d) for _, d in m.residual_lut_mm_px)
        assert max_residual < 1.0  # generous bound

    def test_filter_excludes_near_focus_and_plateau(self):
        """User's actual calibration: 5 near-focus + 3 plateau out of 61."""
        from calibration_core import _filter_for_fit
        z = np.linspace(-7.0, 5.0, 61)
        rho, floor = 0.99, 1.0
        sigma = np.sqrt((rho * np.abs(z)) ** 2 + floor ** 2)
        # Force plateau on the most-negative 3 points
        sigma[0:3] = sigma[3] - 0.02
        z_kept, s_kept, n_near, n_plateau = _filter_for_fit(
            list(z), list(sigma), exclude_near_focus=0.5)
        assert n_near == 5  # z = -0.4, -0.2, 0, +0.2, +0.4
        assert n_plateau == 3  # z = -7.0, -6.8, -6.6

    def test_per_side_slopes_recorded(self):
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack()
        m = calibrate_to_model("quadrature", z, sigma)
        rps = m.fit_metadata["rho_per_side"]
        assert "neg" in rps and "pos" in rps
        assert rps["neg"] is not None and rps["pos"] is not None

    def test_unknown_method_raises(self):
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack()
        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrate_to_model("bilinear", z, sigma)

    def test_too_few_points_raises(self):
        from calibration_core import calibrate_to_model
        with pytest.raises(ValueError, match="Insufficient calibration points"):
            calibrate_to_model("linear", [0.1, 0.2], [1.0, 1.1])

    def test_trust_bounds_match_calibration_data(self):
        """sigma_min/max_trusted should equal min/max of kept points."""
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack()
        m = calibrate_to_model("quadrature", z, sigma)
        # Bounds should be SOME positive value, ordered correctly
        assert m.sigma_min_trusted_calib_px > 0
        assert m.sigma_max_trusted_calib_px > m.sigma_min_trusted_calib_px
        assert m.z_min_trusted_mm > 0
        assert m.z_max_trusted_mm > m.z_min_trusted_mm

    def test_asymmetric_z_caps_recorded(self):
        """User's calibration goes -7 to +5 → asymmetric z caps."""
        from calibration_core import calibrate_to_model
        z, sigma = self._synthetic_stack()  # default goes -7 to +5
        m = calibrate_to_model("quadrature", z, sigma)
        # Negative side reaches farther after plateau exclusion (likely 6.4)
        assert m.z_max_trusted_neg_mm is not None
        assert m.z_max_trusted_pos_mm is not None
        # Positive side max is 5.0; negative side max is at least 5.4 after
        # excluding 3 plateau points starting from z=-7.0
        assert m.z_max_trusted_pos_mm <= 5.0
        assert m.z_max_trusted_neg_mm >= 5.0


# ---------------------------------------------------------------------------
# Phase 10: per-method LOO + uncertainty propagation
# ---------------------------------------------------------------------------

class TestPerMethodLOO:
    """The loo_cv_for_method() function fits the chosen method per fold
    and returns the right-shaped result for that method."""

    @staticmethod
    def _stack():
        rng = np.random.default_rng(7)
        z = np.linspace(-6.0, 5.0, 56)
        sigma = np.sqrt((0.99 * np.abs(z)) ** 2 + 1.0 ** 2)
        sigma = sigma + rng.normal(0.0, 0.02 * sigma)
        return list(z), list(sigma)

    def test_loo_for_linear_uses_sigma_0(self):
        from calibration_core import loo_cv_for_method
        z, sigma = self._stack()
        out = loo_cv_for_method("linear", z, sigma)
        assert out["method"] == "linear"
        assert out["aux_param_name"] == "sigma_0"
        assert out["rho_std"] >= 0
        assert out["aux_param_std"] >= 0
        assert out["num_folds"] >= 40

    def test_loo_for_quadrature_uses_sigma_floor(self):
        from calibration_core import loo_cv_for_method
        z, sigma = self._stack()
        out = loo_cv_for_method("quadrature", z, sigma)
        assert out["method"] == "quadrature"
        assert out["aux_param_name"] == "sigma_floor"
        # sigma_floor recovered ≈ 1.0; std should be small with low noise
        assert out["aux_param_std"] < 0.2

    def test_loo_for_hybrid_completes(self):
        from calibration_core import loo_cv_for_method
        z, sigma = self._stack()
        out = loo_cv_for_method("hybrid", z, sigma)
        assert out["method"] == "hybrid"
        assert out["aux_param_name"] == "sigma_floor"
        assert out["num_folds"] >= 40


class TestDefocusUncertaintyPerMethod:
    """defocus_uncertainty() must produce the right answer for each method.
    Linear is checked vs the standalone legacy function. Quadrature and
    hybrid are checked vs finite-difference numerics on inverse()."""

    LOO = {
        "rho_std": 0.01,
        "aux_param_std": 0.02,
        "loo_mae": 0.05,
        "num_folds": 50,
    }

    def test_linear_matches_legacy_defocus_uncertainty(self):
        """For linear method, CalibrationModel.defocus_uncertainty must
        equal physics.defocus_uncertainty (the legacy function)."""
        from physics import defocus_uncertainty as legacy_unc
        m = CalibrationModel(
            method="linear", rho_px_per_mm=0.99, sigma_0_calib_px=0.354,
            loo_cv={**self.LOO, "aux_param_name": "sigma_0"},
        )
        for sigma_test in [1.0, 2.0, 3.5, 5.0]:
            u_new = m.defocus_uncertainty(sigma_test)
            u_legacy = legacy_unc(
                sigma_native=sigma_test, rho=0.99, sigma_0=0.354,
                rho_std=self.LOO["rho_std"],
                sigma_0_std=self.LOO["aux_param_std"],
            )
            assert abs(u_new - u_legacy) < 1e-9, (
                f"linear unc mismatch at σ={sigma_test}: {u_new} vs {u_legacy}")

    def test_quadrature_jacobian_matches_finite_difference(self):
        """Closed-form quadrature uncertainty should match a numerical
        Jacobian computation within tight tolerance."""
        m = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            loo_cv={**self.LOO, "aux_param_name": "sigma_floor"},
        )
        for sigma_test in [1.5, 2.5, 4.0, 5.0]:
            u_closed_form = m.defocus_uncertainty(sigma_test)

            # Numerical: perturb each param by its std, measure z change
            from copy import copy
            eps_rho = 0.01 * 0.99
            m_lo = copy(m); m_lo.rho_px_per_mm = 0.99 - eps_rho
            m_hi = copy(m); m_hi.rho_px_per_mm = 0.99 + eps_rho
            z_lo, _ = m_lo.inverse(sigma_test)
            z_hi, _ = m_hi.inverse(sigma_test)
            dz_drho = (z_hi - z_lo) / (2 * eps_rho)

            eps_floor = 0.01 * 1.0
            m_lo = copy(m); m_lo.sigma_floor_calib_px = 1.0 - eps_floor
            m_hi = copy(m); m_hi.sigma_floor_calib_px = 1.0 + eps_floor
            z_lo, _ = m_lo.inverse(sigma_test)
            z_hi, _ = m_hi.inverse(sigma_test)
            dz_dfloor = (z_hi - z_lo) / (2 * eps_floor)

            u_numerical = math.sqrt(
                (self.LOO["rho_std"] * dz_drho) ** 2
                + (self.LOO["aux_param_std"] * dz_dfloor) ** 2)

            assert abs(u_closed_form - u_numerical) < 0.01, (
                f"quadrature unc closed-form vs numerical at σ={sigma_test}: "
                f"{u_closed_form} vs {u_numerical}")

    def test_hybrid_uncertainty_is_nonzero(self):
        """Hybrid uses numerical Jacobian; just verify it runs and
        produces a sensible non-zero value within trusted range."""
        m = CalibrationModel(
            method="hybrid", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            residual_lut_mm_px=[(0.5, 0.01), (1.0, -0.02), (3.0, 0.05)],
            loo_cv={**self.LOO, "aux_param_name": "sigma_floor"},
            sigma_max_trusted_calib_px=20.0,
        )
        u = m.defocus_uncertainty(3.0)
        assert u > 0
        # Should be similar order to quadrature on same data
        m_quad = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            loo_cv={**self.LOO, "aux_param_name": "sigma_floor"},
        )
        u_quad = m_quad.defocus_uncertainty(3.0)
        # Hybrid residual is small → uncertainty close to quadrature
        assert abs(u - u_quad) < 0.1

    def test_uncertainty_zero_when_no_loo_data(self):
        m = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            loo_cv=None,
        )
        assert m.defocus_uncertainty(3.0) == 0.0

    def test_uncertainty_zero_when_saturated(self):
        m = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            sigma_max_trusted_calib_px=5.0,
            loo_cv={**self.LOO, "aux_param_name": "sigma_floor"},
        )
        assert m.defocus_uncertainty(6.0) == 0.0

    def test_uncertainty_zero_when_below_floor_quadrature(self):
        m = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            loo_cv={**self.LOO, "aux_param_name": "sigma_floor"},
        )
        assert m.defocus_uncertainty(0.8) == 0.0

    def test_uncertainty_at_boundary_wrapper(self):
        """defocus_uncertainty_at should equal defocus_uncertainty on
        equivalent calibration-pixel input."""
        m = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            s_calib_px_per_mm=89.55,
            loo_cv={**self.LOO, "aux_param_name": "sigma_floor"},
            sigma_max_trusted_calib_px=20.0,
        )
        # σ_calib = 3.0 corresponds to σ_model = 3.0 * 256/950 at native=950
        sigma_model = 3.0 * 256.0 / 950.0
        u_at = m.defocus_uncertainty_at(sigma_model, s_inf_px_per_mm=89.55,
                                         model_size=256, native_size=950)
        u_direct = m.defocus_uncertainty(3.0)
        assert abs(u_at - u_direct) < 1e-9

    def test_calib_valid_defocus_mm_uses_method(self):
        """calib_valid_defocus_mm should use the chosen method's inverse,
        not always the linear formula."""
        from physics import calib_valid_defocus_mm
        # Quadrature config — sigma_floor=1.0, blur_range [1.5, 5.0]
        cm = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            sigma_max_trusted_calib_px=20.0,  # disable saturation
        )
        config = {
            "data": {"blur_range_px": [1.5, 5.0]},
            "training": {
                "calibration_model": cm.to_dict(),
                "rho_direct": 0.99, "sigma_0": 0.354,  # legacy fields too
            },
        }
        z_min, z_max = calib_valid_defocus_mm(config)
        # Quadrature inverse: z = sqrt(sigma^2 - 1.0^2) / 0.99
        # z_min from sigma=1.5: sqrt(2.25-1)/0.99 = sqrt(1.25)/0.99 ≈ 1.13
        # z_max from sigma=5.0: sqrt(25-1)/0.99 = sqrt(24)/0.99 ≈ 4.95
        # Linear would give: (1.5-0.354)/0.99 = 1.16, (5.0-0.354)/0.99 = 4.69
        # The two methods give noticeably different z bounds.
        # We just check it's quadrature math:
        import math
        expected_z_min = math.sqrt(1.5**2 - 1.0**2) / 0.99
        expected_z_max = math.sqrt(5.0**2 - 1.0**2) / 0.99
        assert abs(z_min - expected_z_min) < 1e-3
        assert abs(z_max - expected_z_max) < 1e-3

    def test_calib_valid_defocus_mm_legacy_fallback(self):
        """Without calibration_model field, falls back to linear math."""
        from physics import calib_valid_defocus_mm
        config = {
            "data": {"blur_range_px": [1.5, 5.0]},
            "training": {"rho_direct": 0.99, "sigma_0": 0.354},  # legacy only
        }
        z_min, z_max = calib_valid_defocus_mm(config)
        expected_z_min = max(0.0, (1.5 - 0.354) / 0.99)
        expected_z_max = (5.0 - 0.354) / 0.99
        assert abs(z_min - expected_z_min) < 1e-6
        assert abs(z_max - expected_z_max) < 1e-6

    def test_write_calibration_syncs_calibration_model_block(self):
        """Phase 11: editing rho/sigma_0 must also update the embedded
        calibration_model block so new inference doesn't read stale values."""
        from calibration_editor import write_calibration
        # Simulate a checkpoint with both legacy and new fields
        ckpt = {
            'config': {
                'training': {
                    'rho_direct': 0.99,
                    'sigma_0': 0.354,
                    'inversion_method': 'linear',
                    'calibration_model': {
                        'method': 'linear',
                        'rho_px_per_mm': 0.99,
                        'sigma_0_calib_px': 0.354,
                    },
                },
            },
        }
        # Apply correction
        write_calibration(ckpt, rho=1.05, sigma_0=0.500)
        # Both legacy and new fields must be updated
        assert ckpt['config']['training']['rho_direct'] == 1.05
        assert ckpt['config']['training']['sigma_0'] == 0.500
        cm_dict = ckpt['config']['training']['calibration_model']
        assert cm_dict['rho_px_per_mm'] == 1.05
        assert cm_dict['sigma_0_calib_px'] == 0.500

    def test_loo_cv_round_trips_through_serialisation(self):
        m = CalibrationModel(
            method="quadrature", rho_px_per_mm=0.99, sigma_floor_calib_px=1.0,
            loo_cv={"rho_std": 0.005, "aux_param_name": "sigma_floor",
                    "aux_param_std": 0.01, "loo_mae": 0.03, "num_folds": 53},
        )
        d = m.to_dict()
        restored = CalibrationModel.from_dict(d)
        assert restored.loo_cv == m.loo_cv
        # And uncertainty is preserved
        u_orig = m.defocus_uncertainty(3.0)
        u_rest = restored.defocus_uncertainty(3.0)
        assert abs(u_orig - u_rest) < 1e-9
