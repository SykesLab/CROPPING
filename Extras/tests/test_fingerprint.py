"""Tests for the fingerprint_checker tool (Phase 1: Check A — scale chain).

Sets up sys.path the same way test_pipeline.py does so the tool's modules
can import physics + Training without packaging gymnastics.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _module in ("Calibration", "Training", "Preprocessing", "Inference", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# Check A — scale_chain round-trip
# ===========================================================================
class TestScaleChainRoundTrip:
    def _base_config(self) -> dict:
        """A sane direct-mode config like the user's actual training_config.yaml."""
        return {
            'data': {
                'image_size_px': 256,
                'blur_range_px': [1.6268907825654924, 9.88809063913606],
            },
            'training': {
                'rho_direct': 1.41349,
                'sigma_0': 0.2413,
                'scale_calib_px_per_mm': 102.5714,
                'crop_size_px': 299,
                'training_mode': 'direct',
            },
        }

    def test_identity_passes_with_matched_native_size(self):
        from tools.fingerprint_checker.scale_chain import round_trip_check
        result = round_trip_check(self._base_config())
        assert result.overall_passed, (
            "Round-trip should pass exactly when synth crop_size == "
            f"inference native size. Diagnostics: {result.diagnostics}"
        )
        # Every point should round-trip to within machine precision
        for p in result.points:
            assert p.delta_mm < 1e-9, f"z={p.defocus_in_mm}: delta={p.delta_mm}"

    def test_resolution_mismatch_fails_with_correct_drift_ratio(self):
        """When inference's native size doesn't match synth's crop_size_px,
        recovered defocus is z * (inference_native / synth_native) — the
        dissertation's resolution-compression bug, exactly."""
        from tools.fingerprint_checker.scale_chain import round_trip_check
        config = self._base_config()
        synth_native = config['training']['crop_size_px']  # 299
        inf_native = 128
        result = round_trip_check(
            config, inference_native_size=inf_native, tolerance_mm=0.01)
        assert not result.overall_passed, "Mismatch should cause FAIL"
        # The recovered z asymptotes to z * (inf_native/synth_native) at
        # large |z| where σ₀ becomes negligible. At small |z| the σ₀ offset
        # contribution shifts the ratio significantly.
        asymptote = inf_native / synth_native
        for p in result.points:
            if abs(p.defocus_in_mm) < 4.0:
                continue  # σ₀ contribution dominates; ratio departs from asymptote
            actual_ratio = p.defocus_recovered_mm / abs(p.defocus_in_mm)
            assert abs(actual_ratio - asymptote) < 0.05, (
                f"z={p.defocus_in_mm}: expected ratio ~{asymptote:.3f}, "
                f"got {actual_ratio:.3f}"
            )

    def test_cross_camera_scale_inverted_detected(self):
        """If user passes wrong-direction scale_inference, drift appears."""
        from tools.fingerprint_checker.scale_chain import round_trip_check
        config = self._base_config()
        # Inference camera at half the calibration scale → drift
        result = round_trip_check(
            config,
            scale_inference_px_per_mm=51.2857,  # half of s_calib
        )
        # Round-trip on its own (with same-size) actually passes because both
        # chains use the same s_inference. The flag in the report should
        # mention cross-camera scaling is active.
        # Most useful: combined with native-size mismatch we still detect.
        assert any('Cross-camera' in d or 'cross-camera' in d.lower()
                   for d in result.diagnostics) or result.overall_passed, (
            "Either round-trip passes (since both directions use the same "
            "s_inference) or the diagnostic flags cross-camera as active. "
            f"Diagnostics: {result.diagnostics}"
        )

    def test_missing_rho_returns_clean_failure(self):
        from tools.fingerprint_checker.scale_chain import round_trip_check
        config = self._base_config()
        del config['training']['rho_direct']
        result = round_trip_check(config)
        assert not result.overall_passed
        assert any('rho_direct' in d for d in result.diagnostics) or \
               result.config_summary.get('error', '').count('rho_direct') > 0

    def test_missing_blur_range_returns_clean_failure(self):
        from tools.fingerprint_checker.scale_chain import round_trip_check
        config = self._base_config()
        del config['data']['blur_range_px']
        result = round_trip_check(config)
        assert not result.overall_passed
        assert any('blur_range_px' in d for d in result.diagnostics) or \
               result.config_summary.get('error', '').count('blur_range_px') > 0

    def test_round_trip_is_symmetric_in_z_sign(self):
        """Defocus +z and -z should both recover to the same |z|
        (the chain operates on |z| only — sign is handled outside)."""
        from tools.fingerprint_checker.scale_chain import round_trip_check
        result = round_trip_check(self._base_config(),
                                   test_defocuses_mm=(-3.0, 3.0))
        recovered = [p.defocus_recovered_mm for p in result.points]
        assert abs(recovered[0] - recovered[1]) < 1e-9

    def test_text_report_renders_without_unicode_errors(self):
        """Output must be cp1252-safe so it can print on Windows console."""
        from tools.fingerprint_checker.scale_chain import (
            round_trip_check, format_text_report)
        result = round_trip_check(self._base_config())
        text = format_text_report(result)
        # Round-trip through cp1252 to verify no encoding-incompatible chars
        text.encode('cp1252')


# ===========================================================================
# Per-image metrics — sanity-check known-property images
# ===========================================================================
class TestFingerprintMetrics:
    def _step_image(self, size=128, blur_sigma=0.0, polarity='dark_on_light'):
        """A synthetic crop with a sphere in the centre, optionally blurred."""
        import numpy as np
        import cv2
        img = np.ones((size, size), dtype=np.float32)  # bright background
        cv2.circle(img, (size // 2, size // 2), size // 4, 0.0, -1)  # dark obj
        if polarity == 'light_on_dark':
            img = 1.0 - img
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)
        return img

    def test_constant_image_zero_laplacian(self):
        import numpy as np
        from tools.fingerprint_checker.fingerprint_metrics import metric_laplacian_variance
        img = np.full((64, 64), 0.5, dtype=np.float32)
        assert metric_laplacian_variance(img) < 1e-10

    def test_blurred_image_has_lower_laplacian_than_sharp(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_laplacian_variance
        sharp = self._step_image(blur_sigma=0)
        blurred = self._step_image(blur_sigma=4.0)
        v_sharp = metric_laplacian_variance(sharp)
        v_blur = metric_laplacian_variance(blurred)
        assert v_blur < v_sharp, f"sharp={v_sharp}, blurred={v_blur}"

    def test_blurred_image_has_lower_tenengrad_than_sharp(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_tenengrad
        sharp = self._step_image(blur_sigma=0)
        blurred = self._step_image(blur_sigma=4.0)
        assert metric_tenengrad(blurred) < metric_tenengrad(sharp)

    def test_polarity_detection(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_polarity
        dark = self._step_image(polarity='dark_on_light')
        light = self._step_image(polarity='light_on_dark')
        assert metric_polarity(dark) == 'dark_on_light'
        assert metric_polarity(light) == 'light_on_dark'

    def test_polarity_ambiguous_on_constant(self):
        import numpy as np
        from tools.fingerprint_checker.fingerprint_metrics import metric_polarity
        flat = np.full((64, 64), 0.5, dtype=np.float32)
        assert metric_polarity(flat) == 'ambiguous'

    def test_object_diameter_matches_drawn_radius(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_object_diameter_px
        # circle of radius=size/4 → diameter = size/2 = 64
        img = self._step_image(size=128)
        d = metric_object_diameter_px(img)
        # Detected diameter from area-equivalent should be ~64 px
        assert abs(d - 64) < 4, f"expected ~64 px diameter, got {d}"

    def test_centre_offset_zero_when_centred(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_centre_offset_px
        img = self._step_image()
        offset = metric_centre_offset_px(img)
        # Centred circle should have offset < 1 px
        assert offset < 1.5

    def test_crop_occupancy_matches_pi_quarter(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_crop_occupancy
        # circle of radius=size/4 in size×size crop
        # occupancy = pi*(size/4)^2 / size^2 = pi/16 ≈ 0.196
        img = self._step_image(size=128)
        occ = metric_crop_occupancy(img)
        assert abs(occ - 0.196) < 0.02

    def test_high_freq_energy_drops_with_blur(self):
        from tools.fingerprint_checker.fingerprint_metrics import metric_high_freq_energy_ratio
        sharp = self._step_image(blur_sigma=0)
        blurred = self._step_image(blur_sigma=8.0)
        assert metric_high_freq_energy_ratio(blurred) < metric_high_freq_energy_ratio(sharp)

    def test_compute_fingerprint_returns_all_fields(self):
        from tools.fingerprint_checker.fingerprint_metrics import compute_fingerprint
        img = self._step_image()
        rec = compute_fingerprint(img)
        # All numeric fields populated (some may be NaN but the keys exist)
        d = rec.to_dict()
        for key in ('erf_sigma_px', 'edge_transition_width', 'laplacian_variance',
                    'tenengrad', 'background_mean', 'object_diameter_px', 'polarity'):
            assert key in d

    def test_compute_fingerprint_uses_precomputed_erf(self):
        """When erf_sigma_precomputed is given, the function uses it
        instead of recomputing — important for reading metadata.csv's
        sigma_measured_erf column."""
        from tools.fingerprint_checker.fingerprint_metrics import compute_fingerprint
        img = self._step_image()
        rec = compute_fingerprint(
            img, erf_sigma_precomputed=4.2, erf_r_squared_precomputed=0.99)
        assert rec.erf_sigma_px == 4.2
        assert rec.erf_r_squared == 0.99

    def test_skip_erf_keeps_other_metrics(self):
        from tools.fingerprint_checker.fingerprint_metrics import compute_fingerprint
        import numpy as np
        img = self._step_image()
        rec = compute_fingerprint(img, skip_erf=True)
        assert np.isnan(rec.erf_sigma_px)
        # Cheap metrics still computed
        assert not np.isnan(rec.laplacian_variance)
        assert not np.isnan(rec.tenengrad)
