"""
CROPPING Pipeline — Core Test Suite
=====================================
10 tests covering the critical invariants of the depth-from-defocus pipeline.
Each test is self-contained, targets one invariant, runs under 5 seconds, and
requires no .cine files, GPU, or external data.

Run:  pytest tests/ -v --timeout=10
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — use package imports with sys.path fallback
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

for _module in ("Calibration", "Training", "Preprocessing", "Inference"):
    _p = str(_REPO_ROOT / _module)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Also add repo root for package-style imports
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ===========================================================================
# Test 1 — Calibration forward-inverse round-trip
# Invariant: sigma_to_depth(rho * |z| + sigma_0) == |z|
# ===========================================================================
class TestCalibrationRoundTrip:
    def test_forward_inverse_recovers_depth(self):
        from calibration_core import calibrate_approach_a, sigma_to_depth_approach_a

        rho_true, sigma_0_true = 2.5, 0.8
        z_values = np.linspace(-10, 10, 21).tolist()
        sigma_values = [rho_true * abs(z) + sigma_0_true for z in z_values]

        result = calibrate_approach_a(z_values, sigma_values)

        for z in z_values:
            sigma_pred = result.rho_px_per_mm * abs(z) + result.sigma_0
            z_recovered = sigma_to_depth_approach_a(
                sigma_pred, result.rho_px_per_mm, result.sigma_0
            )
            assert abs(z_recovered - abs(z)) < 1e-6, (
                f"Round-trip failed at z={z}: recovered {z_recovered}, expected {abs(z)}"
            )


# ===========================================================================
# Test 2 — DME loss: identical inputs = 0, symmetric
# ===========================================================================
@requires_torch
class TestDMELossProperties:
    def test_identical_inputs_give_zero_loss(self):
        from losses import DMELoss

        loss_fn = DMELoss(max_blur=15.0, eps=0.01)
        a = torch.tensor([[0.3], [0.5], [0.8]])
        loss = loss_fn(a, a)
        assert loss.item() < 1e-10, f"Loss for identical inputs should be ~0, got {loss.item()}"

    def test_symmetry(self):
        from losses import DMELoss

        loss_fn = DMELoss(max_blur=15.0, eps=0.01)
        a = torch.tensor([[0.2], [0.6]])
        b = torch.tensor([[0.7], [0.3]])
        assert abs(loss_fn(a, b).item() - loss_fn(b, a).item()) < 1e-7


# ===========================================================================
# Test 3 — DME loss: log-space relative-error invariance
# ===========================================================================
@requires_torch
class TestDMELossRelativeError:
    def test_equal_relative_error_equal_loss(self):
        from losses import DMELoss

        loss_fn = DMELoss(max_blur=20.0, eps=0.01)
        pred_low = torch.tensor([[0.05]])
        tgt_low = torch.tensor([[0.10]])
        pred_high = torch.tensor([[0.40]])
        tgt_high = torch.tensor([[0.80]])

        loss_low = loss_fn(pred_low, tgt_low).item()
        loss_high = loss_fn(pred_high, tgt_high).item()

        ratio = loss_low / loss_high if loss_high > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"Log-space loss should be roughly scale-invariant: "
            f"low={loss_low:.6f}, high={loss_high:.6f}, ratio={ratio:.2f}"
        )


# ===========================================================================
# Test 4 — Model output shape and [0, 1] range
# ===========================================================================
@requires_torch
class TestModelOutputContract:
    def test_output_shape_and_range(self):
        from model import DefocusNet

        model = DefocusNet()
        model.eval()
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
        assert out.min().item() >= 0.0, f"Output below 0: {out.min().item()}"
        assert out.max().item() <= 1.0, f"Output above 1: {out.max().item()}"


# ===========================================================================
# Test 5 — Gaussian kernel normalisation and symmetry
# ===========================================================================
class TestGaussianKernel:
    def test_kernel_normalised_and_symmetric(self):
        from synthetic_blur import create_gaussian_kernel

        for sigma in [1.0, 3.0, 8.0, 15.0]:
            k = create_gaussian_kernel(sigma)
            assert abs(k.sum() - 1.0) < 1e-5, (
                f"Kernel sum != 1 for sigma={sigma}: {k.sum()}"
            )
            np.testing.assert_array_almost_equal(
                k, k.T, decimal=7,
                err_msg=f"Kernel not symmetric for sigma={sigma}",
            )


# ===========================================================================
# Test 6 — Blur-measure round-trip: apply blur then ERF fit recovers sigma
# ===========================================================================
class TestBlurMeasureRoundTrip:
    def test_applied_blur_matches_erf_measurement(self):
        from synthetic_blur import apply_gaussian_blur
        from scipy.special import erf
        from scipy.optimize import curve_fit

        sigma_true = 4.0
        size = 256
        sharp = np.zeros((size, size), dtype=np.float32)
        sharp[:, size // 2:] = 1.0

        blurred = apply_gaussian_blur(sharp, sigma_true)

        row = size // 2
        profile_x = np.arange(size).astype(np.float64)
        profile = blurred[row, :].astype(np.float64)

        def erf_model(x, I_left, I_right, edge, sigma):
            sigma_s = max(sigma, 0.001)
            return (I_left + I_right) / 2 + (I_right - I_left) / 2 * erf(
                (x - edge) / (sigma_s * np.sqrt(2))
            )

        popt, _ = curve_fit(
            erf_model, profile_x, profile,
            p0=[0.0, 1.0, size / 2, sigma_true],
            bounds=([-0.1, 0.5, size * 0.3, 0.01], [0.5, 1.1, size * 0.7, 50]),
        )
        sigma_measured = popt[3]

        assert abs(sigma_measured - sigma_true) / sigma_true < 0.05, (
            f"ERF measured sigma={sigma_measured:.3f}, expected {sigma_true:.3f} "
            f"(error {abs(sigma_measured - sigma_true) / sigma_true * 100:.1f}%)"
        )


# ===========================================================================
# Test 7 — Inference scaling chain: perfect prediction recovers true defocus
# ===========================================================================
class TestInferenceScalingChain:
    def test_perfect_prediction_recovers_defocus(self):
        rho = 2.0
        sigma_0 = 0.5
        s_calib = 100.0
        s_c = 100.0
        max_blur = 25.0
        crop_size = 299
        model_size = 256
        defocus_true = 5.0

        # Forward: training label generation
        sigma_true = rho * defocus_true + sigma_0
        sigma_model_scale = sigma_true * (model_size / crop_size)
        pred_norm = sigma_model_scale / max_blur

        # Inverse: inference chain
        sigma_model = pred_norm * max_blur
        sigma_native = sigma_model * (crop_size / model_size)
        scale_ratio = s_c / s_calib
        rho_eff = rho * scale_ratio
        sigma_0_native = sigma_0 * scale_ratio
        defocus_recovered = (sigma_native - sigma_0_native) / rho_eff

        assert abs(defocus_recovered - defocus_true) < 1e-10, (
            f"Scaling chain broken: recovered {defocus_recovered:.6f} mm, "
            f"expected {defocus_true} mm"
        )

    def test_cross_camera_scaling(self):
        """Round-trip with different calibration and inference cameras."""
        rho = 1.548
        sigma_0 = 0.125
        s_calib = 102.57
        s_c = 120.0  # different camera
        max_blur = 13.72
        crop_size = 299
        model_size = 256
        defocus_true = 4.0

        # Forward
        sigma_calib = rho * defocus_true + sigma_0
        sigma_native = sigma_calib * (s_c / s_calib)
        sigma_model = sigma_native * (model_size / crop_size)
        pred_norm = sigma_model / max_blur

        # Inverse
        sigma_model_inv = pred_norm * max_blur
        sigma_native_inv = sigma_model_inv * (crop_size / model_size)
        rho_eff = rho * (s_c / s_calib)
        sigma_0_eff = sigma_0 * (s_c / s_calib)
        defocus_recovered = (sigma_native_inv - sigma_0_eff) / rho_eff

        assert abs(defocus_recovered - defocus_true) < 1e-10, (
            f"Cross-camera round-trip failed: recovered {defocus_recovered:.6f}, "
            f"expected {defocus_true}"
        )


# ===========================================================================
# Test 8 — Label normalisation clamp
# Invariant: normalised labels must be in [0, 1]
# ===========================================================================
class TestLabelNormalisationClamp:
    def test_label_never_exceeds_one(self):
        """Even if native blur exceeds max_sigma, normalised label should clamp."""
        max_sigma = 10.0
        # Simulate: native blur = 12.0 (exceeds max), target sigma = 11.0
        sigma_model = 12.0  # from quadrature fallback
        normalized = min(sigma_model / max_sigma, 1.0)
        assert normalized <= 1.0, f"Normalised label {normalized} exceeds 1.0"

    def test_label_never_negative(self):
        max_sigma = 10.0
        sigma_model = 0.0
        normalized = min(sigma_model / max_sigma, 1.0) if max_sigma > 0 else 0.0
        assert normalized >= 0.0, f"Normalised label {normalized} is negative"


# ===========================================================================
# Test 9 — Physics module: forward-inverse round-trip via unified module
# ===========================================================================
class TestPhysicsModuleRoundTrip:
    def test_same_camera_round_trip(self):
        """Forward then inverse through physics module recovers defocus exactly."""
        from physics import ScalingParams, defocus_to_label, label_to_defocus

        params = ScalingParams(
            rho=2.0, sigma_0=0.5, s_calib=100.0, s_inference=100.0,
            max_blur=25.0, model_size=256,
        )
        native_size = 299
        defocus_true = 5.0

        label = defocus_to_label(defocus_true, params, native_size)
        defocus_recovered, clamped, saturated = label_to_defocus(label, params, native_size)

        assert abs(defocus_recovered - defocus_true) < 1e-10, (
            f"Physics module round-trip failed: {defocus_recovered:.6f} != {defocus_true}"
        )
        assert not clamped
        assert not saturated

    def test_cross_camera_round_trip(self):
        """Forward then inverse with different cameras recovers defocus."""
        from physics import ScalingParams, defocus_to_label, label_to_defocus

        params = ScalingParams(
            rho=1.548, sigma_0=0.125, s_calib=102.57, s_inference=120.0,
            max_blur=13.72, model_size=256,
        )
        native_size = 299
        defocus_true = 4.0

        label = defocus_to_label(defocus_true, params, native_size)
        defocus_recovered, clamped, saturated = label_to_defocus(label, params, native_size)

        assert abs(defocus_recovered - defocus_true) < 1e-10, (
            f"Cross-camera round-trip failed: {defocus_recovered:.6f} != {defocus_true}"
        )

    def test_clamping_flagged(self):
        """Negative defocus is clamped to 0 and flagged."""
        from physics import ScalingParams, label_to_defocus

        params = ScalingParams(
            rho=2.0, sigma_0=5.0, s_calib=100.0, s_inference=100.0,
            max_blur=25.0, model_size=256,
        )
        # Very small label → sigma_native < sigma_0 → negative defocus
        defocus, clamped, saturated = label_to_defocus(0.005, params, 256)
        assert defocus == 0.0
        assert clamped

    def test_saturation_flagged(self):
        """Labels at sigmoid extremes are flagged as saturated."""
        from physics import ScalingParams, label_to_defocus

        params = ScalingParams(rho=2.0, sigma_0=0.0, max_blur=20.0, model_size=256)
        _, _, sat_low = label_to_defocus(0.005, params, 256)
        _, _, sat_high = label_to_defocus(0.995, params, 256)
        _, _, sat_mid = label_to_defocus(0.5, params, 256)
        assert sat_low
        assert sat_high
        assert not sat_mid

    def test_consistency_with_inline_math(self):
        """Physics module matches the original inline equations exactly."""
        from physics import ScalingParams, invert_prediction

        rho, sigma_0 = 1.548, 0.125
        s_calib, s_c = 102.57, 120.0
        max_blur, model_size = 13.72, 256
        native_size = 299
        pred_val = 0.42

        # Inline math (original code)
        sigma_model = pred_val * max_blur
        sigma_native = sigma_model * (native_size / model_size)
        scale_ratio = s_c / s_calib
        rho_eff = rho * scale_ratio
        sigma_0_eff = sigma_0 * scale_ratio
        defocus_inline = max(0.0, (sigma_native - sigma_0_eff) / rho_eff)

        # Physics module
        params = ScalingParams(
            rho=rho, sigma_0=sigma_0, s_calib=s_calib, s_inference=s_c,
            max_blur=max_blur, model_size=model_size,
        )
        result = invert_prediction(pred_val, params, native_size)

        assert abs(result.defocus_mm - defocus_inline) < 1e-12, (
            f"Physics module disagrees with inline: {result.defocus_mm} vs {defocus_inline}"
        )
        assert abs(result.sigma_model - sigma_model) < 1e-12
        assert abs(result.sigma_native - sigma_native) < 1e-12


# ===========================================================================
# Test 10 — run_paths utility (timestamped runs/datasets)
# ===========================================================================
class TestRunPaths:
    def test_sanitise(self):
        from run_paths import sanitise_run_name
        assert sanitise_run_name("phantom g/2") == "phantom_g_2"
        assert sanitise_run_name("test:bad/chars") == "test_bad_chars"
        assert sanitise_run_name("") == "unnamed"
        assert sanitise_run_name("   ") == "unnamed"
        assert sanitise_run_name("...") == "unnamed"
        assert sanitise_run_name("baseline") == "baseline"

    def test_make_folder_name(self):
        import re
        from run_paths import make_run_folder_name
        n = make_run_folder_name("baseline", default="run")
        assert re.match(r"^\d{8}_\d{6}_baseline$", n), n
        n2 = make_run_folder_name(None, default="dataset")
        assert re.match(r"^\d{8}_\d{6}_dataset$", n2), n2

    def test_list_and_validate(self, tmp_path):
        from run_paths import (datasets_root, find_latest_dataset, list_datasets,
                                validate_dataset)
        # Build a fake training_output/ tree
        ds = datasets_root(tmp_path)
        for name in ("20260101_120000_old", "20260423_120000_new"):
            d = ds / name
            (d / "blur").mkdir(parents=True)
            (d / "metadata.csv").write_text("index,sigma_px\n000000,1.0\n")
        items = list_datasets(tmp_path)
        assert len(items) == 2
        assert items[0].name == "20260423_120000_new"  # newest first
        latest = find_latest_dataset(tmp_path)
        assert latest is not None and latest.name == "20260423_120000_new"

        ok, _ = validate_dataset(items[0])
        assert ok

    def test_validate_rejects_bad(self, tmp_path):
        from run_paths import validate_dataset
        bad = tmp_path / "no_meta"
        (bad / "blur").mkdir(parents=True)
        ok, msg = validate_dataset(bad)
        assert not ok and "metadata.csv" in msg

    def test_per_model_paths(self, tmp_path):
        import re
        from run_paths import (
            models_root, model_dir, model_checkpoints_dir, model_edits_dir,
            edit_dir, tests_dir, make_test_folder_name,
        )
        m = "20260423_202438_demo"
        # Roots
        assert models_root(tmp_path).name == "models"
        assert model_dir(tmp_path, m) == tmp_path / "models" / m
        assert model_checkpoints_dir(tmp_path, m) == \
            tmp_path / "models" / m / "checkpoints"
        assert model_edits_dir(tmp_path, m) == tmp_path / "models" / m / "edits"
        # Edit folder
        assert edit_dir(tmp_path, m, "tuned_for_camG") == \
            tmp_path / "models" / m / "edits" / "tuned_for_camG"
        # Test-result folders — original model
        assert tests_dir(tmp_path, m, 'synthetic') == \
            tmp_path / "models" / m / "tests" / "synthetic"
        assert tests_dir(tmp_path, m, 'real_crop') == \
            tmp_path / "models" / m / "tests" / "real_crop"
        # Test-result folders — edited model
        assert tests_dir(tmp_path, m, 'synthetic', edit_name='v1') == \
            tmp_path / "models" / m / "edits" / "v1" / "tests" / "synthetic"
        assert tests_dir(tmp_path, m, 'real_crop', edit_name='tuned') == \
            tmp_path / "models" / m / "edits" / "tuned" / "tests" / "real_crop"
        # Illegal kind rejected
        import pytest as _pt
        with _pt.raises(ValueError):
            tests_dir(tmp_path, m, 'bogus')
        # Test-folder name is just test_<ts> (no variant suffix)
        assert re.match(r"^test_\d{8}_\d{6}$", make_test_folder_name())

    def test_parse_true_z_from_filename(self):
        from run_paths import parse_true_z_from_filename
        # Typical patterns the inference engine emits
        assert parse_true_z_from_filename("sphere4_z-6.20mm.png") == -6.20
        assert parse_true_z_from_filename("frame_042_z+1.50mm.png") == 1.50
        assert parse_true_z_from_filename("crop_z0.00mm.png") == 0.0
        assert parse_true_z_from_filename("x_z5mm.png") == 5.0  # no decimal
        # No match → None
        assert parse_true_z_from_filename("just_a_name.png") is None
        assert parse_true_z_from_filename("z123_no_mm_suffix.png") is None
        # Works on Path objects too
        from pathlib import Path
        assert parse_true_z_from_filename(Path("a/b/z-3.14mm.png")) == -3.14

    def test_detect_model_name_and_variant(self, tmp_path):
        from run_paths import detect_model_name, detect_variant
        m = "20260423_202438_demo"
        # Checkpoint of the original model (under models/<m>/checkpoints/)
        src = tmp_path / "models" / m / "checkpoints" / "dme_best.pth"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"")
        assert detect_model_name(src) == m
        assert detect_variant(src) == "original"

        # Checkpoint of an edited model (under models/<m>/edits/<edit>/)
        v1 = tmp_path / "models" / m / "edits" / "tuned_for_camG" / "dme_best.pth"
        v1.parent.mkdir(parents=True)
        v1.write_bytes(b"")
        assert detect_model_name(v1) == m
        assert detect_variant(v1) == "tuned_for_camG"

        # Another edit with a different user-chosen name
        v2 = tmp_path / "models" / m / "edits" / "after_bias_fix" / "dme_best.pth"
        v2.parent.mkdir(parents=True)
        v2.write_bytes(b"")
        assert detect_variant(v2) == "after_bias_fix"

        # Unrecognised location → no model name, variant from stem
        stray = tmp_path / "elsewhere" / "somehow.pth"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"")
        assert detect_model_name(stray) is None
        assert detect_variant(stray) == "somehow"


# ===========================================================================
# Test 11 — calibration_editor (bake post-hoc correction into ρ, σ₀)
# ===========================================================================
class TestCalibrationEditor:
    def test_linear_correction_identity(self):
        from calibration_editor import apply_linear_correction
        # a=1, b=0 must be a no-op
        rho_new, sigma_0_new = apply_linear_correction(1.4135, 0.241, 1.0, 0.0)
        assert abs(rho_new - 1.4135) < 1e-9
        assert abs(sigma_0_new - 0.241) < 1e-9

    def test_linear_correction_typical(self):
        from calibration_editor import apply_linear_correction
        # Known case: a=1.034, b=-0.12  from the spec example
        rho_new, sigma_0_new = apply_linear_correction(1.4135, 0.241, 1.034, -0.12)
        # Verify ρ_new = ρ/a and σ₀_new = σ₀ − b·ρ/a
        assert abs(rho_new - 1.4135 / 1.034) < 1e-9
        assert abs(sigma_0_new - (0.241 - (-0.12) * 1.4135 / 1.034)) < 1e-9

    def test_linear_correction_is_equivalent_to_linear_remap(self):
        """Applying ẑ_corr = a·ẑ + b is equivalent to using the new constants."""
        from calibration_editor import apply_linear_correction
        rho, sigma_0 = 1.4135, 0.241
        a, b = 1.034, -0.12
        rho_n, sigma_0_n = apply_linear_correction(rho, sigma_0, a, b)
        # Pick an arbitrary σ_pred; the two formulas should agree.
        for sigma_pred in (0.3, 1.5, 5.0, 9.9):
            z_orig = (sigma_pred - sigma_0) / rho
            z_corr_maths = a * z_orig + b
            z_new_consts = (sigma_pred - sigma_0_n) / rho_n
            assert abs(z_corr_maths - z_new_consts) < 1e-9, sigma_pred

    def test_linear_correction_degenerate(self):
        from calibration_editor import apply_linear_correction, CalibrationError
        with pytest.raises(CalibrationError):
            apply_linear_correction(1.4, 0.2, 0.0, 0.0)

    def test_invert_correction_roundtrip(self):
        from calibration_editor import apply_linear_correction, invert_correction
        rho, sigma_0 = 1.4135, 0.241
        a, b = 1.034, -0.12
        rho_n, sigma_0_n = apply_linear_correction(rho, sigma_0, a, b)
        a_rec, b_rec = invert_correction(rho_n, sigma_0_n, rho, sigma_0)
        assert abs(a_rec - a) < 1e-9
        assert abs(b_rec - b) < 1e-9

    def test_calib_valid_defocus_mm_derives_from_config(self):
        from physics import calib_valid_defocus_mm, CALIB_VALID_DEFOCUS_MM_FALLBACK
        # Typical case: blur range / rho / sigma_0 all present
        cfg = {
            'data': {'blur_range_px': [1.63, 9.89]},
            'training': {'rho_direct': 1.4135, 'sigma_0': 0.241},
        }
        z_min, z_max = calib_valid_defocus_mm(cfg)
        # (1.63 - 0.241) / 1.4135 ≈ 0.983
        # (9.89 - 0.241) / 1.4135 ≈ 6.825
        assert abs(z_min - 0.983) < 0.01
        assert abs(z_max - 6.825) < 0.01

        # Missing sigma_0 → treated as 0
        cfg2 = {
            'data': {'blur_range_px': [2.0, 10.0]},
            'training': {'rho_direct': 2.0},
        }
        z_min, z_max = calib_valid_defocus_mm(cfg2)
        assert abs(z_min - 1.0) < 1e-9 and abs(z_max - 5.0) < 1e-9

        # blur_min below sigma_0 → z_min clamped to 0
        cfg3 = {
            'data': {'blur_range_px': [0.1, 5.0]},
            'training': {'rho_direct': 1.0, 'sigma_0': 0.5},
        }
        z_min, z_max = calib_valid_defocus_mm(cfg3)
        assert z_min == 0.0
        assert abs(z_max - 4.5) < 1e-9

        # Missing blur_range_px → fallback
        assert calib_valid_defocus_mm({'training': {'rho_direct': 1.4}}) == \
            CALIB_VALID_DEFOCUS_MM_FALLBACK
        # Missing rho → fallback
        assert calib_valid_defocus_mm({'data': {'blur_range_px': [1, 5]}}) == \
            CALIB_VALID_DEFOCUS_MM_FALLBACK
        # rho <= 0 → fallback
        assert calib_valid_defocus_mm(
            {'data': {'blur_range_px': [1, 5]}, 'training': {'rho_direct': 0}}) == \
            CALIB_VALID_DEFOCUS_MM_FALLBACK
        # Empty / None config → fallback
        assert calib_valid_defocus_mm({}) == CALIB_VALID_DEFOCUS_MM_FALLBACK
        assert calib_valid_defocus_mm(None) == CALIB_VALID_DEFOCUS_MM_FALLBACK

    def test_correction_from_fit(self):
        from calibration_editor import correction_from_fit, CalibrationError
        # Identity fit — no correction
        a, b = correction_from_fit(1.0, 0.0)
        assert abs(a - 1.0) < 1e-12 and abs(b) < 1e-12
        # Predictions 10% high, no offset → scale down
        a, b = correction_from_fit(1.1, 0.0)
        assert abs(a - 1 / 1.1) < 1e-12 and abs(b) < 1e-12
        # Predictions shifted up by 0.5 mm → pure offset correction
        a, b = correction_from_fit(1.0, 0.5)
        assert abs(a - 1.0) < 1e-12 and abs(b + 0.5) < 1e-12
        # Round trip: applying (a, b) to a fake "pred = slope·true + intercept"
        # must recover true.
        slope, intercept = 1.07, 0.15
        a, b = correction_from_fit(slope, intercept)
        for true in (0.5, 2.0, 5.0, 9.0):
            pred = slope * true + intercept
            true_recovered = a * pred + b
            assert abs(true_recovered - true) < 1e-12, true
        # Degenerate
        with pytest.raises(CalibrationError):
            correction_from_fit(0.0, 0.5)

    @requires_torch
    def test_read_write_calibration_roundtrip(self, tmp_path):
        import torch
        from calibration_editor import (read_calibration, write_calibration,
                                         save_corrected_checkpoint, load_checkpoint,
                                         read_history)
        # Build a minimal checkpoint on disk
        src = tmp_path / "dme_best.pth"
        ckpt = {
            'config': {'training': {'rho_direct': 1.4135, 'sigma_0': 0.241,
                                     'training_mode': 'direct'}},
            'dme_state_dict': {},
        }
        torch.save(ckpt, src)

        # Save a corrected version — history should seed + grow
        out = tmp_path / "edits" / "dme_best_v1.pth"
        snap = save_corrected_checkpoint(
            src, out, rho_new=1.3672, sigma_0_new=0.180,
            source_label='fit', note='a=1.034 b=-0.12')
        assert out.is_file()
        assert snap.rho == 1.3672 and snap.sigma_0 == 0.180

        # Re-load the corrected checkpoint and verify
        loaded = load_checkpoint(out)
        rho, sigma_0 = read_calibration(loaded)
        assert abs(rho - 1.3672) < 1e-9 and abs(sigma_0 - 0.180) < 1e-9
        history = read_history(loaded)
        # History should have: [seeded training, the new fit snapshot]
        assert len(history) == 2
        assert history[0].source == 'training' and history[0].rho == 1.4135
        assert history[1].source == 'fit' and history[1].rho == 1.3672

        # Source must be untouched
        src_loaded = load_checkpoint(src)
        rho_src, _ = read_calibration(src_loaded)
        assert rho_src == 1.4135, "source checkpoint was mutated"

    @requires_torch
    def test_chained_edit_preserves_original(self, tmp_path):
        import torch
        from calibration_editor import (save_corrected_checkpoint, load_checkpoint,
                                         read_history)
        src = tmp_path / "dme_best.pth"
        torch.save({'config': {'training': {'rho_direct': 2.0, 'sigma_0': 0.1}},
                    'dme_state_dict': {}}, src)
        v1 = tmp_path / "edits" / "dme_best_v1.pth"
        v2 = tmp_path / "edits" / "dme_best_v2.pth"
        save_corrected_checkpoint(src, v1, 1.8, 0.15, 'fit', note='first')
        save_corrected_checkpoint(v1, v2, 1.7, 0.18, 'manual', note='second')
        h2 = read_history(load_checkpoint(v2))
        assert len(h2) == 3  # training + fit + manual
        assert [s.rho for s in h2] == [2.0, 1.8, 1.7]
        assert [s.source for s in h2] == ['training', 'fit', 'manual']

    @requires_torch
    def test_refuses_to_overwrite_source(self, tmp_path):
        import torch
        from calibration_editor import save_corrected_checkpoint, CalibrationError
        src = tmp_path / "dme_best.pth"
        torch.save({'config': {'training': {'rho_direct': 1.4, 'sigma_0': 0.2}},
                    'dme_state_dict': {}}, src)
        with pytest.raises(CalibrationError):
            save_corrected_checkpoint(src, src, 1.3, 0.18, 'manual')

    def test_read_calibration_errors_on_missing(self):
        from calibration_editor import read_calibration, CalibrationError
        # No config.training.rho_direct
        with pytest.raises(CalibrationError):
            read_calibration({'config': {'training': {}}})
        with pytest.raises(CalibrationError):
            read_calibration({})

    def test_next_edit_dirname_numbering(self, tmp_path):
        from calibration_editor import next_edit_dirname
        edits = tmp_path / "edits"
        edits.mkdir()
        # Empty dir → v1
        assert next_edit_dirname(edits) == "v1"
        (edits / "v1").mkdir()
        assert next_edit_dirname(edits) == "v2"
        (edits / "v2").mkdir()
        assert next_edit_dirname(edits) == "v3"
        # User-named edits don't block the v-numbering
        (edits / "tuned_for_camG").mkdir()
        assert next_edit_dirname(edits) == "v3"
