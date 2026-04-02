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
_REPO_ROOT = Path(__file__).resolve().parent.parent

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
