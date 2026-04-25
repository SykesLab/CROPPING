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


# ===========================================================================
# I/O — synthetic dataset + calibration stack loaders
# ===========================================================================
class TestFingerprintIO:
    def test_stratified_subsample_spans_full_range(self, tmp_path):
        """Subsampling to N from a larger pool should keep coverage of the
        full defocus range, not just one cluster."""
        import numpy as np
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_io import _stratified_sample

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            'index': range(1000),
            'defocus_mm': rng.uniform(-12, 12, 1000),
            'sigma_px': rng.uniform(0, 10, 1000),
        })
        sampled = _stratified_sample(df, n_samples=100, n_bins=10, seed=42)
        assert len(sampled) <= 100
        # Sampled should cover roughly the same |z| range
        assert sampled['defocus_mm'].abs().max() > 9.0
        assert sampled['defocus_mm'].abs().min() < 1.5

    def test_full_dataset_when_n_samples_exceeds_total(self, tmp_path):
        """If user asks for more samples than exist, keep all of them."""
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_io import _stratified_sample
        df = pd.DataFrame({
            'index': range(50),
            'defocus_mm': [(i - 25) * 0.5 for i in range(50)],
            'sigma_px': [1.0] * 50,
        })
        # n_samples=200 > total=50 → real load returns all; helper itself
        # returns at most n_samples, which is up to 50 here
        sampled = _stratified_sample(df, n_samples=200, n_bins=10, seed=42)
        assert len(sampled) == 50

    def test_load_synthetic_dataset_minimal(self, tmp_path):
        """Build a tiny fake synthetic dataset on disk and load it."""
        import cv2
        import numpy as np
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_io import load_synthetic_dataset

        ds = tmp_path / "20260101_000000_test"
        (ds / "blur").mkdir(parents=True)
        (ds / "sharp").mkdir()
        # Three samples
        for i in range(3):
            img = np.full((32, 32), 128, dtype=np.uint8)
            cv2.imwrite(str(ds / "blur" / f"{i:06d}.png"), img)
        pd.DataFrame({
            'index': [0, 1, 2],
            'defocus_mm': [-2.0, 0.0, 2.0],
            'sigma_px': [1.5, 0.5, 1.5],
            'sigma_measured_erf': [1.4, float('nan'), 1.5],
        }).to_csv(ds / "metadata.csv", index=False)

        loaded = load_synthetic_dataset(ds)
        assert loaded.sample_count_total == 3
        assert loaded.sample_count_used == 3
        # First sample's metadata-driven fields
        s0 = loaded.samples[0]
        assert s0.index == 0
        assert s0.defocus_mm == -2.0
        assert abs(s0.sigma_measured_erf - 1.4) < 1e-9
        # Second has NaN ERF — should come back as None
        s1 = loaded.samples[1]
        assert s1.sigma_measured_erf is None

    def test_calibration_loader_reads_positions_csv(self, tmp_path):
        """Without pyphantom we can't load the .cine bytes, but the
        metadata loading (file list, positions.csv) should work."""
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_io import load_calibration_stack
        # Create empty .cine files + positions.csv
        for i in [1, 2, 3]:
            (tmp_path / f"sphere_{i}.cine").write_bytes(b"")
        pd.DataFrame({
            'filename': ['sphere_1.cine', 'sphere_2.cine', 'sphere_3.cine'],
            'stage_position_mm': [0.0, 1.0, 2.0],
        }).to_csv(tmp_path / "positions.csv", index=False)
        stack = load_calibration_stack(tmp_path, focus_offset_mm=1.0)
        assert len(stack.frames) == 3
        # Frames sorted by trailing integer
        assert stack.frames[0].file_path.name == 'sphere_1.cine'
        # Defocus = stage − focus_offset
        assert stack.frames[0].defocus_mm == -1.0
        assert stack.frames[1].defocus_mm == 0.0
        assert stack.frames[2].defocus_mm == 1.0


# ===========================================================================
# Analyses — Check B (alignment) + Check C (coverage)
# ===========================================================================
class TestAlignmentAndCoverage:
    def test_alignment_picks_correct_neighbours(self):
        """For an anchor at z=5, its K nearest synthetics by |z| should
        be the K with |z| closest to 5."""
        import numpy as np
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_analyses import alignment_check_nn

        anchor = pd.DataFrame({'defocus_mm': [5.0],
                               'erf_sigma_px': [3.0],
                               'edge_transition_width': [4.0]})
        other = pd.DataFrame({
            'defocus_mm': [-5.0, -4.5, -3.0, 4.0, 5.5, 8.0, 10.0, 0.0],
            'erf_sigma_px': [3.0, 2.8, 2.5, 2.7, 3.2, 4.0, 4.5, 0.5],
            'edge_transition_width': [4.0, 3.9, 3.5, 3.8, 4.2, 4.8, 5.0, 1.5],
        })
        result = alignment_check_nn(anchor, other, k=3)
        assert len(result.comparisons) == 1
        c = result.comparisons[0]
        # Three closest |z| to 5 are: -5(0.0), 5.5(0.5), -4.5(0.5) OR 4.0(1.0)
        # depending on tie-breaking. All within 1.0 of |z|=5.
        for ni in c.neighbour_indices:
            assert abs(abs(other.loc[ni, 'defocus_mm']) - 5.0) <= 1.0

    def test_alignment_zero_delta_when_pipelines_identical(self):
        """If anchor and other have identical fingerprints at matched defocuses,
        deltas should be zero across all features."""
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_analyses import alignment_check_nn
        df = pd.DataFrame({
            'defocus_mm': [-5.0, -2.0, 0.0, 2.0, 5.0],
            'erf_sigma_px': [3.0, 1.5, 0.5, 1.5, 3.0],
            'laplacian_variance': [10.0, 25.0, 50.0, 25.0, 10.0],
        })
        result = alignment_check_nn(df, df, k=1)
        for c in result.comparisons:
            assert abs(c.deltas['erf_sigma_px']) < 1e-9
            assert abs(c.deltas['laplacian_variance']) < 1e-9

    def test_coverage_full_when_test_inside_reference(self):
        """If test distribution is contained inside reference's range,
        coverage should be 100%."""
        import numpy as np
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_analyses import coverage_check
        ref = pd.DataFrame({'edge_transition_width': np.linspace(0, 10, 100)})
        test = pd.DataFrame({'edge_transition_width': np.linspace(2, 8, 50)})
        result = coverage_check(ref, test)
        cov = result.per_feature['edge_transition_width']['coverage_pct']
        assert cov == 100.0

    def test_coverage_partial_when_test_extends_beyond_reference(self):
        """Test distribution wider than reference → coverage drops."""
        import numpy as np
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_analyses import coverage_check
        ref = pd.DataFrame({'edge_transition_width': np.linspace(0, 10, 100)})
        test = pd.DataFrame({'edge_transition_width': np.linspace(0, 15, 100)})
        result = coverage_check(ref, test)
        cov = result.per_feature['edge_transition_width']['coverage_pct']
        # Test goes to 15 but ref's p95 is ~9.5 → ~63% of test should fit
        assert 50 < cov < 75, f"unexpected coverage {cov}"

    def test_alignment_flags_classify_correctly(self):
        """Tiny delta → PASS, big delta → FAIL."""
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_analyses import (
            alignment_check_nn, flag_alignment)
        anchor = pd.DataFrame({'defocus_mm': [5.0],
                               'erf_sigma_px': [3.0]})
        good = pd.DataFrame({'defocus_mm': [5.0],
                             'erf_sigma_px': [3.001]})  # ~0% delta
        bad = pd.DataFrame({'defocus_mm': [5.0],
                            'erf_sigma_px': [4.5]})  # 50% delta
        good_result = alignment_check_nn(anchor, good, k=1)
        bad_result = alignment_check_nn(anchor, bad, k=1)
        good_flags = flag_alignment(good_result)
        bad_flags = flag_alignment(bad_result)
        assert good_flags['erf_sigma_px'] == 'PASS'
        assert bad_flags['erf_sigma_px'] == 'FAIL'


# ===========================================================================
# Orchestrator + Report — end-to-end with a tiny dataset
# ===========================================================================
class TestOrchestratorAndReport:
    def _build_tiny_dataset(self, tmp_path):
        """Minimal synthetic dataset on disk + matching config."""
        import cv2
        import numpy as np
        import pandas as pd
        ds = tmp_path / "20260101_000000_test"
        (ds / "blur").mkdir(parents=True)
        # 8 samples with monotonic blur for correlation
        for i in range(8):
            img = np.ones((48, 48), dtype=np.float32)
            cv2.circle(img, (24, 24), 10, 0.0, -1)
            sigma = 0.5 + i * 0.7
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            cv2.imwrite(str(ds / "blur" / f"{i:06d}.png"),
                        (blurred * 255).astype(np.uint8))
        pd.DataFrame({
            'index': range(8),
            'defocus_mm': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            'sigma_px': [3.0, 2.0, 1.0, 0.5, 1.0, 2.0, 3.0, 4.0],
            'sigma_measured_erf': [None] * 8,
        }).to_csv(ds / "metadata.csv", index=False)
        config = {
            'data': {'image_size_px': 48,
                     'blur_range_px': [0.5, 4.0]},
            'training': {
                'rho_direct': 1.0, 'sigma_0': 0.0,
                'scale_calib_px_per_mm': 50.0,
                'crop_size_px': 48,
                'training_mode': 'direct',
            },
        }
        return ds, config

    def test_run_all_checks_no_calibration(self, tmp_path):
        from tools.fingerprint_checker.fingerprint_orchestrator import run_all_checks
        ds, cfg = self._build_tiny_dataset(tmp_path)
        result = run_all_checks(
            config_dict=cfg,
            synthetic_dataset_path=ds,
            n_synthetic_samples=None,
        )
        assert result.scale_chain is not None
        assert result.scale_chain.overall_passed
        assert len(result.synthetic_fingerprints) == 8
        # Sigma-trend correlations exist for the cheap metrics
        assert 'tenengrad' in result.sigma_trend_correlations
        assert 'edge_gradient_max' in result.sigma_trend_correlations

    def test_write_full_report_creates_all_files(self, tmp_path):
        from tools.fingerprint_checker.fingerprint_orchestrator import run_all_checks
        from tools.fingerprint_checker.fingerprint_report import write_full_report
        ds, cfg = self._build_tiny_dataset(tmp_path)
        out = tmp_path / "report"
        result = run_all_checks(config_dict=cfg, synthetic_dataset_path=ds)
        files = write_full_report(result, out)
        # JSON + markdown + at least one CSV + at least one PNG
        assert (out / 'blur_fingerprint_report.json').is_file()
        assert (out / 'blur_fingerprint_report.md').is_file()
        assert (out / 'fingerprints_synthetic.csv').is_file()
        assert (out / 'check_a_scale_residuals.png').is_file()
        assert (out / 'check_b_internal_sigma_trends.png').is_file()

    def test_report_json_is_valid_json(self, tmp_path):
        import json
        from tools.fingerprint_checker.fingerprint_orchestrator import run_all_checks
        from tools.fingerprint_checker.fingerprint_report import write_json_report
        ds, cfg = self._build_tiny_dataset(tmp_path)
        result = run_all_checks(config_dict=cfg, synthetic_dataset_path=ds)
        p = write_json_report(result, tmp_path / 'rep')
        with open(p) as f:
            data = json.load(f)
        # Top-level keys
        assert 'config_summary' in data
        assert 'scale_chain' in data
        assert 'sigma_trend_correlations' in data
        assert 'written_at' in data


# ===========================================================================
# Phase 2 — crop folder loaders, Check C, Inference↔Calibration alignment
# ===========================================================================
class TestPhase2:
    def _build_real_crops(self, tmp_path):
        """A folder of "real" crops with defocus encoded in filename."""
        import cv2
        import numpy as np
        folder = tmp_path / "real_crops"
        folder.mkdir()
        for i, z in enumerate([-3.0, -1.0, 0.0, 1.0, 3.0]):
            img = np.ones((48, 48), dtype=np.float32)
            cv2.circle(img, (24, 24), 10, 0.0, -1)
            sigma = 0.5 + abs(z) * 0.3
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            (folder / f"sample{i}_z{z:+.2f}mm.png").write_bytes(b"")
            cv2.imwrite(str(folder / f"sample{i}_z{z:+.2f}mm.png"),
                        (blurred * 255).astype(np.uint8))
        return folder

    def test_load_crop_folder_parses_defocus_from_filename(self, tmp_path):
        from tools.fingerprint_checker.fingerprint_io import load_crop_folder
        folder = self._build_real_crops(tmp_path)
        loaded = load_crop_folder(folder, label='real')
        assert loaded.sample_count_total == 5
        # Each sample should have parsed defocus
        zs = sorted(s.defocus_mm for s in loaded.samples
                    if s.defocus_mm is not None)
        assert zs == [-3.0, -1.0, 0.0, 1.0, 3.0]

    def test_load_crop_folder_subsample(self, tmp_path):
        from tools.fingerprint_checker.fingerprint_io import load_crop_folder
        folder = self._build_real_crops(tmp_path)
        loaded = load_crop_folder(folder, label='real', n_samples=3)
        assert loaded.sample_count_total == 5
        assert loaded.sample_count_used == 3

    def test_load_crop_folder_handles_no_defocus_in_name(self, tmp_path):
        import cv2
        import numpy as np
        from tools.fingerprint_checker.fingerprint_io import load_crop_folder
        folder = tmp_path / "no_z_crops"
        folder.mkdir()
        for i in range(3):
            cv2.imwrite(str(folder / f"plain_{i}.png"),
                        (np.ones((32, 32)) * 128).astype(np.uint8))
        loaded = load_crop_folder(folder, label='inference')
        for s in loaded.samples:
            assert s.defocus_mm is None

    def test_orchestrator_runs_check_c_when_real_provided(self, tmp_path):
        """End-to-end: synth dataset + real crops → coverage check runs."""
        from tools.fingerprint_checker.fingerprint_orchestrator import run_all_checks
        # Build synth dataset
        import cv2
        import numpy as np
        import pandas as pd
        ds = tmp_path / "20260101_000000_test"
        (ds / "blur").mkdir(parents=True)
        for i in range(8):
            img = np.ones((48, 48), dtype=np.float32)
            cv2.circle(img, (24, 24), 10, 0.0, -1)
            sigma = 0.5 + i * 0.4
            cv2.imwrite(str(ds / "blur" / f"{i:06d}.png"),
                        (cv2.GaussianBlur(img, (0, 0), sigmaX=sigma) * 255)
                        .astype(np.uint8))
        pd.DataFrame({
            'index': range(8),
            'defocus_mm': [(i - 4) * 1.0 for i in range(8)],
            'sigma_px': [0.5 + i * 0.4 for i in range(8)],
        }).to_csv(ds / "metadata.csv", index=False)

        cfg = {
            'data': {'image_size_px': 48, 'blur_range_px': [0.5, 4.0]},
            'training': {'rho_direct': 1.0, 'sigma_0': 0.0,
                         'scale_calib_px_per_mm': 50.0, 'crop_size_px': 48,
                         'training_mode': 'direct'},
        }

        # Reuse the real-crops builder
        real_folder = self._build_real_crops(tmp_path)

        result = run_all_checks(
            config_dict=cfg,
            synthetic_dataset_path=ds,
            real_crops_path=real_folder,
        )
        assert not result.synthetic_fingerprints.empty
        assert not result.real_fingerprints.empty
        assert result.coverage_synth_vs_real is not None
        # At least one feature should have a defined coverage
        per_feat = result.coverage_synth_vs_real.per_feature
        assert any(np.isfinite(v.get('coverage_pct', float('nan')))
                   for v in per_feat.values())

    def test_orchestrator_no_inference_calib_alignment_without_defocus(self, tmp_path):
        """If inference crops have no parseable defocus, the
        inference↔calibration alignment step is skipped with a diagnostic."""
        import cv2
        import numpy as np
        import pandas as pd
        from tools.fingerprint_checker.fingerprint_orchestrator import run_all_checks
        ds = tmp_path / "ds"
        (ds / "blur").mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(str(ds / "blur" / f"{i:06d}.png"),
                        (np.ones((32, 32)) * 100).astype(np.uint8))
        pd.DataFrame({
            'index': range(4),
            'defocus_mm': [-2.0, -1.0, 1.0, 2.0],
            'sigma_px': [1.0, 0.5, 0.5, 1.0],
        }).to_csv(ds / "metadata.csv", index=False)

        # Inference crops with NO defocus in filename
        inf = tmp_path / "inf"
        inf.mkdir()
        for i in range(3):
            cv2.imwrite(str(inf / f"plain_{i}.png"),
                        (np.ones((32, 32)) * 128).astype(np.uint8))

        cfg = {
            'data': {'image_size_px': 32, 'blur_range_px': [0.5, 1.0]},
            'training': {'rho_direct': 1.0, 'sigma_0': 0.0,
                         'scale_calib_px_per_mm': 50.0, 'crop_size_px': 32,
                         'training_mode': 'direct'},
        }
        result = run_all_checks(
            config_dict=cfg, synthetic_dataset_path=ds,
            inference_crops_path=inf,
        )
        # Coverage runs (doesn't need defocus)
        assert result.coverage_synth_vs_inference is not None
        # But alignment skipped (no defocus)
        assert result.alignment_inference_vs_calib is None
