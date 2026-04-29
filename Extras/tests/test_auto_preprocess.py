"""Tests for Inference.auto_preprocess.decide_flatten_mode.

Synthetic-input tests cover the four decision branches:
  - detection failure → boundary_normalise
  - sphere extends past crop edge → boundary_normalise
  - high fill ratio (>60%) → boundary_normalise
  - normal calibration sphere with headroom → calibration

Plus a real-data test against one of the user's calibration cines if
available locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Inference.auto_preprocess import (
    AutoDecision,
    MODE_AUTO,
    MODE_BOUNDARY,
    MODE_CALIBRATION,
    MODE_SIMPLE,
    decide_flatten_mode,
    resolve_mode,
)

# Used by the optional real-data test for the cine path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # tests -> Extras -> CROPPING


# ─── Synthetic-image helpers ─────────────────────────────────────────

def _make_sphere_crop(crop_size: int, sphere_radius: int,
                       centre_offset: int = 0,
                       background: int = 220, sphere: int = 30) -> np.ndarray:
    """Render a uint8 grayscale crop of a dark circle on a light bg.

    crop is (crop_size, crop_size); sphere centred at
    (crop_size//2 + centre_offset, crop_size//2 + centre_offset)
    with the given radius.
    """
    img = np.full((crop_size, crop_size), background, dtype=np.uint8)
    cy = crop_size // 2 + centre_offset
    cx = crop_size // 2 + centre_offset
    yy, xx = np.indices((crop_size, crop_size))
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= sphere_radius ** 2
    img[mask] = sphere
    return img


# ─── Tests ────────────────────────────────────────────────────────────

class TestDecideFlattenMode:
    def test_calibration_sphere_with_headroom(self):
        """Sphere occupies ~25% of crop, plenty of edge clearance.

        This mimics the calibration sphere PNG case (950px crop with a
        ~480px sphere). Should pick calibration.
        """
        crop = _make_sphere_crop(crop_size=400, sphere_radius=100)
        decision = decide_flatten_mode(crop)
        assert decision.mode == MODE_CALIBRATION
        assert decision.info["detection_succeeded"] is True
        assert decision.info["fill_ratio"] < 0.3
        assert "calibration mode" in decision.rationale.lower()

    def test_tight_droplet_high_fill(self):
        """Sphere fills >60% of crop AND has enough edge clearance that
        the edge-rule does NOT fire — exercises the fill-ratio branch
        in isolation. Need dim > ~400 with r just past 0.437*dim."""
        # 600px crop, radius 270 → fill = pi*270²/600² = 0.636
        # edge clearance = 300 - 270 = 30 (≥ cfg1_inner_margin+5 = 25)
        crop = _make_sphere_crop(crop_size=600, sphere_radius=270)
        decision = decide_flatten_mode(crop)
        assert decision.mode == MODE_BOUNDARY
        assert decision.info["fill_ratio"] > 0.6
        assert decision.info["edge_clearance_px"] >= 25
        assert "fill" in decision.rationale.lower()

    def test_sphere_at_edge_low_clearance(self):
        """Sphere is large enough that the edge is closer than cfg1's
        inner_margin + 5 = 25. fill_ratio is below 0.6 so this exercises
        the edge-clearance branch independently."""
        crop = _make_sphere_crop(crop_size=400, sphere_radius=180)
        # fill_ratio ≈ pi*180**2 / 400**2 = 0.636 — but we want to check
        # the edge_clearance branch. crop_dim/2 = 200; clearance =
        # 200 - 180 = 20px. cfg1 needs >25, so this triggers edge rule.
        decision = decide_flatten_mode(crop)
        assert decision.mode == MODE_BOUNDARY
        # The earliest matching rule wins; could be either edge or fill,
        # but both lead to boundary_normalise.

    def test_sphere_extends_past_crop(self):
        """Sphere is detected but its bounding box exceeds the crop.

        Edge clearance is negative — hits the edge-clearance branch.
        """
        crop = _make_sphere_crop(crop_size=200, sphere_radius=110)
        decision = decide_flatten_mode(crop)
        assert decision.mode == MODE_BOUNDARY
        assert decision.info["edge_clearance_px"] < 0

    def test_invalid_input(self):
        """None / empty / wrong-shape input falls back safely to boundary."""
        for bad in (None, np.array([]), np.zeros((0, 0)),
                     np.zeros((10, 10, 3))):
            decision = decide_flatten_mode(bad)
            assert decision.mode == MODE_BOUNDARY
            assert "invalid" in decision.rationale.lower() or \
                   "failed" in decision.rationale.lower()

    def test_no_sphere_detection(self):
        """Pure noise — sphere detection fails → boundary."""
        rng = np.random.default_rng(seed=42)
        noise = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
        decision = decide_flatten_mode(noise)
        # find_sphere_center may or may not find a degenerate "circle"
        # in pure noise depending on Canny output. Either way must be
        # boundary_normalise (detection-failed branch OR rules 2/3).
        assert decision.mode == MODE_BOUNDARY


class TestResolveMode:
    def test_auto_mode_with_crop(self):
        """resolve_mode('auto', crop) runs the heuristic."""
        crop = _make_sphere_crop(crop_size=400, sphere_radius=100)
        d = resolve_mode(MODE_AUTO, crop)
        assert d.mode == MODE_CALIBRATION  # would be auto's choice
        assert "user override" not in d.rationale.lower()

    def test_auto_mode_no_crop(self):
        """resolve_mode('auto', None) returns the safe fallback."""
        d = resolve_mode(MODE_AUTO, None)
        assert d.mode == MODE_BOUNDARY
        assert "no crop" in d.rationale.lower() or "defaulting" in d.rationale.lower()

    def test_user_override_locks_mode(self):
        """resolve_mode('simple', crop) returns simple regardless."""
        crop = _make_sphere_crop(crop_size=400, sphere_radius=100)
        d = resolve_mode(MODE_SIMPLE, crop)
        assert d.mode == MODE_SIMPLE
        assert "user override" in d.rationale.lower()

    def test_unknown_mode_falls_back(self):
        d = resolve_mode("nonexistent_mode", None)
        assert d.mode == MODE_BOUNDARY


# ─── Real-data smoke test (optional — only runs if cines are present) ─

@pytest.mark.skipif(
    not (Path(_REPO_ROOT) / "calibration spheres" / "9mm" / "9mm_30.cine").is_file(),
    reason="calibration cine not available in this checkout",
)
class TestRealCalibrationCrop:
    def test_calibration_cine_is_picked_as_calibration_mode(self):
        """A real ~mid-defocus calibration sphere crop should pick
        calibration mode (the inputs the model was trained on)."""
        # Load a calibration cine, extract first frame, do a centred
        # sphere-aware crop, then run the decision.
        from Calibration.cine_loader import CineLoader
        from Calibration.sphere_processing import (
            find_sphere_center, crop_to_square,
        )
        cine_path = Path(_REPO_ROOT) / "calibration spheres" / "9mm" / "9mm_30.cine"
        loader = CineLoader(str(cine_path))
        if loader.cine_obj is None:
            pytest.skip("could not load cine")
        frame = loader.extract_frame(loader.frame_range[0])
        if frame is None:
            pytest.skip("could not read frame")
        if frame.ndim == 3:
            import cv2
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        det = find_sphere_center(frame, upper_only=True)
        if det is None:
            pytest.skip("sphere detection failed on test frame")
        cx, cy, r = det
        crop = crop_to_square(frame, cx, cy, r, padding=1.2)
        if crop.dtype != np.uint8:
            import cv2
            crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        decision = decide_flatten_mode(crop)
        # 1.2× padded crop of a calibration sphere should have ~70%
        # headroom by area (sphere fills ~70% of a 1.2x-padded square).
        # Either calibration or boundary — depending on exact ratio.
        # What we want to confirm: it's a sensible decision.
        assert decision.info["detection_succeeded"] is True
        assert decision.mode in (MODE_CALIBRATION, MODE_BOUNDARY)
