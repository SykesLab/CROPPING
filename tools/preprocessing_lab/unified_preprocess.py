"""Candidate `unified_preprocess()` recipe.

This is a SHIM. It uses existing project functions
(`crop_to_square`, `flatten_sphere_crop`) without modifying them, and
implements the candidate "unified" preprocessing recipe under
evaluation:

  1. Crop to square at user-chosen padding (defaults to existing
     calibration GUI's 1.2× radius).
  2. Apply `flatten_sphere_crop` with feather computed from a
     model-space target (so the same feather looks the same in model
     space regardless of source resolution).
  3. Resize to model_size (final step — all pipelines should go
     through the same downsampling).

If user testing shows this recipe gives good model behaviour, it
becomes the basis for production changes (separate plan).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project-relative path setup — required so we can import from
# Calibration/, Training/ regardless of where the user runs from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

from sphere_processing import crop_to_square, flatten_sphere_crop


def unified_preprocess(
    raw_frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    target_feather_model_px: float = 30.0,
    model_size: int = 256,
    crop_padding: float = 1.2,
    inner_margin_model_px: float = 0.0,
) -> np.ndarray:
    """Process one raw .cine frame through the candidate recipe.

    Args:
        raw_frame: float32 image at native source resolution.
        cx, cy, radius: sphere geometry (typically the consensus
            geometry from `find_consensus_sphere`).
        target_feather_model_px: feather width SPECIFIED in model-space
            pixels. Will be converted to source pixels at runtime so
            the post-resize appearance is consistent across pipelines.
        model_size: final image side length (e.g. 256).
        crop_padding: passed to `crop_to_square`. Default 1.2 matches
            the existing calibration GUI.
        inner_margin_model_px: in case interior preservation is desired
            (default 0 = full binary fill).

    Returns:
        uint8 array of shape (model_size, model_size). Ready to feed to
        the trained model or save as PNG.
    """
    # Step 1: crop to square around the sphere
    crop = crop_to_square(raw_frame, cx, cy, radius, padding=crop_padding)
    if crop.dtype == np.uint8:
        crop_f = crop.astype(np.float32) / 255.0
    else:
        crop_f = crop.astype(np.float32)
        if crop_f.max() > 1.5:
            crop_f = crop_f / 255.0

    # Step 2: compute source-pixel feather from the model-space target
    h, w = crop_f.shape[:2]
    source_size = max(h, w)
    feather_src = max(1, round(target_feather_model_px * source_size / model_size))
    inner_src = max(0, round(inner_margin_model_px * source_size / model_size))

    # Step 3: flatten at source resolution
    flat, info = flatten_sphere_crop(
        crop_f,
        feather=feather_src,
        inner_margin=inner_src,
        flatten_exterior=True,
    )
    if info is None:
        # Detection failed — return the un-flattened crop, resized
        flat = crop_f

    # Step 4: resize to model size as final step
    interp = (cv2.INTER_AREA if max(flat.shape) > model_size
              else cv2.INTER_CUBIC)
    resized = cv2.resize(
        np.clip(flat, 0, 1),
        (model_size, model_size),
        interpolation=interp,
    )
    return (resized * 255).astype(np.uint8)


def derived_feather_source_px(
    source_size: int,
    target_feather_model_px: float = 30.0,
    model_size: int = 256,
) -> int:
    """Convenience: what feather (source-px) corresponds to the
    chosen model-space target for a given source_size."""
    return max(1, round(target_feather_model_px * source_size / model_size))
