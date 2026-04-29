"""auto_preprocess — decide which flatten mode best fits a given crop.

Pure logic, no Tk. Inputs: a crop array. Returns an ``AutoDecision``
with the recommended mode + a human-readable rationale.

Decision rule (in order of precedence):

  1. If sphere detection fails → 'boundary_normalise' (most robust to
     heavily-defocused or noisy inputs).
  2. If detected sphere extends within (cfg1_inner_margin + 5) pixels
     of the crop edge → 'boundary_normalise' (cfg1 would zero out
     most of the interior — see the 4mm borosilicate bug from
     2026-04-28).
  3. If sphere fill ratio (π·r² / crop_area) > 0.6 → 'boundary_normalise'
     (tight droplet framing).
  4. Otherwise → 'calibration' (cfg1 — preserves blur fidelity for
     calibration sphere inputs with headroom).

The modes returned are the values used by ``Inference.inference_engine
.preprocess_crop()``: ``"calibration"``, ``"boundary_normalise"``,
``"simple"``, ``"none"``. The auto-decide function never returns
``"simple"`` or ``"none"`` — those are user-override-only choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from Calibration.sphere_processing import find_sphere_center


# Mode constants — must match the values inference_engine.preprocess_crop reads
MODE_CALIBRATION = "calibration"
MODE_BOUNDARY = "boundary_normalise"
MODE_SIMPLE = "simple"
MODE_NONE = "none"
MODE_AUTO = "auto"

# All valid override choices (the dropdown values in the UI)
ALL_MODES = (MODE_AUTO, MODE_CALIBRATION, MODE_BOUNDARY, MODE_SIMPLE, MODE_NONE)


@dataclass
class AutoDecision:
    """Recorded outcome of decide_flatten_mode.

    Attributes
    ----------
    mode : str
        Recommended flatten mode (one of MODE_*, never MODE_AUTO).
    rationale : str
        One-line human-readable explanation suitable for direct UI display.
    info : dict
        Diagnostic numbers — fill_ratio, edge_clearance, sphere_radius,
        crop_dim, detection_succeeded. Useful for tooltips / debugging.
    """
    mode: str
    rationale: str
    info: dict = field(default_factory=dict)


def decide_flatten_mode(
    crop: np.ndarray,
    cfg1_inner_margin: int = 20,
    fill_ratio_threshold: float = 0.6,
) -> AutoDecision:
    """Pick the best flatten mode for the given crop.

    Parameters
    ----------
    crop : np.ndarray
        Grayscale crop (uint8 or float). 2D image.
    cfg1_inner_margin : int
        The inner_margin used by calibration (cfg1) mode. Determines
        how much edge clearance cfg1 needs to be useful.
    fill_ratio_threshold : float
        If sphere fill ratio exceeds this, cfg1 is unsuitable (sphere
        too tightly framed) → boundary_normalise.

    Returns
    -------
    AutoDecision
    """
    if crop is None or crop.ndim != 2 or crop.size == 0:
        return AutoDecision(
            mode=MODE_BOUNDARY,
            rationale="invalid crop input → boundary_normalise (most robust)",
            info={"detection_succeeded": False, "reason": "invalid_input"},
        )

    h, w = crop.shape[:2]
    crop_area = float(h * w)
    crop_dim = min(h, w)

    # --- Run sphere detection ---
    det = find_sphere_center(crop, upper_only=True)
    if det is None:
        return AutoDecision(
            mode=MODE_BOUNDARY,
            rationale="sphere detection failed → boundary_normalise (more "
                       "forgiving on heavily-defocused inputs)",
            info={"detection_succeeded": False, "crop_dim": crop_dim},
        )

    cx, cy, radius = det
    sphere_area = float(np.pi * radius * radius)
    fill_ratio = sphere_area / crop_area if crop_area > 0 else 0.0

    # Edge clearance: minimum distance from the sphere's bounding circle
    # to any crop edge. We want the smallest of (cx - r, w - cx - r,
    # cy - r, h - cy - r). All values can be negative if the sphere
    # extends past the crop edge.
    edge_clearance = float(min(
        cx - radius,
        w - cx - radius,
        cy - radius,
        h - cy - radius,
    ))

    info = {
        "detection_succeeded": True,
        "sphere_radius_px": float(radius),
        "sphere_centre_xy": (int(cx), int(cy)),
        "crop_dim_px": crop_dim,
        "crop_area_px2": crop_area,
        "sphere_area_px2": sphere_area,
        "fill_ratio": fill_ratio,
        "edge_clearance_px": edge_clearance,
        "cfg1_inner_margin": cfg1_inner_margin,
    }

    # --- Decision rules ---

    # Rule 2: edge clearance too small for cfg1's inner_margin to work
    cfg1_min_clearance = cfg1_inner_margin + 5
    if edge_clearance < cfg1_min_clearance:
        return AutoDecision(
            mode=MODE_BOUNDARY,
            rationale=(
                f"sphere edge is only {edge_clearance:.0f}px from crop "
                f"boundary (calibration-mode needs >{cfg1_min_clearance}px) "
                f"→ boundary_normalise"
            ),
            info=info,
        )

    # Rule 3: high fill ratio = tightly-framed droplet
    if fill_ratio > fill_ratio_threshold:
        return AutoDecision(
            mode=MODE_BOUNDARY,
            rationale=(
                f"sphere fills {fill_ratio*100:.0f}% of crop "
                f"(>{fill_ratio_threshold*100:.0f}% threshold) → tight "
                f"framing, use boundary_normalise"
            ),
            info=info,
        )

    # Rule 4: default — calibration mode preserves blur fidelity when
    # the sphere has headroom
    return AutoDecision(
        mode=MODE_CALIBRATION,
        rationale=(
            f"sphere fills {fill_ratio*100:.0f}% of crop with "
            f"{edge_clearance:.0f}px headroom → calibration mode "
            f"(preserves blur fidelity)"
        ),
        info=info,
    )


def resolve_mode(setting: str, crop: Optional[np.ndarray]) -> AutoDecision:
    """Translate a user-facing setting (which may be 'auto') into a
    concrete decision.

    If ``setting`` is ``MODE_AUTO``, runs the heuristic on ``crop``.
    Otherwise returns a decision wrapping the user's manual choice with
    a "user-locked" rationale so the UI can display it consistently.
    """
    if setting == MODE_AUTO:
        if crop is None:
            return AutoDecision(
                mode=MODE_BOUNDARY,
                rationale="no crop available yet → defaulting to boundary_normalise",
                info={"detection_succeeded": False, "reason": "no_crop"},
            )
        return decide_flatten_mode(crop)
    if setting in ALL_MODES:
        return AutoDecision(
            mode=setting,
            rationale=f"user override: {setting} (auto-detect bypassed)",
            info={"user_override": True},
        )
    # Unknown — fall back conservatively
    return AutoDecision(
        mode=MODE_BOUNDARY,
        rationale=f"unknown mode '{setting}' → falling back to boundary_normalise",
        info={"unknown_setting": setting},
    )
