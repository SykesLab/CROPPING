"""15 per-image metrics for blur/edge/intensity/geometry fingerprinting.

Each metric is a pure function: ``image (HxW float32 in [0,1]) → scalar``
or, where useful, a dict of named scalars.

Categorisation matches the plan file:
    Blur/edge:     1-5
    Focus:         6-8
    Intensity:     9-11
    Geometry:     12-14
    Domain:        15

Subject-independent (used in cross-pipeline alignment): 1-11, 15
Subject-dependent (used only for distribution coverage): 12-14
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Need cv2 for filters; calibration's blur_measurement for ERF fitting
import cv2  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blur_measurement import measure_blur_erf  # noqa: E402


# ── Subject mask helpers ─────────────────────────────────────────────────


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Coerce input to float32 in [0,1]. Handles uint8 [0,255] and float [0,1]."""
    if image.ndim != 2:
        if image.ndim == 3 and image.shape[2] in (1, 3, 4):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 \
                else image[..., 0]
        else:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    img = image.astype(np.float32)
    if img.max() > 1.5:  # heuristic: still in [0,255] range
        img = img / 255.0
    return img


def detect_object_mask(
    image: np.ndarray,
    polarity: Optional[str] = None,
) -> np.ndarray:
    """Otsu-based object mask. Returns boolean array, True where the object is.

    polarity: 'dark_on_light' / 'light_on_dark' / None (auto-detect).
    """
    img01 = _to_float01(image)
    img8 = (img01 * 255).astype(np.uint8)
    _, mask = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_bool = mask > 127
    # Auto-determine polarity if not provided
    if polarity is None:
        # If "True" pixels (where mask>127, i.e. brighter half) are FEWER than
        # "False" pixels, the object is the bright minority → light_on_dark
        # Otherwise the object is the dark minority → dark_on_light
        if mask_bool.sum() < (~mask_bool).sum():
            polarity = 'light_on_dark'
        else:
            polarity = 'dark_on_light'
    if polarity == 'dark_on_light':
        # Object = the DARK pixels = where mask was below threshold = ~mask_bool
        return ~mask_bool
    else:
        return mask_bool


# ── Metric 1-5: blur / edge ──────────────────────────────────────────────


def metric_erf_sigma(image: np.ndarray) -> Dict[str, float]:
    """Metric #1 — ERF sigma via Calibration's measurement code.

    Returns {'erf_sigma_px', 'erf_r_squared', 'erf_radius_px'}; values may
    be NaN if the fit failed.
    """
    img01 = _to_float01(image)
    try:
        result = measure_blur_erf(img01, num_rays=36, verbose=False)
        return {
            'erf_sigma_px': float(result.sigma_px) if result.sigma_px is not None else float('nan'),
            'erf_r_squared': float(result.r_squared) if result.r_squared is not None else float('nan'),
            'erf_radius_px': float(result.radius_px) if result.radius_px is not None else float('nan'),
        }
    except Exception:
        return {
            'erf_sigma_px': float('nan'),
            'erf_r_squared': float('nan'),
            'erf_radius_px': float('nan'),
        }


def metric_edge_transition_width(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #2 — 10% to 90% edge transition width in pixels.

    Walks radially outward from object centre, extracts the intensity profile
    along the steepest gradient, computes pixel distance between the
    10% and 90% intensity values. Robust when ERF fit fails.
    """
    img01 = _to_float01(image)
    if mask is None:
        mask = detect_object_mask(img01)
    if mask.sum() < 10:
        return float('nan')
    # Find object centre + approx radius
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    radius = np.sqrt(mask.sum() / np.pi)
    # Sample radial profile along right axis (any direction with edge crossing)
    h, w = img01.shape
    r_samples = np.arange(int(max(0, cy)), int(min(h, cy + 2 * radius)))
    if len(r_samples) < 5:
        return float('nan')
    profile = img01[r_samples, int(cx)] if 0 <= int(cx) < w else None
    if profile is None or len(profile) < 5:
        return float('nan')
    # Identify edge midpoint as the steepest gradient
    grad = np.abs(np.diff(profile))
    if grad.max() < 1e-6:
        return float('nan')
    mid_idx = int(np.argmax(grad))
    # Extract a window around the edge
    window = 6
    lo = max(0, mid_idx - window)
    hi = min(len(profile), mid_idx + window + 1)
    sub = profile[lo:hi]
    if len(sub) < 4:
        return float('nan')
    # Normalise sub to [0, 1] within its own range
    s_min, s_max = float(np.min(sub)), float(np.max(sub))
    if (s_max - s_min) < 1e-6:
        return float('nan')
    sub_n = (sub - s_min) / (s_max - s_min)
    # Find indices crossing 10% and 90% (linear interpolation)
    def _crossing(target: float) -> Optional[float]:
        for i in range(len(sub_n) - 1):
            a, b = sub_n[i], sub_n[i + 1]
            if (a - target) * (b - target) <= 0 and a != b:
                return i + (target - a) / (b - a)
        return None
    p10 = _crossing(0.1)
    p90 = _crossing(0.9)
    if p10 is None or p90 is None:
        return float('nan')
    return float(abs(p90 - p10))


def metric_edge_gradient_max(image: np.ndarray) -> float:
    """Metric #3 — maximum gradient magnitude in the image.

    Sharper edges → larger max gradient.
    """
    img01 = _to_float01(image)
    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad.max())


def metric_spatial_blur_uniformity(
    image: np.ndarray,
) -> Dict[str, float]:
    """Metric #4 — 4-quadrant ERF sigma variance.

    Splits the image into 4 quadrants, fits ERF in each, returns the
    variance + ratio of (max sigma) / (min sigma) across the quadrants.
    Probes whether blur is uniform across the crop or spatially varying.

    Returns {'quadrant_sigma_var', 'quadrant_sigma_max_min_ratio',
             'quadrants_fit_count'} (count = 0..4 successful fits).
    """
    img01 = _to_float01(image)
    h, w = img01.shape
    quads = [
        img01[:h // 2, :w // 2],
        img01[:h // 2, w // 2:],
        img01[h // 2:, :w // 2],
        img01[h // 2:, w // 2:],
    ]
    sigmas = []
    for q in quads:
        if q.size < 100:
            continue
        try:
            r = measure_blur_erf(q, num_rays=12, verbose=False)
            if r.sigma_px is not None and r.r_squared > 0.5:
                sigmas.append(float(r.sigma_px))
        except Exception:
            continue
    if len(sigmas) < 2:
        return {
            'quadrant_sigma_var': float('nan'),
            'quadrant_sigma_max_min_ratio': float('nan'),
            'quadrants_fit_count': float(len(sigmas)),
        }
    arr = np.array(sigmas)
    return {
        'quadrant_sigma_var': float(arr.var()),
        'quadrant_sigma_max_min_ratio': float(arr.max() / max(arr.min(), 1e-6)),
        'quadrants_fit_count': float(len(sigmas)),
    }


def metric_edge_symmetry(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Metric #5 — symmetry of edge profiles between opposite sides.

    Compares left-vs-right and top-vs-bottom radial intensity profiles
    around the detected object via L1 distance after normalisation.
    Higher distance = more asymmetric PSF.

    Returns {'edge_symmetry_lr_l1', 'edge_symmetry_tb_l1'} (NaN if not
    measurable).
    """
    img01 = _to_float01(image)
    if mask is None:
        mask = detect_object_mask(img01)
    if mask.sum() < 10:
        return {
            'edge_symmetry_lr_l1': float('nan'),
            'edge_symmetry_tb_l1': float('nan'),
        }
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    radius = np.sqrt(mask.sum() / np.pi)
    h, w = img01.shape
    span = int(min(radius * 2, h // 2, w // 2))
    if span < 4:
        return {
            'edge_symmetry_lr_l1': float('nan'),
            'edge_symmetry_tb_l1': float('nan'),
        }

    def _profile(start_y: int, start_x: int, dy: int, dx: int) -> np.ndarray:
        out = []
        for k in range(span):
            y = start_y + dy * k
            x = start_x + dx * k
            if 0 <= y < h and 0 <= x < w:
                out.append(img01[y, x])
            else:
                break
        return np.array(out, dtype=np.float32)

    cy_i, cx_i = int(cy), int(cx)
    left = _profile(cy_i, cx_i, 0, -1)
    right = _profile(cy_i, cx_i, 0, 1)
    top = _profile(cy_i, cx_i, -1, 0)
    bottom = _profile(cy_i, cx_i, 1, 0)

    def _normalise(p: np.ndarray) -> np.ndarray:
        if len(p) == 0:
            return p
        lo, hi = float(p.min()), float(p.max())
        if hi - lo < 1e-6:
            return np.zeros_like(p)
        return (p - lo) / (hi - lo)

    def _compare(a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        if n < 4:
            return float('nan')
        return float(np.mean(np.abs(_normalise(a[:n]) - _normalise(b[:n]))))

    return {
        'edge_symmetry_lr_l1': _compare(left, right),
        'edge_symmetry_tb_l1': _compare(top, bottom),
    }


# ── Metric 6-8: focus / sharpness ────────────────────────────────────────


def metric_laplacian_variance(image: np.ndarray) -> float:
    """Metric #6 — variance of the Laplacian. Classic sharpness proxy.

    Constant image → 0. Sharp edges → high values.
    """
    img01 = _to_float01(image)
    lap = cv2.Laplacian(img01, cv2.CV_32F)
    return float(lap.var())


def metric_tenengrad(image: np.ndarray) -> float:
    """Metric #7 — Tenengrad: mean(Sobel_x^2 + Sobel_y^2).

    Same family as Laplacian variance; matches the dissertation's
    classical-baseline focus measure.
    """
    img01 = _to_float01(image)
    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def metric_high_freq_energy_ratio(
    image: np.ndarray, threshold_fraction: float = 0.25,
) -> float:
    """Metric #8 — fraction of FFT power above a normalised radial threshold.

    threshold_fraction: cutoff radius as a fraction of Nyquist (0..0.5).
    Defaults to 0.25 = mid-frequency. High value → image has lots of HF
    content (sharp). Low value → mostly low frequencies (blurred).
    """
    img01 = _to_float01(image)
    h, w = img01.shape
    f = np.fft.fft2(img01)
    f_shift = np.fft.fftshift(f)
    power = np.abs(f_shift) ** 2
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices(power.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (min(h, w) / 2.0)  # normalised so 1.0 = Nyquist
    total = power.sum()
    if total < 1e-12:
        return float('nan')
    high = power[r_norm > threshold_fraction].sum()
    return float(high / total)


# ── Metric 9-11: intensity / contrast ────────────────────────────────────


def _bg_obj_stats(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Helper returning (bg_mean, bg_std, obj_mean, obj_std)."""
    img01 = _to_float01(image)
    if mask is None:
        mask = detect_object_mask(img01)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return (float('nan'),) * 4
    obj = img01[mask]
    bg = img01[~mask]
    return (
        float(bg.mean()), float(bg.std()),
        float(obj.mean()), float(obj.std()),
    )


def metric_background_mean(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #9 — mean intensity outside the detected object."""
    bg_mean, _, _, _ = _bg_obj_stats(image, mask)
    return bg_mean


def metric_background_std(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #10 — std of intensity outside the detected object (noise floor)."""
    _, bg_std, _, _ = _bg_obj_stats(image, mask)
    return bg_std


def metric_object_bg_contrast(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #11 — signed contrast (obj_mean − bg_mean) / bg_std.

    Sign captures polarity. NaN if bg_std == 0.
    """
    bg_mean, bg_std, obj_mean, _ = _bg_obj_stats(image, mask)
    if not np.isfinite(bg_std) or bg_std < 1e-6:
        return float('nan')
    return (obj_mean - bg_mean) / bg_std


# ── Metric 12-14: geometry (subject-dependent) ───────────────────────────


def metric_object_diameter_px(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #12 — equivalent diameter of detected object in pixels.

    Computed from area: d = 2 * sqrt(area / pi).
    """
    img01 = _to_float01(image)
    if mask is None:
        mask = detect_object_mask(img01)
    area = mask.sum()
    if area < 1:
        return float('nan')
    return float(2.0 * np.sqrt(area / np.pi))


def metric_centre_offset_px(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #13 — distance from object centre-of-mass to crop centre, in px."""
    img01 = _to_float01(image)
    if mask is None:
        mask = detect_object_mask(img01)
    if mask.sum() < 1:
        return float('nan')
    h, w = img01.shape
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    return float(np.hypot(cy - h / 2.0, cx - w / 2.0))


def metric_crop_occupancy(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    """Metric #14 — fraction of crop area occupied by the object."""
    img01 = _to_float01(image)
    if mask is None:
        mask = detect_object_mask(img01)
    return float(mask.sum() / mask.size)


# ── Metric 15: domain ───────────────────────────────────────────────────


def metric_polarity(
    image: np.ndarray, mask: Optional[np.ndarray] = None,
) -> str:
    """Metric #15 — 'dark_on_light' / 'light_on_dark' / 'ambiguous'.

    Compares mean intensity of the detected object vs background. Polarity
    is determinable if the obj/bg mean difference exceeds either the
    background noise floor (real images) or 5% of the image's dynamic range
    (clean / synthetic images where bg_std≈0).
    """
    img01 = _to_float01(image)
    bg_mean, bg_std, obj_mean, _ = _bg_obj_stats(img01, mask)
    if not np.isfinite(bg_mean) or not np.isfinite(obj_mean):
        return 'ambiguous'
    diff = obj_mean - bg_mean
    # Threshold: max(noise floor, 5% of dynamic range). The dynamic-range
    # fallback handles clean images where noise is zero.
    img_range = float(img01.max() - img01.min())
    noise_floor = bg_std if (np.isfinite(bg_std) and bg_std > 1e-6) else 0.0
    threshold = max(noise_floor, 0.05 * img_range)
    if abs(diff) < threshold:
        return 'ambiguous'
    return 'dark_on_light' if diff < 0 else 'light_on_dark'


# ── Top-level: compute all 15 metrics for one image ──────────────────────


@dataclass
class FingerprintRecord:
    """All 15 metrics for one image, plus optional pre-known metadata."""
    # Blur / edge (1-5)
    erf_sigma_px: float = float('nan')
    erf_r_squared: float = float('nan')
    erf_radius_px: float = float('nan')
    edge_transition_width: float = float('nan')
    edge_gradient_max: float = float('nan')
    quadrant_sigma_var: float = float('nan')
    quadrant_sigma_max_min_ratio: float = float('nan')
    quadrants_fit_count: float = float('nan')
    edge_symmetry_lr_l1: float = float('nan')
    edge_symmetry_tb_l1: float = float('nan')
    # Focus / sharpness (6-8)
    laplacian_variance: float = float('nan')
    tenengrad: float = float('nan')
    high_freq_energy_ratio: float = float('nan')
    # Intensity / contrast (9-11)
    background_mean: float = float('nan')
    background_std: float = float('nan')
    object_bg_contrast: float = float('nan')
    # Geometry (12-14)
    object_diameter_px: float = float('nan')
    centre_offset_px: float = float('nan')
    crop_occupancy: float = float('nan')
    # Domain (15)
    polarity: str = 'ambiguous'
    # Metadata pass-through (not computed; useful when streaming results)
    source_path: str = ''
    source_type: str = ''  # 'synthetic' / 'calibration' / 'real' / 'inference'
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v
        return d


def compute_fingerprint(
    image: np.ndarray,
    *,
    erf_sigma_precomputed: Optional[float] = None,
    erf_r_squared_precomputed: Optional[float] = None,
    skip_erf: bool = False,
    skip_uniformity: bool = False,
    skip_symmetry: bool = False,
) -> FingerprintRecord:
    """Compute all 15 metrics for one image.

    Args:
        image: HxW grayscale array (uint8 [0,255] or float32 [0,1]).
        erf_sigma_precomputed: If provided (e.g. from metadata.csv's
            ``sigma_measured_erf`` column), use it instead of recomputing.
        erf_r_squared_precomputed: Likewise for the fit quality.
        skip_erf: Skip the ERF fit entirely (faster). Quadrant uniformity
            and explicit ERF metrics will be NaN.
        skip_uniformity: Skip 4-quadrant ERF (~4x ERF cost).
        skip_symmetry: Skip the symmetry metric (small saving, but it does
            its own profile sampling).
    """
    img01 = _to_float01(image)
    mask = detect_object_mask(img01)
    rec = FingerprintRecord()

    # Metric 1: ERF (use precomputed if available)
    if erf_sigma_precomputed is not None and not np.isnan(erf_sigma_precomputed):
        rec.erf_sigma_px = float(erf_sigma_precomputed)
        if erf_r_squared_precomputed is not None:
            rec.erf_r_squared = float(erf_r_squared_precomputed)
    elif not skip_erf:
        erf = metric_erf_sigma(img01)
        rec.erf_sigma_px = erf['erf_sigma_px']
        rec.erf_r_squared = erf['erf_r_squared']
        rec.erf_radius_px = erf['erf_radius_px']

    # Metric 2-3: edge width, edge gradient
    rec.edge_transition_width = metric_edge_transition_width(img01, mask)
    rec.edge_gradient_max = metric_edge_gradient_max(img01)

    # Metric 4: spatial uniformity (4-quadrant ERF)
    if not skip_uniformity:
        u = metric_spatial_blur_uniformity(img01)
        rec.quadrant_sigma_var = u['quadrant_sigma_var']
        rec.quadrant_sigma_max_min_ratio = u['quadrant_sigma_max_min_ratio']
        rec.quadrants_fit_count = u['quadrants_fit_count']

    # Metric 5: edge symmetry
    if not skip_symmetry:
        s = metric_edge_symmetry(img01, mask)
        rec.edge_symmetry_lr_l1 = s['edge_symmetry_lr_l1']
        rec.edge_symmetry_tb_l1 = s['edge_symmetry_tb_l1']

    # Metric 6-8: focus / sharpness
    rec.laplacian_variance = metric_laplacian_variance(img01)
    rec.tenengrad = metric_tenengrad(img01)
    rec.high_freq_energy_ratio = metric_high_freq_energy_ratio(img01)

    # Metric 9-11: intensity / contrast
    bg_mean, bg_std, obj_mean, _ = _bg_obj_stats(img01, mask)
    rec.background_mean = bg_mean
    rec.background_std = bg_std
    rec.object_bg_contrast = metric_object_bg_contrast(img01, mask)

    # Metric 12-14: geometry
    rec.object_diameter_px = metric_object_diameter_px(img01, mask)
    rec.centre_offset_px = metric_centre_offset_px(img01, mask)
    rec.crop_occupancy = metric_crop_occupancy(img01, mask)

    # Metric 15: polarity
    rec.polarity = metric_polarity(img01, mask)

    return rec


# Subject-classification of metrics — used by analyses to decide which
# fingerprint columns to compare across pipelines (Check B) vs use for
# distribution coverage (Check C).
SUBJECT_INDEPENDENT_METRICS = (
    'erf_sigma_px', 'edge_transition_width', 'edge_gradient_max',
    'quadrant_sigma_var', 'quadrant_sigma_max_min_ratio',
    'edge_symmetry_lr_l1', 'edge_symmetry_tb_l1',
    'laplacian_variance', 'tenengrad', 'high_freq_energy_ratio',
    'background_mean', 'background_std', 'object_bg_contrast',
    'polarity',
)
SUBJECT_DEPENDENT_METRICS = (
    'object_diameter_px', 'centre_offset_px', 'crop_occupancy',
)
ALL_NUMERIC_METRICS = tuple(
    m for m in SUBJECT_INDEPENDENT_METRICS + SUBJECT_DEPENDENT_METRICS
    if m != 'polarity'
)
