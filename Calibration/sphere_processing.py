"""
Sphere detection and processing utilities for calibration images.

Functions for detecting spheres, mirroring to remove stage artifacts,
blackening interiors to remove hot spots, flattening crops, and cropping to square.
"""

import cv2
import numpy as np


# Default feather width for flattening (pixels)
FLATTEN_FEATHER = 3


def find_sphere_center(
    img: np.ndarray,
    upper_only: bool = True,
    upper_fraction: float = 0.60,
) -> tuple | None:
    """
    Detect sphere edge and fit circle.

    Args:
        img: Grayscale image (2D numpy array)
        upper_only: If True, use only the upper portion of the contour
                    for circle fitting (avoids stage reflection)
        upper_fraction: Fraction of contour height to use when upper_only=True

    Returns:
        (cx, cy, radius) or None if detection fails
    """
    if img.ndim != 2:
        return None

    # Canny requires uint8 — normalize if needed
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img

    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return None

    pts = cnt.reshape(-1, 2)

    if upper_only:
        y_min = pts[:, 1].min()
        y_max = pts[:, 1].max()
        y_cut = y_min + upper_fraction * (y_max - y_min)
        upper_pts = pts[pts[:, 1] <= y_cut]
        if len(upper_pts) >= 20:
            pts = upper_pts

    (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))

    h, w = img.shape[:2]
    if r_i <= 0 or r_i > max(h, w):
        return None
    if not (0 <= cx_i < w and 0 <= cy_i < h):
        return None

    return cx_i, cy_i, r_i


def mirror_from_center(img: np.ndarray, center_y: int) -> np.ndarray:
    """Mirror about y=center_y using top half -> flipped bottom half."""
    h, _ = img.shape[:2]
    center_y = int(np.clip(center_y, 0, h - 1))
    top_half = img[:center_y + 1, :]
    bottom_half = cv2.flip(top_half, 0)
    return np.vstack([top_half, bottom_half])


def blacken_sphere_interior(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    radius_fraction: float = 0.98,
    edge_margin_px: int = 2,
) -> np.ndarray:
    """
    Fill sphere interior with the sphere's own colour, stopping slightly short of the fitted edge.

    Samples the median intensity from the deep interior (within 50% of radius)
    and fills with that value — like a colour pipette — so the filled region
    matches the sphere rather than introducing a black artefact.

    Args:
        img: Input image
        center_x, center_y: Sphere center coordinates
        radius: Sphere radius in pixels
        radius_fraction: Fraction of radius to fill (default 0.98)
        edge_margin_px: Additional margin in pixels from edge (default 2)
    """
    if radius <= 0:
        return img

    h, w = img.shape[:2]
    r_eff = int(min(radius * radius_fraction, radius - edge_margin_px))
    if r_eff <= 0:
        return img

    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2

    # Sample sphere colour from deep interior (within 50% of radius, far from edge)
    sample_r = int(radius * 0.5)
    if sample_r > 0:
        interior_mask = dist_sq < (sample_r ** 2)
        if np.any(interior_mask):
            fill_value = float(np.median(img[interior_mask]))
        else:
            fill_value = 0.0
    else:
        fill_value = 0.0

    img[dist_sq < (r_eff ** 2)] = fill_value
    return img


def _detect_sphere_contour(image: np.ndarray) -> tuple | None:
    """
    Detect sphere boundary via Canny edges, selecting the contour that
    contains the image center (since crops are centered on the sphere).

    Uses fillPoly to create a proper filled mask for distance-map computation.

    Returns:
        (contour, cx, cy, radius) or None if detection fails.
    """
    if image.dtype != np.uint8:
        img_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = image

    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # The sphere is at the center of the crop. Find the contour containing
    # the center point — that's the sphere boundary.
    h, w = image.shape[:2]
    center_pt = (w // 2, h // 2)

    cnt = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.pointPolygonTest(c, center_pt, False) >= 0:
            cnt = c
            break

    if cnt is None:
        # Fallback: largest contour
        cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return None

    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    radius = int(round(np.mean(dists)))

    return cnt, cx, cy, radius


def _detect_sphere_contour_otsu(image: np.ndarray) -> tuple | None:
    """
    Detect sphere boundary via Otsu thresholding. More robust than Canny on
    blurry images (calibration z-stacks) because it finds the global intensity
    boundary rather than edge gradients.

    Handles both dark-on-bright and bright-on-dark spheres by checking which
    side of the threshold contains the image center.

    Returns:
        (contour, cx, cy, radius) or None if detection fails.
    """
    if image.dtype != np.uint8:
        img_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = image

    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Check if center is inside the mask — if not, the sphere is bright-on-dark
    h, w = image.shape[:2]
    if thresh[h // 2, w // 2] == 0:
        thresh = 255 - thresh

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 20:
        return None

    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    radius = int(round(np.mean(dists)))

    return cnt, cx, cy, radius


def flatten_sphere_crop(
    image: np.ndarray,
    contour: np.ndarray | None = None,
    feather: int = FLATTEN_FEATHER,
    inner_margin: int = 0,
    flatten_exterior: bool = True,
) -> tuple[np.ndarray, dict | None]:
    """
    Flatten sphere crop: interior to 0, optionally background to 1, with feather.

    Detects the sphere boundary via Canny edge detection (default) or Otsu
    thresholding (when inner_margin > 0, for calibration z-stacks), then uses
    a signed distance map to flatten the image.

    Two modes:
        Simple (default): inner_margin=0, flatten_exterior=True.
            Interior=0, exterior=1, 3px feather. For sharp preprocessing crops.
            Validated on 62 crops: 100% CLEAN, +1.0% mean error.

        Calibration: inner_margin=20, flatten_exterior=False.
            Only interior beyond 20px from edge is zeroed. Exterior untouched.
            Preserves blur-widened edges on defocused z-stack frames.
            Per-frame Otsu detection adapts boundary to each frame's blur.

    Args:
        image: Grayscale image (uint8 or float32)
        contour: Pre-detected contour, or None to auto-detect
        feather: Width of transition zone in pixels (default 3)
        inner_margin: Pixels inside contour to preserve before flattening (default 0)
        flatten_exterior: Whether to set exterior to 1.0 (default True)

    Returns:
        (flattened_float32_image, info_dict) or (original_copy, None) on failure
    """
    # Normalise to float [0, 1]
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)

    # Auto-detect contour if not provided
    if contour is None:
        if inner_margin > 0:
            # Calibration mode: use Otsu (works reliably across blur levels)
            detection = _detect_sphere_contour_otsu(img_f)
        else:
            # Simple mode: use Canny (sharper boundary for sharp crops)
            detection = _detect_sphere_contour(img_f)
        if detection is None:
            return img_f.copy(), None
        contour, cx, cy, radius = detection
    else:
        # Compute center/radius from provided contour
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return img_f.copy(), None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        pts = contour.reshape(-1, 2)
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        radius = int(round(np.mean(dists)))

    # Build filled mask using fillPoly
    h, w = img_f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 2)], 255)

    # Signed distance map: positive = inside sphere, negative = outside
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    signed_dist = np.where(mask > 0, dist_inside, -dist_outside)

    out = img_f.copy()
    fw = max(feather, 1)
    m_in = inner_margin

    # Interior: set to 0 beyond (inner_margin + feather) from contour
    out[signed_dist > (m_in + fw)] = 0.0

    # Inner feather: smooth transition from original to 0
    mfi = (signed_dist > m_in) & (signed_dist <= (m_in + fw))
    if np.any(mfi):
        t = np.clip((signed_dist[mfi] - m_in) / fw, 0, 1)
        out[mfi] = (1 - t) * img_f[mfi]

    # Exterior (only if requested)
    if flatten_exterior:
        out[signed_dist < -fw] = 1.0
        mfo = (signed_dist < 0) & (signed_dist >= -fw)
        if np.any(mfo):
            t = np.clip(-signed_dist[mfo] / fw, 0, 1)
            out[mfo] = img_f[mfo] + t * (1.0 - img_f[mfo])

    method = 'contour_distmap_calibration' if inner_margin > 0 else 'contour_distmap_simple'
    return out, {
        'center': (cx, cy), 'radius': radius,
        'feather': fw, 'inner_margin': m_in,
        'flatten_exterior': flatten_exterior,
        'method': method,
    }


def crop_to_square(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    padding: float = 1.2,
) -> np.ndarray:
    """
    Crop to a square centered on (center_x, center_y).

    Args:
        img: Input image
        center_x, center_y: Center coordinates
        radius: Sphere radius
        padding: Multiplier for half-size (1.2 = 20% padding beyond radius)
    """
    h, w = img.shape[:2]
    half_size = int(radius * padding)

    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)

    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    size = min(ch, cw)
    if size <= 0:
        return crop

    cx_local = min(cw - 1, max(0, center_x - x1))
    cy_local = min(ch - 1, max(0, center_y - y1))

    sx1 = cx_local - size // 2
    sy1 = cy_local - size // 2
    sx2 = sx1 + size
    sy2 = sy1 + size

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > cw:
        sx1 -= (sx2 - cw)
        sx2 = cw
    if sy2 > ch:
        sy1 -= (sy2 - ch)
        sy2 = ch

    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(cw, sx2), min(ch, sy2)

    return crop[sy1:sy2, sx1:sx2]


def _apply_sphere_pipeline(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    output_size: int | None = None,
    mirror: bool = True,
    blacken: bool = True,
    flatten: bool = False,
    flatten_mode: str = "default",
) -> np.ndarray:
    """Apply crop -> normalize to uint8 -> resize, with optional mirror, blacken, and flatten steps.

    Args:
        flatten_mode: Flattening mode.
            "default" — per-frame Otsu contour detection, interior-only flattening with 20px
                inner margin and 3px feather. Exterior left untouched. Designed for calibration
                z-stacks where blur widens the edge and exterior must be preserved.
            "simple" — per-frame Canny contour detection, zero-margin flattening.
                Interior=0, exterior=1, 3px feather. For sharp preprocessing crops.
            "inference" — per-frame Otsu contour detection, full flattening (interior=0,
                exterior=1) with 40px feather. Designed for inference on defocused images:
                Otsu handles blurry boundaries, wide feather preserves blur-widened edges,
                full exterior flatten matches training domain.
    """
    import logging as _logging

    processed = img.copy()
    if mirror:
        processed = mirror_from_center(processed, cy)
    if blacken:
        processed = blacken_sphere_interior(
            processed, cx, cy, radius, radius_fraction=0.50, edge_margin_px=0
        )
    processed = crop_to_square(processed, cx, cy, radius, padding=1.2)

    if flatten:
        # Convert to float [0, 1] for flattening
        if processed.dtype == np.uint8:
            proc_f = processed.astype(np.float32) / 255.0
        else:
            proc_f = processed.astype(np.float32)
            if proc_f.max() > 1.0:
                proc_f /= proc_f.max()

        if flatten_mode == "simple":
            flat, info = flatten_sphere_crop(proc_f)
        elif flatten_mode == "inference":
            # Otsu detection (robust to blur) + full flatten + 40px feather
            # The wide feather preserves blur-widened edge profiles while
            # removing background gradients that confuse the model.
            flat, info = flatten_sphere_crop(proc_f, inner_margin=1,
                                             flatten_exterior=True, feather=40)
        else:
            # Calibration mode: per-frame Otsu, interior only, 20px inner margin
            flat, info = flatten_sphere_crop(proc_f, inner_margin=20, flatten_exterior=False)

        if info is not None:
            processed = flat
        else:
            _logging.getLogger(__name__).warning(
                "_apply_sphere_pipeline: flatten detection failed — using unflattened crop")

    # Normalize to uint8 after processing (not before) so that the
    # blackened interior doesn't compress edge dynamic range
    if processed.dtype != np.uint8:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if output_size is not None and output_size > 0:
        h, w = processed.shape[:2]
        if h != output_size or w != output_size:
            processed = cv2.resize(
                processed, (output_size, output_size),
                interpolation=cv2.INTER_AREA
            )

    return processed


def process_sphere_image(
    img: np.ndarray,
    upper_only: bool = True,
    output_size: int | None = None,
) -> tuple[np.ndarray | None, tuple | None]:
    """
    Full sphere processing pipeline: detect → mirror → blacken → crop → resize.

    Args:
        img: Grayscale input image
        upper_only: Use upper contour only for circle fitting
        output_size: If set, resize the cropped image to this square size

    Returns:
        (processed_image, (cx, cy, radius)) or (None, None) if detection fails
    """
    result = find_sphere_center(img, upper_only=upper_only)
    if result is None:
        return None, None

    cx, cy, radius = result
    processed = _apply_sphere_pipeline(img, cx, cy, radius, output_size)
    return processed, (cx, cy, radius)


def find_consensus_sphere(
    images: list[np.ndarray],
    upper_only: bool = True,
) -> tuple[int, int, int] | None:
    """
    Detect sphere on all frames and return the consensus (median) circle.

    In a z-stack the sphere doesn't move — only focus changes. Out-of-focus
    frames may detect noise instead of the sphere. By taking the median of
    all detections, we get the true sphere position.

    Returns:
        (cx, cy, radius) or None if no detections at all
    """
    detections = []
    for img in images:
        result = find_sphere_center(img, upper_only=upper_only)
        if result is not None:
            detections.append(result)

    if not detections:
        return None

    cxs = [d[0] for d in detections]
    cys = [d[1] for d in detections]
    radii = [d[2] for d in detections]

    cx = int(np.median(cxs))
    cy = int(np.median(cys))
    radius = int(np.median(radii))

    return cx, cy, radius


def process_sphere_stack(
    images: list[np.ndarray],
    upper_only: bool = True,
    output_size: int | None = None,
    mirror: bool = True,
    blacken: bool = True,
    flatten: bool = False,
    flatten_mode: str = "default",
) -> tuple[list[np.ndarray], tuple | None]:
    """
    Process a z-stack of sphere images using a single consensus detection.

    Detects the sphere across all frames, finds the consensus center/radius,
    then applies the pipeline to every frame with the same parameters.
    Flattening uses per-frame contour detection (adapts to each frame's blur).

    Args:
        images: List of grayscale images (z-stack)
        upper_only: Use upper contour only for circle fitting
        output_size: If set, resize output to this square size
        mirror: Apply vertical mirror about sphere centre (calibration only)
        blacken: Fill sphere interior with median value (calibration only)
        flatten: Flatten crops (interior=0, background=1, small feather)
        flatten_mode: "default" for calibration (inner margin, no exterior),
                      "simple" for preprocessing (zero margin, full flatten)

    Returns:
        (processed_images, (cx, cy, radius)) or (original_images, None) if detection fails
    """
    sphere = find_consensus_sphere(images, upper_only=upper_only)
    if sphere is None:
        return list(images), None

    cx, cy, radius = sphere

    processed = []
    for img in images:
        processed.append(_apply_sphere_pipeline(img, cx, cy, radius, output_size,
                                                mirror=mirror, blacken=blacken,
                                                flatten=flatten,
                                                flatten_mode=flatten_mode))

    return processed, (cx, cy, radius)
