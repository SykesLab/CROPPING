"""Focus quality metrics for defocus quantification.

Provides edge-based sharpness metrics to assess image focus quality.
These serve as:
    1. Immediate quality filters for crops
    2. Baseline comparison for future CNN-based methods
    3. Potential features for focus classification

Metrics implemented:
    - Laplacian Variance: var(∇²I)
    - Tenengrad: Σ(Gx² + Gy²) using Sobel
    - Brenner Gradient: Σ(I[x+2] - I[x])²
    - Normalised variants for cross-image comparison
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Compute Laplacian variance as focus measure.

    The Laplacian highlights edges (second derivative). Sharp images
    have strong edges → high variance. Blurry images have weak edges
    → low variance.

    Args:
        gray: Grayscale image as uint8 or float array.

    Returns:
        Variance of Laplacian response. Higher = sharper.
    """
    # Use 64-bit float to avoid overflow
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    """Compute Tenengrad focus measure using Sobel gradients.

    Computes sum of squared gradient magnitudes. Sharp images have
    strong gradients at edges → high Tenengrad value.

    Args:
        gray: Grayscale image as uint8 or float array.
        ksize: Sobel kernel size (3, 5, or 7).

    Returns:
        Sum of squared gradient magnitudes. Higher = sharper.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Sum of squared magnitudes
    gradient_magnitude_sq = gx ** 2 + gy ** 2
    return float(np.sum(gradient_magnitude_sq))


def tenengrad_variance(gray: np.ndarray, ksize: int = 3) -> float:
    """Compute variance of Tenengrad gradient magnitude.

    Normalised version of Tenengrad that's less sensitive to image
    size and content. Better for comparing across different images.

    Args:
        gray: Grayscale image as uint8 or float array.
        ksize: Sobel kernel size (3, 5, or 7).

    Returns:
        Variance of gradient magnitude. Higher = sharper.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.var(gradient_magnitude))


def brenner_gradient(gray: np.ndarray) -> float:
    """Compute Brenner focus measure.

    Uses horizontal gradient with 2-pixel spacing:
        B = Σ (I[x+2, y] - I[x, y])²

    Less sensitive to noise than single-pixel gradients.

    Args:
        gray: Grayscale image as uint8 or float array.

    Returns:
        Brenner gradient sum. Higher = sharper.
    """
    gray_float = gray.astype(np.float64)
    
    # Horizontal difference with 2-pixel spacing
    diff = gray_float[:, 2:] - gray_float[:, :-2]
    return float(np.sum(diff ** 2))


def normalised_laplacian(gray: np.ndarray) -> float:
    """Compute normalised Laplacian variance.

    Normalised by image mean to reduce sensitivity to overall
    brightness variations.

    Args:
        gray: Grayscale image as uint8 or float array.

    Returns:
        Normalised Laplacian variance. Higher = sharper.
    """
    mean_val = np.mean(gray)
    if mean_val < 1e-6:
        return 0.0
    
    lap_var = laplacian_variance(gray)
    return float(lap_var / (mean_val ** 2))


def energy_of_gradient(gray: np.ndarray) -> float:
    """Compute energy of gradient focus measure.

    Sum of squared first derivatives in both directions.

    Args:
        gray: Grayscale image as uint8 or float array.

    Returns:
        Energy of gradient. Higher = sharper.
    """
    gray_float = gray.astype(np.float64)
    
    # First derivatives using simple differences
    dx = np.diff(gray_float, axis=1)
    dy = np.diff(gray_float, axis=0)
    
    return float(np.sum(dx ** 2) + np.sum(dy ** 2))


def compute_all_focus_metrics(gray: np.ndarray) -> Dict[str, float]:
    """Compute all focus metrics for an image.

    Args:
        gray: Grayscale image as uint8 array.

    Returns:
        Dictionary with all computed metrics:
            - laplacian_var: Laplacian variance
            - tenengrad: Tenengrad sum
            - tenengrad_var: Tenengrad variance (normalised)
            - brenner: Brenner gradient
            - norm_laplacian: Normalised Laplacian
            - energy_gradient: Energy of gradient
    """
    return {
        "laplacian_var": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "tenengrad_var": tenengrad_variance(gray),
        "brenner": brenner_gradient(gray),
        "norm_laplacian": normalised_laplacian(gray),
        "energy_gradient": energy_of_gradient(gray),
    }


def compute_focus_score(
    gray: np.ndarray,
    method: str = "laplacian_var",
) -> float:
    """Compute single focus score using specified method.

    Args:
        gray: Grayscale image as uint8 array.
        method: One of 'laplacian_var', 'tenengrad', 'tenengrad_var',
                'brenner', 'norm_laplacian', 'energy_gradient'.

    Returns:
        Focus score (higher = sharper).

    Raises:
        ValueError: If method is not recognised.
    """
    methods = {
        "laplacian_var": laplacian_variance,
        "tenengrad": tenengrad,
        "tenengrad_var": tenengrad_variance,
        "brenner": brenner_gradient,
        "norm_laplacian": normalised_laplacian,
        "energy_gradient": energy_of_gradient,
    }
    
    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {list(methods.keys())}"
        )
    
    return methods[method](gray)


def classify_focus(
    score: float,
    sharp_threshold: float,
    blur_threshold: float,
) -> str:
    """Classify focus quality based on score and thresholds.

    Args:
        score: Focus score from any metric.
        sharp_threshold: Scores above this are "sharp".
        blur_threshold: Scores below this are "blurry".

    Returns:
        Classification: "sharp", "medium", or "blurry".
    """
    if score >= sharp_threshold:
        return "sharp"
    elif score <= blur_threshold:
        return "blurry"
    else:
        return "medium"


def compute_focus_metrics_for_crop(
    crop: np.ndarray,
    include_classification: bool = False,
    sharp_threshold: Optional[float] = None,
    blur_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute focus metrics for a crop image.

    Convenience function that handles both grayscale and colour inputs,
    computes all metrics, and optionally classifies focus quality.

    Args:
        crop: Image as numpy array (grayscale or BGR).
        include_classification: If True, include focus classification.
        sharp_threshold: Threshold for "sharp" (uses laplacian_var).
        blur_threshold: Threshold for "blurry" (uses laplacian_var).

    Returns:
        Dictionary with metrics and optional classification.
    """
    # Convert to grayscale if needed
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    metrics = compute_all_focus_metrics(gray)
    
    if include_classification:
        if sharp_threshold is None or blur_threshold is None:
            raise ValueError(
                "Must provide sharp_threshold and blur_threshold "
                "when include_classification=True"
            )
        metrics["focus_class"] = classify_focus(
            metrics["laplacian_var"],
            sharp_threshold,
            blur_threshold,
        )
    
    return metrics


# ============================================================
# BATCH PROCESSING UTILITIES
# ============================================================

def compute_dataset_statistics(
    scores: np.ndarray,
) -> Dict[str, float]:
    """Compute statistics for a batch of focus scores.

    Useful for determining classification thresholds.

    Args:
        scores: Array of focus scores.

    Returns:
        Dictionary with min, max, mean, std, and percentiles.
    """
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
    }


def suggest_thresholds(
    scores: np.ndarray,
    sharp_percentile: float = 75.0,
    blur_percentile: float = 25.0,
) -> Tuple[float, float]:
    """Suggest classification thresholds based on score distribution.

    Args:
        scores: Array of focus scores from dataset.
        sharp_percentile: Percentile above which images are "sharp".
        blur_percentile: Percentile below which images are "blurry".

    Returns:
        Tuple of (sharp_threshold, blur_threshold).
    """
    sharp_thresh = float(np.percentile(scores, sharp_percentile))
    blur_thresh = float(np.percentile(scores, blur_percentile))
    return sharp_thresh, blur_thresh


def classify_folder_focus(
    folder_scores: np.ndarray,
    sharp_percentile: float = 75.0,
    blur_percentile: float = 25.0,
) -> Tuple[np.ndarray, float, float]:
    """Classify focus within a single folder using per-folder thresholds.

    Args:
        folder_scores: Array of focus scores for one folder.
        sharp_percentile: Percentile above which images are "sharp".
        blur_percentile: Percentile below which images are "blurry".

    Returns:
        Tuple of (classifications array, sharp_threshold, blur_threshold).
    """
    sharp_thresh, blur_thresh = suggest_thresholds(
        folder_scores, sharp_percentile, blur_percentile
    )
    
    classifications = []
    for score in folder_scores:
        if score >= sharp_thresh:
            classifications.append("sharp")
        elif score <= blur_thresh:
            classifications.append("blurry")
        else:
            classifications.append("medium")
    
    return np.array(classifications), sharp_thresh, blur_thresh
