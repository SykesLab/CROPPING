"""
Focus quality metrics for assessing image sharpness.

These edge-based metrics quantify how in-focus a crop is. Sharp images
have strong edges (high gradient values), while blurry images have
weak, smeared edges.

Metrics:
  - Laplacian variance: var(∇²I) - most commonly used
  - Tenengrad: sum of squared Sobel gradients
  - Brenner: horizontal gradient with 2px spacing (noise-robust)
  - Normalised variants for cross-image comparison
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian - higher means sharper."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    """Sum of squared Sobel gradient magnitudes."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.sum(gx ** 2 + gy ** 2))


def tenengrad_variance(gray: np.ndarray, ksize: int = 3) -> float:
    """Variance of gradient magnitude - normalised version of Tenengrad."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.var(gradient_magnitude))


def brenner_gradient(gray: np.ndarray) -> float:
    """Brenner focus measure: sum of (I[x+2] - I[x])² horizontally."""
    gray_float = gray.astype(np.float64)
    diff = gray_float[:, 2:] - gray_float[:, :-2]
    return float(np.sum(diff ** 2))


def normalised_laplacian(gray: np.ndarray) -> float:
    """Laplacian variance normalised by mean intensity."""
    mean_val = np.mean(gray)
    if mean_val < 1e-6:
        return 0.0
    return float(laplacian_variance(gray) / (mean_val ** 2))


def energy_of_gradient(gray: np.ndarray) -> float:
    """Sum of squared first derivatives in both directions."""
    gray_float = gray.astype(np.float64)
    dx = np.diff(gray_float, axis=1)
    dy = np.diff(gray_float, axis=0)
    return float(np.sum(dx ** 2) + np.sum(dy ** 2))


def compute_all_focus_metrics(gray: np.ndarray) -> Dict[str, float]:
    """Compute all available focus metrics."""
    return {
        "laplacian_var": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "tenengrad_var": tenengrad_variance(gray),
        "brenner": brenner_gradient(gray),
        "norm_laplacian": normalised_laplacian(gray),
        "energy_gradient": energy_of_gradient(gray),
    }


def compute_focus_score(gray: np.ndarray, method: str = "laplacian_var") -> float:
    """Compute single focus score using the specified method."""
    methods = {
        "laplacian_var": laplacian_variance,
        "tenengrad": tenengrad,
        "tenengrad_var": tenengrad_variance,
        "brenner": brenner_gradient,
        "norm_laplacian": normalised_laplacian,
        "energy_gradient": energy_of_gradient,
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods.keys())}")

    return methods[method](gray)


def classify_focus(score: float, sharp_threshold: float, blur_threshold: float) -> str:
    """Classify as 'sharp', 'medium', or 'blurry' based on thresholds."""
    if score >= sharp_threshold:
        return "sharp"
    elif score <= blur_threshold:
        return "blurry"
    return "medium"


def compute_focus_metrics_for_crop(
    crop: np.ndarray,
    include_classification: bool = False,
    sharp_threshold: Optional[float] = None,
    blur_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute focus metrics for a crop, optionally with classification."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    metrics = compute_all_focus_metrics(gray)

    if include_classification:
        if sharp_threshold is None or blur_threshold is None:
            raise ValueError("Must provide thresholds when include_classification=True")
        metrics["focus_class"] = classify_focus(
            metrics["laplacian_var"], sharp_threshold, blur_threshold
        )

    return metrics


# --- Batch processing utilities ---

def compute_dataset_statistics(scores: np.ndarray) -> Dict[str, float]:
    """Compute statistics for determining classification thresholds."""
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
    """Suggest sharp/blur thresholds based on score distribution."""
    sharp_thresh = float(np.percentile(scores, sharp_percentile))
    blur_thresh = float(np.percentile(scores, blur_percentile))
    return sharp_thresh, blur_thresh


def classify_folder_focus(
    folder_scores: np.ndarray,
    sharp_percentile: float = 75.0,
    blur_percentile: float = 25.0,
) -> Tuple[np.ndarray, float, float]:
    """Classify focus within a folder using per-folder thresholds."""
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
