"""
Image loading and preprocessing utilities for CINE frames.
"""

from typing import Any, Tuple

import cv2
import numpy as np

from cine_io import utils


def load_frame_gray(cine_obj: Any, idx: int) -> np.ndarray:
    """Load a single frame from a CINE file as uint8 grayscale."""
    frame_range = utils.FrameRange(idx, idx)
    frame = cine_obj.get_images(frame_range, Option=1)
    arr = np.squeeze(frame).astype(np.float32)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)


def otsu_mask(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Otsu thresholding to separate dark regions (droplet/sphere)
    from the bright background.

    Returns (original_gray, dark_mask) where dark_mask is True for dark pixels.
    """
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, (mask == 0)
