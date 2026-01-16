"""Image loading and preprocessing utilities.

Handles loading individual frames from .cine files and basic image processing.
"""

from typing import Any, Tuple

import cv2
import numpy as np

from phantom_silence_modular import utils


def load_frame_gray(cine_obj: Any, idx: int) -> np.ndarray:
    """Load a single cine frame as uint8 grayscale.

    Args:
        cine_obj: Loaded cine object from pyphantom.
        idx: Absolute frame index to load.

    Returns:
        Grayscale image as uint8 array with values in [0, 255].
    """
    frame_range = utils.FrameRange(idx, idx)
    frame = cine_obj.get_images(frame_range, Option=1)
    arr = np.squeeze(frame).astype(np.float32)

    # Convert to grayscale if needed
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Normalise to full 8-bit range
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)


def otsu_mask(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Otsu thresholding to identify dark regions.

    Args:
        gray: Grayscale image as uint8 array.

    Returns:
        Tuple of (original_gray, dark_mask) where dark_mask is a boolean
        array that is True where pixels are 'dark' (droplet/sphere).
    """
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return gray, (mask == 0)
