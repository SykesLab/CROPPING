# frame_selector_test_4mm_viz.py
#
# Utilities for analysing a single .cine:
#   - build Otsu-based darkness curve
#   - pick best frame (max dark fraction)
#   - return best frame image + mask for visualisation

import numpy as np
import cv2
from scipy.signal import savgol_filter
from pyphantom import utils  # for FrameRange


def _get_frame_and_mask(c, frame_index):
    """
    Load a single frame from cine `c` and compute an Otsu mask.

    Returns:
        frame_raw   : 2D numpy array (original intensity, typically uint16)
        mask_dark   : boolean array, True where pixel is 'dark' (droplet)
        dark_frac   : float, fraction of dark pixels
    """
    fr = utils.FrameRange(frame_index, frame_index)
    frame = c.get_images(fr, Option=1)  # shape (1, H, W) or (H, W, ...)
    arr = np.squeeze(frame)

    # Ensure 2D grayscale
    if arr.ndim == 3:
        # If it's (H, W, C), convert to grayscale
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    frame_raw = arr.astype(np.float32)

    # Normalise to 0–255 for Otsu
    norm = cv2.normalize(
        frame_raw, None, 0, 255, cv2.NORM_MINMAX
    )
    norm_u8 = norm.astype(np.uint8)

    # Otsu: returns threshold + binary mask (255 = bright, 0 = dark)
    _, mask = cv2.threshold(
        norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # droplet assumed darker than background → mask_dark is inverse
    mask_dark = (mask == 0)

    dark_frac = float(mask_dark.mean())

    return frame_raw, mask_dark, dark_frac


def analyze_cine(c):
    """
    Parameter-free robust selection using curve shape:
      - Smooth darkness curve
      - Compute derivative
      - Find rising edge
      - Select first frame where derivative ~ 0 after rise
    """

    first, last = c.range
    total = (last - first) + 1

    # ---- Build darkness curve ----
    darkness_vals = []
    for idx in range(first, last + 1):
        _, mask_dark, dark_frac = _get_frame_and_mask(c, idx)
        darkness_vals.append(dark_frac)

    darkness = np.array(darkness_vals, dtype=np.float32)

    # ---- Smooth the curve (parameter free) ----
    # Window = min(total, 11)
    win = min(total if total % 2 == 1 else total - 1, 11)
    smooth = savgol_filter(darkness, window_length=win, polyorder=2)

    # ---- First derivative ----
    deriv = np.gradient(smooth)

    # ---- Step 1: find start of rising edge ----
    rise_peak = int(np.argmax(deriv))  # steepest rise

    # ---- Step 2: find first near-zero derivative after rise ----
    post_rise = deriv[rise_peak:]

    # absolute derivative threshold = automatic, median-based
    threshold = np.median(np.abs(post_rise)) * 0.5

    zeros = np.where(np.abs(post_rise) < threshold)[0]

    if len(zeros) == 0:
        # fallback: best is max darkness
        best_pos = int(np.argmax(smooth))
    else:
        best_pos = rise_peak + int(zeros[0])  # first zero-crossing

    best_frame = first + best_pos
    best_dark = float(darkness[best_pos])

    # ---- Load best frame for visualisation ----
    best_frame_image, best_mask_dark, _ = _get_frame_and_mask(c, best_frame)

    return {
        "first_frame": first,
        "last_frame": last,
        "total_frames": total,
        "best_frame": best_frame,
        "best_dark_fraction": best_dark,
        "darkness_curve": darkness,
        "best_frame_image": best_frame_image,
        "best_mask_dark": best_mask_dark,
        "smooth_curve": smooth,
        "derivative": deriv,
    }
