# image_utils_modular.py
#
# Loads individual cine frames as grayscale.
# Uses silent pyphantom imports from phantom_silence_modular.py

import numpy as np
import cv2

from phantom_silence_modular import cine, utils  # <-- Silent imports!


def load_frame_gray(c, idx):
    """
    Load a single cine frame as uint8 grayscale [0,255].
    """
    fr = utils.FrameRange(idx, idx)
    frame = c.get_images(fr, Option=1)
    arr = np.squeeze(frame).astype(np.float32)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)


def otsu_mask(gray):
    """
    Apply Otsu threshold and return (gray, dark_mask_bool).
    dark_mask is True where pixel is considered 'dark' (droplet/sphere).
    """
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, (mask == 0)
