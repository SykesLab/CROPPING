# frame_cropping.py
#
# Automatically detect the dark bottom sphere (large dark region at the bottom)
# and crop away everything below its top-most point.

import numpy as np
import cv2


def crop_below_dark_sphere(frame_raw):
    """
    Detect the bottom dark sphere and crop the image to keep only
    everything ABOVE the sphere's top-most point.

    Returns:
        {
            "crop_y": int,
            "cropped": 2D array,
            "sphere_mask": 2D bool mask
        }
    """

    # --- 1. Normalise to 0–255 ---
    img_u8 = cv2.normalize(
        frame_raw, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --- 2. Blur ---
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)

    # --- 3. Otsu threshold (dark = sphere + droplet) ---
    thresh_val, mask_dark = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask_dark = (mask_dark == 255)  # boolean

    H, W = mask_dark.shape

    # --- 4. Remove objects near the top (droplet) ---
    # DROPLET is always near top → mask out top ~30% region
    top_cut = int(H * 0.30)
    mask_bottom_region = np.zeros_like(mask_dark, dtype=bool)
    mask_bottom_region[top_cut:, :] = mask_dark[top_cut:, :]

    # --- 5. Find connected components in the lower region ---
    num_labels, labels = cv2.connectedComponents(
        mask_bottom_region.astype(np.uint8)
    )

    if num_labels <= 1:
        # fallback: nothing detected
        return {
            "crop_y": H,
            "cropped": frame_raw.copy(),
            "sphere_mask": np.zeros_like(mask_dark),
        }

    # --- 6. Choose LARGEST connected component (bottom sphere) ---
    largest_label = None
    largest_size = -1

    for lab in range(1, num_labels):  # skip background
        size = np.sum(labels == lab)
        if size > largest_size:
            largest_size = size
            largest_label = lab

    sphere_mask = (labels == largest_label)

    # --- 7. Find top-most pixel of the sphere ---
    ys, xs = np.where(sphere_mask)
    if len(ys) == 0:
        crop_y = H
    else:
        crop_y = int(ys.min())

    # --- 8. Crop the image above the sphere ---
    cropped = frame_raw[:crop_y, :]

    return {
        "crop_y": crop_y,
        "cropped": cropped,
        "sphere_mask": sphere_mask,
    }
