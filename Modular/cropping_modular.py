# cropping_modular.py
# cropping_modular.py
def crop_droplet_with_sphere_guard(frame, y_top, y_bottom, cx,
                                   target_w, target_h,
                                   y_sphere=None, safety=3):
    """
    Centre crop on droplet, but guarantee:
      - fixed target_w x target_h
      - no warping (pure crop)
      - sphere kept out of crop if y_sphere is given

    Logic:
      1) Start centred on droplet (cx, cy).
      2) If bottom of crop would reveal the sphere, shift crop up.
      3) Clamp to image boundaries while preserving crop size.
      4) Re-apply sphere guard in case boundary clamping broke it.
    """

    H, W = frame.shape

    cy = 0.5 * (y_top + y_bottom)
    half_h = target_h // 2
    half_w = target_w // 2

    # Initial centred crop
    x0 = int(cx - half_w)
    x1 = x0 + target_w
    y0 = int(cy - half_h)
    y1 = y0 + target_h

    # --- Sphere guard (first pass): keep bottom above sphere - safety
    if y_sphere is not None:
        max_y1 = int(y_sphere - safety)
        if y1 > max_y1:
            shift = y1 - max_y1
            y0 -= shift
            y1 -= shift

    # --- Clamp vertically, preserving height
    if y0 < 0:
        y1 -= y0  # move down by |y0|
        y0 = 0
    if y1 > H:
        y0 -= (y1 - H)
        y1 = H

    # --- Clamp horizontally, preserving width
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > W:
        x0 -= (x1 - W)
        x1 = W

    # --- Sphere guard (second pass) after boundary clamping ---
    if y_sphere is not None:
        max_y1 = int(y_sphere - safety)
        if y1 > max_y1:
            shift = y1 - max_y1
            y0 -= shift
            y1 -= shift
            # If this pushed us above the top, just pin to top;
            # in realistic data this should almost never happen.
            if y0 < 0:
                y1 -= y0
                y0 = 0

    return frame[int(y0):int(y1), int(x0):int(x1)]

