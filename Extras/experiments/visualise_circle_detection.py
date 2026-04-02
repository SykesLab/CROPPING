"""
Visualise circle detection on a sample of camera-m sharp crops.

Runs _detect_circle() from crop_blur_measurement.py on N random camera-m
crops and saves an annotated grid so you can see whether the detected
centre/radius is correct.

Output: spheres/circle_detection_camm.png
"""

import sys
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COURSEWORK = Path(__file__).parent.parent
PREPROCESSING_DIR = COURSEWORK / "Preprocessing/CROPPING/Preprocessing"
SHARP_CSV = COURSEWORK / "Preprocessing/CROPPING/Preprocessing/OUTPUTNEW/Focus/sharp_crops.csv"
OUT_PNG = Path(__file__).parent / "circle_detection_camm.png"

sys.path.insert(0, str(PREPROCESSING_DIR))
from crop_blur_measurement import _detect_circle, measure_erf_blur  # noqa: E402


def _resolve_path(crop_path: str) -> Path:
    p = Path(crop_path)
    if not p.exists():
        parts = p.parts
        fixed = Path(*[("OUTPUTNEW" if part == "OUTPUT" else part) for part in parts])
        if fixed.exists():
            return fixed
    return p


def main(n_samples: int = 12, camera: str = "m", seed: int = 42):
    crops = pd.read_csv(SHARP_CSV)
    cam_rows = crops[crops["camera"] == camera].copy()
    print(f"Found {len(cam_rows)} camera-{camera} rows in sharp_crops.csv")

    random.seed(seed)
    sample = cam_rows.sample(min(n_samples, len(cam_rows)), random_state=seed)

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sample.iterrows()):
        p = _resolve_path(row["crop_path"])
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        if img is None:
            ax.set_title("LOAD FAILED", fontsize=8, color="red")
            continue

        # Normalise to uint8 the same way crop_blur_measurement does
        img_f = img.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0
        img_u8 = (img_f * 255).astype(np.uint8)

        result = _detect_circle(img_u8)
        sigma = measure_erf_blur(img)

        # Draw annotated image (RGB for matplotlib)
        annotated = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

        h, w = img_u8.shape
        cx_img, cy_img = w / 2, h / 2

        if result is not None:
            cx, cy, radius = result
            # Detected circle — green
            cv2.circle(annotated, (cx, cy), radius, (0, 220, 0), 1)
            # Centre dot — green
            cv2.circle(annotated, (cx, cy), 2, (0, 220, 0), -1)
            # Diameter line — cyan
            cv2.line(annotated, (cx - radius, cy), (cx + radius, cy), (0, 220, 220), 1)
            title = (
                f"cx={cx} cy={cy}  r={radius}px\n"
                f"d={2*radius}px  σ={sigma:.1f}px" if sigma is not None
                else f"cx={cx} cy={cy}  r={radius}px\nd={2*radius}px  σ=None"
            )
            color = "green"
        else:
            title = "NO CIRCLE DETECTED"
            color = "red"

        # Image centre crosshair — red
        cv2.line(annotated, (int(cx_img) - 5, int(cy_img)), (int(cx_img) + 5, int(cy_img)), (220, 0, 0), 1)
        cv2.line(annotated, (int(cx_img), int(cy_img) - 5), (int(cx_img), int(cy_img) + 5), (220, 0, 0), 1)

        ax.imshow(annotated)
        ax.set_title(title, fontsize=7, color=color)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fname = Path(row["crop_path"]).name
        ax.set_xlabel(fname, fontsize=6, color="gray")

    legend = [
        mpatches.Patch(color="green", label="Detected circle + centre"),
        mpatches.Patch(color="cyan", label="Diameter span"),
        mpatches.Patch(color="red", label="Image centre"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=False)

    plt.suptitle(f"Circle detection — camera {camera}  (n={len(sample)})", fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(OUT_PNG, dpi=120)
    plt.close()
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
