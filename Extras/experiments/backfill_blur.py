"""
Backfill native_blur_sigma into an existing sharp_crops.csv.

Loads each crop image and calls the same measure_erf_blur() used by
focus_classification.py — per-ray erf fitting, median sigma, identical
to calibration/blur_measurement.py logic.
"""

import sys
from pathlib import Path

import cv2
import pandas as pd

COURSEWORK = Path(__file__).parent.parent
PREPROCESSING_DIR = COURSEWORK / "Preprocessing/CROPPING/Preprocessing"
SHARP_CSV = COURSEWORK / "Preprocessing/CROPPING/Preprocessing/OUTPUTNEW/Focus/sharp_crops.csv"

sys.path.insert(0, str(PREPROCESSING_DIR))
from crop_blur_measurement import measure_erf_blur  # noqa: E402


def _resolve_path(crop_path: str) -> Path:
    p = Path(crop_path)
    if not p.exists():
        parts = p.parts
        fixed = Path(*[('OUTPUTNEW' if part == 'OUTPUT' else part) for part in parts])
        if fixed.exists():
            return fixed
    return p


def main():
    crops = pd.read_csv(SHARP_CSV)
    print(f"Loaded {len(crops)} rows from sharp_crops.csv")

    native_blur = []
    n_measured = 0
    n_failed = 0

    for i, row in crops.iterrows():
        sigma = None
        try:
            p = _resolve_path(row['crop_path'])
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sigma = measure_erf_blur(img)
                if sigma is not None:
                    n_measured += 1
                else:
                    n_failed += 1
            else:
                n_failed += 1
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            n_failed += 1

        native_blur.append(sigma)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(crops)} processed...")

    crops['native_blur_sigma'] = native_blur

    measured = crops['native_blur_sigma'].dropna()
    print(f"\nMeasured: {n_measured}  |  Failed: {n_failed}")
    if len(measured):
        print(f"sigma — mean: {measured.mean():.3f} px  std: {measured.std():.3f} px  "
              f"min: {measured.min():.3f}  max: {measured.max():.3f}")

    # Per-camera summary
    for cam, grp in crops.groupby('camera'):
        m = grp['native_blur_sigma'].dropna()
        print(f"  camera {cam}: {len(m)}/{len(grp)} measured  mean={m.mean():.3f} px" if len(m) else
              f"  camera {cam}: 0/{len(grp)} measured")

    crops.to_csv(SHARP_CSV, index=False)
    print(f"\nWritten: {SHARP_CSV}")


if __name__ == "__main__":
    main()
